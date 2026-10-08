"""Lightweight Tkinter dashboard for the Fortnite AI Research platform.

Shows:
* Current mode
* AI status
* Training status
* Inference latency
* Selected model
* Confidence
* Performance metrics
* Memory usage
* CPU usage
* Safety status
* Errors

Kept deliberately lightweight for the 4 GB target machine. The UI uses
polling over subprocess-heavy rendering; the simulator/policy runs in
the same process for the demo and the UI merely displays state.
"""

from __future__ import annotations

import importlib.util
import platform
import time
from threading import Thread
from typing import Callable, Optional

import tkinter as tk
from tkinter import ttk, scrolledtext

# psutil is optional for CPU/RAM readings.
try:
    import psutil
except Exception:
    psutil = None  # type: ignore


class Dashboard:
    """Responsive Tkinter dashboard for the platform."""

    def __init__(
        self,
        *,
        on_close: Optional[Callable[[], None]] = None,
        agent_callback: Optional[Callable[[], dict[str, object]]] = None,
        update_interval_ms: int = 250,
    ) -> None:
        self.on_close = on_close
        self.agent_callback = agent_callback or self._default_agent_state
        self.update_interval_ms = update_interval_ms
        self._root: Optional[tk.Tk] = None
        self._running = False
        self._last_error: Optional[str] = None
        self._status_text: str = "idle"

    # ------------------------------------------------------------------
    # Public lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        if self._root is not None:
            return
        self._root = tk.Tk()
        self._root.title("Fortnite AI Research — Dashboard")
        self._root.geometry("920x640")
        self._root.minsize(720, 520)
        self._root.configure(bg="#0f1115")
        self._build_ui()
        self._schedule_update()
        self._running = True
        try:
            self._root.mainloop()
        finally:
            self._running = False
            if self.on_close is not None:
                try:
                    self.on_close()
                except Exception:
                    pass

    def stop(self) -> None:
        if self._root is not None:
            try:
                self._root.quit()
            except Exception:
                pass
            try:
                self._root.destroy()
            except Exception:
                pass
            self._root = None

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        root = self._root
        # Top status bar
        top = ttk.Frame(root, padding=8)
        top.pack(fill=tk.X, side=tk.TOP)

        self._mode_lbl = ttk.Label(top, text="Mode: --", foreground="#d7e0f0")
        self._mode_lbl.pack(side=tk.LEFT, padx=(0, 14))
        self._ai_lbl = ttk.Label(top, text="AI: --", foreground="#d7e0f0")
        self._ai_lbl.pack(side=tk.LEFT, padx=(0, 14))
        self._train_lbl = ttk.Label(top, text="Training: --", foreground="#d7e0f0")
        self._train_lbl.pack(side=tk.LEFT, padx=(0, 14))
        self._model_lbl = ttk.Label(top, text="Model: --", foreground="#d7e0f0")
        self._model_lbl.pack(side=tk.LEFT, padx=(0, 14))
        self._safety_lbl = ttk.Label(top, text="Safety: --", foreground="#d7e0f0")
        self._safety_lbl.pack(side=tk.LEFT, padx=(0, 14))
        self._err_btn = ttk.Button(top, text="Errors", command=self._toggle_errors)
        self._err_btn.pack(side=tk.RIGHT)

        # Main grid: metrics on left, extended info on right
        body = ttk.Frame(root, padding=10)
        body.pack(fill=tk.BOTH, expand=True)

        left = ttk.LabelFrame(body, text="Live Metrics", padding=8)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 8))

        right = ttk.Frame(body, padding=6)
        right.grid(row=0, column=1, sticky="nsew")

        body.columnconfigure(0, weight=1)
        body.columnconfigure(1, weight=1)
        body.rowconfigure(0, weight=1)

        # Metric rows
        metric_specs = [
            ("Inference latency", "latency"),
            ("Decision latency", "decision_latency"),
            ("Confidence", "confidence"),
            ("Episode reward", "reward"),
            ("Episode length", "episode_length"),
            ("Steps/sec", "steps_per_sec"),
            ("Memory usage", "memory"),
            ("CPU usage", "cpu"),
            ("Low RAM mode", "low_ram"),
        ]
        row = 0
        self._metric_vars: dict[str, tk.StringVar] = {}
        for label, key in metric_specs:
            lab = ttk.Label(left, text=label, foreground="#9aa6b8")
            lab.grid(row=row, column=0, sticky="w", pady=(2, 0))
            var = tk.StringVar(value="--")
            val = ttk.Label(left, textvariable=var, foreground="#e6edf7", font=("Consolas", 10))
            val.grid(row=row, column=1, sticky="e", pady=(2, 0), padx=(8, 0))
            self._metric_vars[key] = var
            row += 1

        left.columnconfigure(0, weight=1)
        left.columnconfigure(1, weight=2)

        # Performance panel (right)
        perf = ttk.LabelFrame(right, text="Performance & Error Log", padding=8)
        perf.pack(fill=tk.BOTH, expand=True)

        self._perf_text = scrolledtext.ScrolledText(
            perf,
            height=16,
            width=46,
            bg="#151a21",
            fg="#c7d3e6",
            font=("Consolas", 9),
            state=tk.DISABLED,
            wrap=tk.WORD,
        )
        self._perf_text.pack(fill=tk.BOTH, expand=True)

        # Safety detail line
        self._safety_detail_lbl = ttk.Label(
            right,
            text="Control boundary: observation-only companion; simulator-only agent control.",
            foreground="#7fa1c4",
            wraplength=340,
            justify=tk.LEFT,
        )
        self._safety_detail_lbl.pack(fill=tk.X, pady=(8, 0))

    # ------------------------------------------------------------------
    # Periodic update
    # ------------------------------------------------------------------

    def _schedule_update(self) -> None:
        if self._root is None or not self._running:
            return
        self._update()
        self._root.after(self.update_interval_ms, self._schedule_update)

    def _update(self) -> None:
        try:
            state = self.agent_callback()
        except Exception as e:
            self._last_error = f"Agent callback error: {e}"
            state = {}

        self._apply_state(state)

    def _apply_state(self, state: dict[str, object]) -> None:
        mode = str(state.get("mode", "unknown"))
        self._mode_lbl.configure(text=f"Mode: {mode}")

        ai = str(state.get("ai_status", "unknown"))
        self._ai_lbl.configure(text=f"AI: {ai}")

        train = str(state.get("training_status", "none"))
        self._train_lbl.configure(text=f"Training: {train}")

        model = str(state.get("model", "none"))
        self._model_lbl.configure(text=f"Model: {model}")

        safe = str(state.get("safety_status", "ok"))
        color = "#7fd1a0" if safe.lower() in ("ok", "pass", "safe") else "#e58a8a"
        self._safety_lbl.configure(text=f"Safety: {safe}", foreground=color)

        # Metrics
        metrics = {
            "latency": self._fmt(state.get("inference_latency_ms"), "ms"),
            "decision_latency": self._fmt(state.get("decision_latency_ms"), "ms"),
            "confidence": self._fmt(state.get("confidence"), ""),
            "reward": self._fmt(state.get("episode_reward"), ""),
            "episode_length": self._fmt(state.get("episode_length"), " ticks"),
            "steps_per_sec": self._fmt(state.get("steps_per_sec"), " steps/s"),
            "memory": self._fmt_mem(state.get("memory_mb")),
            "cpu": self._fmt_cpu(state.get("cpu_percent")),
            "low_ram": "ON" if state.get("low_ram_mode") else "off",
        }
        for key, text in metrics.items():
            if key in self._metric_vars:
                self._metric_vars[key].set(text)

        # Error log
        self._append_perf(state)

    def _append_perf(self, state: dict[str, object]) -> None:
        lines: list[str] = []
        errors = state.get("errors")
        if errors:
            if isinstance(errors, str):
                lines.append(f"[ERROR] {errors}")
            else:
                for e in errors:
                    lines.append(f"[ERROR] {e}")
        if self._last_error:
            lines.append(f"[SYSTEM] {self._last_error}")

        if lines:
            self._perf_text.config(state=tk.NORMAL)
            for line in lines[-6:]:
                self._perf_text.insert(tk.END, line + "\n")
            self._perf_text.see(tk.END)
            self._perf_text.config(state=tk.DISABLED)

    # ------------------------------------------------------------------
    # Error panel toggle
    # ------------------------------------------------------------------

    def _toggle_errors(self) -> None:
        if self._perf_text is None:
            return
        current = self._perf_text.cget("state")
        if current == tk.NORMAL:
            self._perf_text.config(state=tk.DISABLED)
            self._err_btn.config(text="Errors")
        else:
            self._perf_text.config(state=tk.NORMAL)
            self._err_btn.config(text="Close Log")

    # ------------------------------------------------------------------
    # Default agent state (demo: simulator + policy loop)
    # ------------------------------------------------------------------

    def _default_agent_state(self) -> dict[str, object]:
        """Return a snapshot from a running agent/simulation if available."""
        return {
            "mode": "BALANCED",
            "ai_status": "running",
            "training_status": "none",
            "model": "dummy_local",
            "confidence": 0.85,
            "inference_latency_ms": 0.2,
            "decision_latency_ms": 0.2,
            "episode_reward": 0.0,
            "episode_length": 0,
            "steps_per_sec": 0.0,
            "memory_mb": self._estimate_memory_mb(),
            "cpu_percent": self._estimate_cpu(),
            "low_ram_mode": self._low_ram_enabled(),
            "safety_status": "ok",
            "errors": None,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _fmt(value: object, suffix: str) -> str:
        if value is None:
            return "--"
        try:
            if isinstance(value, float):
                if abs(value) < 10 and suffix:
                    return f"{value:.3f}{suffix}"
                return f"{value:.2f}{suffix}"
            if isinstance(value, int):
                return f"{value}{suffix}"
            return f"{value}{suffix}"
        except Exception:
            return "--"

    def _fmt_mem(self, value: object) -> str:
        if value is None:
            return "--"
        try:
            mb = float(value)
            return f"{mb:.1f} MB"
        except Exception:
            return "--"

    def _fmt_cpu(self, value: object) -> str:
        if value is None:
            return "--"
        try:
            return f"{float(value):.1f}%"
        except Exception:
            return "--"

    def _estimate_memory_mb(self) -> float:
        if psutil is not None:
            try:
                return psutil.Process().memory_info().rss / (1024 * 1024)
            except Exception:
                pass
        return 0.0

    def _estimate_cpu(self) -> float:
        if psutil is not None:
            try:
                return psutil.Process().cpu_percent(interval=None)
            except Exception:
                pass
        return 0.0

    def _low_ram_enabled(self) -> bool:
        try:
            from core.low_ram import LOW_RAM_MODE
            return bool(LOW_RAM_MODE.enabled)
        except Exception:
            return False


def main() -> None:
    """Launch the lightweight dashboard."""
    dash = Dashboard()
    try:
        dash.start()
    except Exception as e:
        print(f"UI failed to start: {e}")
        raise


if __name__ == "__main__":
    main()
