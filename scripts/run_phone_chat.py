"""Double-click-friendly launcher for Peter's iPhone text chat."""
from __future__ import annotations

import argparse
import socket

from core.chat import PhoneChatServer
from core.chat.peter_responder import respond


def detect_lan_ip() -> str:
    """Best-effort LAN IPv4 detection without sending application data."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect(("192.0.2.1", 80))  # documentation-only address; no packet is sent
        ip = sock.getsockname()[0]
        if ip and not ip.startswith("127."):
            return ip
    except OSError:
        pass
    finally:
        sock.close()
    return "127.0.0.1"


def main() -> None:
    parser = argparse.ArgumentParser(description="Start Peter's iPhone text chat server")
    parser.add_argument("--ip", help="PC LAN IPv4 address (auto-detected if omitted)")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    ip = args.ip or detect_lan_ip()
    server = PhoneChatServer(respond, port=args.port)
    print("\nPeter Phone Chat is starting…", flush=True)
    print(f"On your iPhone (same Wi-Fi), open: {server.chat_url(ip)}", flush=True)
    print("Keep that private link secret. Leave this window open while chatting.", flush=True)
    print("Press Ctrl+C to stop Peter.\n", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()


if __name__ == "__main__":
    main()
