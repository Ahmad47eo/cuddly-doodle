"""Tiny LAN text-chat server for talking to Peter from an iPhone."""
from __future__ import annotations
import json
import secrets
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Protocol
from urllib.parse import parse_qs, urlparse

class ChatResponder(Protocol):
    def __call__(self, message: str) -> str: ...

_PAGE = """<!doctype html><html lang="en"><head><meta name="viewport" content="width=device-width,initial-scale=1"><meta name="theme-color" content="#111827"><title>Peter Chat</title><style>*{box-sizing:border-box}body{margin:0;background:#0b1020;color:#f3f4f6;font:16px system-ui,sans-serif}main{max-width:720px;margin:auto;min-height:100vh;display:flex;flex-direction:column;padding:16px}h1{font-size:20px;margin:4px 0 12px}.status{color:#9ca3af;font-size:13px;margin-bottom:10px}#messages{flex:1;overflow:auto;padding:4px 0 12px}.msg{padding:10px 12px;border-radius:14px;margin:8px 0;white-space:pre-wrap;overflow-wrap:anywhere;max-width:88%}.you{margin-left:auto;background:#2563eb}.peter{background:#1f2937}.system{color:#9ca3af;font-size:13px;text-align:center}form{display:flex;gap:8px;position:sticky;bottom:0;background:#0b1020;padding:10px 0 2px}input{min-width:0;flex:1;padding:13px;border-radius:12px;border:1px solid #374151;background:#111827;color:#fff;font-size:16px}button{border:0;border-radius:12px;padding:0 18px;background:#2563eb;color:#fff;font-weight:700;font-size:16px;min-width:76px}button:disabled{opacity:.5}</style></head><body><main><h1>Peter 🤖</h1><div class="status">Text chat only — no microphone needed.</div><div id="messages" aria-live="polite"></div><form id="form"><input id="input" autocomplete="off" placeholder="Message Peter…" aria-label="Message Peter"><button id="send">Send</button></form></main><script>const token=decodeURIComponent(location.hash.slice(1));const box=document.querySelector('#messages'),input=document.querySelector('#input'),send=document.querySelector('#send');function add(who,text){const d=document.createElement('div');d.className='msg '+who;d.textContent=text;box.appendChild(d);box.scrollTop=box.scrollHeight}async function submit(){const message=input.value.trim();if(!message)return;input.value='';send.disabled=true;add('you',message);try{const r=await fetch('/chat?token='+encodeURIComponent(token),{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({message})});const j=await r.json();if(!r.ok)throw new Error(j.error||'Request failed');add('peter',j.reply)}catch(e){add('system','Could not reach Peter: '+e.message)}finally{send.disabled=false;input.focus()}}document.querySelector('#form').addEventListener('submit',e=>{e.preventDefault();submit()});input.focus();</script></body></html>"""

class PhoneChatServer:
    def __init__(self, responder: ChatResponder, host: str = "0.0.0.0", port: int = 8765, token: str | None = None) -> None:
        if not callable(responder): raise TypeError("responder must be callable")
        if not 1024 <= port <= 65535: raise ValueError("port must be between 1024 and 65535")
        self.responder, self.host, self.port = responder, host, port
        self.token = token or secrets.token_urlsafe(18)
        self._server: ThreadingHTTPServer | None = None

    def start(self) -> ThreadingHTTPServer:
        responder, token = self.responder, self.token
        class Handler(BaseHTTPRequestHandler):
            def _json(self, status: int, payload: dict[str, str]) -> None:
                raw=json.dumps(payload).encode("utf-8"); self.send_response(status); self.send_header("Content-Type","application/json; charset=utf-8"); self.send_header("Content-Length",str(len(raw))); self.send_header("Cache-Control","no-store"); self.end_headers(); self.wfile.write(raw)
            def do_GET(self) -> None:
                if urlparse(self.path).path != "/": self._json(404,{"error":"Not found"}); return
                raw=_PAGE.encode("utf-8"); self.send_response(200); self.send_header("Content-Type","text/html; charset=utf-8"); self.send_header("Content-Length",str(len(raw))); self.send_header("Cache-Control","no-store"); self.end_headers(); self.wfile.write(raw)
            def do_POST(self) -> None:
                parsed=urlparse(self.path); qtoken=parse_qs(parsed.query).get("token",[""])[0]
                if parsed.path!="/chat" or qtoken!=token: self._json(401,{"error":"Invalid chat token"}); return
                try:
                    length=min(int(self.headers.get("Content-Length","0")),16384); payload=json.loads(self.rfile.read(length) or b"{}"); message=str(payload.get("message","")).strip()
                    if not message: raise ValueError("message is empty")
                    if len(message)>2000: raise ValueError("message is too long")
                    reply=str(responder(message)).strip() or "I don't have a response yet."; self._json(200,{"reply":reply[:8000]})
                except (ValueError,json.JSONDecodeError) as exc: self._json(400,{"error":str(exc)})
                except Exception: self._json(500,{"error":"Peter failed to respond"})
            def log_message(self, fmt: str, *args: object) -> None: return
        self._server=ThreadingHTTPServer((self.host,self.port),Handler); self._server.daemon_threads=True; return self._server
    def serve_forever(self) -> None:
        (self._server or self.start()).serve_forever()
    def stop(self) -> None:
        if self._server: self._server.shutdown(); self._server.server_close(); self._server=None
    def chat_url(self, local_ip: str) -> str:
        return f"http://{local_ip}:{self.port}/#{self.token}"
