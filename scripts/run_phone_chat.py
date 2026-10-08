"""Run Peter's phone chat transport with an existing responder callable.

Example:
  python scripts/run_phone_chat.py --responder my_peter_module:respond

The responder receives one text message and returns one text reply.
"""
from __future__ import annotations
import argparse
import importlib
from core.chat import PhoneChatServer

def load_responder(spec: str):
    module_name, sep, attr = spec.partition(":")
    if not sep or not module_name or not attr:
        raise ValueError("responder must look like module_name:function_name")
    return getattr(importlib.import_module(module_name), attr)

def main() -> None:
    parser=argparse.ArgumentParser(description="Start Peter's iPhone text chat server")
    parser.add_argument("--responder",required=True,help="Python callable: module:function")
    parser.add_argument("--ip",required=True,help="PC LAN IPv4 address shown to the iPhone")
    parser.add_argument("--port",type=int,default=8765)
    args=parser.parse_args()
    server=PhoneChatServer(load_responder(args.responder),port=args.port)
    print("Peter phone chat:",server.chat_url(args.ip),flush=True)
    print("Keep this URL private and use the same Wi-Fi network on the iPhone.",flush=True)
    try: server.serve_forever()
    except KeyboardInterrupt: pass
    finally: server.stop()

if __name__ == "__main__": main()
