"""Run Peter's iPhone text chat server.

Example:
  python scripts/run_phone_chat.py --ip 192.168.1.20
"""
from __future__ import annotations

import argparse

from core.chat import PhoneChatServer
from core.chat.peter_responder import respond


def main() -> None:
    parser = argparse.ArgumentParser(description="Start Peter's iPhone text chat server")
    parser.add_argument("--ip", required=True, help="PC LAN IPv4 address shown to the iPhone")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    server = PhoneChatServer(respond, port=args.port)
    print("Peter phone chat:", server.chat_url(args.ip), flush=True)
    print("Keep this URL private and use the same Wi-Fi network on the iPhone.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()


if __name__ == "__main__":
    main()
