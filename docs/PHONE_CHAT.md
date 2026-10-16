# Peter phone chat

Peter now has a built-in, text-only phone chat responder.

## Start

On the Peter PC, from the repository root:

```powershell
ipconfig
python scripts/run_phone_chat.py --ip YOUR_PC_IPV4
```

Replace `YOUR_PC_IPV4` with the PC's IPv4 address, for example `192.168.1.20`.

The terminal prints a private URL containing a temporary session token. Open that URL on the iPhone while both devices are on the same Wi-Fi.

No microphone or audio is required.

## Safety boundary

The responder is local and dependency-free. It does not inject keyboard/mouse input into Windows or the Fortnite client. Virtual input remains available only to the custom/authorized simulator or Neo integration boundary.

Do not port-forward 8765 or expose the chat server to the public internet.
