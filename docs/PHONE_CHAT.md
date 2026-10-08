# iPhone text chat for Peter

This is a dependency-free, text-only phone chat transport. No microphone, audio, OS input injection, or real Fortnite control is involved.

Architecture: iPhone browser -> local HTTP chat -> Peter responder -> chat reply.

Pass Peter's existing response function into PhoneChatServer:

    from core.chat import PhoneChatServer
    server = PhoneChatServer(responder=peter.respond)
    print(server.chat_url("192.168.1.20"))
    server.serve_forever()

Use the PC's LAN IP address in place of 192.168.1.20. The server generates a random session token in the URL. Keep that URL private while the server is running.

Setup: start Peter and the chat server on Windows, put the iPhone and PC on the same Wi-Fi, then open the printed URL on the iPhone. If Windows Firewall asks, allow the Python server on Private networks only. Do not expose port 8765 to the public internet.

The responder should call Peter's existing model/router. Keep this transport separate from VirtualInputController and RealFortniteControlInterface.
