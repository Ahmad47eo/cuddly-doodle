from core.chat.phone_chat import PhoneChatServer

def test_chat_url_contains_token():
    server=PhoneChatServer(lambda message: "Peter: "+message, port=8765, token="test-token")
    assert server.chat_url("192.168.1.20")=="http://192.168.1.20:8765/#test-token"

def test_token_generated():
    assert len(PhoneChatServer(lambda message: "ok").token)>=20
