from core.chat.peter_responder import respond


def test_basic_chat():
    assert "Peter" in respond("hello")


def test_move_classification():
    assert "movement" in respond("help me move").lower()


def test_empty_message():
    assert "here" in respond("").lower()
