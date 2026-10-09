from core.training.virtual_input import MouseState, VirtualInputController


def test_keyboard_state_is_in_memory_only():
    controller = VirtualInputController()
    controller.key_down("w")
    controller.key_down("Space")

    snapshot = controller.snapshot()

    assert snapshot.keys == frozenset({"W", "SPACE"})
    assert isinstance(snapshot.mouse, MouseState)


def test_mouse_state_is_clamped_and_snapshot_is_immutable():
    controller = VirtualInputController()
    controller.set_mouse(dx=5000, dy=-5000, left=True, wheel=100)

    snapshot = controller.snapshot()

    assert snapshot.mouse.dx == 1000
    assert snapshot.mouse.dy == -1000
    assert snapshot.mouse.left is True
    assert snapshot.mouse.wheel == 20


def test_clear_resets_keyboard_and_mouse():
    controller = VirtualInputController()
    controller.key_down("W")
    controller.set_mouse(dx=10, dy=20, right=True)

    controller.clear()

    snapshot = controller.snapshot()
    assert snapshot.keys == frozenset()
    assert snapshot.mouse == MouseState()
