# Neo integration boundary

Peter can expose keyboard and mouse actions as virtual state to a permitted
custom training environment or simulator.

The integration boundary is intentionally:

Peter policy
  -> VirtualInputController
  -> VirtualInputSnapshot
  -> authorized/custom environment adapter

VirtualInputController never calls OS input APIs, never moves the real mouse,
never presses real keys, and never injects input into a real game client.

This makes it suitable for testing Peter's movement, camera, building, and
strategy policies in a simulator/custom environment that explicitly exposes
an input API.

## Example

from core.training.virtual_input import VirtualInputController

inputs = VirtualInputController()
inputs.key_down("W")
inputs.key_down("SPACE")
inputs.set_mouse(dx=12, dy=-4, left=False)

snapshot = inputs.snapshot()
# Pass snapshot to the authorized simulator adapter.

A future Neo adapter should consume VirtualInputSnapshot through Neo's
documented/custom-server interface. It must not translate the snapshot into
desktop/OS automation or Fortnite-client injection.
