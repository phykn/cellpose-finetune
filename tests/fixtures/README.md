`cellpose_a54cb488.npz` contains synthetic instance masks and the float32
outputs of the unmodified `masks_to_flows_gpu(..., device=cpu)` from
[Cellpose dynamics.py at a54cb488](https://github.com/MouseLand/cellpose/blob/a54cb48849b7e225a81e8e43dcb042d42427f543/cellpose/dynamics.py).
The cases cover bounding boxes with odd origins and even dimensions (rounding
ties), a concave object, image borders, and empty labels. No biological data
or model weights are included.

Regenerate from a locally inspected copy of that exact source:

```powershell
python scripts/generate_flow_fixture.py <upstream-dynamics.py> tests/fixtures/cellpose_a54cb488.npz
```

The script executes only the four upstream flow functions, avoiding unrelated
Cellpose imports. It does not download or change the installed dependencies.
Cellpose attribution and license are in `THIRD_PARTY_NOTICES.md` and `LICENSE`.
