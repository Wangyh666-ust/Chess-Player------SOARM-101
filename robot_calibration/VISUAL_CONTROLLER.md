# Local visual jogging

From the repository root, launch the **mock** panel (no devices opened):

```bash
.venv-control/bin/python -m robot_calibration.collection.visual_controller \
  --output "runs/robot-calibration/visual-mock-$(date +%Y%m%d-%H%M%S-%N)"
```

Select Joint or Wrist mode. Each press sends one selected increment (0.25–1 degree or 0.05–1 mm); holding continues on the 10 Hz feedback schedule without a
key-repeat delay. The gripper has a separate increment in calibrated normalized
0–100 units. A short click released before feedback finishes becomes exactly
one tap. There is no FIFO of clicks or held movements: newer input replaces old
input. Release cancels held continuation and any in-flight IK for that hold;
a single already-sent increment can finish. Focus loss, pointer exit, a mode
change and a stale UI heartbeat cancel unsent work. The UI heartbeat expires
after 250 ms. No input resumes automatically after expiration.

Hold commands are limited to 0.5 degrees/normalized units per 100 ms, and wrist
steps to 0.5 mm per 100 ms. Smaller selected steps give slower holds. IK moves
that would exceed the joint increment bound are rejected, not subdivided into
queued motion. Choose a smaller step. Tap targets are at most one degree/unit
from the preceding goal. Repeated deliberate taps have no click cooldown.

The controlled wrist point is the **wrist_roll axis, at the gripper_link
origin**, not either fingertip. The coordinates are the URDF **base_link**
frame: forward +X, left +Y, up +Z. These are model axes, not camera/screen axes;
verify the installed base orientation before physical operation. The UI shows
measured joint positions and encoder-derived wrist XYZ in mm. Cartesian values
are model estimates, not independently verified physical measurements.

Wrist IK uses the existing URDF FK/IK and calibrated hardware/model mapping,
starting from each fresh measured pose. It preserves the orientation at the
start of a press within 3 degrees, rejects position residuals over 0.3 mm (or
20% of the step, whichever is smaller), and requires a 5-degree model joint
margin. A five-axis arm cannot maintain arbitrary full orientation during every
translation; infeasible requests stop the hold with a visible rejection.
Joint limits and IK are not collision detection.

**Esc pauses jogging.** Release the held control and press a jog control to continue.
**STOP ends the session.** Hardware enters the existing supervised recovery in
the launching terminal. Closing the window, terminal input, or a guard failure also enters recovery. Normal release
holds the last bounded target without torque-off. Bus failures make physical
holding uncertain; the owner stays in supervised recovery. Software torque
release remains disabled. Support the arm and any piece, remove physical motor
power yourself, then click **Confirm physical power is off…** and confirm in the panel. The owner terminal also accepts `power-off-supported`.

## Hardware preparation and launch

Read [AGENTS.md](AGENTS.md), [the handoff](CODEX_ARM_HANDOFF.md) and
[the operator guide](ARM_OPERATOR_GUIDE.md) first. Resolve any existing serial
owner without killing its controller. This panel starts a new passive session;
it does not take over an existing powered controller.

Install the numerical solver only in the dedicated hardware environment:

```bash
.venv-robot-calibration/bin/python -m pip install -r robot_calibration/requirements-controller.txt
```

Prepare a local session JSON from a **fresh passive capture**, with these fields:

- `hardware`: the exact adapter snapshot (calibration, gains, mode and all six torque flags).
- `initial`: all six measured normalized positions (arm degrees, gripper 0–100).
- `start_raw`: all six measured integer encoder positions.
- `profile`: absolute path to the reviewed joint mapping profile.
- `limits`: absolute path to a filled `configs/limits.example.json`-format file.
- `recovery`: absolute path to a filled `configs/recovery.example.json`-format file.

Use measured conservative limits for a collision-free, operator-reviewed jog
volume, including every joint and the gripper. The mock limits and identity
mapping are deliberately created only in mock mode; they are not hardware
configuration. Do not repurpose a powered hold's hardware snapshot as a passive
startup snapshot. No gains, IDs, calibration or torque-off commands are written.

```bash
.venv-robot-calibration/bin/python -m robot_calibration.collection.visual_controller \
  --backend lerobot --session /absolute/path/to/reviewed-visual-session.json \
  --port /dev/serial/by-id/YOUR_VERIFIED_DEVICE --robot-id YOUR_SAVED_ROBOT_ID \
  --output "runs/robot-calibration/visual-hardware-$(date +%Y%m%d-%H%M%S-%N)"
```

Launching the hardware app confirms mechanical support and accessible power-off.
Click OK after checking clearance for the displayed limits to start powered hold.
The `scripts/jog-arm` shortcut forwards the same options to the Python entry
point. The original bench session also accepts a fresh stationary start inside
its reviewed envelope on subsequent launches.
The controller checks ownership, exact hardware identity and the start within
two ticks, prepares recovery, and reuses the single energizing seed and bounded
startup verification. During operation it checks complete feedback, goals,
torque, faults, temperatures, position envelopes, eight-tick tracking on all
six joints, mode, lock, a 0.5-second read deadline and 0.2-second schedule
lateness. One worker owns all runtime bus reads/writes. Input contains no
hardware operations. It continues monitoring while no button is pressed.

The default jogging session records raw feedback and commands to
`samples.jsonl`, startup and configuration evidence in that trace and
`metadata.json`, and supervised recovery in `recovery.jsonl`. The calibration
recording modes also capture fixed and wrist video and attach camera and
registration evidence to each mark. Use the [fixed-tip worksheet](FIXED_TIP_WORKSHEET.md)
for the 23-point sequence and keep the entire output directory.

## Offline verification

```bash
.venv-control/bin/python -m unittest discover -s robot_calibration/tests -v
```

The new tests exercise taps including short clicks, bounded holds, release,
stale input, gripper units/limits, stop, release during IK, measured-pose IK,
infeasible targets and feedback failure. With a display available, a real Tk
window also exercises mouse and keyboard events, mode switching, focus loss
and Stop. Without a display that UI test skips explicitly. SciPy is needed for
controller tests; the original stdlib test command skips this optional module
when its numerical dependencies are absent.

The [calibration results](README.md#what-has-been-validated) describe the recorded
supervised bench checks and the scope of the physical position measurements.
