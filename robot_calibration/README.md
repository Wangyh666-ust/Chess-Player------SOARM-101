# Robot measurement collection

Start with joint calibration and a read-only recording, then measure the response of one joint. For pickup experiments, record current/load telemetry and video while replaying a locally taught trajectory. This collector is separate from the frozen PyBullet runtime. It does not plan grasps from board coordinates, estimate forces or fit a physics model.

## Files

- `AGENTS.md`: instructions for an agent assisting with installation, tests and supervised hardware collection.
- `physical_measurements.json`: the user's measurements of 11 November 2026, in grams and millimetres. Grid units are confirmed as centimetres and converted to millimetres. Unknown force values remain `null`. Nothing is automatically applied to Blender or PyBullet.
- `collect_robot_calibration.py`: passive recording (`read`), a small relative joint sweep (`joint_response`), and taught trajectory replay (`pickup`). Defaults to a mock backend that never opens a serial port.
- `pickup_trial.py`: validates/interpolates taught pickups and records camera video in a separate thread.
- `pickup.example.json`: intentionally blank, non-executable waypoint template; fill measured local poses.
- `annotate_separation.py`: exports numbered video frames and saves a human-reviewed separation interval.
- `limits.example.json`: blank template for a verified local motion envelope. It intentionally cannot enable hardware motion until every interval is filled.
- `requirements-hardware.txt`: optional, separately installed LeRobot dependencies.
- `test_collect_robot_calibration.py` and `test_pickup_trial.py`: tests without hardware, including failure and cleanup paths. Fake camera tests do not validate real camera drivers/codecs.

The 198 mm grid measurement and eight 24.5 mm cells differ by 2 mm. Confirm the seam's inclusion and axis orientation before deriving square centres. The new king height is 43 mm; earlier geometry was 45 mm. Top-ring widths do not specify the height at which to grasp.

## Handoff to a teammate using Codex

Give Codex this starting request, explicitly naming the local instructions even
if the session starts at the repository root:

> Read robot_calibration/AGENTS.md and robot_calibration/README.md. Run the tests
> and a mock joint-response recording in a new output directory. Help me install
> the pinned dependencies in the separate environment and establish passive
> hardware readings and camera recording. Before physical movement, review the
> measured limits and locally taught poses with me and wait for my readiness
> confirmation. Preserve failures and report what has actually been verified.

The teammate still needs to supply the connected port, robot ID, calibrated arm,
camera device, measured local limits and rehearsed pickup poses. The scripts do
not provide a teleoperation interface for teaching those poses; use the local
arm software as described below. Installation and hardware compatibility must
be checked on that computer before collecting evidence.

Return the complete trial directories, including failed trials, through your
shared data storage. Keep the videos, frame timestamps, sample logs, metadata,
trajectory snapshots and separation reviews together. Also include the local
limits and taught trajectory files, the exact commands, `git rev-parse HEAD`,
`python --version`, and `python -m pip freeze` from the hardware environment.
Record the camera view, piece identity, board square, grasp height and any changes
to placement or support in a short session note. These details make later
comparisons interpretable. Outputs belong under the ignored `runs/` directory,
not in the source commit.

## Try the recorder without installing anything

From the repository root:

```bash
python3 robot_calibration/collect_robot_calibration.py \
  --mode joint_response --joint shoulder_pan \
  --output runs/robot-calibration/mock-001

python3 -m unittest discover -s robot_calibration -v
```

Mock runs execute quickly using synthetic positions. `planned_s` is the simulated sample schedule; host timestamps reflect the actual fast execution. Mock files are marked as mock and must never be used as physical calibration evidence.

## Establish the first hardware connection

Use an isolated Python 3.12 environment, separate from the simulator's dependencies:

```bash
python3.12 -m venv .venv-robot-calibration
source .venv-robot-calibration/bin/activate
python -m pip install -r robot_calibration/requirements-hardware.txt
lerobot-find-port
```

Follow the [SO-101 setup and calibration guide](https://huggingface.co/docs/lerobot/so101). If motor IDs have not been configured, follow its motor setup process first. Do not blindly reassign an already assembled arm's IDs.

Calibrate the follower with the actual port and a consistent ID (replace the example port):

```bash
lerobot-calibrate \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM0 \
  --robot.id=chess_arm
```

The guided tool needs manual movement. Support the arm as instructed. Calibration files are managed by LeRobot outside this project environment; keep a copy and reuse the same robot ID. Removing the virtual environment later does not remove those calibration files or recordings.

Neither dependency installation nor physical calibration has been performed by adding this module. The hardware adapter targets LeRobot 0.7.0 and refuses another version until reviewed. Its bus calls were checked against the versioned upstream source; actual motor communication is still unverified.

## Record joint readings without commanding motion

Close other applications controlling the arm. This mode requires saved calibration matching the motors. It reads positions and existing settings; it does not send position goals, configure PID gains, enable torque or disable torque on exit.

```bash
python robot_calibration/collect_robot_calibration.py \
  --backend lerobot --mode read \
  --port /dev/ttyACM0 --robot-id chess_arm \
  --duration-s 5 --output runs/robot-calibration/read-001
```

The five arm readings are LeRobot-calibrated **degrees**, not PyBullet joint coordinates. The gripper is **normalized 0–100**, not degrees, millimetres or force. The mapping to the simulator must be checked separately.

The collector opens the motor bus directly rather than calling the higher-level robot connection routine, because that routine can configure motor gains. Existing gains and torque limits are logged, not overwritten. Add `--telemetry` to read current/load and `--camera 0` to capture video, including in read mode. Do not run a second controller on the same port; read mode is not a concurrent teleoperation recorder.

## Prepare a small movement test

1. Secure the base. Remove chess pieces and clear the intended small sweep. Start with the empty arm in a supported pose, with all motor torque off. Never force a powered joint by hand.
2. Use calibrated observations and manual/teleoperated inspection to choose a collision-free local envelope. Copy `limits.example.json` to `runs/robot-calibration/limits.json` and replace every `null` with a measured limit. These are local experiment limits, not automatically discovered mechanical limits.
3. Check that the current pose and the selected joint's full positive/negative excursion fit inside that envelope. Joint bounds alone do not establish that a pose is clear of the table, itself or a person.
4. Keep physical power-off accessible and support the arm when the run finishes or aborts. The collector disables torque after a motion run, which can let the arm fall under gravity. Do not use the joint-response test while holding a piece. Pickup trials can also drop the piece on an abort; use a small lift over a protected area.

For the first run, use a 1-degree amplitude and one cycle:

```bash
python robot_calibration/collect_robot_calibration.py \
  --backend lerobot --mode joint_response --execute \
  --port /dev/ttyACM0 --robot-id chess_arm \
  --joint shoulder_pan --amplitude-deg 1 \
  --ramp-s 3 --hold-s 1 --cycles 1 --hz 30 \
  --limits runs/robot-calibration/limits.json \
  --output runs/robot-calibration/shoulder-pan-001
```

The selected joint follows: initial hold → positive ramp/hold → centre ramp/hold → negative ramp/hold → initial ramp/hold. Other joints, including the gripper, hold their starting readings. There is no absolute home movement and no automatic retry.

Before enabling torque, the script requires position mode and torque off, checks the saved calibration and all pose bounds, and seeds the motor goals with current readings. It then enables torque without altering gains. Only the five arm joints can be selected for this first response test; automated gripper-gap calibration is a later mode.

Joint-response commands are capped at 5 degrees of excursion and 5 degrees/second. These are software caps, not a claim that every permitted trajectory is safe. Samples use a monotonic clock. Excessive lateness, out-of-bound readings, excessive tracking error, invalid values or communication errors abort the run and preserve partial data. Ctrl+C and SIGTERM use the same cleanup path.

## Current and load telemetry

```bash
python robot_calibration/collect_robot_calibration.py \
  --backend lerobot --mode read --telemetry --camera 0 --hz 10 \
  --port /dev/ttyACM0 --robot-id chess_arm \
  --duration-s 10 --output runs/robot-calibration/telemetry-001
```

The collector records `Present_Current`, `Present_Load`, `Present_Voltage`, `Present_Temperature` and `Goal_Position` for all six motors. Current/load are kept as register values, without guessing sign encoding, amperes, joint torque or gripper force. Goal positions use the same degree/normalized-gripper units as commands. Each register read has its own timestamp interval; these reads are sequential, not simultaneous.

These register names are supported by the [STS3215 driver definition](https://docs.rs/rustypot/latest/rustypot/servo/feetech/sts3215/index.html). The actual firmware/driver must answer every requested register. A failed preflight read prevents movement; a failed read during motion aborts and retains partial evidence. We do not replace missing values with zero. Start at 10 Hz because the extra reads increase bus traffic; the observed timestamps determine the effective sample rate.

## Teach a pickup before repeating it

Copy `pickup.example.json` to `runs/robot-calibration/pawn-pickup.json`. Fill it with poses measured on the real robot after calibration. Never copy simulation joint coordinates or the synthetic test fixture into hardware.

Use the teammate's working teleoperation/jogging software to establish poses. Stop that software before running the collector on the same port. A short `--mode read` run saves all six current values in `metadata.json.initial_positions`; copy that dictionary into the corresponding waypoint. A saved encoder pose does not prove that the path between poses is collision-free: rehearse the complete path locally, slowly and initially without the piece.

The required phases are:

| Phase | Pose to supply |
| --- | --- |
| `start` | Supported, repeatable pose with the empty gripper open, near the piece |
| `approach` | Fingers around the piece at a measured grasp height; same opening as start |
| `close` | Same arm angles as approach, smaller tested gripper opening |
| `lift` | Locally verified small upward movement; retain the closing command |
| `hold` | Identical pose to lift, later timestamp to establish a hold period |
| `lower` | Back to the supported piece position, gripper still closed |
| `release` | Same arm angles as lower, larger gripper opening |
| `return` | Exactly the initial pose and opening |

Times are seconds since trajectory start. Increase segment durations when needed; limits are 5 deg/s for arm joints, 20 normalized units/s for the gripper and 20 degrees total arm excursion from start. The final pose is held for one second. A brief alignment ramp removes a starting discrepancy of at most 0.5 units; larger discrepancies are rejected. Software checks do not establish a vertical Cartesian lift or board clearance. Verify those physically. The interpolated path uses joint-space linear ramps, not the simulator's collision-checked IK planner.

## Repeat matched pickup trials

```bash
python robot_calibration/collect_robot_calibration.py \
  --backend lerobot --mode pickup --execute \
  --port /dev/ttyACM0 --robot-id chess_arm \
  --trajectory runs/robot-calibration/pawn-pickup.json \
  --limits runs/robot-calibration/limits.json \
  --piece-id white-pawn-1 --condition magnetic-board \
  --camera 0 --camera-fps 30 --hz 10 --trials 3 \
  --output runs/robot-calibration/pawn-series-001
```

Pickup mode automatically requires telemetry and, on hardware, a camera. It does not begin until telemetry and a first video frame are available. Before every trial the operator must reset the same piece to the marked square, offset and orientation, check the start pose and torque-off state, then type `ready`. A failure stops the series. There is no automatic retry. Motor gains are preserved and logged.

Each repetition gets `trial-001`, `trial-002`, etc. A copied trajectory plus its SHA-256, calibration snapshot, limits, piece ID and condition label accompany each trial. Keep the same physical piece, grasp height, closing command, path, lift speed, board location and camera view. Manual placement is not guaranteed identical by the code; mark positions and review the video. If comparing a non-magnetic reference, match its top-surface height and path, and label the changed condition. Do not infer attraction solely from a current difference without checking repeatability and confounding factors.

`status: completed` means command playback finished. It does not prove the piece was grasped or detached; `pickup_outcome` remains `unreviewed` until visual inspection. Real pose teaching and trials must be done with the teammate's arm. No real pickup has been validated by the mock tests.

## Review the instant of separation

Aim the camera low enough to see the piece base and board contact. Each trial saves `video.avi` and `frames.jsonl`. The video uses MJPEG; timestamps in the JSONL are authoritative for alignment, since nominal playback FPS may differ from capture timing. Camera and motor reads share one host monotonic time origin. This is software timestamp alignment, not hardware synchronization: exposure and camera buffering latency remain unknown.

Watch the video, then export numbered frames around the event (example range only):

```bash
python robot_calibration/annotate_separation.py \
  --trial runs/robot-calibration/pawn-series-001/trial-001 \
  --export-frames 20 50
```

Inspect the JPEGs and record the last frame with contact and the first frame with a visible gap. Replace the example frame numbers with observed values:

```bash
python robot_calibration/annotate_separation.py \
  --trial runs/robot-calibration/pawn-series-001/trial-001 \
  --reviewer teammate --outcome separated \
  --last-contact-frame 32 --first-clear-frame 33
```

This writes a new `separation.json` with frame indices, a host-time interval, matching telemetry sample indices and evidence hashes. It never overwrites a review. Use `--outcome no-separation` or `--outcome occluded` without frame indices when appropriate. A frame interval is more honest than claiming an exact physical separation time. No automatic computer-vision separation detector is included.

A blocked driver call, killed process, disconnected cable or failed torque-disable command cannot be stopped reliably by this Python loop. Physical power-off is the fallback. There is no hardware watchdog, torque/force control, collision model or current-based contact detector in this collector.

## Read the results

Each new output directory contains:

- `metadata.json`: backend, status, settings, units, starting positions, limits, hardware calibration/settings, source hash and a tracking summary for motion runs. Failure and cleanup status are explicit.
- `samples.jsonl`: one record per sample, with requested joint positions, measured positions, phase and host-side command/read timestamps. Read-only records have `command: null`.
- `video.avi` and `frames.jsonl`, when a camera is selected: video plus frame acquisition intervals on the telemetry clock.
- `separation.json`, after review: human-labelled separation or uncertainty; no force estimate.

The metadata contains the full calibration snapshot; the record is not just tied to an ID that might be recalibrated later. Each run uses a new directory, and existing results are never overwritten. Outputs under `runs/` are already ignored by Git.

Read timings bound the host acquisition interval; they are not exact motor sensor timestamps. The tracking RMSE is a diagnostic, not an automatically estimated motor delay. External video/markers will be needed to distinguish encoder tracking from mechanical play and gripper position error. An offline fitting module can be added after real recordings exist.

## Force measurements can wait

A force gauge is a physical push/pull instrument, not a software setting. If none is available, leave contact values unknown for now. Pickup recordings can reveal repeatable load changes and separation timing, but do not by themselves yield magnetic attraction in newtons. Do not convert a motor current or torque setting directly into a measured jaw force.

Board alignment through ArUco still needs camera calibration and a camera-to-robot relationship. A single known arm pose does not generally determine all unknown camera/tool transforms. This module does not solve that calibration yet.
