# Calibration tool reference

Start with the [calibration and recalibration guide](README.md). Dated paths below refer to local experiment archives.

See [ARCHITECTURE.md](ARCHITECTURE.md) for the initial commissioning workflow,
reconnect checks, protractor measurements, moving gripper centre and independent
final validation. Those orchestrated workflows are planned; the commands below
are the existing measurement and offline-analysis tools.

The code is now grouped into `collection/`, `analysis/`, `configs/` and `tests/`.
Run commands from the repository root with `python -m ...`; the previous flat
script paths have moved. Shared record helpers live in `records.py`. Hardware
dependencies and analysis dependencies remain separate at this directory's root.

Start with joint calibration and a read-only recording, then measure the response of one joint. For pickup experiments, record current/load telemetry and video while replaying a locally taught trajectory. This collector is separate from the frozen PyBullet runtime. It does not plan grasps from board coordinates, estimate forces or fit a physics model.

## Files

- `configs/board_target.json`: measured custom ArUco target, including 40 measured corners (±0.5 mm) and verified decoded corner order.
- `camera_calibration.py`: offline fixed/wrist intrinsics fitting, held-out reprojection checks and original/undistorted previews. Results remain candidates.
- `AGENTS.md`: instructions for an agent assisting with installation, tests and supervised hardware collection.
- `configs/physical_measurements.json`: the user's measurements of 11 November 2026, in grams and millimetres. Grid units are confirmed as centimetres and converted to millimetres. Unknown force values remain `null`. Nothing is automatically applied to Blender or PyBullet.
- `collection/collect_robot_calibration.py`: passive recording (`read`), a small relative joint sweep (`joint_response`), and taught trajectory replay (`pickup`). Defaults to a mock backend that never opens a serial port.
- `collection/pickup_trial.py`: validates/interpolates taught pickups and records camera video in a separate thread.
- `configs/pickup.example.json`: intentionally blank, non-executable waypoint template; fill measured local poses.
- `collection/annotate_separation.py`: exports numbered video frames and saves a human-reviewed separation interval.
- `configs/limits.example.json`: blank template for a verified local motion envelope. It intentionally cannot enable hardware motion until every interval is filled.
- `requirements-hardware.txt`: optional, separately installed LeRobot dependencies.
- `tests/test_collect_robot_calibration.py` and `tests/test_pickup_trial.py`: tests without hardware, including failure and cleanup paths. Fake camera tests do not validate real camera drivers/codecs.
- `analysis/prepare_recordings.py`: converts the manual handoff to calibrated arm degrees, pairs earlier camera frames with observations, and marks excluded segments and samples.
- `configs/recording_review_20261112.json`: editable review of known setup, hand-reset and robot-regrip intervals for the 12 November session. Boundaries are conservative host-time intervals, not exact contact times.
- `analysis/compare_joint_response.py`: offline PyBullet comparison using the frozen robot loader and controller. Includes a collision audit and explicit pose assumptions.
- `analysis/audit_joint_alignment.py`: static pose audit of the returned session, with timestamp-paired video frames, illustrative model renders, joint-limit counts and a folded-pose geometry check. It does not fit or approve a hardware mapping.
- `requirements-analysis.txt` and `tests/test_prepare_recordings.py`: isolated analysis dependencies and unit/alignment/segmentation checks.

The grid is 202 mm along a1→h1 and 198 mm along a1→a8, with a
1.5 mm seam between ranks 4 and 5. Nominal cells are 25 × 24.5 mm (X/Y).
Use measured marker coordinates for calibration, not dimensions inferred from
printed colour widths. The playing grid is 1 mm above the border (user confirmation,
19 November 2026). The 16 mm body plus 3 mm feet puts the outer black border at 19 mm above
the table. The grid, grey surround and thin black outline are at 20 mm. The king height is 43 mm.

## Camera calibration from photos

Calibrate the fixed and wrist cameras separately.
Keep fixed/wrist originals and calibration outputs under ignored `runs/`.
Use original images with one resolution and unchanged focus/crop settings per camera.
The board is a custom ArUco target, not a standard ChArUco target.

From the repository root, with the installed camera environment:

```bash
.venv-robot-calibration/bin/python -m robot_calibration.camera_calibration \
  --camera fixed \
  --images runs/robot-calibration/camera-intrinsics-20261117/fixed \
  --validation picture_2026-11-17_16-00-11.jpg picture_2026-11-17_16-01-05.jpg \
  --output runs/robot-calibration/camera-intrinsics-20261117/fixed-fit-02

.venv-robot-calibration/bin/python -m robot_calibration.camera_calibration \
  --camera wrist \
  --images runs/robot-calibration/camera-intrinsics-20261117/wrist \
  --validation picture_2026-11-17_16-05-12.jpg picture_2026-11-17_16-07-08.jpg picture_2026-11-17_16-08-21.jpg \
  --output runs/robot-calibration/camera-intrinsics-20261117/wrist-fit-02
```

Choose a fresh output directory on each run. Input discovery is recursive;
previews must stay outside the input directory. At least four markers per image
and four training views are required as an initial screening, not an accuracy guarantee.
Reserve validation filenames before fitting. The script uses OpenCV's five-term
pinhole distortion model; it never applies the old recognition radial correction.

`observations.json` saves detected IDs, raw pixels, measured coordinates and image
hashes. `intrinsics.json` saves the lens fit, target hash, training/validation
membership and per-image errors. `preview_*.jpg` shows original left, corrected
right, in report order. Same-matrix undistortion can crop the edges.

These are candidate intrinsics, not approved robot calibration. Held-out board
poses are estimated separately while the lens remains fixed. Pixel residuals
are not millimetre accuracy; measured corners have roughly ±0.5 mm uncertainty.
Inspect detections, straightness, image coverage and repeatability before use.
Changing optics, focus, crop or capture mode requires checking calibration again.
No fitted values are automatically installed in recognition or robot control.

The first 17 November fit used 6 fixed images (1920×1080) and 11 wrist images
(1280×720). Fixed held-out RMS: 1.72 and 2.04 pixels. Wrist held-out RMS: 1.48,
8.23 and 1.58 pixels. The wrist outlier needs investigation (corner detection,
lens model, board geometry/flatness or capture effects); this fit is not approved.
The fixed fit also needs broader coverage and independent metric validation.
If these validation views guide changes, reserve new views for final validation.

### Validate a saved camera fit

```bash
.venv-robot-calibration/bin/python -m robot_calibration.analysis.validate_cameras \
  --fit runs/robot-calibration/camera-intrinsics-20261117/wrist-fit-01 \
  --output runs/robot-calibration/camera-validation-review/wrist
```

Repeat for the fixed fit, using a fresh output directory. The module verifies
source hashes and measured coordinates, draws detected/predicted corner overlays,
withholds each marker from board-pose estimation, refits the lens leaving one
photo out at a time, and screens inverse/forward distortion consistency across
the image. `validation.json` contains all results; green overlay circles are
detections, red crosses are predictions and yellow lines connect them.

Marker exclusion tests the pose, not full independence from lens fitting. The
photo-subset tests reuse one capture session. The distortion round-trip screen
checks numerical consistency, not independent optical accuracy or a mathematical
proof of invertibility. Neither a small residual nor this report authorizes
robot movement. Review stability, frame coverage and fresh independent evidence
before publishing a calibration profile.

### Compare candidate lens models

```bash
.venv-robot-calibration/bin/python -m robot_calibration.analysis.compare_camera_models \
  --fit runs/robot-calibration/camera-intrinsics-20261117/wrist-fit-01 \
  --output runs/robot-calibration/camera-model-review/wrist
```

Repeat with the fixed fit and a fresh output folder. Fixed-camera candidates
are five-term Brown distortion, two radial terms, one radial term and no
distortion. Wrist candidates are Brown distortion and OpenCV fisheye with two
or four free coefficients (skew fixed to zero). Identical photo folds and pixel-space
pose refinement make the comparisons consistent; all failures remain in the report.
The optimizer uses 100 iterations and 1e-9 tolerance, unlike the initial fit's
OpenCV defaults. Results may therefore improve through convergence alone.

`comparison.json` records each fit, excluded-photo errors, parameter ranges and
edge-inversion checks. JPEGs show original left / corrected right. These are
model-selection results, not independent validation. The all-photo candidates
also include earlier validation photos. Fisheye coefficients must be used with
OpenCV fisheye functions, never as ordinary Brown coefficients.
The models remain offline candidates; active recognition settings are unchanged.

### Test frozen candidates on new photos

```bash
.venv-robot-calibration/bin/python -m robot_calibration.analysis.validate_camera_photos \
  --candidate runs/robot-calibration/camera-models-20261117/wrist/candidate.json \
  --images runs/robot-calibration/camera-validation-new/wrist \
  --output runs/robot-calibration/camera-fresh-review/wrist
```

Repeat for `fixed` with its own candidate and images. You can supply the fitter's
`intrinsics.json` or a `MODEL-candidate.json` from the comparison command. This checks image resolution,
measured target geometry and file hashes to reject repeated training images.
Only board pose is fitted; lens values are frozen. Results include detected-corner
observations, per-photo residuals and original/corrected previews. Unusable views
remain listed. No automatic accuracy approval or active configuration update occurs.

## Estimate the board pose offline

`board_pose.py` converts decoded marker corners and frozen intrinsics into
`T_camera_from_board`. Its translation is in **metres**; rotation vectors are
in radians. Board origin is the outer grid corner at a1, X points toward h1,
Y toward a8 and Z out of the playing surface. Camera X points right, Y down
and Z forward. Multiplying a homogeneous board point by this transform gives
the point in camera coordinates; it is not a robot-base transform.

```bash
.venv-robot-calibration/bin/python -m robot_calibration.board_pose \
  --candidate runs/robot-calibration/fixed-camera-investigation-20261117/candidate.json \
  --images runs/robot-calibration/camera-validation-final-fixed-20261117 \
  --holdout-ids 8 9 \
  --output runs/robot-calibration/board-pose-review/fixed

.venv-robot-calibration/bin/python -m robot_calibration.board_pose \
  --candidate runs/robot-calibration/camera-models-20261117/wrist/candidate.json \
  --images runs/robot-calibration/camera-validation-new/wrist \
  --holdout-ids 8 9 \
  --output runs/robot-calibration/board-pose-review/wrist
```

Use a fresh output directory. Inputs must retain the calibration resolution and
orientation; do not apply the legacy radial correction first. The CLI reuses
`camera_calibration.collect_views()` for raw ArUco detections. The reusable
`estimate_board_pose()` function accepts IDs and canonical corners directly,
so a live vision caller does not need another detector or its old metric layout.

At least four fitting markers must remain after excluding the requested checks.
Missing reserved markers and unknown IDs are reported. `board_poses.json` stores
the transform, fitting/check-marker errors, image coverage, source hashes and
both refined planar-pose candidates. Both initial solutions can converge to the
same pose; two entries do not necessarily mean two distinct poses. A competing
solution must be reviewed, not automatically treated as equally plausible.
`pose_*.jpg` projects measured marker outlines (green for fit, magenta for checks)
and 30 mm board axes onto the raw image. Failures remain in the report.

`camera_geometry.py` shares projection and pixel-space pose refinement between
this module and the lens-model comparison; it does not fit lens parameters.
The held-out markers test pose prediction, but their measured geometry and the
lens fit are not independent physical ground truth. No grid centres are invented
from inconsistent nominal pitches. Outputs always keep `motion_ready=false`.
Base pose, live integration, cross-camera alignment and metric acceptance remain
separate work. The software does not command the arm.

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
python3 -m robot_calibration.collection.collect_robot_calibration \
  --mode joint_response --joint shoulder_pan \
  --output runs/robot-calibration/mock-001

python3 -m unittest discover -s robot_calibration/tests -v
```

Mock runs execute quickly using synthetic positions. `planned_s` is the simulated sample schedule; host timestamps reflect the actual fast execution. Mock files are marked as mock and must never be used as physical calibration evidence.

## Process the returned recordings offline

The 12 November handoff is attached to the GitHub release
[`chessarm-recordings-2026-11-12`](https://github.com/Wangyh666-ust/Chess-Player------SOARM-101/releases/tag/chessarm-recordings-2026-11-12).
These commands assume its ZIP has been extracted to the `data` directory below.
Outputs must be new directories; choose another name when rerunning. Neither
processor connects to hardware or alters the raw files.

```bash
python3 -m robot_calibration.analysis.prepare_recordings \
  --source runs/robot-calibration/teammate-2026-11-12/data \
  --output runs/robot-calibration/teammate-2026-11-12/prepared
```

Preparation uses only the Python standard library. Each trial gets a
`samples.jsonl` and `metadata.json`; `manifest.json` records the source hashes,
units and exclusions, and `segments.json` lists separate segments with their
source and candidate sample indices. Videos are referenced by absolute path,
not duplicated. Re-run preparation after moving the source recordings.

Arm values follow the [LeRobot 0.7.0 normalization implementation](https://raw.githubusercontent.com/huggingface/lerobot/v0.7.0/src/lerobot/motors/motors_bus.py):
for non-inverted STS3215 joints, `degrees = normalized * (range_max - range_min)
* 360 / (200 * 4095)`. The motor homing offset is already applied. This yields
degrees relative to the calibration midpoint, **not verified URDF angles**.
Gripper values stay normalized 0–100; no jaw-width or force conversion is known.
Normalized endpoint observations may have been clipped upstream and are excluded
from imitation candidates, rather than silently reconstructed as precise angles.

Each observation uses the latest frame whose acquisition ended before that
observation started. `--max-frame-age-s` defaults to 0.15; missing/stale frames
remain visible in the output but exclude the sample from imitation candidates.
Frame age describes host acquisition timing; camera exposure latency is unknown.
Commands are paired with the preceding observation from the same recorded row.
Intermediate commands were not all logged, so this is not an exact replay file.

The review JSON separates the knight's failed attempt, conservative 18–25 s hand
reset/transition exclusion, and retry. It separates the pawn's initial transfer,
21–29 s regrip/placement interval, and withdrawal; that entire pawn recording is
excluded from simple imitation candidates. The setup-only bishop is excluded.
Elsewhere, outcomes are inherited from the supplied recording-level review;
idle time is retained and no dense intervention audit is claimed. An
`imitation_candidate` is a candidate for later review, not a certified expert
action. Keep all segments of a recording together when splitting datasets; this
single session does not provide an independent session-level test set.

For simulation and plots, use Python 3.12 in a removable environment:

```bash
python3.12 -m venv .venv-control
source .venv-control/bin/activate
python -m pip install -r robot_calibration/requirements-analysis.txt

python -m robot_calibration.analysis.compare_joint_response \
  --source runs/robot-calibration/teammate-2026-11-12/data \
  --output runs/robot-calibration/teammate-2026-11-12/comparison-recorded \
  --pose recorded-assumed

python -m robot_calibration.analysis.compare_joint_response \
  --source runs/robot-calibration/teammate-2026-11-12/data \
  --output runs/robot-calibration/teammate-2026-11-12/comparison-neutral \
  --pose neutral-reference
```

PyBullet may compile from source. If your Python build requests an unavailable
compiler, use an installed toolchain explicitly, for example
`CC=gcc CXX=g++ python -m pip install -r robot_calibration/requirements-analysis.txt`.
The robot model, gains, force caps and 240 Hz step come from the current simulation;
the frozen runtime is not edited. The probe contains only the robot under gravity,
without board, pieces or external support. Commands follow recorded send times,
and simulated observations use the preceding physics tick (at most one tick old).

`recorded-assumed` directly treats measured degrees as URDF angles. In this
session that assumption produces shoulder/jaw self-penetration, so its large
errors **must not be used to fit motor parameters**. `neutral-reference` instead
starts all five arm joints at zero and applies the measured relative movements.
It retains the collision model and offers a collision-free controller diagnostic
at a different pose. Neither mode establishes the missing physical joint mapping.
The gripper is fixed at the explicit `--gripper-reference-rad` value (default 0),
since its angle mapping is also uncalibrated.

Outputs are `comparison.json` (assumptions, collisions, hashes and metrics), one
CSV per joint test, and `joint_response_comparison.png`. No parameters are fitted
or applied to the simulator. Real and simulated RMS errors are tracking errors
against the recorded command, not Cartesian accuracy or force estimates.

### Inspect the physical-to-model pose mapping

After preparing the recordings, run this offline inspection in the analysis environment:

```bash
python -m robot_calibration.analysis.audit_joint_alignment \
  --source runs/robot-calibration/teammate-2026-11-12/data \
  --prepared runs/robot-calibration/teammate-2026-11-12/prepared \
  --output runs/robot-calibration/teammate-2026-11-12/alignment-review
```

Use a new output path when rerunning. `ffmpeg` must be available to extract the
actual frame indices; nominal video playback time is not used. The six session
review poses are listed in `REVIEW_POSES` in the script. Outputs include
`alignment.json`, original fixed/wrist frames, `pose_*_comparison.jpg` panels and
the folded joint-test pose. The panels rotate the fixed view for readability;
the simulated viewpoint is illustrative, **not a calibrated camera overlay**.
The gripper is held at zero radians as an explicit diagnostic assumption.

The 13 November inspection found matching joint names/IDs and identical arm
joint origins/axes between the original and active URDF. The active model narrows
the original arm limits by about five degrees at each end; this is not an angle
offset. Direct degree-to-radian mapping broadly reproduces visible pickup bends,
but the shoulder/base are cropped and the wrist-roll range in the reviewed poses
is small. No signs or offsets were fitted or validated.

All 7,385 pickup samples were inspected statically under that hypothesis, without
clamping or stepping physics. None had self-penetration above 1 mm with the
assumed gripper pose, but 1,503 shoulder-lift and 3,959 wrist-flex samples exceed
the active limits. These counts can overlap and include censored/failed samples.
The folded joint-test pose still collides with both the original and modified
collision geometry. Thus its collision does not establish a global sign error,
nor can it be fixed by changing only the moving jaw's opening.

Keep this mapping provisional. Before fitting actuator physics, resolve the
folded-pose discrepancy using a known physical pose; before physical deployment,
validate the joint offsets/signs, gripper opening, tool frame and camera/board
transforms. Do not widen limits, add guessed offsets or remove collision pairs
just to make the diagnostic pass. The audit does not change the frozen runtime.

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

The adapter targets LeRobot 0.7.0. See the [calibration guide](README.md#what-has-been-validated) for the scope of the recorded hardware checks.

## Record joint readings without commanding motion

Close other applications controlling the arm. This mode requires saved calibration matching the motors. It reads positions and existing settings; it does not send position goals, configure PID gains, enable torque or disable torque on exit.

```bash
python -m robot_calibration.collection.collect_robot_calibration \
  --backend lerobot --mode read \
  --port /dev/ttyACM0 --robot-id chess_arm \
  --duration-s 5 --output runs/robot-calibration/read-001
```

The five arm readings are LeRobot-calibrated **degrees**, not PyBullet joint coordinates. The gripper is **normalized 0–100**, not degrees, millimetres or force. The mapping to the simulator must be checked separately.

The collector opens the motor bus directly rather than calling the higher-level robot connection routine, because that routine can configure motor gains. Existing gains and torque limits are logged, not overwritten. Add `--telemetry` to read current/load and `--camera 0` to capture video, including in read mode. Do not run a second controller on the same port; read mode is not a concurrent teleoperation recorder.

## Prepare a small movement test

1. Secure the base. Remove chess pieces and clear the intended small sweep. Start with the empty arm in a supported pose, with all motor torque off. Never force a powered joint by hand.
2. Use calibrated observations and manual/teleoperated inspection to choose a collision-free local envelope. Copy `configs/limits.example.json` to `runs/robot-calibration/limits.json` and replace every `null` with a measured limit. These are local experiment limits, not automatically discovered mechanical limits.
3. Check that the current pose and the selected joint's full positive/negative excursion fit inside that envelope. Joint bounds alone do not establish that a pose is clear of the table, itself or a person.
4. Keep physical power-off accessible and support the arm when the run finishes or aborts. Completion or abort enters supervised recovery; torque is not automatically disabled. Provide mechanical support against a drop before explicitly releasing torque. Do not use the joint-response test while holding a piece. Pickup trials can also drop the piece on an abort; use a small lift over a protected area.

For the first run, use a 1-degree amplitude and one cycle:

```bash
python -m robot_calibration.collection.collect_robot_calibration \
  --backend lerobot --mode joint_response --execute --recovery-config runs/robot-calibration/recovery.json \
  --port /dev/ttyACM0 --robot-id chess_arm \
  --joint shoulder_pan --amplitude-deg 1 \
  --ramp-s 3 --hold-s 1 --cycles 1 --hz 30 \
  --limits runs/robot-calibration/limits.json \
  --output runs/robot-calibration/shoulder-pan-001
```

The selected joint follows: initial hold → positive ramp/hold → centre ramp/hold → negative ramp/hold → initial ramp/hold. Other joints, including the gripper, hold their starting readings. There is no absolute home movement and no automatic retry.

Before enabling torque, the script requires position mode and torque off, checks the saved calibration and all pose bounds, and seeds the motor goals with current readings. It then enables torque without altering gains. Only the five arm joints can be selected for this first response test; automated gripper-gap calibration is a later mode.

Joint-response commands are capped at 5 degrees of excursion and 5 degrees/second. These are software caps, not a claim that every permitted trajectory is safe. Samples use a monotonic clock. Excessive lateness, out-of-bound readings, excessive tracking error, invalid values or communication errors abort the run and preserve partial data. Ctrl+C and SIGTERM use the same supervised recovery path; they do not automatically disable torque or disconnect.

**Supervised stop configuration (software tested, hardware validation pending).**
Copy `configs/recovery.example.json` to a local run configuration and fill all
values after reviewing the intended operating envelope. Null examples cannot run.
`drift_ticks` is the allowed arm displacement from the stopped encoder position;
`max_temperature` is the motor temperature register threshold; `read_deadline_s`
and `poll_deadline_s` bound feedback acquisition and polling gaps;
`hold_deadline_s` requests intervention after a prolonged hold; `poll_s` sets the
poll interval and must be smaller than its deadline. These are not force limits.
Raw position bounds are converted from `--limits` using the pinned SDK calibration.

A local terminal (`/dev/tty`) and writable `recovery.jsonl` are required before
motion. At completion or failure, the trajectory is permanently cancelled.
Where fresh, valid feedback permits, the five arm goals are replaced by measured
positions while the existing gripper goal is retained. Readback is evidence of a
goal being stored, **not proof that the arm can physically hold it**. Faults,
drift, excessive temperature, missed deadlines and lost feedback latch an
intervention state; no automatic retry, re-enable or timeout torque-off occurs.

After securing the arm and any held piece on mechanical support, the operator
can type `supported-release`; torque-off feedback must be verified before
closing the connection. If communication cannot verify release, physically
remove motor power with the arm supported, then type `power-off-supported`.
That fallback is logged as an operator assertion, not telemetry verification.
Do not enter release commands in advance. Repeated Ctrl+C and lost terminal
input do not release torque; physical support and power isolation remain needed
if software or communication fails. Run this from an attended local terminal.
The Python deadlines detect late responses but cannot interrupt a blocked SDK
call, guarantee real-time stopping, or protect against process kill/power loss.

The stop implementation is `collection/stop_recovery.py`; fake-bus regressions
are in `tests/test_stop_recovery.py`. This change does not validate motor tracking,
robot geometry, or autonomous movement. Validate recovery with mechanical
support before relying on it during a physical experiment.


## Current and load telemetry

```bash
python -m robot_calibration.collection.collect_robot_calibration \
  --backend lerobot --mode read --telemetry --camera 0 --hz 10 \
  --port /dev/ttyACM0 --robot-id chess_arm \
  --duration-s 10 --output runs/robot-calibration/telemetry-001
```

The collector records `Present_Current`, `Present_Load`, `Present_Voltage`, `Present_Temperature` and `Goal_Position` for all six motors. Current/load are kept as register values, without guessing sign encoding, amperes, joint torque or gripper force. Goal positions use the same degree/normalized-gripper units as commands. Each register read has its own timestamp interval; these reads are sequential, not simultaneous.

These register names are supported by the [STS3215 driver definition](https://docs.rs/rustypot/latest/rustypot/servo/feetech/sts3215/index.html). The actual firmware/driver must answer every requested register. A failed preflight read prevents movement; a failed read during motion aborts and retains partial evidence. We do not replace missing values with zero. Start at 10 Hz because the extra reads increase bus traffic; the observed timestamps determine the effective sample rate.

## Teach a pickup before repeating it

Copy `configs/pickup.example.json` to `runs/robot-calibration/pawn-pickup.json`. Fill it with poses measured on the real robot after calibration. Never copy simulation joint coordinates or the synthetic test fixture into hardware.

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
python -m robot_calibration.collection.collect_robot_calibration \
  --backend lerobot --mode pickup --execute --recovery-config runs/robot-calibration/recovery.json \
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
python -m robot_calibration.collection.annotate_separation \
  --trial runs/robot-calibration/pawn-series-001/trial-001 \
  --export-frames 20 50
```

Inspect the JPEGs and record the last frame with contact and the first frame with a visible gap. Replace the example frame numbers with observed values:

```bash
python -m robot_calibration.collection.annotate_separation \
  --trial runs/robot-calibration/pawn-series-001/trial-001 \
  --reviewer teammate --outcome separated \
  --last-contact-frame 32 --first-clear-frame 33
```

This writes a new `separation.json` with frame indices, a host-time interval, matching telemetry sample indices and evidence hashes. It never overwrites a review. Use `--outcome no-separation` or `--outcome occluded` without frame indices when appropriate. A frame interval is more honest than claiming an exact physical separation time. No automatic computer-vision separation detector is included.

A blocked driver call, killed process, disconnected cable or failed torque-disable command cannot be stopped reliably by this Python loop. Physical power-off is the fallback. There is no hardware watchdog, torque/force control, collision model or current-based contact detector in this collector.

## Read the results

Each new output directory contains:

- `recovery.jsonl` (hardware motion): stop goals, feedback, latched faults and explicit release evidence.
- `metadata.json`: backend, status, settings, units, starting positions, limits, hardware calibration/settings, source hash and a tracking summary for motion runs. Failure and cleanup status are explicit.
- `samples.jsonl`: one record per sample, with requested joint positions, measured positions, phase and host-side command/read timestamps. Read-only records have `command: null`.
- `video.avi` and `frames.jsonl`, when a camera is selected: video plus frame acquisition intervals on the telemetry clock.
- `separation.json`, after review: human-labelled separation or uncertainty; no force estimate.

The metadata contains the full calibration snapshot; the record is not just tied to an ID that might be recalibrated later. Each run uses a new directory, and existing results are never overwritten. Outputs under `runs/` are already ignored by Git.

Read timings bound the host acquisition interval; they are not exact motor sensor timestamps. The tracking RMSE is a diagnostic, not an automatically estimated motor delay. External video/markers will be needed to distinguish encoder tracking from mechanical play and gripper position error. An offline fitting module can be added after real recordings exist.

## Force measurements can wait

A force gauge is a physical push/pull instrument, not a software setting. If none is available, leave contact values unknown for now. Pickup recordings can reveal repeatable load changes and separation timing, but do not by themselves yield magnetic attraction in newtons. Do not convert a motor current or torque setting directly into a measured jaw force.

Board alignment through ArUco still needs camera calibration and a camera-to-robot relationship. A single known arm pose does not generally determine all unknown camera/tool transforms. This module does not solve that calibration yet.

## Powered fingertip measurements from a terminal
For the complete bench procedure, including serial ownership, raw register reads and writes, startup retries and operator handoff, see the [SO-ARM101 operator guide](ARM_OPERATOR_GUIDE.md).

Preview a local plan with `scripts/measure-chess-arm preview --plan "$REVIEWED_PLAN"`.
Use the [operator guide](ARM_OPERATOR_GUIDE.md) before starting any powered hold.

1. From the local terminal, run the command below with the operator-reviewed plan,
   its reviewed SHA-256, the verified devices, and a fresh output directory.
   The first goal write can energize all six motors.

   ```bash
   scripts/measure-chess-arm run --plan "$REVIEWED_PLAN" \
     --reviewed-plan-sha256 "$REVIEWED_PLAN_SHA256" --robot-id "$ROBOT_ID" \
     --port "$ARM_PORT" --camera "$FIXED_CAMERA" --output "$NEW_RECORDING"
   ```

2. Wait for **REACHED**. Measure from the playing surface to the same fixed fingertip. Enter height and ruler uncertainty in millimetres on one line, for example `70.4 0.5`. The program records the observation with the current encoders. Each hold has a 180-second deadline including time to select the next target.
3. Choose the next board location using the observation. In a second terminal use the [offline planning command](ARM_OPERATOR_GUIDE.md#5-measure-and-continue-without-cycling-power) with the saved hold, desired square and modeled height. Use a new output filename for each plan. This command only previews the path. Review joint travel and whole-arm clearance. Enter `next /tmp/next-placement.json` in the first terminal to run that one target on the same connection. Use the corresponding numbered hold for each subsequent plan.
4. Enter `stop` when finished. Completion, a fault, or a deadline enters supervised recovery. Support the arm, physically remove motor power and enter `power-off-supported` once that is true. During startup, a mixed torque reading keeps the arm in read-only verification. Two consecutive fully verified all-on samples are required within the original 0.5-second window. A torque loss during movement still cancels the trajectory. Only passive checks can be repeated without starting another motion attempt.

Keep the entire output directory. In `samples.jsonl`, `operator_measurement_received` links the square, pose number, ruler reading, ruler uncertainty and encoder registers. Use elapsed times with `frames.jsonl` and `video.avi` to find the corresponding footage. Ruler uncertainty covers the ruler reading only. Report encoder-model error separately from requested-height error. On this setup the marker mounting limits confidence in absolute placement and the 5 mm clearance remains unvalidated.

Use `measure-start` in place of `run` with the same options to measure the reviewed starting pose. A clean startup enters a powered measurement hold with zero trajectory commands. If startup torque fails to stabilize before the deadline, only a verified, mechanically supported recovery hold can accept `supported-height HEIGHT UNCERTAINTY`. This observation is stored in `recovery.jsonl` with fresh registers and its load condition. Compare supported readings separately from free-standing powered placements. A fault or expired recovery deadline closes this measurement window.

For the measured 20 November setup, `scripts/measure-chess-arm plan` with `--height-calibration FILE` previews a hover using the local rank-4 fixed-tip height calibration. It adds the measured encoder-to-ruler bias to the IK height and carries both height conventions in the output. For example, use the final g4 hold and a requested physical height of 69.2 mm. The estimate assumes the joints achieve the IK goal. Actual joint tracking adds placement error. The calibration is restricted to the recorded model, frame, hardware, gripper opening, tested squares and hover postures. The original `plan` command still takes modeled mm. Both commands only generate a preview. Corrected commanded placement has not been physically validated.
