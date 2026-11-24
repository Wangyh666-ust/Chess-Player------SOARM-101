# Calibration architecture

This design separates reusable robot measurements from the geometry of today's
setup. The collection/analysis code below exists. Offline camera/board pose estimation is implemented. Commissioning, startup,
live integration and independent physical validation remain planned.
No physical calibration or movement was performed during this reorganization.

## Scope and assumptions

- The robot is clamped at the table edge with the board in front. The workspace
  contains the table, mounting clamp/base, fixed camera and its support, board
  and pieces. Do not build a general obstacle-recognition system for this setup.
  Record these known surfaces and require the assumed clear area to remain clear.
- Calibrate the fixed and wrist cameras separately.
- The wrist camera is rigidly mounted. Its starting view points upward; acquire
  the board from the fixed camera before following an approved departure path
  to the wrist observation pose.
- Pieces weigh roughly 2–3 g. Start with empty-arm kinematic measurements, then
  validate representative pickups. Do not make extensive payload sweeps a
  prerequisite; investigate load effects if observed errors justify them.
- One gripper jaw is fixed and one moves. Model the grasp centre as a function
  of opening and grasp height; do not assume symmetric finger motion.
- Never infer a physical motor limit from stalled commanded movement. Record
  ranges manually with torque off and support, then select conservative software
  bounds. Mechanical range, joint/model mapping and scene clearance are distinct.

## Current source tree

```text
robot_calibration/
  README.md                       commands for existing tools
  ARCHITECTURE.md                  workflows and implementation plan
  AGENTS.md                       hardware operation rules
  requirements-hardware.txt       pinned robot environment dependencies
  requirements-analysis.txt       offline analysis dependencies
  records.py                      shared joint order, JSON and file hashes
  camera_calibration.py           offline photo fitting and held-out checks
  camera_geometry.py              shared projection and pose refinement
  board_pose.py                   camera-from-board transform and marker checks
  collection/
    collect_robot_calibration.py   passive reads, guarded joint tests, taught pickup
    pickup_trial.py                trajectory validation and camera recording
    annotate_separation.py         video event review
  analysis/
    prepare_recordings.py          units, timing, segments and eligibility
    audit_joint_alignment.py       static real/model comparison
    compare_joint_response.py      recorded-response comparison in simulation
  configs/
    board_target.json             measured physical calibration target
    physical_measurements.json     existing manual measurements and unknowns
    limits.example.json           blank local motion-bound template
    pickup.example.json           blank taught-motion template
    recording_review_20261112.json historical handoff annotations
  tests/                          existing hardware-free regression tests
```

Python packages also have `__init__.py`. Launch tools from the repository root
with `python -m robot_calibration.collection.collect_robot_calibration`, etc.
Old flat script paths have moved; there are no duplicate legacy implementations.
Manipulation and evaluation use `records.py` rather than importing the historical
handoff processor for generic file helpers. Frozen simulation and historical run
records retain their original contents and command strings.

## Modules to implement next

Keep ordinary functions and JSON records. Add these files only when implementing
their behaviour; do not create placeholder packages or a plugin framework.

| Planned module | Functions and responsibility |
| --- | --- |
| `hardware.py` | Extract the existing `LeRobotArm` and `MockArm` implementation here when commissioning needs it. Reuse passive connection, read, guarded enable/send and close. Preserve its preflight and supervised stop/release lifecycle in `collection/stop_recovery.py`; no second motor driver or unconditional torque-off cleanup. |
| `commission.py` | `commission()` coordinates selected initial measurements; `record_reference_pose()` collects user-confirmed angle measurements and simultaneous encoder data; `fit_joint_alignment()` proposes a mapping; `save_profile()` publishes a new version only after independent validation. |
| `startup.py` | `startup()` runs reconnect checks; `check_identity()` checks device/calibration identity; `check_start_pose()` checks the saved starting region; `build_session()` combines the robot profile and current scene; only a passed final validation can set readiness. |
| `camera_calibration.py` (partly implemented) | `collect_views()` detects markers in offline photos; `fit_intrinsics()` fits pinhole intrinsics; `validate_intrinsics()` evaluates held-out views. Live capture and wrist-mount transform fitting remain future work. |
| `board_pose.py` (partly implemented) | `estimate_board_pose()` consumes raw decoded IDs/corners and measured target geometry, supports both lens models, reserves check markers and returns camera-from-board transforms. The offline CLI reuses the calibration detector. Live vision integration, base pose, temporal stability and square targets remain future work. |
| `geometry.py` | `joint_to_model()` applies the reviewed sign/offset mapping; `grasp_frame()` returns the tool frame for opening/height; `check_path()` checks bounds and known scene geometry using the existing simulation/planner integration. No duplicated IK solver. |
| `validation.py` | `validate_profile()` scores held-out commissioning observations; `validate_session()` performs independent reconnect checks and reports pass/fail/not-tested per criterion; `write_report()` saves evidence and reasons. It is not the learned RL critic. |

The collector stays responsible for measurement experiments. The offline
analysis tools stay responsible for fitting/comparison; startup never silently
refits gains, force settings or joint offsets. Future simulation comparisons use
a materialized workspace/explicit profile, not edits to the frozen export.

### Reuse detection, keep physical target measurements separate

Camera calibration will use `configs/board_target.json`, independently of the
recognition layout. It contains the user's measurements, with unresolved fields
explicitly left unmeasured. Treat the opened playing surface as planar as the
user confirmed; the seam still needs explicit in-plane geometry. Use a custom
ArUco target rather than assuming the taped markers form a standard ChArUco
layout. Canonical marker ordering has now been checked; offline candidate fitting
is implemented in `camera_calibration.py`. Independent accuracy validation remains.

`recognition/vision_eval/aruco_live_service_v10.py` already provides
`GridLocator.detect()`: DICT_4X4_50 marker detection, known-marker layout matching,
board homography, grid/cell projection and geometric/reprojection checks. Its
helpers include `camera1_radial_grid_v2.py` and `guided_grid_adapter_v2.py`.
The board-confirmation evidence includes `calibration/layout.json` and
`calibration/radial.json`, selected through `CHESS_ARUCO_ASSETS`.

Reuse detection through a small shared interface for marker IDs and raw pixel
corners. Obtain physical corner coordinates from the independently validated
calibration target, not recognition's historical layout. Preserve
recognition's existing output contract when exposing those correspondences.
The current grid result is an image-space localization, not a verified
camera-to-board or robot-to-board rigid transform. Calibration adds metric pose,
the base-marker relationship and independent physical validation.

Before reuse, review camera-specific input rotation/resolution and occlusion
masks, the provisional wrist radial model, and the layout's square pitch against
the measured board/seam geometry. The code defaults to 26 mm squares if the
layout does not supply a value. Do not treat those assumptions or image-space
validity thresholds as approved robot positioning accuracy. Keep ArUco detection
owned by vision; the calibration adapter owns coordinate-frame integration.

## Local data layout

```text
runs/robot-calibration/
  local-state/                    existing local install caches/calibration files
  profiles/<robot-id>/<revision>/
    profile.json                  reviewed values, units, identities and hashes
    evidence/                     source measurements, images and fit reports
  commissioning/<run-id>/
    measurements.jsonl            append-only observations and user annotations
    images/                       fixed and wrist reference images
    report.json                   completion, failures and independent results
  sessions/<run-id>/
    session.json                  profile identity, scene transforms, readiness
    evidence/                     observations used for the scene fit
    validation.json               separate checks excluded from fitting
    events.jsonl                  transitions, communication faults and resets
```

These are planned output locations; existing recordings are not relocated.
Keep profiles immutable by revision. Sessions reference exact profile and camera
calibration hashes. Store input measurements and proposed corrections separately.
No unreviewed fit replaces the SDK calibration or a previously valid profile.

`profile.json` owns joint mapping, measured bounds, start-region angles, approved
departure procedure, tool/opening geometry, camera intrinsics and mounting
transforms, marker geometry, units, measurement uncertainty and validation criteria.
`session.json` owns the current board/base/camera/table transforms, known scene,
task-specific operating envelope and final readiness. Device IDs are configuration,
not guesses based on `/dev/video0` or port order.

Use metres/radians internally for geometry, but label hardware degrees and
normalized gripper 0–100 explicitly at boundaries. A normalized gripper value is
not an angle, jaw width or force. Define transforms as `T_A_from_B`: they convert
coordinates in B into A. Joint/protractor definitions must state parent link,
child link, reference axis, sign and whether the reading is an included angle.

## Initial commissioning workflow

1. Connect passively and verify identity, saved motor calibration and readable
   joint state. Save an unchanged calibration snapshot. Recalibrate SDK ranges
   only if necessary, using its guided supported manual procedure.
2. Record manual geometry and marker layout. Reconcile the reported 202 × 198 mm grid (X/Y) with nominal 25 × 24.5 mm cells
   and measured three-cell spans before deriving centres. Include the seam and
   board surface elevation, not only its outer dimensions.
3. Establish the supported starting pose, acceptable repeatability and a clear
   departure path. A pose at manually reached stops is a reference observation,
   never permission to drive into those stops. Midpoints for base/wrist roll need
   marks or a jig; do not reset offsets from an approximate pose.
4. Collect reference poses with the protractor procedure below. Fit joint signs
   and offsets, then test separate poses. One pose cannot validate the whole arm.
5. Calibrate fixed/wrist cameras. Fit
   using diverse views; validate separate images. Record resolution/crop/focus.
   Determine rigid base-marker and wrist-camera transforms.
6. Measure fixed/moving finger contact locations at several openings and grasp
   heights. For parallel contact faces at x_fixed and x_moving, the nominal
   centre is their midpoint; use measured kinematics when faces rotate. Keep
   finger-face offsets and piece contact geometry explicit.
7. Record bounded empty-arm response tests. 'Both directions' means approach
   the same target from below and above in that joint's angle, with other joints
   held at the approved pose. The board stays in front. Use results to assess
   backlash; add complexity only when repeatability requires it.
8. Conduct controlled representative pickup trials and fit practical grasp
   settings. Distinguish missed contact, slip, failed magnetic separation and
   placement failure. These settings are not force measurements in newtons.
9. Validate on poses/attempts excluded from fitting. Save the approved profile
   with measured error and applicability; retain failed candidates and raw data.

### Protractor reference-pose procedure

The arm may move to a reference pose only using a previously established safe
hardware target/path and operator readiness. Before joint/model alignment exists,
use a manually taught pose; do not send guessed simulation angles to the arm.

1. Record commanded hardware angles, measured encoder angles and any provisional
   model-angle predictions as distinct fields. Show all five arm joints; show
   gripper command/opening separately unless its angle mapping is established.
2. Support the arm before releasing torque. Record the readings immediately
   before and after torque-off to expose gravity-induced motion.
3. Measure the actual relative link angle with the protractor. Enter that value,
   its definition and estimated uncertainty; capture fresh encoder readings and
   an image at the same accepted measurement event.
4. If the operator manually adjusts a joint, record the resulting protractor
   value and encoder value as a second observation. Encoder change alone is not
   an independent measurement of the true angle.
5. Repeat across multiple poses. With established signs and conventions, fit
   `q_model = sign * q_hardware + offset` (with explicit degree/radian conversion).
   Do not conceal geometric/model errors by forcing every joint into this fit.
6. Propose corrections and assess separate validation poses. Keep physical/model
   mapping separate from LeRobot normalization/EEPROM; do not rewrite it silently.

Support is necessary because a torque-off pose can change under gravity even
with no chess piece attached. No unsupervised torque-on/torque-off cycle is implied.

## Reconnect workflow and independent final validation

`DISCONNECTED → PASSIVE_CHECK → START_POSE_CHECK → SCENE_ESTIMATE →
PATH_CHECK → SUPERVISED_MOTION_CHECK → INDEPENDENT_VALIDATION → READY`

1. Load the correct approved robot profile; compare identity/calibration and
   camera modes. Verify current joint readings against the start region.
2. Estimate board and rigid base target from fixed-camera observations. The
   camera may have moved: solve its new relationship to both targets. Base-wall
   markers are valid only with measured rigid mounting geometry; clamp markers
   qualify only if their relationship to the base is mechanically fixed.
3. Rebuild the known scene and compute task-specific clearance/reachability.
   Saved motor limits remain unchanged; the scene restricts them further. No
   automatic hard-stop search or general clutter detector is needed.
4. After operator readiness, verify local joint response and follow the approved
   departure path to a wrist observation pose. Compare its board estimate with
   the fixed camera, accounting for both estimates' measured uncertainty.
5. Run mandatory independent final checks: held-out board landmarks/markers not
   used for the fit where possible, known grid centres across both halves,
   wrist-versus-fixed agreement, and validated non-contact approach checkpoints
   using independent visual/measured evidence. A commanded pose matching its
   own encoder target alone does not establish Cartesian accuracy.
6. Use tolerance values established during commissioning, store residuals and
   evidence, and require every applicable check to pass. Missing evidence is
   `not_tested`, never a pass. If a check is used to adjust the fit, reserve new
   independent checks before granting readiness.

On failure, preserve evidence, invalidate readiness and require a supervised
recovery. Loss of communication may prevent software stopping the arm, so retain
physical power-off access and the established support procedure. Do not blindly
retry or automatically resume on reconnect. Board/base movement or changed
camera settings invalidates the session. A moved wrist mount, changed fingers or
motor calibration invalidates the affected parts of the reusable profile too.

## First implementation milestone

1. Extract/reuse the hardware adapter; implement passive identity and calibration
   checks, without enabling torque.
2. Implement a reference-pose recorder for manually supported protractor readings
   and simultaneous joint state. Save start-region evidence without fitting from
   a single pose. Confirm actual marker dictionary, IDs, sizes and mounting layout.
3. Calibrate the fixed camera and establish the rigid base-marker reference.
   Implement board/base pose estimation and overlays; independently check grid
   orientation and measured points.
4. Add an end-of-session validation report. Report `read_only_ready` separately
   from `motion_ready`: the latter stays false until the approved path, joint
   mapping and supervised independent motion checks are complete.

This milestone comes before automated reference-pose movement, wrist hand-eye
fitting, grasp-parameter experiments or physics fitting. Those later stages use
the same evidence/profile/session structure rather than a separate pipeline.
