# SO-ARM101 calibration and recalibration

Start here to commission an SO-ARM101 for chess or recover a previously
calibrated setup. The calibration defines where the arm, board, cameras and
fingertip are relative to each other.

## Choose your workflow

| Your situation | Start here |
|---|---|
| First time on this rig | [Run the offline smoke check](#first-time-software-setup). If the follower has no saved calibration, follow the [guided calibration steps](#after-replacing-a-motor), then work through the calibration order below. |
| Same rig, no physical change | [Read the arm passively](#passive-check-and-what-to-save), then verify a known working pose against a physical reference. If it still meets the task tolerance, use the saved fit. |
| Motor or joint mounting changed | [Recalibrate after a motor replacement](#after-replacing-a-motor). If the workspace also moved, register it afterward. |
| Board, base, or camera moved | [Refresh the scene registration](#after-moving-the-setup). Keep the motor calibration and lens fits that still match the hardware. |
| Fixed finger, pads, or jaw changed | [Check the geometry that changed](FIXED_TIP_WORKSHEET.md). Refit a changed fixed-tip point and remeasure the gap if the pad contact surfaces or gripper opening changed. |

## Calibration order and reuse

| Step | What you measure and save | When to repeat it |
|---|---|---|
| **1. Arm and joint mapping** | Save the LeRobot calibration and conservative local limits. Use physical reference poses to check how encoder positions map to model angles. A passive read and a reviewed small joint-response test check identity, tracking, and return. [Joint reference measurements](ARCHITECTURE.md#protractor-reference-pose-procedure) · [Joint response](TOOL_REFERENCE.md#prepare-a-small-movement-test) | After changing a servo, joint mounting, or encoder calibration. If only control settings changed, repeat the affected response checks. |
| **2. Board and pieces** | Measure the playing surface, grid, seam, and marker corners. Reuse the board target for the same marker layout. Keep piece sizes and masses with the measured pieces. [Board target](configs/board_target.json) · [Physical measurements](configs/physical_measurements.json) | When the board, markers, or pieces change. Moving an unchanged board calls for scene registration. |
| **3. Camera lenses and mounts** | Fit each lens from original photos and check it on reserved photos. Record the capture mode. Survey the wrist camera's pose relative to its link and compare views at the intended wrist poses if using metric wrist feedback. [Lens fitting](TOOL_REFERENCE.md#camera-calibration-from-photos) · [Wrist checks](ARM_CONTROL_HISTORY.md) | Recheck the lens after changes to focus, crop, resolution, or optics. Refresh the wrist mounting transform after a remount. |
| **4. Board relative to the arm** | Use fresh board images and a surveyed rigid base target to locate the board in robot coordinates. Save the scene with the camera settings and survey used. [Scene registration](#after-moving-the-setup) | When the board, base, or fixed camera moves relative to the others, or a marker shifts. |
| **5. Fixed-tip fit and jaw gap** | Use one [23-mark jogger session](FIXED_TIP_WORKSHEET.md#record-the-23-marks) to fit the fixed fingertip. It includes the 13 surface contacts and adds elevated marks. Use the same data to check the fit by leaving out whole intersections. For grasping, measure the inside jaw gap at stated depths and reserve an opening to check interpolation. | Recheck after joint remapping, finger or pad changes, or repeated misses at known poses. Refit the affected geometry if the physical check fails. |
| **6. Independent checks at working poses** | Freeze the fit and measure actual XYZ at supervised holds that cover the intended positions and heights. Measure clearance as a separate quantity during the same session. Repeat one hold if you need to measure return repeatability. [Validation worksheet](FIXED_TIP_WORKSHEET.md#validate-the-frozen-fit) | After a change that affects the task, or before using a new route or working height. |

Keep the **arm profile** with the hardware, joint mapping and tool geometry.
Keep a separate **scene record** for the current positions of the board, base,
and cameras. A validation report should identify both revisions and show
which poses and clearances were physically checked.

### Which checks can share a session?

- **Use the 23-mark session as the fixed-tip fit.** It covers the separate
  13-contact run. Earlier spot contacts and the rank-4 ruler correction are
  reference evidence for this bench. There is no need to repeat them when
  fitting the tool with the 23 marks.
- **Combine hover, near-board, and return checks in one independent session.**
  Use working locations and heights for the task. A low hold can serve as an
  XYZ check if you can measure all three axes independently. Add a higher
  hold if the task uses it or the low views cannot resolve XYZ. Revisit one
  hold if you need to measure repeatability. The separate 60 mm repeat
  sequence is available for diagnosing inconsistent returns. If you use a
  check to change the fit, reserve fresh observations to validate the revision.
- **Check jaws and loaded behavior when the task uses them.** The fixed-tip
  fit covers a single point on the arm. Measure jaw gap for the relevant
  opening and depth, and use supervised piece trials for retention, support,
  release, and retreat. Wrist mounting checks are needed when using the
  wrist camera for metric feedback. [Pickup trials](TOOL_REFERENCE.md#repeat-matched-pickup-trials)

### What another rig needs

Bring the follower's own motor calibration and limits, a measured board
target, lens fits from its cameras, and a rigid base-marker survey. Use the
[registration template](configs/registration.example.json) to give the
23-mark jogger those inputs. The [worksheet](FIXED_TIP_WORKSHEET.md) covers
marking, fitting, and validation. New recordings include a per-mark
board-to-base transform and target so the offline fitter can read them
without the original bench archives. The staged e4/f4 ACT runner still uses
bench-specific taught references.

## What has been validated

**On 24 November 2026, the fitted fixed fingertip was validated in 3D at three
supervised, unloaded board-relative hovers.** The operator's measurements put
the 3D position error below 1.43 mm at each tested pose. Follow-up checks
validated near-board fixed-tip positioning at e4 and f4 within the measured
scope below.

| Check | Recorded result and scope |
|---|---|
| Saved motor calibration and guarded motion | All six motors matched the saved LeRobot calibration on the replacement **USB controller board**, with the same servos. A 3° wrist-roll test moved 2.73° and returned within 0.35° of the start. Two g5 handoffs and an empty e4 rehearsal completed under supervision. A later operator-confirmed e4–f4–e4 knight round trip started with the knight already gripped. This evidence is specific to those paths and that installation. |
| Jaw opening | A single opening excluded from fitting was checked at 5 mm and 10 mm from the fixed tip. The largest difference was 0.83 mm. This checks interpolation for the recorded taped jaws. Ruler uncertainty, powered compression, and grasp-centre accuracy remain open. |
| Camera lens fits | Fresh-photo reprojection RMS was 2.04 px median / 3.62 px maximum for the fixed camera (13 photos) and 1.40 px / 2.04 px for the wrist camera (12 photos). The fits describe image projection. Wrist camera mounting and independent metric accuracy still need validation. |
| Fixed fingertip relative to the board | Three higher hovers checked XYZ position. Two near-board descents checked height at two holds per square. All used the same fitted fingertip point. |

| Board intersection | Height above board | 3D error bound from operator measurements |
|---|---:|---:|
| d4/e4/d5/e5 | 60 mm | < 1.43 mm |
| f4/g4/f5/g5 | 92 mm | < 0.62 mm |
| f6/g6/f7/g7 | 60 mm | < 1.14 mm |

The 3D bounds combine the reported axis errors and measurement resolution. The
first measurement is attributable to the first target, with the exact run
unspecified.

| Near-board square | Requested heights | Measured heights (±1 mm) | Measured minus requested |
|---|---:|---:|---:|
| e4 | 19.2 → 14.2 mm | 21.2 → 15.7 mm | +2.0, +1.5 mm |
| f4 | 18.2 → 13.2 mm | 19.7 → 13.5 mm | +1.5, +0.3 mm |

All four holds had a measured height within 2.0 mm of the request before
accounting for the stated ±1 mm measurement uncertainty. The operator recorded
zero XY offset at all four holds, with ±1 mm uncertainty per axis. The operator
also confirmed at least 5 mm between the playing surface and the lowest pad at
each final hold.

The 5 mm minimum clearance along the full motion path, grasp centre and
retention under load, and wrist-camera mounting accuracy across poses remain
open. After a motor or setup change, recheck the intended working poses using
the procedures below.

This is the calibration baseline for the supervised e4 → f4 workflow.
The staged ACT trial logged a successful pickup and assisted carry. At f4 the
placement policy kept requesting descent until it reached the configured
floor, and the trace recorded no release command. Placement ACT validation
continues separately using that baseline.

See [arm bench history](ARM_CONTROL_HISTORY.md) and [the tool reference](TOOL_REFERENCE.md)
for methods and detailed commands. If you have the original recordings, the
summary sources under `runs/robot-calibration/` are
`calibration-delivery-20261119/CALIBRATION_STATUS.txt`,
`calibration-delivery-20261119-update/CURRENT_STATUS.txt`,
`camera-final-fixed-check-20261117/RESULTS.md`, and
`powered-placement-20261120/rank4-height-calibration.json`. For the November 24
checks, see `board23-tip-analysis/filtered/results.json` and
`board23-tip-analysis/filtered/trial-1-operator-measurement.json`, plus
`operator-measurement.json` in runs `tip-validation-2-1795517325086542632` and
`tip-validation-3-1795517595458971477`. The near-board measurements are in
`operator-measurement.json` in runs `near-board-4-1795522676812525537` and
`near-board-5-1795523363677430887`. These archives are local and optional for
reading this guide.

## First-time software setup

Run commands from the repository root. The mock recorder runs without the arm or
LeRobot. Full regression coverage requires the optional camera and analysis
dependencies described in the [tool reference](TOOL_REFERENCE.md).

```bash
python3 -m robot_calibration.collection.collect_robot_calibration \
  --backend mock --mode joint_response --joint shoulder_pan \
  --output "runs/robot-calibration/mock-$(date +%Y%m%d-%H%M%S-%N)"
```

For real hardware, use a separate Python 3.12 environment. Set `ARM_PORT` to
the port found on this computer and `ROBOT_ID` to the ID of this follower's saved
calibration. Keep the ID consistent between calibration and the collector.

```bash
python3.12 -m venv .venv-robot-calibration
source .venv-robot-calibration/bin/activate
python -m pip install -r robot_calibration/requirements-hardware.txt
lerobot-find-port
```

### Offline tests

The [calibration workflow](../.github/workflows/calibration.yml) runs the
offline suites on GitHub Actions. From the repository root, run these in a
Python 3.12 environment with NumPy, SciPy and headless OpenCV installed:

```bash
python -m unittest discover -s robot_calibration/tests -v
python -m unittest discover -s manipulation/tests -p test_calibration.py -v
```

These suites use mock hardware and saved or synthetic data. The graphical
test skips on runners without a display.

## After replacing a motor

1. **Save the previous revision.** Back up the LeRobot calibration JSON, robot
   profile, and approved limits. Record which joint changed. Support the arm
   and remove motor power before replacing the servo. Confirm the replacement
   matches the follower's motor and gearing. If its ID or baud rate needs
   programming, follow LeRobot's single-motor setup procedure with the replacement
   isolated from the other servos. [LeRobot SO-101 motor setup](https://huggingface.co/docs/lerobot/so101#configure-the-motors)
2. **Run the guided follower calibration.** Close all other controllers, clear
   the manual sweep area and keep the arm supported. This tool writes motor
   settings. If LeRobot 0.7.0 offers to reuse an existing calibration, type
   `c` and press Enter to start a fresh calibration. Pressing Enter alone writes
   the old calibration back to the motors. Follow the guided midpoint and manual
   range capture.
   [LeRobot follower calibration](https://huggingface.co/docs/lerobot/so101#calibrate)

   ```bash
   lerobot-calibrate --robot.type=so101_follower \
     --robot.port="${ARM_PORT:?Set ARM_PORT}" --robot.id="${ROBOT_ID:?Set ROBOT_ID}"
   ```

3. **Check the new readings with the passive command below.** Confirm that all
   six joints are present, the saved calibration matches the motors, and the
   readings and settings agree with the hardware. Recheck the joint references
   against physical measurements. Review measured motion limits and taught
   poses under the new calibration before any powered test. If the gripper or
   pads changed, repeat jaw-gap and contact checks. If the wrist camera mount
   changed, re-estimate its pose relative to the wrist before relying on that view.
   [Joint reference measurements](ARCHITECTURE.md#protractor-reference-pose-procedure)
4. **Verify a small supervised empty-arm movement first.** Use a locally
   reviewed path, compare physical direction, travel and return with the
   encoder record, and save the resulting calibration as a new revision.
   Recheck the intended board positions using the
   [independent checks](FIXED_TIP_WORKSHEET.md#validate-the-frozen-fit).
   [Guarded joint-response procedure](TOOL_REFERENCE.md#prepare-a-small-movement-test)

## After moving the setup

1. **Check what moved relative to what.** If the whole assembly moved rigidly,
   verify that the clamp, marker mounts, cameras and board stayed fixed relative
   to each other. If the board, base or fixed camera moved relative to the others,
   take fresh measurements of the layout. Keep the saved motor calibration if
   the motor and joint assemblies are unchanged and the
   [passive check](#passive-check-and-what-to-save) confirms a match.
   Lens fits can be reused after checking that the camera, focus, crop,
   orientation and resolution match. If a wrist camera shifted in its mount,
   its wrist-relative transform needs a new fit and checks.
2. **Photograph the current scene.** With the arm stationary, capture raw
   fixed-camera photos showing the board markers and a rigid base target with
   measured coordinates relative to the robot base. Reserve visible board
   markers for a check. If the marker arrangement differs from the checked-in
   target, supply your measured target JSON via `--target`. Set
   `FIXED_INTRINSICS` to the saved lens fit and `SCENE_PHOTOS` to the new photo
   directory. Rerun with the matching wrist fit and photos if you use the wrist view.
   [Fit or check lens intrinsics](TOOL_REFERENCE.md#camera-calibration-from-photos)

   ```bash
   python -m robot_calibration.board_pose \
     --candidate "${FIXED_INTRINSICS:?Set FIXED_INTRINSICS}" \
     --images "${SCENE_PHOTOS:?Set SCENE_PHOTOS}" --holdout-ids 8 9 \
     --output "runs/robot-calibration/board-pose-$(date +%Y%m%d-%H%M%S-%N)"
   ```

   Choose other reserved IDs if 8 and 9 are hidden. Keep at least four known
   markers for fitting, and inspect the held-out results and overlays. This
   command locates the board **in the camera frame**. The jogger's explicit registration
   inputs let it also estimate the board position relative to the arm at
   each mark, assuming a level board at the measured height. If you lack
   a verified base survey, stop at the camera-frame result. Use independent
   physical checks before relying on the new registration for motion.
   [Scene registration design](ARCHITECTURE.md#reconnect-workflow-and-independent-final-validation)
3. **Review clearance and measure independent points.** Check the new board
   height, reachability, and clearance along the entire path. Start with
   supervised, empty-arm, non-contact holds at independently measured
   landmarks on both board halves. Compare measured XYZ and uncertainty against
   the tolerance for the intended task. Compare the two views if both are used.
   Missing observations or errors outside tolerance leave the setup unverified.
   [Operator guide](ARM_OPERATOR_GUIDE.md)

## Passive check and what to save

Use one terminal owner for the serial port. Support the arm and keep physical
power-off accessible. Read mode records the present state without sending
position goals or changing torque.

```bash
python -m robot_calibration.collection.collect_robot_calibration \
  --backend lerobot --mode read --telemetry --duration-s 5 \
  --port "${ARM_PORT:?Set ARM_PORT}" --robot-id "${ROBOT_ID:?Set ROBOT_ID}" \
  --output "runs/robot-calibration/read-$(date +%Y%m%d-%H%M%S-%N)"
```

Check `metadata.json` and `samples.jsonl` for a complete six-joint record and
resolve any mismatch or read failure before planning motion. Before any powered
test, review the specific plan, start pose, clearances, and recovery procedure
with the operator. Save the hardware and camera IDs, calibration and scene
revisions, raw observations, independent errors with units, and pass/fail/not
tested status. Keep the raw runs and machine-specific calibration files local.
[Operator recovery procedure](ARM_OPERATOR_GUIDE.md#6-stop-recover-and-diagnose-a-timeout)
