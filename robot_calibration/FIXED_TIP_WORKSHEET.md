# Fixed fingertip and jaw calibration worksheet

Use this worksheet after checking the arm, camera, and current board-to-base
registration in the [calibration README](README.md). Use the 23-point session
to fit a new tool offset. To verify a saved offset or check a new working
height, go directly to [Validate the frozen fit](#validate-the-frozen-fit).

## Prepare the rig and measurement references

- Photograph the exact point you will measure on the fixed finger. Use the
  same point throughout, including after moving the ruler or camera. Record
  the pad condition and jaw opening. The jogger's displayed wrist point is
  the model origin of `gripper_link`, so it differs from the point you measure.
- Define **0 mm** at the playing surface. Measure vertical heights from that
  surface to the fixed tip. Record the ruler resolution and any datum offset.
  Mark the board intersection precisely and use a clear side view or a height
  reference to reduce parallax. Board X runs from a1 toward h1, Y from a1
  toward a8, and Z points up from the playing surface.
- Save the reviewed jog session, local limits and recovery configuration,
  hardware calibration, measured board target, lens fit, and base-marker
  survey. Check the current setup and the permitted jog volume with the
  operator before moving. [Jogger setup](VISUAL_CONTROLLER.md#hardware-preparation-and-launch) · [Recovery](ARM_OPERATOR_GUIDE.md#6-stop-recover-and-diagnose-a-timeout)

## Configure and open the jogger

Copy [the registration template](configs/registration.example.json) into your
local calibration directory and fill it in. The two paths are relative to the
registration JSON. Use your measured board target and fixed-camera lens fit.
List at least four board anchor IDs and keep another visible board marker
available for checking. Give each rigid base marker its ID and four ordered XYZ
corners in the robot-base frame in metres, in the same corner order returned by
the detector. Base IDs must differ from board IDs. The board height is the
playing surface Z in the robot-base frame. The two cell pitches are measured
along board X and Y, in metres. The registration fits X, Y and yaw for a level
board. See [jogger setup](VISUAL_CONTROLLER.md#hardware-preparation-and-launch)
for the separately reviewed session, limits, and recovery configuration.

With the local operator present, set the environment variables to your
reviewed session and verified devices, then run from the repository root:

```bash
scripts/jog-arm --backend lerobot --board-23 \
  --session "${JOG_SESSION:?Set JOG_SESSION}" --registration "${REGISTRATION:?Set REGISTRATION}" \
  --port "${ARM_PORT:?Set ARM_PORT}" --robot-id "${ROBOT_ID:?Set ROBOT_ID}" \
  --fixed-camera "${FIXED_CAMERA:?Set FIXED_CAMERA}" --wrist-camera "${WRIST_CAMERA:?Set WRIST_CAMERA}" \
  --output "runs/robot-calibration/board-23-$(date +%s%N)"
```

Fixed capture uses the resolution in the fixed lens fit. The wrist stream
defaults to 1280 × 720 and can be set with `--wrist-resolution WIDTH HEIGHT`.
The standalone `--board-contacts` and `--repeat-60` modes use the same inputs.
See the [operator recovery procedure](ARM_OPERATOR_GUIDE.md#6-stop-recover-and-diagnose-a-timeout).

## Record the 23 marks

The jogger names each intersection by its four surrounding squares. The map
uses the **lower-left square** as shorthand, so `d4` here means the shared
corner of d4/e4/d5/e5. It differs from the centre of d4.

```text
Lower-left square at each intersection, viewed from above

Rank      a    b    c    d    e    f    g
  7      12    ·    ·    ·    ·    ·    9
  6       ·   11    ·   10    ·    8    ·
  5       ·    ·    ·    ·    ·    ·    ·
  4       ·   13    ·    1    ·    7    ·
  3       ·    ·    ·    ·    ·    ·    ·
  2       ·    3    ·    2    ·    5    ·
  1       4    ·    ·    ·    ·    ·    6
```

| Marks | Height above the playing surface | Lower-left squares in order |
|---|---:|---|
| 1–13 | 0 mm | d4, d2, b2, a1, f2, g1, f4, f6, g7, d6, b6, a7, b4 |
| 14–18 | 60 mm | d4, a1, g1, g7, a7 |
| 19–23 | 92 mm | d4, a1, g1, g7, a7 |

1. Jog to the prompted intersection and height under local supervision.
   Approach surface points gently and check the actual point and height before
   saving. At elevated marks, verify horizontal alignment as well as height.
2. When the arm is stationary, select **Save measurement and advance**. If the
   panel asks for fresh camera and joint feedback, wait and save the same mark.
   Save an operator note for uncertain alignment, an obscured tip, or a
   questionable height. Use the mark number and run directory to identify it.
   Keep a saved questionable mark in the log and flag it for review. The
   displayed height is the requested height. Record any measured discrepancy
   with the mark so the fit can use the actual observation.
3. Keep `metadata.json`, `samples.jsonl`, both camera streams, and the
   operator notes. For an interrupted recording, pass
   `--resume-board-23 PREVIOUS_RUN` with `--board-23` and a fresh output
   directory. The jogger checks numbering and starts at the next mark. Preserve
   each segment of the session. If the rig moved between segments, refresh the
   scene registration and record the change before combining measurements.

Copy this row for each mark or check.

| Run and mark | Board intersection or square centre | Target height (mm) | Measured XYZ or gap (mm) | Uncertainty and visibility | Use for fit, check, or exclude |
|---|---|---:|---|---|---|
| | | | | | |

## Turn the marks into a fit

The marks in `samples.jsonl` have type `board_contact` or
`board_manual_hover`. Use the recorded joint state, camera registration and
board landmark for each mark. A `board_only` or `partial` observation leaves
out a valid per-mark board-to-base pose. Keep those rows and any uncertain
operator marks in the record with the reason for excluding them.

For each usable mark, express the board landmark at its measured height in the
robot base frame. Then use the recorded joint angles and arm model to express
that target in `gripper_link` coordinates. Fit one fixed point from those
per-mark estimates, using lower weight for less certain measurements.
When checking the fit, omit every mark at the same intersection in all height
groups. Compare the predicted tip with that intersection's observed board-frame
position and report signed XYZ and total error in millimetres. Freeze the
chosen fit before collecting independent checks.

**Worked example from the recorded bench.** Mark 1 was the shared
d4/e4/d5/e5 corner at 0 mm. Its complete board-to-base observation made it
usable for fitting. When that intersection was held out, the reported 3D
prediction error for mark 1 was 1.33 mm. The final fit used **16 of 23 marks**.
Four were excluded for uncertain horizontal alignment, two were board-only,
and one was partial. The resulting point was approximately
`[-3.96, -5.24, -110.58]` mm in `gripper_link` coordinates.

Run the offline fitter with the completed recording directories. When a session
was resumed, supply each segment in order:

```bash
python -m robot_calibration.analysis.fit_fixed_tip \
  --run "$FIRST_RUN" "$CONTINUED_RUN" --review "$MARK_REVIEW" \
  --output "$TIP_FIT"
```

Omit the second run if the session was completed in one recording. Omit
`--review` if you have no overrides. The review JSON is an object keyed by
`run-directory-name/mark-number`. For an uncertain mark use
`{"exclude": "horizontal alignment obscured"}`. To record a measured target
instead of the prompted one, use `{"measured_board_m": [x, y, z]}` with the
measured coordinates in metres. Add `uncertainty_mm` to set the fit weight.
Marks without a review weight have equal weight. The output gives the fixed
point in metres in `gripper_link`, mark status and held-out residuals, input
hashes, and `motion_ready: false`.

Older runs stored the registration parameters and prompted mark separately.
Pass `--legacy-pitch-m X Y` and `--legacy-board-height-m Z` with the measured
geometry from those runs. To reproduce the archived fit also use its exclusion
notes and weights. The original `fit.py` and `results.json` remain under
`runs/robot-calibration/board23-tip-analysis/filtered/`.

## Validate the frozen fit

| Check | What to measure and record |
|---|---|
| **Independent working holds** | With the fit frozen, choose fresh, reviewed holds spanning the task positions and heights. Measure physical X, Y and Z, and record uncertainty. For near-board placement, use the intended square centres and working heights. Add higher holds if they cover a different operating condition or let you measure axes that were obscured. |
| **Clearance** | At the same holds, measure the gap from the lowest pad or other relevant part to the playing surface or obstacle. Record the part and reference surface. A gap measured at a hold covers that hold. For a path claim, verify clearance along the motion as well. |
| **Return repeatability, when needed** | Revisit an independent hold without refitting. Compare the physical measurements between visits. For a dedicated repeatability check, `--repeat-60` visits the d4 and g7 intersections twice at 60 mm. Resume with `--resume-repeat-60 PREVIOUS_RUN` and a fresh output directory. |
| **Jaw geometry, for grasping** | At known openings, measure the inside gap between the pads at stated depths from the fixed tip. Record the encoder opening with each measurement and reserve an intermediate opening to check the map. The original check used 5 and 10 mm depths. Assess the two-jaw grasp centre and loaded retention with supervised piece trials when needed. |

Before the check, write down the task, the arm profile and scene revisions,
the locations and heights, and the acceptable XYZ error and minimum clearance
for the task. Afterward, report each signed error as measured minus requested.
Compare the error with the task tolerance, accounting for measurement
uncertainty. Mark a result inconclusive if the measurement or its uncertainty
cannot distinguish pass from fail. Record the outcome and exact scope with the
saved fit.

For example, the e4 low hold requested **14.2 mm** and measured **15.7 ±1 mm**.
The signed height error was **+1.5 mm**, with measurement uncertainty kept
separate. The operator recorded **0 ±1 mm on each horizontal axis** and
confirmed **at least 5 mm** between the lowest pad and the board at that hold.
[Recorded calibration results](README.md#what-has-been-validated)
