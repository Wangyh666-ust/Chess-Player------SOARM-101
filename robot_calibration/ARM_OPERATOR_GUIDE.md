# Operating the SO-ARM101 chess follower from a terminal

See the [arm control incident history](ARM_CONTROL_HISTORY.md) for the failures, changes and successful bench runs behind this procedure.

For a Codex-operated session or an incomplete motor read, start with the [Codex arm handoff and retry guide](CODEX_ARM_HANDOFF.md).
This guide describes the single-pose powered-placement workflow on the chess-arm desktop, checked against the runner and saved recordings on 22 November 2026. A person at the bench controls motor power, supports the arm and takes the measurements. One terminal process owns the serial bus, sends incremental position goals, reads the servos and records the evidence. A second terminal can calculate a plan from files while the first process continues holding. No chat or leader arm is required.

**The arm falls when torque goes off. On this replacement board, writing the current positions back to Goal_Position has energized all six motors. Treat that first goal write as the start of powered operation, even for a stationary measurement.** A lit servo LED, USB enumeration and an old log entry each answer different questions. Only a fresh complete register sample from the owner establishes the reported torque state.

## 1. Identify the installation and the current owner

Run these in Bash from the repository root on the arm computer. These commands inspect the OS and files without opening the serial port.

```bash
arm_port="${ARM_PORT:?Set ARM_PORT to the verified follower port}"
readlink -e "$arm_port"
fuser -v "$arm_port"
ps -eo pid,tty,etime,args | rg 'powered_placement|collect_robot_calibration|/run.py|recovery'
find runs/robot-calibration -name recovery.jsonl -printf '%T@ %p\n' | sort -nr | head
```

Match the PID, terminal and output directory from the command line. Read the latest lines of that specific run's `samples.jsonl`, `recovery.jsonl` and `metadata.json` using `tail` or `jq`. Check the file modification time with `stat`, then check whether the log advances. A stale file does not establish that torque is off. An empty ownership result is meaningful only when the device exists and the ownership check succeeded.

**If an owner exists**, use its original terminal for operator input. A new shell or chat can inspect the logs but cannot assume ownership of its prompt. If the old terminal is inaccessible, arrange handoff with its operator. If recovery must end, physically support the arm and anything held, remove motor power and confirm this in the owning terminal when available. Resolve the old owner before opening the bus again. Never kill a holding controller to make the port available.

Desktop hardware identity is `yhwang_follower_arm`, with the replacement USB board in the incident history, the existing follower motor supply (12 V on this installation), six saved servo calibrations and LeRobot 0.7.0 in `.venv-robot-calibration`. Verify the supply rating on any different installation. The operator keeps physical power-off accessible. Revisit path clearance when the board, piece, arm start or surrounding setup changes.

## 2. Preview offline or read the arm passively

```bash
scripts/measure-chess-arm preview --plan "$REVIEWED_PLAN"
.venv-robot-calibration/bin/python -m robot_calibration.collection.powered_placement --help
```

Set `REVIEWED_PLAN` to the local plan file. Preview describes the plan
without opening devices. For execution, supply the current robot ID, port,
camera, and a fresh output directory. A new first plan also needs the SHA-256
from the operator's local review. The runner compares it with the exact file
it loads. Keep old bench plans and their historical device settings with the
rig they describe.

For **one passive capture**, after establishing exclusive access and with the servo electronics powered:

```bash
.venv-robot-calibration/bin/python -m robot_calibration.collection.collect_robot_calibration \
  --backend lerobot --mode read --telemetry --hz 10 --duration-s 5 \
  --port "$arm_port" --robot-id "${ROBOT_ID:?Set ROBOT_ID to the saved follower ID}" \
  --output "runs/robot-calibration/read-$(date +%Y%m%d-%H%M%S-%N)"
```

The adapter opens the Feetech bus through `arm.bus.connect()`, verifies saved calibration against the motors, and reads existing settings. Passive mode sends no goal and disconnects without disabling torque. Do not use it as a second reader alongside teleoperation. For an offline rehearsal of the collector, use the README's mock joint-response command and a fresh output directory.

## 3. Understand the values before writing

| Value | Meaning |
|---|---|
| Present_Position | Measured shaft encoder position. Raw reads use integer ticks. The SDK converts the first five joints to calibrated hardware degrees and the gripper to normalized opening 0–100. |
| Goal_Position | Servo target register. Reading it confirms the stored target. Compare Present_Position separately to see whether the shaft arrived. |
| Operating_Mode, Torque_Enable, Lock, Status | Check position mode, reported torque on every motor, lock readback and servo fault flags. |
| P/I/D_Coefficient, Max_Torque_Limit | Actual servo control settings read at connection. Compare all six with the reviewed plan before motion. |
| Present_Temperature, Present_Load, Present_Current | Temperature guard and diagnostic telemetry. Load and current alone do not measure pad contact force. |

The archived first powered g5 handoff had the following **readbacks**. These are comparison evidence, not settings to write after a mismatch.

| Register | Shoulder pan | Shoulder lift | Elbow flex | Wrist flex | Wrist roll | Gripper |
|---|---:|---:|---:|---:|---:|---:|
| P | 16 | 64 | 64 | 32 | 16 | 16 |
| I | 0 | 0 | 0 | 0 | 0 | 0 |
| D | 32 | 32 | 32 | 32 | 32 | 32 |
| Max torque limit | 1000 | 1000 | 1000 | 1000 | 1000 | 500 |

Simulation radians, board millimetres, motor ticks, hardware degrees and normalized jaw opening are distinct. Use the saved hardware calibration for tick conversion and the recorded joint mapping for simulation replay. Model-only joint offsets, taped-pad meshes and camera corrections are not servo commands. Even perfect shaft tracking can leave a fingertip placement error.

## 4. Start one reviewed hold

Use a local interactive terminal. With the arm supported at the reviewed start, supply on and torque off on all six motors, choose **one** command:

| Command | First action after verified startup |
|---|---|
| `scripts/measure-chess-arm measure-start` | Hold the current reviewed start for measurement. Zero trajectory commands, but startup still writes the energizing goal. |
| `scripts/measure-chess-arm run` | Move along the reviewed desktop e4 path, then hold. |

Supply the same execution options to either mode. See the [powered measurement command](TOOL_REFERENCE.md#powered-fingertip-measurements-from-a-terminal). The powered runner follows this sequence:

1. Check port ownership, match the full hardware snapshot including gains, compare actual start ticks with the reviewed start (within two ticks), check the path and configure raw travel envelopes. Prepare the operator console and recovery log. Start video and collect ten passive encoder samples.
2. Read existing goals individually by motor. Require torque off and position mode. Write and verify Lock on each motor. Verify fresh positions, faults and three stationary passive samples against the intended seed.
3. Issue **one synchronized raw Goal_Position write** for all six quantized starting positions. The runner marks the arm as potentially energized before sending it so failures go through recovery.
4. Read torque immediately and read each goal back by motor address. Check positions, status, lock and temperature. A mixed torque sample keeps startup in read-only verification. Require two consecutive complete all-on checks inside the original 0.5-second window. One checksum error on torque or addressed goal readback, or one addressed goal-read timeout, can be discarded and re-read after additional checks. No second seed write. Failure to verify cancels the trajectory.
5. Stream the reviewed path on the same connection. The placement runner subdivides it into checkpoints of at most one degree per moving joint, with scheduled command slots 0.1 seconds apart. Before each write it checks video, operator input and raw feedback against the **previous** expected goal. `arm.send(command)` uses the SDK's normalized synchronized Goal_Position write. The runner records both the joint command and its raw equivalent. Final arrival has its own tracking check. This is timed host-side position streaming; each servo runs its own local position controller.

During movement, a feedback batch has a 0.5-second deadline. The runner checks complete integer samples, exact goal readback, all-on torque, status, temperature, joint envelopes, drift and tracking. Its elbow window is 26 ticks during intermediate travel and 8 at checkpoints and arrival; other active joints have an 8-tick window. The motion schedule allows at most 0.2 seconds lateness and 55 seconds total. These are implementation limits for this runner. Occupied-square handoffs additionally need a reviewed collision screen for the actual piece position, jaw opening, links and placement uncertainty.

## 5. Measure and continue without cycling power

Wait for **REACHED** in the standard ruler workflow. Measure the same fixed-finger extremity, record the reference surface and enter two numbers in mm, for example `70.4 0.5`. The second number is ruler uncertainty. If a 3.7 mm datum offset applies, retain the raw reading, sign and reference surfaces so the offset is applied exactly once. Distinguish vertical measurements from visual horizontal centering and report parallax when relevant. A corrected reading at the same pose supersedes the original and counts once.

Plan one next pose from the saved hold in a **second terminal**. Example syntax, with the actual hold path and a fresh plan filename:

```bash
scripts/measure-chess-arm plan --hold /path/to/session/hold-1.json \
  --profile "$ARM_PROFILE" --scene "$SCENE" --tip "$TIP_FIT" \
  --square d5 --height-mm 80 --output /tmp/placement-next-01.json
```

Here 80 means modeled height in mm. Inspect the proposed target, confirmed scene, joint travel and whole-arm clearance. Once reviewed, type `next /tmp/placement-next-01.json` in the **original controller terminal**. The runner checks the held goal and live start before continuing. It also accepts `snapshot` during a hold to save measured encoders and the held goal separately. The 180-second hold deadline includes planning time. The operator needs to measure; the software handles the intermediate chunks.

`plan --height-calibration FILE` takes physical height and applies the saved **7.75 mm local rank-4 fixed-tip bias** within its recorded scope. Never apply it to the shoulder, all board ranks or every gripper surface. The November 24 near-board trials confirmed at least 5 mm under the lowest pad at the final holds on e4 and f4. Clearance along the full motion path still needs verification. See the [calibration results](README.md#what-has-been-validated).

## 6. Stop, recover and diagnose a timeout

Enter `stop` in the hold terminal to finish. During a trajectory, operator input cancels movement. Completion, a fault or a deadline can leave torque **on** in supervised recovery. Follow the owning console and recovery log. The placement wrapper disables the software release command. Once the arm and any piece have mechanical support, the operator physically removes supply power, **then** enters `power-off-supported`. This is confirmation of an action already completed. A routine transition between verified holds needs no power cycle.

| Last evidence | Next action and reason |
|---|---|
| Passive read failed before any goal write | Inspect the exact failed register, exclusive ownership, selected port and supply state. Once stable, repeat the passive read after a short wait. |
| Seed write attempted, mixed torque, or goal readback uncertain | Treat the arm as energized. Keep the owner. Inspect the six torque values, addressed goal reads, drift and elapsed times. Only the runner's bounded verification retries are appropriate here. |
| Tracking, fault, thermal, collision or timing stop during the path | Preserve the trace. Compare the last expected and returned goals, actual ticks, offending joint, status, temperatures, camera status and deadlines. Correct the specific cause and review a path from the actual start before any new trajectory. |
| Original terminal unavailable or logs stale | Resolve ownership and physical state as in section 1. Do not start another controller or repeatedly power-cycle to clear an unexplained stop. |

## 7. Keep an auditable recording

Keep the entire run directory, including failed attempts. Save the plan, scene registration and physical measurement notes alongside it. The standard launcher uses the fixed camera; the historical ACT handoffs used a separate recorder with **fixed and wrist** video. The generic side-camera option labels its second stream Angetube. Verify the physical camera identity, resolution and framing before treating any second stream as the wrist feed.

- `metadata.json` records hardware identity, calibration, gains, status, errors and recovery outcome.
- `samples.jsonl` records monotonic event times, passive samples, expected goals, feedback, arrival, hold and operator measurement events. `hold-N.json` stores the commanded hold; a `measured-*.json` snapshot distinguishes actual ticks.
- `recovery.jsonl` records the stop sequence, feedback and operator confirmation. Check the last events even when the target was reached.
- Match the ruler event or steady hold to the nearest video frame using its stream's `frames.jsonl` and monotonic read interval. Report timing uncertainty. For accuracy, compare **requested height**, **encoder-predicted height during the hold** and **physical reading** separately.

For a worked dual-camera example, inspect [the first g5 hold](../runs/robot-calibration/wrist-mount-validation-20261120/g5-higher-handoff-recording-01/) and its [reviewed runner](../runs/robot-calibration/wrist-mount-validation-20261120/g5-higher-handoff-01/run.py). For startup evidence, inspect [the replacement-board archive](../runs/robot-calibration/powered-placement-20261120/board-replacement-20261120-171724/), especially the verified seed run's recovery log. Run archives are local and may need to be copied separately when sharing the guide.

Implementation references: [desktop launcher](../scripts/measure-chess-arm), [passive bus adapter](collection/collect_robot_calibration.py), [placement loop](collection/powered_placement.py), [register guards and startup retry rules](collection/placement_guards.py), [supervised recovery](collection/stop_recovery.py), and [installation, mock runs and collector options](TOOL_REFERENCE.md).
