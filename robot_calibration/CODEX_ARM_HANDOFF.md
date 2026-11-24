# Using Codex when an SO-ARM101 motor read looks incomplete

See the [arm control incident history](ARM_CONTROL_HISTORY.md) for the failures, changes and successful bench runs behind this procedure.

**First identify who owns the port and when the read happened.** A missing reply leaves that motor's state unknown. A complete reply containing both zeroes and ones reports mixed torque for that sample. The placement runner handles these cases differently. This guide applies to the chess desktop's replacement board and the current single-pose runner.

## Paste this into a new Codex chat

> Work from the repository root. Read robot_calibration/AGENTS.md, robot_calibration/CODEX_ARM_HANDOFF.md and the operator guide. First inspect the controller PID, terminal, serial ownership and freshness of the matching run logs without opening the port. Report the phase, register, six joint names and observed values or missing replies, and the last verified event. If a fresh passive retry is appropriate, do it on the authorized read-only path without cycling power. If startup verification is already running, follow its trace. Keep one controller and all motion guards. I control power and torque release. Ask me only when a physical action or terminal handoff is actually required.

## 1. Recover the live context without touching the bus

Use the [operator guide's ownership commands](ARM_OPERATOR_GUIDE.md#1-identify-the-installation-and-the-current-owner). Match the controller's PID and TTY to its output directory. Inspect that run, since the newest directory can belong to a failed attempt. With `run_dir` set to the matched directory, these commands only read files:

```bash
stat "$run_dir"/{metadata.json,samples.jsonl,recovery.jsonl}
tail -n 12 "$run_dir/samples.jsonl"
tail -n 12 "$run_dir/recovery.jsonl"
jq '{status,error,recovery}' "$run_dir/metadata.json"
```

Some files appear only after their phase begins. Check timestamps twice if freshness matters. Use the raw JSON line if a pretty-printed tool response was truncated. Count the actual keys for **one register in one sample**: shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll and gripper. Five returned ones cannot verify six motors. Never fill the missing value from an earlier packet. The status of a previous run, an illuminated LED and the word “on” do not give per-motor torque readback.

If the old process is holding or in recovery, use its existing terminal and logs. A read-only collector still opens the serial port and would compete with that process. If Codex lacks the old terminal, tell the operator which PID, TTY and run need attention and which prompt is pending. Keep the arm supported if power must be removed. Leave physical power removal and its acknowledgement to the operator.

## 2. Choose the retry by phase

| Last established phase and evidence | What Codex does |
|---|---|
| Standalone passive read timed out before **any** write; prior read-only process has exited | Inspect the error and verify the port is free. If power is already on, allow a few seconds for the bus to settle, then repeat the authorized passive capture once using the operator guide's read command and a new output directory. Compare all six replies. A repeated missing joint or checksum error is a diagnostic finding; report its name, register and timing before another attempt. |
| Fresh startup, before seed, all six torque flags zero | This is the expected passive state. The runner checks saved calibration, actual P/I/D and torque limits, exact joint coverage, start position, operating mode, locks and faults. A nonzero flag at this stage requires resolving the existing powered state before fresh startup. |
| After the seed, all six flags present but mixed, or all zero | The **owning runner** polls again. It needs two consecutive fully verified all-on samples within the original 0.5 seconds. Each pass also checks addressed goals, status, locks, drift and temperature. Look for `mixed_torque_pending`, followed by `all_powered_verified` or an error. Nothing for the operator to switch while this verification is progressing. |
| After the seed, a checksum error on torque or addressed goal readback, or an addressed goal-read timeout | The runner may discard **one** such failed read across that startup window, reset the consecutive-success count, and check status, lock and position before retrying. Look for `startup_read_discarded`. The deadline stays fixed. |
| After the seed, a torque reply times out or a torque key is missing | Keep the seeded goal latched, check status, locks and position, and re-read until two complete all-on confirmations within the existing startup window. During guarded placement feedback, poll missing torque replies within the feedback deadline before issuing the next command. |
| After the seed, another joint key is missing, a value is invalid, or the deadline expires | The current runner treats this as failed verification. Follow recovery and inspect the trace. Treat every servo as potentially energized. A later passive diagnostic is a separate operation after ownership and physical state are resolved. |
| A flag drops, a goal changes, or tracking, fault, temperature or timing trips during motion or hold | Let the guard cancel the trajectory. Inspect the last expected goals and the actual registers. A fresh motion attempt needs its own reviewed live start. A startup read retry does not authorize replaying the interrupted move. |

**Retry reads only in the applicable phase.** The single synchronized seed write to Goal_Position can energize all six motors on this board. Even an exception around that write can leave torque on. Never resend it to “see if it worked,” issue a torque toggle, restart the old paired wrist experiment, or launch a replacement controller to clear a timeout. The runner reads each Goal_Position by motor address during seed verification. It keeps torque checks, goals and measured positions distinct.

## 3. Continue or hand off

After `all_powered_verified`, wait for arrival and the hold prompt. Once a powered measurement is saved, the operator or Codex can prepare the next reviewed plan from files and use the original controller's `next PATH` prompt. This is how to collect several poses without power-cycling between them. If recovery is latched, follow its actual prompt. Only the operator enters `power-off-supported`, after mechanical support and physical power removal.

For a handoff, record **PID/TTY, port, run path, last fresh timestamp, phase, last write attempted, register and six raw values or missing names, guard outcome and next required action**. If torque state is unknown, say so. If no operator action is needed, continue the permitted log check or passive retry instead of asking for another “on.”

[Register guards and bounded startup verification](collection/placement_guards.py) · [Powered runner](collection/powered_placement.py) · [Commands, measurement and recovery](ARM_OPERATOR_GUIDE.md)
