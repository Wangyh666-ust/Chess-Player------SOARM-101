# Robot calibration handoff

For serial ownership, incomplete six-motor reads, mixed startup torque or a chat handoff, consult [CODEX_ARM_HANDOFF.md](CODEX_ARM_HANDOFF.md) before opening the bus.
Read `README.md` in this directory before running the collector. Commands in that
README assume the repository root as the working directory. This module records
measurements; it does not implement autonomous grasp planning or force control.

Read `ARCHITECTURE.md` before implementing commissioning or reconnect workflows.
Use the `collection`, `analysis`, `configs` and `tests` subdirectories and package
entry points (`python -m robot_calibration...`). Planned workflows are not proof
that hardware readiness or physical validation has already been implemented.

- Start with `python3 -m unittest discover -s robot_calibration/tests -v` and the README's
  mock joint-response run, using a fresh output directory. Treat the recorded
  hardware results as evidence for their original installation and capture modes.
- Install hardware dependencies only in `.venv-robot-calibration`, separate from
  simulation. The adapter targets LeRobot 0.7.0; review compatibility before
  changing that pin. Do not alter the frozen simulation runtime for this handoff.
- Establish the actual port, robot ID and saved calibration with the local
  operator. Close other serial controllers before collection. Begin with passive
  readings, then passive telemetry/video, before attempting movement.
- Before physical motion, obtain the local operator's readiness confirmation for
  the specific experiment, including a clear path, supported arm and accessible
  power-off. Completion and abort enter supervised recovery; torque release requires
  confirmed mechanical support for the arm and any held piece. Do not supply
  the pickup prompt's `ready` or recovery release commands on the operator's behalf.
- Never invent joint limits or taught poses, use test fixtures on hardware, or
  copy simulation radians into this collector. Arm units are calibrated degrees;
  gripper units are normalized 0–100. Joint bounds do not prove collision clearance.
- Preserve preflight checks, tracking/timing limits, manual resets and aborts.
  Investigate failures instead of bypassing guards, changing motor gains/IDs or
  automatically retrying movement. The example JSON files deliberately contain
  non-executable null values.
- Use new output directories under `runs/robot-calibration/`. Retain failed runs
  and raw recordings. Do not convert current/load registers to newtons or label
  a completed playback as a successful pickup. Separation needs video review;
  an occluded event remains uncertain.
- At handoff, report commands, commit/version, actual hardware checks performed,
  output paths and unresolved failures. Follow the [results checklist](TOOL_REFERENCE.md#read-the-results);
  keep recordings and local configurations out of the source commit.
- Offer the visual jogger at the beginning of supervised motion tests so the
  operator can position the arm. Keep one serial owner through handoff, record
  setup separately from policy actions, and register from fresh frames afterward.
- For supervised tests, accept the operator's confirmation of the task pose
  and read the current encoders. When a taught pose is required, plan the
  approach from those readings rather than asking the operator to reproduce
  saved ticks. Apply taught registration offsets only where motion uses them.
