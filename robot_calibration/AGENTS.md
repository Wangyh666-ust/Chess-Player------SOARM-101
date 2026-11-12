# Robot calibration handoff

Read `README.md` in this directory before running the collector. Commands in that
README assume the repository root as the working directory. This module records
measurements; it does not implement autonomous grasp planning or force control.

- Start with `python3 -m unittest discover -s robot_calibration -v` and the README's
  mock joint-response run, using a fresh output directory. Mock success is not
  hardware validation. Real serial communication and camera codecs remain untested.
- Install hardware dependencies only in `.venv-robot-calibration`, separate from
  simulation. The adapter targets LeRobot 0.7.0; review compatibility before
  changing that pin. Do not alter the frozen simulation runtime for this handoff.
- Establish the actual port, robot ID and saved calibration with the local
  operator. Close other serial controllers before collection. Begin with passive
  readings, then passive telemetry/video, before attempting movement.
- Before physical motion, obtain the local operator's readiness confirmation for
  the specific experiment, including a clear path, supported arm and accessible
  power-off. Torque is disabled after motion and the arm can fall. Do not supply
  the pickup prompt's `ready` response on the operator's behalf.
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
  output paths and unresolved failures. Follow the README's results checklist;
  keep recordings and local configurations out of the source commit.
