# Recorded simulation evidence

Source experiment: `ideal_grasp_continuous_3x10_v1` (2026-10-12).

- `raw-audit.json`: same-GPT whole-batch audit; 6/10, 10/10, 10/10 completed plies. Original local trace paths and SHA-256 identifiers describe archived source records; those full traces are not bundled here.
- `replay-provenance.json`: selected `random2` action/home segments, frame indices, recorded timestamps, pose hashes and encoded MP4 hash. Source-relative paths identify the original local experiment, not additional downloadable files in this repository.
- `decode-qa.json`: all-frame decode checks, dimensions, duration and source-pose checks. The separate `visual-review.json` records completed human-style GPT inspection.
- `recorded-poses.json.gz`: all 877 selected robot/piece poses used to render the video, in sequence; contains all ten actions and returns. These are playback data, not a control policy.
- `visual-review.json`: review of the decoded poster and ten-move montage.

Published video: [`../../docs/assets/simulation-random2-10plies.mp4`](../../docs/assets/simulation-random2-10plies.mp4).

MP4 SHA-256: `0f6974ade01a9377de9dc7dc1c9518a9477475a54d4b6cd99228132901419374`.

The replay adds zero physics steps and uses variable playback speed. The manipulation run uses an **ideal fixed relative translation and rotation** during carrying, detached before release. See [the experiment description](../../SIMULATION.md) for the full scope and limitations.
