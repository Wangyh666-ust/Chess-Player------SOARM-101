# SO-ARM101 Chess Player — Simulation Control

**Current control snapshot: 10 November 2026.** The PyBullet workflow uses the project's scanned-piece geometry, 25.223 mm board squares and native-contact holding with natural rotation. It performs measured-grasp path rebuilding, support feedback before opening, settled release, and independently checked withdrawal/home.

**[English control handoff](CONTROL_HANDOFF.md)** · **[Install and run](simulation/README.md)** · [Download handoff ZIP](https://wangyh666-ust.github.io/Chess-Player------SOARM-101/assets/control-handoff-20261110-v2.zip) · [Current evidence](artifacts/control-20261110/README.md) · [Project page](https://wangyh666-ust.github.io/Chess-Player------SOARM-101/)

This handoff is for a developer taking over **simulation first**. It covers the exact environment, current models and frames, source map, control phases, reproducible commands and parameters that may be unnecessarily strict. Running it requires no API key, GPU, vision model or real arm.

| Session | Complete / planned | Outcome |
| --- | ---: | --- |
| seed 101 | 0 / 10 | Queen placed/settled; post-release exit planning blocked |
| seed 202 | **10 / 10** | Complete with checked physical home after every action |
| seed 303 | 5 / 10 | Knight settled at 3.043 mm XY error, above the 3 mm criterion |

**15/30 complete actions, 17 physical starts, 13 unattempted; 1/3 complete games.** The remaining issues and possible tolerance revisions are documented without rescoring the baseline. See [portability validation](artifacts/control-20261110/PORTABILITY.md) for the separate Windows and Ubuntu/WSL static loading checks.

## Included in this release

- [Portable control runtime and current robot/piece assets](simulation), pinned dependencies and three ten-action session inputs.
- [English control handoff](CONTROL_HANDOFF.md), including frames, source map, phase-by-phase behaviour and parameter limits.
- [Setup and gated launch instructions](simulation/README.md).
- [Current result evidence and portability checks](artifacts/control-20261110/README.md).
- [Current asset provenance](THIRD_PARTY.md).

Start with `simulation/README.md`. The root `requirements.txt` forwards to the same control dependencies. The handoff ZIP contains this current simulation package.

Control uses known simulator state and frozen move instructions. Camera-driven execution, a learned VLA policy and hardware validation are later stages.
