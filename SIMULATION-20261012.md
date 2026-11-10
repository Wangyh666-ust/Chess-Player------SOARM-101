# Preliminary SO-ARM101 virtual manipulation

Experiment: `ideal_grasp_continuous_3x10_v1`, 2026-10-12.

The project now extends beyond recognition: a PyBullet prototype executes consecutive chess-piece pickup, transfer, placement, release, and checked return-home actions. This is an initial motion-workflow experiment, separate from the synthetic vision benchmark.

## Video

[Watch in the online demo](https://wangyh666-ust.github.io/Chess-Player------SOARM-101/#simulation) · [Download MP4](docs/assets/simulation-random2-10plies.mp4)

[![Recorded simulation](docs/assets/simulation-poster.jpg)](https://wangyh666-ust.github.io/Chess-Player------SOARM-101/#simulation)

The 89.7-second clip shows `random2`, the first fully completed game in the original ordering. It includes all ten action segments and all ten return-home segments. The source run spans 432.9 simulated seconds including initialization. Display uses recorded poses, approximately one-second samples plus phase boundaries, at 10 fps, so playback speed is variable. Exporting the video performed **zero physics steps**.

Ten plies are ten halfmoves, not ten complete white/black move pairs. The displayed sequence is:

`e5e3, c6d6, e3e4, d6c5, f6f5, d3g3, e4f4, g3e3, f4h4, e3g5`.

## Method and assumptions

- Nominal SO-ARM101 same-table installation, 30 mm board squares, three fixed sparse king/queen endings with legal noncapture moves.
- One persistent physical scene per game; no resets of piece state between actions. Each completed action includes settled release and checked physical home before the next ply.
- Known simulator state, numeric IK, collision-checked paths and OMPL. Queen grasp preparation includes current-scene centering.
- **Ideal grasp B:** real bilateral closure precedes a measured `JOINT_FIXED` attachment. Both relative translation and rotation are constrained during lift, transfer and descent. The attachment is removed before intentional opening and on cleanup. Collisions and actual released board support remain enabled.
- Native contact-force values are recorded without replacement. This is not a constant-pressure or natural-friction grasp controller. The ideal constraint reaction capacity is distinct from the nominal motor torque limit.
- A failed task stops its game; the next independently initialized game can continue. Program/evidence errors stop the batch. No automatic retries.

## Whole-batch results

| Game | Completed / planned | Outcome |
| --- | ---: | --- |
| random1 | 6 / 10 | Ply 7, queen d4→a1, blocked by candidate/placement-IK preflight |
| random2 | 10 / 10 | Complete; featured video |
| random3 | 10 / 10 | Complete |

**26/30 planned plies completed; 2/3 complete games.** There were 27 workflow attempts: 26 positive-physics actions and one zero-physics planning block. Three subsequent plies were unattempted. All 26 positive-physics actions completed actual release, settled support, and checked home; no drops or unintended collisions were observed in reviewed records/guards.

| Audited quantity | Result |
| --- | ---: |
| Maximum final XY error | 0.808 mm |
| Minimum recorded continuous post-release support before retreat | 0.600 s |
| Maximum fixed-constraint position residual | 0.0578 mm |
| Maximum fixed-constraint rotation residual | 0.2508° |
| Action / home / initialization physics steps | 247,897 / 27,854 / 360 |
| Total physics steps | 276,111 |
| Process wall time | 1,632.781 s |

The blocked d4→a1 action produced 32 candidate records: 28 `fit` rejections and four `destination_place_ik` rejections. No physical action or native transfer-OMPL call occurred for that ply. This does not establish that a1 is globally unreachable.

Native jaw-force-zero samples still occurred in 14/26 completed actions: 12,508 of 143,155 attachment samples had at least one zero-force side. Ideal attachment bypassed their effect on carrying; it did not repair native contact dynamics.

## Evidence and limits

[Published evidence package](artifacts/simulation-v1/README.md) includes the whole-batch raw audit, video provenance, decode checks, and the compact recorded poses used for the clip. The audit checked 27 action traces and 26 home traces. Every video frame decoded successfully, and all 877 selected poses were matched to their source indices/hashes. Full high-frequency raw traces, meshes, and the local simulation runtime are not bundled; this update is a browsable results/demo package, not a standalone simulator distribution.

The same GPT separately reviewed the raw evidence (`independent=false`); no independent second reviewer was used. Three fixed games are not a general reliability estimate. Full contact vectors are regular/event samples; runtime collision checks ran at each physics tick. Translation **and** rotation are idealized while grasped. These results do not establish hardware capability, natural friction reliability, live vision-driven control, or a trained VLA action policy.

Next work is to resolve the d4→a1 candidate/placement-IK block and connect the separately evaluated vision and manipulation components.
