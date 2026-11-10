# Current control evidence — 10 November 2026

Start with [the English handoff](../../CONTROL_HANDOFF.md) and [setup guide](../../simulation/README.md).

The recorded original three-game outcomes are **0/10, 10/10, 5/10** complete actions, **17 physical starts**, **15 complete actions**, **13 unattempted**, and **1/3 full games**. Recorded physics: 139,663 ticks at 240 Hz. This handoff adds environment verification; it does not rerun or rescore those games.

| Artifact | What it contains |
| --- | --- |
| [Combined raw audit](combined-raw-audit.json) | Byte-exact original same-GPT audit: game counts, persistence, timebase, support, force and motion measurements |
| [Initial stop audit](initial-stop-audit.json) | Queen f7→d7 placed/settled; exit plan rejected at its starting pose; original program-error label retained alongside derived interpretation |
| [Remaining actions audit](remaining-actions-audit.json) | Original second/third-game action-level audit; filename was a progress filename, content is the final retained audit |
| [Knight final XY measurements](knight-final-xy.json) | Original pre-open, open and settled positions; descriptive, no isolated cause claimed |
| [Selected evidence](selected-evidence) | Original small results, selected raw frames, source/ledger snapshots, timebase QA and report gate results |
| [Portability validation](PORTABILITY.md) | Current packaging/program tests and zero-step Windows/Linux environment checks |

Original registered identities: `staged_continuous_3x10_20261110_v1`, `staged_remaining_2x10_20261110_v1`, and the aggregate-only `staged_continuous_3x10_assessment_20261110_v1`. The execution profile names the current classification-only v2 entrypoint separately from the physically executed legs.

Original evidence is preserved, including condensed English and Chinese historical notes. This folder is a public subset: original path references identify the maintainer's source workspace; not every high-frequency recording referenced by those audits is bundled. Do not claim full raw-trace reproducibility from this subset alone. Runtime reconstruction is supplied separately under [simulation](../../simulation).

Review was performed by the same GPT in a separate raw-evidence pass; `independent=false`. No new subagents or manipulation trials were used for this handoff. Implementation API usage for the portable helpers is recorded separately in [handoff provenance](handoff-provenance.json).
