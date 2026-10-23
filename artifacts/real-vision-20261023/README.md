# Real-camera evidence package — 2026-10-23

Full report: [<kbd> English </kbd>](../../REAL_VISION.en.md) · [<kbd> 简体中文 </kbd>](../../REAL_VISION.md). Both versions cover methods, figures, interpretation and limitations.

| File | Meaning |
| --- | --- |
| `r1-phone-training.csv` | All 57 epochs of phone-photo fine-tuning; selected epoch 37 |
| `r2-dual-pilot-training.csv` | All 45 epochs of initial paired-camera fine-tuning; selected epoch 25 |
| `r3-groups28-training.csv` | All 66 epochs of 28-group fine-tuning; selected epoch 46 |
| `training-history.json` | Exact training arguments, initialization/selected checkpoint hashes, per-run held-out AP |
| `groups28-split.json` | 23/3/2 group split, 56 image identities, source/image/label hashes and ROI sizes |
| `groups28-evaluation.json` | Official AP, fixed-threshold diagnostics, class results and remaining main-camera errors |
| `checkpoint.json` | Latest candidate identity and offline-only deployment status; no binary weights |
| `live-acceptance.json` | Historical live class/grid acceptance with R1; host address omitted |
| `grid-registration-audit.json` | Two real-image grid cases and two image-translation checks; not absolute grid accuracy |
| `asset-provenance.json` | Copied image/CSV hashes and plot derivation |

The original CSVs and selected camera/experiment images are copied without modification. The new plots read these CSVs and use every epoch without smoothing. The original acquisition dataset, credentials, host configuration and model binaries are not included.

All three real-image stages use different test sets. Before/after figures are paired within each stage; they are not a common-test-set learning curve. R3's fixed-threshold 53/56 main-camera result is class-and-box correctness on reviewed board crops, not square assignment, full-board, or fusion accuracy. Labels and analysis were reviewed by the same GPT, not an independent second reviewer.
