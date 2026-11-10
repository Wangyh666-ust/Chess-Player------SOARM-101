# Portability validation — 10 November 2026

These are **static environment checks**, separate from the original 0/10, 10/10, 5/10 physical game results. No manipulation was rerun for this handoff.

| Check | Windows | Ubuntu / WSL2 |
| --- | --- | --- |
| Python | 3.10.20 | 3.12.3 |
| OS report | Windows build 26200 | Linux 6.18.33.2 Microsoft WSL2, glibc 2.39 |
| NumPy / SciPy | 1.26.4 / 1.14.1 | 1.26.4 / 1.14.1 |
| PyBullet | 3.2.7, existing Windows build | 3.2.7, freshly compiled Linux wheel |
| python-chess / chess | 1.999 / 1.11.2 | 1.999 / 1.11.2 |
| Portable source and asset manifest | 383 files | Same 383 files |
| Complete runtime entry import | Passed | Passed |
| Actual bodies / geometry types | 12 / 6 | 12 / 6 |
| Actual joint names | Five arm joints + `gripper` | Same |
| Constructor contact threshold | 0.002 forwarded | 0.002 forwarded |
| Timestep | 1/240 s | 1/240 s |
| Native step calls attempted / completed | 0 / 0 | 0 / 0 |
| Static 1200×800 preview | Written and visually inspected | Written and decoded |
| New manipulation / full-game tests | None | None |

Results: [Windows](windows-static-result.json), [Ubuntu/WSL](ubuntu-static-result.json). Previews: [Windows](windows-static-preview.png), [Ubuntu/WSL](ubuntu-static-preview.png). The plain board in this physics preview is untextured; it is an asset-load view, not a camera scene or recorded movement.

Windows used an independently materialized directory. Linux used the published setup flow from the public worktree: materialization → pin actual configuration/source bytes → gate start check → gated static inspection. The full three-game template was also pinned and passed its **start check only**. That command was not executed.

The first Windows package check stopped during import because `query_world_reuse_v2.py` dynamically compiles a predecessor file that a normal import inventory did not list. No worlds or physics started. The missing original file was added without editing the controller; the corrected package passed the no-world import check and the subsequent registered static check. The initial stop remains in the local gate record and handoff provenance.

Program checks cover portable JSON/URDF path handling, source-byte preservation, hash/overwrite rejection, actual checker namespace binding and restoration on constructor/close failure. They do not demonstrate contact stability. PyBullet 3.2.7 required compilation for the tested Linux Python 3.12 setup; pip installation took several minutes.

Source and API provenance: [handoff-provenance.json](handoff-provenance.json). The same GPT reviewed the evidence; `independent=false`. Linux dynamic reproducibility, other layouts and hardware remain unverified.
