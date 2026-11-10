# Start here: simulation control

This package is for running the virtual arm before working on hardware. Read [CONTROL_HANDOFF.md](../CONTROL_HANDOFF.md) for the workflow, current results, models and parameter limitations.

## 1. Clone and install

Commands below start at the repository root. Use a new 64-bit Python virtual environment. Python 3.10.20 is the recorded Windows physics baseline. The separate [portability record](../artifacts/control-20261110/PORTABILITY.md) states which Linux setup was actually checked.

If you downloaded the handoff ZIP, extract it, enter its top-level folder and skip the two Git commands below. The remaining setup commands work from that folder.

```bash
git clone https://github.com/Wangyh666-ust/Chess-Player------SOARM-101.git
cd Chess-Player------SOARM-101
```

Ubuntu/Linux:

```bash
python3 -m venv .venv-control
source .venv-control/bin/activate
python -m pip install -r simulation/requirements.txt
mkdir -p runs
```

If the distribution does not provide `venv`, install its Python venv package first. If pip must compile PyBullet, install the distribution's C/C++ build tools and Python development headers. A source build can take several minutes. GPU/CUDA and ROS are not required.

Windows PowerShell, with Python 3.10 available:

```powershell
py -3.10 -m venv .venv-control
.\.venv-control\Scripts\Activate.ps1
python -m pip install -r simulation/requirements.txt
New-Item -ItemType Directory -Force runs | Out-Null
```

You can invoke `.venv-control/Scripts/python.exe` explicitly if PowerShell activation is unavailable. Do not copy an existing Windows virtual environment onto Linux.

## 2. Materialize a portable workspace

```bash
python simulation/tools/portable_snapshot.py prepare --bundle simulation --out runs/control-workspace
```

The destination must be new. This checks all bundle hashes, copies the runtime and expands JSON path tokens to your absolute workspace path. URDF meshes already use relative paths. Keep `simulation/runtime` and `simulation/manifest.json` as the immutable export. Edit a separately versioned workspace when developing.

The result contains `materialization.json`, dated runtime source directories, current assets and all three ten-action input sessions. The original directory structure is required by source-relative imports.

## 3. Check imports and static asset loading

```bash
python simulation/tools/pin_record.py --root . --template simulation/records/environment-smoke.template.json --out runs/environment-smoke.json
python simulation/runtime/experiment_review/experiment_gate.py --root . --ledger simulation/review/ledger check --record runs/environment-smoke.json --phase start
python simulation/runtime/experiment_review/experiment_gate.py --root . --ledger simulation/review/ledger run --record runs/environment-smoke.json
```

Expected: `runs/environment-smoke/result.json` reports 12 pieces, six geometry types, five arm joint names plus `gripper`, `dt_s=1/240`, threshold 0.002, zero attempted `stepSimulation` calls and zero physics steps. Open `runs/environment-smoke/preview.png` to inspect the **static** scene. This proves environment/asset loading only.

The gate's process result deliberately leaves scientific outcome unknown until raw review. Inspect the checker result and image. Do not infer a manipulation pass from `ok: true` in a process wrapper.

## 4. Reproduce the three-game control baseline

The known physical results are **0/10, 10/10, 5/10**. Expect the documented exit/XY blocks to remain until separately corrected; this package does not claim all games pass. The v2 entrypoint has a classification-only fix, not another recorded full-batch physical pass.

The supplied reproduction template fixes the original scenes, moves, physics and budgets. It pins every declared materialized JSON input and runtime Python file. Review its criteria before launch. If you change a source or configuration, prepare a new reviewed version rather than bypassing a hash mismatch.

```bash
python simulation/tools/pin_record.py --root . --template simulation/records/three-games.template.json --out runs/three-games.json
python simulation/runtime/experiment_review/experiment_gate.py --root . --ledger simulation/review/ledger check --record runs/three-games.json --phase start
python simulation/runtime/experiment_review/experiment_gate.py --root . --ledger simulation/review/ledger run --record runs/three-games.json
```

Allow up to 2,700 seconds for the batch. The scene is headless; detailed files appear under `runs/three-games/`. Each game retains its physical state between actions. Task failures stop that game; program/evidence faults stop the batch. There are no automatic retries. Keep the output and gate logs even when a task stops early.

The setup/portability work for this handoff does **not** execute this three-game command. It is provided for the receiving developer's subsequent reproduction.

## 5. Troubleshooting

| Symptom | Check |
| --- | --- |
| `@WORKSPACE@` in a missing filename | Materialize first; do not launch runtime templates directly. |
| Missing URDF/OBJ or a bundle hash mismatch | Complete clone/download; inspect `simulation/manifest.json`. Do not replace current models with the old Dubrovnik set. |
| Missing `query_world_reuse_v1.py` | The pinned v2 adapter dynamically compiles its predecessor. Keep the entire exported source closure. |
| `ModuleNotFoundError` | Activate the control environment; install `simulation/requirements.txt`; keep the original directory layout. |
| Output already exists | Preserve it. Prepare a new versioned record and new output directory. |
| Immediate queen `empty_return_no_plan` | Known post-release planning block; read the empty-exit section of the handoff. |
| Knight final XY slightly above 3 mm | Known task tolerance failure; investigate release/centring and predeclare any tolerance revision. |
| Planning timeouts on a slower computer | Resource block; report actual wall budget separately from physics/task outcome. |
| Threshold log shows 0.0002 requested by legacy code | Check forwarded **requested** value: the wrapper must forward 0.002. Never launch legacy PhysicsWorld alone for a new trial. |

The source tree still imports historical modules; inclusion does not make those older controllers the active default. The three-game entrypoint and actual runner bindings define the active flow.
