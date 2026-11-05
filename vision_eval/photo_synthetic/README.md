# Synthetic chess dataset tools

Generate legal chess setups, render them from fixed and arm cameras with randomized
diffuse lighting, and check the images and labels before using them for training.
No input games or PGN files are required. Run every command below from the project root.

## Workflow

1. **Configure and generate setups.** Edit [configs/setups.json](configs/setups.json),
   then run `create_setups.py`. Stockfish plays legal games; the module samples
   positions and saves FENs, game histories and dataset splits. It does not load
   models or choose physical poses.
2. **Configure the dataset.** Edit [configs/dataset.json](configs/dataset.json), then
   give the saved setup JSON to `create_dataset.py`. It samples piece offsets and
   rotations, cameras, and lighting within the configured bounds, using the scene
   and assets in `configs/scene.json`.
3. **Render both views.** The runner launches Blender with `render_dataset.py`,
   which uses `reference_style.py` for materials, the table and lights. Each fixed/arm
   pair shares the same pieces, placement and lighting. Source models are unchanged.
4. **Validate automatically.** `check_dataset.py` checks labels, masks and paired
   views; `check_lighting.py` checks brightness and reflections. A passing run gets
   `data/dataset.yaml`. A failed run keeps diagnostics and is not published for training.

Keep the whole setup JSON so game histories, splits and provenance survive.
Piece placement, cameras and lighting are saved later in the dataset render plan.
Changing any physical randomization does not require playing the games again.

## Quick start

### 1. Check the local installation

A fresh clone does **not** include `.local-blender/` or generated datasets because
these paths are ignored by Git. Before running, you need:

- Python with `chess` (installed by `python-chess`), NumPy and Pillow in
  `.local-blender/venv/`.
- Blender and an executable `.local-blender/start-blender.sh` launcher.
- Stockfish at the `engine_path` in `configs/setups.json`.
- The approved files under `assets/chess/{board,pawn,knight,bishop,rook,queen,king}/`,
  especially each `model.blend` and `dimensions.json`. Obtain any missing assets
  from the project owner; do not regenerate the manually sculpted knight.

Follow [Local runtime installation](#local-runtime-installation) if this environment
is missing. Blender runs in the background; neither an open Blender window nor
Blender MCP is required for dataset creation.

```bash
.local-blender/venv/bin/python -c "import chess, numpy, PIL; print('Python dependencies ready')"
./.local-blender/start-blender.sh --version
```

### 2. Generate legal game setups

Adjust [configs/setups.json](configs/setups.json), then choose a fresh batch name:

```bash
.local-blender/venv/bin/python vision_eval/photo_synthetic/create_setups.py \
  --config vision_eval/photo_synthetic/configs/setups.json \
  --output datasets/synthetic/my_batch.config.json \
  --dataset-output datasets/synthetic/my_batch
```

This writes the setup JSON only; no images are rendered yet. Defaults request
24 positions from 12 games, giving **48 images** with the default one variant per
position and both camera views. The general count is
**source positions × `variants_per_setup` × 2 views**.
Piece placement is chosen in the next stage: the whole base stays inside its cell,
with all available space used by default, and ±180° rotation covers a full turn.

### 3. Check the rendering plan

Adjust [configs/dataset.json](configs/dataset.json) for piece placement, cameras, lighting and render quality:

```bash
.local-blender/venv/bin/python vision_eval/photo_synthetic/create_dataset.py \
  --setups datasets/synthetic/my_batch.config.json \
  --config vision_eval/photo_synthetic/configs/dataset.json \
  --dry-run
```

This validates the input and prints the position/image counts and output location.
It does not write files, start Blender or check rendered image quality.

### 4. Try two setups first

```bash
.local-blender/venv/bin/python vision_eval/photo_synthetic/create_dataset.py \
  --setups datasets/synthetic/my_batch.config.json \
  --output datasets/synthetic/my_batch_preview \
  --limit 2
```

With the default `variants_per_setup: 1`, this renders four images and runs both checks. Inspect
`datasets/synthetic/my_batch_preview/data/label-review.jpg` and `run.json` in the
preview directory. The preview uses the same default dataset settings as a full run.
For faster trials, copy `configs/dataset.json`, lower `samples` and/or
`resolution_scale`, and pass that copy with `--config`. Very small images can leave
too few measurable squares for the lighting checker.

For a 48-image trial with repeated positions, set `count: 6` in a copy of the
setup settings and `variants_per_setup: 4` in a copy of the dataset settings.
Generate the six legal positions, then render with that dataset settings file.
Each position produces four independently randomized physical scenes, each seen
by both cameras. All four variants keep the source position's train/val/test split.

### 5. Render the full batch

```bash
.local-blender/venv/bin/python vision_eval/photo_synthetic/create_dataset.py \
  --setups datasets/synthetic/my_batch.config.json \
  --config vision_eval/photo_synthetic/configs/dataset.json
```

The runner automatically renders, checks and records the outcome. You do not need
to launch the renderer or checkers separately. Output files and directories must be
new; use a different name for another run instead of overwriting previous results.

### 6. Inspect the result and reuse setups

- `datasets/synthetic/my_batch/run.json`: must say `"status": "passed"`.
- `datasets/synthetic/my_batch/data/dataset.yaml`: training entry point after success.
- `data/images/` and `data/labels/`: rendered RGB images and YOLO annotations.
- `data/label-review.jpg`, `data/qa.json`, `data/lighting_qa.json`: review and checks.
- `render_config.json`: exact FENs, piece poses, camera poses and lighting draws used for the run.
- `render.log`: Blender diagnostics, including failures.

To reuse the same chess setups with new physical variations, edit `configs/dataset.json`
(for example its seed, piece bounds, camera bounds or preset selection) and use a fresh output directory:

```bash
.local-blender/venv/bin/python vision_eval/photo_synthetic/create_dataset.py \
  --setups datasets/synthetic/my_batch.config.json \
  --output datasets/synthetic/my_batch_new_lighting
```

If a run fails, inspect `run.json`, `render.log` and any QA reports. Fix the cause
and rerun into a new directory. The runner does not automatically retry, resume,
discard failed pairs or relax thresholds. Passing these checks does not establish
that synthetic augmentation improves the real detector; that requires evaluation
on untouched real-camera images.

## What to edit

| Change you want | Where to edit | What to rerun |
| --- | --- | --- |
| Game count, sampled positions, phases, split weights or move variety | `configs/setups.json` | Generate a new setup JSON, then create its dataset. |
| Piece displacement or rotation bounds | `configs/dataset.json` → `piece_randomization` | Create a new dataset from the same game JSON. |
| Camera angle, distance, target, roll, focal length or framing | `configs/dataset.json` → `camera_randomization.fixed` / `.arm` | Create a new dataset from the same game JSON. |
| Lighting presets, lighting bounds, seed, resolution or render samples | `configs/dataset.json` | Run `create_dataset.py` with a fresh output directory. |
| Preset definitions or lighting QA thresholds | `configs/lighting_presets.json` | Create a new dataset; retain realistic thresholds when tuning. |
| Materials, table, markers or reference cameras | `configs/scene.json` (selected by `dataset.json` → `base_config`) | Create a new dataset; no game regeneration needed. |
| Piece or board geometry | `assets/chess/<piece>/model.blend` and matching dimensions | Create a new dataset; physical bounds are recomputed from the dimensions. |

`create_dataset.py` reads the scene config fresh for each run. Game files are
independent of scene assets. Editing a generated `.blend` scene does not update
the source models or JSON. Previously rendered runs retain their exact saved plan.

## Generate chess setups

No input games or PGN files are needed. `create_setups.py` starts from the normal
initial board and uses Stockfish for both sides. At each turn it analyses up to
three candidate moves and randomly chooses one within 80 centipawns (0.8 pawn)
of its best evaluated move. This produces varied engine games, not a calibrated
model of human playing strength. Search is limited to 20,000 nodes per turn;
increasing `engine_nodes` improves search quality at a speed cost. Rules, including
castling, promotion and en passant, are handled by
[python-chess](https://python-chess.readthedocs.io/en/stable/engine.html).

Edit `configs/setups.json` to change these controls:

| Setting | Meaning |
| --- | --- |
| `seed` | Controls move selection, position sampling and game splits only. |
| `games`, `max_plies` | Number of games and maximum half-moves per game. Games stop earlier on game-over, including claimable draws. |
| `count` | Number of unique setups to save. Fails with an explanation if too few were generated. |
| `min_ply`, `sample_every_plies` | Earliest sampled half-move and spacing between candidates. Use `min_ply: 0` to include the initial position as a candidate. |
| `positions_per_phase_per_game` | Cap samples from one game in each visual phase bucket. |
| `candidate_moves`, `max_loss_cp`, `engine_nodes` | Move variety, tolerated evaluation loss and search effort. More variety is not necessarily more human-like. |
| `split_weights` | Relative shares of sampled games assigned to train/val/test, rounded to whole games. Very small batches may lack a split; these are not exact image quotas. |
| `mode`, `engine_path` | Default `engine` uses the project-relative Stockfish executable. Explicit `random` uses uniform legal moves for stress testing; it is less realistic. |

Sampling favours a mixture of available phases: opening through ply 20,
endgame at 12 or fewer pieces, middlegame otherwise. These are approximate visual
buckets. Missing phases are not invented; inspect `setup_generation.phase_counts`
and `split_counts`. Exact repeated piece layouts are removed across all splits;
similar openings from different games can still occur. Keep real-camera evaluation
separate from synthetic validation.

Each output position has an ID, FEN, split, game ID, ply and phase. The
`setup_generation.games` list stores full UCI histories, game seeds and results;
replay the first `ply` moves to recover a sample. Games cut off at the ply limit
have result `*`. No dimensions, piece transforms, scene materials or camera poses
are generated by this module. Seeded random mode is repeatable; engine regeneration
also depends on its version/build. Saved FENs are the authoritative logical setups.

Run checks with `.local-blender/venv/bin/python vision_eval/photo_synthetic/tests/test_create_setups.py`.

## Create and check the dataset

`create_dataset.py` consumes the **whole setup JSON**, not just the FEN strings.
It preserves the FENs, game IDs and splits, verifies game histories when present,
and rejects duplicate source layouts. It expands each source into `variants_per_setup`
physical scenes, loads `base_config`, then samples piece poses,
fixed/arm camera poses and lighting. The exact draws are saved before Blender starts.
Legacy setup files with embedded scene settings or transforms remain readable;
those physical settings are replaced by the current dataset and scene configuration.

Inside Blender, `render_dataset.py` loads the assets, uses `reference_style.py`
for the table/materials/lights, applies each saved lighting draw, and renders the
fixed and arm cameras. Both views share pieces and lighting. Presets can have two
or three fixtures; unused lights are disabled when switching between them.
Materials come from the scene config. Reference camera locations/lenses anchor
the camera sampler. Renderer-level piece and camera jitter are disabled so the
saved plan is applied exactly once. Both views share the sampled piece placement.

Edit `configs/dataset.json`:

- `base_config`: scene JSON, relative to `dataset.json`; default `scene.json`.
- `variants_per_setup`: positive integer, default 1. Each source FEN produces this
  many physical scenes with independent piece, camera and lighting draws.
- `seed`: changes physical poses, cameras and lighting while preserving FENs and
  splits. Independent streams per position/view keep changes to one set of bounds
  from changing other draws; reordering positions does not change those draws.
- `piece_randomization`: displacement/rotation bounds described below.
- `camera_randomization`: separate fixed and arm camera bounds described below.
- `lighting_presets`: IDs from `lighting_presets.json` eligible for random selection.
- `lighting_bounds`: direction and power bounds; source distances and diffuser
  dimensions remain unchanged. The defaults are the bounds tested in the pilot.
- `samples`: Cycles samples per image (quality/runtime, not number of images).
- `resolution_scale`: multiplies both saved camera resolutions; default 1.
- `threads`: Blender CPU thread count.

`base_config` and `presets_file` are relative to `dataset.json`. The setup JSON's `output` becomes the
run directory, or pass `--output datasets/synthetic/a_new_run`. Existing directories
are refused. `--dry-run` validates and prints the image count without writing or
rendering; `--limit 2` limits source positions before expansion, so it produces
`2 × variants_per_setup × 2` images. Changed piece, camera or lighting bounds require only a new dataset run, not new
games. To reproduce an existing physical scene exactly, use that run's saved
`render_config.json` with the low-level renderer, changing only its output path.

### Piece placement bounds

Edit `piece_randomization` in `configs/dataset.json`:

```json
"piece_randomization": {
  "offset_mm": null,
  "cell_margin_mm": 0.0,
  "yaw_degrees": 180.0
}
```

| Setting | Meaning |
| --- | --- |
| `offset_mm` | Default `null` uses all available cell space. A number caps displacement per axis in mm, still limited by each piece's base size. |
| `cell_margin_mm` | Extra clearance between the base and cell edge; default 0. |
| `yaw_degrees` | Maximum deviation from the usual heading. Default 180 means ±180°, covering a full 360° turn; saved angles are normalized to [0, 360). |

Offsets are sampled independently along X and Y. Each piece's limit is
`(square_mm - base_diameter_mm) / 2 - cell_margin_mm`, optionally capped by
`offset_mm`. The saved circular base stays inside its square even at the corners
of that allowed region. Current limits with no extra margin are ±4.01 mm per axis
for the king and ±5.36 mm for the knight. This is a base-footprint constraint, not
a general sculpted-mesh collision solver; keep asset dimensions synchronized.

### Camera bounds and framing

Both `camera_randomization.fixed` and `.arm` contain these settings:

| Setting | Meaning and defaults |
| --- | --- |
| `enabled` | `true` samples a new camera; `false` preserves its reference pose, lens shifts and possible cropping. |
| `azimuth_offset_degrees` | Orbit left/right around the board, relative to the reference location; `[-10, 10]`. |
| `elevation_offset_degrees` | Change orbit elevation relative to the reference; `[-5, 5]`. Cameras remain above the board. |
| `distance_scale` | Multiply the reference distance from the board-volume centre; fixed `[1.4, 1.8]`, arm `[1.15, 1.45]`. |
| `target_offset_mm` | Three `[min, max]` intervals for X/Y/Z aim offset from the board-volume centre; default ±5/±5/±2 mm. |
| `roll_degrees` | Tilt the image in its plane; `[-3, 3]`. |
| `lens_scale` | Multiply the reference focal length; `[0.9, 1.0]`. Smaller values widen the field of view. |
| `margin_fraction` | Minimum framing margin on each image edge; `0.04` means 4% of width/height. |
| `max_attempts` | Maximum candidate poses to try per view; 100. Failure stops planning with a bounds error. |

The camera looks toward a box covering the full outer board and the tallest piece,
with 2 mm extra vertical clearance. Its reference location supplies the nominal
orbit; sampled cameras are re-aimed and their old photographic lens shifts are
reset to zero. The fixed-camera distance range is farther back because its old
reference crop did not contain that full volume. This avoids cropping edge pieces.

Candidates must keep every corner of the box inside the configured image margin.
This conditions the resulting distribution: requested ranges are limits, not a
promise that all combinations are accepted equally. Bounds are never widened
silently. If no candidate fits, widen the distance range, reduce focal length or
relax the framing margin explicitly. Equal interval endpoints lock a parameter.

Each position saves `piece_transforms`, `piece_seed`, `camera_setup` (including
sampled parameters and matrices), and `lighting_setup` in `render_config.json`.
Blender checks the framing volume again with its actual camera projection before
rendering; the dataset checker independently verifies the saved projections.
With `enabled: false`, that view retains the reference and skips the new full-volume
framing requirement. Lighting QA still runs for every view: valid framing does
not guarantee acceptable reflections at new angles.

A run contains:

```text
<run_name>/
├── settings.json          # Dataset settings, preset snapshot, setup source/hash
├── render_config.json     # Exact FENs, poses and lighting draws sent to Blender
├── render.log             # Blender output, useful if rendering fails
├── run.json               # rendering / checking / passed / failed
└── data/
    ├── config.json, manifest.json, scene.blend
    ├── images/{train,val,test}/
    ├── labels/{train,val,test}/
    ├── masks/{train,val,test}/
    ├── records/{train,val,test}/
    ├── qa.json, lighting_qa.json, label-review.jpg
    └── dataset.yaml       # Published only when both checks pass
```

`check_dataset.py` verifies expected view completeness, labels against masks,
paired geometry/lighting and placement split consistency. Its review sheet shows
at most 48 images, while all images are checked. `check_lighting.py` checks black
square washout, white brightness/evenness and border/table glare. It rejects views
with too few measurable empty squares rather than treating them as successful.
Very small previews or heavy occlusion can make that check inconclusive.

A failed run retains diagnostic files. If rendering finished but checks failed,
the YAML remains `dataset.unvalidated.yaml` instead of the training entry point.
Earlier render failures may not produce a YAML or QA reports. Inspect `run.json` and fix the cause before rerunning
into a new directory. This first runner fails the entire batch; it does not silently
drop failed pairs, retry with different lights, or relax quality thresholds.
A passing batch is structurally and visually checked, not proven to improve real
recognition accuracy. Testing with the real detector remains the next evaluation.

## Positions, cameras and exports

`create_setups.py` generates legal positions and stores them as FENs in `positions`. Uppercase symbols are white, lowercase black; digits are empty runs.
The parser validates placement and, by default, one king per colour. The partial
photo reconstruction sets `require_kings: false`; its diagnostic `test` split is
not a claim of legal positions or untouched model evaluation.

The dataset planner writes explicit `positions[].piece_transforms` and
`positions[].camera_setup` and sets legacy renderer jitter to zero. Direct-render
configs can still use `piece_jitter_mm`, `piece_yaw_jitter_degrees`,
`camera_position_jitter_mm` and `camera_target_jitter_mm`. In that legacy mode a
saved `matrix_world` overrides the target-based orientation. Do not mix those
legacy controls with the dataset planner's explicit poses.
Saved reference poses are approximate pinhole poses, without lens distortion. Robots, grippers,
sensor noise and the complete room are not modeled.

Each rendered dataset contains:

- `images/{split}/*.png`: RGB views.
- `labels/{split}/*.txt`: normalized YOLO class/centre/size boxes covering the full
  projected piece, including occluded geometry, when sufficiently visible.
- `masks/{split}/*.png`: visible instance IDs (zero is background).
- `records/{split}/*.json`: FEN, classes/squares, full/visible boxes, visibility,
  transforms, camera matrices, board corners and lighting/material metadata.
  `source_setup_id` identifies the original legal position; `variant_index` is its
  zero-based physical variant. `qa.json` distinguishes source positions from
  paired physical scenes.
- `config.json`, `manifest.json`, `dataset.yaml`: exact settings, asset hashes,
  sample listing, class definitions and dataset location.
- `scene.blend`: editable first scene, except bound probes omit duplicate scenes.
- `qa.json`, `label-review.jpg`, `lighting_qa.json`: validation outputs where run.

Both views and all variants of an arrangement must remain in the same split;
`create_setups.py` also groups positions from the same source game. Real
held-out images will be needed to measure the benefit of synthetic augmentation.
## Models and dimensions

Editable assets remain in `assets/chess/{board,pawn,knight,bishop,rook,queen,king}/`.
Each has `model.blend`, `dimensions.json`, `create.py`, `README.md`, `preview.png`
and `validation.json`. The renderer loads the evaluated saved geometry.
**Do not rerun the knight's `create.py`: it would replace the approved manual sculpt.**

- Outer board: 248 × 245.8 mm; black frame: 7.5 mm.
- Square pitch: 25.223229 mm; grid: 201.785833 mm square.
- Grey surround: nominal area increased by 25%, before lettering/slit cutouts.
- Heights: pawn 23, knight approximately 27.495, bishop 32, rook 25.5, queen 35,
  king 45 mm. Rook top diameter: 11.52 mm.

Board width/depth are measured; pitch, thickness and piece widths remain estimates.
The renderer reads saved Blender board controls. Keep the board's
`dimensions.json` synchronized with those controls. Labels and marker centres follow the
board's dimension drivers. Source assets were not regenerated during reorganization.

## Appearance and lighting

The reference uses Standard display response, None look and exposure 1; the older
environment comparison uses AgX. White pieces are cream and black pieces/squares
near-black. Shared material finishes and camera response matter: reusing lights
with different finishes can restore glare.

| Preset | Ceiling / window emitted power |
| --- | --- |
| `01_reference_room` | Strips 40/20 W; diffuse window 6 W |
| `02_overhead_strips` | Strips 40/20 W; no window |
| `03_overhead_panels` | Panels 40/20 W; no window |
| `04_strips_daylight` | Strips 40/20 W; window 10 W |
| `05_panels_daylight` | Panels 40/20 W; window 6 W |
| `06_dim_room` | Strips 36/18 W; window 5.4 W; 90% room fill |
| `07_bright_room` | Strips 44/22 W; window 6.6 W; 110% room fill |

These are Blender emitted watts, not electrical fixture ratings. Ceiling sources
are 3.2 m high, offset toward +Y; strips measure 1.2 × 0.3 m and panels 0.8 × 0.8 m.
The vertical 2 × 1.5 m window is 1.5 m away in plan (centre distance 1.76 m),
representing diffuse sky/bounce only. There is no direct sun or point spotlight.
World strength 0.5 and colour [0.53, 0.88, 1] approximate broad room illumination
and the cool camera response. These are visual estimates, not reconstructed lux.

- `lights[]`: `location_m`, `target_m`, `power_w`, `shape`, `size_m`, `size_y_m`,
  `color`. Source size relative to distance affects softness; direction affects
  reflection placement. `rig_rotation_degrees` preserves diffuser axes when sampling.
- `world_strength`, `world_color`, `exposure`, `view_transform`, `view_look`:
  room fill and camera response.
- `surface_finishes`: board, frame, white/black squares, pieces and colour-specific
  piece overrides. `base_color` uses linear RGB; `roughness`, `coat`,
  `coat_roughness` and `specular` control reflections. Specific overrides take
  precedence over common ones.
- `wood`: procedural colour/grain, warp, roughness, coat and specular settings.
- `aruco_enabled`: hide stickers when false; otherwise respect saved visibility.

Editing a generated Blender scene does not rewrite JSON; changing JSON does not
update an already-open scene. Save model edits in the source assets and edit the
appropriate config to retain scene settings in future renders.

### Bounded randomization

For the normal workflow, edit `lighting_bounds` in `configs/dataset.json`.
The planner saves explicit draws in `positions[].lighting_setup`; Blender does not
sample them again. For direct renderer use without `create_dataset.py`, add this
to a renderer config, or use an empty object for current defaults:

```json
"lighting_randomization": {
  "azimuth_degrees": [-30, 90],
  "power_scale": [0.8, 1.0],
  "per_light_power_scale": [0.65, 1.0],
  "world_scale": [1.0, 1.0]
}
```

Omit the field for fixed explicit lights. Equal endpoints lock a parameter.
Invalid, reversed, non-finite and unknown bounds raise `ValueError`.
All sources/targets rotate together around the board's Z axis. Heights, distances,
diffuser shapes/sizes and relative orientations remain fixed. Global power scales
fixtures and room fill; per-light power changes their individual balance; world
scale is an additional fill multiplier. All draws are uniform within bounds.

For direct use, put `vision_eval/photo_synthetic` on Python's import path:

```python
import json
from pathlib import Path
from randomized_lighting import sample_lighting

presets = json.loads(Path("vision_eval/photo_synthetic/configs/lighting_presets.json").read_text())
preset = presets["variants"][0]["overrides"]
setup = sample_lighting(preset["lights"], preset["world_strength"], seed=1234)
```

In that direct-renderer mode, the renderer samples once per position using `seed + position_index * 100003`,
independently of piece/camera RNG draws. Both views share the sample, recorded
under `sampled_lighting` with exact specs, seed and bounds. Reordering positions
changes seeds. The controlled lighting-preset pilot deliberately disables this.

## Quality checks

`configs/lighting_presets.json` contains the unchanged visual limits: black-square
p95 ≤65/255, white-square median p90 210–254.95, white spread ≤40, frame p95 ≤100,
grey surround p95 ≤205, tabletop p99 ≤215. The reference clips some white colour
channels; these are displayed-image heuristics, not lux measurements.

The checker projects inset empty-square regions and excludes occluding pieces
plus a two-pixel antialiasing halo. It requires four visible empty squares of each
colour. Frame checks assume the current 7.5 mm frame. Invisible regions are null;
the lighting pilot retains a wider-camera reflection diagnostic. Keep these checks
when changing cameras/materials. A preset does not guarantee glare-free renders
at every viewpoint. `check_lighting.py OUTPUT --self-test` exercises rejection.

## File structure

```text
photo_synthetic/
├── README.md
├── create_setups.py
├── create_dataset.py
├── render_dataset.py
├── reference_style.py
├── randomized_lighting.py
├── randomized_cameras.py
├── check_dataset.py
├── check_lighting.py
├── configs/
│   ├── scene.json
│   ├── setups.json
│   ├── dataset.json
│   └── lighting_presets.json
├── tests/
│   ├── test_create_setups.py
│   ├── test_create_dataset.py
│   └── test_randomized_lighting.py
├── pilots/
│   ├── environment/       # config.json, run.py, results/
│   ├── lighting/          # config.json, run.py, results/
│   └── lighting_bounds/   # config.json, run.py, results/
└── references/
```

Generated batches belong in `datasets/synthetic/<run_name>/` at the project root.
Game generation and dataset rendering are separate commands: saved setups can be
reused with new lighting settings without replaying Stockfish.

## Core files

| File | Use |
| --- | --- |
| `create_setups.py` | Play legal games and save unique FENs, histories and splits without scene dependencies. Runs outside Blender. |
| `create_dataset.py` | Sample piece poses, cameras and lighting, run Blender, then run both checks. Publish a training YAML only after success. |
| `configs/dataset.json` | Piece/camera/lighting bounds, scene config, render quality and CPU threads. |
| `configs/setups.json` | Game generation, logical position sampling and split weights. |
| `render_dataset.py` | Load approved assets, place pieces, render paired views, export RGB, masks, YOLO labels and metadata. Runs inside Blender. |
| `reference_style.py` | Apply procedural wood, materials, lights and background props. Runs inside Blender. |
| `randomized_lighting.py` | Sample bounded lighting with a seed; pure Python, called by the dataset planner or directly by the renderer. |
| `randomized_cameras.py` | Sample bounded board-facing cameras and reject cropped framing volumes without Blender. |
| `check_dataset.py` | Check annotations, masks, view pairing and split consistency; save `qa.json` and `label-review.jpg`. |
| `check_lighting.py` | Reject washed-out black squares, clipping, uneven lighting and excessive frame/table glare. |
| `configs/scene.json` | Assets, materials, cameras and current reference arrangement, plus renderer settings. Default output: `datasets/synthetic/reference_preview`. This is a 24-piece partial photo reconstruction, not a complete legal game. |
| `configs/lighting_presets.json` | Seven reusable lighting definitions and visual-quality limits. |
| `tests/test_randomized_lighting.py` | Test seeded sampling, input bounds, preserved geometry and Blender diffuser axes. |
| `tests/test_create_dataset.py` | Check independent physical draws, camera framing/bounds, game preservation and failed-run handling. |
| `tests/test_create_setups.py` | Replay generated games, check FENs, reproducibility and Stockfish integration. |

Config `assets.<kind>.file` and `output` paths are relative to the **project root**.
Pilot `base_config` and `presets_file` paths are relative to that pilot's config
file. Generated configs retain exact settings and relocated output paths.
Historical source hashes inside results describe the code used for those renders;
they are not rewritten to pretend the current reorganized code produced them.

## Local runtime installation

These instructions describe the Linux layout expected by the runners. All runtime
files stay under the ignored `.local-blender/` directory. This project was tested
with Blender 4.5.15 LTS, Stockfish 19, and Python 3.14 with python-chess 1.999
(chess 1.11.2), NumPy 2.5.4 and Pillow 13.0.0. The installed environment's exact
packages are recorded locally in `.local-blender/requirements.lock.txt`.

1. Create an isolated Python environment and install the generator/checker dependencies:

   ```bash
   python3 -m venv .local-blender/venv
   .local-blender/venv/bin/python -m pip install python-chess==1.999 numpy Pillow
   ```

2. Download the matching Linux Blender archive from the
   [official Blender 4.5 releases](https://download.blender.org/release/Blender4.5/)
   and extract it inside `.local-blender/`. The launcher below expects
   `.local-blender/blender-4.5.15-linux-x64/blender`; adjust its final line if your
   extracted directory differs. Keep the approved model files separate from this runtime.

3. Download Stockfish for your architecture from the
   [official Stockfish distribution](https://stockfishchess.org/download/) and
   extract it under `.local-blender/stockfish/`. Keep its license and source.
   Set `engine_path` in `configs/setups.json` to the actual executable, relative to
   the project root. The current value targets `stockfish-linux-x86-64-universal`.

4. If the launcher is missing, save this as `.local-blender/start-blender.sh`:

   ```bash
   #!/usr/bin/env bash
   set -euo pipefail
   BLENDER_ENV_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
   export BLENDER_USER_CONFIG="$BLENDER_ENV_DIR/profile/config"
   export BLENDER_USER_SCRIPTS="$BLENDER_ENV_DIR/profile/scripts"
   export BLENDER_USER_DATAFILES="$BLENDER_ENV_DIR/profile/datafiles"
   export BLENDER_USER_EXTENSIONS="$BLENDER_ENV_DIR/profile/extensions"
   export XDG_CACHE_HOME="$BLENDER_ENV_DIR/cache"
   export TMPDIR="$BLENDER_ENV_DIR/tmp"
   export DISABLE_TELEMETRY=true
   mkdir -p "$BLENDER_USER_CONFIG" "$BLENDER_USER_SCRIPTS" \
     "$BLENDER_USER_DATAFILES" "$BLENDER_USER_EXTENSIONS" "$XDG_CACHE_HOME" "$TMPDIR"
   exec "$BLENDER_ENV_DIR/blender-4.5.15-linux-x64/blender" "$@"
   ```

   Make it executable:

   ```bash
   chmod +x .local-blender/start-blender.sh
   ```

The dataset scripts do not require the Blender MCP addon. Remove
`.local-blender/stockfish/` to remove the engine, or `.local-blender/` to remove the
whole local runtime. Save models and datasets outside that directory before cleanup.

## Regression checks

Run the Python suite with the isolated interpreter:

```bash
.local-blender/venv/bin/python -m unittest discover \
  -s vision_eval/photo_synthetic/tests -p 'test_*.py'
```

The Stockfish integration check runs when the configured engine exists. The
Blender-specific diffuser-axis test is skipped by ordinary Python; run it in Blender:

```bash
./.local-blender/start-blender.sh --background --threads 2 --python-exit-code 1 \
  --python vision_eval/photo_synthetic/tests/test_randomized_lighting.py
```

## Pilots and references

| Folder | Purpose and entry points |
| --- | --- |
| `pilots/environment/` | Reconstruct the 18-piece arrangement in `references/environment.jpg`. `run.py` assembles the photo/render comparison from existing renders without changing its config. See `results/comparison.jpg` and `results/scene.blend`. |
| `pilots/lighting/` | Compare seven diffuse presets with fixed and arm cameras. `run.py` renders and validates them. See `results/reference_comparison.jpg`, `results/comparison.html`, and each variant's scene/QA files. |
| `pilots/lighting_bounds/` | Replay accepted/rejected bound probes. `run.py` checks completed matching cases or renders missing cases. See `results/selected_comparison.jpg`, `results/summary.json`, `results/verification.json`. No detector is run yet. |
| `references/` | Original photographs, published annotated examples, provenance and fitted camera information. Keep these as inputs. |

Each pilot's `config.json` contains its experiment settings. Results are
regenerable and excluded from Git, so result links below may be absent in a fresh
clone. Pilots are diagnostic experiments; they are not required to generate a dataset.

Reference provenance is recorded in `references/source.json` and historical camera fits
in `references/camera-fit.json` and `references/fixed-camera-match.json`.
The camera-fitting utility has been removed. Existing camera poses and piece
placements remain saved in the configs. Camera randomization uses those reference
locations as starting points without fitting new photos. The environment runner only
creates comparison images and checks their alignment; it does not render or refit.
The stored Camera 2 comparison was based on an annotated group16 image that was
already absent from `references/` when this reorganization began. The active
lighting config therefore renders preset comparisons only; its saved historical
photo comparison is retained. An optional `reference_comparison` can be added
again when its source photo is available. Group16 influenced appearance tuning
and is not an untouched recognition test. The environment pilot retains its
earlier photo appearance; the lighting pilot has the later Camera 2 finishes
and colour response. Both use the same approved asset geometry.

### Pilot and direct-render commands

Run from the project root. Copy a config and change its output before rendering
if the output already exists; the renderer refuses to overwrite a dataset.

```bash
# Reference preview, or pass --config with your own renderer configuration.
./.local-blender/start-blender.sh --background --threads 4 \
  --python vision_eval/photo_synthetic/render_dataset.py -- \
  --config vision_eval/photo_synthetic/configs/scene.json
.local-blender/venv/bin/python vision_eval/photo_synthetic/check_dataset.py datasets/synthetic/reference_preview
.local-blender/venv/bin/python vision_eval/photo_synthetic/check_lighting.py datasets/synthetic/reference_preview

# Environment: render with the saved camera and placements, then compare.
./.local-blender/start-blender.sh --background --threads 4 \
  --python vision_eval/photo_synthetic/render_dataset.py -- \
  --config vision_eval/photo_synthetic/pilots/environment/config.json
.local-blender/venv/bin/python vision_eval/photo_synthetic/pilots/environment/run.py

# Preset comparison. Existing results are protected; --output chooses a fresh run.
.local-blender/venv/bin/python vision_eval/photo_synthetic/pilots/lighting/run.py --threads 4

# Bound sweep: existing completed matching cases are reused.
.local-blender/venv/bin/python vision_eval/photo_synthetic/pilots/lighting_bounds/run.py

```

The lighting runner accepts `--base-config` for custom positions/cameras and
`--output` for a fresh destination. Its pilot config sets samples, resolution,
shared presets and optional real-photo comparison. It disables placement, camera
and lighting jitter for controlled comparisons. It renders each preset as a
separate YOLO dataset; it does not merge them into a training dataset.
The bound runner accepts `--config` for a different sweep. Use a new output after
changing code/assets, because matching completed configs are reused.
Both runners retain image-quality/label checks. Failed visual probes and failed
dataset runs remain available for diagnosis; they must not become training samples.

### Why these lighting bounds

The visual sweep tested **44 setups / 88 renders**: 31 setups passed the existing
image-quality checks and 13 were rejected. All 20 targeted checks of the selected
range passed: dim/bright endpoints for seven presets, two reflection-edge checks,
and four seeded mixed samples. The current defaults are −30° to +90° azimuth,
0.80–1.00 overall power, 0.65–1.00 individual light power and 1.00 additional room fill.

| Probe | Observed result |
| --- | --- |
| 40–50% reference power | Rejected: white-square brightness about 177–195/255, too dim for this room reference. |
| 60% reference power alone | Passed, but combining it with weaker fixtures failed; unsuitable as a shared lower bound across presets. |
| 120–140% reference power | Rejected: white-square brightness clipped to 255. |
| Double fixture power | Rejected: clipped whites and an overly bright grey surround. |
| −90° rotation | Rejected: fixed-view black-square p95 reached 83/255. |
| +120° to +180° rotation | Rejected: excessive frame/square reflections in the arm view. |
| Selected combined bounds | Passed the targeted endpoint and mixed-sample checks in both views. |

Overall power scales both fixtures and room fill; independent multipliers then
change the balance of individual lights. Room fill therefore varies even though
its additional multiplier is fixed at 1. The bright preset already provides
higher illumination. The asymmetric angle range avoids the two cameras' different
reflection directions, while source distances, heights and diffuser geometry stay fixed.

The accepted dim-room endpoint reduces black-piece p95 brightness in the fixed
view from **52.1 to 39.7/255**, about 24% lower. This weakens highlights and shape
cues without making the scene unusably dark. It is a potential detection challenge,
not a measured accuracy drop.

Evidence remains in [the selected comparisons](pilots/lighting_bounds/results/selected_comparison.jpg),
[the full metrics](pilots/lighting_bounds/results/summary.json), and
[verification counts and source hashes](pilots/lighting_bounds/results/verification.json).
The sweep used one 24-piece arrangement, two camera poses, 24 Cycles samples,
denoising, and images of 559×428 (fixed) and 640×360 (arm). Geometry, exposure
and materials were held constant. These are image-space checks guided by the
reference, not measured room illuminance or an exhaustive guarantee for new scenes.
Rejected probes are diagnostics, not training samples.

### Next step: test the bounds with the recognition software

1. Obtain the actual deployed or fine-tuned recognition checkpoint. Record its
   hash, class mapping, image crop/resize procedure, confidence threshold and NMS
   settings so the comparison uses the same pipeline as the real cameras.
2. Render matched baseline/lighting variants across several arrangements, game
   phases and seeds, including crowded boards and black pieces on black squares.
   Keep placement and cameras identical within each comparison, and use the same
   sampled setup for both camera views.
3. Compare predictions with synthetic ground truth. Measure missed pieces, false
   detections, colour/type errors and, if testing the full recognition pipeline,
   square-assignment errors. Report fixed and arm results separately, plus fused
   results if fusion is implemented. Define visible versus full projected boxes
   consistently before calculating overlap-based scores.
4. Expand or shrink one bound at a time, then test combinations. Keep the current
   visual-quality checks: an image should not enter training merely because it
   breaks the detector. Prefer plausible, informative failures and retain ordinary
   reference-like samples rather than training only on difficult extremes.
5. Freeze selected bounds using development scenes. Compare real-only training
   against real-plus-synthetic training on a separate, untouched real-image test
   set. Group related positions/views/lighting variants into the same split.
   The reference images used for appearance tuning are not an untouched test set.

Recognition-based scoring and automatic rejection/resampling are future work;
`pilots/lighting_bounds/run.py` currently measures visual quality only.
