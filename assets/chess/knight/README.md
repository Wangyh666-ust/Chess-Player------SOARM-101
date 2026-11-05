# Editable knight model

The final manually refined knight is in `model.blend`, scene
**Knight — photo modeling**, object **Knight — sculpt working copy**.
The dataset's `configs/scene.json` selects this object for both piece colours.

The sculpt has rounded mane scallops, an inset eye and eyelid, a mouth crease,
soft jaw relief, a broad central body recess and two rounded pedestal tiers.
Its height is approximately 27.5 mm and its base diameter is 14.6 mm. Widths,
depths and small details remain photographic estimates rather than scan results.

**Do not run `create.py` over the final sculpt.** The generator rebuilds the
parameter-driven version from `dimensions.json`; it does not reproduce the final
manual edits. Save a separate copy before rebuilding.

## Contents

- `model.blend`: final sculpt, aligned camera, orthographic view cameras and
  alignment guides. The separate **Knight — black and white photo reconstruction**
  scene retains the parameter-controlled white and black variants.
- `create.py`: procedural reconstruction script, run with Blender.
- `dimensions.json`: reconstruction parameters; these do not control the final sculpt.
- `preview.png`: historical comparison of the photograph, detailed sculpt and an
  oblique view. The photograph in this preview is not packed into `model.blend`.
- `validation.json`: recorded geometry checks and historical refinement measurements.
- `README.md`: editing and asset usage instructions.

The parameter-controlled variants expose height, base diameter, body width and
body depth as Custom Properties. Detailed profiles are in `dimensions.json`.
The final sculpt has baked geometry and is edited directly in Sculpt or Edit Mode.
The embedded **Molded details.json** text records detail depths; it is not a live
geometry control.

## Editing the final sculpt

1. Open `model.blend` and choose **Knight — photo modeling**.
2. Select **Knight — sculpt working copy** and enter Sculpt Mode.
3. Use Y symmetry for silhouette changes. Inspect from several directions before
   changing shallow recesses or the mane.
4. Keep the underside at Z=0 and check that the height remains approximately 27.5 mm.
5. Save `model.blend` to retain the edits.

The nose points along +X and the symmetry plane is Y=0.

| View | Direction |
| --- | --- |
| Numpad 1 | Left side profile |
| Ctrl + Numpad 1 | Opposite side profile |
| Numpad 3 | Front, looking towards the nose |
| Ctrl + Numpad 3 | Back/mane |
| Numpad 0 | Saved perspective-matched right-side camera |

The **MATCH right photo — perspective** camera preserves the user's alignment.
The **Right photo — alignment guides** collection retains the base ring and table
wire guides. These cameras and guides remain useful without a photograph overlay.

## Reference photographs and file cleanup

On 2026-11-05, packed reference photographs, image overlays and camera background
images were removed. Old sculpt snapshots, unrelated piece-study/default scenes
and orphaned data were also removed; the remaining geometry and materials were
verified unchanged. The file is saved with Blender compression enabled.

Original photographs remain outside the Blender file in the locally ignored
`assets/chess_pieces_pictures/{white_knight,black_knight}/` folders, when available.
They are not included in a fresh clone. To use them again, add an image reference
or camera background in Blender and align it to the retained camera. The original
photo objects and their opacity/alignment settings are available in the local
pre-cleanup backup, not in the lightweight model.

Local backups of the pre-cleanup files and the cleanup report are under
`.local-blender/backups/blend-cleanup-20261105-165537/`, mirroring project paths.
This folder is ignored by Git. Historical backup-object names in `validation.json`
refer to the pre-cleanup file, not the current Outliner.
