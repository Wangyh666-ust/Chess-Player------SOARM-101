# Folding chessboard

Open `model.blend`, scene **Board • photo reference**. `preview.png` is the Cycles render.

Measured outer dimensions: **248 × 245.8 mm**. The 25.2232 mm square size,
15 mm case thickness, 7.5 mm frame, 0.6 mm fold gap, corner rounding and lettering
size are estimates from IMG_4449, IMG_4450 and IMG_4452. They are not measurements.

Select **Board controls > Object Properties > Custom Properties** to edit the
dimensions live. All dimensions are in millimetres. The grid and lettering move
with square size, while the border and shell follow the outer dimensions. Keep
`8 * square_mm < min(width_mm, depth_mm) - 2 * frame_mm` and use a positive seam
smaller than one square. Extreme incompatible values can cause intersections.

The model has two rounded case halves, a gray printed surround, 64 named squares,
coordinates on all four edges, a three-step glossy black frame and a recessed black fold slit between ranks 4 and 5.
Files run along +X, ranks along +Y; a1 is dark. The origin is centred at the case
bottom. Playing surface height is `thickness_mm + 0.06` mm; the rim reaches
`thickness_mm + rim_height_mm`. The width/depth controls refer to the outer case.

Materials are editable in the Shader Editor. Lighting reflections are rendered,
not copied from the photographs. Ten separate ArUco stickers reproduce the
patterns and placement in IMG_4467.JPEG. The board is modeled open; concealed hinge/latch mechanisms,
underside cavities, wear and an opening animation are not reconstructed.

`create.py` generates the scene and validates the square colors, count,
outer case dimensions and live width control. It also works in a fresh Blender:

```bash
./.local-blender/start-blender.sh --background --python assets/chess/board/create.py
```

Regeneration saves to `model.blend`; save manual edits under another filename
first. Render Image (F12) regenerates the preview. Use the Board controls parent
to move the whole asset; studio objects are separate.

## Standard folder layout

- `model.blend`: editable Blender asset. Save manual model changes here.
- `create.py`: rebuilds the asset from the parameters; overwrites model.blend.
- `dimensions.json`: initial dimension/profile parameters in millimetres. Changes require regeneration.
- `preview.png`: rendered preview; use F12 in Blender and save the render here to refresh it.
- `validation.json`: geometry/dimension check results, not a measure of photographic accuracy.
- `README.md`: asset-specific instructions.

Preserve manual edits before running create.py. JSON parameters and Blender Custom Properties
are separate: changing one does not automatically rewrite the other.

## Marker layer

Select **Board controls → Object Properties → Custom Properties → aruco_enabled**
to show/hide all markers in both viewport and renders. The **ArUco markers**
collection also groups the ten separate objects for individual editing.

Markers: h1=4, d1=5, a2=6, h3=3, f3=11, f5=9, c6=8, h7=2, e8=1, a8=0.
The patterns match DICT_4X4_50. Sizes, border width, offsets and angles are estimates
from the photograph; the marker IDs and orientations were detected with OpenCV.
The clean paper/ink geometry does not reproduce tape, wear or photographed glare.

Edit the `aruco` section of `dimensions.json` before regeneration: `enabled` sets
initial visibility, `white_border_mm` sets paper margin, and each marker has
`square`, `black_size_mm`, `offset_mm` and `rotation_degrees`. `pattern_rows` stores
the complete 6×6 pattern including black border (1=black); changing an ID alone
does not regenerate its pattern. Markers follow live square-size/thickness changes.
No external texture files or OpenCV installation are needed to render them.

Dataset configs accept `"aruco_enabled": false` to omit stickers. `true` includes
stickers that are enabled in the saved board asset; it does not override hidden
objects or collections. Save Blender edits before generating a dataset.

## Revised frame and coordinate print

The gray coordinate surround has its own material, separate from the white squares.
The black frame has three ledges: inner shelf flush with the grid, middle shelf
halfway up, outer lip `rim_height_mm` above the grid. `frame_mm` controls the full
width (each ledge occupies one third). `slit_depth_mm` controls the recessed black
floor below the fold opening. The structural halves still have a real gap.
These profile dimensions are estimates from the board photographs, not measurements.

`label_size_mm` controls font size; `label_stroke_mm` adds a small stroke expansion.
The packed Liberation Sans font approximates the printed lettering. From the white
player's end, left numbers and bottom letters are upright; right numbers and top
letters are rotated 180 degrees. Materials remain editable in the Shader Editor.
The frame is glossier than the laminate. Dataset `surface_finishes.frame` can adjust
it independently; the general `surface_finishes.board` override now excludes the
frame, printed ink and slit interior so their distinct finishes survive import.

The frame is now 7.5 mm wide. `grid_outline_mm` controls the thin black outline
outside the 8×8 grid (0.35 mm initial estimate). White squares use a brighter,
slightly warm material distinct from the gray coordinate surround. The environment
presets were updated so their color overrides retain this contrast.

The grey surround's nominal surface area has been increased by 25%, from
10,220.71 to 12,775.89 mm². This means the planar band inside the black frame
and outside the grid outline, before lettering and the narrow fold slit.
The outer board and 7.5 mm black frame stay fixed. `square_mm` is now
25.223229104719604 instead of 26, giving a 201.7858 mm square grid. The pieces
are not resized; their size relative to a square increases by about 3.08%.
Lettering and marker centres follow the grid drivers; marker paper sizes remain
unchanged. Edit **Board controls → square_mm** to tune the surround further:
smaller squares give a wider grey band. Update `dimensions.json` too if the
same change should survive regeneration with `create.py`.
