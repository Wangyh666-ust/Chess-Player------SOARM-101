# Bishop

Open `model.blend` to edit the white and black bishop models.
Select a piece and use Object Properties > Custom Properties for dimensions in mm.
Measured height: 32 mm; other proportions are estimates from the photos.
Geometry, materials, shape keys and dimension controls are preserved from the original asset.

`dimensions.json` contains this piece's profile and detail parameters.
`create.py` rebuilds this piece and saves `model.blend`; save manual edits elsewhere before regenerating.
From the project root:

```bash
.local-blender/start-blender.sh --background --python assets/chess/bishop/create.py
```

The script reuses profile interpolation from `../pawn/create.py`.
`validation.json` contains the original model's geometry checks.
The preview render path is `preview.png`; render with F12 to create it.
Dataset presets now load this file. They use the white mesh for both colors and replace piece materials.
Keep scene/object names unchanged or update the presets accordingly.

## Standard folder layout

- `model.blend`: editable Blender asset. Save manual model changes here.
- `create.py`: rebuilds the asset from the parameters; overwrites model.blend.
- `dimensions.json`: initial dimension/profile parameters in millimetres. Changes require regeneration.
- `preview.png`: rendered preview; use F12 in Blender and save the render here to refresh it.
- `validation.json`: geometry/dimension check results, not a measure of photographic accuracy.
- `README.md`: asset-specific instructions.

Preserve manual edits before running create.py. JSON parameters and Blender Custom Properties
are separate: changing one does not automatically rewrite the other.
