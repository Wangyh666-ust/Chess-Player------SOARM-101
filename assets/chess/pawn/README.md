# Photo-reference pawn

Open `model.blend`, scene `Pawn • photo reference`. The original startup scene is preserved.
`preview.png` is a Cycles render of the model.

Measured: total height 23 mm. Estimated from IMG_4435.JPEG and IMG_4439.JPEG:
base diameter 14.6 mm, head width 8.6 mm, collar width 8.5 mm, nominal neck width 4.2 mm.
This is an approximate reconstruction, not a scan. The underside is a flat closed
surface because its real construction is not shown in the references.

Select **Pawn**, then **Object Properties > Custom Properties**:

- `height_mm`: total height, measured from the flat base to the head tip.
- `base_diameter_mm`: maximum base width.
- `head_diameter_mm`: horizontal head width.
- `collar_diameter_mm`: horizontal collar width.
- `stem_diameter_mm`: nominal stem width control.

Drivers update the geometry automatically. Height scales all vertical proportions;
width controls blend locally through the connected surface. Changing head width
alone makes an ellipsoid, rather than changing its vertical size as well.
For individual vertical dimensions, edit `profile_z_radius_mm` in `dimensions.json`
and regenerate. All unmeasured dimensions remain explicit editable estimates.

The mesh is one continuous closed surface, with outward normals and a base at Z=0.
One Blender unit is one metre. The cream material is an estimated glossy plastic;
its color and roughness are editable in the Shader Editor. Studio objects are
separate from the pawn and are not part of the asset geometry.

The generator creates a new scene and saves to `model.blend`; save any manual
changes under a different filename before regenerating. From the project root:

```bash
./.local-blender/start-blender.sh --background --python assets/chess/pawn/create.py
```

The generator checks closed topology, outward orientation, height, and its height driver.
The preview can be regenerated with Blender's Render Image command (F12).

## Standard folder layout

- `model.blend`: editable Blender asset. Save manual model changes here.
- `create.py`: rebuilds the asset from the parameters; overwrites model.blend.
- `dimensions.json`: initial dimension/profile parameters in millimetres. Changes require regeneration.
- `preview.png`: rendered preview; use F12 in Blender and save the render here to refresh it.
- `validation.json`: geometry/dimension check results, not a measure of photographic accuracy.
- `README.md`: asset-specific instructions.

Preserve manual edits before running create.py. JSON parameters and Blender Custom Properties
are separate: changing one does not automatically rewrite the other.
