"""Build the queen asset from dimensions.json using Blender's Python.

Run this script through Blender, not system Python. Dimensions are in millimetres;
mesh coordinates are in metres. Geometry, live controls, studio setup, and
validation are built in that order. Saving overwrites model.blend.
"""

import bpy
import bmesh
import math
import json
import ast
from pathlib import Path
from mathutils import Vector

# Keep asset inputs and outputs beside this script.
OUT = Path(__file__).resolve().parent
# Reuse the pawn's overshoot-free profile interpolation without executing its scene builder.
# Extract only the pure interpolation function; importing the whole pawn script
# would also build and save a pawn scene as a side effect.
source = ast.parse((OUT.parent / "pawn/create.py").read_text())
node = next(
    n
    for n in source.body
    if isinstance(n, ast.FunctionDef) and n.name == "profile_samples"
)
exec(compile(ast.Module(body=[node], type_ignores=[]), "pawn profile helper", "exec"))
PARAMETERS = {"queen": json.loads((OUT / "dimensions.json").read_text())}
scene = bpy.data.scenes.new("Queen — individual asset")
bpy.context.window.scene = scene
scene.unit_settings.system = "METRIC"
scene.unit_settings.length_unit = "MILLIMETERS"


def material(name, color, rough):
    """Create a Principled shader; colors are linear RGB, roughness is unitless."""
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    m.diffuse_color = (*color, 1)
    shader = next(n for n in m.node_tree.nodes if n.type == "BSDF_PRINCIPLED")
    shader.inputs["Base Color"].default_value = (*color, 1)
    shader.inputs["Roughness"].default_value = rough
    shader.inputs["IOR"].default_value = 1.46
    return m


cream = material("Ivory molded plastic", (0.81, 0.70, 0.47), 0.25)
black = material("Black molded plastic", (0.009, 0.008, 0.007), 0.23)


def lathe(name, parameters):
    """Revolve a millimetre profile into a closed mesh stored in metres."""
    profile = profile_samples(parameters["profile_z_radius_mm"], 4)
    n = 128
    vertices = []
    faces = []
    for z, r in profile:
        # A shallow molded scallop, not the tall battlements of a generic rook.
        fade = (
            max(0, min(1, (z - parameters["scallop_start_mm"]) / 1.0))
            if parameters.get("scallops", 0)
            else 0
        )
        for j in range(n):
            a = 2 * math.pi * j / n
            radius = r * (
                1
                - parameters.get("scallop_depth_fraction", 0)
                * fade
                * (0.5 + 0.5 * math.cos(parameters.get("scallops", 0) * a))
            )
            vertices.append(
                (radius * math.cos(a) / 1000, radius * math.sin(a) / 1000, z / 1000)
            )
    for i in range(len(profile) - 1):
        for j in range(n):
            a = i * n + j
            b = i * n + (j + 1) % n
            faces.append((a, b, b + n, a + n))
    bottom = len(vertices)
    vertices.append((0, 0, profile[0][0] / 1000))
    top = len(vertices)
    vertices.append((0, 0, profile[-1][0] / 1000))
    for j in range(n):
        k = (j + 1) % n
        faces.append((bottom, k, j))
        a = (len(profile) - 1) * n
        faces.append((a + j, a + k, top))
    mesh = bpy.data.meshes.new(name + " continuous profile")
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    scene.collection.objects.link(obj)
    for f in mesh.polygons:
        f.use_smooth = True
    return obj


def active(obj):
    """Select only this object so subsequent bpy operators target it safely."""
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj


def boolean(obj, cutter, operation):
    """Apply a union or difference, then remove the temporary cutter object."""
    active(obj)
    mod = obj.modifiers.new(operation.title(), "BOOLEAN")
    mod.operation = operation
    mod.object = cutter
    bpy.ops.object.modifier_apply(modifier=mod.name)
    bpy.data.objects.remove(cutter, do_unlink=True)


def box(name, center, size, bevel=0):
    """Create a box from the supplied dimensions and location."""
    bpy.ops.mesh.primitive_cube_add(size=1, location=Vector(center) / 1000)
    obj = bpy.context.object
    obj.name = name
    obj.dimensions = Vector(size) / 1000
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    if bevel:
        mod = obj.modifiers.new("Molded edge rounding", "BEVEL")
        mod.width = bevel / 1000
        mod.segments = 3
        bpy.ops.object.modifier_apply(modifier=mod.name)
    return obj


def add_driver(owner, path, index, obj, prop, expr):
    """Bind a dimension property to a scale component or shape-key value."""
    f = owner.driver_add(path, index) if index is not None else owner.driver_add(path)
    d = f.driver
    v = d.variables.new()
    v.name = "d"
    v.type = "SINGLE_PROP"
    v.targets[0].id = obj
    v.targets[0].data_path = '["' + prop + '"]'
    d.expression = expr


def controls(obj, parameters):
    """Expose height and radial widths without changing the original mesh."""
    obj["height_mm"] = float(parameters["height_mm"])
    add_driver(obj, "scale", 2, obj, "height_mm", f'd/{parameters["height_mm"]}')
    obj.shape_key_add(name="Basis")
    # Three radial zones with smooth transitions; parameters do not alter height.
    b, s, h = parameters["width_zone_heights_mm"]
    for part, diam in [
        ("base", parameters["base_diameter_mm"]),
        ("stem", parameters["stem_diameter_mm"]),
        ("head", parameters["head_diameter_mm"]),
    ]:
        prop = part + "_diameter_mm"
        obj[prop] = float(diam)
        key = obj.shape_key_add(name=part.title() + " width")
        key.slider_min = -0.9
        key.slider_max = 3
        for v, k in zip(obj.data.vertices, key.data):
            z = v.co.z * 1000
            t = max(0, min(1, (z - b) / (s - b)))
            t = t * t * (3 - 2 * t)
            u = max(0, min(1, (z - s) / (h - s)))
            u = u * u * (3 - 2 * u)
            w = {"base": 1 - t, "stem": t * (1 - u), "head": u}[part]
            k.co.x = v.co.x * (1 + w)
            k.co.y = v.co.y * (1 + w)
        add_driver(key, "value", None, obj, prop, f"d/{diam}-1")
    for prop in [
        "height_mm",
        "base_diameter_mm",
        "stem_diameter_mm",
        "head_diameter_mm",
    ]:
        obj.id_properties_ui(prop).update(
            min=0.5,
            max=100,
            description="mm; only total height measured. Detailed profile and top parameters in dimensions.json.",
        )
    obj["reference_note"] = (
        "Both black and white photos; widths estimated. King cross from black king; white king photos appear queen-like."
    )


report = []
objects = []
for i, (name, parameters) in enumerate(PARAMETERS.items()):
    obj = lathe("White " + name.title(), parameters)
    if name == "bishop":
        cut = box(
            "Mitre diagonal slit",
            parameters["slit_center_mm"],
            parameters["slit_size_mm"],
        )
        cut.rotation_euler.y = math.radians(parameters["slit_angle_deg"])
        boolean(obj, cut, "DIFFERENCE")
        active(obj)
        mod = obj.modifiers.new("Soft slit edges", "BEVEL")
        mod.width = 0.09 / 1000
        mod.segments = 2
        mod.limit_method = "ANGLE"
        bpy.ops.object.modifier_apply(modifier=mod.name)
    if name == "king":
        # One extruded cross silhouette, with rounded molded edges.
        w = parameters["cross_width_mm"] / 2
        t = parameters["cross_stem_width_mm"] / 2
        z = parameters["cross_bottom_mm"]
        h = parameters["height_mm"]
        mid = parameters["cross_bar_center_mm"]
        bh = parameters["cross_bar_height_mm"] / 2
        outline = [
            (-t, z),
            (t, z),
            (0.7 * t, mid - bh),
            (w, mid - bh),
            (w, mid + bh),
            (0.7 * t, mid + 0.65 * bh),
            (t, h),
            (-t, h),
            (-0.7 * t, mid + 0.65 * bh),
            (-w, mid + bh),
            (-w, mid - bh),
            (-0.7 * t, mid - bh),
        ]
        depth = parameters["cross_depth_mm"] / 2
        vertices = [
            (x / 1000, y / 1000, zz / 1000)
            for y in [-depth, depth]
            for x, zz in outline
        ]
        n = len(outline)
        faces = [tuple(range(n - 1, -1, -1)), tuple(range(n, 2 * n))] + [
            (j, (j + 1) % n, (j + 1) % n + n, j + n) for j in range(n)
        ]
        mesh_data = bpy.data.meshes.new("Cross")
        mesh_data.from_pydata(vertices, [], faces)
        mesh_data.update()
        cross = bpy.data.objects.new("Cross", mesh_data)
        scene.collection.objects.link(cross)
        editable_mesh = bmesh.new()
        editable_mesh.from_mesh(mesh_data)
        bmesh.ops.recalc_face_normals(editable_mesh, faces=list(editable_mesh.faces))
        editable_mesh.to_mesh(mesh_data)
        editable_mesh.free()
        active(cross)
        mod = cross.modifiers.new("Rounded cross", "BEVEL")
        mod.width = parameters["cross_bevel_mm"] / 1000
        mod.segments = 4
        bpy.ops.object.modifier_apply(modifier=mod.name)
        boolean(obj, cross, "UNION")
    editable_mesh = bmesh.new()
    editable_mesh.from_mesh(obj.data)
    bmesh.ops.recalc_face_normals(editable_mesh, faces=list(editable_mesh.faces))
    editable_mesh.to_mesh(obj.data)
    assert all(e.is_manifold for e in editable_mesh.edges), name + " not closed"
    volume = editable_mesh.calc_volume(signed=True)
    assert volume > 3e-7, name + " missing volume"
    editable_mesh.free()
    # A rounded cross bevel slightly lowers its apex: normalize to the measured height.
    zmax = max(v.co.z for v in obj.data.vertices)
    for v in obj.data.vertices:
        v.co.z *= parameters["height_mm"] / 1000 / zmax
    controls(obj, parameters)
    obj.data.materials.append(cream)
    obj.location = (-0.015, 0, 0)
    objects.append(obj)
    # Own mesh/key datablock so each color's dimensions can be edited independently.
    other = obj.copy()
    other.data = obj.data.copy()
    other.name = "Black " + name.title()
    scene.collection.objects.link(other)
    other.data.materials.clear()
    other.data.materials.append(black)
    other.location.x = 0.015
    for owner in [other, other.data.shape_keys]:
        if owner.animation_data:
            for f in owner.animation_data.drivers:
                for variable in f.driver.variables:
                    for target in variable.targets:
                        if target.id == obj:
                            target.id = other
    objects.append(other)
    bpy.context.view_layer.update()
    assert abs(obj.dimensions.z * 1000 - parameters["height_mm"]) < 0.001
    obj["height_mm"] = float(parameters["height_mm"]) + 1
    obj.update_tag()
    bpy.context.view_layer.update()
    assert abs(obj.dimensions.z * 1000 - parameters["height_mm"] - 1) < 0.001
    obj["height_mm"] = float(parameters["height_mm"])
    obj.update_tag()
    bpy.context.view_layer.update()
    report.append(
        {
            "piece": name,
            "height_mm": parameters["height_mm"],
            "vertices": len(obj.data.vertices),
            "closed": True,
            "volume_mm3": round(volume * 1e9, 2),
        }
    )

bpy.ops.mesh.primitive_plane_add(size=2, location=(0, 0, -0.00005))
floor = bpy.context.object
floor.name = "Studio floor"
floor.data.materials.append(material("Slate studio", (0.13, 0.16, 0.19), 0.65))


def aim(obj, point):
    """Point a camera or light toward a world-space target using Blender axes."""
    obj.rotation_euler = (
        (Vector(point) - obj.location).to_track_quat("-Z", "Y").to_euler()
    )


for name, loc, power, size in [
    ("Key", (-0.08, -0.08, 0.16), 4, 0.12),
    ("Rim", (0.07, 0.05, 0.13), 3, 0.09),
    ("Fill", (0.02, -0.13, 0.07), 1, 0.1),
]:
    d = bpy.data.lights.new(name, "AREA")
    d.energy = power
    d.shape = "DISK"
    d.size = size
    obj = bpy.data.objects.new(name, d)
    scene.collection.objects.link(obj)
    obj.location = loc
    aim(obj, (0, 0, 0.02))
d = bpy.data.cameras.new("Comparison camera")
camera = bpy.data.objects.new("Comparison camera", d)
scene.collection.objects.link(camera)
camera.location = (0.006, -0.24, 0.16)
aim(camera, (0, 0, 0.019))
d.type = "ORTHO"
d.ortho_scale = 0.085
d.clip_start = 0.001
scene.camera = camera
scene.world = bpy.data.worlds.new("Soft studio ambient")
scene.world.use_nodes = True
background_shader = next(
    n for n in scene.world.node_tree.nodes if n.type == "BACKGROUND"
)
background_shader.inputs["Color"].default_value = (0.65, 0.7, 0.8, 1)
background_shader.inputs["Strength"].default_value = 0.3
scene.view_settings.exposure = -1.6
scene.render.engine = "CYCLES"
scene.cycles.samples = 48
scene.cycles.use_denoising = True
scene.render.resolution_x = 1400
scene.render.resolution_y = 1100
scene.render.resolution_percentage = 100
scene.render.image_settings.file_format = "PNG"
scene.render.filepath = str(OUT / "preview.png")
active(objects[0])
# Set a useful opening view; these UI settings do not affect the mesh.
for screen in bpy.data.screens:
    for area in screen.areas:
        if area.type == "VIEW_3D":
            area.spaces.active.clip_start = 0.0001
            area.spaces.active.region_3d.view_distance = 0.12
            area.spaces.active.region_3d.view_location = (0, 0, 0.02)
            area.spaces.active.region_3d.view_rotation = (
                camera.rotation_euler.to_quaternion()
            )
            area.spaces.active.shading.color_type = "MATERIAL"
        if area.type == "PROPERTIES":
            area.spaces.active.context = "OBJECT"
for filename in ["create.py", "dimensions.json", "README.md"]:
    text = bpy.data.texts.new(filename)
    text.write((OUT / filename).read_text())
(OUT / "validation.json").write_text(json.dumps(report, indent=2))
# Save the editable asset after geometry and driver checks have passed.
bpy.ops.wm.save_as_mainfile(filepath=str(OUT / "model.blend"))
print(json.dumps(report))
