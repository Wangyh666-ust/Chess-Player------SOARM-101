"""Build the knight asset from dimensions.json using Blender's Python.

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
PARAMETERS = json.loads((OUT / "dimensions.json").read_text())
# Reuse the monotone interpolator; do not execute the pawn scene builder.
# Extract only the pure interpolation function; importing the whole pawn script
# would also build and save a pawn scene as a side effect.
source = ast.parse((OUT.parent / "pawn/create.py").read_text())
node = next(
    n
    for n in source.body
    if isinstance(n, ast.FunctionDef) and n.name == "profile_samples"
)
exec(compile(ast.Module(body=[node], type_ignores=[]), "pawn interpolation", "exec"))
scene = bpy.data.scenes.new("Knight — black and white photo reconstruction")
bpy.context.window.scene = scene
scene.unit_settings.system = "METRIC"
scene.unit_settings.length_unit = "MILLIMETERS"


def mesh_object(name, vertices, faces):
    """Convert millimetre vertices to metres and recalculate outward normals."""
    mesh_data = bpy.data.meshes.new(name)
    mesh_data.from_pydata([tuple(v / 1000 for v in co) for co in vertices], [], faces)
    mesh_data.update()
    editable_mesh = bmesh.new()
    editable_mesh.from_mesh(mesh_data)
    bmesh.ops.recalc_face_normals(editable_mesh, faces=list(editable_mesh.faces))
    editable_mesh.to_mesh(mesh_data)
    editable_mesh.free()
    ob = bpy.data.objects.new(name, mesh_data)
    scene.collection.objects.link(ob)
    return ob


def rings_object(name, rings):
    """Connect equal-sized rings into a closed surface with end caps."""
    n = len(rings[0])
    vs = [v for ring in rings for v in ring]
    fs = []
    for i in range(len(rings) - 1):
        for j in range(n):
            a = i * n + j
            b = i * n + (j + 1) % n
            fs.append((a, b, b + n, a + n))
    fs.append(tuple(reversed(range(n))))
    fs.append(tuple(range(len(vs) - n, len(vs))))
    return mesh_object(name, vs, fs)


def active(obj):
    """Select only this object so subsequent bpy operators target it safely."""
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj


def ellipsoid(name, center, radii):
    """Create an ellipsoid from a millimetre center and axis radii."""
    bpy.ops.mesh.primitive_uv_sphere_add(
        segments=32, ring_count=20, location=Vector(center) / 1000
    )
    obj = bpy.context.object
    obj.name = name
    obj.scale = Vector(radii) / 1000
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    return obj


def subtract(obj, cut):
    """Carve one solid with another and discard the temporary cutter."""
    active(obj)
    m = obj.modifiers.new("Carved molded detail", "BOOLEAN")
    m.operation = "DIFFERENCE"
    m.object = cut
    bpy.ops.object.modifier_apply(modifier=m.name)
    bpy.data.objects.remove(cut, do_unlink=True)


def tube(name, points, radius):
    """Turn a millimetre path into a capped mesh used to carve a groove."""
    c = bpy.data.curves.new(name, "CURVE")
    c.dimensions = "3D"
    c.resolution_u = 8
    c.bevel_depth = radius / 1000
    c.bevel_resolution = 3
    c.use_fill_caps = True
    s = c.splines.new("BEZIER")
    s.bezier_points.add(len(points) - 1)
    for v, point in zip(s.bezier_points, points):
        v.co = Vector(point) / 1000
        v.handle_left_type = "AUTO"
        v.handle_right_type = "AUTO"
    obj = bpy.data.objects.new(name, c)
    scene.collection.objects.link(obj)
    active(obj)
    bpy.ops.object.convert(target="MESH")
    return bpy.context.object


# +X is the nose; Y is width. Loft sections trace the side silhouette at each height.
N = 96
base = rings_object(
    "Rounded stepped base",
    [
        [
            (r * math.cos(2 * math.pi * j / N), r * math.sin(2 * math.pi * j / N), z)
            for j in range(N)
        ]
        for z, r in profile_samples(PARAMETERS["base_profile_z_radius_mm"], 5)
    ],
)
sections = PARAMETERS["body_sections_z_back_front_halfwidth_mm"]
fields = [profile_samples([(s[0], s[k]) for s in sections], 5) for k in [1, 2, 3]]
groove = PARAMETERS["neck_groove_path_xz_mm"]


def path_distance(x, z):
    """Find the closest distance to the neck-groove polyline in millimetres."""
    nearest = 100
    for (ax, az), (bx, bz) in zip(groove, groove[1:]):
        dx = bx - ax
        dz = bz - az
        t = max(0, min(1, ((x - ax) * dx + (z - az) * dz) / (dx * dx + dz * dz)))
        nearest = min(nearest, math.hypot(x - ax - t * dx, z - az - t * dz))
    return nearest


rings = []
for (z, back), (_, front), (_, width) in zip(*fields):
    ring = []
    cx = (back + front) / 2
    rx = (front - back) / 2
    for j in range(N):
        a = 2 * math.pi * j / N
        cs = math.cos(a)
        sn = math.sin(a)
        x = cx + rx * cs
        y = width * sn
        # Broad cheek and shallow flowing relief are molded into the surface.
        bulge = PARAMETERS["cheek_bulge_mm"] * math.exp(
            -(((x - PARAMETERS["cheek_center_xz_mm"][0]) / 1.9) ** 2)
            - ((z - PARAMETERS["cheek_center_xz_mm"][1]) / 1.6) ** 2
        )
        bulge += (
            PARAMETERS["brow_radii_mm"][1]
            * 0.5
            * math.exp(
                -(
                    (
                        (x - PARAMETERS["brow_center_mm"][0])
                        / PARAMETERS["brow_radii_mm"][0]
                    )
                    ** 2
                )
                - (
                    (z - PARAMETERS["brow_center_mm"][2])
                    / PARAMETERS["brow_radii_mm"][2]
                )
                ** 2
            )
        )
        groove_depth = PARAMETERS["neck_groove_depth_mm"] * math.exp(
            -((path_distance(x, z) / PARAMETERS["neck_groove_width_mm"]) ** 2)
        )
        y += (bulge - groove_depth) * sn
        ring.append((x, y, z))
    rings.append(ring)
body = rings_object("Traced horse silhouette", rings)
parts = [base, body]
# A continuous narrow mane strip, scalloped along its rear edge; not separate beads.
mane = []
steps = 240
for i in range(steps + 1):
    z = (
        PARAMETERS["mane_bottom_mm"]
        + (PARAMETERS["mane_top_mm"] - PARAMETERS["mane_bottom_mm"]) * i / steps
    )
    back = min(fields[0], key=lambda item: abs(item[0] - z))[1]
    wave = 0.5 + 0.5 * math.cos(
        2
        * math.pi
        * (z - PARAMETERS["mane_bottom_mm"])
        / PARAMETERS["mane_tooth_pitch_mm"]
    )
    depth = PARAMETERS["mane_depth_mm"] + PARAMETERS["mane_tooth_depth_mm"] * wave
    # Both ends taper into the body.
    taper = min(
        1,
        (z - PARAMETERS["mane_bottom_mm"]) / 0.7 + 0.15,
        (PARAMETERS["mane_top_mm"] - z) / 0.7 + 0.15,
    )
    ring = []
    for j in range(24):
        a = 2 * math.pi * j / 24
        ring.append(
            (
                back + 0.1 - depth * taper * (0.5 + 0.5 * math.cos(a)),
                PARAMETERS["mane_halfwidth_mm"] * math.sin(a),
                z,
            )
        )
    mane.append(ring)
parts.append(rings_object("Continuous serrated mane", mane))
for center in PARAMETERS["ear_centers_mm"]:
    parts.append(ellipsoid("Small molded ear", center, PARAMETERS["ear_radii_mm"]))
# Join intersecting solids and voxel fuse for a continuous molded body.
active(base)
for obj in parts:
    obj.select_set(True)
bpy.ops.object.join()
knight = bpy.context.object
knight.name = "White Knight — rebuilt"
rem = knight.modifiers.new("Fuse molded shape", "REMESH")
rem.mode = "VOXEL"
rem.voxel_size = PARAMETERS["voxel_size_mm"] / 1000
rem.use_smooth_shade = True
bpy.ops.object.modifier_apply(modifier=rem.name)
smooth = knight.modifiers.new("Polish fusion", "SMOOTH")
smooth.factor = 0.45
smooth.iterations = 3
bpy.ops.object.modifier_apply(modifier=smooth.name)
# Small eye and nostril impressions, kept subtle as in the reference crops.
for sign in [-1, 1]:
    for label in ["eye", "nostril"]:
        center = PARAMETERS[label + "_center_mm"].copy()
        center[1] *= sign
        subtract(
            knight,
            ellipsoid(label + " impression", center, PARAMETERS[label + "_radii_mm"]),
        )
    z = PARAMETERS["mouth_z_mm"]
    points = [
        (3.9, sign * 2.35, z),
        (4.9, sign * 1.95, z - 0.02),
        (5.55, sign * 0.95, z),
    ]
    subtract(
        knight, tube("Short mouth crease", points, PARAMETERS["mouth_groove_radius_mm"])
    )
# Normalize bottom and full height after voxelization, preserving measured size.
z0 = min(v.co.z for v in knight.data.vertices)
z1 = max(v.co.z for v in knight.data.vertices)
for v in knight.data.vertices:
    v.co.z = (v.co.z - z0) * PARAMETERS["height_mm"] / 1000 / (z1 - z0)
for f in knight.data.polygons:
    f.use_smooth = True
editable_mesh = bmesh.new()
editable_mesh.from_mesh(knight.data)
assert all(e.is_manifold for e in editable_mesh.edges), "Mesh must be closed"
volume = editable_mesh.calc_volume(signed=True)
assert volume > 7e-7, "Missing solid volume"
# A single connected component catches detached ears or carving fragments.
remaining = set(editable_mesh.verts)
components = 0
while remaining:
    todo = [remaining.pop()]
    components += 1
    while todo:
        for e in todo.pop().link_edges:
            for v in e.verts:
                if v in remaining:
                    remaining.remove(v)
                    todo.append(v)
assert components == 1, ("Disconnected mesh", components)
editable_mesh.free()


def drive(owner, path, index, obj, prop, expr):
    """Drive a Blender property from the editable dimension controls."""
    f = owner.driver_add(path, index) if index is not None else owner.driver_add(path)
    v = f.driver.variables.new()
    v.name = "d"
    v.type = "SINGLE_PROP"
    v.targets[0].id = obj
    v.targets[0].data_path = '["' + prop + '"]'
    f.driver.expression = expr


knight["height_mm"] = float(PARAMETERS["height_mm"])
drive(knight, "scale", 2, knight, "height_mm", f'd/{PARAMETERS["height_mm"]}')
knight.shape_key_add(name="Basis")
for prop in ["base_diameter_mm", "body_depth_mm", "body_width_mm"]:
    knight[prop] = float(PARAMETERS[prop])
    key = knight.shape_key_add(name=prop)
    key.slider_min = -0.8
    key.slider_max = 3
    for v, k in zip(knight.data.vertices, key.data):
        z = v.co.z * 1000
        t = max(0, min(1, (z - 5.5) / 2))
        t = t * t * (3 - 2 * t)
        if prop == "base_diameter_mm":
            k.co.x *= 2 - t
            k.co.y *= 2 - t
        elif prop == "body_depth_mm":
            k.co.x *= 1 + t
        else:
            k.co.y *= 1 + t
    drive(key, "value", None, knight, prop, f"d/{PARAMETERS[prop]}-1")
for prop in ["height_mm", "base_diameter_mm", "body_depth_mm", "body_width_mm"]:
    knight.id_properties_ui(prop).update(
        min=1.0,
        max=100.0,
        description="Millimeters; only total height measured. Profile and detail settings are in dimensions.json.",
    )
knight["reference"] = (
    "All black_knight and white_knight views. Photo-guided approximation; widths and fine details estimated."
)


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


knight.data.materials.append(material("Knight ivory plastic", (0.81, 0.70, 0.47), 0.24))
other = knight.copy()
other.data = knight.data.copy()
other.name = "Black Knight — rebuilt"
scene.collection.objects.link(other)
for owner in [other, other.data.shape_keys]:
    for f in owner.animation_data.drivers:
        for variable in f.driver.variables:
            for target in variable.targets:
                if target.id == knight:
                    target.id = other
other.data.materials.clear()
other.data.materials.append(
    material("Knight black plastic", (0.009, 0.008, 0.007), 0.23)
)
knight.location.x = -0.013
other.location.x = 0.013
other.rotation_euler.z = math.radians(-45)
bpy.context.view_layer.update()
assert abs(knight.dimensions.z - PARAMETERS["height_mm"] / 1000) < 1e-6
knight["height_mm"] = 30.0
knight.update_tag()
bpy.context.view_layer.update()
assert abs(knight.dimensions.z - 0.03) < 1e-6
knight["height_mm"] = PARAMETERS["height_mm"]
knight.update_tag()
bpy.context.view_layer.update()
bpy.ops.mesh.primitive_plane_add(size=2, location=(0, 0, -0.00004))
bpy.context.object.name = "Knight studio floor"
bpy.context.object.data.materials.append(
    material("Knight slate", (0.11, 0.14, 0.17), 0.65)
)


def aim(obj, p):
    """Point a camera or light toward a world-space target using Blender axes."""
    obj.rotation_euler = (Vector(p) - obj.location).to_track_quat("-Z", "Y").to_euler()


for name, loc, power, size in [
    ("Key", (-0.045, -0.065, 0.085), 0.45, 0.055),
    ("Rim", (0.045, 0.025, 0.065), 0.4, 0.045),
    ("Fill", (0.01, -0.065, 0.035), 0.08, 0.05),
]:
    d = bpy.data.lights.new("Knight " + name, "AREA")
    d.energy = power
    d.shape = "DISK"
    d.size = size
    obj = bpy.data.objects.new("Knight " + name, d)
    scene.collection.objects.link(obj)
    obj.location = loc
    aim(obj, (0, 0, 0.014))
d = bpy.data.cameras.new("Knight comparison")
camera = bpy.data.objects.new("Knight comparison", d)
scene.collection.objects.link(camera)
camera.location = (0.0, -0.12, 0.048)
aim(camera, (0, 0, 0.013))
d.type = "ORTHO"
d.ortho_scale = 0.055
d.clip_start = 0.001
scene.camera = camera
scene.world = bpy.data.worlds.new("Knight studio")
scene.world.use_nodes = True
background_shader = next(
    n for n in scene.world.node_tree.nodes if n.type == "BACKGROUND"
)
background_shader.inputs["Strength"].default_value = 0.22
scene.render.engine = "CYCLES"
scene.cycles.samples = 48
scene.cycles.use_denoising = True
scene.render.resolution_x = 1200
scene.render.resolution_y = 900
scene.render.resolution_percentage = 100
scene.view_settings.exposure = -0.5
scene.render.image_settings.file_format = "PNG"
scene.render.filepath = str(OUT / "preview.png")
active(knight)
# Set a useful opening view; these UI settings do not affect the mesh.
for screen in bpy.data.screens:
    for area in screen.areas:
        if area.type == "VIEW_3D":
            area.spaces.active.clip_start = 0.0001
            area.spaces.active.region_3d.view_distance = 0.085
            area.spaces.active.region_3d.view_location = (0, 0, 0.014)
            area.spaces.active.region_3d.view_rotation = (
                camera.rotation_euler.to_quaternion()
            )
            area.spaces.active.shading.color_type = "MATERIAL"
        if area.type == "PROPERTIES":
            area.spaces.active.context = "OBJECT"
report = {
    "height_mm": PARAMETERS["height_mm"],
    "vertices": len(knight.data.vertices),
    "closed": True,
    "connected_components": components,
    "volume_mm3": volume * 1e9,
    "height_driver_checked": True,
}
(OUT / "validation.json").write_text(json.dumps(report, indent=2))
for filename in ["create.py", "dimensions.json", "README.md"]:
    t = bpy.data.texts.new("Knight rebuild " + filename)
    t.write((OUT / filename).read_text())
# Save the editable asset after geometry and driver checks have passed.
bpy.ops.wm.save_as_mainfile(filepath=str(OUT / "model.blend"))
print(json.dumps(report))
