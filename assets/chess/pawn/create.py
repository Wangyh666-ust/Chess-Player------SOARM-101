"""Build the pawn asset from dimensions.json using Blender's Python.

Run this script through Blender, not system Python. Dimensions are in millimetres;
mesh coordinates are in metres. Geometry, live controls, studio setup, and
validation are built in that order. Saving overwrites model.blend.
"""

import bpy
import math
import bmesh
import json
from mathutils import Vector
from pathlib import Path

# Keep asset inputs and outputs beside this script.
OUT = Path(__file__).resolve().parent
OUT.mkdir(parents=True, exist_ok=True)
PARAMETERS = json.loads((OUT / "dimensions.json").read_text())
# Millimetres. Height is measured; all widths/profile details are photo estimates.
HEIGHT = PARAMETERS["height_mm"]
BASE_DIAMETER = PARAMETERS["base_diameter_mm"]
HEAD_DIAMETER = PARAMETERS["head_diameter_mm"]
COLLAR_DIAMETER = PARAMETERS["collar_diameter_mm"]
STEM_DIAMETER = PARAMETERS["stem_diameter_mm"]
SEGMENTS = PARAMETERS["segments"]
# (height, radius): rounded foot, raised shoulder, concave stem, thin collar.
PROFILE = [
    (z * HEIGHT / PARAMETERS["profile_reference_height_mm"], r)
    for z, r in PARAMETERS["profile_z_radius_mm"]
]


# Monotone cubic interpolation avoids overshoot at the foot and collar.
def profile_samples(points, steps=4):
    """Interpolate height/radius pairs without overshooting the measured outline."""
    slopes = [(b[1] - a[1]) / (b[0] - a[0]) for a, b in zip(points, points[1:])]
    tangents = [slopes[0]]
    for a, b in zip(slopes, slopes[1:]):
        tangents.append(0 if a * b <= 0 else 2 * a * b / (a + b))
    tangents.append(slopes[-1])
    result = []
    for i, (a, b) in enumerate(zip(points, points[1:])):
        dz = b[0] - a[0]
        for j in range(steps):
            t = j / steps
            r = (
                (2 * t**3 - 3 * t * t + 1) * a[1]
                + (t**3 - 2 * t * t + t) * dz * tangents[i]
                + (-2 * t**3 + 3 * t * t) * b[1]
                + (t**3 - t * t) * dz * tangents[i + 1]
            )
            result.append((a[0] + t * dz, r))
    result.append(points[-1])
    return result


profile = profile_samples(PROFILE)
head_r = HEAD_DIAMETER / 2
head_c = HEIGHT - head_r
start = math.asin((PROFILE[-1][0] - head_c) / head_r)
# The shared join follows the spherical head exactly.
profile[-1] = (PROFILE[-1][0], head_r * math.cos(start))
for i in range(1, 49):
    angle = start + (math.pi / 2 - start) * i / 49
    profile.append((head_c + head_r * math.sin(angle), head_r * math.cos(angle)))

scene = bpy.data.scenes.new("Pawn • photo reference")
bpy.context.window.scene = scene
scene.unit_settings.system = "METRIC"
scene.unit_settings.length_unit = "MILLIMETERS"
scene.unit_settings.scale_length = 1.0
vertices = []
for z, r in profile:
    for j in range(SEGMENTS):
        a = 2 * math.pi * j / SEGMENTS
        vertices.append((r * math.cos(a) / 1000, r * math.sin(a) / 1000, z / 1000))
faces = []
for i in range(len(profile) - 1):
    for j in range(SEGMENTS):
        a = i * SEGMENTS + j
        b = i * SEGMENTS + (j + 1) % SEGMENTS
        faces.append((a, b, b + SEGMENTS, a + SEGMENTS))
bottom = len(vertices)
vertices.append((0, 0, 0))
top = len(vertices)
vertices.append((0, 0, HEIGHT / 1000))
for j in range(SEGMENTS):
    k = (j + 1) % SEGMENTS
    faces.append((bottom, k, j))
    a = (len(profile) - 1) * SEGMENTS
    faces.append((a + j, a + k, top))
mesh = bpy.data.meshes.new("Pawn continuous revolved profile")
mesh.from_pydata(vertices, [], faces)
mesh.update()
pawn = bpy.data.objects.new("Pawn", mesh)
scene.collection.objects.link(pawn)
for p in mesh.polygons:
    p.use_smooth = True
for p in mesh.polygons:
    if bottom in p.vertices:
        p.use_smooth = False
pawn["height_mm"] = HEIGHT
pawn["base_diameter_mm"] = BASE_DIAMETER
pawn["head_diameter_mm"] = HEAD_DIAMETER
pawn["collar_diameter_mm"] = COLLAR_DIAMETER
pawn["stem_diameter_mm"] = STEM_DIAMETER
pawn["reference"] = "IMG_4435.JPEG, IMG_4439.JPEG; only 23 mm height is measured"
for key in [
    "height_mm",
    "base_diameter_mm",
    "head_diameter_mm",
    "collar_diameter_mm",
    "stem_diameter_mm",
]:
    pawn.id_properties_ui(key).update(
        min=1.0,
        max=100.0,
        description="Millimetres. Widths are estimates from the reference photos.",
    )


def driver(owner, path, index, prop, expr):
    """Connect a pawn custom property to an animated dimension or shape key."""
    curve = (
        owner.driver_add(path, index) if index is not None else owner.driver_add(path)
    )
    d = curve.driver
    d.type = "SCRIPTED"
    v = d.variables.new()
    v.name = "dimension"
    v.type = "SINGLE_PROP"
    v.targets[0].id = pawn
    v.targets[0].data_path = '["' + prop + '"]'
    d.expression = expr


driver(pawn, "scale", 2, "height_mm", f"dimension / {HEIGHT}")
pawn.shape_key_add(name="Basis")
# Local radial controls blend across the transitions; total height stays fixed.
anchors = [
    (0, "base"),
    (6.6, "base"),
    (10.8, "stem"),
    (13.6, "stem"),
    (14.0, "collar"),
    (14.7, "collar"),
    (15.05, "head"),
    (HEIGHT, "head"),
]
anchors = [
    (z * HEIGHT / PARAMETERS["profile_reference_height_mm"], part)
    for z, part in anchors[:-1]
] + [(HEIGHT, "head")]


def weight(z, part):
    """Blend radial shape-key influence between neighboring height anchors."""
    for (za, pa), (zb, pb) in zip(anchors, anchors[1:]):
        if za <= z <= zb:
            t = (z - za) / (zb - za)
            t = t * t * (3 - 2 * t)
            return (1 - t) * (pa == part) + t * (pb == part)
    return 0


for part, diam in [
    ("base", BASE_DIAMETER),
    ("stem", STEM_DIAMETER),
    ("collar", COLLAR_DIAMETER),
    ("head", HEAD_DIAMETER),
]:
    key = pawn.shape_key_add(name=part.title() + " width")
    key.slider_min = -1
    key.slider_max = 10
    for v, k in zip(mesh.vertices, key.data):
        w = weight(v.co.z * 1000, part)
        k.co.x = v.co.x * (1 + w)
        k.co.y = v.co.y * (1 + w)
    driver(key, "value", None, part + "_diameter_mm", f"dimension / {diam} - 1.0")
mat = bpy.data.materials.new("Cream glossy molded plastic")
mat.use_nodes = True
shader = next(n for n in mat.node_tree.nodes if n.type == "BSDF_PRINCIPLED")
shader.inputs["Base Color"].default_value = (0.82, 0.745, 0.53, 1)
shader.inputs["Roughness"].default_value = 0.23
shader.inputs["IOR"].default_value = 1.46
mat.diffuse_color = (0.92, 0.87, 0.72, 1)
pawn.data.materials.append(mat)


def material(name, color, roughness):
    """Create a Principled shader; colors are linear RGB, roughness is unitless."""
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    n = next(n for n in m.node_tree.nodes if n.type == "BSDF_PRINCIPLED")
    n.inputs["Base Color"].default_value = (*color, 1)
    n.inputs["Roughness"].default_value = roughness
    return m


bpy.ops.mesh.primitive_plane_add(size=2, location=(0, 0, -0.00004))
floor = bpy.context.object
floor.name = "Studio floor"
floor.data.materials.append(material("Warm slate", (0.095, 0.12, 0.14), 0.6))


def aim(obj, point):
    """Point a camera or light toward a world-space target using Blender axes."""
    obj.rotation_euler = (
        (Vector(point) - obj.location).to_track_quat("-Z", "Y").to_euler()
    )


for name, loc, power, size in [
    ("Large softbox", (-0.045, -0.035, 0.065), 0.28, 0.045),
    ("Right reflection strip", (0.035, 0.005, 0.045), 0.18, 0.025),
    ("Soft fill", (0.0, -0.04, 0.025), 0.04, 0.04),
]:
    data = bpy.data.lights.new(name, "AREA")
    data.energy = power
    data.shape = "DISK"
    data.size = size
    obj = bpy.data.objects.new(name, data)
    scene.collection.objects.link(obj)
    obj.location = loc
    aim(obj, (0, 0, 0.012))
camera_data = bpy.data.cameras.new("Pawn portrait")
camera = bpy.data.objects.new("Pawn portrait", camera_data)
scene.collection.objects.link(camera)
camera.location = (0.037, -0.065, 0.032)
aim(camera, (0, 0, 0.0115))
camera_data.type = "ORTHO"
camera_data.ortho_scale = 0.033
camera_data.clip_start = 0.001
scene.camera = camera
scene.world = bpy.data.worlds.new("Studio ambient")
scene.world.use_nodes = True
background_shader = next(
    n for n in scene.world.node_tree.nodes if n.type == "BACKGROUND"
)
background_shader.inputs["Color"].default_value = (0.5, 0.55, 0.65, 1)
background_shader.inputs["Strength"].default_value = 0.25
scene.render.engine = "CYCLES"
scene.cycles.samples = 48
scene.cycles.use_denoising = True
scene.render.resolution_x = 800
scene.render.resolution_y = 900
scene.render.resolution_percentage = 100
scene.render.image_settings.file_format = "PNG"
scene.render.filepath = str(OUT / "preview.png")
for obj in bpy.context.selected_objects:
    obj.select_set(False)
pawn.select_set(True)
bpy.context.view_layer.objects.active = pawn
# Set a useful opening view; these UI settings do not affect the mesh.
for screen in bpy.data.screens:
    for area in screen.areas:
        if area.type == "VIEW_3D":
            area.spaces.active.clip_start = 0.0001
            area.spaces.active.region_3d.view_distance = 0.065
            area.spaces.active.region_3d.view_location = (0, 0, 0.0115)
            area.spaces.active.region_3d.view_rotation = (
                camera.rotation_euler.to_quaternion()
            )
            area.spaces.active.shading.color_type = "MATERIAL"
        if area.type == "PROPERTIES":
            area.spaces.active.context = "OBJECT"
bpy.context.view_layer.update()
editable_mesh = bmesh.new()
editable_mesh.from_mesh(mesh)
assert all(e.is_manifold for e in editable_mesh.edges), "Mesh must be closed"
assert editable_mesh.calc_volume(signed=True) > 0, "Normals must face outwards"
editable_mesh.free()
assert abs(pawn.dimensions.z - HEIGHT / 1000) < 1e-7
# Exercise a parameter to check its driver, then restore the measured height.
pawn["height_mm"] = HEIGHT + 2
pawn.update_tag()
bpy.context.view_layer.update()
assert abs(pawn.dimensions.z - (HEIGHT + 2) / 1000) < 1e-7
pawn["height_mm"] = HEIGHT
pawn.update_tag()
bpy.context.view_layer.update()
# Save the editable asset after geometry and driver checks have passed.
bpy.ops.wm.save_as_mainfile(filepath=str(OUT / "model.blend"))
report = {
    "file": str(OUT / "model.blend"),
    "dimensions_mm": [round(v * 1000, 4) for v in pawn.dimensions],
    "vertices": len(mesh.vertices),
    "closed_mesh": True,
}
(OUT / "validation.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report))
