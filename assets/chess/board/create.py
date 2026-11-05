"""Build the board asset from dimensions.json using Blender's Python.

Run this script through Blender, not system Python. Dimensions are in millimetres;
mesh coordinates are in metres. Geometry, live controls, studio setup, and
validation are built in that order. Saving overwrites model.blend.
"""

import bpy
import math
import json
from pathlib import Path
from mathutils import Vector

# Keep asset inputs and outputs beside this script.
OUT = Path(__file__).resolve().parent
OUT.mkdir(parents=True, exist_ok=True)
scene = bpy.data.scenes.new("Board • photo reference")
bpy.context.window.scene = scene
scene.unit_settings.system = "METRIC"
scene.unit_settings.length_unit = "MILLIMETERS"
root = bpy.data.objects.new("Board controls", None)
scene.collection.objects.link(root)
root.empty_display_size = 0.03
PARAMS = json.loads((OUT / "dimensions.json").read_text())
for k, v in PARAMS.items():
    if k == "aruco":
        continue
    root[k] = v
    root.id_properties_ui(k).update(
        min=0.01,
        max=1000,
        description="Millimetres. Only outer width and depth are measured.",
    )
root["notes"] = (
    "Only 248 x 245.8 mm is measured. Square size, frame, thickness and seam are photo estimates."
)
root["orientation"] = (
    "Files a-h along +X; ranks 1-8 along +Y; a1 dark. Playing surface Z = thickness_mm + 0.06 mm."
)
names = {
    "W": "width_mm",
    "D": "depth_mm",
    "T": "thickness_mm",
    "S": "square_mm",
    "F": "frame_mm",
    "G": "seam_mm",
    "R": "rim_height_mm",
    "C": "corner_radius_mm",
    "L": "label_size_mm",
    "B": "label_stroke_mm",
    "Q": "slit_depth_mm",
    "O": "grid_outline_mm",
}


def drive(obj, path, index, expression):
    """Drive a Blender property from the editable dimension controls."""
    f = obj.driver_add(path, index) if index is not None else obj.driver_add(path)
    d = f.driver
    d.type = "SCRIPTED"
    for symbol, prop in names.items():
        if symbol in expression:
            v = d.variables.new()
            v.name = symbol
            v.type = "SINGLE_PROP"
            v.targets[0].id = root
            v.targets[0].data_path = '["' + prop + '"]'
    d.expression = "(" + expression + ")/1000"


def material(name, color, roughness, coat=0):
    """Create a Principled shader; colors are linear RGB, roughness is unitless."""
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    shader = next(n for n in m.node_tree.nodes if n.type == "BSDF_PRINCIPLED")
    shader.inputs["Base Color"].default_value = (*color, 1)
    shader.inputs["Roughness"].default_value = roughness
    shader.inputs["Coat Weight"].default_value = coat
    shader.inputs["Coat Roughness"].default_value = 0.16
    m.diffuse_color = (*color, 1)
    return m


black = material("Board • glossy black squares", (0.008, 0.009, 0.011), 0.24, 0.02)
white = material("Board • warm white laminate", (0.82, 0.81, 0.77), 0.27, 0.02)
frame = material("Board • molded charcoal frame", (0.008, 0.009, 0.011), 0.18, 0.25)
ink = material("Board • printed dark coordinates", (0.002, 0.002, 0.002), 0.48)
edge = material("Board • inset trim", (0.006, 0.007, 0.009), 0.14, 0.3)

surround = material("Board • gray coordinate surround", (0.38, 0.39, 0.39), 0.32)
slit = material("Board • black slit interior", (0.001, 0.001, 0.001), 0.65)
# Ink and the cavity should not acquire a gray specular veil from the studio.
for matte_material in [ink, slit]:
    shader = next(n for n in matte_material.node_tree.nodes if n.type == "BSDF_PRINCIPLED")
    shader.inputs["Specular IOR Level"].default_value = 0


def box(name, dims, loc, mat):
    """Create a box from the supplied dimensions and location."""
    bpy.ops.mesh.primitive_cube_add(size=1)
    obj = bpy.context.object
    obj.name = name
    obj.parent = root
    for i, e in enumerate(dims):
        drive(obj, "scale", i, e)
    for i, e in enumerate(loc):
        drive(obj, "location", i, e)
    obj.data.materials.append(mat)
    return obj


# Apply driver-controlled dimensions through mesh geometry nodes before bevel,
# preserving physically meaningful bevel radii while dimensions remain editable.
def rounded_box(name, dims, loc, mat, radius):
    """Scale geometry before beveling so the edge radius stays physically sized."""
    obj = box(name, ["1", "1", "1"], loc, mat)
    for i in range(3):
        obj.driver_remove("scale", i)
    obj.scale = (1, 1, 1)
    group = bpy.data.node_groups.new(name + " dimensions", "GeometryNodeTree")
    group.interface.new_socket(
        name="Geometry", in_out="INPUT", socket_type="NodeSocketGeometry"
    )
    group.interface.new_socket(
        name="Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry"
    )
    inp = group.nodes.new("NodeGroupInput")
    out = group.nodes.new("NodeGroupOutput")
    trans = group.nodes.new("GeometryNodeTransform")
    group.links.new(inp.outputs["Geometry"], trans.inputs["Geometry"])
    group.links.new(trans.outputs["Geometry"], out.inputs["Geometry"])
    for i, e in enumerate(dims):
        drive(trans.inputs["Scale"], "default_value", i, e)
    mod = obj.modifiers.new("Editable dimensions", "NODES")
    mod.node_group = group
    bevel = obj.modifiers.new("Rounded molded edges", "BEVEL")
    bevel.segments = 5
    drive(bevel, "width", None, radius)
    obj.modifiers.new("Face normals", "WEIGHTED_NORMAL")
    return obj


for sign in [-1, 1]:
    rounded_box(
        "Case half " + str(sign),
        ["W", "(D-G)/2", "T"],
        ["0", f"{sign}*(D+G)/4", "T/2"],
        frame,
        "C",
    )
    # Printed laminate: flat coplanar surfaces, no artificial bevels around squares.
    box(
        "Laminate half " + str(sign),
        ["W-2*F", "(D-2*F-G)/2", ".3"],
        ["0", f"{sign}*(D-2*F+G)/4", "T-.11"],
        surround,
    )
    # Three ledges rise outwards: the inner shelf is level with the grid.
    # Front/back bars span the width; side bars meet them without overlapping.
    # Each outer tier occupies one third of the total frame width.
    for tier in range(3):
        inset = f"{2-tier}*F/3"
        height = f".06+{tier}*R/2"
        width = "F/3"
        rounded_box(
            f"Frame step {tier} end {sign}",
            [f"W-2*({inset})", width, height],
            ["0", f"{sign}*(D/2-({inset})-F/6)", f"T+({height})/2"],
            edge, ".06",
        )
        for side in [-1, 1]:
            length = f"D/2-G/2-({inset})-F/3"
            rounded_box(
                f"Frame step {tier} side {side} half {sign}",
                [width, length, height],
                [f"{side}*(W/2-({inset})-F/6)",
                 f"{sign}*(G/2+({length})/2)", f"T+({height})/2"],
                edge, ".06",
            )
# A recessed black floor keeps the fold slit dark instead of exposing the table.
box("Fold slit interior", ["W-.4", "G+.1", ".2"],
    ["0", "0", "T+.06-Q-.1"], slit)
squares = []
for rank in range(8):
    for file in range(8):
        y = f"({rank}-3.5)*S"
        sy = "S"
        if rank == 3:
            y += "-G/4"
            sy = "S-G/2"
        if rank == 4:
            y += "+G/4"
            sy = "S-G/2"
        sq = box(
            chr(97 + file) + str(rank + 1),
            ["S", sy, ".02"],
            [f"({file}-3.5)*S", y, "T+.05"],
            white if (rank + file) % 2 else black,
        )
        sq["square"] = chr(97 + file) + str(rank + 1)
        squares.append(sq)
# Fine printed boundary around the checker pattern.
for sign in [-1, 1]:
    box(
        "Grid outline",
        ["8*S+2*O", "O", ".022"],
        ["0", f"{sign}*(4*S+O/2)", "T+.051"],
        ink,
    )
    box(
        "Grid outline",
        ["O", "8*S", ".022"],
        [f"{sign}*(4*S+O/2)", "0", "T+.051"],
        ink,
    )


# Pack a readable sans-serif font into the .blend; use Blender's built-in font
# if regenerating on a machine without this Linux font installation.
font_path = Path("/usr/share/fonts/liberation-sans-fonts/LiberationSans-Regular.ttf")
coordinate_font = bpy.data.fonts.load(str(font_path)) if font_path.exists() else None
if coordinate_font:
    coordinate_font.pack()


def label(text, x, y, angle):
    """Place printed coordinate text using board dimension expressions."""
    c = bpy.data.curves.new("Printed " + text, "FONT")
    c.body = text
    if coordinate_font:
        c.font = coordinate_font
    drive(c, "offset", None, "B")
    c.align_x = "CENTER"
    c.align_y = "CENTER"
    c.extrude = 0
    c.resolution_u = 8
    obj = bpy.data.objects.new("Coordinate " + text, c)
    scene.collection.objects.link(obj)
    obj.parent = root
    drive(c, "size", None, "L")
    drive(obj, "location", 0, x)
    drive(obj, "location", 1, y)
    drive(obj, "location", 2, "T+.075")
    obj.rotation_euler.z = angle
    c.materials.append(ink)


for i in range(8):
    pos = f"({i}-3.5)*S"
    label(chr(97 + i), pos, "-(4*S+D/2-F)/2", 0)
    label(chr(97 + i), pos, "(4*S+D/2-F)/2", math.pi)
    label(str(i + 1), "-(4*S+W/2-F)/2", pos, 0)
    label(str(i + 1), "(4*S+W/2-F)/2", pos, math.pi)
def add_aruco_markers(scene, root, settings):
    """Add independently hideable, vector-sharp marker stickers above the board.

    Patterns include the black border: 1 means black, 0 means white. Their rows
    run down the printed marker, whereas Blender's local +Y runs up the marker.
    No external image textures or OpenCV installation are needed for rendering.
    """
    root['aruco_enabled'] = bool(settings.get('enabled', True))
    root.id_properties_ui('aruco_enabled').update(
        description='Show marker stickers in viewport and renders'
    )
    collection = bpy.data.collections.new('ArUco markers')
    scene.collection.children.link(collection)
    materials = []
    for name, color in [
        ('Marker paper', (0.86, 0.85, 0.81, 1)),
        ('Marker ink', (0.004, 0.004, 0.004, 1)),
    ]:
        material = bpy.data.materials.new(name)
        material.use_nodes = True
        shader = next(
            node for node in material.node_tree.nodes
            if node.type == 'BSDF_PRINCIPLED'
        )
        shader.inputs['Base Color'].default_value = color
        shader.inputs['Roughness'].default_value = 0.65
        material.diffuse_color = color
        materials.append(material)

    def control(obj, path, index, property_name, expression):
        curve = obj.driver_add(path) if index is None else obj.driver_add(path, index)
        variable = curve.driver.variables.new()
        variable.name = 'value'
        variable.type = 'SINGLE_PROP'
        variable.targets[0].id = root
        variable.targets[0].data_path = '["' + property_name + '"]'
        curve.driver.expression = expression

    for marker in settings['markers']:
        rows = marker['pattern_rows']
        assert len(rows) == 6 and all(
            len(row) == 6 and set(row) <= {'0', '1'} for row in rows
        )
        black_size = marker['black_size_mm'] / 1000
        half_paper = black_size / 2 + settings['white_border_mm'] / 1000
        vertices = [
            (-half_paper, -half_paper, 0), (half_paper, -half_paper, 0),
            (half_paper, half_paper, 0), (-half_paper, half_paper, 0),
        ]
        faces = [(0, 1, 2, 3)]
        cell_size = black_size / 6
        for row, bits in enumerate(rows):
            for column, bit in enumerate(bits):
                if bit == '0':
                    continue
                x = -black_size / 2 + column * cell_size
                y = black_size / 2 - (row + 1) * cell_size
                first = len(vertices)
                # One micron separates ink from paper to avoid depth fighting.
                vertices.extend([
                    (x, y, 0.000001), (x + cell_size, y, 0.000001),
                    (x + cell_size, y + cell_size, 0.000001),
                    (x, y + cell_size, 0.000001),
                ])
                faces.append(tuple(range(first, first + 4)))
        mesh = bpy.data.meshes.new('ArUco ' + str(marker['id']) + ' pattern')
        mesh.from_pydata(vertices, [], faces)
        mesh.update()
        for material in materials:
            mesh.materials.append(material)
        for polygon in mesh.polygons:
            polygon.material_index = 0 if polygon.index == 0 else 1
        obj = bpy.data.objects.new('ArUco %02d — %s' % (marker['id'], marker['square']), mesh)
        collection.objects.link(obj)
        obj.parent = root
        obj['aruco_marker'] = True
        obj['aruco_id'] = marker['id']
        obj['square'] = marker['square']
        obj['dictionary'] = settings['dictionary']
        file = ord(marker['square'][0]) - ord('a')
        rank = int(marker['square'][1]) - 1
        dx, dy = marker['offset_mm']
        control(obj, 'location', 0, 'square_mm', f'(({file}-3.5)*value+{dx})/1000')
        control(obj, 'location', 1, 'square_mm', f'(({rank}-3.5)*value+{dy})/1000')
        offset = 0.06 + settings['surface_offset_mm']
        control(obj, 'location', 2, 'thickness_mm', f'(value+{offset})/1000')
        obj.rotation_euler.z = math.radians(marker['rotation_degrees'])
        control(obj, 'hide_render', None, 'aruco_enabled', 'not value')
        control(obj, 'hide_viewport', None, 'aruco_enabled', 'not value')
    return collection


if "aruco" in PARAMS:
    add_aruco_markers(scene, root, PARAMS["aruco"])

# The fold opening has real depth and a recessed black floor.
# A neutral studio keeps the actual board material easy to inspect.
bpy.ops.mesh.primitive_plane_add(size=200, location=(0, 0, -0.0001))
floor = bpy.context.object
floor.name = "Board studio floor"
floor.data.materials.append(
    material("Board studio • warm gray", (0.12, 0.135, 0.15), 0.75)
)


def aim(obj, point):
    """Point a camera or light toward a world-space target using Blender axes."""
    obj.rotation_euler = (
        (Vector(point) - obj.location).to_track_quat("-Z", "Y").to_euler()
    )


for name, loc, power, size in [
    ("Board LED left", (-1.6, 1.4, 2.4), 65, 1.2),
    ("Board LED right", (1.6, 1.4, 2.4), 65, 1.2),
]:
    l = bpy.data.lights.new(name, "AREA")
    l.energy = power
    l.shape = "RECTANGLE"
    l.size_y = 0.15
    l.size = size
    obj = bpy.data.objects.new(name, l)
    scene.collection.objects.link(obj)
    obj.location = loc
    aim(obj, (0, 0, 0))
world = bpy.data.worlds.new("Board studio ambient")
world.use_nodes = True
next(n for n in world.node_tree.nodes if n.type == "BACKGROUND").inputs[
    "Strength"
].default_value = 0.015
scene.world = world
scene.view_settings.exposure = 0.15
data = bpy.data.cameras.new("Board camera")
camera = bpy.data.objects.new("Board camera", data)
scene.collection.objects.link(camera)
camera.location = (0.24, 0.35, 0.39)
aim(camera, (0, 0, 0.006))
data.type = "ORTHO"
data.ortho_scale = 0.365
data.clip_start = 0.001
scene.camera = camera
scene.render.engine = "CYCLES"
scene.cycles.samples = 48
scene.cycles.use_denoising = True
scene.render.resolution_x = 1100
scene.render.resolution_y = 1000
scene.render.resolution_percentage = 100
scene.render.image_settings.file_format = "PNG"
scene.render.filepath = str(OUT / "preview.png")
bpy.context.view_layer.update()
assert len(squares) == 64
assert squares[0].data.materials[0] == black and squares[7].data.materials[0] == white
assert (
    root["square_mm"] * 8
    < min(root["width_mm"], root["depth_mm"]) - 2 * root["frame_mm"]
)


# Check evaluated outer dimensions across the two shells.
def case_bounds():
    """Measure evaluated shell bounds, including geometry nodes and drivers."""
    deps = bpy.context.evaluated_depsgraph_get()
    pts = []
    for obj in scene.objects:
        if obj.name.startswith("Case half"):
            e = obj.evaluated_get(deps)
            pts.extend(e.matrix_world @ Vector(v) for v in e.bound_box)
    return [max(v[i] for v in pts) - min(v[i] for v in pts) for i in range(3)]


assert all(
    abs(a - b) < 1e-6
    for a, b in zip(
        case_bounds(),
        tuple(PARAMS[k] / 1000 for k in ["width_mm", "depth_mm", "thickness_mm"]),
    )
), case_bounds()
root["width_mm"] = PARAMS["width_mm"] + 2
root.update_tag()
bpy.context.view_layer.update()
assert abs(case_bounds()[0] - (PARAMS["width_mm"] + 2) / 1000) < 1e-6
root["width_mm"] = PARAMS["width_mm"]
root.update_tag()
bpy.context.view_layer.update()
bpy.ops.object.select_all(action="DESELECT")
root.select_set(True)
bpy.context.view_layer.objects.active = root
# Set a useful opening view; these UI settings do not affect the mesh.
for screen in bpy.data.screens:
    for area in screen.areas:
        if area.type == "VIEW_3D":
            area.spaces.active.clip_start = 0.001
            area.spaces.active.region_3d.view_perspective = "CAMERA"
        if area.type == "PROPERTIES":
            area.spaces.active.context = "OBJECT"
# Check printing and markers as well as dimensions before saving the asset.
coordinates = [o for o in scene.objects if o.type == "FONT"]
assert len(coordinates) == 32
for obj in coordinates:
    if obj.data.body.isdigit():
        expected = 0 if obj.location.x < 0 else math.pi
        assert abs(obj.rotation_euler.z - expected) < 1e-5
markers = [o for o in scene.objects if o.get("aruco_marker")]
assert len(markers) == len(PARAMS.get("aruco", {}).get("markers", []))
# Save the editable asset after geometry and driver checks have passed.
# Embed the rebuild inputs alongside the editable geometry.
for filename in ["create.py", "dimensions.json", "README.md"]:
    block = bpy.data.texts.get(filename) or bpy.data.texts.new(filename)
    block.clear()
    block.write((OUT / filename).read_text())
bpy.context.preferences.filepaths.save_version = 0
bpy.ops.wm.save_as_mainfile(filepath=str(OUT / "model.blend"))
report = {
    "outer_case_mm": [round(x * 1000, 3) for x in case_bounds()],
    "square_count": len(squares),
    "square_mm": root["square_mm"],
    "grid_width_mm": 8 * root["square_mm"],
    # Planar grey band before subtracting lettering and the narrow folding slit.
    "grey_surround_area_mm2": (
        (root["width_mm"] - 2 * root["frame_mm"])
        * (root["depth_mm"] - 2 * root["frame_mm"])
        - (8 * root["square_mm"] + 2 * root["grid_outline_mm"]) ** 2
    ),
    "frame_steps": 3,
    "frame_width_mm": root["frame_mm"],
    "rim_above_grid_mm": root["rim_height_mm"],
    "slit_depth_mm": root["slit_depth_mm"],
    "coordinate_count": len(coordinates),
    "aruco_count": sum(bool(o.get("aruco_marker")) for o in scene.objects),
    "measured": ["width_mm", "depth_mm"],
}
(OUT / "validation.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report))
