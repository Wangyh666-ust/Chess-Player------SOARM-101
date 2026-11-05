"""Render paired photo-set samples using Blender's bundled Python and numpy.
Usage: .local-blender/start-blender.sh --background --python THIS_FILE -- --config CONFIG
"""

import argparse
import hashlib
import json
import math
import random
import struct
import sys
import zlib
from pathlib import Path
import bpy
import numpy as np
from mathutils import Vector, Matrix

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from reference_style import wood_material, set_lights, add_props, surface_finish
from randomized_lighting import sample_lighting

NAMES = [
    "white-pawn",
    "white-rook",
    "white-knight",
    "white-bishop",
    "white-queen",
    "white-king",
    "black-pawn",
    "black-rook",
    "black-knight",
    "black-bishop",
    "black-queen",
    "black-king",
    "empty",
]
TYPES = dict(zip("PRNBQK", ["pawn", "rook", "knight", "bishop", "queen", "king"]))


def parse_placement(fen, require_kings=True):
    """Read FEN placement as file/rank/symbol tuples; this is not a legality check."""
    # FEN lists rank 8 first; the renderer uses rank 1 at Y = 0 in board indices.
    rows = fen.split()[0].split("/")
    assert len(rows) == 8
    pieces = []
    for row, text in enumerate(rows):
        file = 0
        for symbol in text:
            if symbol.isdigit():
                assert symbol in "12345678"
                file += int(symbol)
            else:
                assert symbol.upper() in TYPES and file < 8
                pieces.append((file, 7 - row, symbol))
                file += 1
        assert file == 8
    assert not require_kings or (
        sum(s == "K" for _, _, s in pieces) == 1
        and sum(s == "k" for _, _, s in pieces) == 1
    )
    return pieces


def png_gray(path, array):
    """Write lossless 8-bit instance IDs without requiring Pillow inside Blender."""
    h, w = array.shape

    def chunk(kind, data):
        """Encode a PNG chunk with its length and checksum."""
        return (
            struct.pack("!I", len(data))
            + kind
            + data
            + struct.pack("!I", zlib.crc32(kind + data) & 0xFFFFFFFF)
        )

    data = b"".join(b"\x00" + row.tobytes() for row in array.astype(np.uint8))
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack("!2I5B", w, h, 8, 0, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(data))
        + chunk(b"IEND", b"")
    )


def material(name, color, roughness):
    """Create a Principled shader; colors are linear RGB, roughness is unitless."""
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    m.diffuse_color = (*color, 1)
    shader = next(n for n in m.node_tree.nodes if n.type == "BSDF_PRINCIPLED")
    shader.inputs["Base Color"].default_value = (*color, 1)
    shader.inputs["Roughness"].default_value = roughness
    shader.inputs["IOR"].default_value = 1.46
    return m


def load_assets(config, scene):
    """Bake evaluated source geometry; preserve board materials and replace piece materials."""
    loaded = {}
    meshes = {}
    points = {}
    provenance = {}
    for kind, spec in config["assets"].items():
        path = ROOT / spec["file"]
        if spec["file"] not in provenance:
            provenance[spec["file"]] = hashlib.sha256(path.read_bytes()).hexdigest()
        key = (str(path), spec["scene"])
        if key not in loaded:
            with bpy.data.libraries.load(str(path), link=False) as (src, dst):
                assert spec["scene"] in src.scenes, (spec["scene"], src.scenes)
                dst.scenes = [spec["scene"]]
            loaded[key] = dst.scenes[0]
        source = loaded[key]
        bpy.context.window.scene = source
        source.frame_set(source.frame_current)
        bpy.context.view_layer.update()
        obj = source.objects.get(spec["object"])
        assert obj is not None, (kind, spec["object"])
        if kind == "board":
            board_dims = {
                k: obj[k] for k in ["width_mm", "depth_mm", "square_mm", "thickness_mm"]
            }
            # Collection render switches must survive the bake into the dataset scene.
            render_visible = set()

            def collect_visible(collection, hidden=False):
                hidden = hidden or collection.hide_render
                if not hidden:
                    render_visible.update(collection.objects)
                for nested in collection.children:
                    collect_visible(nested, hidden)

            collect_visible(source.collection)
            for child in obj.children_recursive:
                if child.hide_render or child not in render_visible:
                    continue
                if child.get("aruco_marker") and not config.get("aruco_enabled", True):
                    continue
                if child.type not in {"MESH", "FONT", "CURVE"}:
                    continue
                evaluated = child.evaluated_get(bpy.context.evaluated_depsgraph_get())
                matrix = evaluated.matrix_world.copy()
                mesh_data = bpy.data.meshes.new_from_object(
                    evaluated, depsgraph=bpy.context.evaluated_depsgraph_get()
                )
                mesh_data.transform(matrix)
                copy = bpy.data.objects.new("Board " + child.name, mesh_data)
                scene.collection.objects.link(copy)
        else:
            evaluated = obj.evaluated_get(bpy.context.evaluated_depsgraph_get())
            mesh_data = bpy.data.meshes.new_from_object(
                evaluated, depsgraph=bpy.context.evaluated_depsgraph_get()
            )
            # Bake local scale (including height driver), discard display translation/yaw.
            for v in mesh_data.vertices:
                v.co.x *= obj.scale.x
                v.co.y *= obj.scale.y
                v.co.z *= obj.scale.z
            mesh_data.materials.clear()
            meshes[kind] = mesh_data
            xyz = np.empty(len(mesh_data.vertices) * 3, dtype=np.float32)
            mesh_data.vertices.foreach_get("co", xyz)
            points[kind] = np.column_stack(
                [xyz.reshape(-1, 3), np.ones(len(mesh_data.vertices))]
            )
    bpy.context.window.scene = scene
    bpy.context.view_layer.update()
    assert all(
        max(o.dimensions) < 0.3 for o in scene.objects if o.type == "MESH"
    ), "Invalid board transform after asset import"
    # Drop imported studio scenes, keeping the copied geometry only.
    for source in loaded.values():
        objects = list(source.objects)
        bpy.data.scenes.remove(source)
        for obj in objects:
            if obj.users == 0:
                bpy.data.objects.remove(obj)
    return meshes, points, board_dims, provenance


def project(points, matrix, width, height):
    """Project homogeneous vertices to pixels with a top-left image origin."""
    clip_coordinates = points @ matrix.T
    assert np.all(
        clip_coordinates[:, 3] > 0
    ), "Geometry behind camera; adjust configured pose"
    # Perspective division maps clip coordinates to [-1, 1]; flip Y for image rows.
    normalized_device_coordinates = (
        clip_coordinates[:, :3] / clip_coordinates[:, 3, None]
    )
    return np.column_stack(
        [
            (normalized_device_coordinates[:, 0] + 1) * width / 2,
            (1 - normalized_device_coordinates[:, 1]) * height / 2,
        ]
    )


def create_environment(config, scene):
    """Create the tabletop, optional props, softboxes, and ambient world."""
    bpy.ops.mesh.primitive_plane_add(size=2)
    table = bpy.context.object
    table.name = "Randomized tabletop"
    mat = material("Table material", (0.2, 0.1, 0.05), 0.65)
    table.data.materials.append(mat)
    shader = next(n for n in mat.node_tree.nodes if n.type == "BSDF_PRINCIPLED")
    noise = mat.node_tree.nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 22
    bump = mat.node_tree.nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.16
    bump.inputs["Distance"].default_value = 0.0005
    mat.node_tree.links.new(noise.outputs["Fac"], bump.inputs["Height"])
    mat.node_tree.links.new(bump.outputs["Normal"], shader.inputs["Normal"])
    if "wood" in config:
        wood_material(mat, config["wood"])
    if "props" in config:
        add_props(config["props"])
    lights = []
    light_count = max([len(config.get("lights", [0, 1]))] + [
        len(group.get("lighting_setup", {}).get("lights", []))
        for group in config["positions"]])
    for i in range(light_count):
        data = bpy.data.lights.new("Softbox " + str(i), "AREA")
        obj = bpy.data.objects.new(data.name, data)
        scene.collection.objects.link(obj)
        lights.append(obj)
    scene.world = bpy.data.worlds.new("Randomized ambient")
    scene.world.use_nodes = True
    background_shader = next(
        n for n in scene.world.node_tree.nodes if n.type == "BACKGROUND"
    )
    if "world_color" in config:
        background_shader.inputs["Color"].default_value = (*config["world_color"], 1)
    return shader, lights, background_shader


def create_cameras(config, scene):
    """Create cameras; a supplied matrix overrides the location/target pose."""
    cameras = {}
    for name, spec in config["cameras"].items():
        data = bpy.data.cameras.new(name + " camera")
        obj = bpy.data.objects.new(name + " camera", data)
        scene.collection.objects.link(obj)
        data.type = "PERSP"
        data.sensor_fit = "HORIZONTAL"
        data.clip_start = 0.001
        data.clip_end = 100
        data.lens = spec["lens_mm"]
        data.sensor_width = spec["sensor_width_mm"]
        obj.location = spec["location_m"]
        obj.rotation_euler = (
            (Vector(spec["target_m"]) - obj.location)
            .to_track_quat("-Z", "Y")
            .to_euler()
        )
        if "matrix_world" in spec:
            obj.matrix_world = Matrix(spec["matrix_world"])
        cameras[name] = obj
    scene.camera = next(iter(cameras.values()))
    return cameras


def configure_render_passes(config, scene, output_dir):
    """Configure Cycles RGB output and an unfiltered object-index EXR pass."""
    width, height = config["resolution"]
    scene.render.resolution_x = width
    scene.render.resolution_y = height
    scene.render.resolution_percentage = 100
    scene.render.engine = "CYCLES"
    scene.cycles.samples = config["samples"]
    scene.cycles.use_denoising = True
    # Camera response matters when comparing a clipped webcam image with a render.
    scene.view_settings.view_transform = config.get("view_transform", "AgX")
    scene.view_settings.look = config.get("view_look", "None")
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    # RGB is for training; IndexOB preserves exact instance IDs for visibility checks.
    scene.view_layers[0].use_pass_object_index = True
    scene.use_nodes = True
    nodes = scene.node_tree.nodes
    nodes.clear()
    render_layers = nodes.new("CompositorNodeRLayers")
    render_layers.scene = scene
    output = nodes.new("CompositorNodeOutputFile")
    output.base_path = str(output_dir / "_passes")
    output.format.file_format = "OPEN_EXR"
    output.format.color_mode = "RGB"
    output.format.color_depth = "32"
    scene.node_tree.links.new(render_layers.outputs["IndexOB"], output.inputs[0])
    composite = nodes.new("CompositorNodeComposite")
    scene.node_tree.links.new(render_layers.outputs["Image"], composite.inputs[0])
    scene.frame_set(1)
    return output


def build_annotations(
    config, piece_objects, instances, local_points, matrix, width, height, mask
):
    """Build full projected boxes and visible-mask metadata for one camera view."""
    records = []
    lines = []
    # Amodal boxes include occluded geometry. The mask decides whether enough of
    # the piece is visible to label it, and also provides its smaller visible box.
    for obj, instance in zip(piece_objects, instances):
        xy = project(
            local_points[instance["kind"]],
            matrix @ np.array(obj.matrix_world),
            width,
            height,
        )
        low = np.maximum(xy.min(axis=0), [0, 0])
        high = np.minimum(xy.max(axis=0), [width, height])
        amodal = [*low, *high]
        ys, xs = np.where(mask == instance["instance_id"])
        visible = len(xs)
        visible_box = (
            [int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)]
            if visible
            else None
        )
        labelled = visible >= config["min_visible_pixels"] and bool(np.all(high > low))
        if labelled:
            box = [
                *(low + high) / 2 / np.array([width, height]),
                *(high - low) / np.array([width, height]),
            ]
            assert all(0 <= v <= 1 for v in box)
            lines.append(
                str(instance["class_id"]) + " " + " ".join(f"{v:.8f}" for v in box)
            )
        records.append(
            {
                **instance,
                "amodal_box": list(map(float, amodal)),
                "visible_box": visible_box,
                "visible_pixels": visible,
                "labelled": labelled,
                "world_matrix": np.array(obj.matrix_world).tolist(),
                "base_image_point": project(
                    np.array([[0, 0, 0, 1]]),
                    matrix @ np.array(obj.matrix_world),
                    width,
                    height,
                )[0].tolist(),
            }
        )
    return records, lines


def main():
    """Load one preset and generate paired RGB images, masks, boxes, and metadata."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default=str(Path(__file__).parent / "configs/scene.json")
    )
    args = parser.parse_args(
        sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    )
    config = json.loads(Path(args.config).read_text())
    if "lighting_randomization" in config:
        # Validate bounds before creating output folders or loading heavy assets.
        sample_lighting(config.get("lights"), config["world_strength"],
                        seed=config["seed"], bounds=config["lighting_randomization"])
    output_dir = (ROOT / config["output"]).resolve()
    assert output_dir.is_relative_to(ROOT), "Output must be inside this project"
    # Fail before any overwrite; a new config/output folder creates a new dataset version.
    output_dir.mkdir(parents=True, exist_ok=False)
    groups = config["positions"]
    seen = {}
    source_placements = {}
    for g in groups:
        assert g["split"] in ["train", "val", "test"]
        parse_placement(g["fen"], config.get("require_kings", True))
        placement = g["fen"].split()[0]
        source_id = g.get("source_setup_id", g["id"])
        owner = (source_id, g["split"])
        assert placement not in seen or seen[placement] == owner, "Placement crosses source setups or splits"
        assert source_id not in source_placements or source_placements[source_id] == placement
        seen[placement] = owner
        source_placements[source_id] = placement
    assert len({g["id"] for g in groups}) == len(groups)
    (output_dir / "config.json").write_text(json.dumps(config, indent=2))
    for folder in ["images", "labels", "masks", "records"]:
        for split in ["train", "val", "test"]:
            (output_dir / folder / split).mkdir(parents=True)
    (output_dir / "_passes").mkdir()
    scene = bpy.data.scenes.new("Photo set — synthetic data pilot")
    bpy.context.window.scene = scene
    scene.unit_settings.system = "METRIC"
    scene.unit_settings.length_unit = "MILLIMETERS"
    meshes, local_points, board_dimensions, provenance = load_assets(config, scene)
    white = material("Synthetic ivory plastic", (0.81, 0.70, 0.47), 0.24)
    black = material("Synthetic black plastic", (0.009, 0.008, 0.007), 0.23)
    # Finish overrides affect generated scenes only, leaving source assets editable.
    finishes = config.get("surface_finishes", {})
    surface_finish(white, finishes.get("pieces", {}))
    surface_finish(black, finishes.get("pieces", {}))
    surface_finish(white, finishes.get("white_pieces", {}))
    surface_finish(black, finishes.get("black_pieces", {}))
    board_materials = {
        slot.material for obj in scene.objects for slot in obj.material_slots
        if slot.material and slot.material.name.startswith("Board •")
    }
    for board_material in board_materials:
        name = board_material.name
        if name.startswith(("Board • molded charcoal", "Board • inset trim")):
            surface_finish(board_material, finishes.get("frame", {}))
        elif name.startswith(("Board • printed dark", "Board • black slit")):
            pass  # Printed ink and the recessed slit retain their own finish.
        else:
            surface_finish(board_material, finishes.get("board", {}))
        if board_material.name.startswith("Board • warm white laminate"):
            surface_finish(board_material, finishes.get("white_squares", {}))
        elif board_material.name.startswith("Board • glossy black squares"):
            surface_finish(board_material, finishes.get("black_squares", {}))
    shader, lights, background_shader = create_environment(config, scene)
    cameras = create_cameras(config, scene)
    output = configure_render_passes(config, scene, output_dir)
    piece_objects = []
    manifest = []
    # The board is centered at the origin. The printed surface sits 0.06 mm
    # above the case, matching the board asset generator.
    square_size_m = board_dimensions["square_mm"] / 1000
    board_surface_m = (board_dimensions["thickness_mm"] + 0.06) / 1000
    corner_xyz = np.array(
        [
            [-4 * square_size_m, -4 * square_size_m, board_surface_m, 1],
            [4 * square_size_m, -4 * square_size_m, board_surface_m, 1],
            [4 * square_size_m, 4 * square_size_m, board_surface_m, 1],
            [-4 * square_size_m, 4 * square_size_m, board_surface_m, 1],
        ]
    )
    for group_index, group in enumerate(groups):
        # One repeatable scene per FEN; both cameras share pieces and lighting.
        rng = random.Random(config["seed"] + group_index * 100003)
        for obj in piece_objects:
            bpy.data.objects.remove(obj, do_unlink=True)
        piece_objects = []
        instances = []
        for index, (file, rank, symbol) in enumerate(
            parse_placement(group["fen"], config.get("require_kings", True)), 1
        ):
            kind = TYPES[symbol.upper()]
            color = "white" if symbol.isupper() else "black"
            obj = bpy.data.objects.new(
                f"{color} {kind} {chr(97+file)}{rank+1}", meshes[kind]
            )
            scene.collection.objects.link(obj)
            # Object-linked material slots allow all instances to share one mesh.
            if not obj.data.materials:
                obj.data.materials.append(white)
            obj.material_slots[0].link = "OBJECT"
            obj.material_slots[0].material = white if symbol.isupper() else black
            jitter = config["piece_jitter_mm"] / 1000
            obj.location = (
                (file - 3.5) * square_size_m + rng.uniform(-jitter, jitter),
                (rank - 3.5) * square_size_m + rng.uniform(-jitter, jitter),
                board_surface_m,
            )
            obj.rotation_euler.z = math.radians(
                (90 if symbol.isupper() else -90)
                + rng.uniform(
                    -config["piece_yaw_jitter_degrees"],
                    config["piece_yaw_jitter_degrees"],
                )
            )
            # Optional photo-matching adjustments, measured relative to square centers.
            square = chr(97 + file) + str(rank + 1)
            adjustment = group.get("piece_transforms", {}).get(square, {})
            offset = adjustment.get("offset_mm", [0, 0])
            obj.location.x += offset[0] / 1000
            obj.location.y += offset[1] / 1000
            if "yaw_degrees" in adjustment:
                obj.rotation_euler.z = math.radians(adjustment["yaw_degrees"])
            obj.pass_index = index
            piece_objects.append(obj)
            instances.append(
                {
                    "instance_id": index,
                    "square": chr(97 + file) + str(rank + 1),
                    "symbol": symbol,
                    "class_id": NAMES.index(color + "-" + kind),
                    "kind": kind,
                }
            )
        table_color = rng.choice(config["table_colors"])
        shader.inputs["Base Color"].default_value = (*table_color, 1)
        shader.inputs["Roughness"].default_value = (
            config["wood"]["roughness"] if "wood" in config else rng.uniform(0.4, 0.8)
        )
        for light in lights:
            light.location = (
                rng.uniform(-0.4, 0.4),
                rng.uniform(-0.4, 0.4),
                rng.uniform(0.4, 0.65),
            )
            light.rotation_euler = (
                (Vector((0, 0, 0.02)) - light.location)
                .to_track_quat("-Z", "Y")
                .to_euler()
            )
            light.data.energy = rng.uniform(*config["light_power_w"])
            light.data.size = rng.uniform(*config["light_size_m"])
            light.data.color = (1, rng.uniform(0.85, 1), rng.uniform(0.7, 1))
        # Keep random draws in the same order even when explicit lights override them.
        if "lights" in config:
            set_lights(lights, config["lights"])
        background_shader.inputs["Strength"].default_value = rng.uniform(
            *config["world_strength"]
        )
        scene.view_settings.exposure = rng.uniform(*config["exposure"])
        sampled_lighting = None
        if "lighting_setup" in group:
            # The dataset planner sampled this once; both cameras reuse it exactly.
            sampled_lighting = group["lighting_setup"]
            set_lights(lights, sampled_lighting["lights"])
            background_shader.inputs["Strength"].default_value = sampled_lighting["world_strength"]
            background_shader.inputs["Color"].default_value = (*sampled_lighting["world_color"], 1)
            scene.view_settings.exposure = sampled_lighting["exposure"]
        elif "lighting_randomization" in config:
            # A separate stream keeps lighting independent of piece/camera RNG draws.
            # Sample once per position, before the camera loop, for paired views.
            sampled_lighting = sample_lighting(
                config["lights"], config["world_strength"],
                seed=config["seed"] + group_index * 100003,
                bounds=config["lighting_randomization"],
            )
            set_lights(lights, sampled_lighting["lights"])
            background_shader.inputs["Strength"].default_value = sampled_lighting["world_strength"]
        for view, spec in group.get("camera_setup", config["cameras"]).items():
            camera = cameras[view]
            camera_data = camera.data
            scene.camera = camera
            width, height = spec.get("resolution", config["resolution"])
            scene.render.resolution_x = width
            scene.render.resolution_y = height
            sample_id = group["id"] + "_" + view
            split = group["split"]
            jitter = config["camera_position_jitter_mm"] / 1000
            camera.location = Vector(spec["location_m"]) + Vector(
                [rng.uniform(-jitter, jitter) for _ in range(3)]
            )
            t = config["camera_target_jitter_mm"] / 1000
            target = Vector(spec["target_m"]) + Vector(
                [rng.uniform(-t, t) for _ in range(3)]
            )
            camera.rotation_euler = (
                (target - camera.location).to_track_quat("-Z", "Y").to_euler()
            )
            # A calibrated matrix also replaces translation; target jitter is ignored.
            if "matrix_world" in spec:
                camera.matrix_world = Matrix(spec["matrix_world"])
                camera.location += Vector(
                    [rng.uniform(-jitter, jitter) for _ in range(3)]
                )
            camera_data.lens = spec["lens_mm"]
            camera_data.sensor_width = spec["sensor_width_mm"]
            # Off-center principal points preserve a fitted camera when cropping a photo.
            camera_data.shift_x = spec.get("shift_x", 0)
            camera_data.shift_y = spec.get("shift_y", 0)
            bpy.context.view_layer.update()
            projection = np.array(
                camera.calc_matrix_camera(
                    bpy.context.evaluated_depsgraph_get(), x=width, y=height
                )
            )
            world_to_camera = np.array(camera.matrix_world.inverted())
            matrix = projection @ world_to_camera
            corners = project(corner_xyz, matrix, width, height)
            if "framing_points_m" in spec:
                # Check the planner's conservative volume with Blender's actual camera.
                frame = np.column_stack((spec["framing_points_m"], np.ones(8)))
                clip = frame @ matrix.T
                pixels = project(frame, matrix, width, height)
                margin = spec["framing_margin_fraction"]
                assert np.all(clip[:, 3] > 0) and np.all(
                    (pixels >= np.array([width, height]) * margin)
                    & (pixels <= np.array([width, height]) * (1 - margin))
                ), sample_id + " randomized camera crops the board/piece volume"
            assert config.get("allow_cropped_board", False) or np.all(
                (corners[:, 0] > 2)
                & (corners[:, 0] < width - 2)
                & (corners[:, 1] > 2)
                & (corners[:, 1] < height - 2)
            ), (sample_id + " board cropped")
            scene.render.filepath = str(
                output_dir / "images" / split / (sample_id + ".png")
            )
            output.file_slots[0].path = sample_id + "_"
            bpy.ops.render.render(write_still=True, scene=scene.name)
            exr = output_dir / "_passes" / (sample_id + "_0001.exr")
            assert exr.exists(), exr
            img = bpy.data.images.load(str(exr), check_existing=False)
            pixels = np.empty(width * height * 4, dtype=np.float32)
            img.pixels.foreach_get(pixels)
            # Blender stores pixels bottom-up; PNG masks and boxes use top-left origin.
            mask = np.rint(pixels.reshape(height, width, 4)[::-1, :, 0]).astype(
                np.uint8
            )
            bpy.data.images.remove(img)
            assert mask.max() <= len(instances) and np.count_nonzero(mask) > 0
            png_gray(output_dir / "masks" / split / (sample_id + ".png"), mask)
            # The saved lossless PNG mask replaces the temporary object-index EXR.
            exr.unlink()
            records, lines = build_annotations(
                config,
                piece_objects,
                instances,
                local_points,
                matrix,
                width,
                height,
                mask,
            )
            (output_dir / "labels" / split / (sample_id + ".txt")).write_text(
                "\n".join(lines) + "\n"
            )
            record = {
                "id": sample_id,
                "group": group["id"],
                "source_setup_id": group.get("source_setup_id", group["id"]),
                "variant_index": group.get("variant_index", 0),
                "split": split,
                "view": view,
                "fen": group["fen"],
                "width": width,
                "height": height,
                "instances": records,
                "corners_order": ["a1 outer", "h1 outer", "h8 outer", "a8 outer"],
                "corners": corners.tolist(),
                "camera_to_world": np.array(camera.matrix_world).tolist(),
                "projection_matrix": projection.tolist(),
                "lens_mm": camera_data.lens,
                "sensor_width_mm": camera_data.sensor_width,
                "sampled_camera": spec.get("sampling"),
                "framing_points_m": spec.get("framing_points_m"),
                "framing_margin_fraction": spec.get("framing_margin_fraction"),
                "pixel_origin": "top-left",
                "camera_local_axes": "+X right, +Y up, -Z forward",
                "table_color": table_color,
                "wood": config.get("wood"),
                "surface_finishes": config.get("surface_finishes", {}),
                "lighting": [
                    {
                        "location": list(l.location),
                        "rotation_euler": list(l.rotation_euler),
                        "energy": l.data.energy,
                        "size": l.data.size,
                        "shape": l.data.shape,
                        "size_y": l.data.size_y,
                        "color": list(l.data.color),
                    }
                    for l in lights
                ],
                "world_strength": background_shader.inputs["Strength"].default_value,
                "world_color": list(background_shader.inputs["Color"].default_value)[:3],
                "sampled_lighting": sampled_lighting,
                "exposure": scene.view_settings.exposure,
                "view_transform": scene.view_settings.view_transform,
                "view_look": scene.view_settings.look,
            }
            (output_dir / "records" / split / (sample_id + ".json")).write_text(
                json.dumps(record, indent=2)
            )
            manifest.append(
                {
                    "id": sample_id,
                    "split": split,
                    "group": group["id"],
                    "view": view,
                    "labels": len(lines),
                }
            )
            print("SAMPLE_DONE", sample_id, len(lines), flush=True)
            if group_index == 0 and view == next(iter(config["cameras"])):
                # An inspectable assembled scene without any change to the source assets.
                bpy.ops.wm.save_as_mainfile(
                    filepath=str(output_dir / "scene.blend"), compress=True
                )
    dataset = (
        "path: "
        + json.dumps(str(output_dir))
        + "\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n"
        + "".join(f"  {i}: {name}\n" for i, name in enumerate(NAMES))
    )
    (output_dir / "dataset.yaml").write_text(dataset)
    (output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "samples": manifest,
                "asset_sha256": provenance,
                "board_dimensions_mm": board_dimensions,
                "label_policy": "projected complete mesh boxes; include only instances with >= min_visible_pixels in object-index pass",
                "scope": "Plumbing pilot, not a performance benchmark; camera poses approximate; no lens distortion; ArUco dimensions approximate",
            },
            indent=2,
        )
    )
    (output_dir / "_passes").rmdir()
    print("DATASET_COMPLETE", output_dir, flush=True)


if __name__ == "__main__":
    main()
