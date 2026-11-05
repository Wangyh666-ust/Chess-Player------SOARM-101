"""Editable procedural table and distant softboxes for the branch photo references."""

import bpy
import math
from mathutils import Vector, Quaternion


def wood_material(material, settings):
    """Build broad grain and fine fibers from world-space procedural noise."""
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    shader = next(n for n in nodes if n.type == "BSDF_PRINCIPLED")
    # World-space coordinates keep the grain scale consistent across table sizes.
    geometry = nodes.new("ShaderNodeNewGeometry")
    warp = nodes.new("ShaderNodeTexNoise")
    warp.inputs["Scale"].default_value = 6
    warp.inputs["Detail"].default_value = 2
    links.new(geometry.outputs["Position"], warp.inputs["Vector"])
    amplitude = nodes.new("ShaderNodeVectorMath")
    amplitude.operation = "MULTIPLY"
    amplitude.inputs[1].default_value = (0, settings.get("grain_warp_m", 0.04), 0)
    links.new(warp.outputs["Color"], amplitude.inputs[0])
    warped = nodes.new("ShaderNodeVectorMath")
    warped.operation = "ADD"
    links.new(geometry.outputs["Position"], warped.inputs[0])
    links.new(amplitude.outputs[0], warped.inputs[1])
    # Combine stretched coarse noise with much finer fibers to suggest wood grain.
    textures = []
    for name, stretch in [
        ("Long wood grain", settings["grain_scale"]),
        ("Fine fibers", settings["fiber_scale"]),
    ]:
        scale = nodes.new("ShaderNodeVectorMath")
        scale.operation = "MULTIPLY"
        scale.inputs[1].default_value = stretch
        links.new(warped.outputs[0], scale.inputs[0])
        noise = nodes.new("ShaderNodeTexNoise")
        noise.label = name
        noise.inputs["Scale"].default_value = 1
        noise.inputs["Detail"].default_value = 4
        noise.inputs["Roughness"].default_value = 0.7
        links.new(scale.outputs[0], noise.inputs["Vector"])
        textures.append(noise)
    mix = nodes.new("ShaderNodeMixRGB")
    mix.blend_type = "MIX"
    mix.inputs[0].default_value = 0.17
    links.new(textures[0].outputs["Fac"], mix.inputs[1])
    links.new(textures[1].outputs["Fac"], mix.inputs[2])
    ramp = nodes.new("ShaderNodeValToRGB")
    ramp.label = "Honey walnut laminate"
    stops = settings["colors"]
    ramp.color_ramp.elements.remove(ramp.color_ramp.elements[1])
    for i, (position, color) in enumerate(stops):
        element = (
            ramp.color_ramp.elements[0]
            if i == 0
            else ramp.color_ramp.elements.new(position)
        )
        element.position = position
        element.color = (*color, 1)
    links.new(mix.outputs[0], ramp.inputs["Fac"])
    links.new(ramp.outputs["Color"], shader.inputs["Base Color"])
    shader.inputs["Roughness"].default_value = settings["roughness"]
    shader.inputs["Coat Weight"].default_value = settings.get("coat", 0.12)
    shader.inputs["Coat Roughness"].default_value = settings.get("coat_roughness", 0.3)
    shader.inputs["Specular IOR Level"].default_value = settings.get("specular", 0.5)
    # Bump changes surface normals only; it does not displace the tabletop mesh.
    bump = nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.12
    bump.inputs["Distance"].default_value = 0.00008
    links.new(textures[1].outputs["Fac"], bump.inputs["Height"])
    links.new(bump.outputs[0], shader.inputs["Normal"])


def set_lights(lights, specs):
    """Apply diffuser shape and dimensions; power is emitted power, not wall draw."""
    assert len(lights) >= len(specs)
    # Reuse the light pool when presets contain different numbers of fixtures.
    for light in lights[len(specs):]:
        light.data.energy = 0
    for spec in specs:
        assert spec.get("shape", "SQUARE") in {"SQUARE", "RECTANGLE", "DISK", "ELLIPSE"}
        assert spec["size_m"] > 0 and spec.get("size_y_m", spec["size_m"]) > 0
        assert spec["power_w"] >= 0
    for light, spec in zip(lights, specs):
        light.name = spec.get("name", light.name)
        light.location = spec["location_m"]
        direction = Vector(spec.get("target_m", [0, 0, 0])) - light.location
        # Rotate the entire diffuser, including a strip's long axis. Re-aiming
        # alone would leave downward-facing rectangular lights aligned to world Y.
        rotation = Quaternion((0, 0, 1), math.radians(spec.get("rig_rotation_degrees", 0)))
        original_direction = rotation.inverted() @ direction
        light.rotation_euler = (rotation @ original_direction.to_track_quat("-Z", "Y")).to_euler()
        light.data.energy = spec["power_w"]
        light.data.shape = spec.get("shape", "SQUARE")
        light.data.size = spec["size_m"]
        if light.data.shape in {"RECTANGLE", "ELLIPSE"}:
            light.data.size_y = spec.get("size_y_m", spec["size_m"])
        light.data.color = spec["color"]


def add_props(settings):
    """Create the approximate stand disk; props are background, not labelled pieces."""
    material = bpy.data.materials.new("Reference matte black accessories")
    material.use_nodes = True
    shader = next(n for n in material.node_tree.nodes if n.type == "BSDF_PRINCIPLED")
    shader.inputs["Base Color"].default_value = (0.008, 0.01, 0.011, 1)
    shader.inputs["Roughness"].default_value = 0.65
    shader.inputs["Specular IOR Level"].default_value = 0.2
    stand = settings["stand"]
    bpy.ops.mesh.primitive_cylinder_add(
        vertices=96,
        radius=stand["radius_m"],
        depth=stand["height_m"],
        location=stand["location_m"],
    )
    obj = bpy.context.object
    obj.name = "Approximate camera stand base"
    obj.data.materials.append(material)
    bevel = obj.modifiers.new("Rounded rim", "BEVEL")
    bevel.width = 0.002
    bevel.segments = 3


def surface_finish(material, settings):
    """Override finish and optional base color, preserving existing texture links.

    Roughness broadens reflections; coat adds a separate reflective layer.
    Optional noise varies roughness in world-space metres, not mesh dimensions.
    """
    shader = next(n for n in material.node_tree.nodes if n.type == "BSDF_PRINCIPLED")
    for key, socket in [("roughness", "Roughness"), ("coat", "Coat Weight"),
                        ("coat_roughness", "Coat Roughness"),
                        ("specular", "Specular IOR Level")]:
        if key in settings:
            assert 0 <= settings[key] <= 1, key
            shader.inputs[socket].default_value = settings[key]
    if "base_color" in settings:
        color = settings["base_color"]
        assert len(color) == 3 and all(0 <= value <= 1 for value in color)
        shader.inputs["Base Color"].default_value = (*color, 1)
        material.diffuse_color = (*color, 1)
    variation = settings.get("roughness_variation", 0)
    if variation:
        roughness = settings["roughness"]
        assert 0 <= variation <= min(roughness, 1 - roughness)
        nodes, links = material.node_tree.nodes, material.node_tree.links
        geometry = nodes.new("ShaderNodeNewGeometry")
        noise = nodes.new("ShaderNodeTexNoise")
        noise.inputs["Scale"].default_value = settings.get("texture_scale_per_m", 600)
        ramp = nodes.new("ShaderNodeMapRange")
        ramp.inputs["To Min"].default_value = roughness - variation
        ramp.inputs["To Max"].default_value = roughness + variation
        links.new(geometry.outputs["Position"], noise.inputs["Vector"])
        links.new(noise.outputs["Fac"], ramp.inputs["Value"])
        links.new(ramp.outputs["Result"], shader.inputs["Roughness"])
