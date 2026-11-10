# Sources and third-party components

## Starting detector and recognition data

- Starting YOLOv8m checkpoint: [alpkirisci/chess_detection_computer_vision](https://github.com/alpkirisci/chess_detection_computer_vision), revision d450e86fac94e7b6fa9c0e4fe156517995767095, models/piece_best.pt.
- Upstream describes this detector as trained on ChessReD2K. The synthetic Dubrovnik fine-tuning dataset is a separate dataset created for this experiment.
- ChessReD citation: Athanasios Masouris and Jan van Gemert, *End-to-End Chess Recognition*, VISAPP 2024, pp. 393–403, DOI [10.5220/0012370200003660](https://doi.org/10.5220/0012370200003660).

The portable known-corner mapping helper implements the same bottom-strip geometry used for the comparison, with outputs checked against stored predictions. Upstream inference source and its corner model are not bundled.

The inspected upstream snapshot did not include a standalone license file. This repository does not grant additional rights to the upstream checkpoint, original recognition data, or external mesh assets.

## Historical vision / October piece geometry

The supplied STL asset set is named ezeeone_Dubrovnik_1960_set_02_stls. Mesh identity is recorded by filename and SHA-256 in the split plan. Raw meshes are external inputs and are not redistributed in this repository. Rendered preview images and experiment outputs were produced for this project.

## Dependencies

Model training/inference uses [Ultralytics](https://github.com/ultralytics/ultralytics), with its upstream AGPL-3.0 and enterprise licensing options. Rendering and processing use VTK, NumPy, OpenCV, Pillow, python-chess, SciPy, matplotlib, PyYAML, and PyTorch. Each dependency retains its own license; consult its source distribution for terms.

No external font, JavaScript framework, or CDN is needed for the demo.

## Preliminary robot simulation

The October recorded motion experiment uses PyBullet, OMPL, SciPy, NumPy, python-chess, Pillow, OpenCV, and imageio-ffmpeg. The video depicts the project's SO-ARM101 and Dubrovnik geometry in a nominal virtual installation. That historical results update included rendered video, selected recorded poses and audits. Its ideal fixed grasp constrains both translation and rotation.

## November control handoff assets

The current `simulation/runtime` snapshot additionally includes the project's current robot URDF/STL assets and generated scanned-piece OBJ/NPZ/URDF and convex collision slabs. They are separate from the external Dubrovnik STL set used in the vision benchmark.

- Piece source: project-supplied Blender scene `calibration/scene_alignment/manual_layout_20261109/camera_mount_v3/package/initial_mount.blend`; SHA-256 `ae537174266ce94104420e45b7d530ce439f16b5a58d171d5b485babf0148790`. The authoring scene is not bundled. `assets/profiles.json` records source object names, dimensions and generated asset paths. The scanned inputs have no separate upstream license file in the inspected project; this handoff does not invent an additional asset license or certify provenance beyond that recorded source.
- Robot source: the local `so101_new_calib.urdf` identifies an onshape-to-robot export of [this Onshape document](https://cad.onshape.com/documents/7715cc284bb430fe6dab4ffd/w/4fd0791b683777b02f8d975a/e/826c553ede3b7592eb9ca800). The imported SO-ARM101 CAD/STL and SRDF are retained as project inputs. The wider [TheRobotStudio SO-ARM100/101 project](https://github.com/TheRobotStudio/SO-ARM100) publishes an [Apache-2.0 license](https://github.com/TheRobotStudio/SO-ARM100/blob/main/LICENSE); that link alone does not establish byte identity or the complete licensing chain of this local calibrated export. Original applicable asset terms remain with their respective owners.
- Export changes: Python control sources and mesh bytes are unchanged. Machine-specific JSON paths become portable tokens; URDF mesh filenames become relative. XML/JSON formatting may change. Original and exported hashes and transformations are recorded in `simulation/manifest.json`; original kinematic URDF header provenance remains in its byte-exact copy.
- Current runtime dependencies are PyBullet, NumPy, SciPy and python-chess/chess. Their own licenses apply. OMPL and vision dependencies are not required for the supplied current control entrypoint.

The 3 g piece masses, approximate inertia, contact envelope and friction values are simulation assumptions rather than measured physical material specifications. See the English handoff before using the assets to infer real-arm behaviour.
