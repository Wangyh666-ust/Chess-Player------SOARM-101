# Sources and third-party components

## Starting detector and recognition data

- Starting YOLOv8m checkpoint: [alpkirisci/chess_detection_computer_vision](https://github.com/alpkirisci/chess_detection_computer_vision), revision d450e86fac94e7b6fa9c0e4fe156517995767095, models/piece_best.pt.
- Upstream describes this detector as trained on ChessReD2K. The synthetic Dubrovnik fine-tuning dataset is a separate dataset created for this experiment.
- ChessReD citation: Athanasios Masouris and Jan van Gemert, *End-to-End Chess Recognition*, VISAPP 2024, pp. 393–403, DOI [10.5220/0012370200003660](https://doi.org/10.5220/0012370200003660).

The portable known-corner mapping helper implements the same bottom-strip geometry used for the comparison, with outputs checked against stored predictions. Upstream inference source and its corner model are not bundled.

The inspected upstream snapshot did not include a standalone license file. This repository does not grant additional rights to the upstream checkpoint, original recognition data, or external mesh assets.

## Piece geometry

The supplied STL asset set is named ezeeone_Dubrovnik_1960_set_02_stls. Mesh identity is recorded by filename and SHA-256 in the split plan. Raw meshes are external inputs and are not redistributed in this repository. Rendered preview images and experiment outputs were produced for this project.

## Dependencies

Model training/inference uses [Ultralytics](https://github.com/ultralytics/ultralytics), with its upstream AGPL-3.0 and enterprise licensing options. Rendering and processing use VTK, NumPy, OpenCV, Pillow, python-chess, SciPy, matplotlib, PyYAML, and PyTorch. Each dependency retains its own license; consult its source distribution for terms.

No external font, JavaScript framework, or CDN is needed for the demo.

## Preliminary robot simulation

The recorded motion experiment uses PyBullet, OMPL, SciPy, NumPy, python-chess, Pillow, OpenCV, and imageio-ffmpeg. The video depicts the project's SO-ARM101 and Dubrovnik geometry in a nominal virtual installation. Raw robot/CAD/STL assets and the full local simulation runtime are not bundled by this results update; rendered video, selected recorded poses, and audit data are included. Each dependency and source asset retains its own terms.
