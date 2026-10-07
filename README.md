# SO-ARM101 Chess Player — Vision Module

An initial chess-vision result for the planned SO-ARM101 chess-playing robot. This release fine-tunes a YOLOv8m piece detector for **one specific piece model: ezeeone_Dubrovnik_1960_set_02_stls**.

**Scope: synthetic RGB images, known board corners, and chess-piece recognition.** These results do not measure automatic board localization, real camera performance, other chess sets, or robot/VLA execution.

## Interactive demo

**[Open the online demo](https://wangyh666-ust.github.io/Chess-Player------SOARM-101/)**.

![Original and fine-tuned Dubrovnik detector comparison](docs/assets/demo-preview.png)

Open [the demo](docs/index.html) **after cloning or downloading the repository**, or run a local server from the repository root:

~~~bash
python -m http.server 8000 --bind 127.0.0.1 --directory docs
~~~

Then visit **http://127.0.0.1:8000/**. The demo needs no Python packages, GPU, backend inference service, API key, or Internet connection when opened locally.

The English interface shows recorded original/fine-tuned detections side by side, ground-truth and reconstructed boards, camera elevations, and square-level errors. It includes **all 14 failed fine-tuned test frames**, the first six test frames, twelve evenly spaced frames, and the nine earlier scenes. Benchmark metrics always cover the complete evaluation set, independent of the preview filter. Detection overlays and board reconstructions come from saved predictions.

To host this static demo on GitHub Pages, select **Settings → Pages → Deploy from a branch → main → /docs → Save**. GitHub's source view does not execute HTML.

## Measured result

Held-out test: **456 synthetic images from 57 positions**, with **10,784 occupied ground-truth squares**.

| Metric | Original YOLOv8m | Dubrovnik fine-tuned |
| --- | ---: | ---: |
| Occupancy F1 | 89.97% | 99.99% |
| Piece color accuracy | 75.09% | 99.93% |
| Piece type accuracy | 66.00% | 99.89% |
| Piece color + type accuracy | 60.32% | **99.85%** |
| Completely correct boards | 0 / 456 | **442 / 456 (96.93%)** |
| Median processing time | 13.76 ms | 13.81 ms |
| 95th-percentile processing time | 27.87 ms | 21.89 ms |

Color/type/joint accuracy use **ground-truth occupied squares** as the denominator; missed pieces count as incorrect. Exact-board accuracy checks all 64 squares, including empty squares.

Timing was measured on an **NVIDIA RTX 5070 Laptop GPU (8 GB)** with batch size 1 and FP32 inference. It includes preprocessing, inference, and square mapping, and excludes disk reads, camera capture, and automatic corner detection. It is not a measured end-to-end robot frame rate.

The earlier nine scenes improved from **116/222 (52.25%)** to **222/222 (100%)** correct piece color + type, and from 0/9 to 9/9 completely correct boards. See [EXPERIMENT.md](EXPERIMENT.md) for the protocol, failures, and limitations.

## What is included

- [Trained checkpoint](models/dubrovnik-yolov8m-v1.pt), approximately 49.6 MiB.
- [English interactive demo](docs/index.html), preview images, and recorded detections.
- [Full recorded evaluation](artifacts/dubrovnik-v1/comparison.json), [summary CSV](artifacts/dubrovnik-v1/summary.csv), [training log](artifacts/dubrovnik-v1/training-results.csv), and plots.
- [Split plan and source-mesh hashes](artifacts/dubrovnik-v1/split-plan.json) and [dataset statistics](artifacts/dubrovnik-v1/dataset-statistics.json).
- Synthetic data generation, fine-tuning, evaluation, and geometry helpers under [vision_eval](vision_eval).
- [Artifact verifier](tools/verify_artifacts.py) for checkpoint identity, stored prediction/mapping consistency, split isolation, and demo integrity.

The full 3,000-image dataset and original STL assets are generated/local inputs and are not bundled. Their absence does not prevent browsing the demo, verifying the saved results, or using the trained detector.

## Use the checkpoint

Install a CUDA-enabled PyTorch build suitable for your system and the [tested dependencies](requirements.txt). The original run used Python 3.10.20, PyTorch 2.7.1+cu128, torchvision 0.22.1+cu128, and Ultralytics 8.4.173.

For standalone piece detection:

~~~bash
yolo predict model=models/dubrovnik-yolov8m-v1.pt source=your_image.jpg imgsz=640 conf=0.25 iou=0.7 device=0
~~~

This produces bounding boxes and piece classes. Reconstructing square occupancy also requires ordered board corners and a known board orientation; the provided mapper supports **known corners** only.

For verification of the published artifacts, a GPU and PyTorch are unnecessary:

~~~bash
python -m pip install numpy==1.26.4 opencv-python==4.11.0.86 Pillow==12.3.0
python tools/verify_artifacts.py
~~~

## Reproduce the experiment

See [REPRODUCING.md](REPRODUCING.md) for asset preparation and commands. It documents the external original checkpoint, source mesh files, tested environment, deterministic split, and full evaluation.

The intended later pipeline is:

~~~text
RGB image → board/piece recognition → verified board state → Stockfish
RGB + robot state + move instruction → VLA policy → SO-ARM101
New RGB image → recognition → execution verification
~~~

This repository currently implements and documents the initial recognition experiment. Board localization, real-camera adaptation, turn handling, and VLA action learning remain subsequent work.

## Attribution and model workflow

The starting detector is from [alpkirisci/chess_detection_computer_vision](https://github.com/alpkirisci/chess_detection_computer_vision), trained on ChessReD2K. See [THIRD_PARTY.md](THIRD_PARTY.md) for sources and dependency information.

GPT handled experiment design, planning, analysis, decisions, and review. DeepSeek deepseek-flash handled code implementation through its official API. No credential or model-generation service is required to run this repository or demo.
