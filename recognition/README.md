# Dual-camera chessboard recognition

This source snapshot pairs the fixed C2 camera with the wrist-mounted C1 camera. The Ubuntu service uses ArUco localization and YOLOv8m to assign pieces to squares. It combines per-square class scores with quality-weighted C2:C1 weights of 7:3 and a confidence exception for strong C1 evidence. The Windows preview provides Chinese and English controls. **Recognize board** combines five two-second windows and marks unresolved squares with `?` for operator review.

The Ubuntu service and Windows preview operate independently of the arm controller. The recognition code expects a compatible detector checkpoint, camera calibration, and local reference frames. This directory contains source, example configuration and the project's marker calibration. Each installation supplies its own checkpoint and camera images.

## Ubuntu recognition service

Run these commands from the `recognition` directory. Install a CUDA-compatible PyTorch build if using an NVIDIA GPU, then install the remaining dependencies.

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements-ubuntu.txt
cp vision_eval/collection_config.example.json vision_eval/collection_config.json
cp vision_eval/grid_config.example.json vision_eval/grid_config.json
cp vision_eval/camera_mapping.example.json vision_eval/camera_mapping.json
```

In `collection_config.json`, enter the checkpoint's absolute path, its SHA-256 from `sha256sum`, and a writable absolute capture directory. The deployed model expects class IDs `0`–`11` in the order listed in `grid_config.example.json`. Use a trusted YOLO checkpoint trained with that class order.

Put camera reference frames at `vision_eval/references/c1.png` and `vision_eval/references/c2.png` to match the dimensions and orientations in `camera_mapping.json`. Live localization uses the visible ArUco markers and their calibrated board coordinates. For another physical placement, update `vision_eval/aruco_live_assets/layout.json` with surveyed marker corners in `a1`-origin board coordinates and retain detector corner order. The bundled C1 radial correction and occlusion rectangle describe the project's wrist-camera setup.

The first device is the wrist camera and the second is the fixed camera. Replace the host and video devices in this example. Bind to loopback for same-host testing or to your Ubuntu machine's trusted private-network interface for Windows access.

```sh
python vision_eval/aruco_live_service_v10.py \
  --host 192.0.2.10 --port 8765 \
  --camera-a /dev/video0 --camera-b /dev/video2 --max-seconds 7200
```

Keep the board update interval at **2 s** for five-window confirmation. The Ubuntu server uses `vision_eval/voting_settings.json` for the interval.

## Windows board preview

Start the Ubuntu recognition service first. From `recognition` in PowerShell:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-windows.txt
Copy-Item vision_eval/dual_camera_launcher_config.example.json vision_eval/dual_camera_launcher_config.json
```

Edit `host`, `port`, and `url` in the Windows config to match the Ubuntu service. The manual `--serve` entry reads these settings and uses the already-running upstream service.

```powershell
.\.venv\Scripts\python.exe -X utf8 vision_eval/local_preview_with_review.py --serve --port 8766 --default-root "$PWD\captures"
```

Open `http://127.0.0.1:8766/` and use **Language / 语言** to switch the preview. Keep the board still for recognition. A piece class needs at least three of the five window votes. A square with piece evidence but no majority is shown as unresolved. Review flagged squares before freezing the result.

The local workflow requires no cloud credentials. The optional Qwen review reads `DashScopeAPI.txt` from `recognition` and takes `reference_compact24.jpg` and `manifest_compact24.json` from `vision_eval/board_confirmation/field_references/`. The manifest supplies `count` (24) and `jpeg_sha256`. Enabling Qwen sends the selected crops and reference sheet to the provider.

## Source entry points

- `vision_eval/aruco_live_service_v10.py` runs the Ubuntu recognition service.
- `vision_eval/weighted_temporal_fusion_v4.py` implements the square-level fusion policy.
- `vision_eval/second_window_vote_v2.py` and `vision_eval/voting_controls_v2.py` manage time-window aggregation.
- `vision_eval/board_confirmation/five_window_vote.py` tallies the final board.
- `vision_eval/local_preview_with_review.py` and `vision_eval/preview_language*.py` serve and localize the Windows interface.

The detector uses [Ultralytics YOLO](https://github.com/ultralytics/ultralytics). Board markers and image geometry use [OpenCV ArUco](https://docs.opencv.org/4.x/d5/dae/tutorial_aruco_detection.html).
