# Dual-camera chessboard recognition

This source snapshot pairs the fixed C2 camera with the wrist-mounted C1 camera. The Ubuntu service uses ArUco localization and YOLOv8m to assign pieces to squares. It combines per-square class scores with quality-weighted C2:C1 weights of 7:3 and a confidence exception for strong C1 evidence. The Windows preview provides Chinese and English controls. **Recognize board** combines five two-second windows and marks unresolved squares with `?` for operator review.

The Ubuntu service and Windows preview operate independently of the arm controller. The recognition code expects a compatible detector checkpoint, camera calibration, and local reference frames. This directory contains source, example configuration and the project's marker calibration. Each installation supplies its own checkpoint and camera images.

## Deployment and portability

In our project setup, both USB cameras were connected to an Ubuntu computer. Ubuntu captured frames, ran YOLO and ArUco, fused the two views, and served the results over HTTP. A separate Windows computer connected to that service and hosted the bilingual preview and final board confirmation. The optional SSH launcher can start the service remotely. The preview's `--serve` mode connects over HTTP to a service that is already running.

| Setup | Recognition service | Preview connection |
| --- | --- | --- |
| Separate Ubuntu and preview computers | Bind `--host` to a trusted private-network address on Ubuntu, with TCP port `8765` reachable from the preview computer. | Set `host`, `port`, and `url` in `dual_camera_launcher_config.json` to that address and port. |
| Both processes on one Ubuntu computer | Connect both cameras to that computer and bind `--host 127.0.0.1 --port 8765`. | Set `host` to `127.0.0.1`, `port` to `8765`, and `url` to `http://127.0.0.1:8765/`. |

In both cases, open `http://127.0.0.1:8766/` in a browser on the computer running the preview. The recognition service uses plain HTTP without user authentication. Keep it on loopback or a trusted private network.

The capture code uses OpenCV's Linux V4L2 backend and expects two distinct, readable devices such as `/dev/video0` and `/dev/video2`. Native Windows or macOS recognition requires adapting the camera backend, device selection, and capture settings. Under WSL, the Linux process must be able to open both cameras as V4L2 devices. Check camera access and calibration on the target machine before relying on the same-host setup.

## Ubuntu recognition service

Run these commands from `recognition` on the computer connected to the cameras. Install a CUDA-compatible PyTorch build if using an NVIDIA GPU, then install the remaining dependencies.

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements-ubuntu.txt
cp vision_eval/collection_config.example.json vision_eval/collection_config.json
cp vision_eval/grid_config.example.json vision_eval/grid_config.json
cp vision_eval/camera_mapping.example.json vision_eval/camera_mapping.json
```

In `collection_config.json`, enter the checkpoint's absolute path on the recognition computer, its SHA-256 from `sha256sum`, and a writable absolute capture directory on that computer. Use a trusted YOLO checkpoint with class IDs `0`–`11` in the order listed in `grid_config.example.json`.

Put camera reference frames at `vision_eval/references/c1.png` and `vision_eval/references/c2.png` to match the dimensions and orientations in `camera_mapping.json`. After changing cameras or mounts, check the capture resolution, the canonical rotations (C1 at 0° and C2 at 270° in this snapshot), and the square mapping. Verify C1's occlusion mask in `camera_mapping.json` and its radial correction in `vision_eval/aruco_live_assets/radial.json`. If the board or markers change, update the board dimensions and surveyed marker corners in `vision_eval/aruco_live_assets/layout.json` using the `a1` origin and detector corner order.

Set `--camera-a` to the wrist camera (C1, index `0`) and `--camera-b` to the fixed camera (C2, index `1`). Replace the paths below with the actual devices. For separate computers, substitute the Ubuntu computer's private address for `127.0.0.1`.

```sh
python vision_eval/aruco_live_service_v10.py \
  --host 127.0.0.1 --port 8765 \
  --camera-a /dev/video0 --camera-b /dev/video2 --max-seconds 7200
```

Keep the board update interval at **2 s** for five-window confirmation. The Ubuntu server uses `vision_eval/voting_settings.json` for the interval.

## Bilingual board preview

Start the Ubuntu recognition service first. Run the commands for the preview computer's operating system from `recognition`. In Windows PowerShell:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-windows.txt
Copy-Item vision_eval/dual_camera_launcher_config.example.json vision_eval/dual_camera_launcher_config.json
```

On the same Ubuntu computer, reuse the recognition virtual environment and add the preview's Python dependencies:

```sh
. .venv/bin/activate
python -m pip install -r requirements-windows.txt
cp vision_eval/dual_camera_launcher_config.example.json vision_eval/dual_camera_launcher_config.json
```

Set `host`, `port`, and `url` in the copied config as shown in the table. `192.0.2.10` in the example is a documentation placeholder. Use an IP address and a root URL in the form `http://<IP>:<port>/`. Keep the other example fields because the config loader validates the full schema. In `--serve` mode, the SSH credentials and remote paths are unused. `--default-root` sets the directory for preview-side captures.

In Windows PowerShell:

```powershell
.\.venv\Scripts\python.exe -X utf8 vision_eval/local_preview_with_review.py --serve --port 8766 --default-root "$PWD\captures"
```

On Ubuntu:

```sh
python -X utf8 vision_eval/local_preview_with_review.py --serve --port 8766 --default-root "$PWD/captures"
```

Use **Language / 语言** to switch the preview. Keep the board still for recognition. A piece class needs at least three of the five window votes. A square with piece evidence but no majority is shown as unresolved. Review flagged squares before freezing the result.

The local workflow requires no cloud credentials. The optional Qwen review reads `DashScopeAPI.txt` from `recognition` and takes `reference_compact24.jpg` and `manifest_compact24.json` from `vision_eval/board_confirmation/field_references/`. The manifest supplies `count` (24) and `jpeg_sha256`. If the calibration changes, sync the marker layout and radial correction in `vision_eval/board_confirmation/calibration/` for this crop path. Enabling Qwen sends the selected crops and reference sheet to the provider.

## Remote launcher and HTTP integration

The optional SSH startup path needs this installation's `username`, `credential_file`, `known_hosts`, `remote_python`, `remote_script`, `remote_lock`, `remote_log`, and `camera_devices` in the launcher config. Install Paramiko on the launcher's computer and set `expected_source_sha256` to the digest of the entry script on Ubuntu. Store credentials locally. The launcher checks the remote host key, script and devices before starting the HTTP service.

Check `GET /status` on the recognition service from the preview computer to verify connectivity and camera status. For **Recognize board**, the upstream also needs to serve both annotated views at `/view/0.jpg` and `/view/1.jpg` and expose both camera states, all 64 square entries, and completed 2-second voting windows in the `collection.grid_pipeline` section of `/status`. If replacing the upstream server or transport, preserve these interfaces or adapt [the board sampler](vision_eval/board_confirmation/live_capture_budget.py) and [the preview bridge](vision_eval/local_capture_bridge_v9.py). The bilingual preview itself serves the `/board-review/` routes.

## Source entry points

- `vision_eval/aruco_live_service_v10.py` runs the Ubuntu recognition service.
- `vision_eval/weighted_temporal_fusion_v4.py` implements the square-level fusion policy.
- `vision_eval/second_window_vote_v2.py` and `vision_eval/voting_controls_v2.py` manage time-window aggregation.
- `vision_eval/board_confirmation/five_window_vote.py` tallies the final board.
- `vision_eval/local_preview_with_review.py` and `vision_eval/preview_language*.py` serve and localize the bilingual preview.

The detector uses [Ultralytics YOLO](https://github.com/ultralytics/ultralytics). Board markers and image geometry use [OpenCV ArUco](https://docs.opencv.org/4.x/d5/dae/tutorial_aruco_detection.html).
