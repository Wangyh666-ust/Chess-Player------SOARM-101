# Dubrovnik-specific YOLOv8m fine-tuning — trial v1

Experiment date: 2026-10-07.

This is a targeted **synthetic-domain chess-piece detection experiment** for ezeeone_Dubrovnik_1960_set_02_stls. It does not establish performance on real printed pieces or other chess sets.

## Task and model

YOLOv8m predicts object bounding boxes and piece-color/type classes. It is **object detection**, not instance segmentation: the network was trained on bounding boxes, while renderer ID masks were used only for visibility checks and annotation quality assurance.

The starting checkpoint is models/piece_best.pt from upstream revision d450e86fac94e7b6fa9c0e4fe156517995767095 of [alpkirisci/chess_detection_computer_vision](https://github.com/alpkirisci/chess_detection_computer_vision). The existing 13-output class head is retained: six white piece types, six black piece types, and the upstream empty class. There are no empty-square bounding-box annotations; empty squares are inferred from the reconstructed board.

All model layers remain trainable. Training initializes from pretrained weights with a new optimizer rather than resuming the upstream training state. This is supervised domain-specific fine-tuning; no old/new-data replay or formal continual-learning guarantee was evaluated.

Loss components follow the Ultralytics detection implementation: CIoU box loss, BCE classification loss, and distribution focal loss (DFL), with gains 7.5, 0.5, and 1.5 respectively. DFL supervises the distributions over distances from feature-map candidate locations to the four box edges.

## Synthetic environment

The environment is a static VTK offscreen renderer. It does not simulate robot motion, grasping, contact, or physics.

The board is procedural geometry: 64 planar squares, each 50 mm, making a 400 × 400 mm playing surface. A 15 mm border and a table plane are added. It does not use a downloaded chessboard mesh. The STL set is uniformly scaled to a 78 mm king.

The bishop body and separate head are assembled by a geometric stem-insertion heuristic; printed assembly fit is not measured. Bottom caps are omitted. Materials are approximate, with no physically accurate print textures or cast shadows.

Views randomize camera elevation (25–80°), azimuth (±15°), distance (800–1,050 mm), vertical field of view (35–43°), piece position jitter (±2 mm), piece yaw (±12°), colors, light, background, exposure, and small image blur. Images are 960 × 720; the earlier nine scenes are 1,600 × 1,200.

## Data and annotation

There are 375 distinct chess placements, with eight views per placement:

| Split | Positions | Images |
| --- | ---: | ---: |
| Train | 262 | 2,096 |
| Validation | 56 | 448 |
| Test | 57 | 456 |
| Total | 375 | 3,000 |

The split is by **piece placement**, not by image. No placement crosses splits. All three placements used in the earlier nine-scene comparison are excluded from every generated split. Seed: 20261007.

Boxes are the projected full-mesh (amodal) extents, rather than only the visible silhouette. ID masks verify that each annotated piece has at least 20 visible pixels. The completed dataset has 72,272 labeled piece instances and no pieces omitted by the visibility threshold.

YOLO annotations are class ID and normalized box center/width/height. The split plan contains placements, image seeds, and SHA-256 hashes for the seven source meshes.

## Training

| Setting | Value |
| --- | --- |
| Initial model | Upstream ChessReD2K YOLOv8m piece checkpoint |
| Epochs completed | 15 |
| Optimizer | AdamW |
| Initial learning rate | 0.0005 |
| Learning-rate schedule | Cosine, final fraction 0.1 |
| Weight decay | 0.0005 |
| Batch / nominal batch | 4 / 16 |
| Image size | 640 |
| Warmup / patience | 1 epoch / 6 epochs |
| Mixed precision | Enabled during training |
| Data-loader workers | 0 |
| Flips / mosaic / mixup | Disabled |
| Layer freezing | None |
| Elapsed training time | About 23.1 minutes |

Minor color, translation, and scale augmentation are enabled. The released file is the training run's selected best.pt, not a newly trained model during demo construction.

## Evaluation and board reconstruction

Both checkpoints use the same evaluation pipeline: batch 1, image size 640, confidence threshold 0.25, NMS IoU threshold 0.7, FP32, CUDA device 0, four PyTorch CPU threads, and zero OpenCV worker threads.

**Board corners are supplied by the renderer in semantic TL → TR → BR → BL order.** Automatic corner detection is unchanged and is not included in the reported result. A direct homography maps these corners into an 800 × 800 board. Sum/difference-based geometric corner reordering is avoided because it can duplicate selected vertices for low-angle, rotated views.

For each detection, the bottom 25% strip of its box is projected into the rectified board. The square with the highest rectangle-overlap IoU is selected; bottom-center projection is the fallback if overlap is negligible. When detections compete for a square, the higher-confidence detection wins. Both models use this same mapping.

The repository's portable geometry mapper is checked against all **930 saved model predictions**: two checkpoints × (9 earlier scenes + 456 held-out images). It does not require upstream corner weights or importing upstream inference code.

## Main results

| Held-out metric | Original | Fine-tuned |
| --- | ---: | ---: |
| Occupancy F1 | 89.9689% | 99.9861% |
| Color accuracy, GT occupied squares | 75.0927% | 99.9258% |
| Type accuracy, GT occupied squares | 65.9959% | 99.8887% |
| Color + type correct | 6,505 / 10,784 | 10,768 / 10,784 |
| Color + type accuracy | 60.3208% | 99.8516% |
| Completely correct boards | 0 / 456 | 442 / 456 |
| Exact-board accuracy | 0% | 96.9298% |
| Median processing time | 13.7645 ms | 13.8117 ms |
| 95th-percentile processing time | 27.8711 ms | 21.8933 ms |

Joint accuracy measures correct class **at the correct square** on ground-truth occupied squares. A missing piece counts as wrong. Exact-board accuracy checks all 64 symbols, including empty squares. These are board-recognition metrics, not object-detection mAP.

On the earlier nine scenes: original color + type is 116/222 (52.2523%), fine-tuned is 222/222 (100%); exact boards improve from 0/9 to 9/9.

Timing includes preprocessing, GPU inference, result transfer, and square mapping after image loading. It excludes disk I/O, camera capture, and corner detection. Hardware: RTX 5070 Laptop GPU with 8 GB VRAM and 24 GB system memory. This timing is not an end-to-end robot throughput measurement.

## Residual failures

Fourteen held-out frames have an imperfect board, totaling sixteen wrong occupied squares: three misses and thirteen class/color mistakes. Thirteen of these frames have camera elevations below 40°. All empty squares are correctly reconstructed by the fine-tuned pipeline in this test.

| Elevation range | Frames | Exact boards | Correct occupied-square color + type |
| --- | ---: | ---: | ---: |
| 25–<40° | 141 | 128 | 3,380 / 3,394 |
| 40–<60° | 158 | 158 | 3,647 / 3,647 |
| 60–80° | 157 | 156 | 3,741 / 3,743 |

Failure frame IDs:

g0318_v01, g0322_v02, g0329_v03, g0330_v02, g0330_v05, g0340_v02, g0342_v01, g0346_v05, g0348_v02, g0348_v03, g0360_v02, g0360_v06, g0372_v00, g0373_v07.

The demo includes all of them, in addition to deterministic overview samples. Preview metrics are never substituted for full-test metrics.

The recorded ideal-box mapping diagnostic reconstructs all 456 boards correctly when supplied ground-truth boxes and classes. This checks the mapping on these annotations; it is **not a model inference result or a guaranteed performance upper bound**.

## Checkpoint identity

Original SHA-256:

~~~text
514a9a95f9eb54f133a4dff4c9fa60430ea44137f76fc538a13099a6f2582345
~~~

Released fine-tuned SHA-256:

~~~text
021ad9f46ad2c31856a9aa5e36b25e10c6d10be8b8005efcf9982908a4c11a9b
~~~

Released size: 52,026,770 bytes.

## Interpretation and next work

The experiment demonstrates adaptation to this specific **rendered piece geometry**, under controlled board calibration. Accuracy may change with print finish, real illumination, camera optics, motion blur, occlusion, board design, or different pieces.

Next steps are real-camera validation and labeled real images, board localization/calibration, temporal state checking, turn handling, and integration with Stockfish. VLA demonstrations and SO-ARM101 execution require separate action data and training; this vision result does not provide an action policy.
