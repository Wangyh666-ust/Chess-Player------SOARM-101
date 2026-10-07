# Reproducing the vision experiment

The static demo and stored-artifact checks work without downloading the original model or STL files. Regenerating the training data and rerunning the experiment require those external inputs.

Commands below run from the repository root. Python 3.10 was used. Rendering and training were tested on Windows with an NVIDIA GPU; timings and pixel-level rendering can vary across hardware or graphics drivers.

## Environment

Create a virtual environment, activate it, and install the CUDA-enabled PyTorch build appropriate for the machine. The measured run used:

~~~bash
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
~~~

For stored-artifact verification alone:

~~~bash
python -m pip install numpy==1.26.4 opencv-python==4.11.0.86 Pillow==12.3.0
python tools/verify_artifacts.py
~~~

The verifier checks the released weight hash, all 930 recorded board mappings, canonical result counts, split isolation, and demo sample fidelity.

## Source meshes

Prepare the user-provided ezeeone_Dubrovnik_1960_set_02_stls asset set under:

~~~text
vision_eval/prepared/ezeeone_Dubrovnik_1960_set_02_stls/
~~~

The required files are:

~~~text
obj_5_1960_Pawn_01.step.stl
obj_3_1960_knight.stl
obj_13_1960_Set_v1 v9_2.stl
obj_4_1960_Rook_01.step.stl
obj_2_1960_Set_v1 v9.stl
obj_1_1960_Set_v1 v9.stl
obj_12_1960_Bishop_01.step_1.stl
~~~

Expected hashes are in [split-plan.json](artifacts/dubrovnik-v1/split-plan.json). The portable loader requires only Dubrovnik assets; it does not require the other two candidate chess sets from the earlier comparison.

The source STLs are not bundled. The nine previous rendered scenes and their filtered manifest are included.

## Original checkpoint

Download [the original piece checkpoint at the recorded upstream revision](https://github.com/alpkirisci/chess_detection_computer_vision/raw/d450e86fac94e7b6fa9c0e4fe156517995767095/models/piece_best.pt) to:

~~~text
vision_eval/alternatives/vendor/chessred_yolov8/models/piece_best.pt
~~~

Expected SHA-256:

~~~text
514a9a95f9eb54f133a4dff4c9fa60430ea44137f76fc538a13099a6f2582345
~~~

The known-corners evaluation does not need a corner checkpoint or upstream inference script.

## Generate the data

~~~bash
python vision_eval/build_dubrovnik_dataset_v4.py --groups 375 --views 8 --seed 20261007
python vision_eval/build_dubrovnik_dataset_v4.py --audit-only
~~~

Default output is vision_eval/dubrovnik_finetune/dataset_v1/. The generator writes images, labels, masks, records, a split plan, statistics, and dataset.yaml. The default full generation contains 3,000 images. The --limit option can create a rendering pilot; that incomplete pilot is not a valid full-training dataset.

Compare generated placements/seeds and mesh hashes with the published plan. VTK rendering differences across machines can change pixels even when placements and camera parameters agree.

## Fine-tune

~~~bash
python vision_eval/train_dubrovnik_yolov8_v4.py --name reproduction_v1 --epochs 15 --batch 4 --imgsz 640
~~~

The driver validates data completeness and placement isolation before full training. A distinct run name avoids overwriting a previous run. The recorded settings are in [EXPERIMENT.md](EXPERIMENT.md).

## Evaluate

Evaluate a new training run:

~~~bash
python vision_eval/evaluate_dubrovnik_finetune_v6.py --weights vision_eval/dubrovnik_finetune/runs/reproduction_v1/weights/best.pt --output vision_eval/dubrovnik_finetune/evaluation/reproduction_v1
~~~

To evaluate the released weight, first copy models/dubrovnik-yolov8m-v1.pt into a location under vision_eval/, such as vision_eval/dubrovnik_finetune/released/best.pt, and pass that path with --weights. The evaluation driver's output/input safety checks require paths under vision_eval/.

Evaluation reads the nine included scenes and the regenerated 456 test images. It writes per-image predictions and aggregate recognition metrics for both original and fine-tuned checkpoints. Both use the same known corners and square mapper.

Published results are the completed 2026-10-07 run. Reruns can vary due to rendering, dependency versions, and GPU execution; do not replace published numbers with a limited pilot.
