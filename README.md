# Road Sign Detection: YOLOv8n vs Faster R-CNN vs RetinaNet

A controlled comparison of three object detectors on a 4-class road sign dataset
(speed limit, stop, crosswalk, traffic light). All models use the same
train/val/test split, the same COCO-pretrained initialization, the same
checkpoint selection rule and one common mAP implementation.

Author: Bijoy Ahmed, Southwest Petroleum University

## Results (test set, 135 images)

| Model                          | mAP50 | mAP50-95 | Inference (ms) | FPS  | Params (M) |
| ------------------------------ | ----- | -------- | -------------- | ---- | ---------- |
| YOLOv8n                        | 0.855 | 0.720    | 15.4           | 64.9 | 3.01       |
| Faster R-CNN (MobileNetV3-FPN) | 0.819 | 0.637    | 36.4           | 27.5 | 18.95      |
| RetinaNet (ResNet50-FPN)       | 0.818 | 0.666    | 165.9          | 6.0  | 32.23      |

Measured on a laptop NVIDIA MX330 (2 GB). YOLOv8n gives the best
accuracy-efficiency trade-off. Traffic light is the hardest class for all models.

## Dataset

Road Sign Detection dataset (Kaggle): https://www.kaggle.com/datasets/andrewmvd/road-sign-detection

877 images, 4 classes. Download it and place it under `data/`.
The split used in the paper (70/15/15, stratified, seed 42) is in `data/splits.csv`.

## Project structure

```
src/        training, evaluation and plotting scripts
results/    metrics (json/csv) and figures
report/     full paper (PDF)
data/       dataset (not included) and splits.csv
```

## Setup

```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Tested with Python 3.13, PyTorch 2.7.1 (CUDA 11.8), torchvision 0.22.1, Ultralytics 8.4.213.

## How to run

Run all commands from the project root.

```
python src/check_env.py
python src/prepare_data.py
python src/train_yolo.py
python src/train_frcnn.py
python src/train_retinanet.py
python src/eval_yolo_common.py
python src/compare.py
python src/visualize.py
```

## Notes and limitations

- Small test set (135 images); minority classes have only 10-14 test images.
- Single training run, no confidence intervals.
- Annotation has one box per image, which hurts the traffic light class.
- YOLOv8n was trained for 50 epochs with strong augmentation; the other two for 30 epochs with flip only.

Trained weights (.pt files) are not stored in this repository.

## Report

See `report/Road_Sign_Detection_Report.pdf`.
