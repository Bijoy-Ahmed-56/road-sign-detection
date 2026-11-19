# Road Sign Detection: YOLOv8n vs Faster R-CNN vs RetinaNet

A comparative study of three deep learning object detection models for road sign detection using a 4-class dataset.

This project evaluates **YOLOv8n, Faster R-CNN, and RetinaNet** under a common evaluation framework to compare their detection accuracy and inference efficiency.

**Author:** Bijoy Ahmed  
**Institution:** Southwest Petroleum University

## Project Overview

The objective of this project is to compare different object detection architectures for road sign detection and analyze their performance in terms of accuracy, inference speed, and model complexity.

The dataset contains four classes:

- Speed Limit
- Stop
- Crosswalk
- Traffic Light

## Models Compared

| Model | Architecture |
|---|---|
| YOLOv8n | YOLO |
| Faster R-CNN | MobileNetV3-FPN |
| RetinaNet | ResNet50-FPN |

## Experimental Results

The following results were obtained on the test set containing 135 images.

| Model | mAP50 | mAP50-95 | Inference (ms) | FPS | Parameters (M) |
|---|---:|---:|---:|---:|---:|
| YOLOv8n | 0.855 | 0.720 | 15.4 | 64.9 | 3.01 |
| Faster R-CNN | 0.819 | 0.637 | 36.4 | 27.5 | 18.95 |
| RetinaNet | 0.818 | 0.666 | 165.9 | 6.0 | 32.23 |

**Hardware:** NVIDIA GeForce MX330 (2 GB VRAM)

### Key Findings

- YOLOv8n achieved the highest mAP50 and mAP50-95 among the evaluated models.
- YOLOv8n provided the fastest inference speed.
- Faster R-CNN and RetinaNet required more parameters than YOLOv8n.
- Traffic Light was the most challenging class for all three models.

Based on these experimental results, YOLOv8n provided the best accuracy-efficiency trade-off in this experiment.

## Dataset

**Dataset:** [Road Sign Detection Dataset — Kaggle](https://www.kaggle.com/datasets/andrewmvd/road-sign-detection)

- Total images: 877
- Number of classes: 4
- Train/Validation/Test split: 70% / 15% / 15%
- Split strategy: Stratified
- Random seed: 42

The dataset is not included in this repository. Download it from Kaggle and place it inside the `data/` directory according to the expected dataset structure.

The split information is available in `data/splits.csv`.

## Project Structure

```text
road-sign-detection/
│
├── data/
│   └── splits.csv
│
├── report/
│   └── Road_Sign_Detection_Report.pdf
│
├── results/
│   ├── Metrics and evaluation files
│   └── Figures and visualizations
│
├── src/
│   ├── check_env.py
│   ├── prepare_data.py
│   ├── train_yolo.py
│   ├── train_frcnn.py
│   ├── train_retinanet.py
│   ├── eval_yolo_common.py
│   ├── compare.py
│   └── visualize.py
│
├── .gitignore
├── requirements.txt
└── README.md
```

## Installation

### Requirements

- Python 3.13
- PyTorch 2.7.1
- torchvision 0.22.1
- Ultralytics 8.4.213
- CUDA 11.8 (for the tested GPU configuration)

### 1. Clone the Repository

```bash
git clone https://github.com/Bijoy-Ahmed-56/road-sign-detection.git
cd road-sign-detection
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

### 3. Activate the Environment

**Windows:**

```bash
venv\Scripts\activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

## How to Run

Run the following commands from the project root directory.

### 1. Check Environment

```bash
python src/check_env.py
```

### 2. Prepare Dataset

```bash
python src/prepare_data.py
```

### 3. Train Models

```bash
python src/train_yolo.py
python src/train_frcnn.py
python src/train_retinanet.py
```

### 4. Evaluate and Compare Models

```bash
python src/eval_yolo_common.py
python src/compare.py
```

### 5. Generate Visualizations

```bash
python src/visualize.py
```

## Experimental Limitations

The following limitations should be considered when interpreting the results:

- The test set contains only 135 images.
- Minority classes have only 10–14 test images.
- Each model was trained using a single run, without confidence intervals.
- The annotations contain one bounding box per image, which affects the Traffic Light class.
- YOLOv8n was trained for 50 epochs with strong augmentation, while Faster R-CNN and RetinaNet were trained for 30 epochs with flip augmentation only.

These differences should be considered when interpreting the model comparison.

## Trained Weights

Trained model weights (`.pt` files) are not included in this repository.

## Project Report

For detailed methodology, experiments, and analysis, see:

`report/Road_Sign_Detection_Report.pdf`

## Author

**Bijoy Ahmed**

- GitHub: [Bijoy-Ahmed-56](https://github.com/Bijoy-Ahmed-56)

