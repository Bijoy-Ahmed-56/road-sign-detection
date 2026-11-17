import os
import json
from ultralytics import YOLO

if __name__ == "__main__":
    # Use the actual path where ultralytics saved the weights
    weights = r"runs/detect/results/runs/yolov8n/weights/best.pt"
    data = "data/yolo_road/data.yaml"
    out_dir = os.path.abspath("results/runs")   # absolute path avoids the extra runs/detect prefix

    model = YOLO(weights)

    # Evaluate on the held-out TEST split (135 images, never used for training or selection)
    m = model.val(
        data=data, split="test", imgsz=448, batch=8, device=0, workers=0,
        project=out_dir, name="yolov8n_test", exist_ok=True,
    )

    names = ["speedlimit", "stop", "crosswalk", "trafficlight"]
    result = {
        "model": "YOLOv8n",
        "test_precision": round(float(m.box.mp), 4),
        "test_recall": round(float(m.box.mr), 4),
        "test_mAP50": round(float(m.box.map50), 4),
        "test_mAP50_95": round(float(m.box.map), 4),
        "per_class_mAP50": {n: round(float(v), 4) for n, v in zip(names, m.box.ap50)},
        "inference_ms_per_image": round(float(m.speed["inference"]), 2),
        "train_time_hours": 0.776,
    }
    print(json.dumps(result, indent=2))

    # Save for the final comparison table
    with open("results/metrics_yolov8n.json", "w") as f:
        json.dump(result, f, indent=2)