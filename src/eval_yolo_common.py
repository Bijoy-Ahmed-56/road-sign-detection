import json
import time
import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO

from common import load_split, evaluate

if __name__ == "__main__":
    weights = r"runs/detect/results/runs/yolov8n/weights/best.pt"
    model = YOLO(weights)

    test = load_split("test")
    paths = test["img"].tolist()

    # Low confidence threshold so that the full precision-recall curve is available for mAP
    kw = dict(imgsz=448, conf=0.001, iou=0.6, max_det=100, device=0, verbose=False)
    results = model.predict(paths, **kw)

    preds, gts = [], []
    for r, (_, row) in zip(results, test.iterrows()):
        preds.append({
            "boxes": r.boxes.xyxy.cpu().numpy(),
            "scores": r.boxes.conf.cpu().numpy(),
            "labels": r.boxes.cls.cpu().numpy().astype(int),
        })
        gts.append({
            "boxes": np.array([[row["x1"], row["y1"], row["x2"], row["y2"]]], dtype=float),
            "labels": np.array([int(row["class"])]),
        })

    res = evaluate(preds, gts)
    res["model"] = "YOLOv8n"
    res["train_time_hours"] = 0.776
    res["params_million"] = 3.13

    # Timing: load all images into memory first (no disk reads), warm up the GPU,
    # then time preprocess + inference + NMS per image, like the torchvision models
    imgs = [np.array(Image.open(p).convert("RGB"))[:, :, ::-1] for p in paths]  # BGR arrays
    for im in imgs[:5]:                      # warm-up
        model.predict(im, **kw)
    torch.cuda.synchronize()
    t0 = time.time()
    for im in imgs:
        model.predict(im, **kw)
    torch.cuda.synchronize()
    res["inference_ms_per_image"] = round((time.time() - t0) / len(imgs) * 1000, 2)

    print(json.dumps(res, indent=2))
    with open("results/metrics_yolov8n_common.json", "w") as f:
        json.dump(res, f, indent=2)