import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from ultralytics import YOLO
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn, retinanet_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.retinanet import RetinaNetClassificationHead
from torchvision.transforms.functional import to_tensor

from common import load_split, NAMES

THR = 0.5                      # only draw boxes with score >= 0.5
device = torch.device("cuda")
COLORS = {"YOLOv8n": "#2a6fdb", "Faster R-CNN": "#e8830c", "RetinaNet": "#2a9d6f"}

# One test image per class (first one in the test split)
test = load_split("test")
rows = [test[test["class"] == c].iloc[0] for c in range(4)]
paths = [r["img"] for r in rows]
gts = [(np.array([r["x1"], r["y1"], r["x2"], r["y2"]]), int(r["class"])) for r in rows]


def top_dets(boxes, scores, labels):
    """Keep boxes above the threshold, best 3 only, to avoid clutter."""
    keep = np.where(scores >= THR)[0]
    keep = keep[np.argsort(-scores[keep])][:3]
    return [(boxes[i], float(scores[i]), int(labels[i])) for i in keep]


@torch.no_grad()
def run_tv(model):
    """Run a torchvision detector on the selected images (labels are shifted back by -1)."""
    model.to(device).eval()
    out = []
    for p in paths:
        x = to_tensor(Image.open(p).convert("RGB")).to(device)
        o = model([x])[0]
        out.append(top_dets(o["boxes"].cpu().numpy(), o["scores"].cpu().numpy(),
                            o["labels"].cpu().numpy() - 1))
    return out


preds = {}

# Model 1: YOLOv8n
yolo = YOLO(r"runs/detect/results/runs/yolov8n/weights/best.pt")
out = []
for p in paths:
    r = yolo.predict(p, imgsz=448, conf=THR, device=0, verbose=False)[0]
    out.append(top_dets(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(),
                        r.boxes.cls.cpu().numpy().astype(int)))
preds["YOLOv8n"] = out
del yolo
torch.cuda.empty_cache()

# Model 2: Faster R-CNN (pretrained weights come from the local cache, no download)
m = fasterrcnn_mobilenet_v3_large_fpn(weights="DEFAULT", min_size=320, max_size=480)
m.roi_heads.box_predictor = FastRCNNPredictor(m.roi_heads.box_predictor.cls_score.in_features, 5)
m.load_state_dict(torch.load("results/frcnn_best.pt", map_location="cpu"))
preds["Faster R-CNN"] = run_tv(m)
del m
torch.cuda.empty_cache()

# Model 3: RetinaNet
m = retinanet_resnet50_fpn(weights="DEFAULT", min_size=320, max_size=480)
m.head.classification_head = RetinaNetClassificationHead(
    256, m.head.classification_head.num_anchors, 5)
m.load_state_dict(torch.load("results/retinanet_best.pt", map_location="cpu"))
preds["RetinaNet"] = run_tv(m)
del m
torch.cuda.empty_cache()


def draw(ax, box, text, color):
    """Draw one labelled box (x1, y1, x2, y2)."""
    x1, y1, x2, y2 = box
    ax.add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False,
                                   edgecolor=color, linewidth=2))
    ax.text(x1, max(y1 - 3, 8), text, color="white", fontsize=7,
            bbox=dict(facecolor=color, alpha=0.85, pad=1, edgecolor="none"))


titles = ["Ground truth"] + list(preds.keys())
fig, axes = plt.subplots(4, 4, figsize=(15, 9))
for r in range(4):
    img = Image.open(paths[r]).convert("RGB")
    for c, title in enumerate(titles):
        ax = axes[r, c]
        ax.imshow(img)
        ax.axis("off")
        if r == 0:
            ax.set_title(title, fontsize=12)
        if c == 0:
            draw(ax, gts[r][0], NAMES[gts[r][1]], "#22aa44")
        else:
            for b, s, l in preds[title][r]:
                draw(ax, b, f"{NAMES[l]} {s:.2f}", COLORS[title])
fig.tight_layout()
fig.savefig("results/fig_qualitative.png", dpi=200)
print("Saved results/fig_qualitative.png")