import json
import os
import time
import torch
from PIL import Image
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms.functional import to_tensor

from common import load_split, evaluate

EPOCHS = 30
BATCH = 4
device = torch.device("cuda")


class RoadDS(torch.utils.data.Dataset):
    """Road sign dataset. Labels are shifted by +1 because torchvision uses 0 for background."""

    def __init__(self, split, train):
        self.df = load_split(split)
        self.train = train

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        img = Image.open(r["img"]).convert("RGB")
        W = img.size[0]
        t = to_tensor(img)
        boxes = torch.tensor([[r["x1"], r["y1"], r["x2"], r["y2"]]], dtype=torch.float32)
        if self.train and torch.rand(1).item() < 0.5:      # random horizontal flip
            t = t.flip(-1)
            boxes[:, [0, 2]] = W - boxes[:, [2, 0]]
        return t, {"boxes": boxes, "labels": torch.tensor([int(r["class"]) + 1])}


def collate(b):
    return tuple(zip(*b))


@torch.no_grad()
def predict(model, ds):
    """Run the model on a dataset and return predictions and ground truths for evaluate()."""
    model.eval()
    loader = torch.utils.data.DataLoader(ds, batch_size=8, collate_fn=collate)
    preds, gts = [], []
    for imgs, tg in loader:
        out = model([im.to(device) for im in imgs])
        for o, t in zip(out, tg):
            preds.append({"boxes": o["boxes"].cpu().numpy(),
                          "scores": o["scores"].cpu().numpy(),
                          "labels": o["labels"].cpu().numpy() - 1})
            gts.append({"boxes": t["boxes"].numpy(), "labels": t["labels"].numpy() - 1})
    return preds, gts


if __name__ == "__main__":
    torch.manual_seed(42)
    os.makedirs("results", exist_ok=True)

    # COCO-pretrained lightweight Faster R-CNN; images are small, so use a small input size
    model = fasterrcnn_mobilenet_v3_large_fpn(weights="DEFAULT", min_size=320, max_size=480)
    in_f = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_f, 5)   # 4 classes + background
    model.roi_heads.score_thresh = 0.001                          # keep low-score boxes for mAP
    model.to(device)

    train_loader = torch.utils.data.DataLoader(
        RoadDS("train", True), batch_size=BATCH, shuffle=True, collate_fn=collate)
    val_ds, test_ds = RoadDS("val", False), RoadDS("test", False)

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

    best, t0 = -1, time.time()
    for ep in range(EPOCHS):
        model.train()
        total = 0
        for imgs, tg in train_loader:
            imgs = [i.to(device) for i in imgs]
            tg = [{k: v.to(device) for k, v in t.items()} for t in tg]
            loss = sum(model(imgs, tg).values())
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
        sched.step()

        # Select the best epoch on the validation split
        p, g = predict(model, val_ds)
        m = evaluate(p, g)
        print(f"epoch {ep + 1}/{EPOCHS} loss {total / len(train_loader):.4f} "
              f"val mAP50 {m['mAP50']:.4f} mAP50-95 {m['mAP50_95']:.4f}", flush=True)
                # Same selection rule as YOLO: 10% mAP50 + 90% mAP50-95
        fitness = 0.1 * m["mAP50"] + 0.9 * m["mAP50_95"]
        if fitness > best:
            best = fitness
            torch.save(model.state_dict(), "results/frcnn_best.pt")
    train_hours = (time.time() - t0) / 3600

    # Final evaluation on the TEST split with the best checkpoint
    model.load_state_dict(torch.load("results/frcnn_best.pt"))
    p, g = predict(model, test_ds)
    res = evaluate(p, g)

    # Inference speed: one image at a time
    model.eval()
    imgs = [test_ds[i][0].to(device) for i in range(len(test_ds))]
    with torch.no_grad():
        model([imgs[0]])
        torch.cuda.synchronize()
        s = time.time()
        for im in imgs:
            model([im])
        torch.cuda.synchronize()
    res.update({
        "model": "Faster R-CNN MobileNetV3-FPN",
        "inference_ms_per_image": round((time.time() - s) / len(imgs) * 1000, 2),
        "train_time_hours": round(train_hours, 3),
        "params_million": round(sum(x.numel() for x in model.parameters()) / 1e6, 2),
    })
    print(json.dumps(res, indent=2))
    with open("results/metrics_frcnn.json", "w") as f:
        json.dump(res, f, indent=2)