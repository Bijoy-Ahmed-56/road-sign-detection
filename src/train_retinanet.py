import json
import time
import torch
from torchvision.models.detection import retinanet_resnet50_fpn
from torchvision.models.detection.retinanet import RetinaNetClassificationHead

from common import evaluate
# Reuse the dataset, collate function and predict() from the Faster R-CNN script
from train_frcnn import RoadDS, collate, predict, device

EPOCHS = 30
BATCH = 2        # RetinaNet-ResNet50 is heavier; keep the batch small for 2 GB VRAM
LR = 0.0025      # lower learning rate for the smaller batch

if __name__ == "__main__":
    torch.manual_seed(42)

    # COCO-pretrained RetinaNet; small input size because our images are 447x300
    model = retinanet_resnet50_fpn(weights="DEFAULT", min_size=320, max_size=480)
    # Replace the classification head: 4 classes + background
    num_anchors = model.head.classification_head.num_anchors
    model.head.classification_head = RetinaNetClassificationHead(256, num_anchors, 5)
    model.score_thresh = 0.001      # keep low-score boxes so mAP can be computed
    model.to(device)

    train_loader = torch.utils.data.DataLoader(
        RoadDS("train", True), batch_size=BATCH, shuffle=True, collate_fn=collate)
    val_ds, test_ds = RoadDS("val", False), RoadDS("test", False)

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=LR, momentum=0.9, weight_decay=5e-4)
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
            torch.nn.utils.clip_grad_norm_(params, 10.0)   # avoid exploding gradients
            opt.step()
            total += loss.item()
        sched.step()

        # Select the best epoch on validation (same rule as YOLO and Faster R-CNN)
        p, g = predict(model, val_ds)
        m = evaluate(p, g)
        print(f"epoch {ep + 1}/{EPOCHS} loss {total / len(train_loader):.4f} "
              f"val mAP50 {m['mAP50']:.4f} mAP50-95 {m['mAP50_95']:.4f}", flush=True)
        fitness = 0.1 * m["mAP50"] + 0.9 * m["mAP50_95"]
        if fitness > best:
            best = fitness
            torch.save(model.state_dict(), "results/retinanet_best.pt")
    train_hours = (time.time() - t0) / 3600

    # Final evaluation on the TEST split
    model.load_state_dict(torch.load("results/retinanet_best.pt"))
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
        "model": "RetinaNet ResNet50-FPN",
        "inference_ms_per_image": round((time.time() - s) / len(imgs) * 1000, 2),
        "train_time_hours": round(train_hours, 3),
        "params_million": round(sum(x.numel() for x in model.parameters()) / 1e6, 2),
    })
    print(json.dumps(res, indent=2))
    with open("results/metrics_retinanet.json", "w") as f:
        json.dump(res, f, indent=2)