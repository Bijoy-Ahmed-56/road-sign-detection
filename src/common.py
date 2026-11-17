import numpy as np
import pandas as pd

NAMES = ["speedlimit", "stop", "crosswalk", "trafficlight"]


def load_split(split):
    """Return the rows of data/splits.csv belonging to one split (train/val/test)."""
    df = pd.read_csv("data/splits.csv")
    return df[df["split"] == split].reset_index(drop=True)


def box_iou(a, b):
    """IoU between one box a (4,) and many boxes b (M,4), format x1 y1 x2 y2."""
    x1 = np.maximum(a[0], b[:, 0]); y1 = np.maximum(a[1], b[:, 1])
    x2 = np.minimum(a[2], b[:, 2]); y2 = np.minimum(a[3], b[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a + area_b - inter + 1e-9)


def _ap(rec, prec):
    """101-point interpolated average precision (COCO style)."""
    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([1.0], prec, [0.0]))
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    x = np.linspace(0, 1, 101)
    return float(np.mean(np.interp(x, mrec, mpre)))


def evaluate(preds, gts, nc=4):
    """
    preds: list (one per image) of dicts {boxes Nx4, scores N, labels N}
    gts:   list (one per image) of dicts {boxes Mx4, labels M}
    Labels are 0..nc-1. Returns mAP50, mAP50-95 and per-class AP50.
    """
    thrs = np.arange(0.5, 0.96, 0.05)
    ap = np.full((nc, len(thrs)), np.nan)
    for c in range(nc):
        n_gt = sum(int((g["labels"] == c).sum()) for g in gts)
        if n_gt == 0:
            continue
        dets = []
        for i, p in enumerate(preds):
            for b, s, l in zip(p["boxes"], p["scores"], p["labels"]):
                if l == c:
                    dets.append((float(s), i, b))
        dets.sort(key=lambda d: -d[0])
        for t, thr in enumerate(thrs):
            used = [np.zeros(len(g["labels"]), dtype=bool) for g in gts]
            tp = np.zeros(len(dets))
            for d, (_, i, b) in enumerate(dets):
                idx = np.where(gts[i]["labels"] == c)[0]
                if len(idx) == 0:
                    continue
                ious = box_iou(b, gts[i]["boxes"][idx])
                j = int(ious.argmax())
                if ious[j] >= thr and not used[i][idx[j]]:
                    tp[d] = 1
                    used[i][idx[j]] = True
            ctp = np.cumsum(tp)
            cfp = np.cumsum(1 - tp)
            rec = ctp / n_gt
            prec = ctp / (ctp + cfp + 1e-9)
            ap[c, t] = _ap(rec, prec) if len(dets) else 0.0
    return {
        "mAP50": float(np.nanmean(ap[:, 0])),
        "mAP50_95": float(np.nanmean(ap)),
        "per_class_mAP50": {n: float(ap[i, 0]) for i, n in enumerate(NAMES)},
    }