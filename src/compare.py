import json
import pandas as pd
import matplotlib.pyplot as plt

FILES = {
    "YOLOv8n": "results/metrics_yolov8n_common.json",
    "Faster R-CNN": "results/metrics_frcnn.json",
    "RetinaNet": "results/metrics_retinanet.json",
}
COLORS = {"YOLOv8n": "#2a6fdb", "Faster R-CNN": "#e8830c", "RetinaNet": "#2a9d6f"}
CLASSES = ["speedlimit", "stop", "crosswalk", "trafficlight"]

# Build the comparison table
rows = []
for name, path in FILES.items():
    d = json.load(open(path))
    row = {"Model": name, "mAP50": d["mAP50"], "mAP50-95": d["mAP50_95"]}
    for c in CLASSES:
        row[c] = d["per_class_mAP50"][c]
    row["Inference (ms)"] = d["inference_ms_per_image"]
    row["FPS"] = 1000 / d["inference_ms_per_image"]
    row["Params (M)"] = d["params_million"]
    row["Train time (h)"] = d["train_time_hours"]
    rows.append(row)
df = pd.DataFrame(rows)
df.round(3).to_csv("results/comparison.csv", index=False)
print(df.round(3).to_string(index=False))

models = df["Model"].tolist()
cols = [COLORS[m] for m in models]

# Figure 1: overall accuracy
fig, ax = plt.subplots(figsize=(6, 4))
w = 0.35
x = range(len(models))
ax.bar([i - w / 2 for i in x], df["mAP50"], w, label="mAP50", color="#4c78a8")
ax.bar([i + w / 2 for i in x], df["mAP50-95"], w, label="mAP50-95", color="#9ecae9")
for i, (a, b) in enumerate(zip(df["mAP50"], df["mAP50-95"])):
    ax.text(i - w / 2, a + 0.005, f"{a:.3f}", ha="center", fontsize=8)
    ax.text(i + w / 2, b + 0.005, f"{b:.3f}", ha="center", fontsize=8)
ax.set_xticks(list(x))
ax.set_xticklabels(models)
ax.set_ylim(0, 1)
ax.set_ylabel("Score")
ax.set_title("Detection accuracy on the test set")
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig("results/fig_overall.png", dpi=200)

# Figure 2: per-class AP50
fig, ax = plt.subplots(figsize=(7, 4))
w = 0.25
for k, m in enumerate(models):
    vals = [df.loc[k, c] for c in CLASSES]
    ax.bar([i + (k - 1) * w for i in range(len(CLASSES))], vals, w, label=m, color=cols[k])
ax.set_xticks(range(len(CLASSES)))
ax.set_xticklabels(CLASSES)
ax.set_ylim(0, 1)
ax.set_ylabel("AP50")
ax.set_title("Per-class AP50 on the test set")
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig("results/fig_per_class.png", dpi=200)

# Figure 3: speed vs accuracy trade-off
fig, ax = plt.subplots(figsize=(6, 4))
for k, m in enumerate(models):
    ax.scatter(df.loc[k, "Inference (ms)"], df.loc[k, "mAP50-95"], s=120, color=cols[k])
    ax.annotate(m, (df.loc[k, "Inference (ms)"], df.loc[k, "mAP50-95"]),
                textcoords="offset points", xytext=(8, 6), fontsize=9)
ax.set_xscale("log")
ax.set_xlabel("Inference time per image (ms, log scale)")
ax.set_ylabel("mAP50-95")
ax.set_title("Speed vs accuracy")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("results/fig_speed_accuracy.png", dpi=200)
print("Saved figures to results/")