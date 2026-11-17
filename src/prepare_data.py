import os, shutil
import pandas as pd
from PIL import Image

SEED = 42
CSV = "data/object_detection/dataset.csv"
OUT = "data/yolo_road"
NAMES = ["speedlimit", "stop", "crosswalk", "trafficlight"]  

df = pd.read_csv(CSV)
b = df["new_bb"].str.split(expand=True).astype(int)
df["y1"], df["x1"], df["y2"], df["x2"] = b[0], b[1], b[2], b[3]
df["img"] = df["new_path"].str.replace("\\", "/", regex=False).str.replace(r"^\./", "", regex=True)


parts = []
for c, g in df.groupby("class"):
    g = g.sample(frac=1, random_state=SEED)
    n = len(g)
    n_tr, n_va = int(n * 0.70), int(n * 0.15)
    g = g.copy()
    g["split"] = ["train"] * n_tr + ["val"] * n_va + ["test"] * (n - n_tr - n_va)
    parts.append(g)
df = pd.concat(parts).sort_values("index").reset_index(drop=True)
df.to_csv("data/splits.csv", index=False)
print(pd.crosstab(df["class"], df["split"]))


for s in ["train", "val", "test"]:
    os.makedirs(f"{OUT}/images/{s}", exist_ok=True)
    os.makedirs(f"{OUT}/labels/{s}", exist_ok=True)

for _, r in df.iterrows():
    name = os.path.basename(r["img"])
    W, H = Image.open(r["img"]).size
    xc = (r["x1"] + r["x2"]) / 2 / W
    yc = (r["y1"] + r["y2"]) / 2 / H
    w = (r["x2"] - r["x1"]) / W
    h = (r["y2"] - r["y1"]) / H
    shutil.copy(r["img"], f"{OUT}/images/{r['split']}/{name}")
    with open(f"{OUT}/labels/{r['split']}/{name.replace('.png', '.txt')}", "w") as f:
        f.write(f"{int(r['class'])} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")


root = os.path.abspath(OUT).replace("\\", "/")
with open(f"{OUT}/data.yaml", "w") as f:
    f.write(f"path: {root}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n")
    for i, n in enumerate(NAMES):
        f.write(f"  {i}: {n}\n")


for c in [1, 2]:
    print("class", c, "example:", df[df["class"] == c]["img"].iloc[0])
print("Done.")