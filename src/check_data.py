import os
import pandas as pd
from PIL import Image, ImageDraw

CSV = r"data/object_detection/dataset.csv"
ROOT = r"data/object_detection/road_signs"

df = pd.read_csv(CSV)
print("Rows:", len(df))
print(df["class"].value_counts().sort_index())


missing = [p for p in df["new_path"] if not os.path.exists(p)]
print("Missing images:", len(missing))
if missing:
    print("Example missing:", missing[0])


sizes = {}
for p in df["new_path"][:100]:
    if os.path.exists(p):
        sizes[Image.open(p).size] = sizes.get(Image.open(p).size, 0) + 1
print("Image sizes (width, height):", sizes)


os.makedirs("results", exist_ok=True)
for i in [0, 3, 13]:
    row = df.iloc[i]
    a, b, c, d = map(int, row["new_bb"].split())
    img = Image.open(row["new_path"]).convert("RGB")
    dr = ImageDraw.Draw(img)
    dr.rectangle([b, a, d, c], outline="lime", width=3)   # y1 x1 y2 x2
    dr.rectangle([a, b, c, d], outline="red", width=3)    # x1 y1 x2 y2
    img.save(f"results/check_box_{i}.png")
    print(i, "class", row["class"], "bb", row["new_bb"])


for r, ds, fs in os.walk(ROOT):
    xmls = [f for f in fs if f.endswith(".xml")]
    print(r, "| files:", len(fs), "| xml:", len(xmls))
    break
print("Folders in road_signs:", os.listdir(ROOT))