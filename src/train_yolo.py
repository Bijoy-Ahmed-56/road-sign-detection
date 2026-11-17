from ultralytics import YOLO

if __name__ == "__main__":
    # Load the pretrained YOLOv8 nano model (smallest, fits in 2 GB VRAM)
    model = YOLO("yolov8n.pt")

    # Train on our road sign dataset
    model.train(
        data="data/yolo_road/data.yaml",  # dataset config (paths + class names)
        epochs=50,
        imgsz=448,        # images are 447x300, so 448 (multiple of 32)
        batch=8,          # reduce to 4 if CUDA out of memory
        device=0,         # use the first GPU
        workers=0,        # avoids multiprocessing issues on Windows
        amp=False,        # mixed precision not useful on MX330; also skips the AMP check
        seed=42,          # fixed seed for reproducibility
        project="results/runs",
        name="yolov8n",
        exist_ok=True,    # overwrite the folder if it already exists
    )

    # Evaluate the best checkpoint on the held-out test split
    best = YOLO("results/runs/yolov8n/weights/best.pt")
    m = best.val(
        data="data/yolo_road/data.yaml",
        split="test",
        imgsz=448,
        batch=8,
        device=0,
        workers=0,
        project="results/runs",
        name="yolov8n_test",
        exist_ok=True,
    )

    # Print the key metrics for the report
    print("TEST mAP50:", round(m.box.map50, 4))
    print("TEST mAP50-95:", round(m.box.map, 4))
    print("Per-class mAP50:", [round(x, 4) for x in m.box.ap50])