import sys
from pathlib import Path
from ultralytics import YOLO

WEIGHTS = Path('../runs/detect/oil_spill_v1/weights/best.pt')


def predict(image_path: str) -> None:
    model = YOLO(str(WEIGHTS))
    results = model.predict(source=image_path, save=True, conf=0.25)

    for result in results:
        print(f"Image: {result.path}")
        if result.boxes is None or len(result.boxes) == 0:
            print("  No detections.")
            continue
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = box.conf[0].item()
            cls = int(box.cls[0].item())
            label = model.names[cls]
            print(f"  [{label}] conf={conf:.3f}  bbox=({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f})")
        print(f"  Output saved to: {result.save_dir}")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python predict.py <image_path>")
        sys.exit(1)
    predict(sys.argv[1])
