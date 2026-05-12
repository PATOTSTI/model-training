from ultralytics import YOLO


def main():
    model = YOLO(r"F:\model-training\runs\2026-05-12_20-26-07\oil_detection_v2\weights\last.pt")  # resume from V2

    results = model.train(
        data=r"F:\model-training\dataset\data.yaml",
        epochs=350,
        imgsz=640,
        batch=16,
        device=0,
        workers=2,        # kept low for your RAM
        cache=False,      # kept off for your RAM
        patience=75,
        optimizer='AdamW',
        lr0=0.0005,       # halved — finer tuning from where V2 left off
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        flipud=0.5,
        fliplr=0.5,
        mosaic=1.0,
        degrees=15.0,
        translate=0.1,
        scale=0.5,
        project=r"F:\model-training\runs",
        name='oil_detection_v3',
    )

if __name__ == '__main__':
    main()