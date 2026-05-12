from ultralytics import YOLO

def main():
    model = YOLO('yolov8s.pt')

    results = model.train(
        data='../dataset/data.yaml',
        epochs=100,
        imgsz=640,
        batch=8,
        device=0,
        workers=4,
        cache=True,
        project='../runs',
        name='oil_spill_v1',
    )

    print(f"Training complete. Results saved to: {results.save_dir}")

if __name__ == '__main__':
    main()
