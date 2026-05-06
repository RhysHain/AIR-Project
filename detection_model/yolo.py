from ultralytics import YOLO
# Majority of setup and validation is from https://docs.ultralytics.com/modes/val/#usage-examples
# Targetted debugging, needed it within a function when choosing device 0
if __name__ == '__main__':
    # 1. Train the model
    print("Starting training")
    model = YOLO('yolov8n.pt')
    results = model.train(
        data='detection_model/signs_data/data.yaml',
        epochs=40, # Found that 40 was enough to get a reliable AI model
        imgsz=640, # Dependent on GPU
        batch=4, # Dependent on GPU
        device=0, # Choose 0 for GPU, enter 'cpu' for cpu
        name='signs_v1',
    )

    # 2. Validate
    print("\nValidating model")
    metrics = model.val()
    # T
    print(f"mAP50: {metrics.box.map50:.3f}")
    print(f"mAP50-95: {metrics.box.map:.3f}")

    # 3. Test on sample images
    print("\nTesting on validation images")
    model.predict(
        'signs_data/images/val/',
        save=True,
        conf=0.5
    )

    print("\nTraining complete")