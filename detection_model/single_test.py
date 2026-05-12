from ultralytics import YOLO

# Load the best model located within the following directory
model = YOLO('detection_model/runs/detect/signs_v1-6/weights/best.pt')

# Test it
results = model.predict('detection_model/test_image.png', conf=0.6)
results[0].show()