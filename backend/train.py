from ultralytics import YOLO

# Load a model
model = YOLO("yolov8m.yaml")  # build a new model from scratch

# Use the model
model.train(data="dataset.yaml", epochs=100)  # train the model
