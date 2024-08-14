from ultralytics import YOLO
import torch

torch.cuda.set_device(0)

# Load a model
model = YOLO("yolov8n.yaml")  # build a new model from scratch

# Use the model
model.train(data="config.yaml", epochs=30, workers=0, batch=1)  # train the model