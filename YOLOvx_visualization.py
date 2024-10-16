# Importing essential libraries
import torch
from ultralytics import YOLO
import matplotlib.pyplot as plt
from visualtorch import layered_view

# loading the models and formating them so that the become readable in visualtorch
model_YOLOv3 = YOLO('yolov3.pt').model.model
model_YOLOv5n = YOLO('yolov5n.pt').model.model
model_YOLOv8n = YOLO('yolov8n.pt').model.model
model_YOLOv8m = YOLO('yolov8m.pt').model.model
model_YOLOv8l = YOLO('yolov8l.pt').model.model

# Defining the proper input shape for the models
input_shape = (1, 3, 224, 224)

# Defining a function for illustrating the visualization for input model
def yolo_model_illustrate(model_YOLOvx):
  img = visualtorch.layered_view(
    torch.nn.Sequential(*list(model_YOLOvx.children())[:3]),
    input_shape=input_shape,
    one_dim_orientation="x",
    spacing=10,legend=True,
  )
  plt.axis("off")
  plt.tight_layout()
  plt.imshow(img)
  plt.show()

# visuliaztion for the first 3 children's block layers in YOLOv3 model
yolo_model_illustrate(model_YOLOv3)

# visuliaztion for the first 3 children's block layers in YOLOv5n model
yolo_model_illustrate(model_YOLOv5n)

# visuliaztion for the first 3 children's block layers in YOLOv8n model
yolo_model_illustrate(model_YOLOv8n)

# visuliaztion for the first 3 children's block layers in YOLOv3 model
yolo_model_illustrate(model_YOLOv8m)

# visuliaztion for the first 3 children's block layers in YOLOv3 model
yolo_model_illustrate(model_YOLOv8l)
