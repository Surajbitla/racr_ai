"""module to tie in implemented models"""

from torchvision import models
from ultralytics import YOLO


def model_selector(model_name, weights_path=None):
    if "alexnet" in model_name:
        return models.alexnet(weights="DEFAULT")
    elif "yolo" in model_name:
        if weights_path:
            yolo = YOLO(weights_path)
        else:
            yolo = YOLO(str(model_name) + ".pt")
        return yolo.model  # pop the real model out of their wrapper for now
    else:
        raise NotImplementedError
