from .yoloV2 import get_yolo
from .resnet18 import get_resnet18_model


def get_model(name):
    print(f"Getting {name}")
    if name == "yolo5":
        return get_yolo(num_boxes=5)
    elif name == "yolo7":
        return get_yolo(num_boxes=7)
    elif name == "yolo10":
        return get_yolo(num_boxes=10)
    elif name == "resnet":
        return get_resnet18_model(False)
    else:
        raise ValueError(f"Model {name} does not exist.")
