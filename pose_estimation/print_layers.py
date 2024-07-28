from torchvision import models
from torch.nn import Linear
from torch import nn
from torchinfo import summary

import torch

# from torchviz import make_dot
from models import get_model


def _get_model():
    model = get_model("yolo10")
    model = model.to("cuda")
    return model


def print_model(model, num_channels=1):
    summary(model, (1, num_channels, 128, 128))
    input = torch.randn(1, 1, 128, 128)


if __name__ == "__main__":
    model = _get_model()
    print_model(model)
