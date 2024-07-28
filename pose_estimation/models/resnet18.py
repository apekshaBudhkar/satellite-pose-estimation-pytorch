from torchvision import models
from torch.nn import Linear
from torch import nn


class Resnet18Model(nn.Module):
    def __init__(self, is_pretrained):
        super(Resnet18Model, self).__init__()
        self.backbone = models.resnet18(pretrained=is_pretrained)
        num_ftrs = self.backbone.fc.out_features

        self.position_branch = Linear(num_ftrs, 2)
        self.orientation_branch = Linear(num_ftrs, 1)

    def forward(self, x):
        output = self.backbone(x)
        position = self.position_branch(output)
        orientation = self.orientation_branch(output)

        return position, orientation


def get_resnet18_model(pretrained):
    return Resnet18Model(is_pretrained=pretrained)
