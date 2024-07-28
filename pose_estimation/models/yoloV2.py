import torch
import torch.nn as nn


def conv2dLayer(kernel_size, in_channels=10, out_channels=10, stride=1, padding="same"):
    return nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)


class YOLO10(nn.Module):
    def __init__(self, num_boxes=10, in_c=1):
        super(YOLO10, self).__init__()

        self.backbone = nn.Sequential(
            # Stage 1:
            conv2dLayer(kernel_size=num_boxes, in_channels=in_c, out_channels=16),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            #
            # Stage 2:
            conv2dLayer(kernel_size=3, in_channels=16, out_channels=32),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            #
            # Stage 3:
            conv2dLayer(kernel_size=1, in_channels=32, out_channels=96),
            nn.LeakyReLU(),
            conv2dLayer(kernel_size=3, in_channels=96, out_channels=128),
            nn.LeakyReLU(),
            conv2dLayer(kernel_size=1, in_channels=128, out_channels=96),
            nn.LeakyReLU(),
            conv2dLayer(kernel_size=3, in_channels=96, out_channels=128),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            #
            # Stage 4:
            conv2dLayer(kernel_size=1, in_channels=128, out_channels=256),
            nn.LeakyReLU(),
            conv2dLayer(kernel_size=3, in_channels=256, out_channels=128),
            nn.LeakyReLU(),
        )
        # Position branch
        self.position_branch = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Conv2d(128, num_boxes, kernel_size=1, stride=1),
            nn.Flatten(),
            nn.Linear(128 * 2 * num_boxes, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 2),
        )

        # Orientation branch
        self.orientation_branch = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Conv2d(128, num_boxes, kernel_size=1, stride=1),
            nn.Flatten(),
            nn.Linear(128 * 2 * num_boxes, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        x = self.backbone(x)
        position_output = self.position_branch(x)
        orientation_output = self.orientation_branch(x)

        return position_output, orientation_output


class YOLO7(nn.Module):
    def __init__(self, num_boxes=7, in_c=1):
        super(YOLO10, self).__init__()

        self.backbone = nn.Sequential(
            # Stage 1:
            conv2dLayer(kernel_size=num_boxes, in_channels=in_c, out_channels=16),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            #
            # Stage 2:
            conv2dLayer(kernel_size=3, in_channels=16, out_channels=32),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            #
            # Stage 3:
            conv2dLayer(kernel_size=1, in_channels=32, out_channels=96),
            nn.LeakyReLU(),
            conv2dLayer(kernel_size=3, in_channels=96, out_channels=128),
            nn.LeakyReLU(),
            conv2dLayer(kernel_size=1, in_channels=128, out_channels=96),
            nn.LeakyReLU(),
            conv2dLayer(kernel_size=3, in_channels=96, out_channels=128),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            #
            # Stage 4:
            conv2dLayer(kernel_size=1, in_channels=128, out_channels=256),
            nn.LeakyReLU(),
            conv2dLayer(kernel_size=3, in_channels=256, out_channels=128),
            nn.LeakyReLU(),
        )
        # Position branch
        self.position_branch = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Conv2d(128, num_boxes, kernel_size=1, stride=1),
            nn.Flatten(),
            nn.Linear(128 * 2 * num_boxes, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 2),
        )

        # Orientation branch
        self.orientation_branch = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Conv2d(128, num_boxes, kernel_size=1, stride=1),
            nn.Flatten(),
            nn.Linear(128 * 2 * num_boxes, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        x = self.backbone(x)
        position_output = self.position_branch(x)
        orientation_output = self.orientation_branch(x)

        return position_output, orientation_output


# Create an instance of the modified YOLO model
def get_yolo(num_boxes=10):
    return YOLO10(num_boxes)
