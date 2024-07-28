from torch import nn


def get_decoder_branch(in_features: int = 512, out_features: int = 2) -> nn.Sequential:
    branch = nn.Sequential(
        nn.Flatten(),
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, out_features)
    )

    return branch
