import torch
import torch.optim as optim
from torchvision import transforms


from datetime import datetime as dt
import click
from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np
from print_layers import print_model

import os

from dataloaders.dataloader_infrared import (
    InfraredPoseEstimationDSSimplified as InfraredDS,
)

from models import get_model

default_data_dir = "/vivado/spot_data/2024/experimentv5/"


default_transforms = [transforms.Resize((128, 128)), transforms.ToTensor()]
extra_transforms = [
    transforms.RandomAdjustSharpness(1.1, p=0.5),
    transforms.RandomAutocontrast(p=0.5),
]


def _get_train_val_loaders(
    data_root: str,
    batch_size=8,
    shuffle=True,
    apply_transforms=False,
    single_channel=True,
):
    dataloader_dict = {}
    if apply_transforms:
        transform = transforms.Compose(default_transforms + extra_transforms)
    else:
        transform = transforms.Compose(default_transforms)

    for phase in ["train", "validation"]:

        set = InfraredDS(
            data_root=data_root,
            split=phase,
            transform=transform,
            load_as_single_channel=single_channel,
        )
        dataloader_dict[phase] = torch.utils.data.DataLoader(
            set, batch_size=batch_size, shuffle=shuffle
        )

    return dataloader_dict


def _get_timestamp():
    ts = dt.now()
    timestamp_str = f"{ts.year}-{ts.month}-{ts.day}_{ts.hour}-{ts.minute}"
    return timestamp_str


def _save_model_to_disk(model: any, model_id: str, epoch: str, directory: str):
    ts = _get_timestamp()
    filename = f"epoch{epoch}_{model_id}_{ts}.pth"
    path = Path.joinpath(Path(directory), filename)
    torch.save(model.state_dict(), path)


def _plot_variable(plt, array_of_vals, title, style, color):
    x = range(0, len(array_of_vals))
    plt.plot(x, array_of_vals, label=title, linestyle=style, color=color)
    return plt


def _save_loss_to_disk(
    train_loss_pos,
    train_loss_ori,
    val_loss_pos,
    val_loss_ori,
    model_id,
    directory,
):
    plt.clf()
    plot = plt.axes()

    plot = _plot_variable(
        plot, train_loss_pos, "Training loss: position (m)", "solid", "red"
    )
    plot = _plot_variable(
        plot, train_loss_ori, "Training loss: orientation (rad)", "solid", "black"
    )
    plot = _plot_variable(plot, val_loss_pos, "Val loss: position (m)", "dotted", "red")
    plot = _plot_variable(
        plot, val_loss_ori, "Val loss: orientation (rad)", "dotted", "black"
    )

    plt.title("Loss Curve")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(color="0.95")

    ts = dt.now()
    timestamp_str = f"{ts.year}-{ts.month}-{ts.day}"
    fname = f"epoch{len(train_loss_pos)}_{timestamp_str}_{model_id}.png"

    np.savez(
        f"{directory}/{model_id}",
        train_loss_pos=train_loss_pos,
        train_loss_ori=train_loss_ori,
        val_loss_pos=val_loss_pos,
        val_loss_ori=val_loss_ori,
    )
    path = Path.joinpath(Path(directory), fname)
    print(f"Saving to {path}")
    plt.savefig(path)


def _get_position_and_orientation_labels(poses):
    """
    Separates the position and orientation by slicing the tensor.
    """
    pos_indices = torch.tensor([0, 1]).to("cuda")
    ori_indices = torch.tensor([2]).to("cuda")
    pos_gt = torch.index_select(poses, 1, pos_indices).to("cuda")
    ori_gt = torch.index_select(poses, 1, ori_indices).to("cuda")

    return pos_gt, ori_gt


def OrientationLoss(prediction, actual):
    prediction = (prediction + torch.pi) % (2 * torch.pi) - torch.pi
    actual = (actual + torch.pi) % (2 * torch.pi) - torch.pi

    angle_diff = prediction - actual
    angle_diff = (angle_diff + torch.pi) % (2 * torch.pi) - torch.pi
    angle_error = torch.abs(angle_diff)
    loss = torch.mean(angle_error)
    return loss


def CosineLoss(pred, gt):
    loss = torch.nn.CosineSimilarity()
    # angle_diff = (angle_diff + torch.pi) % (2 * torch.pi) - torch.pi
    output = loss(pred, gt)
    output = torch.abs(output)
    return torch.mean(output)


@click.command()
@click.option("-c", "--checkpoint_dir", required=False, default="./checkpoints/")
@click.option("-m", "--model_name", required=False, default="yolo")
@click.option("-n", "--num_epochs", required=False, default="50")
@click.option("-d", "--data_root", required=False, default=default_data_dir)
@click.option("-t", "--apply_transforms", is_flag=True, default=False)
def main(checkpoint_dir, model_name, num_epochs, data_root, apply_transforms):
    model = get_model(model_name)
    single_channel = True
    if model_name == "resnet" or model_name == "resnet_pretrained":
        single_channel = False
        print_model(model, 3)
    else:
        print_model(model)

    model = model.to("cuda")
    criterion_pos = torch.nn.MSELoss()

    lr_position = 1e-7
    lr_orientation = 1e-4

    momentum = 0.9
    optimizer_pos = optim.SGD(model.parameters(), lr=lr_position, momentum=momentum)
    optimizer_ori = optim.Adam(model.parameters(), lr=lr_orientation)

    dataloaders = _get_train_val_loaders(
        data_root,
        batch_size=4,
        shuffle=True,
        apply_transforms=True,
        single_channel=single_channel,
    )

    log_dict: dict = {
        "train": {"pos": [], "ori": []},
        "validation": {"pos": [], "ori": []},
    }

    running_loss_dict = {
        "train": {"pos": 0, "ori": 0},
        "validation": {"pos": 0, "ori": 0},
    }

    save_directory = f"{checkpoint_dir}/{model_name}_{_get_timestamp()}"
    os.makedirs(save_directory)

    for epoch in range(1, int(num_epochs) + 1):
        if epoch == 25:
            lr_orientation = lr_orientation / 10
            optimizer_ori = optim.Adam(model.parameters(), lr=lr_orientation)

        if epoch == 30:
            lr_orientation = lr_orientation / 10
            optimizer_ori = optim.Adam(model.parameters(), lr=lr_orientation)

        for phase in ["train", "validation"]:
            running_loss_dict[phase]["pos"] = 0
            running_loss_dict[phase]["ori"] = 0
            if phase == "train":
                model.train(True)
            else:
                model.eval()

            dataset = dataloaders[phase]
            for _, (id, images, poses) in enumerate(dataset):
                images = images.to("cuda")
                poses = poses.to("cuda")
                pos_gt, ori_gt = _get_position_and_orientation_labels(poses)

                with torch.set_grad_enabled(phase == "train"):
                    # Zero the gradients for every batch.
                    optimizer_pos.zero_grad()
                    optimizer_ori.zero_grad()

                    # Make a prediction. Separate the position and orientation arguments.
                    outputs = model(images.float())
                    pos_prediction = outputs[0].to("cuda")
                    ori_prediction = outputs[1].to("cuda")

                    loss_pos = criterion_pos(pos_prediction.float(), pos_gt.float())
                    loss_ori = OrientationLoss(ori_prediction.float(), ori_gt.float())

                if phase == "train":
                    loss_pos.backward(retain_graph=True)
                    loss_ori.backward(retain_graph=True)

                    optimizer_pos.step()
                    optimizer_ori.step()

                # Adjust the weights
                running_loss_dict[phase]["pos"] += loss_pos.item()
                running_loss_dict[phase]["ori"] += loss_ori.item()

            print(f"Predict:\t{outputs}")
            print(f"Ground Truth:\t{pos_gt}, {ori_gt}")

            epoch_loss_pos = running_loss_dict[phase]["pos"] / len(dataset)
            epoch_loss_ori = running_loss_dict[phase]["ori"] / len(dataset)
            log_dict[phase]["pos"].append(epoch_loss_pos)
            log_dict[phase]["ori"].append(epoch_loss_ori)

            print(
                f"[{epoch}/{num_epochs}]: {epoch_loss_pos}, {epoch_loss_ori} Phase: {phase}\n"
            )

        if (epoch) % 5 == 0 and phase == "validation":
            _save_model_to_disk(model, model_name, str(epoch), save_directory)
            _save_loss_to_disk(
                log_dict["train"]["pos"],
                log_dict["train"]["ori"],
                log_dict["validation"]["pos"],
                log_dict["validation"]["ori"],
                model_name,
                save_directory,
            )


if __name__ == "__main__":
    main()
