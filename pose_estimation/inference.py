import torch
from dataloaders.dataloader_infrared import (
    InfraredPoseEstimationDSSimplified as InfraredDS,
)
from torchvision import transforms
import torch.nn as nn
import click
import json
from pathlib import Path
import numpy as np

from models.models import get_model

data_root = "/vivado/spot_data/2024/experimentv5"


def save_to_disk(model_name, outputs, output_dir):
    json_path = Path(output_dir).joinpath(f"{model_name}.json")
    with open(json_path, "w") as f:
        json.dump(outputs, f)


def __get_np_array(position_tensor, orientation_tensor):
    pos = position_tensor.detach().cpu().numpy()[0]
    ori = orientation_tensor.detach().cpu().numpy()[0]
    arr = np.array([pos[0], pos[1], ori[0]])

    return arr.tolist()


def extract_filename(fpath):
    return Path(fpath).stem


def get_dataloader(dir, tfs, single_channel):
    dataset_array = []

    for split in ["test"]:
        set = InfraredDS(
            data_root=dir,
            split=split,
            transform=tfs,
            load_as_single_channel=single_channel,
        )
        dataset_array.append(set)

    combined = torch.utils.data.ConcatDataset(dataset_array)
    combined_loader = torch.utils.data.DataLoader(combined, batch_size=1, shuffle=False)

    return combined_loader


def GetOrientationError(prediction, gt):
    delta = prediction - gt
    # loss = torch.atan2(torch.sin(delta), torch.cos(delta))
    loss = torch.divide(torch.sin(delta), torch.cos(delta))

    return torch.abs(torch.mean(loss))


@click.command()
@click.option("-p", "--model_path", required=True)
@click.option("-m", "--model_name", required=False, default="yolo10")
@click.option("-d", "--data_root", required=False, default=data_root)
@click.option("-o", "--output_dir", required=False, default="outputs")
@click.option("-f", "--filename", required=False, default=None)
def main(
    model_path,
    model_name,
    data_root,
    output_dir,
    filename
):

    model = get_model(model_name)
    model = model.to("cuda")
    single_channel = True
    if model_name == "resnet" or model_name == "resnet_pretrained":
        single_channel = False

    model.load_state_dict(torch.load(model_path))
    default_transform = transforms.Compose(
        [transforms.Resize((128, 128)), transforms.ToTensor()]
    )

    test_loader = get_dataloader(data_root, default_transform, single_channel)

    results = []
    pos_loss = []
    ori_loss = []

    total_pos_loss = 0
    total_ori_loss = 0

    loss_distance = nn.MSELoss()

    for _, (id, images, poses) in enumerate(test_loader):
        loss_angle = nn.MSELoss()

        images = images.to("cuda")
        poses = poses.to("cuda")

        # Separate the position and orientation by slicing the tensor.
        pos_indices = torch.tensor([0, 1]).to("cuda")
        ori_indices = torch.tensor([2]).to("cuda")
        pos_gt = torch.index_select(poses, 1, pos_indices).to("cuda")
        ori_gt = torch.index_select(poses, 1, ori_indices).to("cuda")

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        outputs = model(images.float())
        end.record()

        torch.cuda.synchronize()
        time_taken = start.elapsed_time(end)

        pos_pred = outputs[0].to("cuda")
        ori_prediction = outputs[1].to("cuda")

        ori_prediction = torch.atan2(
            torch.sin(ori_prediction), torch.cos(ori_prediction)
        )
        ori_gt = torch.atan2(torch.sin(ori_gt), torch.cos(ori_gt))

        # Compute the current loss
        loss_pos = loss_distance(pos_pred.float(), pos_gt.float())
        loss_ori = GetOrientationError(ori_prediction.float(), ori_gt.float())

        # ori_tensor = torch.select(ori_pred, 1, torch.tensor([0, 1]).to("cuda"))
        # angle_pred = torch.atan2(ori_pred[0][1], ori_pred[0][0])
        data = {
            "id": id,
            "actual": __get_np_array(pos_gt, ori_gt),
            "predictions": __get_np_array(pos_pred, ori_prediction),
            "losses": list([loss_pos.item(), loss_ori.item()]),
            "elapsed_time": time_taken,
        }

        total_pos_loss += loss_pos.item()
        total_ori_loss += loss_ori.item()
        pos_loss.append(loss_pos.item())
        ori_loss.append(loss_ori.item())

        results.append(data)

    mean_pos_loss = total_pos_loss / len(test_loader)
    mean_ori_loss = total_ori_loss / len(test_loader)

    med_pos_loss = np.median(pos_loss)
    med_ori_loss = np.median(ori_loss)
    total_params = sum(p.numel() for p in model.parameters())

    output_json = {}
    output_json["model_performance"] = results
    output_json["model_summary"] = {
        "params": total_params,
        "mean_pos_loss": mean_pos_loss,
        "mean_ori_loss": mean_ori_loss,
        "med_pos_loss": med_pos_loss,
        "med_ori_loss": med_ori_loss,
    }

    save_to_disk(extract_filename(model_path), output_json, output_dir)


if __name__ == "__main__":
    main()
