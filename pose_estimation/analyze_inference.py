import click
import json
from matplotlib import pyplot as plt
import numpy as np

from pathlib import Path
import cv2


def get_json_data_from_file(filepath: str) -> json:
    with open(filepath, "r") as data:
        return json.load(data)


def sort_losses_by_worst_position(model_outputs):
    return sorted(model_outputs, key=lambda x: abs(x["losses"][0]), reverse=True)


def sort_losses_by_worst_orientation(model_outputs):
    return sorted(model_outputs, key=lambda x: x["losses"][1], reverse=True)


dataset_root = "/vivado/spot_data/2024/experimentv4/test/images/"


def plot_image(id):
    print(id)
    img_path = Path.joinpath(Path(dataset_root), id)
    img = cv2.imread(str(img_path))
    cv2.imshow("win", img)
    cv2.waitKey(0)


@click.command()
@click.option("-i", "--input", required=True)
def main(input):
    data = get_json_data_from_file(input)

    orientation_data = sort_losses_by_worst_orientation(data["model_performance"])
    # position_data = sort_losses_by_worst_position(data["model_performance"])

    losses_orientation = []
    for d in data["model_performance"]:
        losses_orientation.append(d["losses"])

    print(np.median(losses_orientation))
    worst_loss = orientation_data[-20:]

    for loss in worst_loss:
        print(loss)
        image_file = loss["id"][0]
        plot_image(image_file)


if __name__ == "__main__":
    main()
