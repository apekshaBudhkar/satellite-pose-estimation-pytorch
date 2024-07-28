from pathlib import Path
import math
import numpy as np
import json
import click

from image_processing import process_image, is_binary_datafile
from ground_truth_processing import get_marker_metadata

import cv2

from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec


DATA_DIRECTORY = "/vivado/spot_data/2024/feb_18_1535"


def load_json(filepath):
    with open(filepath) as f:
        return json.load(f)


@click.command()
@click.option("-d", "--data_root", required=False, default=DATA_DIRECTORY)
@click.option("-o", "--output_filename", required=False, default=None)
def main(data_root, output_filename):
    json_filename = f"{Path(data_root).stem}.json"
    json_filepath = Path(data_root).joinpath(json_filename)
    

    data = load_json(json_filepath)
    for point in data:
        x = data[point]["x"]
        y = data[point]["y"]
        yaw = data[point]["yaw"]

        gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1])
        ax0 = plt.subplot(gs[0])
        ax1 = plt.subplot(gs[1])

        image = cv2.imread(f"{data_root}/{point}.png")
        ax0.imshow(image, cmap="gray")
        ax1.arrow(x, y, 0.2 * np.cos(yaw), 0.2 * np.sin(yaw))
        plt.show()


if __name__ == "__main__":
    main()
