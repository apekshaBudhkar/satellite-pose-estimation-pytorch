from pathlib import Path
import json
import shutil
import glob
import os

from ground_truth_processing import get_marker_metadata
from dataset_processing import get_relative_pose_from_markers

"""
This script is run after the convert_markers_to_poses script. It:

1. Iterates over the folders in the FOLDER_NAMES array, and finds all pngs, copies them over to the "DESTINATION" folder, along with the relevant json files. 
2. Combines all the json files in the directory into a single json.


"""

DATA_ROOT = "/vivado/spot_data/2024/"
FOLDER_NAMES = [
    "feb_18_1455",
    "feb_18_1508",
    "feb_18_1518",
    "feb_18_1528",
    "feb_18_1535",
    "feb_18_1555",
    "feb_18_1648",
    "feb_23_1746",
    "feb_23_1806",
    "feb_23_1824",
    "feb_23_1854",
    "mar21_1945",
    "mar21_1952",
    "mar21_2003",
    "mar21_2013",
    "mar21_2021",
    "mar21_2036",
    "mar23_1836",
    "mar23_1848",
    "mar23_1901",
    "mar23_1920",
    "mar23_1939",
    "mar23_1959",
]
DESTINATION = Path(DATA_ROOT).joinpath("experimentv5/combined")
OUTPUT_JSON_FILE = DESTINATION.joinpath("poses.json")


def convert_markers_to_poses(folder):
    folder_name = Path(folder).name
    json_file_path = Path(folder).joinpath(f"{folder_name}.json")
    gt_json_array: json = {}
    png_file_ids = glob.glob(os.path.join(folder, "*.png"))
    err_count = 0
    for png_file in png_file_ids:
        id = Path(png_file).stem
        text_file = Path(folder).joinpath(f"{id}.txt")
        try:
            markers = get_marker_metadata(str(text_file))
            x, y, yaw = get_relative_pose_from_markers(markers)
            gt_json_array[id] = {"x": x, "y": y, "yaw": yaw}
        except:
            print(f"An error occured while processing {id}")
            err_count += 1

    Path(json_file_path).write_text(json.dumps(gt_json_array))


def main():
    # Copy everything
    for folder in FOLDER_NAMES:
        dataset_path = Path(DATA_ROOT).joinpath(folder)
        convert_markers_to_poses(dataset_path)

        json_file_path = Path(dataset_path).joinpath(f"{folder}.json")
        png_file_ids = glob.glob(os.path.join(dataset_path, "*.png"))

        for png_file in png_file_ids:
            png_filename = Path(png_file).name
            raw_file = Path(png_file).with_suffix(".txt")

            png_dest_path = DESTINATION.joinpath(png_filename)
            raw_dest_path = DESTINATION.joinpath(raw_file.name)

            try:
                shutil.copy(png_file, png_dest_path)
                shutil.copy(raw_file, raw_dest_path)
            except:
                print(f"An error occured. Skipping {raw_file}")
            if not os.path.exists(png_dest_path):
                raise ValueError(f"File {png_dest_path} did not copy.")
        shutil.copy(json_file_path, DESTINATION)

    # Combine JSONS
    jsons = [DESTINATION.joinpath(f"{f}.json") for f in FOLDER_NAMES]

    result: json = {}
    for file in jsons:
        with open(file, "r") as f:
            data = json.load(f)
            for id in data:
                result[id] = data[id]

        print(f"Length of {file} is {len(data)}")
    with open(OUTPUT_JSON_FILE, "w") as f:
        json.dump(result, f)

    with open(OUTPUT_JSON_FILE, "r") as f:
        data = json.load(f)
        print(f"Final length of {OUTPUT_JSON_FILE} is {len(data)}")


if __name__ == "__main__":
    main()
