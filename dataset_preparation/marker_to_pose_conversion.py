import os
import glob
import json

from pathlib import Path

from ground_truth_processing import get_marker_metadata
from dataset_processing import get_relative_pose_from_markers


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
