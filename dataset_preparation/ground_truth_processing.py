import json


def get_marker_data_json(marker_data: str) -> json:
    data = json.loads(marker_data)
    marker_json = {"id": data["id"], "x": data["x"], "y": data["y"]}
    return marker_json


def get_marker_metadata(filename: str) -> list:
    with open(filename, "r") as f:
        lines = f.read().split("\n")
        markers_a = list(map(get_marker_data_json, lines[1:9]))
        # markers_b = list(map(get_marker_data_json, lines[12:20]))
        return markers_a
