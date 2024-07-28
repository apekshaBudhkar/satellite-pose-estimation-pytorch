import numpy as np


def apply_agc(image):
    min = np.min(image)
    max = np.max(image)
    span = max - min
    span = span if (span > 0.0) else 1.0

    result = (image - min) * (255.0 / span)
    np.clip(result, 0, 255, result)
    return result.astype("uint8")


def convert(buffer):
    w, h = 320, 240
    np_matrix = np.resize(buffer, (h, w))
    np_matrix = np_matrix.astype("float32")
    image = apply_agc(np_matrix)
    return image


def get_buffer(path):
    with open(path, mode="rb") as f:
        for b in f:
            np_arr = np.frombuffer(b, dtype=np.float32)
    return np_arr


def process_image(bin_file):
    b = get_buffer(bin_file)
    return convert(b)


def is_binary_datafile(file) -> bool:
    return file.suffix == ".bin" and file.stem.startswith("data_")
