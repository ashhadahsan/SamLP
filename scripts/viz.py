import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import os
import glob
import random
import argparse


def get_random_data_point(root_dir):
    root_dir_train = os.path.join(root_dir, "training")

    track_dirs = [
        d for d in glob.glob(os.path.join(root_dir_train, "track*")) if os.path.isdir(d)
    ]

    txt_files = []
    png_files = []

    for track_dir in track_dirs:
        txt_files.extend(glob.glob(os.path.join(track_dir, "*.txt")))
        png_files.extend(glob.glob(os.path.join(track_dir, "*.png")))

    txt_files.sort()
    png_files.sort()

    zipped_files = list(zip(txt_files, png_files))
    idx = random.randint(0, len(zipped_files))
    return zipped_files[idx]


def extract_metadata(data: str):
    char_positions = [x.split(": ")[1].split() for x in data.splitlines()[8:]]
    vehicle_position = data.splitlines()[1].split(":")[1].split()
    coordinates = data.splitlines()[7].split(":")[1].strip().split(" ")
    corners = []
    for coord in coordinates:
        x, y = coord.split(",")
        corners.append((int(x), int(y)))
    meta_data = data.split("postion_vehicle")[0].replace("\t", "\n").splitlines()
    meta_data = [x for x in meta_data if x != ""]
    camera = meta_data[0].split(":")[1].strip().title()
    type_v = meta_data[2].split(":")[1].strip().title()
    type_make = meta_data[3].split(":")[1].strip().title().strip().title()
    type_model = meta_data[4].split(":")[1].strip().title()
    type_year = meta_data[5].split(":")[1].strip().title()
    type_plate = meta_data[6].split(":")[1].strip().title()
    response = {
        "char_positions": char_positions,
        "vehicle_position": vehicle_position,
        "corners": corners,
        "camera": camera,
        "vehicle_type": type_v,
        "vehicle_make": type_make,
        "model_type": type_model,
        "model_year": type_year,
        "model_plate": type_plate,
    }
    return response


def read_txt_file(path: str):
    with open(path, "r") as f:
        data = f.read()
    return data


def plot_image_and_annotations(root_dir: str):
    meta_data_file_path, image_file_path = get_random_data_point(root_dir=root_dir)
    meta_data = extract_metadata(read_txt_file(meta_data_file_path))
    image = Image.open(image_file_path)

    draw = ImageDraw.Draw(image)

    vehicle_position = meta_data.get("vehicle_position")
    x, y, w, h = map(int, vehicle_position)
    draw.rectangle([x, y, x + w, y + h], outline="red", width=2)

    plate_corners = meta_data.get("corners")
    draw.polygon(plate_corners, outline="blue")

    char_positions = meta_data.get("char_positions")

    for pos in char_positions:
        cx, cy, cw, ch = map(int, pos)
        draw.rectangle([cx, cy, cx + cw, cy + ch], outline="green", width=2)

    text_info = [
        f"Camera: {meta_data['camera']}",
        f"Type: {meta_data['vehicle_type']}",
        f"Make: {meta_data['vehicle_make']}",
        f"Model: {meta_data['model_type']}",
        f"Year: {meta_data['model_year']}",
        f"Plate: {meta_data['model_plate']}",
    ]

    try:
        font = ImageFont.truetype("arial.ttf", size=25)
    except IOError:
        font = ImageFont.load_default()

    text_y = int(y + h + 10)
    for line in text_info:
        draw.text((x, text_y), line, fill="red", font=font)
        text_y += 20

    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.axis("off")
    plt.savefig("test.png", bbox_inches="tight", pad_inches=0)
    plt.show()


parser = argparse.ArgumentParser()
parser.add_argument(
    "--root_dir",
    type=str,
    default="/tmp/ahsan/sqfs/storage_local/datasets/public/ufpr-alpr",
    help="root dir for data",
)
args = parser.parse_args()

if __name__ == "__main__":
    plot_image_and_annotations(args.root_dir)
