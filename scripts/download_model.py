import requests
import os


def download_model():
    check_point_url = (
        "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
    )
    response = requests.get(check_point_url)
    fname = "sam_vit_b_01ec64.pth"
    cpkt_path = os.path.join(os.getcwd(), "checkpoints", fname)
    if not os.path.exists(cpkt_path):
        print("Checkpoint does not exist...")
        print("Downloding model..")
        os.makedirs(cpkt_path)
        with open(cpkt_path, "wb") as f:
            f.write(response.content)
    else:
        print("Inital Model Exists..")
        print(
            "If this is your second time traning the model, make sure to use the latest .pth file under exp path"
        )
