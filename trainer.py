import os
from scripts.train import main as trainer

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
trainer()
