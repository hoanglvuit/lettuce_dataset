from model import Model2Class
from utils import set_seed
import torch
import argparse
set_seed(22520465)

if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="resnet18")
    args = parser.parse_args()
    model_name = args.model_name

    # load trained model from .pt path 
    model = Model2Class(model_name=model_name)
    model.load_state_dict(torch.load(f"models/{model_name}.pth"))
    

