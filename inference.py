from PIL import Image
from model import Model2Class
from utils import set_seed
import torch
from utils import transform
set_seed(22520465)
device = "cuda" if torch.cuda.is_available() else "cpu"

def inference(model_path, image_path, device = torch.device("cuda" if torch.cuda.is_available() else "cpu"), model_name="resnet18"): 
    # load trained model from .pt path 
    model = Model2Class(model_name=model_name)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    model = model.to(device)

    # load image
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)  
    output = model(image)
    probs = torch.sigmoid(output)
    predicted = (probs > 0.5).int()
    if predicted == 0:
        return "Healthy"
    else:
        return "UnHealthy"