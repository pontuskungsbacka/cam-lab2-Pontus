# --- Imports libraries ---
from pathlib import Path
import torch
from PIL import Image
from torchvision import transforms
from torchvision.io import decode_image
from torchvision.models import get_model, get_model_weights
from torchcam.methods import LayerCAM
from torchcam.utils import overlay_mask
from torchvision.transforms.functional import to_pil_image
import matplotlib.pyplot as plt

# --- Initializes the model with the specified name and the preprocess function ---
def initialize_model(model_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(model_name, weights=get_model_weights(model_name).DEFAULT).to(device).eval()
    weights = get_model_weights(model_name).DEFAULT
    preprocess = weights.transforms()
    return model, preprocess