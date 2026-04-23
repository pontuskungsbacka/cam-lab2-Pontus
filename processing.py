# --- Imports libraries ---
from pathlib import Path
from xml.parsers.expat import model
import torch
from PIL import Image
from torchvision import transforms
from torchvision.io import decode_image
from torchvision.models import get_model, get_model_weights
from torchcam.methods import LayerCAM
from torchcam.utils import overlay_mask
from torchvision.transforms.functional import to_pil_image
from torch.nn.functional import softmax
import numpy as np
import json
import matplotlib.pyplot as plt

class ImageProcessor:
    def __init__(self):
        # --- Initializes the model, weights, and preprocess function to None ---
        self.model = None
        self.weights = None
        self.preprocess = None
        self.device = None
        self.categories = None

    # --- Initializes the model with the specified name and the preprocess function ---
    def load_imagenet_json(self, json_path):
        with open(json_path, "r") as f:
            raw = json.load(f)
        categories = [None] * len(raw)

        for idx_str, pair in raw.items():
            idx = int(idx_str)
            wnid, class_name = pair
            categories[idx] = class_name

        self.categories = categories
        return categories


    def initialize_model(self, model_name):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = get_model(model_name, weights=get_model_weights(model_name).DEFAULT).to(device).eval()
        weights = get_model_weights(model_name).DEFAULT
        preprocess = weights.transforms()

        self.model = model
        self.preprocess = preprocess
        self.device = device

        return model, preprocess, device
    
    # --- Processes the image and returns the CAM result ---
    def process_image(self, image_path):
        # --- Loads and preprocesses the image ---
        image = Image.open(image_path).convert("RGB")
        batch = self.preprocess(image).unsqueeze(0).to(self.device)
        return image, batch
    
    def find_class_index(self, class_name):
        class_name_lower = class_name.lower()
        categories = self.categories

        exact_matches = [idx for idx, name in enumerate(categories) if name.lower() == class_name_lower]

        if exact_matches:
            return exact_matches[0]
        
        partial_matches = [idx for idx, name in enumerate(categories) if class_name_lower in name.lower()]

        if len(partial_matches) == 1:
            return partial_matches[0]
        elif len(partial_matches) > 1:
            print(f"Multiple matches found for '{class_name}':")
            for idx in partial_matches:
                print(f"{idx}: {categories[idx]}")
            selected_idx = int(input("Please enter the index of the correct class: "))
            if selected_idx in partial_matches:
                return selected_idx
            else:
                raise ValueError("Invalid index selected.")
        raise ValueError(f"No matches found for class name '{class_name}'.")
    
    def top_k_from_scores(self, scores, k=5):
        categories = self.categories
        probs = softmax(scores.squeeze(), dim=0).detach().cpu()
        values, indices = torch.topk(probs, k)
        rows = []
        for value, idx in zip(values, indices):
            rows.append(
                {
                    "class_index": idx.item(),
                    "class_name": categories[idx.item()],
                    "probability": value.item(),
                }
            )
        return rows
    
    def top_k_logits(self, scores, k=10):
        categories = self.categories
        logits = scores.squeeze().detach().cpu()
        values, indices = torch.topk(logits, k)
        rows = []
        for value, idx in zip(values, indices):
            rows.append(
                {
                    "class_index": idx.item(),
                    "class_name": categories[idx.item()],
                    "logit": value.item(),
                }
            )
        return rows
    
    def get_class_ranking(self, scores, class_index):
        probs = softmax(scores.squeeze(), dim=0).detach().cpu()
        ranking = torch.argsort(probs, descending=True)
        rank = int((ranking == class_index).nonzero().item()) + 1
        prob = float(probs[class_index].item())
        return rank, prob

    def analyse_image_CAM_with_layers(self,
            image_path,
            target_class_index,
            alpha=0.5):
        target_idx = self.find_class_index(target_class_index)
        image, batch = self.process_image(image_path)

        with LayerCAM(self.model) as cam_extractor:
            scores = self.model(batch)
            activation_map = cam_extractor(target_idx, scores)[0].detach().cpu()
        
        predicted_rows = self.top_k_from_scores(scores, k=5)
        target_rank, target_prob = self.get_class_ranking(scores, target_idx)
        predicted_class_name = predicted_rows[0]["class_name"]

        overlay = overlay_mask(image, to_pil_image(activation_map, mode="F"), alpha=alpha)
        return {
            "image_path": image_path,
            "target_class_name": self.categories[target_idx],
            "predicted_class_name": predicted_class_name,
            "target_rank": target_rank,
            "target_probability": target_prob,
            "top5_predictions": predicted_rows,
            "top10_logits": self.top_k_logits(scores, k=10),
            "image": image,
            "activation_map": activation_map,
            "overlay": overlay,
        }
    
    def display_cam_results(self, results, layers_name=["layer1", "layer3", "layer4"], alpha=0.4):
        fig, axes = plt.subplots(1, len(layers_name) + 1, figsize=(5 * (len(layers_name) + 1), 5))
        axes[0].imshow(results["image"])
        axes[0].set_title(f"Original Image\n(Target: {results['target_class_name']})")
        axes[0].axis("off")

        for ax, layer_name in zip(axes[1:], layers_name):
            with LayerCAM(self.model, target_layer=layer_name) as cam_extractor:
                scores = self.model(self.preprocess(results["image"]).unsqueeze(0).to(self.device))
                activation_map = cam_extractor(self.find_class_index(results["target_class_name"]), scores)[0].detach().cpu()
            
            activation_map_2d = activation_map.squeeze()

            overlay = overlay_mask(results["image"],
                                    to_pil_image(activation_map_2d, mode="F"), alpha=alpha)
            
            ax.imshow(overlay)
            ax.set_title(f"{layer_name} Activation Map")
            ax.axis("off")

        plt.tight_layout()
        plt.show()

        print(f"Filename: {results['image_path'].name}")
        print(f"Target Class: {results['target_class_name']}")
        print(f"Predicted Class: {results['predicted_class_name']}")
        print(f"Target Class Rank: {results['target_rank']}")
        print(f"Target Class Probability: {results['target_probability']:.4f}")
        print("\nTop 5 Predictions:")
        for idx, row in enumerate(results["top5_predictions"], 1):
            print(f"{idx}. {row['class_name']} (Index: {row['class_index']}, Probability: {row['probability']:.4f})")
