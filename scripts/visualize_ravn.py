import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import os
import sys

# If LayerDataset is in another file, make sure you import it correctly.
# For example: from my_data_file import LayerDataset
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scripts.train_ravn import LayerGenerator  # or replicate the class here if needed
from src.data import LayerDataset

def visualize(img_tensor, title=""):
    """
    Utility to visualize a tensor image. Handles both 1-channel (grayscale) and 3-channel images.
    """
    # Remove batch dimension if present
    if img_tensor.ndim == 4:  # (B, C, H, W)
        img_tensor = img_tensor[0]
    # Move channels last: from (C, H, W) -> (H, W, C)
    img_tensor = img_tensor.permute(1, 2, 0).detach().cpu().numpy()
    # If there's only one channel, matplotlib's imshow expects a 2D array
    if img_tensor.shape[2] == 1:
        img_tensor = img_tensor.squeeze(-1)
    plt.imshow(img_tensor, cmap="gray" if len(img_tensor.shape) == 2 else None)
    plt.axis("off")
    plt.title(title)

def main():
    # Paths and parameters
    model_path = "layer_generator.pth"  # The file saved by train_simple.py
    h5_file = "data/kenney_dataset_1000.h5"
    dataset_name = "images"
    img_size = 64
    greyscale = True  # or False, depending on how you trained your model
    in_chans = 1 if greyscale else 4

    # Instantiate the model with the same hyperparameters used during training
    model = LayerGenerator(
        img_size=img_size,
        patch_size=8,
        in_chans=in_chans,
        embed_dim=256,
        num_transformer_layers=6,
        num_heads=8
    )
    # Load weights
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    # Create a small dataset & dataloader
    dataset = LayerDataset(
        h5_file=h5_file,
        dataset_name=dataset_name,
        img_size=img_size,
        greyscale=greyscale,
        subset_fraction=0.01,  # just grab a small fraction for visualization
    )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # Run inference on a few samples and visualize
    num_to_show = 3  # how many samples you want to visualize
    count = 0

    with torch.no_grad():
        for input_image, target_image in dataloader:
            # Forward pass
            prediction = model(input_image)

            # Visualize input, prediction, and target side by side
            plt.figure(figsize=(10, 3))
            plt.subplot(1, 3, 1)
            visualize(input_image, title="Input")

            plt.subplot(1, 3, 2)
            visualize(prediction, title="Prediction")

            plt.subplot(1, 3, 3)
            visualize(target_image, title="Target")

            plt.tight_layout()
            plt.show()

            count += 1
            if count >= num_to_show:
                break

if __name__ == "__main__":
    main()