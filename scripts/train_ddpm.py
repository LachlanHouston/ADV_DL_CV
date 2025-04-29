"""
Main training script for the Hybrid Layer Generator model.
Imports the model architecture from 'model.py' and visualization
utilities from 'visualization.py'.

This script handles:
- Hyperparameter definition
- Dataset loading
- Training loop setup (optimizer, scheduler, loss)
- Epoch iteration with training steps
- Periodic visualization and checkpoint saving
- Final model saving
"""

import os
import sys
import time
import numpy as np
import torch
import torch.nn.functional as F # Keep if used directly, e.g., for F.interpolate in loss check
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import matplotlib.pyplot as plt
# import torchvision.transforms as transforms # Keep if needed for LayerPtDataset preprocessing
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data import LastLayerPtDataset # Ensure this returns (input, target, layer_idx)
from src.utils.weighted_proportional_mse_loss import WeightedProportionalMSELoss
from src.models.model_ddpm import Diffusion, UNet
from src.utils.visualization import save_image_grid, save_rollout_grid


# --- Hyperparameters ---
subset_fraction = 1.0       # Use 1.0 for full dataset, smaller for testing
greyscale = True          # Set to True for grayscale (1 channel), False for color (e.g., 4 channels)
img_size = 64             # Input/Output image size
batch_size = 32           # Adjust based on GPU memory
num_epochs = 200          # Number of training epochs
learning_rate = 1e-3      # Initial learning rate
cnn_depth = 4             # Number of down/up sampling stages in CNN U-Net part
transformer_layers = 6    # Number of layers in the Transformer bottleneck
transformer_heads = 8     # Number of attention heads in the Transformer
weight_decay = 0.05       # Weight decay for AdamW optimizer
max_timestep = 500
beta_start = 1e-4
beta_end = 0.02
time_dim = 256
unet_channels = 32

# Loss Function Weights (IMPORTANT: TUNE THESE)
lambda_mse = 1.0          # Weight for WeightedProportionalMSELoss
lambda_perceptual = 0.005   # Weight for VGGPerceptualLoss

VISUALIZE_AND_CHECKPOINT_FREQUENCY = 10     # Save images and model every N epochs
SAVE_MODEL = True                          # Set to True to save checkpoints and final model
visualize_rollouts = True                  # Set to True to generate rollout visualizations


# --- Configuration ---
# Update this path to your actual dataset file
data_file = 'data/full_dataset.pt'
# Create a unique directory for each run based on timestamp
run_timestamp = time.strftime("%Y%m%d_%H%M%S")
save_dir = f'results/ddpm_{run_timestamp}/'
experiment_name = f"ddpm_{run_timestamp}"
# Note: Checkpoints are now saved within the training loop with epoch number

def save_images(images, path, show=True, title=None, nrow=10):
    grid = torchvision.utils.make_grid(images, nrow=nrow)
    ndarr = grid.permute(1, 2, 0).to('cpu').numpy()
    if title is not None:
        plt.title(title)
    plt.imshow(ndarr)
    plt.axis('off')
    if path is not None:
        plt.savefig(path, bbox_inches='tight', pad_inches=0)
    if show:
        plt.show()
    plt.close()

# --- Training Function ---
def train_model(greyscale=True, subset_fraction=1.0):
    """ Trains the LayerGeneratorHybrid model """
    in_chans = 1 if greyscale else 4 # Input channels (1 for grayscale, 4 for RGBA assumed)
    out_chans = in_chans # Output channels usually match input

    print(f"--- Training Configuration ---")
    print(f"Timestamp: {run_timestamp}")
    print(f"Dataset: {data_file}, Greyscale: {greyscale}, Subset: {subset_fraction*100:.1f}%")
    print(f"Image Size: {img_size}x{img_size}, Batch Size: {batch_size}, Epochs: {num_epochs}")
    print(f"LR: {learning_rate}, Weight Decay: {weight_decay}")
    print(f"Saving results to: {save_dir}")
    print("-" * 30)

    # Ensure save directory exists
    os.makedirs(save_dir, exist_ok=True)

    print(f"Loading dataset from: {data_file}")
    try:
        # Pass any necessary transforms here if needed by LayerPtDataset
        dataset = LastLayerPtDataset(data_file, img_size=img_size, transform=None,
                                 greyscale=greyscale, subset_fraction=subset_fraction)
        print(f"Successfully loaded dataset. Number of samples: {len(dataset)}")

        # Determine max layers for embedding (important for the model)
        if hasattr(dataset, 'num_layers') and dataset.num_layers is not None:
             num_layers_max = dataset.num_layers
             print(f"Using num_layers_max = {num_layers_max} from dataset.")
        else:
             # Infer from data or set a default if attribute missing/None
             # Example: try to find max layer_idx in a subset (can be slow)
             # Or set a reasonable default if known
             num_layers_max = 18 # Default value if not found in dataset
             print(f"Warning: Dataset does not provide 'num_layers'. Using default max_layers={num_layers_max} for embedding.")

    except FileNotFoundError:
         print(f"Error: Dataset file not found at {data_file}. Please check the path.")
         return
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # Check if dataset is empty
    if len(dataset) == 0:
        print("Error: Loaded dataset is empty. Cannot train.")
        return

    # dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
    #                         num_workers=min(4, os.cpu_count() // 2), # Adjust num_workers based on system
    #                         pin_memory=True, drop_last=True)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    print(f"Dataloader created. Batches per epoch: {len(dataloader)}")
    if len(dataloader) == 0:
        print("Warning: Dataloader has zero batches. Check batch_size vs dataset size.")
        # Optionally exit if no batches can be formed
        # return

    # Setup device, model, optimizer, losses, scheduler, scaler
    if torch.cuda.is_available():
        device = torch.device('cuda')
    # elif torch.backends.mps.is_available(): # MPS support can be less stable
    #     device = torch.device('mps')
    else:
        device = torch.device('cpu')

    print(f"Using device: {device}")

    model = UNet(
    img_size=img_size,
    c_in=in_chans,      # <-- single‐channel input
    c_out=out_chans,     # <-- single‐channel output
    device=device
    ).to(device)

    diffusion = Diffusion(
        T=max_timestep,
        beta_start=beta_start,
        beta_end=beta_end,
        img_size=img_size,
        img_channels=in_chans,  # <-- match your data
        device=device
    )

    # Print model parameter count
    try:
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model Parameters: {total_params / 1e6:.2f} M")
    except Exception as e:
        print(f"Could not calculate model parameters: {e}")


    # --- Loss Functions ---
    # Weighted MSE Loss (ensure it's imported correctly)
    try:
        criterion_mse = torch.nn.MSELoss()
    except NameError:
        print("Error: WeightedProportionalMSELoss not found. Make sure it's imported correctly.")
        return

    # Perceptual Loss (imported from model.py)
    # criterion_perceptual = VGGPerceptualLoss(feature_layers=[2, 7, 16, 25, 34]).to(device)

    # --- Optimizer and Scheduler ---
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    # Ensure total_steps is at least 1 for scheduler
    total_steps = max(1, num_epochs * len(dataloader))
    # Cosine annealing scheduler: warms down LR towards eta_min
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6)
    # # Automatic Mixed Precision (AMP) scaler for potential speedup on CUDA
    # # Enable AMP only if device is CUDA
    # use_amp = (device.type == 'cuda')
    # scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    # print(f"Using Automatic Mixed Precision (AMP): {use_amp}")


    print(f"\n--- Starting Training ---")
    start_train_time = time.time()

    print(next(iter(dataloader)).shape)

    best_loss = float('inf')

    for epoch in range(1, num_epochs + 1):
        pbar = tqdm(dataloader)

        running_loss = 0.0
        for i, images in enumerate(pbar):
            images = images.to(device)

            # TASK 4: implement the training loop
            t = diffusion.sample_timesteps(images.shape[0]).to(device) # line 3 from the Training algorithm
            x_t, noise = diffusion.q_sample(images, t) # inject noise to the images (forward process), HINT: use q_sample
            predicted_noise = diffusion.p_sample(UNet(), x_t, t) # predict noise of x_t using the UNet
            # loss = criterion_mse(predicted_noise, noise) # calculate the loss
            loss = torch.norm(predicted_noise - noise, p=2) # calculate the loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


            pbar.set_postfix(MSE=loss.item())

            running_loss += loss.item()
        
        epoch_loss = running_loss / len(dataloader)
        print(f"[Epoch {epoch}] Loss: {epoch_loss:.4f}")

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            print(f"New best loss: {best_loss:.4f}")
        # Save model checkpoint
        if SAVE_MODEL:
            checkpoint_path = os.path.join(save_dir, f"checkpoint_epoch_{epoch:03d}.pth")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Checkpoint saved at {checkpoint_path}")

        # ←— HERE: at the end of each epoch, generate & save samples:
        model.eval()
        with torch.no_grad():
            samples = diffusion.p_sample_loop(
                model,
                batch_size=batch_size
            )  # returns a uint8 tensor [B, C, H, W]
        # Compute a square-ish grid size:
        print(f"Generated {samples.shape[0]} samples.")
        nrow = int(np.sqrt(batch_size))
        if nrow * nrow < batch_size:
            nrow = batch_size // nrow

        # save_images() was defined above
        out_path = os.path.join(save_dir, f"samples_epoch_{epoch:03d}.png")
        save_images(
            samples,
            path=out_path,
            show=False,
            title=f"Epoch {epoch} samples",
            nrow=nrow
        )
        print(f"[Epoch {epoch}] Sample images saved to {out_path}")
        model.train()

        # Update the learning rate
        scheduler.step()

    # --- End of Training Loop ---
    end_train_time = time.time()
    total_training_time = end_train_time - start_train_time
    print("-" * 50)
    print(f"Training completed in {total_training_time / 60:.2f} minutes ({total_training_time:.1f} seconds).")

    # Save the final model state dictionary (only the model weights)
    if SAVE_MODEL:
        final_model_path = os.path.join(save_dir, 'hybrid_layer_generator_final.pth')
        torch.save(model.state_dict(), final_model_path)
        print(f"Final model state_dict saved as {final_model_path}")

    print(f"Results (checkpoints, visualizations) saved in: {save_dir}")
    print("-" * 50)



if __name__ == '__main__':
    train_model(greyscale=greyscale, subset_fraction=subset_fraction)
    print("Training completed successfully.")