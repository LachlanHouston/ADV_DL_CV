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
from torch.utils.data import DataLoader, random_split
import torchvision
import matplotlib.pyplot as plt
# import torchvision.transforms as transforms # Keep if needed for LayerPtDataset preprocessing
from tqdm import tqdm
from torchmetrics.image.fid import FrechetInceptionDistance
from ignite.metrics import FID

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data import LastLayerPtDataset # Ensure this returns (input, target, layer_idx)
from src.utils.weighted_proportional_mse_loss import WeightedProportionalMSELoss
from src.models.model_ddpm import DDPM, UNet
from src.utils.visualization import save_image_grid, save_rollout_grid, convert_for_imshow
import pdb

# --- Hyperparameters ---
subset_fraction = 1.0       # Use 1.0 for full dataset, smaller for testing
greyscale = False          # Set to True for grayscale (1 channel), False for color (e.g., 4 channels)
img_size = 128             # Input/Output image size
batch_size = 32           # Adjust based on GPU memory
num_epochs = 400          # Number of training epochs
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
data_file = 'data/full_dataset_128.pt'
# Create a unique directory for each run based on timestamp
run_timestamp = time.strftime("%Y%m%d_%H%M%S")
save_dir = f'results/ddpm_{run_timestamp}/'
experiment_name = f"ddpm_{run_timestamp}"

# Split Proportions
train_split = 0.8
val_split = 0.1
test_split = 0.1
if not np.isclose(train_split + val_split + test_split, 1.0):
    raise ValueError("Split proportions must sum to 1.0")

def save_images(images, path, show=True, title=None, nrow=10):
    # ─── normalise input ────────────────────────────────────────────────────────
    if isinstance(images, np.ndarray):
        images = torch.from_numpy(images)

    # (B, H, W, C) ➜ (B, C, H, W)
    if images.ndim == 4 and images.shape[-1] in {1, 3, 4}:
        images = images.permute(0, 3, 1, 2)

    images = images.detach().cpu()

    # ─── convert every sample to something imshow can digest ───────────────────
    # convert_for_imshow(img) returns (np_img, kwargs); we only need np_img here
    tensor_list = []
    for img in images:                              # img shape: (C, H, W)
        np_img, _ = convert_for_imshow(img)
        np_img    = np_img.astype(np.float32) / 255.0 if np_img.dtype == np.uint8 else np_img
        # (H, W) ➜ (C=1, H, W)   |  (H, W, C) ➜ (C, H, W)
        if np_img.ndim == 2:
            tensor_list.append(torch.from_numpy(np_img)[None, ...])
        else:                                       # 3- or 4-channel
            tensor_list.append(torch.from_numpy(np_img).permute(2, 0, 1))

    rgb_images = torch.stack(tensor_list)           # (B, C, H, W)

    # ─── make the thumbnail grid ───────────────────────────────────────────────
    grid = torchvision.utils.make_grid(
        rgb_images,
        nrow=nrow,
        value_range=(0, 1),                         # assume data now in [0,1]
        pad_value=1.0                               # white padding
    )

    ndarr = grid.permute(1, 2, 0).numpy()           # (H, W, C)

    if title is not None:
        plt.title(title)
    plt.imshow(ndarr, cmap='gray' if ndarr.shape[2] == 1 else None)
    plt.axis('off')

    if path is not None:
        plt.savefig(path, bbox_inches='tight', pad_inches=0)
    if show:
        plt.show()
    else:
        plt.close()

# --- Training Function ---
def train_model(greyscale=False, subset_fraction=1.0):
    """ Trains the DDPM model """
    in_chans = 1 if greyscale else 3 # Input channels (1 for grayscale, 4 for RGBA assumed)
    out_chans = in_chans # Output channels usually match input

    no_samples = 4

    print(f"--- Training Configuration ---")
    print(f"Timestamp: {run_timestamp}")
    print(f"Dataset: {data_file}, Greyscale: {greyscale}, Subset: {subset_fraction*100:.1f}%")
    print(f"Image Size: {img_size}x{img_size}, Batch Size: {batch_size}, Epochs: {num_epochs}")
    print(f"LR: {learning_rate}, Weight Decay: {weight_decay}")
    print(f"Saving results to: {save_dir}")
    print("-" * 30)

    os.makedirs(save_dir, exist_ok=True)

    print(f"Loading dataset from: {data_file}")
    try:
        dataset = LastLayerPtDataset(data_file, img_size=img_size, transform=None,
                                 greyscale=greyscale, subset_fraction=subset_fraction)
        print(f"Successfully loaded dataset. Number of samples: {len(dataset)}")
        print(next(iter(dataset)).shape)  # Check shape of a sample

        if hasattr(dataset, 'num_layers') and dataset.num_layers is not None:
             num_layers_max = dataset.num_layers
             print(f"Using num_layers_max = {num_layers_max} from dataset.")
        else:
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

    total_size = len(dataset)
    train_size = int(train_split * total_size)
    val_size = int(val_split * total_size)
    test_size = total_size - train_size - val_size # Ensure all data is used

    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size], generator=generator
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, drop_last=True)
    # val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, drop_last=False)
    # test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, drop_last=False)

    print(f"Dataloader created. Batches per epoch: {len(train_loader)}")
    if len(train_loader) == 0:
        print("Warning: Dataloader has zero batches. Check batch_size vs dataset size.")

    # Setup device, model, optimizer, losses, scheduler, scaler
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    print(f"Using device: {device}")

    model = UNet(grey_scale=False,
             channels=(32, 64, 128, 256, 512, 1024, 2048)).to(device)

    diffusion = DDPM(
        network=model,
        beta_1=beta_start,
        beta_T=beta_end,
        T=max_timestep,
        p_unconditional=1.0,).to(device)

    # Print model parameter count
    try:
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model Parameters: {total_params / 1e6:.2f} M")
    except Exception as e:
        print(f"Could not calculate model parameters: {e}")

    try:
        criterion_mse = torch.nn.MSELoss()
    except NameError:
        print("Error: WeightedProportionalMSELoss not found. Make sure it's imported correctly.")
        return
    
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    total_steps = max(1, num_epochs * len(train_loader))
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6)
    use_amp = (device.type == 'cuda')
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    print(f"Using Automatic Mixed Precision (AMP): {use_amp}")

    print(f"\n--- Starting Training ---")
    start_train_time = time.time()

    print(next(iter(train_loader)).shape)

    best_loss = float('inf')

    for epoch in range(1, num_epochs + 1):
        pbar = tqdm(train_loader)

        running_loss = 0.0
        for i, images in enumerate(pbar):
            images = images[:, :3, :, :].to(device)
            
            optimizer.zero_grad()
            loss = diffusion(images)  # Forward pass through the diffusion model
            
            loss.backward()
            optimizer.step()
            pbar.set_postfix(MSE=loss.item())

            running_loss += loss.item()
            break
        
        epoch_loss = running_loss / len(train_loader)
        print(f"[Epoch {epoch}] Loss: {epoch_loss:.4f}")

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            print(f"New best loss: {best_loss:.4f}")
        # Save model checkpoint
        if SAVE_MODEL:
            checkpoint_path = os.path.join(save_dir, f"checkpoint_epoch_{epoch:03d}.pth")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Checkpoint saved at {checkpoint_path}")

        if epoch % 5 == 0:
            model.eval()
            with torch.no_grad():
                print(f"Generating {no_samples} samples.")
                samples = diffusion.sample((no_samples, in_chans, img_size, img_size), keep_steps=False)
                samples = samples / 2 + 0.5

            nrow = int(np.sqrt(no_samples))
            if nrow * nrow < no_samples:
                nrow = no_samples // nrow

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

        try:
            # Scale the loss for AMP
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

        except Exception as backward_err:
                print(f"\nError during backward pass or optimizer step at epoch {epoch+1}, batch {i}: {backward_err}")
                optimizer.zero_grad(set_to_none=True)
                continue # Skip this batch update
        
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

def sample_checkpoint(checkpoint_path, num_samples=36, img_size=64, grayscale=False, device='cpu'):
    """ Load a checkpoint and generate samples """
    # Load the model state dict
    model = UNet(grey_scale=False,
             channels=(32, 64, 128, 256, 512, 1024, 2048)).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    diffusion = DDPM(
        network=model,
        beta_1=beta_start,
        beta_T=beta_end,
        T=max_timestep,
        p_unconditional=1.0,
    ).to(device)

    in_chans = 1 if grayscale else 4 # Input channels (1 for grayscale, 4 for RGBA assumed)

    with torch.no_grad():
        samples = diffusion.sample((num_samples, in_chans, img_size, img_size), keep_steps=False)
        samples = samples / 2 + 0.5

    print(f"Generated {num_samples} samples.")
    nrow = int(np.sqrt(num_samples))
    if nrow * nrow < num_samples:
        nrow = num_samples // nrow

    # Save the generated samples
    out_path = 'results/sampled_images.png'
    save_images(samples, path=out_path, show=False, title="Sampled Images from the DDPM", nrow=nrow)
    print(f"Sampled images saved to {out_path}")

if __name__ == '__main__':
    # train_model(greyscale=greyscale, subset_fraction=subset_fraction)
    # print("Training completed successfully.")

    # Gather 100 images from the test set
    # and save them to a file

    # full_dataset = LastLayerPtDataset(data_file, img_size=img_size, transform=None,
    #                              greyscale=greyscale, subset_fraction=subset_fraction)

    # # --- Split the dataset ---
    # total_size = len(full_dataset)
    # train_size = int(train_split * total_size)
    # val_size = int(val_split * total_size)
    # test_size = total_size - train_size - val_size # Ensure all data is used

    # # Use random_split for reproducibility
    # generator = torch.Generator().manual_seed(42)
    # _, _, test_dataset = random_split(
    #     full_dataset, [train_size, val_size, test_size], generator=generator
    # )

    # --- Create DataLoaders ---
    # num_workers = min(4, os.cpu_count() // 2)
    # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
    #                           num_workers=num_workers, pin_memory=True, drop_last=True)
    # val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, # No shuffle for val/test
    #                         num_workers=num_workers, pin_memory=True, drop_last=False)
    # test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, pin_memory=True, drop_last=False)
    
    
    # # test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, drop_last=False)
    # test_images = []

    # pbar = tqdm(test_loader)
    # for i, images in enumerate(pbar):
    #     test_images.append(images)
    # test_images = torch.cat(test_images, dim=0)
    # print(f"Test images shape: {test_images.shape}")

    # # Save the test images to a file
    # test_images_path = 'data/test_images.pt'
    # torch.save(test_images, test_images_path)

    # # Load the model state dict
    # device = 'cpu'
    # model = UNet(grey_scale=False,
    #          channels=(32, 64, 128, 256, 512, 1024, 2048)).to(device)
    # model.load_state_dict(torch.load('results/DDPM_epoch_400.pth', map_location=device))
    # input = torch.rand((1, 4, img_size, img_size))
    # t = torch.rand((1, 1))
    # diffusion = DDPM(
    #     network=model,
    #     beta_1=beta_start,
    #     beta_T=beta_end,
    #     T=max_timestep,
    #     p_unconditional=1.0,
    # ).to(device)

    # in_chans =  4 # Input channels (1 for grayscale, 4 for RGBA assumed)
    # images = []
    # with torch.no_grad():
    #     for i in range(1700):
    #         samples = diffusion.sample((1, in_chans, img_size, img_size), keep_steps=False)
    #         samples = samples / 2 + 0.5
    #         images.append(samples)

    #         print(i)
    # images = torch.cat(images, dim=0)
    # print(f"Test images shape: {images.shape}")

    # # Save the test images to a file
    # test_images_path = 'data/ddpm_images.pt'
    # torch.save(images, test_images_path)

    # from fvcore.nn import FlopCountAnalysis

    # flops = FlopCountAnalysis(model, (input, t))
    # print(flops.total())

    # model = UNet(grey_scale=False,
    #          channels=(32, 64, 128, 256, 512, 1024, 2048)).to('mps')
    # model.load_state_dict(torch.load('results/checkpoint_epoch_190.pth', map_location='mps'))

    # diffusion = DDPM(
    #     network=model,
    #     beta_1=beta_start,
    #     beta_T=beta_end,
    #     T=max_timestep,
    #     p_unconditional=1.0,
    # ).to('mps')

    # in_chans = 3 # Input channels (1 for grayscale, 4 for RGBA assumed)

    # with torch.no_grad():
    #     print('Sampling')
    #     samples = diffusion.sample((100, in_chans, img_size, img_size), keep_steps=False)
    #     samples = samples / 2 + 0.5

    #     print('Sampling done')
    # print(samples.shape)
    # torch.save(samples, 'data/ddpm_new.pt')
    #real = torch.nan_to_num(real_images.float(), nan=0.0, posinf=255.0, neginf=0.0).to(torch.uint8)
    #fake = torch.nan_to_num(ddpm_images.float(), nan=0.0, posinf=255.0, neginf=0.0).to(torch.uint8)
    
    ##FID_score(real_images=real, fake_images=fake)

    # import torch
    # from torch import nn, optim
    # from collections import OrderedDict

    # default_model = nn.Sequential(OrderedDict([
    #     ('base', nn.Linear(128, 64)),
    #     ('fc', nn.Linear(64, 1))
    # ]))
    # fid = FID(num_features=128, device="cpu")                 # keep it on CPU to avoid the MPS SIGTRAP
    # fid.update(ddpm)          # (y_pred, y)
    # print("FID =", fid.compute())
    # ddpm_images = torch.load('data/ddpm_images.pt')
    # images = []
    # for i in range(ddpm_images.shape[0]):
    #     array = ddpm_images[i]
        
    #     indices_swap = [2, 1, 0] + list(range(3, array.shape[0]))
    #     final_image_tensor_swapped = array[indices_swap, :, :]

    #     array_tensor = final_image_tensor_swapped
    #     images.append(array_tensor)

    # tensors = torch.stack(images).reshape(1700, 4, 128, 128)
    # torch.save(tensors, 'data/ddpm_images_converted.pt')

    # Data provided
    # Data
    # DDPM_FLOPS  = 1166540800
    # SMALL_FLOPS = 713838592
    # BIG_FLOPS   = 11751161856
    # VAR_FLOPS   = 12531927040

    # SMALL_FID = 0.6405
    # SMALL_DIFF_FID = 0.9716
    # SMALL_NOPERC_FID = 0.9597

    # BIG_FID = 0.6502
    # BIG_DIFF_FID = 0.7753

    # DDPM_FID = 14.7980
    # VAR_FID = 0.6875

    # # Point list with base jitter in x
    # points = [
    #     ("Small model", SMALL_FLOPS, SMALL_FID, 0.2, -0.05),
    #     ("Small model diff", SMALL_FLOPS, SMALL_DIFF_FID, 0.2, 0.00),      # upward label offset
    #     ("Small model no perception", SMALL_FLOPS, SMALL_NOPERC_FID, 0.2,  -0.2),  # downward label offset
        
    #     ("Big model", BIG_FLOPS, BIG_FID, -1.75, -0.1),
    #     ("Big model diff", BIG_FLOPS, BIG_DIFF_FID, -2, 0.0),
        
    #     #("DDPM", DDPM_FLOPS, DDPM_FID, 0.2, -2.0),
    #     ("VAR", VAR_FLOPS, VAR_FID, -0.3, 0.05)
    # ]

    # # Prepare arrays for scatter
    # x, x_offset, y, labels, y_offsets = [], [], [], [], []
    # for name, flops, fid, jitter_x, jitter_y in points:
    #     x.append(flops / 1e9)
    #     x_offset.append(jitter_x)
    #     y.append(fid)
    #     labels.append(name)
    #     y_offsets.append(jitter_y)

    # plt.figure(figsize=(6*2, 4.5*2))
    # plt.scatter(x, y, s=80)

    # # Annotate with custom y-offsets
    # for label, xi, x_off, yi, yoff in zip(labels, x, x_offset, y, y_offsets):
    #     plt.text(xi + x_off, yi + yoff, f"{label}\nFID={yi:.2f}",
    #             ha="left", va="bottom", fontsize=9*2)

    # plt.xlim(0, 14)
    # # plt.yscale('log')
    # plt.ylabel("FID (log scale)", fontsize=14*2)
    # plt.xlabel("FLOPs (billions)", fontsize=14*2)
    # plt.title("FID vs. FLOPs", fontsize=18*2)
    # plt.grid(True, which="both", linestyle="--", alpha=1, linewidth=2)
    # plt.tight_layout()
    # plt.savefig('FLOPS_FID.png', dpi=300)
    # plt.show()

    # images = torch.load('data/big_diff.pt', map_location='cpu').permute(0, 2, 3, 1)
    # print(images.shape)

    # indices_swap = [2, 1, 0] + list(range(3, images.shape[0]))
    # final_images = images[indices_swap, :, :]
    # print(final_images.shape)

    # plt.imshow(final_images[75])
    # plt.axis('off')
    # plt.savefig("our_kenny_2.png", bbox_inches='tight')
    # plt.show()
    real_images = torch.load('data/test_images.pt', map_location='cpu')[:100, :3, :, :]
    fake_images = torch.load('data/ddpm_new.pt', map_location='cpu')[:100, :3, :, :]
    indices_swap = [0, 2, 1] + list(range(3, fake_images.shape[0]))
    fake_images = fake_images[indices_swap, :, :]


    FID = FrechetInceptionDistance(feature=64, normalize=True)
    FID.update(real_images, real=True)
    FID.update(fake_images, real=False)
    print(FID.compute())
    