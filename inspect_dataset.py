import h5py
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

# Global configuration variables.
NUM_DISPLAY = 6
DATASET_FILE = 'data/kenney_dataset_10.h5'
DATASET_NAME = 'images'

def main():
    # Open the HDF5 file in read mode.
    with h5py.File(DATASET_FILE, 'r') as hf:
        dataset = hf[DATASET_NAME]
        total_samples, num_layers, height, width, channels = dataset.shape
        num_to_display = min(NUM_DISPLAY, total_samples)
        
        # Create a subplot grid: one row per character, one column per layer.
        fig, axs = plt.subplots(num_to_display, num_layers, figsize=(num_layers*2, num_to_display*2))
        
        # If there is only one character, ensure axs is a 2D array.
        if num_to_display == 1:
            axs = np.expand_dims(axs, axis=0)
        
        # Loop over each character and each layer.
        for i in range(num_to_display):
            for j in range(num_layers):
                img = dataset[i, j, :, :, :]
                # Convert from BGRA to RGBA for proper display in matplotlib.
                img_rgba = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
                ax = axs[i, j]
                ax.imshow(img_rgba)
                ax.axis('off')
                # Optionally, add layer titles at the top row.
                if i == 0:
                    ax.set_title(f"Layer {j+1}", fontsize=8)
        
        plt.suptitle("Dataset Inspection: Each row is a character, columns are progressive layers", fontsize=12)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        # plt.show()
        plt.savefig("kenney_layers.png", dpi=300)

if __name__ == "__main__":
    main()