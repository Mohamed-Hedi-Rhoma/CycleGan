import rasterio
import matplotlib.pyplot as plt
import numpy as np

def plot_rgb(date_path):
    # Read RGB bands
    with rasterio.open(f"{date_path}/red.tif") as src:
        red = src.read(1)
    with rasterio.open(f"{date_path}/green.tif") as src:
        green = src.read(1)
    with rasterio.open(f"{date_path}/blue.tif") as src:
        blue = src.read(1)
    
    # Stack to create RGB image [H, W, 3]
    rgb = np.stack([red, green, blue], axis=-1)
    
    # Normalize to 0-1 range (adjust percentiles if needed)
    rgb = np.clip(rgb, np.percentile(rgb, 2), np.percentile(rgb, 98))
    rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min())
    
    # Plot
    plt.figure(figsize=(10, 10))
    plt.imshow(rgb)
    plt.axis('off')
    plt.title(date_path.split('/')[-1])
    plt.show()

# Usage
plot_rgb("/home/mrhouma/Documents/CycleGan/CycleGan/landsat_data/EU_Forest_1/2020-08-20")