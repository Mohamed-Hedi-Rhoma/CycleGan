import rasterio 
import os 
import numpy as np
import shutil
import torch 
from rasterio.warp import reproject, Resampling
from torchvision.transforms.functional import resize

def stats(path_data, sensor="Landsat"):
    band_names = ['blue.tif', 'green.tif', 'red.tif', 'nir.tif', 'swir1.tif', 'swir2.tif']
    
    # Collect all pixel values per band for quantile calculation
    all_pixels = [[] for _ in range(6)]  # One list per band
    
    for subdir in os.listdir(path_data):
        dir_path = os.path.join(path_data, subdir)
        if os.path.isdir(dir_path):
            for date in os.listdir(dir_path):
                date_path = os.path.join(dir_path, date)
                if os.path.isdir(date_path):
                    ref_file = os.path.join(date_path, 'blue.tif')
                    with rasterio.open(ref_file) as ref_src:
                        ref_shape = ref_src.shape
                        ref_transform = ref_src.transform
                        ref_crs = ref_src.crs
                    
                    skip_date = False
                    for i, file_name in enumerate(band_names):
                        file_path = os.path.join(date_path, file_name)
                        if not os.path.exists(file_path):
                            print(f"Warning: Missing {file_path}, skipping date")
                            skip_date = True
                            break
                        
                        with rasterio.open(file_path) as src:
                            if (file_name == 'swir1.tif' or file_name == 'swir2.tif') and sensor == "Sentinel":
                                data = np.empty(ref_shape, dtype=src.dtypes[0])
                                reproject(
                                    source=rasterio.band(src, 1),
                                    destination=data,
                                    src_transform=src.transform,
                                    src_crs=src.crs,
                                    dst_transform=ref_transform,
                                    dst_crs=ref_crs,
                                    resampling=Resampling.bilinear
                                )
                            else:
                                data = src.read(1)
                        if sensor == "Sentinel":
                            data = data.astype(np.float32) / 10000.0  # Sentinel-2: divide by 10000
                        else:  # Landsat
                            data = data.astype(np.float32) * 0.0000275 - 0.2  # Landsat 8/9 Collection 2
                        data_tensor = torch.from_numpy(data).unsqueeze(0)
                        target_size = [384, 384] if sensor == "Sentinel" else [128, 128]
                        data_resized = resize(data_tensor, target_size).squeeze(0)
                        
                        # Store flattened pixels for this band
                        all_pixels[i].append(data_resized.flatten())
                    
                    if not skip_date:
                        print(f"Processed {date_path}")
    
    # Concatenate all pixels per band and compute quantile-based stats
    means = []
    stds = []
    max_samples = int(1e7)
    
    for i in range(6):
        if len(all_pixels[i]) > 0:
            band_data = torch.cat(all_pixels[i]).float()
            
            # Sample if too large
            if band_data.numel() > max_samples:
                indices = torch.randperm(band_data.numel())[:max_samples]
                band_data_sampled = band_data[indices]
            else:
                band_data_sampled = band_data
            
            # Quantile-based normalization (median for mean, IQR for std)
            norm_mean = torch.quantile(band_data_sampled, q=0.5)
            norm_std = torch.quantile(band_data_sampled, q=0.95) - torch.quantile(band_data_sampled, q=0.05)
            
            means.append(norm_mean)
            stds.append(norm_std)
            
            # Free memory
            del band_data, band_data_sampled
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    return torch.tensor(means), torch.tensor(stds)

print(stats(path_data="/home/mrhouma/Documents/CycleGan/CycleGan/landsat_data"))