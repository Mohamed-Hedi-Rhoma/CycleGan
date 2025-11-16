import rasterio 
import os 
import numpy as np
import shutil



def clean_landsat_data(path_landsat,cloud_threshold = 10) : 
    stats = {
        'total_images': 0,
        'deleted_images': 0,
        'kept_images': 0,
        'cloud_percentages': []
    }
    print("Start")
    for subdir in os.listdir(path_landsat) : 
        #print(subdir)
        dir = os.path.join(path_landsat,subdir)
        #print(dir)
        if  os.path.isdir(dir) :
            for dir_date in os.listdir(dir): 
                date_path = os.path.join(dir,dir_date)
                if os.path.isdir(date_path) : 
                    Qa_pixel_path = os.path.join(date_path,"qa_pixel.tif")
                    #print(Qa_pixel_path)
                    with rasterio.open(Qa_pixel_path) as src:
                        data = src.read(1)
                    dilated_cloud = (data & (1 << 1)) > 0  # Bit 1 (Dilated Cloud)
                    cirrus = (data & (1 << 2)) > 0         # Bit 2 (Cirrus)
                    cloud = (data & (1 << 3)) > 0          # Bit 3 (Cloud)
                    cloud_shadow = (data & (1 << 4)) > 0   # Bit 4 (Cloud Shadow)

                    cloud_mask = dilated_cloud | cirrus | cloud | cloud_shadow
                    cloud_mask = cloud_mask.astype(np.uint8)
                    total_pixels = cloud_mask.size  # or cloud_mask.shape[0] * cloud_mask.shape[1]
                    cloud_pixels = np.sum(cloud_mask)  # Count 1s
                    cloud_percentage = (cloud_pixels / total_pixels) * 100
                    stats['total_images'] += 1
                    stats['cloud_percentages'].append(cloud_percentage)
                    if cloud_percentage > cloud_threshold:
                        print(f"Deleting {date_path}: {cloud_percentage:.2f}% clouds")
                        shutil.rmtree(date_path)  # Delete entire date folder
                        stats['deleted_images'] += 1
                    else:
                        stats['kept_images'] += 1
    return stats

def clean_sentinel_data(path_sentinel,cloud_threshold = 10) : 
    stats = {
        'total_images': 0,
        'deleted_images': 0,
        'kept_images': 0,
        'cloud_percentages': []
    }
    print("Start")
    for subdir in os.listdir(path_sentinel) : 
        #print(subdir)
        dir = os.path.join(path_sentinel,subdir)
        #print(dir)
        if  os.path.isdir(dir) :
            for dir_date in os.listdir(dir): 
                date_path = os.path.join(dir,dir_date)
                if os.path.isdir(date_path) : 
                    Qa_pixel_path = os.path.join(date_path,"qa_pixel.tif")
                    #print(Qa_pixel_path)
                    with rasterio.open(Qa_pixel_path) as src:
                        data = src.read(1)
                    opaque_clouds = (data & (1 << 10)) > 0  # Detects 1024
                    cirrus_clouds = (data & (1 << 11)) > 0  # Detects 2048
                    cloud_mask = opaque_clouds | cirrus_clouds

                    cloud_mask = cloud_mask.astype(np.uint8)
                    total_pixels = cloud_mask.size  # or cloud_mask.shape[0] * cloud_mask.shape[1]
                    cloud_pixels = np.sum(cloud_mask)  # Count 1s
                    cloud_percentage = (cloud_pixels / total_pixels) * 100
                    stats['total_images'] += 1
                    stats['cloud_percentages'].append(cloud_percentage)
                    if cloud_percentage > cloud_threshold:
                        print(f"Deleting {date_path}: {cloud_percentage:.2f}% clouds")
                        shutil.rmtree(date_path)  # Delete entire date folder
                        stats['deleted_images'] += 1
                    else:
                        stats['kept_images'] += 1
    return stats
print(clean_landsat_data(path_landsat="/home/mrhouma/Documents/CycleGan/CycleGan/landsat_data")["kept_images"])