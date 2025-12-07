import rasterio 
import os 
import numpy as np
import shutil


def clean_landsat_by_mask(path_landsat, mask_threshold=25):
    """
    Delete Landsat samples if more than mask_threshold% of pixels are masked.
    
    Args:
        path_landsat: Path to Landsat data directory
        mask_threshold: Maximum acceptable masked percentage (default 25%)
    
    Returns:
        dict: Statistics about processed images
    """
    stats = {
        'total_images': 0,
        'deleted_images': 0,
        'kept_images': 0,
        'masked_percentages': []
    }
    
    print("Starting Landsat mask-based cleaning...")
    
    for subdir in os.listdir(path_landsat):
        landsat_site_dir = os.path.join(path_landsat, subdir)
        
        if not os.path.isdir(landsat_site_dir):
            continue
        
        print(f"\nProcessing site: {subdir}")
        
        for dir_date in os.listdir(landsat_site_dir):
            landsat_date_path = os.path.join(landsat_site_dir, dir_date)
            
            if not os.path.isdir(landsat_date_path):
                continue
            
            # Check a reference band (e.g., red band) for Landsat
            ref_band_path = os.path.join(landsat_date_path, "red.tif")
            
            if not os.path.exists(ref_band_path):
                print(f"  Warning: No reference band found for {landsat_date_path}")
                continue
            
            try:
                # Read with masking enabled
                with rasterio.open(ref_band_path) as src:
                    # Read as masked array (respects nodata values)
                    data = src.read(1, masked=True)
                
                # Calculate masked percentage
                total_pixels = data.size
                masked_pixels = np.sum(data.mask)  # True values in mask
                masked_percentage = (masked_pixels / total_pixels) * 100
                
                stats['total_images'] += 1
                stats['masked_percentages'].append(masked_percentage)
                
                if masked_percentage > mask_threshold:
                    print(f"  Deleting {dir_date}: {masked_percentage:.2f}% masked")
                    shutil.rmtree(landsat_date_path)
                    stats['deleted_images'] += 1
                else:
                    print(f"  Keeping {dir_date}: {masked_percentage:.2f}% masked")
                    stats['kept_images'] += 1
                    
            except Exception as e:
                print(f"  Error processing {landsat_date_path}: {e}")
    
    return stats


def clean_sentinel_by_mask(path_sentinel, mask_threshold=25):
    """
    Delete Sentinel samples if more than mask_threshold% of pixels are masked.
    
    Args:
        path_sentinel: Path to Sentinel data directory
        mask_threshold: Maximum acceptable masked percentage (default 25%)
    
    Returns:
        dict: Statistics about processed images
    """
    stats = {
        'total_images': 0,
        'deleted_images': 0,
        'kept_images': 0,
        'masked_percentages': []
    }
    
    print("Starting Sentinel mask-based cleaning...")
    
    for subdir in os.listdir(path_sentinel):
        sentinel_site_dir = os.path.join(path_sentinel, subdir)
        
        if not os.path.isdir(sentinel_site_dir):
            continue
        
        print(f"\nProcessing site: {subdir}")
        
        for dir_date in os.listdir(sentinel_site_dir):
            sentinel_date_path = os.path.join(sentinel_site_dir, dir_date)
            
            if not os.path.isdir(sentinel_date_path):
                continue
            
            # Check a reference band (e.g., red band) for Sentinel
            ref_band_path = os.path.join(sentinel_date_path, "red.tif")
            
            if not os.path.exists(ref_band_path):
                print(f"  Warning: No reference band found for {sentinel_date_path}")
                continue
            
            try:
                # Read with masking enabled
                with rasterio.open(ref_band_path) as src:
                    # Read as masked array (respects nodata values)
                    data = src.read(1, masked=True)
                
                # Calculate masked percentage
                total_pixels = data.size
                masked_pixels = np.sum(data.mask)  # True values in mask
                masked_percentage = (masked_pixels / total_pixels) * 100
                
                stats['total_images'] += 1
                stats['masked_percentages'].append(masked_percentage)
                
                if masked_percentage > mask_threshold:
                    print(f"  Deleting {dir_date}: {masked_percentage:.2f}% masked")
                    shutil.rmtree(sentinel_date_path)
                    stats['deleted_images'] += 1
                else:
                    print(f"  Keeping {dir_date}: {masked_percentage:.2f}% masked")
                    stats['kept_images'] += 1
                    
            except Exception as e:
                print(f"  Error processing {sentinel_date_path}: {e}")
    
    return stats


# Usage for Landsat
print("="*60)
print("LANDSAT CLEANING")
print("="*60)
stats_landsat = clean_landsat_by_mask(
    path_landsat="/home/mrhouma/Documents/CycleGan/CycleGan/landsat_data",
    mask_threshold=12  # Delete if more than 25% is masked
)

# Usage for Sentinel
print("\n" + "="*60)
print("SENTINEL CLEANING")
print("="*60)
stats_sentinel = clean_sentinel_by_mask(
    path_sentinel="/home/mrhouma/Documents/CycleGan/CycleGan/data_sentinel2",
    mask_threshold=12  # Delete if more than 25% is masked
)

print(f"\nLandsat cleaning completed:")
print(f"  Total images processed: {stats_landsat['total_images']}")
print(f"  Images deleted: {stats_landsat['deleted_images']}")
print(f"  Images kept: {stats_landsat['kept_images']}")
if len(stats_landsat['masked_percentages']) > 0:
    print(f"  Average masked percentage: {np.mean(stats_landsat['masked_percentages']):.2f}%")

print(f"\nSentinel cleaning completed:")
print(f"  Total images processed: {stats_sentinel['total_images']}")
print(f"  Images deleted: {stats_sentinel['deleted_images']}")
print(f"  Images kept: {stats_sentinel['kept_images']}")
if len(stats_sentinel['masked_percentages']) > 0:
    print(f"  Average masked percentage: {np.mean(stats_sentinel['masked_percentages']):.2f}%")