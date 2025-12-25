import os
import json
import random
import requests
import rasterio
import numpy as np
from pathlib import Path

def create_test_geotiff_from_landsat_site(site_path, output_path):
    """
    Create a single 6-band GeoTIFF from a specific Landsat site (random date)
    
    Args:
        site_path: Direct path to specific site (e.g., "/path/to/landsat_data/AF_Forest_1")
        output_path: Where to save the output GeoTIFF
    
    Returns:
        angles: List of 4 angle values [sza_cos, vza_cos, saa_cos, vaa_cos]
        site_name: The site name
        date: The date chosen
    """
    
    # Get site name from path
    site_name = os.path.basename(site_path)
    
    print(f"Using site: {site_name}")
    
    # Load metadata from site level
    metadata_path = os.path.join(site_path, "site_metadata.json")
    
    if not os.path.exists(metadata_path):
        raise ValueError(f"No metadata found at {metadata_path}")
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Get all date directories (exclude metadata file)
    date_dirs = [d for d in os.listdir(site_path) 
                 if os.path.isdir(os.path.join(site_path, d))]
    
    if not date_dirs:
        raise ValueError(f"No date directories found in {site_path}")
    
    # Choose random date
    chosen_date = random.choice(date_dirs)
    date_path = os.path.join(site_path, chosen_date)
    
    print(f"Selected date: {chosen_date}")
    
    # Get angles for this date from metadata
    angle_key = None
    for key in metadata["angles"]["SZA"].keys():
        if key.startswith(chosen_date):
            angle_key = key
            break
    
    if angle_key is None:
        raise ValueError(f"No angle data found for date {chosen_date}")
    
    sza = metadata["angles"]["SZA"][angle_key]
    vza = metadata["angles"]["VZA"][angle_key]
    saa = metadata["angles"]["SAA"][angle_key]
    vaa = metadata["angles"]["VAA"][angle_key]
    
    # Convert to cosine(radians)
    sza_cos = np.cos(np.radians(sza))
    vza_cos = np.cos(np.radians(vza))
    saa_cos = np.cos(np.radians(saa))
    vaa_cos = np.cos(np.radians(vaa))
    
    angles = [sza_cos, vza_cos, saa_cos, vaa_cos]
    
    print(f"Angles (cos radians): {angles}")
    
    # Read all 6 bands
    band_names = ['blue.tif', 'green.tif', 'red.tif', 'nir.tif', 'swir1.tif', 'swir2.tif']
    
    # Check if all bands exist
    for band_name in band_names:
        band_path = os.path.join(date_path, band_name)
        if not os.path.exists(band_path):
            raise ValueError(f"Band file not found: {band_path}")
    
    # Read first band to get metadata
    first_band_path = os.path.join(date_path, band_names[0])
    
    with rasterio.open(first_band_path) as src:
        meta = src.meta.copy()
        height = src.height
        width = src.width
        transform = src.transform
        crs = src.crs
    
    # Update metadata for 6 bands
    meta.update({
        'count': 6,
        'dtype': 'float32'
    })
    
    # Create output GeoTIFF with all 6 bands
    with rasterio.open(output_path, 'w', **meta) as dst:
        for i, band_name in enumerate(band_names, start=1):
            band_path = os.path.join(date_path, band_name)
            
            with rasterio.open(band_path) as src:
                data = src.read(1).astype(np.float32)
                dst.write(data, i)
    
    print(f"Created test GeoTIFF: {output_path}")
    print(f"Shape: {height} x {width}, 6 bands")
    
    return angles, site_name, chosen_date


def test_api(input_geotiff, angles, output_dir, generator='L2S', output_format='image'):
    """
    Test the API with a GeoTIFF file
    
    Args:
        input_geotiff: Path to input GeoTIFF
        angles: List of 4 angle values
        output_dir: Directory to save outputs
        generator: 'L2S' or 'S2L'
        output_format: 'geotiff' or 'image'
    """
    
    # API endpoint
    url = "http://localhost:8000/predict"
    
    # Prepare angles as comma-separated string
    angles_str = ','.join([str(a) for a in angles])
    
    # Prepare files and data
    files = {
        'file': open(input_geotiff, 'rb')
    }
    
    data = {
        'generator': generator,
        'angles': angles_str,
        'output_format': output_format
    }
    
    print(f"\nSending request to API...")
    print(f"Generator: {generator}")
    print(f"Angles: {angles_str}")
    print(f"Output format: {output_format}")
    
    # Send request
    response = requests.post(url, files=files, data=data)
    
    files['file'].close()
    
    # Check response
    if response.status_code == 200:
        # Save output
        ext = '.tif' if output_format == 'geotiff' else '.png'
        output_path = os.path.join(output_dir, f'output_{generator}{ext}')
        
        with open(output_path, 'wb') as f:
            f.write(response.content)
        
        print(f"✅ Success! Output saved to: {output_path}")
        return output_path
    else:
        print(f"❌ Error: {response.status_code}")
        print(f"Response: {response.text}")
        return None


def create_comparison_image(input_geotiff, output_png, output_dir):
    """
    Create side-by-side comparison of input Landsat and generated Sentinel
    
    Args:
        input_geotiff: Path to input Landsat GeoTIFF (6 bands)
        output_png: Path to generated Sentinel PNG
        output_dir: Directory to save comparison
    """
    from PIL import Image
    import matplotlib.pyplot as plt
    
    # Read input Landsat (6 bands)
    with rasterio.open(input_geotiff) as src:
        # Read RGB bands (indices 2, 1, 0 for Red, Green, Blue)
        red = src.read(3).astype(np.float32)    # red.tif is band 3
        green = src.read(2).astype(np.float32)  # green.tif is band 2
        blue = src.read(1).astype(np.float32)   # blue.tif is band 1
    
    # Stack to RGB
    landsat_rgb = np.stack([red, green, blue], axis=-1)  # (H, W, 3)
    
    # Apply Landsat scaling
    landsat_rgb = landsat_rgb * 0.0000275 - 0.2
    
    # Percentile stretch for visualization
    p2 = np.percentile(landsat_rgb, 2)
    p98 = np.percentile(landsat_rgb, 98)
    landsat_rgb = np.clip((landsat_rgb - p2) / (p98 - p2), 0, 1)
    landsat_rgb = (landsat_rgb * 255).astype(np.uint8)
    
    # Read generated Sentinel PNG
    sentinel_img = Image.open(output_png)
    sentinel_rgb = np.array(sentinel_img)
    
    # Resize Landsat to match Sentinel size for side-by-side (optional)
    landsat_img = Image.fromarray(landsat_rgb)
    landsat_resized = landsat_img.resize(sentinel_img.size, Image.BILINEAR)
    landsat_rgb_resized = np.array(landsat_resized)
    
    # Create side-by-side comparison
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    axes[0].imshow(landsat_rgb_resized)
    axes[0].set_title('Input: Landsat RGB', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    axes[1].imshow(sentinel_rgb)
    axes[1].set_title('Generated: Sentinel RGB', fontsize=14, fontweight='bold')
    axes[1].axis('off')
    
    plt.tight_layout()
    
    # Save comparison
    comparison_path = os.path.join(output_dir, 'comparison_L2S.png')
    plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Comparison image saved: {comparison_path}")
    
    return comparison_path


def main():
    """Main test function"""
    
    # ============================================
    # CONFIGURATION - SPECIFY YOUR SITE PATH HERE
    # ============================================
    SITE_PATH = "/home/mrhouma/Documents/CycleGan/CycleGan/landsat_data/landsat_data/EU_Grassland_1"
    OUTPUT_DIR = "/home/mrhouma/Documents/CycleGan/CycleGan/api_test_results"
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("="*60)
    print("CREATING TEST GEOTIFF FROM LANDSAT DATA")
    print("="*60)
    
    # Create test GeoTIFF
    test_geotiff_path = os.path.join(OUTPUT_DIR, "test_input.tif")
    
    try:
        angles, site_name, date = create_test_geotiff_from_landsat_site(
            SITE_PATH, 
            test_geotiff_path
        )
    except Exception as e:
        print(f"Error creating test GeoTIFF: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n" + "="*60)
    print("TESTING API - LANDSAT TO SENTINEL")
    print("="*60)
    
    # Test 1: Get image output (PNG)
    print("\n--- Test 1: Image output (PNG) ---")
    output_image = test_api(
        test_geotiff_path, 
        angles, 
        OUTPUT_DIR,
        generator='L2S',
        output_format='image'
    )
    
    # Create comparison image
    if output_image:
        print("\n--- Creating comparison image ---")
        create_comparison_image(test_geotiff_path, output_image, OUTPUT_DIR)
    
    # Test 2: Get GeoTIFF output
    print("\n--- Test 2: GeoTIFF output ---")
    output_geotiff = test_api(
        test_geotiff_path,
        angles,
        OUTPUT_DIR,
        generator='L2S',
        output_format='geotiff'
    )
    
    print("\n" + "="*60)
    print("TEST COMPLETE")
    print("="*60)
    print(f"\nTest info:")
    print(f"  Site: {site_name}")
    print(f"  Date: {date}")
    print(f"\nResults saved in: {OUTPUT_DIR}")
    print(f"  - Input: test_input.tif")
    if output_image:
        print(f"  - Generated Sentinel: output_L2S.png")
        print(f"  - Comparison: comparison_L2S.png")
    if output_geotiff:
        print(f"  - Output GeoTIFF: output_L2S.tif")


if __name__ == "__main__":
    # First, check if API is running
    try:
        response = requests.get("http://localhost:8000/health")
        if response.status_code == 200:
            print("✅ API is running!")
            print(f"Health status: {response.json()}")
            print("\n")
            main()
        else:
            print("❌ API is not responding correctly")
    except Exception as e:
        print("❌ Cannot connect to API. Make sure it's running!")
        print("Run in another terminal: cd src/cyclegan/api && pixi run uvicorn main:app --reload")
        print(f"Error: {e}")