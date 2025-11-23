import ee
import os
import pandas as pd
import numpy as np
import requests
import json
import rasterio
import shutil
import multiprocessing
import psutil
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

# Initialize Earth Engine (make sure to authenticate first)
ee.Initialize(project='ee-get-landsat-data')

# Global constants
OUTPUT_DIR = "landsat_data"  # Base directory for output
MAX_CLOUD_COVER = 20  # Maximum allowed cloud cover percentage
START_DATE = "2015-01-01"  # Start date for data collection
END_DATE = "2022-12-31"  # End date for data collection

# Define 50 sites with different landcover types and geographic locations
# Format: [site_name, latitude, longitude, landcover_type]
SITES = [
    # North America
    ["NA_Forest_1", 45.2041, -68.7103, "forest"],       # Maine forest
    ["NA_Cropland_1", 40.7128, -95.1234, "cropland"],   # Iowa farmland
    ["NA_Urban_1", 40.7128, -74.0060, "urban"],         # New York City
    ["NA_Grassland_1", 39.0997, -105.7631, "grassland"], # Colorado grassland
    ["NA_Wetland_1", 30.2672, -90.1994, "wetland"],     # Louisiana wetland
    
    # South America
    ["SA_Forest_1", -3.4653, -62.2159, "forest"],       # Amazon rainforest
    ["SA_Grassland_1", -31.4201, -64.1888, "grassland"], # Argentinian pampas
    ["SA_Cropland_1", -12.9822, -56.0973, "cropland"],  # Brazilian agriculture
    ["SA_Wetland_1", -15.5989, -55.9586, "wetland"],    # Pantanal wetlands
    ["SA_Urban_1", -23.5505, -46.6333, "urban"],        # São Paulo
    
    # Europe
    ["EU_Forest_1", 46.8182, 8.2275, "forest"],         # Swiss Alps forest
    ["EU_Cropland_1", 48.8566, 2.3522, "cropland"],     # French agriculture
    ["EU_Urban_1", 51.5074, -0.1278, "urban"],          # London
    ["EU_Grassland_1", 42.6977, 25.2872, "grassland"],  # Bulgarian grassland
    ["EU_Wetland_1", 52.9399, 5.7933, "wetland"],       # Netherlands wetland
    
    # Africa
    ["AF_Forest_1", -0.1568, 37.9083, "forest"],        # Kenya forest
    ["AF_Savanna_1", -19.9164, 23.5921, "savanna"],     # Botswana savanna
    ["AF_Desert_1", 23.4162, 25.6628, "desert"],        # Sahara desert
    ["AF_Cropland_1", 5.6037, -0.1870, "cropland"],     # Ghana agriculture
    ["AF_Urban_1", -33.9249, 18.4241, "urban"],         # Cape Town
    
    # Asia
    ["AS_Forest_1", 35.6762, 139.6503, "forest"],       # Japanese forest
    ["AS_Rice_1", 23.1291, 113.2644, "cropland"],       # Chinese rice fields
    ["AS_Urban_1", 22.3193, 114.1694, "urban"],         # Hong Kong
    ["AS_Desert_1", 24.1302, 55.8013, "desert"],        # UAE desert
    ["AS_Wetland_1", 9.1763, 99.3012, "wetland"],       # Thailand mangroves
    
    # Oceania
    ["OC_Forest_1", -42.8821, 147.3272, "forest"],      # Tasmanian forest
    ["OC_Grassland_1", -43.5321, 172.6362, "grassland"], # New Zealand grassland
    ["OC_Urban_1", -33.8688, 151.2093, "urban"],        # Sydney
    ["OC_Desert_1", -25.3444, 131.0369, "desert"],      # Australian outback
    ["OC_Cropland_1", -34.9285, 138.6007, "cropland"],  # Australian agriculture
    
    # Additional varied sites
    ["Tundra_1", 68.3558, -133.7381, "tundra"],         # Arctic tundra
    ["Alpine_1", 46.5197, 11.1027, "alpine"],           # Italian Alps
    ["Coastal_1", 25.0343, -77.3963, "coastal"],        # Bahamas coast
    ["Island_1", 20.7967, -156.3319, "island"],         # Hawaii
    ["Mediterranean_1", 37.9838, 23.7275, "mediterranean"], # Greek landscape
    
    ["Mountain_1", 39.1911, -106.8175, "mountain"],     # Colorado mountains
    ["Canyon_1", 36.0544, -112.1401, "canyon"],         # Grand Canyon
    ["Marsh_1", 29.9499, -89.9504, "marsh"],            # Mississippi marsh
    ["River_1", 1.8312, -67.9272, "river"],             # Amazon River
    ["Lake_1", 46.8131, -71.2075, "lake"],              # Canadian lake region
    
    ["Volcano_1", 19.4094, -155.2834, "volcanic"],      # Hawaiian volcano
    ["Glacier_1", 78.2285, 15.6532, "glacier"],         # Svalbard glacier
    ["Salt_Flat_1", -20.1338, -67.4891, "salt_flat"],   # Bolivian salt flats
    ["Mangrove_1", 25.1304, -80.9000, "mangrove"],      # Florida mangroves
    ["Prairie_1", 41.8781, -87.6298, "prairie"],        # Midwestern prairie
    
    ["Delta_1", 30.0502, 31.2351, "delta"],             # Nile delta
    ["Peninsula_1", 41.9028, 12.4964, "peninsula"],     # Italian peninsula
    ["Plateau_1", 35.8617, 104.1954, "plateau"],        # Tibetan plateau
    ["Fjord_1", 60.4720, 5.4800, "fjord"],              # Norwegian fjord
    ["Coral_Reef_1", -18.1428, 147.7106, "coral_reef"]  # Great Barrier Reef
]

def get_optimal_workers():
    """
    Determines the optimal number of worker processes based on system resources.
    
    Returns:
        tuple: (process_workers, thread_workers)
    """
    # Get CPU info
    total_cores = multiprocessing.cpu_count()
    physical_cores = psutil.cpu_count(logical=False)
    
    # Get memory info
    memory = psutil.virtual_memory()
    available_memory_gb = memory.available / (1024 ** 3)
    
    # Calculate optimal number of process workers
    recommended_process_workers = max(1, int(physical_cores * 0.60))
    memory_based_workers = max(1, int(available_memory_gb / 2))
    process_workers = min(recommended_process_workers, memory_based_workers)
    
    # Calculate thread workers per process
    thread_workers = 2
    
    print(f"System Information:")
    print(f"- Total CPU cores (including logical): {total_cores}")
    print(f"- Physical CPU cores: {physical_cores}")
    print(f"- Available memory: {available_memory_gb:.1f} GB")
    print(f"Recommended configuration:")
    print(f"- Process workers: {process_workers}")
    print(f"- Thread workers per process: {thread_workers}")
    
    return process_workers, thread_workers

def create_bounding_box(lat, lon, pixel_count=128, resolution=30):
    """
    Creates a bounding box around the given latitude and longitude.
    
    Args:
        lat (float): Latitude of the center point
        lon (float): Longitude of the center point
        pixel_count (int): Number of pixels for the bounding box (default: 512)
        resolution (int): Resolution in meters per pixel (default: 30)
        
    Returns:
        ee.Geometry: Earth Engine geometry object representing the bounding box
    """
    half_side = (pixel_count * resolution) / 2
    point = ee.Geometry.Point([lon, lat])
    region = point.buffer(half_side).bounds()
    return region

def calculate_angle_means(angle_files, angle_name):
    """
    Calculate the mean value for an angle from the downloaded TIF files.
    
    Args:
        angle_files (list): List of paths to angle TIF files
        angle_name (str): Name of the angle (SZA, VZA, SAA, VAA)
        
    Returns:
        dict: Dictionary with dates as keys and mean angle values as values
    """
    angle_means = {}
    
    for file_path in angle_files:
        # Extract date from filename (assumes format YYYY-MM-DD_BAND.tif)
        date = os.path.basename(file_path).split('_')[0]
        
        try:
            with rasterio.open(file_path) as src:
                angle_data = src.read(1)
                
                # Get the no-data value from metadata
                nodata = src.nodata
                
                # Filter out no-data values
                if nodata is not None:
                    valid_data = angle_data[angle_data != nodata]
                else:
                    # Landsat uses -32768 or similar for no-data
                    # Valid angles are 0-36000 (for azimuth) or 0-9000 (for zenith)
                    valid_data = angle_data[angle_data >= 0]
                
                if len(valid_data) > 0:
                    # Divide by 100 to get actual angle in degrees
                    mean_value = float(np.mean(valid_data)) / 100.0
                    angle_means[date] = mean_value
                else:
                    print(f"No valid data found in {file_path}")
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    return angle_means

def download_and_process_data(site_name, lat, lon, landcover_type):
    """
    Downloads Landsat 8 reflectance data and angle data for the given site.
    For angles, calculates means and saves to JSON.
    
    Args:
        site_name (str): Name of the site
        lat (float): Latitude of the site
        lon (float): Longitude of the site
        landcover_type (str): Type of landcover at the site
    """
    print(f"Processing site: {site_name} ({landcover_type})")
    
    # Create site directory
    site_dir = os.path.join(OUTPUT_DIR, site_name)
    os.makedirs(site_dir, exist_ok=True)
    
    # Create a temporary directory for angle files
    temp_angles_dir = os.path.join(site_dir, "temp_angles")
    os.makedirs(temp_angles_dir, exist_ok=True)
    
    # Create a JSON metadata file for the site
    site_metadata = {
        "name": site_name,
        "latitude": lat,
        "longitude": lon,
        "landcover_type": landcover_type,
        "data_start": START_DATE,
        "data_end": END_DATE,
        "angles": {}  # Will store angle means by date
    }
    
    # Define the region
    region = create_bounding_box(lat, lon)
    
    # Define the bands to download for reflectance
    reflectance_bands = {
        'SR_B1': 'coastal_aerosol.tif',
        'SR_B2': 'blue.tif',
        'SR_B3': 'green.tif',
        'SR_B4': 'red.tif',
        'SR_B5': 'nir.tif',
        'SR_B6': 'swir1.tif',
        'SR_B7': 'swir2.tif',
        'QA_PIXEL': 'qa_pixel.tif'
    }
    
    # Define angle bands
    angle_bands = ['SZA', 'VZA', 'SAA', 'VAA']
    
    # Load Landsat 8 Collection 2 Tier 1 Level 2 for reflectance data
    reflectance_collection = (ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
                   .filterBounds(region)
                   .filterDate(START_DATE, END_DATE)
                   .filterMetadata('CLOUD_COVER', 'less_than', MAX_CLOUD_COVER)
                   .select(list(reflectance_bands.keys()))
                   .sort('system:time_start'))
    
    # Load Landsat 8 Collection 2 Tier 1 for angle data
    angles_collection = (ee.ImageCollection('LANDSAT/LC08/C02/T1')
                   .filterBounds(region)
                   .filterDate(START_DATE, END_DATE)
                   .filterMetadata('CLOUD_COVER', 'less_than', MAX_CLOUD_COVER)
                   .select(angle_bands)
                   .sort('system:time_start'))
    
    # Get the list of images for reflectance
    reflectance_images = reflectance_collection.toList(reflectance_collection.size()).getInfo()
    print(f"Found {len(reflectance_images)} images for site {site_name}")
    
    # Get the list of images for angles
    angles_images = angles_collection.toList(angles_collection.size()).getInfo()
    print(f"Found {len(angles_images)} images for angle data at site {site_name}")
    
    # Store angle files by type to calculate means later
    angle_files_by_type = {angle: [] for angle in angle_bands}
    
    # Process reflectance and angle data for each acquisition date
    for i, image in enumerate(reflectance_images):
        # Get reflectance image
        img = ee.Image(image['id'])
        date = img.date().format('YYYY-MM-dd').getInfo()
        
        # Create directory for this acquisition date
        date_dir = os.path.join(site_dir, date)
        os.makedirs(date_dir, exist_ok=True)
        
        # Download reflectance bands
        for band, filename in reflectance_bands.items():
            save_path = os.path.join(date_dir, filename)
            
            # Skip if file already exists
            if os.path.exists(save_path):
                print(f"{filename} already exists for {date}. Skipping.")
                continue
            
            print(f"Downloading {band} as {filename} for date {date}...")
            try:
                url = img.select([band]).getDownloadURL({
                    'scale': 30,
                    'format': 'GEO_TIFF',
                    'region': region.getInfo()['coordinates']
                })
                
                response = requests.get(url, stream=True)
                if response.status_code == 200:
                    with open(save_path, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=1024):
                            f.write(chunk)
                    print(f"Downloaded {filename} successfully.")
                else:
                    print(f"Failed to download {filename}. HTTP Status: {response.status_code}")
            except Exception as e:
                print(f"Error downloading {band} on {date}: {e}")
    
    # Process angle data for each date
    for i, image in enumerate(angles_images):
        # Get angle image
        img = ee.Image(image['id'])
        date = img.date().format('YYYY-MM-dd').getInfo()
        
        # Download angle bands to temporary directory
        for band in angle_bands:
            temp_file = os.path.join(temp_angles_dir, f"{date}_{band}.tif")
            angle_files_by_type[band].append(temp_file)
            
            # Skip if file already exists
            if os.path.exists(temp_file):
                print(f"{band} angle file already exists for {date}. Skipping.")
                continue
            
            print(f"Downloading {band} angle for date {date}...")
            try:
                url = img.select([band]).getDownloadURL({
                    'scale': 30,
                    'format': 'GEO_TIFF',
                    'region': region.getInfo()['coordinates']
                })
                
                response = requests.get(url, stream=True)
                if response.status_code == 200:
                    with open(temp_file, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=1024):
                            f.write(chunk)
                    print(f"Downloaded {band} angle for {date} successfully.")
                else:
                    print(f"Failed to download {band} angle. HTTP Status: {response.status_code}")
            except Exception as e:
                print(f"Error downloading {band} angle on {date}: {e}")
    
    # Calculate means for each angle type
    site_metadata["angles"] = {}
    for angle in angle_bands:
        print(f"Calculating mean values for {angle}...")
        angle_means = calculate_angle_means(angle_files_by_type[angle], angle)
        site_metadata["angles"][angle] = angle_means
    
    # Save site metadata with angle means to JSON
    metadata_file = os.path.join(site_dir, "site_metadata.json")
    with open(metadata_file, 'w') as f:
        json.dump(site_metadata, f, indent=4)
    
    print(f"Saved metadata with angle means to {metadata_file}")
    
    # Clean up temporary angle files
    print(f"Removing temporary angle files...")
    shutil.rmtree(temp_angles_dir)
    
    print(f"Completed processing site: {site_name}")

def process_site(site_info):
    """
    Process a single site (wrapper function for parallel processing).
    
    Args:
        site_info (list): [site_name, latitude, longitude, landcover_type]
    """
    site_name, lat, lon, landcover_type = site_info
    try:
        download_and_process_data(site_name, lat, lon, landcover_type)
    except Exception as e:
        print(f"Error processing site {site_name}: {e}")

def main():
    """
    Main function to process all sites with optimal worker configuration.
    """
    print("Starting Landsat 8 data download process...")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Date range: {START_DATE} to {END_DATE}")
    print(f"Max cloud cover: {MAX_CLOUD_COVER}%")
    print(f"Number of sites: {len(SITES)}")
    
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Get optimal number of process workers
    process_workers, _ = get_optimal_workers()
    
    # Process sites in parallel
    with ProcessPoolExecutor(max_workers=process_workers) as executor:
        executor.map(process_site, SITES)
    
    print("All Landsat data downloads and processing completed successfully.")

if __name__ == '__main__':
    main()