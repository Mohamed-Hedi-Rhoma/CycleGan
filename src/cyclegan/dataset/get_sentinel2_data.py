import ee
import os
import pandas as pd
import numpy as np
import requests
import json
import shutil
import multiprocessing
import psutil
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

# Initialize Earth Engine (make sure to authenticate first)
ee.Initialize(project='ee-get-landsat-data')

# Global constants
OUTPUT_DIR = "data_sentinel2"  # Base directory for output
MAX_CLOUD_COVER = 20  # Maximum allowed cloud cover percentage
START_DATE = "2018-01-01"  # Start date for data collection
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

def create_bounding_box(lat, lon, pixel_count=384, resolution=10):
    """
    Creates a bounding box around the given latitude and longitude.
    
    Args:
        lat (float): Latitude of the center point
        lon (float): Longitude of the center point
        pixel_count (int): Number of pixels for the bounding box (default: 512)
        resolution (int): Resolution in meters per pixel (default: 10 for Sentinel-2)
        
    Returns:
        ee.Geometry: Earth Engine geometry object representing the bounding box
    """
    half_side = (pixel_count * resolution) / 2
    point = ee.Geometry.Point([lon, lat])
    region = point.buffer(half_side).bounds()
    return region

def extract_angles_from_metadata(properties):
    """
    Extracts angle information from Sentinel-2 image metadata properties.
    
    Args:
        properties (dict): Image properties dictionary
        
    Returns:
        dict: Dictionary containing mean solar and viewing angles
    """
    angles = {}
    
    # Extract solar angles
    solar_angles = {
        'MEAN_SOLAR_ZENITH_ANGLE': properties.get('MEAN_SOLAR_ZENITH_ANGLE', None),
        'MEAN_SOLAR_AZIMUTH_ANGLE': properties.get('MEAN_SOLAR_AZIMUTH_ANGLE', None),
    }
    angles.update(solar_angles)
    
    # Extract viewing angles - both zenith and azimuth for each band
    # Focus on important bands (visible, NIR, SWIR)
    important_bands = ['B2', 'B3', 'B4', 'B8', 'B11', 'B12']
    
    for band in important_bands:
        zenith_key = f'MEAN_INCIDENCE_ZENITH_ANGLE_{band}'
        azimuth_key = f'MEAN_INCIDENCE_AZIMUTH_ANGLE_{band}'
        
        if zenith_key in properties:
            angles[zenith_key] = properties[zenith_key]
        if azimuth_key in properties:
            angles[azimuth_key] = properties[azimuth_key]
    
    return angles

def download_and_process_data(site_name, lat, lon, landcover_type):
    """
    Downloads Sentinel-2 data for the given site.
    
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
    
    # Create a JSON metadata file for the site
    site_metadata = {
        "name": site_name,
        "latitude": lat,
        "longitude": lon,
        "landcover_type": landcover_type,
        "data_start": START_DATE,
        "data_end": END_DATE,
        "angles_by_date": {}  # Will store angle means by date
    }
    
    # Define the region
    region = create_bounding_box(lat, lon)
    
    # Define the bands to download for Sentinel-2
    # Using the correct band names based on the provided information
    sentinel_bands = {
        'B2': 'blue.tif',      # Blue (10m)
        'B3': 'green.tif',     # Green (10m)
        'B4': 'red.tif',       # Red (10m)
        'B8': 'nir.tif',       # NIR (10m)
        'B11': 'swir1.tif',    # SWIR 1 (20m)
        'B12': 'swir2.tif',    # SWIR 2 (20m)
        'QA60': 'qa_pixel.tif' # Cloud mask (60m)
    }
    
    # Use the updated (non-deprecated) collection as recommended in the warning message
    # COPERNICUS/S2_SR_HARMONIZED is the updated collection for surface reflectance
    collection = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                 .filterBounds(region)
                 .filterDate(START_DATE, END_DATE)
                 .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', MAX_CLOUD_COVER))
                 .select(list(sentinel_bands.keys())))
    
    # Get the list of images
    images = collection.toList(collection.size()).getInfo()
    print(f"Found {len(images)} Sentinel-2 images for site {site_name}")
    
    # Process each image
    for i, image_info in enumerate(images):
        try:
            # Get the image
            image_id = image_info['id']
            img = ee.Image(image_id)
            
            # Get acquisition date
            date = img.date().format('YYYY-MM-dd').getInfo()
            
            # Create directory for this acquisition date
            date_dir = os.path.join(site_dir, date)
            os.makedirs(date_dir, exist_ok=True)
            
            # Extract angle information from metadata
            properties = img.getInfo()['properties']
            angles = extract_angles_from_metadata(properties)
            site_metadata["angles_by_date"][date] = angles
            
            # Download each band
            for band, filename in sentinel_bands.items():
                save_path = os.path.join(date_dir, filename)
                
                # Skip if file already exists
                if os.path.exists(save_path):
                    print(f"{filename} already exists for {date}. Skipping.")
                    continue
                
                print(f"Downloading {band} as {filename} for date {date}...")
                try:
                    # Set the appropriate scale based on the band (10m for visible/NIR, 20m for SWIR, 60m for QA60)
                    scale = 10
                    if band in ['B11', 'B12']:
                        scale = 20
                    elif band == 'QA60':
                        scale = 60
                    
                    url = img.select([band]).getDownloadURL({
                        'scale': scale,
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
        
        except Exception as e:
            print(f"Error processing image {i} for site {site_name}: {e}")
    
    # Save site metadata with angle information to JSON
    metadata_file = os.path.join(site_dir, "site_metadata.json")
    with open(metadata_file, 'w') as f:
        json.dump(site_metadata, f, indent=4)
    
    print(f"Saved metadata with angle information to {metadata_file}")
    print(f"Completed processing site: {site_name}")

def process_site(site_info):
    """
    Process a single site (wrapper function for parallel processing).
    
    Args:
        site_info (list): [site_name, latitude, longitude, landcover_type]
    """
    site_name, lat, lon, landcover_type = site_info
    try:
        # Initialize Earth Engine in this process
        # This is necessary because each process needs its own EE initialization
        ee.Initialize(project='ee-get-landsat-data')
        
        download_and_process_data(site_name, lat, lon, landcover_type)
    except Exception as e:
        print(f"Error processing site {site_name}: {e}")

def main():
    """
    Main function to process all sites with optimal worker configuration.
    """
    print("Starting Sentinel-2 data download process...")
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
    
    print("All Sentinel-2 data downloads and processing completed successfully.")

if __name__ == '__main__':
    main()