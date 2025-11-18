import os 
import json
import torch
import numpy as np

def min_max_to_loc_scale(minimum, maximum):
    loc = (maximum + minimum) / 2
    scale = (maximum - minimum) / 2
    return loc, scale

path_data_sentinel = "/home/mrhouma/Documents/CycleGan/CycleGan/data_sentinel2"
angles = []
for subdir in os.listdir(path_data_sentinel) : 
            subdir_path = os.path.join(path_data_sentinel,subdir)
            if os.path.isdir(subdir_path) : 
                site_name = os.path.basename(subdir_path)
                json_file_path = os.path.join(subdir_path,"site_metadata.json")
                with open(json_file_path,'r') as f : 
                    metadata = json.load(f)
                
                for date , angles_data in metadata["angles_by_date"].items() : 
                    vaa = 0
                    vza = 0
                    sza = angles_data["MEAN_SOLAR_ZENITH_ANGLE"]
                    saa = angles_data["MEAN_SOLAR_AZIMUTH_ANGLE"]
                    for i in [2,3,4,8,11,12] : 
                        vaa += angles_data[f"MEAN_INCIDENCE_AZIMUTH_ANGLE_B{i}"]
                        vza += angles_data[f"MEAN_INCIDENCE_ZENITH_ANGLE_B{i}"]
                    vaa = vaa/6
                    vza = vza/6
                    
                    # Convert to radians and apply cos
                    sza_cos = np.cos(np.radians(sza))
                    vza_cos = np.cos(np.radians(vza))
                    saa_cos = np.cos(np.radians(saa))
                    vaa_cos = np.cos(np.radians(vaa))
                    
                    angles.append(torch.tensor([sza_cos, vza_cos, saa_cos, vaa_cos]))

path_data_landsat = "/home/mrhouma/Documents/CycleGan/CycleGan/landsat_data"
angles_landsat = []
for subdir in os.listdir(path_data_landsat) : 
            subdir_path = os.path.join(path_data_landsat,subdir)
            if os.path.isdir(subdir_path) : 
                
                site_name = os.path.basename(subdir_path)
                json_file_path = os.path.join(subdir_path,"site_metadata.json")
                with open(json_file_path,'r') as f : 
                    metadata = json.load(f)
                for date in metadata["angles"]["SZA"].keys():
                    
                    print(date)
                    date_clean = date[:10]                    
                    sza = metadata["angles"]["SZA"][date] 
                    vza = metadata["angles"]["VZA"][date.replace("SZA", "VZA")] 
                    saa = metadata["angles"]["SAA"][date.replace("SZA", "SAA")] 
                    vaa = metadata["angles"]["VAA"][date.replace("SZA", "VAA")] 
                    
                    # Convert to radians and apply cos
                    sza_cos = np.cos(np.radians(sza))
                    vza_cos = np.cos(np.radians(vza))
                    saa_cos = np.cos(np.radians(saa))
                    vaa_cos = np.cos(np.radians(vaa))
                    
                    angles_landsat.append(torch.tensor([sza_cos, vza_cos, saa_cos, vaa_cos])) 


# Sentinel angles
angles_tensor = torch.stack(angles, dim=0)
print("Sentinel angles shape:", angles_tensor.shape)

min_sentinel = angles_tensor.min(dim=0)[0]
max_sentinel = angles_tensor.max(dim=0)[0]
loc_sentinel, scale_sentinel = min_max_to_loc_scale(min_sentinel, max_sentinel)

print("Sentinel loc:", loc_sentinel)
print("Sentinel scale:", scale_sentinel)


# Landsat angles
angles_tensor_landsat = torch.stack(angles_landsat, dim=0)
print("Landsat angles shape:", angles_tensor_landsat.shape)

min_landsat = angles_tensor_landsat.min(dim=0)[0]
max_landsat = angles_tensor_landsat.max(dim=0)[0]
loc_landsat, scale_landsat = min_max_to_loc_scale(min_landsat, max_landsat)

print("Landsat loc:", loc_landsat)
print("Landsat scale:", scale_landsat)