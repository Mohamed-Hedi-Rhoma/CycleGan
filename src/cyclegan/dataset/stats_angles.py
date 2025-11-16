import os 
import json
import torch

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
                    vaa = 0  # Reset here!
                    vza = 0  # Reset here!
                    sza = angles_data["MEAN_SOLAR_ZENITH_ANGLE"]
                    saa = angles_data["MEAN_SOLAR_AZIMUTH_ANGLE"]
                    for i in [2,3,4,8,11,12] : 
                        vaa += angles_data[f"MEAN_INCIDENCE_AZIMUTH_ANGLE_B{i}"]
                        vza += angles_data[f"MEAN_INCIDENCE_ZENITH_ANGLE_B{i}"]
                    vaa = vaa/6
                    vza = vza/6
                    
                    angles.append(torch.tensor([sza, vza, saa, vaa]))

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
                    
                    angles_landsat.append(torch.tensor([sza, vza, saa, vaa])) 


angles_tensor = torch.stack(angles,dim=0)
print(angles_tensor.shape)
mean_sentinel = angles_tensor.mean(dim=0)
print(mean_sentinel)
std_sentinel = angles_tensor.std(dim=0)
print(std_sentinel)


angles_tensor_landsat = torch.stack(angles_landsat,dim=0)
print(angles_tensor_landsat.shape)
mean_landsat = angles_tensor_landsat.mean(dim=0)
print(mean_landsat)
std_landsat = angles_tensor_landsat.std(dim=0)
print(std_landsat)