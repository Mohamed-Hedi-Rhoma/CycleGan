import os 
import json
import torch
import numpy as np

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
problematic_paths = []

for subdir in os.listdir(path_data_landsat) : 
            subdir_path = os.path.join(path_data_landsat,subdir)
            if os.path.isdir(subdir_path) : 
                
                site_name = os.path.basename(subdir_path)
                json_file_path = os.path.join(subdir_path,"site_metadata.json")
                with open(json_file_path,'r') as f : 
                    metadata = json.load(f)
                for date in metadata["angles"]["SZA"].keys():
                    try:
                        sza = metadata["angles"]["SZA"][date] 
                        vza = metadata["angles"]["VZA"][date]
                        saa = metadata["angles"]["SAA"][date]
                        vaa = metadata["angles"]["VAA"][date]
                        
                        # Convert to radians and apply cos
                        sza_cos = np.cos(np.radians(sza))
                        vza_cos = np.cos(np.radians(vza))
                        saa_cos = np.cos(np.radians(saa))
                        vaa_cos = np.cos(np.radians(vaa))
                        
                        angles_landsat.append(torch.tensor([sza_cos, vza_cos, saa_cos, vaa_cos]))
                            
                    except KeyError as e:
                        problematic_paths.append({
                            'path': json_file_path,
                            'date': date,
                            'issue': f'Missing key: {str(e)}'
                        })


# Sentinel angles - using mean and std of COSINES
angles_tensor = torch.stack(angles, dim=0)
print("="*80)
print("SENTINEL-2 STATISTICS")
print("="*80)
print("Number of observations:", angles_tensor.shape[0])

mean_sentinel = angles_tensor.mean(dim=0)
std_sentinel = angles_tensor.std(dim=0)

print("\nMean of cosines:", mean_sentinel)
print("Std of cosines:", std_sentinel)


# Landsat angles - using mean and std of COSINES
if len(angles_landsat) > 0:
    angles_tensor_landsat = torch.stack(angles_landsat, dim=0)
    print("\n" + "="*80)
    print("LANDSAT STATISTICS")
    print("="*80)
    print("Number of observations:", angles_tensor_landsat.shape[0])

    mean_landsat = angles_tensor_landsat.mean(dim=0)
    std_landsat = angles_tensor_landsat.std(dim=0)

    print("\nMean of cosines:", mean_landsat)
    print("Std of cosines:", std_landsat)
else:
    print("\nNo valid Landsat angles found!")

# Print problematic paths at the end
if problematic_paths:
    print("\n" + "="*80)
    print("PROBLEMATIC PATHS WITH MISSING ANGLE DATA:")
    print("="*80)
    print(f"Total problematic entries: {len(problematic_paths)}")
    for item in problematic_paths[:10]:  # Show first 10
        print(f"Path: {item['path']}")
        print(f"Date: {item['date']}")
        print(f"Issue: {item['issue']}")
        print("-"*80)
    if len(problematic_paths) > 10:
        print(f"... and {len(problematic_paths) - 10} more problematic entries")
else:
    print("\nNo problematic paths found!")

# Print comparison table
print("\n" + "="*80)
print("COMPARISON: Mean ± Std of COSINE values")
print("="*80)
print(f"{'Angle':<12} | {'Sentinel-2':<25} | {'Landsat':<25}")
print("-"*80)
angle_names = ['SZA_cos', 'VZA_cos', 'SAA_cos', 'VAA_cos']
for i, name in enumerate(angle_names):
    sentinel_str = f"{mean_sentinel[i]:6.4f} ± {std_sentinel[i]:.4f}"
    landsat_str = f"{mean_landsat[i]:6.4f} ± {std_landsat[i]:.4f}"
    print(f"{name:<12} | {sentinel_str:<25} | {landsat_str:<25}")

# Save statistics to file for later use
print("\n" + "="*80)
print("SAVING NORMALIZATION PARAMETERS")
print("="*80)

normalization_params = {
    'sentinel': {
        'mean': mean_sentinel.tolist(),
        'std': std_sentinel.tolist()
    },
    'landsat': {
        'mean': mean_landsat.tolist(),
        'std': std_landsat.tolist()
    },
    'angle_names': angle_names,
    'note': 'Statistics computed on cosine-transformed angles (method: mean of cosines)'
}

output_file = '/home/mrhouma/Documents/CycleGan/CycleGan/angle_normalization_params.json'
with open(output_file, 'w') as f:
    json.dump(normalization_params, f, indent=2)
