import os 
import torch
from torch.utils.data import Dataset , DataLoader , random_split
from torchvision import transforms
import os 
import json
import rasterio
import numpy as np
from torchvision.transforms.functional import resize
from rasterio.warp import reproject, Resampling

def torch_select_unsqueeze(tensor, select_dim, nb_dim):
    """Helper function to unsqueeze tensor to match nb_dim"""
    shape = [1] * nb_dim
    shape[select_dim] = -1
    return tensor.view(shape)

def standardize(x, loc, scale, dim=0):
    nb_dim = len(x.size())
    standardized_x = (
        x - torch_select_unsqueeze(loc, select_dim=dim, nb_dim=nb_dim)
    ) / torch_select_unsqueeze(scale, select_dim=dim, nb_dim=nb_dim)
    return standardized_x

def unstandardize(x, loc, scale, dim=0):
    nb_dim = len(x.size())
    unstandardized_x = (
        x * torch_select_unsqueeze(scale, select_dim=dim, nb_dim=nb_dim)
    ) + torch_select_unsqueeze(loc, select_dim=dim, nb_dim=nb_dim)
    return unstandardized_x

class dataset(Dataset):
    def __init__(self , path_data_sentinel,path_data_landsat ):
        super().__init__()

        self.path_list_sentinel = []
        self.path_list_landsat = []
        self.angles_sentinel = {}
        self.angles_landsat = {}
        
        # Landsat statistics
        self.landsat_mean = torch.tensor([0.0607, 0.0893, 0.1058, 0.2282, 0.1923, 0.1370])
        self.landsat_std = torch.tensor([0.3014, 0.3119, 0.4129, 0.4810, 0.4955, 0.4050])
        
        # Sentinel statistics
        self.sentinel_mean = torch.tensor([0.0627, 0.0852, 0.0981, 0.2051, 0.1842, 0.1372])
        self.sentinel_std = torch.tensor([0.2806, 0.3053, 0.3890, 0.4685, 0.4471, 0.3791])
        
        # Angle statistics (loc and scale from cos-transformed angles)
        self.sentinel_angles_loc = torch.tensor([0.5275, 0.9903, -0.0571, -0.0576], dtype=torch.float32)
        self.sentinel_angles_scale = torch.tensor([0.4351, 0.0093, 0.9429, 0.9418], dtype=torch.float32)
        self.landsat_angles_loc = torch.tensor([0.5419, 0.9947, 0.0001, 0.0442], dtype=torch.float32)
        self.landsat_angles_scale = torch.tensor([0.4581, 0.0053, 0.9999, 0.9558], dtype=torch.float32)
        
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
                    
                    # Convert to cos(radians)
                    sza_cos = np.cos(np.radians(sza))
                    vza_cos = np.cos(np.radians(vza))
                    saa_cos = np.cos(np.radians(saa))
                    vaa_cos = np.cos(np.radians(vaa))
                    
                    site_date = site_name +"_"+date
                    self.angles_sentinel[site_date] = [sza_cos, vza_cos, saa_cos, vaa_cos]
                for dir in os.listdir(subdir_path) :
                    subsubdir_path = os.path.join(subdir_path,dir) 
                    if os.path.isdir(subsubdir_path):
                        self.path_list_sentinel.append(subsubdir_path)

            
        for subdir in os.listdir(path_data_landsat) : 
            subdir_path = os.path.join(path_data_landsat,subdir)
            if os.path.isdir(subdir_path) : 
                
                site_name = os.path.basename(subdir_path)
                json_file_path = os.path.join(subdir_path,"site_metadata.json")
                with open(json_file_path,'r') as f : 
                    metadata = json.load(f)
                for date in metadata["angles"]["SZA"].keys():
                    date_clean = date[:10]
                    site_date = f"{site_name}_{date_clean}"
                    
                    sza = metadata["angles"]["SZA"][date] 
                    vza = metadata["angles"]["VZA"][date] 
                    saa = metadata["angles"]["SAA"][date] 
                    vaa = metadata["angles"]["VAA"][date] 
                    
                    # Convert to cos(radians)
                    sza_cos = np.cos(np.radians(sza))
                    vza_cos = np.cos(np.radians(vza))
                    saa_cos = np.cos(np.radians(saa))
                    vaa_cos = np.cos(np.radians(vaa))
                    
                    self.angles_landsat[site_date] = [sza_cos, vza_cos, saa_cos, vaa_cos]
                for dir in os.listdir(subdir_path) : 
                        subsubdir = os.path.join(subdir_path,dir)
                        if os.path.isdir(subsubdir):
                                self.path_list_landsat.append(subsubdir)

    def __len__(self) : 
        return(len(self.path_list_landsat))
    
    def __getitem__(self, index):
        path_landsat = self.path_list_landsat[index]
        band_names = ['blue.tif', 'green.tif', 'red.tif', 'nir.tif', 'swir1.tif', 'swir2.tif']
        date_landsat = os.path.basename(path_landsat)
        site_name_landsat =  os.path.basename(os.path.dirname(path_landsat))
        date_site_landsat = f"{site_name_landsat}_{date_landsat}" 
        list_angles_landsat = self.angles_landsat[date_site_landsat]
        ref_list = []
        for band in band_names : 
            tif_path = os.path.join(path_landsat,band)
            with rasterio.open(tif_path) as src:
                        data = src.read(1)
                        data = data.astype(np.float32) * 0.0000275 - 0.2
            data_tensor = torch.from_numpy(data).unsqueeze(0)
            data_resized = resize(data_tensor, [128,128]).squeeze(0)
            ref_list.append(data_resized)
        data_tensor_landsat = torch.stack(ref_list,dim=0)
        data_tensor_landsat = standardize(data_tensor_landsat, self.landsat_mean, self.landsat_std, dim=0)

        # Random index for Sentinel
        random_index = torch.randint(0, len(self.path_list_sentinel), (1,)).item()
        path_sentinel = self.path_list_sentinel[random_index]
        date_sentinel = os.path.basename(path_sentinel)
        site_name_sentinel  =  os.path.basename(os.path.dirname(path_sentinel))
        date_site_sentinel = f"{site_name_sentinel}_{date_sentinel}" 
        list_angles_sentinel = self.angles_sentinel[date_site_sentinel]
        ref_list_sentinel = []
        
        ref_file = os.path.join(path_sentinel, 'blue.tif')
        with rasterio.open(ref_file) as ref_src:
                        ref_shape = ref_src.shape
                        ref_transform = ref_src.transform
                        ref_crs = ref_src.crs
        for band in band_names : 
            tif_path = os.path.join(path_sentinel,band)
            with rasterio.open(tif_path) as src:
                             
                        if band == 'swir1.tif'or band == 'swir2.tif' : 
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
                            data = data.astype(np.float32) / 10000.0
                        else : 
                            data = src.read(1)
                            data = data.astype(np.float32) / 10000.0
                        
            
            data_tensor = torch.from_numpy(data).unsqueeze(0)
            data_resized = resize(data_tensor, [384, 384]).squeeze(0)
            ref_list_sentinel.append(data_resized)
        data_tensor_sentinel = torch.stack(ref_list_sentinel,dim=0)
        data_tensor_sentinel = standardize(data_tensor_sentinel, self.sentinel_mean, self.sentinel_std, dim=0)

        # Standardize angles using loc and scale
        list_angles_sentinel = standardize(
            torch.tensor(list_angles_sentinel, dtype=torch.float32),
            self.sentinel_angles_loc,
            self.sentinel_angles_scale,
            dim=0
        )
        list_angles_landsat = standardize(
            torch.tensor(list_angles_landsat, dtype=torch.float32),
            self.landsat_angles_loc,
            self.landsat_angles_scale,
            dim=0
        )

        return data_tensor_landsat,list_angles_landsat,data_tensor_sentinel,list_angles_sentinel

#test
data = dataset(path_data_landsat="/home/mrhouma/Documents/CycleGan/CycleGan/landsat_data",path_data_sentinel="/home/mrhouma/Documents/CycleGan/CycleGan/data_sentinel2")
tensor_landsat , angles_landsat ,tensor_sentinel , angles_sentinel = data[500]
print(tensor_landsat.shape , tensor_landsat.min(), tensor_landsat.max())
print(angles_landsat.shape , angles_landsat.min(), angles_landsat.max())
print(tensor_landsat.amax(dim=(1, 2)))  # Max value per channel
print(tensor_sentinel.shape , tensor_sentinel.min(), tensor_sentinel.max())
print(angles_sentinel.shape , angles_sentinel.min(), angles_sentinel.max())
print(tensor_sentinel.amax(dim=(1, 2)))  # Max value per channel
print((tensor_sentinel > 2).sum())  # Count how many pixels are outliers