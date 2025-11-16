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
class dataset(Dataset):
    def __init__(self , path_data_sentinel,path_data_landsat ):
        super().__init__()


        self.path_list_sentinel = []
        self.path_list_landsat = []
        self.angles_sentinel = {}
        self.angles_landsat = {}
        self.normalize_landsat = transforms.Normalize(
            mean=[0.0795, 0.1130, 0.1353, 0.2234, 0.1995, 0.1528],
            std=[0.0651, 0.0767, 0.1059, 0.1264, 0.1359, 0.1125]
        )
        self.normalize_sentinel = transforms.Normalize(
            mean=[0.0826, 0.1048, 0.1230, 0.2006, 0.1860, 0.1494],
            std=[0.0714, 0.0797, 0.1048, 0.1327, 0.1318, 0.1132]
        )
        
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
                    site_date = site_name +"_"+date
                    self.angles_sentinel[site_date]= [sza, vza, saa, vaa]
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
                    
                    self.angles_landsat[site_date] = [sza, vza, saa, vaa] 
                for dir in os.listdir(subdir_path) : 
                        subsubdir = os.path.join(subdir_path,dir)
                        if os.path.isdir(subsubdir):
                                self.path_list_landsat.append(subsubdir)
        


    def __len__(self) : 
        return(len(self.path_list_landsat))
    def normalize_angles(self, angles, mean, std):
        """Normalize angle values using mean and std"""
        angles = torch.tensor(angles, dtype=torch.float32)
        mean = torch.tensor(mean, dtype=torch.float32)
        std = torch.tensor(std, dtype=torch.float32)
        return (angles - mean) / std
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
        data_tensor_landsat = self.normalize_landsat(data_tensor_landsat)

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
        data_tensor_sentinel = self.normalize_sentinel(data_tensor_sentinel)
        data_tensor_sentinel = torch.clamp(data_tensor_sentinel, -3, 3)
        data_tensor_landsat = torch.clamp(data_tensor_landsat, -3, 3)

        list_angles_sentinel = self.normalize_angles(
            list_angles_sentinel,
            [40.3609, 5.8707, 119.3870, 187.6938],
            [14.9709, 2.4810, 50.8305, 78.1721]
        )
        list_angles_landsat = self.normalize_angles(
            list_angles_landsat,
            [40.5064, 3.9567, 109.0848, 17.1651],
            [16.0608, 2.4479, 57.1288, 90.3701]
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


        



        

        