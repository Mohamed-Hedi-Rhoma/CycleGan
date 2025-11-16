import rasterio 
import  numpy as np

with rasterio.open("/home/mrhouma/Documents/CycleGan/CycleGan/data_sentinel2/AS_Wetland_1/2019-01-27/qa_pixel.tif") as src:
                        data = src.read(1)
data = np.array(data)
print(data)
print(data.min() , data.shape , data.max())