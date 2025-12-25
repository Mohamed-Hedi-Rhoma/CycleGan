import os
import yaml
import torch
import rasterio
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
from pathlib import Path
import tempfile
from PIL import Image
import io
from typing import Literal

# Import your models
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from cyclegan.Generator_L2S import Generator_L2S
from cyclegan.Generator_S2L import Generator_S2L
from cyclegan.dataset.dataset import standardize, unstandardize

# Load configuration
config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

# Initialize FastAPI
app = FastAPI(
    title="CycleGAN Satellite Image Translation API",
    description="Translate between Landsat and Sentinel-2 imagery",
    version="1.0.0"
)

# Global variables for models
G_L2S = None
G_S2L = None
device = None

@app.on_event("startup")
async def load_models():
    """Load models on startup"""
    global G_L2S, G_S2L, device
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    checkpoint_path = config['model']['checkpoint_path']
    
    if not os.path.exists(checkpoint_path):
        print(f"ERROR: Checkpoint not found at {checkpoint_path}")
        print("Please run: pixi run dvc pull")
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device,weights_only=False)
    
    # Initialize models
    G_L2S = Generator_L2S(
        in_channels=config['model']['in_channels'],
        n_angles=config['model']['n_angles']
    ).to(device)
    
    G_S2L = Generator_S2L(
        in_channels=config['model']['in_channels'],
        n_angles=config['model']['n_angles']
    ).to(device)
    
    # Load weights
    G_L2S.load_state_dict(checkpoint['G_L2S_state_dict'])
    G_S2L.load_state_dict(checkpoint['G_S2L_state_dict'])
    
    # Set to evaluation mode
    G_L2S.eval()
    G_S2L.eval()
    
    print("Models loaded successfully!")

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "CycleGAN Satellite Image Translation API",
        "version": "1.0.0",
        "endpoints": {
            "/health": "Check API health",
            "/predict": "Translate satellite imagery"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    models_loaded = G_L2S is not None and G_S2L is not None
    return {
        "status": "healthy" if models_loaded else "unhealthy",
        "models_loaded": models_loaded,
        "device": str(device)
    }

def process_geotiff(file_path: str, generator_type: str):
    """Read and process GeoTIFF file"""
    band_names = ['blue.tif', 'green.tif', 'red.tif', 'nir.tif', 'swir1.tif', 'swir2.tif']
    
    # For now, assume uploaded file is a single multi-band GeoTIFF
    # Read all 6 bands
    with rasterio.open(file_path) as src:
        if src.count != 6:
            raise HTTPException(
                status_code=400,
                detail=f"Expected 6 bands, got {src.count}"
            )
        
        # Read all bands
        data = src.read()  # Shape: (6, H, W)
        transform = src.transform
        crs = src.crs
    
    # Apply scaling based on sensor type
    if generator_type == 'L2S':
        # Landsat scaling
        data = data.astype(np.float32) * config['landsat']['scale_factor'] + config['landsat']['offset']
        target_size = config['model']['landsat_size']
    else:  # S2L
        # Sentinel scaling
        data = data.astype(np.float32) * config['sentinel']['scale_factor']
        target_size = config['model']['sentinel_size']
    
    # Convert to tensor and resize
    data_tensor = torch.from_numpy(data).float()  # Shape: (6, H, W)
    
    # Resize each band
    from torchvision.transforms.functional import resize
    data_resized = resize(data_tensor, [target_size, target_size])  # Shape: (6, size, size)
    
    return data_resized, transform, crs

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    generator: Literal['L2S', 'S2L'] = Form(...),
    angles: str = Form(...),
    output_format: Literal['geotiff', 'image'] = Form(...)
):
    """
    Translate satellite imagery
    
    Parameters:
    - file: GeoTIFF file with 6 bands
    - generator: 'L2S' (Landsat to Sentinel) or 'S2L' (Sentinel to Landsat)
    - angles: 4 comma-separated values (e.g., "0.5,0.99,-0.05,-0.06")
    - output_format: 'geotiff' or 'image'
    """
    
    # Validate file size
    if file.size and file.size > config['api']['max_file_size']:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Max size: {config['api']['max_file_size']} bytes"
        )
    
    # Parse angles
    try:
        angles_list = [float(x.strip()) for x in angles.split(',')]
        if len(angles_list) != 4:
            raise ValueError("Expected 4 angle values")
        angles_tensor = torch.tensor(angles_list, dtype=torch.float32)
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid angles format. Expected 4 comma-separated numbers. Error: {str(e)}"
        )
    
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix='.tif') as tmp_input:
        content = await file.read()
        tmp_input.write(content)
        tmp_input_path = tmp_input.name
    
    try:
        # Process input
        input_tensor, transform, crs = process_geotiff(tmp_input_path, generator)
        
        # Add batch dimension
        input_tensor = input_tensor.unsqueeze(0).to(device)  # Shape: (1, 6, size, size)
        angles_tensor = angles_tensor.unsqueeze(0).to(device)  # Shape: (1, 4)
        
        # Standardize input
        if generator == 'L2S':
            # Standardize Landsat input
            landsat_mean = torch.tensor(config['landsat']['mean']).to(device)
            landsat_std = torch.tensor(config['landsat']['std']).to(device)
            input_tensor = standardize(input_tensor, landsat_mean, landsat_std, dim=1)
            
            # Standardize angles (sentinel angles for output)
            angle_loc = torch.tensor(config['angles']['sentinel']['loc']).to(device)
            angle_scale = torch.tensor(config['angles']['sentinel']['scale']).to(device)
            angles_tensor = standardize(angles_tensor, angle_loc, angle_scale, dim=1)
            
            # Run generator
            with torch.no_grad():
                output_tensor = G_L2S(input_tensor, angles_tensor)
            
            # Unstandardize output
            sentinel_mean = torch.tensor(config['sentinel']['mean']).to(device)
            sentinel_std = torch.tensor(config['sentinel']['std']).to(device)
            output_tensor = unstandardize(output_tensor, sentinel_mean, sentinel_std, dim=1)
            
        else:  # S2L
            # Standardize Sentinel input
            sentinel_mean = torch.tensor(config['sentinel']['mean']).to(device)
            sentinel_std = torch.tensor(config['sentinel']['std']).to(device)
            input_tensor = standardize(input_tensor, sentinel_mean, sentinel_std, dim=1)
            
            # Standardize angles (landsat angles for output)
            angle_loc = torch.tensor(config['angles']['landsat']['loc']).to(device)
            angle_scale = torch.tensor(config['angles']['landsat']['scale']).to(device)
            angles_tensor = standardize(angles_tensor, angle_loc, angle_scale, dim=1)
            
            # Run generator
            with torch.no_grad():
                output_tensor = G_S2L(input_tensor, angles_tensor)
            
            # Unstandardize output
            landsat_mean = torch.tensor(config['landsat']['mean']).to(device)
            landsat_std = torch.tensor(config['landsat']['std']).to(device)
            output_tensor = unstandardize(output_tensor, landsat_mean, landsat_std, dim=1)
        
        # Remove batch dimension and move to CPU
        output_tensor = output_tensor.squeeze(0).cpu()  # Shape: (6, size, size)
        output_np = output_tensor.numpy()
        
        # Return based on output format
        if output_format == 'geotiff':
            # Save as GeoTIFF
            with tempfile.NamedTemporaryFile(delete=False, suffix='.tif') as tmp_output:
                tmp_output_path = tmp_output.name
            
            # Write GeoTIFF
            with rasterio.open(
                tmp_output_path,
                'w',
                driver='GTiff',
                height=output_np.shape[1],
                width=output_np.shape[2],
                count=6,
                dtype=output_np.dtype,
                crs=crs,
                transform=transform
            ) as dst:
                for i in range(6):
                    dst.write(output_np[i], i + 1)
            
            return FileResponse(
                tmp_output_path,
                media_type='image/tiff',
                filename=f"translated_{generator}.tif"
            )
        
        else:  # image format
            # Create RGB composite for visualization (bands 2,1,0 = R,G,B)
            rgb = output_np[[2, 1, 0], :, :]  # Shape: (3, H, W)
            rgb = np.transpose(rgb, (1, 2, 0))  # Shape: (H, W, 3)
            
            # Percentile stretch for visualization
            p2 = np.percentile(rgb, 2)
            p98 = np.percentile(rgb, 98)
            rgb = np.clip((rgb - p2) / (p98 - p2), 0, 1)
            rgb = (rgb * 255).astype(np.uint8)
            
            # Convert to PIL Image
            img = Image.fromarray(rgb)
            
            # Save to bytes
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format='PNG')
            img_byte_arr.seek(0)
            
            # Save temporarily and return
            with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_img:
                tmp_img.write(img_byte_arr.getvalue())
                tmp_img_path = tmp_img.name
            
            return FileResponse(
                tmp_img_path,
                media_type='image/png',
                filename=f"translated_{generator}.png"
            )
    
    finally:
        # Clean up input file
        if os.path.exists(tmp_input_path):
            os.remove(tmp_input_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host=config['api']['host'],
        port=config['api']['port']
    )