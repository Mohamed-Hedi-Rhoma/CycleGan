# CycleGAN-SAM: Landsat ↔ Sentinel-2 Translation

## Project Goal
Perform unpaired bidirectional translation between Landsat 8 (30m resolution) and Sentinel-2 (10m resolution) satellite imagery using CycleGAN with angle conditioning and spectral angle preservation.

## Model Architecture

### Inputs
- **Landsat 8**: [B, 6, 128, 128] - Blue, Green, Red, NIR, SWIR1, SWIR2 bands
- **Sentinel-2**: [B, 6, 384, 384] - Blue, Green, Red, NIR, SWIR1, SWIR2 bands  
- **Landsat Angles**: [B, 4] - Solar zenith/azimuth, viewing zenith/azimuth angles (cosine-transformed)
- **Sentinel Angles**: [B, 4] - Solar zenith/azimuth, viewing zenith/azimuth angles (cosine-transformed)

### Architecture Flow

#### Generator L2S (Landsat → Sentinel)
1. **Encoder**: 3 convolutional layers downsample [B, 6, 128, 128] → [B, 256, 32, 32]
2. **Angle MLP**: Transforms angles [B, 4] → [B, 256] for FiLM conditioning
3. **ResBlocks (×9)**: FiLM-conditioned residual blocks maintain [B, 256, 32, 32]
4. **Decoder**: 2 transpose convolutions upsample to [B, 64, 128, 128]
5. **PixelShuffle**: 3× upsampling [B, 64, 128, 128] → [B, 64, 384, 384]
6. **Output Conv**: Final layers produce [B, 6, 384, 384]

#### Generator S2L (Sentinel → Landsat)
1. **Encoder**: Initial conv + PixelUnshuffle 3× downsampling [B, 6, 384, 384] → [B, 576, 128, 128]
2. **Downsampling**: Additional convs reduce to [B, 256, 32, 32]
3. **Angle MLP**: Transforms angles [B, 4] → [B, 256] for FiLM conditioning
4. **ResBlocks (×9)**: FiLM-conditioned residual blocks maintain [B, 256, 32, 32]
5. **Decoder**: 2 transpose convolutions upsample to [B, 64, 128, 128]
6. **Output Conv**: Final layers produce [B, 6, 128, 128]

#### Discriminators
- **Multi-Scale PatchGAN**: 2 scales with average pooling
- **Spectral Normalization**: Applied to all convolutional layers
- **Architecture**: 5 conv layers with InstanceNorm, outputs patch-wise predictions

### FiLM Conditioning
Each ResBlock uses Feature-wise Linear Modulation (FiLM) to inject angle information:
- Angle embedding → Linear layer → Split into γ (scale) and β (shift)
- After each InstanceNorm: `output = γ × normalized + β`

## Training

### Loss Functions
1. **GAN Loss** (λ=1.0): Adversarial loss from multi-scale discriminators
2. **Cycle Consistency Loss** (λ=10.0): L1 loss between original and cycle-reconstructed images
3. **SAM Loss** (λ=5.0): Spectral Angle Mapper preserves spectral signatures

**Total Generator Loss**: L_GAN + 10×L_cycle + 5×L_SAM

### Training Strategy
- **Unpaired Training**: Random Landsat-Sentinel pairing in each batch
- **Alternating Updates**: Train discriminators → Train generators
- **Validation**: 90% train / 10% validation split
- **Early Stopping**: Save only when validation cycle loss improves

### Training Settings
- **Optimizers**: Adam (G: lr=3e-4, D: lr=1e-4, β₁=0.5, β₂=0.999)
- **Scheduler**: ReduceLROnPlateau (factor=0.5, patience=10, min_lr=1e-6)
- **Batch Size**: 4
- **Epochs**: 100
- **Seed**: 20 (for reproducibility)

### Data Preprocessing
- **Landsat**: Scale factor 0.0000275, offset -0.2, resize to 128×128
- **Sentinel**: Scale factor 1/10000, SWIR bands reproject to match resolution, resize to 384×384
- **Standardization**: Per-sensor mean/std normalization
- **Angle Encoding**: Convert to cos(radians), then standardize

### Usage
```bash
python train.py
```

## Dataset Structure
```
landsat_data/
├── site_001/
│   ├── site_metadata.json          # Contains angles by date
│   ├── 2024-01-15/
│   │   ├── blue.tif
│   │   ├── green.tif
│   │   ├── red.tif
│   │   ├── nir.tif
│   │   ├── swir1.tif
│   │   └── swir2.tif
│   └── ...
└── ...

data_sentinel2/
├── site_001/
│   ├── site_metadata.json          # Contains angles by date
│   ├── 2024-01-15/
│   │   ├── blue.tif
│   │   ├── green.tif
│   │   ├── red.tif
│   │   ├── nir.tif
│   │   ├── swir1.tif
│   │   └── swir2.tif
│   └── ...
└── ...
```

## Validation Metrics

### Per-Band MAE
Tracks mean absolute error for each spectral band:
- Blue, Green, Red, NIR, SWIR1, SWIR2
- Identifies problematic bands (typically SWIR bands show highest error)

### Combined Metric
Validation loss = L_cycle + 5×L_SAM  
Used for learning rate scheduling and model checkpointing

## Outputs

### Checkpoints
- **Best Model**: `checkpoints/best_model.pth` (overwrites on improvement)
- **Contains**: Generator/discriminator weights, optimizer states, training history

### Visualizations
Generated on validation improvement:
- **3 Band Combinations**: True RGB, False Color NIR, SWIR Composite
- **2 Batches**: 8 samples total (4 per batch)
- **6 Images per Sample**:
  1. Real Landsat
  2. Fake Sentinel (L→S)
  3. Reconstructed Landsat (cycle)
  4. Real Sentinel
  5. Fake Landsat (S→L)
  6. Reconstructed Sentinel (cycle)
- **Saved to**: `results/epoch_XXX_{combo}_batchY.png`

### Loss Curves
Final training plot saved to `results/loss_curves.png`:
- Discriminator loss
- Generator loss
- Cycle consistency (train + validation)
- GAN loss

## Installation

First, install pixi package manager:
```bash
curl -fsSL https://pixi.sh/install.sh | bash 
```

Install project dependencies:
```bash
pixi install
```

## Key Features

- ✅ **Angle-Aware**: FiLM conditioning accounts for varying illumination/viewing geometry
- ✅ **Spectral Preservation**: SAM loss maintains spectral signatures across translations
- ✅ **Resolution Handling**: PixelShuffle/PixelUnshuffle for efficient 3× upsampling/downsampling
- ✅ **Multi-Scale Discrimination**: Captures both fine details and global structure
- ✅ **Robust Training**: Spectral normalization prevents mode collapse
- ✅ **Validation Tracking**: Per-band error analysis identifies weaknesses