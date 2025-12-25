import sys
sys.path.append('/home/mrhouma/Documents/CycleGan/CycleGan/src/cyclegan')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import itertools
from torch.utils.data import random_split



from cyclegan.Generator_L2S import Generator_L2S
from cyclegan.Generator_S2L import Generator_S2L
from cyclegan.MultiScaleDiscriminator_Landsat import MultiScaleDiscriminator_Landsat
from cyclegan.MultiScaleDiscriminator_Sentinel import MultiScaleDiscriminator_Sentinel

# Import your dataset
from cyclegan.dataset.dataset import Gan_dataset  # or whatever your dataset class is called

from cyclegan.dataset.dataset import unstandardize 

from cyclegan.sam_loss import SAMLoss


seed = 30
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Training configuration
config = {
    # Paths
    'path_landsat': '/kaggle/input/data-landsat/landsat_data',
    'path_sentinel': '/kaggle/input/data-sentinel2/data_sentinel2',
    'checkpoint_dir': '/kaggle/working/checkpoints_seed30_vff',
    'results_dir': '/kaggle/working/results_seed30_vff',
    
    'num_epochs': 180,
    'batch_size': 4,  
    'lr_G': 0.0001,      
    'lr_D': 0.00002,    
    'beta1': 0.5,
    'beta2': 0.999,
    
    'lambda_cycle_L2S': 8.0,  
    'lambda_cycle_S2L': 5.0,  
    'lambda_GAN': 1.0,
    'lambda_SAM': 5.0,
    
    # Gradient clipping
    'max_grad_norm': 1.0,
    
    # NaN handling
    'nan_check_enabled': True,
    'max_nan_batches': 5, 
    
    # Model parameters
    'in_channels': 6,
    'n_angles': 4,
    
    # Logging
    'print_every': 10,
    'shuffle_data': True,
    
    # Device
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
}

# Print device
print(f"Using device: {config['device']}")
os.makedirs(config['checkpoint_dir'], exist_ok=True)
os.makedirs(config['results_dir'], exist_ok=True)

# Dataset
full_dataset = Gan_dataset(path_data_landsat=config['path_landsat'], path_data_sentinel=config['path_sentinel'])

# Split: 90% train, 10% validation
train_size = int(0.9 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

# Create DataLoaders
train_loader = DataLoader(
    train_dataset,
    batch_size=config['batch_size'],
    shuffle=True
)
val_loader = DataLoader(
    val_dataset,
    batch_size=config['batch_size'],
    shuffle=False
)

print(f"Train samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")

# Models
G_L2S = Generator_L2S(in_channels=6, n_angles=4).to(config['device'])
G_S2L = Generator_S2L(in_channels=6, n_angles=4).to(config['device'])
D_Landsat = MultiScaleDiscriminator_Landsat().to(config['device'])
D_Sentinel = MultiScaleDiscriminator_Sentinel().to(config['device'])

# Optimizers
params_G = itertools.chain(G_L2S.parameters(), G_S2L.parameters())
params_D = itertools.chain(D_Landsat.parameters(), D_Sentinel.parameters())

optimizer_G = optim.Adam(
    params_G,
    lr=config['lr_G'],
    betas=(config['beta1'], config['beta2'])
)
optimizer_D = optim.Adam(
    params_D,
    lr=config['lr_D'],
    betas=(config['beta1'], config['beta2'])
)

scheduler_G = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer_G, 
    mode='min',           
    factor=0.5,          
    patience=10,         
    min_lr=1e-6          
)
scheduler_D = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer_D,
    mode='min',
    factor=0.5,
    patience=10,
    min_lr=1e-6,
)

# Loss functions
criterion_GAN = nn.MSELoss()
criterion_cycle = nn.L1Loss()
criterion_SAM = SAMLoss().to(config['device'])

print("Models, optimizers, and losses initialized!")
print(f"G_L2S parameters: {sum(p.numel() for p in G_L2S.parameters()):,}")
print(f"G_S2L parameters: {sum(p.numel() for p in G_S2L.parameters()):,}")
print(f"D_Landsat parameters: {sum(p.numel() for p in D_Landsat.parameters()):,}")
print(f"D_Sentinel parameters: {sum(p.numel() for p in D_Sentinel.parameters()):,}")


def load_checkpoint(checkpoint_path, G_L2S, G_S2L, D_Landsat, D_Sentinel,
                    optimizer_G, optimizer_D, scheduler_G, scheduler_D):
    """Load checkpoint and resume training"""
    print(f"\n{'='*80}")
    print(f"LOADING CHECKPOINT FROM: {checkpoint_path}")
    print(f"{'='*80}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # Load model states
    G_L2S.load_state_dict(checkpoint['G_L2S_state_dict'])
    G_S2L.load_state_dict(checkpoint['G_S2L_state_dict'])
    D_Landsat.load_state_dict(checkpoint['D_Landsat_state_dict'])
    D_Sentinel.load_state_dict(checkpoint['D_Sentinel_state_dict'])
    
    # Load optimizer states
    optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
    optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])
    
    # Load scheduler states
    scheduler_G.load_state_dict(checkpoint['scheduler_G_state_dict'])
    scheduler_D.load_state_dict(checkpoint['scheduler_D_state_dict'])
    
    # Move optimizer states to correct device
    for state in optimizer_G.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.cuda()
    
    for state in optimizer_D.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.cuda()
    
    # Load training info
    start_epoch = checkpoint['epoch']
    history = checkpoint['history']
    best_val_cycle = checkpoint['best_val_cycle']
    
    # Print current learning rates
    current_lr_G = optimizer_G.param_groups[0]['lr']
    current_lr_D = optimizer_D.param_groups[0]['lr']
    print(f"✓ Current LR_G: {current_lr_G:.6f}")
    print(f"✓ Current LR_D: {current_lr_D:.6f}")
    
    print(f"✓ Loaded checkpoint from epoch {start_epoch}")
    print(f"✓ Resuming from epoch {start_epoch + 1}")
    print(f"✓ Best validation cycle loss so far: {best_val_cycle:.6f}")
    print(f"{'='*80}\n")
    
    return start_epoch, history, best_val_cycle


def reload_best_checkpoint(G_L2S, G_S2L, D_Landsat, D_Sentinel,
                          optimizer_G, optimizer_D, checkpoint_dir):
    """Reload best checkpoint when NaN is detected"""
    best_path = os.path.join(checkpoint_dir, 'best_model.pth')
    
    if not os.path.exists(best_path):
        print("⚠️ Best model not found, cannot recover from NaN!")
        return False
    
    print(f"\n{'='*80}")
    print("🔄 RECOVERING FROM NaN - RELOADING BEST MODEL")
    print(f"{'='*80}")
    
    checkpoint = torch.load(best_path, map_location='cpu', weights_only=False)
    
    # Load model states
    G_L2S.load_state_dict(checkpoint['G_L2S_state_dict'])
    G_S2L.load_state_dict(checkpoint['G_S2L_state_dict'])
    D_Landsat.load_state_dict(checkpoint['D_Landsat_state_dict'])
    D_Sentinel.load_state_dict(checkpoint['D_Sentinel_state_dict'])
    
    # Load optimizer states
    optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
    optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])
    
    # Move optimizer states to correct device
    for state in optimizer_G.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.cuda()
    
    for state in optimizer_D.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.cuda()
    
    print(f"✓ Successfully reloaded best model from epoch {checkpoint['epoch']}")
    print(f"{'='*80}\n")
    
    return True


def check_for_nan(tensor, name=""):
    """Check if tensor contains NaN or Inf values"""
    if torch.isnan(tensor).any():
        print(f"⚠️ NaN detected in {name}")
        return True
    if torch.isinf(tensor).any():
        print(f"⚠️ Inf detected in {name}")
        return True
    return False


def visualize_and_save(epoch, G_L2S, G_S2L, 
                       landsat_batch1, angles_L_batch1, sentinel_batch1, angles_S_batch1,
                       landsat_batch2, angles_L_batch2, sentinel_batch2, angles_S_batch2,
                       save_path):
    """Generate and save sample images with multiple band combinations for 2 batches"""
    G_L2S.eval()
    G_S2L.eval()
    
    with torch.no_grad():
        # Process both batches
        batches_data = []
        
        for landsat_real, angles_L, sentinel_real, angles_S in [
            (landsat_batch1, angles_L_batch1, sentinel_batch1, angles_S_batch1),
            (landsat_batch2, angles_L_batch2, sentinel_batch2, angles_S_batch2)
        ]:
            # Generate fakes
            fake_sentinel = G_L2S(landsat_real, angles_S)
            fake_landsat = G_S2L(sentinel_real, angles_L)
            
            # Cycle reconstruction
            rec_landsat = G_S2L(fake_sentinel, angles_L)
            rec_sentinel = G_L2S(fake_landsat, angles_S)
            
            # Move to CPU
            landsat_real = landsat_real.cpu()
            sentinel_real = sentinel_real.cpu()
            fake_sentinel = fake_sentinel.cpu()
            fake_landsat = fake_landsat.cpu()
            rec_landsat = rec_landsat.cpu()
            rec_sentinel = rec_sentinel.cpu()
            
            # Unstandardize
            landsat_mean = torch.tensor([0.0607, 0.0893, 0.1058, 0.2282, 0.1923, 0.1370])
            landsat_std = torch.tensor([0.3014, 0.3119, 0.4129, 0.4810, 0.4955, 0.4050])
            sentinel_mean = torch.tensor([0.0627, 0.0852, 0.0981, 0.2051, 0.1842, 0.1372])
            sentinel_std = torch.tensor([0.2806, 0.3053, 0.3890, 0.4685, 0.4471, 0.3791])
            
            landsat_real = unstandardize(landsat_real, landsat_mean, landsat_std, dim=1)
            fake_landsat = unstandardize(fake_landsat, landsat_mean, landsat_std, dim=1)
            rec_landsat = unstandardize(rec_landsat, landsat_mean, landsat_std, dim=1)
            
            sentinel_real = unstandardize(sentinel_real, sentinel_mean, sentinel_std, dim=1)
            fake_sentinel = unstandardize(fake_sentinel, sentinel_mean, sentinel_std, dim=1)
            rec_sentinel = unstandardize(rec_sentinel, sentinel_mean, sentinel_std, dim=1)
            
            batches_data.append({
                'landsat_real': landsat_real,
                'sentinel_real': sentinel_real,
                'fake_sentinel': fake_sentinel,
                'fake_landsat': fake_landsat,
                'rec_landsat': rec_landsat,
                'rec_sentinel': rec_sentinel
            })
        
        def percentile_stretch(img, p_low=2, p_high=98):
            img_np = img.numpy()
            p2 = np.percentile(img_np, p_low)
            p98 = np.percentile(img_np, p_high)
            if p98 - p2 > 0:
                img_np = (img_np - p2) / (p98 - p2)
            img_np = np.clip(img_np, 0, 1)
            return img_np
        
        def get_composite(tensor, band_indices):
            return tensor[:, band_indices, :, :]
        
        band_combos = {
            'True RGB': [2, 1, 0],
            'False NIR': [3, 2, 1],
            'SWIR Composite': [4, 3, 2],
        }
        
        # Create plots for each band combination and each batch
        for combo_name, bands in band_combos.items():
            for batch_idx, batch_data in enumerate(batches_data):
                
                fig, axes = plt.subplots(4, 6, figsize=(18, 12))
                
                landsat_real = batch_data['landsat_real']
                sentinel_real = batch_data['sentinel_real']
                fake_sentinel = batch_data['fake_sentinel']
                fake_landsat = batch_data['fake_landsat']
                rec_landsat = batch_data['rec_landsat']
                rec_sentinel = batch_data['rec_sentinel']
                
                for i in range(4):
                    L_real = get_composite(landsat_real, bands)[i].permute(1, 2, 0)
                    S_fake = get_composite(fake_sentinel, bands)[i].permute(1, 2, 0)
                    L_rec = get_composite(rec_landsat, bands)[i].permute(1, 2, 0)
                    S_real = get_composite(sentinel_real, bands)[i].permute(1, 2, 0)
                    L_fake = get_composite(fake_landsat, bands)[i].permute(1, 2, 0)
                    S_rec = get_composite(rec_sentinel, bands)[i].permute(1, 2, 0)
                    
                    L_real = percentile_stretch(L_real)
                    S_fake = percentile_stretch(S_fake)
                    L_rec = percentile_stretch(L_rec)
                    S_real = percentile_stretch(S_real)
                    L_fake = percentile_stretch(L_fake)
                    S_rec = percentile_stretch(S_rec)
                    
                    axes[i, 0].imshow(L_real)
                    axes[i, 0].set_title('Real Landsat' if i == 0 else '')
                    axes[i, 0].axis('off')
                    
                    axes[i, 1].imshow(S_fake)
                    axes[i, 1].set_title('Fake Sentinel (L→S)' if i == 0 else '')
                    axes[i, 1].axis('off')
                    
                    axes[i, 2].imshow(L_rec)
                    axes[i, 2].set_title('Rec Landsat' if i == 0 else '')
                    axes[i, 2].axis('off')
                    
                    axes[i, 3].imshow(S_real)
                    axes[i, 3].set_title('Real Sentinel' if i == 0 else '')
                    axes[i, 3].axis('off')
                    
                    axes[i, 4].imshow(L_fake)
                    axes[i, 4].set_title('Fake Landsat (S→L)' if i == 0 else '')
                    axes[i, 4].axis('off')
                    
                    axes[i, 5].imshow(S_rec)
                    axes[i, 5].set_title('Rec Sentinel' if i == 0 else '')
                    axes[i, 5].axis('off')
                
                combo_clean = combo_name.replace(' ', '_').lower()
                plt.suptitle(f'Epoch {epoch} - {combo_name} - Batch {batch_idx + 1}', fontsize=16)
                plt.tight_layout()
                plt.savefig(f'{save_path}/epoch_{epoch:03d}_{combo_clean}_batch{batch_idx + 1}.png', dpi=150, bbox_inches='tight')
                plt.close()
    
    G_L2S.train()
    G_S2L.train()


def validate(G_L2S, G_S2L, val_loader, criterion_cycle, criterion_SAM, device, config):
    G_L2S.eval()
    G_S2L.eval()
    
    val_cycle_losses = []
    val_SAM_losses = []
    
    # Per-band accumulators for Landsat and Sentinel
    landsat_band_errors = torch.zeros(6)
    sentinel_band_errors = torch.zeros(6)
    num_samples = 0
    
    with torch.no_grad():
        for landsat_real, angles_L, sentinel_real, angles_S in val_loader:
            landsat_real = landsat_real.to(device)
            angles_L = angles_L.to(device)
            sentinel_real = sentinel_real.to(device)
            angles_S = angles_S.to(device)
            
            fake_sentinel = G_L2S(landsat_real, angles_S)
            rec_landsat = G_S2L(fake_sentinel, angles_L)
            
            fake_landsat = G_S2L(sentinel_real, angles_L)
            rec_sentinel = G_L2S(fake_landsat, angles_S)
            
            # Cycle loss - ASYMMETRIC WEIGHTING
            loss_cycle_L = criterion_cycle(rec_landsat, landsat_real)
            loss_cycle_S = criterion_cycle(rec_sentinel, sentinel_real)
            loss_cycle = config['lambda_cycle_S2L'] * loss_cycle_L + config['lambda_cycle_L2S'] * loss_cycle_S
            
            # SAM loss
            loss_SAM_L = criterion_SAM(rec_landsat, landsat_real)
            loss_SAM_S = criterion_SAM(rec_sentinel, sentinel_real)
            loss_SAM = loss_SAM_L + loss_SAM_S
            
            val_cycle_losses.append(loss_cycle.item())
            val_SAM_losses.append(loss_SAM.item())
            
            # Compute per-band MAE
            landsat_error = torch.abs(rec_landsat - landsat_real)
            landsat_band_mae = landsat_error.mean(dim=[0, 2, 3])
            landsat_band_errors += landsat_band_mae.cpu()
            
            sentinel_error = torch.abs(rec_sentinel - sentinel_real)
            sentinel_band_mae = sentinel_error.mean(dim=[0, 2, 3])
            sentinel_band_errors += sentinel_band_mae.cpu()
            
            num_samples += 1
    
    # Average over all batches
    landsat_band_errors /= num_samples
    sentinel_band_errors /= num_samples
    
    G_L2S.train()
    G_S2L.train()
    
    return np.mean(val_cycle_losses), np.mean(val_SAM_losses), landsat_band_errors, sentinel_band_errors


def save_checkpoint(epoch, G_L2S, G_S2L, D_Landsat, D_Sentinel, 
                   optimizer_G, optimizer_D, scheduler_G, scheduler_D,
                   history, best_val_cycle, path, is_best=False):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'G_L2S_state_dict': G_L2S.state_dict(),
        'G_S2L_state_dict': G_S2L.state_dict(),
        'D_Landsat_state_dict': D_Landsat.state_dict(),
        'D_Sentinel_state_dict': D_Sentinel.state_dict(),
        'optimizer_G_state_dict': optimizer_G.state_dict(),
        'optimizer_D_state_dict': optimizer_D.state_dict(),
        'scheduler_G_state_dict': scheduler_G.state_dict(),
        'scheduler_D_state_dict': scheduler_D.state_dict(),
        'history': history,
        'best_val_cycle': best_val_cycle
    }
    
    # Save latest checkpoint (overwrites)
    torch.save(checkpoint, f'{path}/latest_checkpoint.pth')
    print(f"✓ Saved checkpoint: {path}/latest_checkpoint.pth")
    
    # Save best model separately
    if is_best:
        torch.save(checkpoint, f'{path}/best_model.pth')
        print(f"★ Best model saved! (epoch {epoch})")



start_epoch = 0
history = {
    'L_D': [],
    'L_G': [],
    'L_cycle': [],
    'L_GAN': [],
    'L_SAM': [],
    'val_cycle': [],
    'val_SAM': []
}
best_val_cycle = float('inf')

checkpoint_path = os.path.join(config['checkpoint_dir'], 'latest_checkpoint.pth')

if os.path.exists(checkpoint_path):
    print(f"\n{'='*80}")
    print(f"🔄 FOUND EXISTING CHECKPOINT!")
    print(f"{'='*80}")
    
    try:
        start_epoch, history, best_val_cycle = load_checkpoint(
            checkpoint_path, G_L2S, G_S2L, D_Landsat, D_Sentinel,
            optimizer_G, optimizer_D, scheduler_G, scheduler_D
        )
        print("✅ Successfully resumed training from checkpoint")
    except Exception as e:
        print(f"❌ Failed to load checkpoint: {e}")
        print("Starting training from scratch...")
        start_epoch = 0
        history = {
            'L_D': [],
            'L_G': [],
            'L_cycle': [],
            'L_GAN': [],
            'L_SAM': [],
            'val_cycle': [],
            'val_SAM': []
        }
        best_val_cycle = float('inf')
else:
    print(f"\nNo existing checkpoint found at {checkpoint_path}")
    print("Starting training from scratch...")


# Fixed samples for visualization - 2 batches (8 samples total)
val_iter = iter(val_loader)
fixed_val_batch1 = next(val_iter)
fixed_val_batch2 = next(val_iter)

fixed_landsat_1 = fixed_val_batch1[0][:4].to(config['device'])
fixed_angles_L_1 = fixed_val_batch1[1][:4].to(config['device'])
fixed_sentinel_1 = fixed_val_batch1[2][:4].to(config['device'])
fixed_angles_S_1 = fixed_val_batch1[3][:4].to(config['device'])

fixed_landsat_2 = fixed_val_batch2[0][:4].to(config['device'])
fixed_angles_L_2 = fixed_val_batch2[1][:4].to(config['device'])
fixed_sentinel_2 = fixed_val_batch2[2][:4].to(config['device'])
fixed_angles_S_2 = fixed_val_batch2[3][:4].to(config['device'])


consecutive_nan_batches = 0

for epoch in range(start_epoch, config['num_epochs']):
    
    G_L2S.train()
    G_S2L.train()
    D_Landsat.train()
    D_Sentinel.train()
    
    epoch_L_D = []
    epoch_L_G = []
    epoch_L_cycle = []
    epoch_L_GAN = []
    epoch_L_SAM = []
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['num_epochs']}")
    
    for batch_idx, (landsat_real, angles_L, sentinel_real, angles_S) in enumerate(pbar):
        
        landsat_real = landsat_real.to(config['device'])
        angles_L = angles_L.to(config['device'])
        sentinel_real = sentinel_real.to(config['device'])
        angles_S = angles_S.to(config['device'])
        
        # ========================================
        # TRAIN DISCRIMINATORS
        # ========================================
        optimizer_D.zero_grad()
        
        fake_sentinel = G_L2S(landsat_real, angles_S)
        fake_landsat = G_S2L(sentinel_real, angles_L)
        
        out1_real, out2_real = D_Landsat(landsat_real)
        out1_fake, out2_fake = D_Landsat(fake_landsat.detach())
        
        loss_D_L_real_1 = criterion_GAN(out1_real, torch.ones_like(out1_real))
        loss_D_L_fake_1 = criterion_GAN(out1_fake, torch.zeros_like(out1_fake))
        loss_D_L_1 = (loss_D_L_real_1 + loss_D_L_fake_1) * 0.5
        
        loss_D_L_real_2 = criterion_GAN(out2_real, torch.ones_like(out2_real))
        loss_D_L_fake_2 = criterion_GAN(out2_fake, torch.zeros_like(out2_fake))
        loss_D_L_2 = (loss_D_L_real_2 + loss_D_L_fake_2) * 0.5
        
        loss_D_Landsat = loss_D_L_1 + loss_D_L_2
        
        out1_real, out2_real = D_Sentinel(sentinel_real)
        out1_fake, out2_fake = D_Sentinel(fake_sentinel.detach())
        
        loss_D_S_real_1 = criterion_GAN(out1_real, torch.ones_like(out1_real))
        loss_D_S_fake_1 = criterion_GAN(out1_fake, torch.zeros_like(out1_fake))
        loss_D_S_1 = (loss_D_S_real_1 + loss_D_S_fake_1) * 0.5
        
        loss_D_S_real_2 = criterion_GAN(out2_real, torch.ones_like(out2_real))
        loss_D_S_fake_2 = criterion_GAN(out2_fake, torch.zeros_like(out2_fake))
        loss_D_S_2 = (loss_D_S_real_2 + loss_D_S_fake_2) * 0.5
        
        loss_D_Sentinel = loss_D_S_1 + loss_D_S_2
        
        loss_D_total = loss_D_Landsat + loss_D_Sentinel
        
        # ========================================
        # NaN CHECK FOR DISCRIMINATOR LOSS
        # ========================================
        if config['nan_check_enabled'] and check_for_nan(loss_D_total, "Discriminator Loss"):
            print(f"Epoch {epoch+1}, Batch {batch_idx}: Skipping batch due to NaN in D loss")
            consecutive_nan_batches += 1
            
            if consecutive_nan_batches >= config['max_nan_batches']:
                print(f"\n⚠️ {consecutive_nan_batches} consecutive NaN batches detected!")
                success = reload_best_checkpoint(
                    G_L2S, G_S2L, D_Landsat, D_Sentinel,
                    optimizer_G, optimizer_D, config['checkpoint_dir']
                )
                if not success:
                    print("Cannot recover. Stopping training.")
                    break
                consecutive_nan_batches = 0
            continue
        
        loss_D_total.backward()
        
        # ========================================
        # GRADIENT CLIPPING FOR DISCRIMINATORS
        # ========================================
        torch.nn.utils.clip_grad_norm_(D_Landsat.parameters(), max_norm=config['max_grad_norm'])
        torch.nn.utils.clip_grad_norm_(D_Sentinel.parameters(), max_norm=config['max_grad_norm'])
        
        optimizer_D.step()
        
        # ========================================
        # TRAIN GENERATORS
        # ========================================
        optimizer_G.zero_grad()
        
        fake_sentinel = G_L2S(landsat_real, angles_S)
        fake_landsat = G_S2L(sentinel_real, angles_L)
        
        rec_landsat = G_S2L(fake_sentinel, angles_L)
        rec_sentinel = G_L2S(fake_landsat, angles_S)
        
        out1_fake, out2_fake = D_Landsat(fake_landsat)
        loss_GAN_S2L_1 = criterion_GAN(out1_fake, torch.ones_like(out1_fake))
        loss_GAN_S2L_2 = criterion_GAN(out2_fake, torch.ones_like(out2_fake))
        loss_GAN_S2L = loss_GAN_S2L_1 + loss_GAN_S2L_2
        
        out1_fake, out2_fake = D_Sentinel(fake_sentinel)
        loss_GAN_L2S_1 = criterion_GAN(out1_fake, torch.ones_like(out1_fake))
        loss_GAN_L2S_2 = criterion_GAN(out2_fake, torch.ones_like(out2_fake))
        loss_GAN_L2S = loss_GAN_L2S_1 + loss_GAN_L2S_2
        
        loss_GAN = loss_GAN_S2L + loss_GAN_L2S
        
        # ASYMMETRIC CYCLE LOSS
        loss_cycle_L = criterion_cycle(rec_landsat, landsat_real)
        loss_cycle_S = criterion_cycle(rec_sentinel, sentinel_real)
        loss_cycle = config['lambda_cycle_S2L'] * loss_cycle_L + config['lambda_cycle_L2S'] * loss_cycle_S
        
        loss_SAM_L = criterion_SAM(rec_landsat, landsat_real)
        loss_SAM_S = criterion_SAM(rec_sentinel, sentinel_real)
        loss_SAM = loss_SAM_L + loss_SAM_S 
        
        # TOTAL GENERATOR LOSS
        loss_G_total = config['lambda_GAN'] * loss_GAN + loss_cycle + config['lambda_SAM'] * loss_SAM
        
        # ========================================
        # NaN CHECK FOR GENERATOR LOSS
        # ========================================
        if config['nan_check_enabled'] and check_for_nan(loss_G_total, "Generator Loss"):
            print(f"Epoch {epoch+1}, Batch {batch_idx}: Skipping batch due to NaN in G loss")
            consecutive_nan_batches += 1
            
            if consecutive_nan_batches >= config['max_nan_batches']:
                print(f"\n⚠️ {consecutive_nan_batches} consecutive NaN batches detected!")
                success = reload_best_checkpoint(
                    G_L2S, G_S2L, D_Landsat, D_Sentinel,
                    optimizer_G, optimizer_D, config['checkpoint_dir']
                )
                if not success:
                    print("Cannot recover. Stopping training.")
                    break
                consecutive_nan_batches = 0
            continue
        
        loss_G_total.backward()
        
        # ========================================
        # GRADIENT CLIPPING FOR GENERATORS
        # ========================================
        torch.nn.utils.clip_grad_norm_(G_L2S.parameters(), max_norm=config['max_grad_norm'])
        torch.nn.utils.clip_grad_norm_(G_S2L.parameters(), max_norm=config['max_grad_norm'])
        
        optimizer_G.step()
        
        # Reset NaN counter on successful batch
        consecutive_nan_batches = 0
        
        epoch_L_D.append(loss_D_total.item())
        epoch_L_G.append(loss_G_total.item())
        epoch_L_cycle.append(loss_cycle.item())
        epoch_L_GAN.append(loss_GAN.item())
        epoch_L_SAM.append(loss_SAM.item())
        
        pbar.set_postfix({
            'L_D': f'{loss_D_total.item():.4f}',
            'L_G': f'{loss_G_total.item():.4f}',
            'cycle': f'{loss_cycle.item():.4f}'
        })
    
    # ========================================
    # VALIDATION
    # ========================================
    val_cycle, val_SAM, landsat_band_errors, sentinel_band_errors = validate(
        G_L2S, G_S2L, val_loader, criterion_cycle, criterion_SAM, config['device'], config
    )
    
    # Store epoch averages
    history['L_D'].append(np.mean(epoch_L_D))
    history['L_G'].append(np.mean(epoch_L_G))
    history['L_cycle'].append(np.mean(epoch_L_cycle))
    history['L_GAN'].append(np.mean(epoch_L_GAN))
    history['L_SAM'].append(np.mean(epoch_L_SAM)) 
    history['val_cycle'].append(val_cycle)
    history['val_SAM'].append(val_SAM)
    
    # Compute combined validation metric
    val_combined = val_cycle + config['lambda_SAM'] * val_SAM
    
    scheduler_G.step(val_combined)
    scheduler_D.step(val_combined)
    
    # Band names for printing
    band_names = ['Blue', 'Green', 'Red', 'NIR', 'SWIR1', 'SWIR2']
    
    print(f"\nEpoch {epoch+1}/{config['num_epochs']} Summary:")
    print(f"  L_D: {history['L_D'][-1]:.4f}")
    print(f"  L_G: {history['L_G'][-1]:.4f}")
    print(f"  Train cycle: {history['L_cycle'][-1]:.4f}")
    print(f"  Val cycle: {val_cycle:.4f}")
    print(f"  Train SAM: {history['L_SAM'][-1]:.4f}")
    print(f"  Val SAM: {val_SAM:.4f}")
    print(f"  L_GAN: {history['L_GAN'][-1]:.4f}")
    print(f"  LR_G: {optimizer_G.param_groups[0]['lr']:.6f}")
    print(f"  LR_D: {optimizer_D.param_groups[0]['lr']:.6f}")
    
    # Print per-band errors
    print("\n  Landsat band MAE (standardized):")
    for i, name in enumerate(band_names):
        print(f"    {name:6s}: {landsat_band_errors[i]:.4f}")
    print(f"    Worst: {band_names[landsat_band_errors.argmax()]}")
    
    print("\n  Sentinel band MAE (standardized):")
    for i, name in enumerate(band_names):
        print(f"    {name:6s}: {sentinel_band_errors[i]:.4f}")
    print(f"    Worst: {band_names[sentinel_band_errors.argmax()]}")
    
    # ========================================
    # SAVE CHECKPOINT EVERY EPOCH
    # ========================================
    is_best = val_cycle < best_val_cycle
    
    if is_best:
        best_val_cycle = val_cycle
        print(f"  ★ New best validation cycle: {best_val_cycle:.4f}")
    
    # Save checkpoint (latest always, best only if improved)
    save_checkpoint(
        epoch + 1, G_L2S, G_S2L, D_Landsat, D_Sentinel,
        optimizer_G, optimizer_D, scheduler_G, scheduler_D,
        history, best_val_cycle, config['checkpoint_dir'], is_best=is_best
    )
    
    # Visualize only on new best
    if is_best:
        visualize_and_save(
            epoch + 1, G_L2S, G_S2L,
            fixed_landsat_1, fixed_angles_L_1, fixed_sentinel_1, fixed_angles_S_1,
            fixed_landsat_2, fixed_angles_L_2, fixed_sentinel_2, fixed_angles_S_2,
            config['results_dir']
        )

# ========================================
# TRAINING COMPLETE
# ========================================
print("\n" + "="*50)
print("Training Complete!")
print(f"Best validation cycle loss: {best_val_cycle:.4f}")
print("="*50)

# Plot loss curves
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

axes[0, 0].plot(history['L_D'])
axes[0, 0].set_title('Discriminator Loss')
axes[0, 0].set_xlabel('Epoch')

axes[0, 1].plot(history['L_G'])
axes[0, 1].set_title('Generator Loss')
axes[0, 1].set_xlabel('Epoch')

axes[1, 0].plot(history['L_cycle'], label='Train')
axes[1, 0].plot(history['val_cycle'], label='Validation')
axes[1, 0].set_title('Cycle Consistency Loss')
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].legend()

axes[1, 1].plot(history['L_GAN'])
axes[1, 1].set_title('GAN Loss')
axes[1, 1].set_xlabel('Epoch')

plt.tight_layout()
plt.savefig(f"{config['results_dir']}/loss_curves.png", dpi=150)
plt.close()

print(f"Loss curves saved to {config['results_dir']}/loss_curves.png")
print(f"Best model saved to {config['checkpoint_dir']}/best_model.pth")
print(f"Latest checkpoint saved to {config['checkpoint_dir']}/latest_checkpoint.pth")


