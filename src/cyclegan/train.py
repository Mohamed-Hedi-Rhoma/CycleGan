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
from cyclegan.dataset import dataset  # or whatever your dataset class is called

# Import your standardization functions for visualization
from cyclegan.dataset import unstandardize




# Training configuration
config = {
    # Paths
    'path_landsat': '/home/mrhouma/Documents/CycleGan/CycleGan/landsat_data',
    'path_sentinel': '/home/mrhouma/Documents/CycleGan/CycleGan/data_sentinel2',
    'checkpoint_dir': './checkpoints',
    'results_dir': './results',
    
    # Training hyperparameters
    'num_epochs': 100,
    'batch_size': 4,  
    'lr_G': 0.0002,   # Generator learning rate
    'lr_D': 0.0002,   # Discriminator learning rate
    'beta1': 0.5,     # Adam beta1
    'beta2': 0.999,   # Adam beta2
    
    # Loss weights
    'lambda_cycle': 10.0,
    'lambda_GAN': 1.0,
    
    # Model parameters
    'in_channels': 6,
    'n_angles': 4,
    
    # Logging
    'print_every': 10,      # Print losses every N batches
    'save_every': 5,        # Save checkpoint every N epochs
    'visualize_every': 5,   # Save sample images every N epochs
    'shuffle_data' : True,
    
    # Device
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    
    'resume': False,
    'checkpoint_path': None,
}

# Print device
print(f"Using device: {config['device']}")
os.makedirs(config['checkpoint_dir'], exist_ok=True)
os.makedirs(config['results_dir'], exist_ok=True)

full_dataset = dataset(path_data_landsat=config['path_landsat'],path_data_sentinel=config['path_sentinel'])
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
    shuffle=False  # No shuffle for validation!
)

print(f"Train samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")

G_L2S = Generator_L2S(in_channels=6, n_angles=4).to(config['device'])
G_S2L = Generator_S2L(in_channels=6, n_angles=4).to(config['device'])
D_Landsat = MultiScaleDiscriminator_Landsat().to(config['device'])
D_Sentinel = MultiScaleDiscriminator_Sentinel().to(config['device'])


params_G = itertools.chain(G_L2S.parameters(), G_S2L.parameters())
params_D = itertools.chain(D_Landsat.parameters(),D_Sentinel.parameters())
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

criterion_GAN = nn.MSELoss()    # For LSGAN
criterion_cycle = nn.L1Loss()   # For cycle consistency

print("Models, optimizers, and losses initialized!")
print(f"G_L2S parameters: {sum(p.numel() for p in G_L2S.parameters()):,}")
print(f"G_S2L parameters: {sum(p.numel() for p in G_S2L.parameters()):,}")
print(f"D_Landsat parameters: {sum(p.numel() for p in D_Landsat.parameters()):,}")
print(f"D_Sentinel parameters: {sum(p.numel() for p in D_Sentinel.parameters()):,}")


# ============================================
# PART 3: Training Loop
# ============================================

# Track losses
history = {
    'L_D': [],
    'L_G': [],
    'L_cycle': [],
    'L_GAN': [],
    'val_cycle': []
}

# For visualization - get fixed samples from validation set
fixed_val_batch = next(iter(val_loader))
fixed_landsat = fixed_val_batch[0][:4].to(config['device'])  # First 4 samples
fixed_angles_L = fixed_val_batch[1][:4].to(config['device'])
fixed_sentinel = fixed_val_batch[2][:4].to(config['device'])
fixed_angles_S = fixed_val_batch[3][:4].to(config['device'])


def visualize_and_save(epoch, G_L2S, G_S2L, landsat_real, angles_L, sentinel_real, angles_S, save_path):
    """Generate and save sample images"""
    G_L2S.eval()
    G_S2L.eval()
    
    with torch.no_grad():
        # Generate fakes
        fake_sentinel = G_L2S(landsat_real, angles_S)
        fake_landsat = G_S2L(sentinel_real, angles_L)
        
        # Cycle reconstruction
        rec_landsat = G_S2L(fake_sentinel, angles_L)
        rec_sentinel = G_L2S(fake_landsat, angles_S)
        
        # Move to CPU for plotting
        landsat_real = landsat_real.cpu()
        sentinel_real = sentinel_real.cpu()
        fake_sentinel = fake_sentinel.cpu()
        fake_landsat = fake_landsat.cpu()
        rec_landsat = rec_landsat.cpu()
        rec_sentinel = rec_sentinel.cpu()
        
        # Denormalize (use your unstandardize function)
        # Get stats from dataset
        landsat_mean = torch.tensor([0.0607, 0.0893, 0.1058, 0.2282, 0.1923, 0.1370])
        landsat_std = torch.tensor([0.3014, 0.3119, 0.4129, 0.4810, 0.4955, 0.4050])
        sentinel_mean = torch.tensor([0.0627, 0.0852, 0.0981, 0.2051, 0.1842, 0.1372])
        sentinel_std = torch.tensor([0.2806, 0.3053, 0.3890, 0.4685, 0.4471, 0.3791])
        
        # Unstandardize
        landsat_real = unstandardize(landsat_real, landsat_mean, landsat_std, dim=1)
        fake_landsat = unstandardize(fake_landsat, landsat_mean, landsat_std, dim=1)
        rec_landsat = unstandardize(rec_landsat, landsat_mean, landsat_std, dim=1)
        
        sentinel_real = unstandardize(sentinel_real, sentinel_mean, sentinel_std, dim=1)
        fake_sentinel = unstandardize(fake_sentinel, sentinel_mean, sentinel_std, dim=1)
        rec_sentinel = unstandardize(rec_sentinel, sentinel_mean, sentinel_std, dim=1)
        
        # Clamp to [0, 1]
        landsat_real = torch.clamp(landsat_real, 0, 1)
        fake_landsat = torch.clamp(fake_landsat, 0, 1)
        rec_landsat = torch.clamp(rec_landsat, 0, 1)
        sentinel_real = torch.clamp(sentinel_real, 0, 1)
        fake_sentinel = torch.clamp(fake_sentinel, 0, 1)
        rec_sentinel = torch.clamp(rec_sentinel, 0, 1)
        
        # Select RGB bands [red, green, blue] = [2, 1, 0]
        def to_rgb(tensor):
            return tensor[:, [2, 1, 0], :, :]
        
        landsat_real_rgb = to_rgb(landsat_real)
        fake_landsat_rgb = to_rgb(fake_landsat)
        rec_landsat_rgb = to_rgb(rec_landsat)
        sentinel_real_rgb = to_rgb(sentinel_real)
        fake_sentinel_rgb = to_rgb(fake_sentinel)
        rec_sentinel_rgb = to_rgb(rec_sentinel)
        
        # Plot
        fig, axes = plt.subplots(4, 6, figsize=(18, 12))
        
        for i in range(4):  # 4 samples
            # Row: Landsat cycle
            axes[i, 0].imshow(landsat_real_rgb[i].permute(1, 2, 0).numpy())
            axes[i, 0].set_title('Real Landsat' if i == 0 else '')
            axes[i, 0].axis('off')
            
            axes[i, 1].imshow(fake_sentinel_rgb[i].permute(1, 2, 0).numpy())
            axes[i, 1].set_title('Fake Sentinel (L→S)' if i == 0 else '')
            axes[i, 1].axis('off')
            
            axes[i, 2].imshow(rec_landsat_rgb[i].permute(1, 2, 0).numpy())
            axes[i, 2].set_title('Rec Landsat (L→S→L)' if i == 0 else '')
            axes[i, 2].axis('off')
            
            axes[i, 3].imshow(sentinel_real_rgb[i].permute(1, 2, 0).numpy())
            axes[i, 3].set_title('Real Sentinel' if i == 0 else '')
            axes[i, 3].axis('off')
            
            axes[i, 4].imshow(fake_landsat_rgb[i].permute(1, 2, 0).numpy())
            axes[i, 4].set_title('Fake Landsat (S→L)' if i == 0 else '')
            axes[i, 4].axis('off')
            
            axes[i, 5].imshow(rec_sentinel_rgb[i].permute(1, 2, 0).numpy())
            axes[i, 5].set_title('Rec Sentinel (S→L→S)' if i == 0 else '')
            axes[i, 5].axis('off')
        
        plt.suptitle(f'Epoch {epoch}', fontsize=16)
        plt.tight_layout()
        plt.savefig(f'{save_path}/epoch_{epoch:03d}.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    G_L2S.train()
    G_S2L.train()


def save_checkpoint(epoch, G_L2S, G_S2L, D_Landsat, D_Sentinel, optimizer_G, optimizer_D, history, path):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'G_L2S_state_dict': G_L2S.state_dict(),
        'G_S2L_state_dict': G_S2L.state_dict(),
        'D_Landsat_state_dict': D_Landsat.state_dict(),
        'D_Sentinel_state_dict': D_Sentinel.state_dict(),
        'optimizer_G_state_dict': optimizer_G.state_dict(),
        'optimizer_D_state_dict': optimizer_D.state_dict(),
        'history': history
    }
    torch.save(checkpoint, f'{path}/checkpoint_epoch_{epoch:03d}.pth')
    print(f"Checkpoint saved: epoch {epoch}")


# ============================================
# MAIN TRAINING LOOP
# ============================================

for epoch in range(config['num_epochs']):
    
    # Set models to training mode
    G_L2S.train()
    G_S2L.train()
    D_Landsat.train()
    D_Sentinel.train()
    
    epoch_L_D = []
    epoch_L_G = []
    epoch_L_cycle = []
    epoch_L_GAN = []
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['num_epochs']}")
    
    for batch_idx, (landsat_real, angles_L, sentinel_real, angles_S) in enumerate(pbar):
        
        # Move to device
        landsat_real = landsat_real.to(config['device'])
        angles_L = angles_L.to(config['device'])
        sentinel_real = sentinel_real.to(config['device'])
        angles_S = angles_S.to(config['device'])
        
        # =============================
        # TRAIN DISCRIMINATORS
        # =============================
        
        optimizer_D.zero_grad()
        
        # Generate fakes
        fake_sentinel = G_L2S(landsat_real, angles_S)
        fake_landsat = G_S2L(sentinel_real, angles_L)
        
        # --- D_Landsat ---
        out1_real, out2_real = D_Landsat(landsat_real)
        out1_fake, out2_fake = D_Landsat(fake_landsat.detach())
        
        # Scale 1 loss
        loss_D_L_real_1 = criterion_GAN(out1_real, torch.ones_like(out1_real))
        loss_D_L_fake_1 = criterion_GAN(out1_fake, torch.zeros_like(out1_fake))
        loss_D_L_1 = (loss_D_L_real_1 + loss_D_L_fake_1) * 0.5
        
        # Scale 2 loss
        loss_D_L_real_2 = criterion_GAN(out2_real, torch.ones_like(out2_real))
        loss_D_L_fake_2 = criterion_GAN(out2_fake, torch.zeros_like(out2_fake))
        loss_D_L_2 = (loss_D_L_real_2 + loss_D_L_fake_2) * 0.5
        
        loss_D_Landsat = loss_D_L_1 + loss_D_L_2
        
        # --- D_Sentinel ---
        out1_real, out2_real = D_Sentinel(sentinel_real)
        out1_fake, out2_fake = D_Sentinel(fake_sentinel.detach())
        
        # Scale 1 loss
        loss_D_S_real_1 = criterion_GAN(out1_real, torch.ones_like(out1_real))
        loss_D_S_fake_1 = criterion_GAN(out1_fake, torch.zeros_like(out1_fake))
        loss_D_S_1 = (loss_D_S_real_1 + loss_D_S_fake_1) * 0.5
        
        # Scale 2 loss
        loss_D_S_real_2 = criterion_GAN(out2_real, torch.ones_like(out2_real))
        loss_D_S_fake_2 = criterion_GAN(out2_fake, torch.zeros_like(out2_fake))
        loss_D_S_2 = (loss_D_S_real_2 + loss_D_S_fake_2) * 0.5
        
        loss_D_Sentinel = loss_D_S_1 + loss_D_S_2
        
        # Total D loss
        loss_D_total = loss_D_Landsat + loss_D_Sentinel
        loss_D_total.backward()
        optimizer_D.step()
        
        # =============================
        # TRAIN GENERATORS
        # =============================
        
        optimizer_G.zero_grad()
        
        # Generate fakes (fresh forward pass)
        fake_sentinel = G_L2S(landsat_real, angles_S)
        fake_landsat = G_S2L(sentinel_real, angles_L)
        
        # Cycle reconstruction
        rec_landsat = G_S2L(fake_sentinel, angles_L)
        rec_sentinel = G_L2S(fake_landsat, angles_S)
        
        # --- GAN Loss (fool discriminators) ---
        out1_fake, out2_fake = D_Landsat(fake_landsat)
        loss_GAN_S2L_1 = criterion_GAN(out1_fake, torch.ones_like(out1_fake))
        loss_GAN_S2L_2 = criterion_GAN(out2_fake, torch.ones_like(out2_fake))
        loss_GAN_S2L = loss_GAN_S2L_1 + loss_GAN_S2L_2
        
        out1_fake, out2_fake = D_Sentinel(fake_sentinel)
        loss_GAN_L2S_1 = criterion_GAN(out1_fake, torch.ones_like(out1_fake))
        loss_GAN_L2S_2 = criterion_GAN(out2_fake, torch.ones_like(out2_fake))
        loss_GAN_L2S = loss_GAN_L2S_1 + loss_GAN_L2S_2
        
        loss_GAN = loss_GAN_S2L + loss_GAN_L2S
        
        # --- Cycle Loss ---
        loss_cycle_L = criterion_cycle(rec_landsat, landsat_real)
        loss_cycle_S = criterion_cycle(rec_sentinel, sentinel_real)
        loss_cycle = loss_cycle_L + loss_cycle_S
        
        # --- Total Generator Loss ---
        loss_G_total = config['lambda_GAN'] * loss_GAN + config['lambda_cycle'] * loss_cycle
        loss_G_total.backward()
        optimizer_G.step()
        
        # =============================
        # LOGGING
        # =============================
        
        epoch_L_D.append(loss_D_total.item())
        epoch_L_G.append(loss_G_total.item())
        epoch_L_cycle.append(loss_cycle.item())
        epoch_L_GAN.append(loss_GAN.item())
        
        # Update progress bar
        pbar.set_postfix({
            'L_D': f'{loss_D_total.item():.4f}',
            'L_G': f'{loss_G_total.item():.4f}',
            'cycle': f'{loss_cycle.item():.4f}'
        })
    
    # =============================
    # END OF EPOCH
    # =============================
    
    # Store epoch averages
    history['L_D'].append(np.mean(epoch_L_D))
    history['L_G'].append(np.mean(epoch_L_G))
    history['L_cycle'].append(np.mean(epoch_L_cycle))
    history['L_GAN'].append(np.mean(epoch_L_GAN))
    
    # Print epoch summary
    print(f"\nEpoch {epoch+1}/{config['num_epochs']} Summary:")
    print(f"  L_D: {history['L_D'][-1]:.4f}")
    print(f"  L_G: {history['L_G'][-1]:.4f}")
    print(f"  L_cycle: {history['L_cycle'][-1]:.4f}")
    print(f"  L_GAN: {history['L_GAN'][-1]:.4f}")
    
    # Save checkpoint
    if (epoch + 1) % config['save_every'] == 0:
        save_checkpoint(
            epoch + 1, G_L2S, G_S2L, D_Landsat, D_Sentinel,
            optimizer_G, optimizer_D, history, config['checkpoint_dir']
        )
    
    # Visualize samples
    if (epoch + 1) % config['visualize_every'] == 0:
        visualize_and_save(
            epoch + 1, G_L2S, G_S2L,
            fixed_landsat, fixed_angles_L, fixed_sentinel, fixed_angles_S,
            config['results_dir']
        )

# ============================================
# TRAINING COMPLETE
# ============================================

print("\n" + "="*50)
print("Training Complete!")
print("="*50)

# Save final checkpoint
save_checkpoint(
    config['num_epochs'], G_L2S, G_S2L, D_Landsat, D_Sentinel,
    optimizer_G, optimizer_D, history, config['checkpoint_dir']
)

# Plot loss curves
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

axes[0, 0].plot(history['L_D'])
axes[0, 0].set_title('Discriminator Loss')
axes[0, 0].set_xlabel('Epoch')

axes[0, 1].plot(history['L_G'])
axes[0, 1].set_title('Generator Loss')
axes[0, 1].set_xlabel('Epoch')

axes[1, 0].plot(history['L_cycle'])
axes[1, 0].set_title('Cycle Consistency Loss')
axes[1, 0].set_xlabel('Epoch')

axes[1, 1].plot(history['L_GAN'])
axes[1, 1].set_title('GAN Loss')
axes[1, 1].set_xlabel('Epoch')

plt.tight_layout()
plt.savefig(f"{config['results_dir']}/loss_curves.png", dpi=150)
plt.close()

print(f"Loss curves saved to {config['results_dir']}/loss_curves.png")



