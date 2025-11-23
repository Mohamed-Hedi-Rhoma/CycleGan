import torch
import torch.nn as nn

class SAMLoss(nn.Module):
    """Spectral Angle Mapper Loss"""
    def __init__(self, eps=1e-8):
        super(SAMLoss, self).__init__()
        self.eps = eps  # Small value to avoid division by zero
    
    def forward(self, pred, target):
        """
        pred: [B, C, H, W] - predicted reflectance
        target: [B, C, H, W] - real reflectance
        
        Returns: mean SAM angle in radians
        """
        # Flatten spatial dimensions: [B, C, H, W] -> [B, C, H*W]
        pred_flat = pred.view(pred.shape[0], pred.shape[1], -1)
        target_flat = target.view(target.shape[0], target.shape[1], -1)
        
        # Compute dot product: [B, H*W]
        dot_product = (pred_flat * target_flat).sum(dim=1)
        
        # Compute L2 norms: [B, H*W]
        pred_norm = torch.sqrt((pred_flat ** 2).sum(dim=1) + self.eps)
        target_norm = torch.sqrt((target_flat ** 2).sum(dim=1) + self.eps)
        
        # Compute cosine similarity
        cos_angle = dot_product / (pred_norm * target_norm + self.eps)
        
        # Clamp to [-1, 1] to avoid numerical issues with arccos
        cos_angle = torch.clamp(cos_angle, -1.0, 1.0)
        
        # Compute angle in radians
        angle = torch.acos(cos_angle)
        
        # Return mean angle across all pixels and batch
        return angle.mean()