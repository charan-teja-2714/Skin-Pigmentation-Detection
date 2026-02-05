import torch
import torch.nn as nn
import torch.nn.functional as F


class MSELoss(nn.Module):
    """Standard Mean Squared Error Loss for regression tasks."""
    
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
    
    def forward(self, predictions, targets):
        return self.mse(predictions, targets)


class SmoothL1Loss(nn.Module):
    """Smooth L1 Loss (Huber Loss) - less sensitive to outliers than MSE."""
    
    def __init__(self, beta=1.0):
        super().__init__()
        self.smooth_l1 = nn.SmoothL1Loss(beta=beta)
    
    def forward(self, predictions, targets):
        return self.smooth_l1(predictions, targets)


class WeightedMSELoss(nn.Module):
    """Weighted MSE Loss - gives more weight to certain severity ranges."""
    
    def __init__(self, mild_weight=1.0, moderate_weight=1.5, severe_weight=2.0):
        super().__init__()
        self.mild_weight = mild_weight
        self.moderate_weight = moderate_weight
        self.severe_weight = severe_weight
    
    def forward(self, predictions, targets):
        # Calculate weights based on target severity
        weights = torch.ones_like(targets)
        
        # Mild: 0.0-0.25
        mild_mask = targets <= 0.25
        weights[mild_mask] = self.mild_weight
        
        # Moderate: 0.25-0.6
        moderate_mask = (targets > 0.25) & (targets <= 0.6)
        weights[moderate_mask] = self.moderate_weight
        
        # Severe: 0.6-1.0
        severe_mask = targets > 0.6
        weights[severe_mask] = self.severe_weight
        
        # Weighted MSE
        mse = (predictions - targets) ** 2
        weighted_mse = weights * mse
        
        return weighted_mse.mean()


def get_loss_function(loss_type="mse", **kwargs):
    """Factory function to get loss function by name."""
    
    if loss_type.lower() == "mse":
        return MSELoss()
    elif loss_type.lower() == "smooth_l1":
        return SmoothL1Loss(**kwargs)
    elif loss_type.lower() == "weighted_mse":
        return WeightedMSELoss(**kwargs)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")