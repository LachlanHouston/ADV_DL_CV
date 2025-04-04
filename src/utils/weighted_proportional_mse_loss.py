import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb

class WeightedProportionalMSELoss(nn.Module):
    """
    Custom MSE loss that weights pixels that change between input and target higher.
    Useful for autoregressive models where most of the image stays the same and only
    a small portion changes.
    """
    def __init__(self, threshold=0.05):
        """
        Initialize the weighted MSE loss.
        
        Args:
            threshold (float): Threshold for considering a pixel changed between input and target.
                              Values above this threshold are considered changed. Default: 0.05
            alpha (float): Weight multiplier for changed pixels. Default: 10.0
        """
        super().__init__()
        self.threshold = threshold
        
    def forward(self, inputs, outputs, targets):
        """
        Calculate weighted MSE loss.
        
        Args:
            inputs (torch.Tensor): Input images [batch_size, channels, height, width]
            outputs (torch.Tensor): Predicted outputs [batch_size, channels, height, width]
            targets (torch.Tensor): Target images [batch_size, channels, height, width]
            
        Returns:
            torch.Tensor: Weighted MSE loss value
        """
        height = inputs.shape[-2]
        width = inputs.shape[-1]
        diff_mask = (torch.abs(targets - inputs) > self.threshold).float()

        weights = torch.sum(diff_mask, (1,2,3))
        weights = (height*width)/weights
        weights = weights.view(-1, 1, 1, 1)

        pixel_losses = (outputs - targets) ** 2

        weighted_losses = pixel_losses * (1 + (weights - 1) * diff_mask)

        loss = weighted_losses.mean()

        return loss
    
    def get_visualization(self, inputs, targets):
        """
        Visualize the weight map for a batch of inputs and targets.
        
        Args:
            inputs (torch.Tensor): Input images [batch_size, channels, height, width]
            targets (torch.Tensor): Target images [batch_size, channels, height, width]
            
        Returns:
            torch.Tensor: Weight map [batch_size, channels, height, width]
        """
        height = inputs.shape[-2]
        width = inputs.shape[-1]
        diff_mask = (torch.abs(targets - inputs) > self.threshold).float()

        weights = torch.sum(diff_mask, (1,2,3))
        weights = (height*width)/weights
        weights.view(-1, 1, 1, 1)
        
        # Create weight map
        weight_map = (1 + (weights - 1) * diff_mask)
        
        return weight_map
    
    def calculate_change_stats(self, inputs, targets):
        """
        Calculate statistics about changed pixels.
        
        Args:
            inputs (torch.Tensor): Input images [batch_size, channels, height, width]
            targets (torch.Tensor): Target images [batch_size, channels, height, width]
            
        Returns:
            dict: Dictionary with statistics
                - 'pct_changed': Percentage of changed pixels
                - 'mean_change': Mean absolute change value
        """
        # Calculate absolute differences
        abs_diff = torch.abs(targets - inputs)
        
        # Create a mask of changed pixels
        diff_mask = (abs_diff > self.threshold).float()
        
        # Calculate percentage of changed pixels
        pct_changed = diff_mask.mean().item() * 100.0
        
        # Calculate mean change among changed pixels
        # Add small epsilon to avoid division by zero
        mean_change = (abs_diff * diff_mask).sum() / (diff_mask.sum() + 1e-8)
        
        return {
            'pct_changed': pct_changed,
            'mean_change': mean_change.item()
        }