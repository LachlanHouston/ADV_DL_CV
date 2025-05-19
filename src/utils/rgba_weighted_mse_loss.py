import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb

class RGBAWeightedMSELoss(nn.Module):
    """
    Custom MSE loss for RGBA images that weights pixels that change between input and target higher.
    Also allows for separate weighting of RGB and alpha channels.
    """
    def __init__(self, threshold=0.05, alpha=10.0, alpha_channel_weight=2.0):
        """
        Initialize the weighted MSE loss.
        
        Args:
            threshold (float): Threshold for considering a pixel changed between input and target.
                              Values above this threshold are considered changed. Default: 0.05
            alpha (float): Weight multiplier for changed pixels. Default: 10.0
            alpha_channel_weight (float): Additional weight multiplier for the alpha channel.
                                         Default: 2.0 (makes alpha twice as important)
        """
        super().__init__()
        self.threshold = threshold
        self.alpha = alpha
        self.alpha_channel_weight = alpha_channel_weight
        
    def forward(self, inputs, outputs, targets):
        """
        Calculate weighted MSE loss for RGBA images.
        
        Args:
            inputs (torch.Tensor): Input images [batch_size, 4, height, width]
            outputs (torch.Tensor): Predicted outputs [batch_size, 4, height, width]
            targets (torch.Tensor): Target images [batch_size, 4, height, width]
            
        Returns:
            torch.Tensor: Weighted MSE loss value
        """
        # Split RGB and Alpha channels
        inputs_rgb, inputs_alpha = inputs[:, :3], inputs[:, 3:4]
        outputs_rgb, outputs_alpha = outputs[:, :3], outputs[:, 3:4]
        targets_rgb, targets_alpha = targets[:, :3], targets[:, 3:4]
        
        # Create masks of changed pixels for RGB and alpha separately
        rgb_diff_mask = (torch.abs(targets_rgb - inputs_rgb) > self.threshold).float().mean(dim=1, keepdim=True)
        alpha_diff_mask = (torch.abs(targets_alpha - inputs_alpha) > self.threshold).float()
        
        # Calculate pixel-wise squared error for RGB and alpha
        rgb_losses = (outputs_rgb - targets_rgb) ** 2
        alpha_losses = (outputs_alpha - targets_alpha) ** 2
        
        # Apply higher weights to changed pixels
        weighted_rgb_losses = rgb_losses.mean(dim=1, keepdim=True) * (1 + (self.alpha - 1) * rgb_diff_mask)
        weighted_alpha_losses = alpha_losses * (1 + (self.alpha - 1) * alpha_diff_mask) * self.alpha_channel_weight
        
        # Combine RGB and alpha losses
        combined_losses = torch.cat([weighted_rgb_losses.expand(-1, 3, -1, -1), weighted_alpha_losses], dim=1)
        
        # Calculate mean loss
        loss = combined_losses.mean()

        return loss
    
    def get_visualization(self, inputs, targets):
        """
        Visualize the weight map for a batch of inputs and targets.
        
        Args:
            inputs (torch.Tensor): Input images [batch_size, 4, height, width]
            targets (torch.Tensor): Target images [batch_size, 4, height, width]
            
        Returns:
            tuple: (rgb_weight_map, alpha_weight_map) weight maps
        """
        # Split RGB and Alpha channels
        inputs_rgb, inputs_alpha = inputs[:, :3], inputs[:, 3:4]
        targets_rgb, targets_alpha = targets[:, :3], targets[:, 3:4]
        
        # Create masks of changed pixels for RGB and alpha separately
        rgb_diff_mask = (torch.abs(targets_rgb - inputs_rgb) > self.threshold).float().mean(dim=1, keepdim=True)
        alpha_diff_mask = (torch.abs(targets_alpha - inputs_alpha) > self.threshold).float()
        
        # Create weight maps
        rgb_weight_map = 1 + (self.alpha - 1) * rgb_diff_mask
        alpha_weight_map = (1 + (self.alpha - 1) * alpha_diff_mask) * self.alpha_channel_weight
        
        return rgb_weight_map, alpha_weight_map
    
    def calculate_change_stats(self, inputs, targets):
        """
        Calculate statistics about changed pixels for RGBA images.
        
        Args:
            inputs (torch.Tensor): Input images [batch_size, 4, height, width]
            targets (torch.Tensor): Target images [batch_size, 4, height, width]
            
        Returns:
            dict: Dictionary with statistics
                - 'pct_changed_rgb': Percentage of changed RGB pixels
                - 'pct_changed_alpha': Percentage of changed alpha pixels
                - 'mean_change_rgb': Mean absolute RGB change value
                - 'mean_change_alpha': Mean absolute alpha change value
        """
        # Split RGB and Alpha channels
        inputs_rgb, inputs_alpha = inputs[:, :3], inputs[:, 3:4]
        targets_rgb, targets_alpha = targets[:, :3], targets[:, 3:4]
        
        # Calculate absolute differences
        abs_diff_rgb = torch.abs(targets_rgb - inputs_rgb)
        abs_diff_alpha = torch.abs(targets_alpha - inputs_alpha)
        
        # Create masks of changed pixels
        rgb_diff_mask = (abs_diff_rgb > self.threshold).float()
        alpha_diff_mask = (abs_diff_alpha > self.threshold).float()
        
        # Calculate percentage of changed pixels
        pct_changed_rgb = rgb_diff_mask.mean().item() * 100.0
        pct_changed_alpha = alpha_diff_mask.mean().item() * 100.0
        
        # Calculate mean change among changed pixels
        # Add small epsilon to avoid division by zero
        mean_change_rgb = (abs_diff_rgb * rgb_diff_mask).sum() / (rgb_diff_mask.sum() + 1e-8)
        mean_change_alpha = (abs_diff_alpha * alpha_diff_mask).sum() / (alpha_diff_mask.sum() + 1e-8)
        
        return {
            'pct_changed_rgb': pct_changed_rgb,
            'pct_changed_alpha': pct_changed_alpha,
            'mean_change_rgb': mean_change_rgb.item(),
            'mean_change_alpha': mean_change_alpha.item()
        }


class RGBAWeightedProportionalMSELoss(nn.Module):
    """
    Custom MSE loss for RGBA images that weights pixels proportionally to how rare changes are.
    Treats RGB and alpha channels separately.
    """
    def __init__(self, threshold=0.05, alpha_channel_weight=2.0):
        """
        Initialize the weighted proportional MSE loss.
        
        Args:
            threshold (float): Threshold for considering a pixel changed between input and target.
                              Values above this threshold are considered changed. Default: 0.05
            alpha_channel_weight (float): Additional weight multiplier for the alpha channel.
                                         Default: 2.0 (makes alpha twice as important)
        """
        super().__init__()
        self.threshold = threshold
        self.alpha_channel_weight = alpha_channel_weight
        
    def forward(self, inputs, outputs, targets):
        """
        Calculate weighted proportional MSE loss for RGBA images.
        
        Args:
            inputs (torch.Tensor): Input images [batch_size, 4, height, width]
            outputs (torch.Tensor): Predicted outputs [batch_size, 4, height, width]
            targets (torch.Tensor): Target images [batch_size, 4, height, width]
            
        Returns:
            torch.Tensor: Weighted MSE loss value
        """
        # Get dimensions
        batch_size, channels, height, width = inputs.shape
        
        # Split RGB and Alpha channels
        inputs_rgb, inputs_alpha = inputs[:, :3], inputs[:, 3:4]
        outputs_rgb, outputs_alpha = outputs[:, :3], outputs[:, 3:4]
        targets_rgb, targets_alpha = targets[:, :3], targets[:, 3:4]
        
        # Create masks of changed pixels for RGB and alpha separately
        rgb_diff_mask = (torch.abs(targets_rgb - inputs_rgb) > self.threshold).float()
        alpha_diff_mask = (torch.abs(targets_alpha - inputs_alpha) > self.threshold).float()
        
        # Calculate weights proportional to rarity of changes
        # For RGB (average across RGB channels)
        rgb_change_sum = rgb_diff_mask.sum(dim=(1, 2, 3), keepdim=True)
        rgb_weights = (3 * height * width) / (rgb_change_sum + 1e-8)  # +1e-8 to avoid division by zero
        
        # For alpha
        alpha_change_sum = alpha_diff_mask.sum(dim=(1, 2, 3), keepdim=True)
        alpha_weights = (height * width) / (alpha_change_sum + 1e-8)
        
        # Calculate pixel-wise squared error
        rgb_losses = (outputs_rgb - targets_rgb) ** 2
        alpha_losses = (outputs_alpha - targets_alpha) ** 2
        
        # Apply proportional weights to changed pixels
        weighted_rgb_losses = rgb_losses * (1 + (rgb_weights - 1) * rgb_diff_mask)
        weighted_alpha_losses = alpha_losses * (1 + (alpha_weights - 1) * alpha_diff_mask) * self.alpha_channel_weight
        
        # Combine all losses
        combined_losses = torch.cat([weighted_rgb_losses, weighted_alpha_losses], dim=1)
        
        # Calculate mean loss
        loss = combined_losses.mean()

        return loss
    
    def get_visualization(self, inputs, targets):
        """
        Visualize the weight map for a batch of inputs and targets.
        
        Args:
            inputs (torch.Tensor): Input images [batch_size, 4, height, width]
            targets (torch.Tensor): Target images [batch_size, 4, height, width]
            
        Returns:
            tuple: (rgb_weight_map, alpha_weight_map) weight maps
        """
        # Get dimensions
        batch_size, channels, height, width = inputs.shape
        
        # Split RGB and Alpha channels
        inputs_rgb, inputs_alpha = inputs[:, :3], inputs[:, 3:4]
        targets_rgb, targets_alpha = targets[:, :3], targets[:, 3:4]
        
        # Create masks of changed pixels for RGB and alpha separately
        rgb_diff_mask = (torch.abs(targets_rgb - inputs_rgb) > self.threshold).float()
        alpha_diff_mask = (torch.abs(targets_alpha - inputs_alpha) > self.threshold).float()
        
        # Calculate weights proportional to rarity of changes
        # For RGB (average across RGB channels)
        rgb_change_sum = rgb_diff_mask.sum(dim=(1, 2, 3), keepdim=True)
        rgb_weights = (3 * height * width) / (rgb_change_sum + 1e-8)
        
        # For alpha
        alpha_change_sum = alpha_diff_mask.sum(dim=(1, 2, 3), keepdim=True)
        alpha_weights = (height * width) / (alpha_change_sum + 1e-8)
        
        # Create weight maps
        rgb_weight_map = 1 + (rgb_weights - 1) * rgb_diff_mask
        alpha_weight_map = (1 + (alpha_weights - 1) * alpha_diff_mask) * self.alpha_channel_weight
        
        return rgb_weight_map, alpha_weight_map
    
    def calculate_change_stats(self, inputs, targets):
        """
        Calculate statistics about changed pixels for RGBA images.
        
        Args:
            inputs (torch.Tensor): Input images [batch_size, 4, height, width]
            targets (torch.Tensor): Target images [batch_size, 4, height, width]
            
        Returns:
            dict: Dictionary with statistics
                - 'pct_changed_rgb': Percentage of changed RGB pixels
                - 'pct_changed_alpha': Percentage of changed alpha pixels
                - 'mean_change_rgb': Mean absolute RGB change value
                - 'mean_change_alpha': Mean absolute alpha change value
        """
        # Split RGB and Alpha channels
        inputs_rgb, inputs_alpha = inputs[:, :3], inputs[:, 3:4]
        targets_rgb, targets_alpha = targets[:, :3], targets[:, 3:4]
        
        # Calculate absolute differences
        abs_diff_rgb = torch.abs(targets_rgb - inputs_rgb)
        abs_diff_alpha = torch.abs(targets_alpha - inputs_alpha)
        
        # Create masks of changed pixels
        rgb_diff_mask = (abs_diff_rgb > self.threshold).float()
        alpha_diff_mask = (abs_diff_alpha > self.threshold).float()
        
        # Calculate percentage of changed pixels
        pct_changed_rgb = rgb_diff_mask.mean().item() * 100.0
        pct_changed_alpha = alpha_diff_mask.mean().item() * 100.0
        
        # Calculate mean change among changed pixels
        # Add small epsilon to avoid division by zero
        mean_change_rgb = (abs_diff_rgb * rgb_diff_mask).sum() / (rgb_diff_mask.sum() + 1e-8)
        mean_change_alpha = (abs_diff_alpha * alpha_diff_mask).sum() / (alpha_diff_mask.sum() + 1e-8)
        
        return {
            'pct_changed_rgb': pct_changed_rgb,
            'pct_changed_alpha': pct_changed_alpha,
            'mean_change_rgb': mean_change_rgb.item(),
            'mean_change_alpha': mean_change_alpha.item()
        }