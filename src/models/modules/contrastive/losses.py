"""
Contrastive losses for self-supervised learning.

EpsInfoNCE: InfoNCE loss with epsilon modification for stability.
Based on standard InfoNCE/NT-Xent loss used in SimCLR and similar methods.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class EpsInfoNCE(nn.Module):
    """
    InfoNCE loss with epsilon modification for numerical stability.
    
    This loss is used for contrastive learning, where we want to maximize
    agreement between positive pairs and minimize agreement with negative pairs.
    
    Args:
        temperature: Temperature scaling parameter (default: 0.07)
        epsilon: Small value added to denominators for numerical stability (default: 0.5)
    """
    
    def __init__(self, temperature: float = 0.07, epsilon: float = 0.5):
        super().__init__()
        self.temperature = temperature
        self.epsilon = epsilon
    
    def forward(self, features: torch.Tensor, labels: torch.Tensor = None) -> torch.Tensor:
        """
        Compute InfoNCE loss.
        
        Args:
            features: Tensor of shape (batch_size, n_views, feature_dim)
                      where n_views=2 typically (original and augmented)
            labels: Optional labels for supervised contrastive learning.
                    If None, treats each sample as its own class (self-supervised)
        
        Returns:
            Scalar loss value
        """
        device = features.device
        batch_size = features.shape[0]
        n_views = features.shape[1]  # typically 2 (original + augmented)
        
        # Reshape features: (batch_size * n_views, feature_dim)
        features = features.view(batch_size * n_views, -1)
        
        # Normalize features
        features = F.normalize(features, dim=1)
        
        # Compute similarity matrix
        # (batch_size * n_views, batch_size * n_views)
        similarity_matrix = torch.matmul(features, features.T)
        
        # Create mask for positive pairs
        # For self-supervised: positives are different views of same sample
        mask_self = torch.eye(batch_size * n_views, device=device, dtype=torch.bool)
        
        if labels is None:
            # Self-supervised: positives are (i, i+batch_size) and (i+batch_size, i)
            labels = torch.cat([torch.arange(batch_size) for _ in range(n_views)], dim=0)
            labels = labels.to(device)
        
        labels = labels.contiguous().view(-1, 1)
        mask_positives = torch.eq(labels, labels.T).float().to(device)
        
        # Remove self from positive mask
        mask_positives = mask_positives * (~mask_self).float()
        
        # Compute logits
        logits = similarity_matrix / self.temperature
        
        # For numerical stability, subtract max
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()
        
        # Compute exp(logits)
        exp_logits = torch.exp(logits)
        
        # Mask out self-similarity
        exp_logits = exp_logits * (~mask_self).float()
        
        # Compute log-sum-exp for denominator (all negatives + positives)
        # Add epsilon for stability
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + self.epsilon)
        
        # Compute mean log-likelihood over positive pairs
        # Only consider pairs where we have positives
        mask_positives_sum = mask_positives.sum(dim=1)
        mean_log_prob_pos = (mask_positives * log_prob).sum(dim=1) / (mask_positives_sum + 1e-8)
        
        # Only consider samples with at least one positive
        valid_samples = mask_positives_sum > 0
        if valid_samples.sum() == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        loss = -mean_log_prob_pos[valid_samples].mean()
        
        return loss


class SupConLoss(nn.Module):
    """
    Supervised Contrastive Learning loss.
    
    Similar to InfoNCE but uses labels to define positive pairs.
    
    Args:
        temperature: Temperature scaling parameter (default: 0.07)
        base_temperature: Base temperature for normalization (default: 0.07)
    """
    
    def __init__(self, temperature: float = 0.07, base_temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature
    
    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Compute supervised contrastive loss.
        
        Args:
            features: Tensor of shape (batch_size, n_views, feature_dim)
            labels: Tensor of shape (batch_size,) with class labels
        
        Returns:
            Scalar loss value
        """
        device = features.device
        batch_size = features.shape[0]
        n_views = features.shape[1]
        
        # Reshape and normalize
        features = features.view(batch_size * n_views, -1)
        features = F.normalize(features, dim=1)
        
        # Extend labels for all views
        labels = labels.contiguous().view(-1, 1)
        labels = labels.repeat(n_views, 1).view(-1, 1)
        
        # Create masks
        mask_self = torch.eye(batch_size * n_views, device=device, dtype=torch.bool)
        mask_positives = torch.eq(labels, labels.T).float().to(device)
        mask_positives = mask_positives * (~mask_self).float()
        
        # Compute similarity
        similarity_matrix = torch.matmul(features, features.T) / self.temperature
        
        # For numerical stability
        logits_max, _ = torch.max(similarity_matrix, dim=1, keepdim=True)
        logits = similarity_matrix - logits_max.detach()
        
        # Compute log_prob
        exp_logits = torch.exp(logits) * (~mask_self).float()
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True))
        
        # Compute mean over positives
        mask_positives_sum = mask_positives.sum(dim=1)
        mean_log_prob_pos = (mask_positives * log_prob).sum(dim=1) / (mask_positives_sum + 1e-8)
        
        # Only valid samples
        valid_samples = mask_positives_sum > 0
        if valid_samples.sum() == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        # Scale by temperature ratio
        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos[valid_samples].mean()
        
        return loss
