"""
Contrastive Encoder - loads a pre-trained SparK encoder from checkpoint

This module provides functionality to load a contrastive encoder that was
trained using the SparK pre-training framework with contrastive loss.
"""

import torch
import torch.nn as nn
from src.models.modules.spark.Spark_2D import SparK_2D
from src.models.modules.spark.models import build_encoder


class ContrastiveEncoder(nn.Module):
    """
    Wrapper for the SparK encoder that extracts features from images.
    Used as a conditioning encoder for the conditional DDPM.
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        sbn = False
        backbone = cfg.get('version', 'resnet50')
        
        # Build the encoder with the same architecture as used during pre-training
        self.encoder = build_encoder(
            backbone,
            cfg.get('cond_dim', 128),
            input_size=int(cfg.imageDim[1] / cfg.rescaleFactor),
            sbn=sbn,
            drop_path_rate=cfg.get('dp', 0),
            verbose=False
        )
    
    def forward(self, x):
        """
        Extract features from input images.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            features: Tensor of shape (B, cond_dim)
        """
        features = self.encoder(x)
        return features


def LoadContrastiveModel(cfg, fold=None):
    """
    Load a pre-trained contrastive encoder from checkpoint.
    
    This function loads the encoder weights from a SparK model checkpoint
    that was trained with the contrastive loss.
    
    Args:
        cfg: Configuration object containing:
            - encoder_path: Path to the checkpoint file
            - cond_dim: Output dimension of the encoder
            - imageDim: Image dimensions
            - rescaleFactor: Rescale factor for input
            - version: Backbone architecture (e.g., 'resnet50')
        fold: Optional fold number (for cross-validation)
        
    Returns:
        encoder: ContrastiveEncoder with loaded weights
    """
    encoder_path = cfg.get('encoder_path', None)
    
    if encoder_path is None:
        raise ValueError("encoder_path must be specified in config to load contrastive encoder")
    
    print(f"[ContrastiveEncoder] Loading pre-trained encoder from: {encoder_path}")
    
    # Create the encoder with the same architecture
    encoder = ContrastiveEncoder(cfg)
    
    # Load the checkpoint
    checkpoint = torch.load(encoder_path, map_location='cpu')
    
    # The checkpoint contains the full Spark_2D Lightning module state
    # We need to extract just the encoder weights
    state_dict = checkpoint.get('state_dict', checkpoint)
    
    # Filter and rename keys to match our encoder
    # The Spark_2D model has: model.sparse_encoder.* 
    # We need to map to: encoder.*
    encoder_state_dict = {}
    
    for key, value in state_dict.items():
        # Look for sparse_encoder weights in the checkpoint
        if 'model.sparse_encoder' in key:
            # Remove 'model.sparse_encoder.' prefix and add 'encoder.' prefix
            new_key = key.replace('model.sparse_encoder.', 'encoder.')
            encoder_state_dict[new_key] = value
        elif 'sparse_encoder' in key:
            # Handle case without 'model.' prefix
            new_key = key.replace('sparse_encoder.', 'encoder.')
            encoder_state_dict[new_key] = value
    
    if len(encoder_state_dict) == 0:
        # Try alternative: maybe the encoder is stored directly
        for key, value in state_dict.items():
            if key.startswith('encoder.'):
                encoder_state_dict[key] = value
            elif key.startswith('model.encoder.'):
                new_key = key.replace('model.', '')
                encoder_state_dict[new_key] = value
    
    if len(encoder_state_dict) == 0:
        print(f"[ContrastiveEncoder] Warning: Could not find encoder weights in checkpoint.")
        print(f"[ContrastiveEncoder] Available keys (first 20): {list(state_dict.keys())[:20]}")
        print(f"[ContrastiveEncoder] Using randomly initialized encoder.")
    else:
        # Load the weights
        missing, unexpected = encoder.load_state_dict(encoder_state_dict, strict=False)
        print(f"[ContrastiveEncoder] Loaded {len(encoder_state_dict)} weight tensors")
        if missing:
            print(f"[ContrastiveEncoder] Missing keys: {missing[:10]}{'...' if len(missing) > 10 else ''}")
        if unexpected:
            print(f"[ContrastiveEncoder] Unexpected keys: {unexpected[:10]}{'...' if len(unexpected) > 10 else ''}")
    
    # Freeze the encoder if specified
    if cfg.get('freeze_encoder', True):
        print("[ContrastiveEncoder] Freezing encoder weights")
        for param in encoder.parameters():
            param.requires_grad = False
    
    return encoder
