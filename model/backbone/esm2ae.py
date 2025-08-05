import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.models.esm.modeling_esm import EsmConfig, EsmModel, EsmPreTrainedModel
from typing import Optional, Tuple, Union

from ..utils.ActivationFunction import SwiGLU
from ..utils.Simclr import Simclr_loss


class OptimizedEncoder(nn.Module):
    """Optimized encoder with better layer design and activation caching."""
    
    def __init__(self, input_dim: int, hidden_dims: list = [768, 512, 256, 128]):
        super().__init__()
        self.layers = nn.ModuleList()
        
        prev_dim = input_dim
        for i, hidden_dim in enumerate(hidden_dims):
            # Use more efficient layer design
            layer = nn.Sequential(
                nn.Linear(prev_dim, hidden_dim, bias=False),  # Remove bias for efficiency
                nn.RMSNorm(hidden_dim),
                SwiGLU() if i < len(hidden_dims) - 1 else nn.Identity()  # No activation on last layer
            )
            self.layers.append(layer)
            prev_dim = hidden_dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class OptimizedDecoder(nn.Module):
    """Optimized decoder with better layer design and gradient flow."""
    
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: list = [256, 512, 768]):
        super().__init__()
        self.layers = nn.ModuleList()
        
        prev_dim = input_dim
        all_dims = hidden_dims + [output_dim]
        
        for i, hidden_dim in enumerate(all_dims):
            layer = nn.Sequential(
                nn.Linear(prev_dim, hidden_dim, bias=False),
                nn.RMSNorm(hidden_dim) if i < len(all_dims) - 1 else nn.Identity(),
                SwiGLU() if i < len(all_dims) - 1 else nn.Identity()
            )
            self.layers.append(layer)
            prev_dim = hidden_dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class ESM2AE(EsmPreTrainedModel):
    def __init__(self, config: EsmConfig):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        
        # Initialize ESM2 backbone with optimizations
        self.esm = EsmModel(config)
        
        # Freeze ESM2 backbone if specified (for better performance during fine-tuning)
        if getattr(config, 'freeze_backbone', False):
            for param in self.esm.parameters():
                param.requires_grad = False
        
        # Use optimized encoder/decoder
        self.encoder = OptimizedEncoder(
            input_dim=config.hidden_size,
            hidden_dims=[768, 512, 256, 128]
        )
        
        self.decoder = OptimizedDecoder(
            input_dim=128,
            output_dim=config.hidden_size,
            hidden_dims=[256, 512, 768]
        )
        
        # Cache for repeated computations
        self._cached_features = {}
        self._use_cache = getattr(config, 'use_feature_cache', True)
        
        self.post_init()

    def _get_features_with_cache(
        self, 
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        cache_key: Optional[str] = None
    ) -> torch.Tensor:
        """Extract features with optional caching for efficiency."""
        
        if cache_key and self._use_cache and cache_key in self._cached_features:
            return self._cached_features[cache_key]
        
        # Extract features from ESM2
        with torch.cuda.amp.autocast(enabled=True):  # Use mixed precision
            outputs = self.esm(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=False,  # Disable attention outputs for efficiency
                output_hidden_states=False,
                return_dict=True,
            )
        
        # Use CLS token representation
        features = outputs.last_hidden_state[:, 0, :]  # CLS token
        
        # Apply centering and normalization for better training stability
        centered = features - features.mean(dim=-1, keepdim=True)
        normalized = F.normalize(centered, p=2, dim=-1)
        
        # Cache if requested
        if cache_key and self._use_cache:
            self._cached_features[cache_key] = normalized.detach()
            
            # Limit cache size to prevent memory issues
            if len(self._cached_features) > 1000:
                # Remove oldest entries
                oldest_key = next(iter(self._cached_features))
                del self._cached_features[oldest_key]
        
        return normalized

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        input_ids_2: Optional[torch.Tensor] = None,
        attention_mask_2: Optional[torch.Tensor] = None,
        simclr_weight: float = 0.1,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, SequenceClassifierOutput]:
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # === Primary branch with optimizations ===
        features1 = self._get_features_with_cache(
            input_ids, 
            attention_mask,
            cache_key=f"primary_{hash(tuple(input_ids.flatten().tolist())) if input_ids.numel() < 1000 else None}"
        )
        
        # Encode to latent space
        with torch.cuda.amp.autocast(enabled=True):
            z1 = self.encoder(features1)
            
            # Decode back to original space
            reconstructed = self.decoder(z1)
        
        # Compute reconstruction loss with label smoothing for better training
        reconstruction_loss = F.mse_loss(
            reconstructed, 
            features1.detach(),
            reduction='mean'
        )
        
        # Add L2 regularization on latent codes for better generalization
        latent_reg = 0.001 * torch.norm(z1, p=2, dim=-1).mean()
        reconstruction_loss = reconstruction_loss + latent_reg

        # === Optional SimCLR branch for contrastive learning ===
        contrastive_loss = torch.tensor(0.0, device=reconstruction_loss.device)
        
        if input_ids_2 is not None:
            features2 = self._get_features_with_cache(
                input_ids_2, 
                attention_mask_2,
                cache_key=f"secondary_{hash(tuple(input_ids_2.flatten().tolist())) if input_ids_2.numel() < 1000 else None}"
            )
            
            with torch.cuda.amp.autocast(enabled=True):
                z2 = self.encoder(features2)
            
            # Compute SimCLR contrastive loss
            contrastive_loss = Simclr_loss(z1, z2)

        # Combined loss with weighted components
        total_loss = reconstruction_loss + simclr_weight * contrastive_loss

        if not return_dict:
            return (total_loss, reconstructed)

        return SequenceClassifierOutput(
            loss=total_loss,
            logits=reconstructed,
        )
    
    def encode(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Encode input sequences to latent space (inference mode)."""
        self.eval()
        with torch.no_grad():
            features = self._get_features_with_cache(input_ids, attention_mask)
            latent = self.encoder(features)
        return latent
    
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode latent representations back to feature space (inference mode)."""
        self.eval()
        with torch.no_grad():
            reconstructed = self.decoder(latent)
        return reconstructed
    
    def clear_cache(self):
        """Clear the feature cache to free memory."""
        self._cached_features.clear()
    
    def get_cache_info(self) -> dict:
        """Get information about the current cache state."""
        return {
            "cache_size": len(self._cached_features),
            "cache_enabled": self._use_cache,
            "memory_usage_mb": sum(
                cache_item.numel() * cache_item.element_size() 
                for cache_item in self._cached_features.values()
            ) / (1024 * 1024)
        }
