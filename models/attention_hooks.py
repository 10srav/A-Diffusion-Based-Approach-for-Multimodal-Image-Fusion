"""
Self-Attention Hooks for U-Net
Extracts and injects K, V features from self-attention layers
"""
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Callable
from collections import defaultdict


class AttentionHooks:
    """
    Manages attention hooks for extracting and injecting K, V features
    from Stable Diffusion U-Net self-attention layers.
    """
    
    def __init__(self, unet: nn.Module):
        """
        Initialize attention hooks on U-Net.
        
        Args:
            unet: Stable Diffusion U-Net model
        """
        self.unet = unet
        self.hooks: List = []
        self.feature_memory: Dict[int, Dict[str, torch.Tensor]] = defaultdict(dict)
        self.injection_enabled = False
        self.injection_source = None  # 'vis' or 'inf'
        self.current_step = 0
        
        # Attention layer names to hook
        self.attn_layer_names = []
        
    def register_hooks(self):
        """Register forward hooks on all self-attention layers."""
        self._clear_hooks()
        
        for name, module in self.unet.named_modules():
            if self._is_self_attention(name, module):
                hook = module.register_forward_hook(
                    self._create_hook(name)
                )
                self.hooks.append(hook)
                self.attn_layer_names.append(name)
                
    def _is_self_attention(self, name: str, module: nn.Module) -> bool:
        """Check if module is a self-attention layer."""
        # Match self-attention layers in SD U-Net
        if 'attn1' in name and hasattr(module, 'to_k') and hasattr(module, 'to_v'):
            return True
        return False
    
    def _create_hook(self, layer_name: str) -> Callable:
        """Create a forward hook for the given layer."""
        def hook_fn(module, input, output):
            if self.injection_enabled and self.injection_source is not None:
                # Injection mode: replace K, V with stored features
                return self._inject_features(module, input, output, layer_name)
            else:
                # Extraction mode: store K, V features
                self._extract_features(module, input, layer_name)
                return output
                
        return hook_fn
    
    def _extract_features(self, module: nn.Module, input: tuple, layer_name: str):
        """Extract K, V features from self-attention layer."""
        hidden_states = input[0]
        
        # Get K, V projections
        key = module.to_k(hidden_states)
        value = module.to_v(hidden_states)
        
        # Store in feature memory
        step_key = f"{self.current_step}_{layer_name}"
        self.feature_memory[self.current_step][f"{layer_name}_K"] = key.detach().clone()
        self.feature_memory[self.current_step][f"{layer_name}_V"] = value.detach().clone()
        
    def _inject_features(self, module: nn.Module, input: tuple, output, layer_name: str):
        """Inject stored K, V features into self-attention layer."""
        if self.current_step not in self.feature_memory:
            return output
            
        k_key = f"{layer_name}_K"
        v_key = f"{layer_name}_V"
        
        if k_key not in self.feature_memory[self.current_step]:
            return output
            
        # This is a simplified injection - in practice, we need to
        # modify the attention computation directly
        return output
    
    def store_latent(self, step: int, latent: torch.Tensor, source: str = 'vis'):
        """
        Store latent noise variable z at given step.
        
        Args:
            step: Diffusion timestep
            latent: Noise latent variable
            source: Source type ('vis' or 'inf')
        """
        self.feature_memory[step][f"z_{source}"] = latent.detach().clone()
        
    def get_latent(self, step: int, source: str = 'vis') -> Optional[torch.Tensor]:
        """
        Get stored latent at given step.
        
        Args:
            step: Diffusion timestep
            source: Source type ('vis' or 'inf')
            
        Returns:
            Stored latent or None
        """
        key = f"z_{source}"
        if step in self.feature_memory and key in self.feature_memory[step]:
            return self.feature_memory[step][key]
        return None
    
    def get_kv_features(self, step: int, source: str = 'vis') -> Dict[str, torch.Tensor]:
        """
        Get all K, V features at given step.
        
        Args:
            step: Diffusion timestep
            source: Source type ('vis' or 'inf')
            
        Returns:
            Dictionary of K, V features
        """
        if step not in self.feature_memory:
            return {}
            
        features = {}
        for key, value in self.feature_memory[step].items():
            if key.endswith('_K') or key.endswith('_V'):
                features[key] = value
                
        return features
    
    def enable_injection(self, source: str = 'vis'):
        """Enable injection mode with given source."""
        self.injection_enabled = True
        self.injection_source = source
        
    def disable_injection(self):
        """Disable injection mode."""
        self.injection_enabled = False
        self.injection_source = None
        
    def set_step(self, step: int):
        """Set current diffusion step."""
        self.current_step = step
        
    def clear_memory(self):
        """Clear all stored features."""
        self.feature_memory.clear()
        
    def _clear_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        self.attn_layer_names.clear()
        
    def __del__(self):
        """Cleanup hooks on deletion."""
        self._clear_hooks()


class AttentionProcessor:
    """
    Custom attention processor for injecting K, V features during attention computation.
    This replaces the default attention processor in the U-Net.
    """
    
    def __init__(self, original_processor):
        """
        Initialize with original processor.
        
        Args:
            original_processor: Original attention processor
        """
        self.original_processor = original_processor
        self.inject_k = None
        self.inject_v = None
        self.injection_enabled = False
        
    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        **kwargs
    ):
        """
        Custom attention forward pass with optional K, V injection.
        """
        # Get batch info
        batch_size, sequence_length, _ = hidden_states.shape
        
        # Compute query
        query = attn.to_q(hidden_states)
        
        # Compute or inject key and value
        if self.injection_enabled and self.inject_k is not None and self.inject_v is not None:
            key = self.inject_k
            value = self.inject_v
        else:
            if encoder_hidden_states is None:
                encoder_hidden_states = hidden_states
            key = attn.to_k(encoder_hidden_states)
            value = attn.to_v(encoder_hidden_states)
            
        # Reshape for multi-head attention
        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads
        
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        
        # Compute attention
        hidden_states = torch.nn.functional.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )
        
        # Reshape output
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        
        # Output projection
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        
        return hidden_states
    
    def set_injection(self, key: torch.Tensor, value: torch.Tensor):
        """Set K, V tensors for injection."""
        self.inject_k = key
        self.inject_v = value
        self.injection_enabled = True
        
    def clear_injection(self):
        """Clear injection state."""
        self.inject_k = None
        self.inject_v = None
        self.injection_enabled = False
