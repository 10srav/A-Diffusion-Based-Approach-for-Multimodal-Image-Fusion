"""
Dual-Modality Feature Memory Bank for Adaptive Diffusion
Stores and retrieves K,V features for BOTH IR and VIS modalities
across timesteps during DDIM inversion and reverse sampling.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .adaptive_unet import _num_groups


class FeatureMemoryBank(nn.Module):
    """
    Single-modality memory bank with K,V projections.
    Used internally by DualModalityFeatureMemory.
    """

    def __init__(self, unet_channels=(32, 64, 128, 256),
                 memory_dim=64, max_stored_steps=5):
        super().__init__()
        self.memory_dim = memory_dim
        self.max_stored_steps = max_stored_steps

        self.scale_configs = {
            'scale2': unet_channels[2],
            'scale3': unet_channels[3],
            'bottleneck': unet_channels[3],
        }

        self.k_projs = nn.ModuleDict()
        self.v_projs = nn.ModuleDict()
        self.q_projs = nn.ModuleDict()
        self.out_projs = nn.ModuleDict()

        for name, channels in self.scale_configs.items():
            self.k_projs[name] = nn.Conv2d(channels, memory_dim, 1)
            self.v_projs[name] = nn.Conv2d(channels, memory_dim, 1)
            self.q_projs[name] = nn.Conv2d(channels, memory_dim, 1)
            self.out_projs[name] = nn.Sequential(
                nn.Conv2d(memory_dim, channels, 1),
                nn.GroupNorm(_num_groups(channels), channels),
            )

        self._memory = {}
        self._stored_order = []

    def clear(self):
        self._memory = {}
        self._stored_order = []

    def store(self, t, features):
        stored = {}
        skips = features['encoder_skips']

        for scale_name, feat in [('scale2', skips[2]),
                                   ('scale3', skips[3]),
                                   ('bottleneck', features['bottleneck'])]:
            k = self.k_projs[scale_name](feat).detach()
            v = self.v_projs[scale_name](feat).detach()
            stored[scale_name] = (k, v)

        self._memory[t] = stored

        if t in self._stored_order:
            self._stored_order.remove(t)
        self._stored_order.append(t)

        while len(self._stored_order) > self.max_stored_steps:
            oldest_t = self._stored_order.pop(0)
            del self._memory[oldest_t]

    def retrieve(self, query_features, scale_name, current_t):
        if not self._memory or scale_name not in self.scale_configs:
            return torch.zeros_like(query_features)

        stored_ts = sorted(self._memory.keys())
        closest_t = min(stored_ts, key=lambda st: abs(st - current_t))

        if scale_name not in self._memory[closest_t]:
            return torch.zeros_like(query_features)

        k_mem, v_mem = self._memory[closest_t][scale_name]

        if k_mem.device != query_features.device:
            k_mem = k_mem.to(query_features.device)
            v_mem = v_mem.to(query_features.device)

        B, C, H, W = query_features.shape
        q = self.q_projs[scale_name](query_features)

        if k_mem.shape[2:] != q.shape[2:]:
            k_mem = F.interpolate(k_mem, size=q.shape[2:], mode='nearest')
            v_mem = F.interpolate(v_mem, size=q.shape[2:], mode='nearest')

        q_flat = q.flatten(2).permute(0, 2, 1)
        k_flat = k_mem.flatten(2).permute(0, 2, 1)
        v_flat = v_mem.flatten(2).permute(0, 2, 1)

        attn_out = F.scaled_dot_product_attention(q_flat, k_flat, v_flat)
        attn_out = attn_out.permute(0, 2, 1).view(B, self.memory_dim, H, W)
        return self.out_projs[scale_name](attn_out)

    @property
    def num_stored(self):
        return len(self._memory)

    @property
    def stored_timesteps(self):
        return sorted(self._memory.keys())


class DualModalityFeatureMemory(nn.Module):
    """
    Stores K,V features for BOTH IR and VIS modalities separately.

    Architecture matching the client diagram:
        Feature Memory (z_t, K, V for both IR & VIS)

    During dual inversion:
      - VIS features stored under 'vis' key
      - IR features stored under 'ir' key

    During reverse diffusion, the UNet retrieves from BOTH
    modalities via cross-attention for rich fusion context.

    ~0.6M params (2x FeatureMemoryBank)
    """

    def __init__(self, unet_channels=(32, 64, 128, 256),
                 memory_dim=64, max_stored_steps=5):
        super().__init__()

        # Separate memory banks for each modality
        self.vis_memory = FeatureMemoryBank(
            unet_channels, memory_dim, max_stored_steps
        )
        self.ir_memory = FeatureMemoryBank(
            unet_channels, memory_dim, max_stored_steps
        )

        # Fusion gate: blends IR and VIS retrieved features
        self.scale_configs = {
            'scale2': unet_channels[2],
            'scale3': unet_channels[3],
            'bottleneck': unet_channels[3],
        }
        self.fusion_gates = nn.ModuleDict()
        for name, channels in self.scale_configs.items():
            self.fusion_gates[name] = nn.Sequential(
                nn.Conv2d(channels * 2, channels, 1),
                nn.Sigmoid(),
            )

        # Also store z_t trajectory for both modalities
        self._z_trajectories = {'vis': {}, 'ir': {}}

    def clear(self):
        """Clear all stored features for both modalities."""
        self.vis_memory.clear()
        self.ir_memory.clear()
        self._z_trajectories = {'vis': {}, 'ir': {}}

    def store(self, t, modality, features, z_t=None):
        """
        Store features for a specific modality at timestep t.

        Args:
            t: int, timestep
            modality: 'ir' or 'vis'
            features: dict with 'encoder_skips' and 'bottleneck'
            z_t: optional [B,C,H,W] latent state at this timestep
        """
        if modality == 'vis':
            self.vis_memory.store(t, features)
        elif modality == 'ir':
            self.ir_memory.store(t, features)

        if z_t is not None:
            self._z_trajectories[modality][t] = z_t.detach()

    def retrieve(self, query_features, scale_name, current_t):
        """
        Retrieve from BOTH modalities and fuse via learned gating.

        The fusion gate learns to weight IR vs VIS features
        based on the current decoder state.

        Args:
            query_features: [B, C, H, W] current UNet features
            scale_name: 'scale2', 'scale3', or 'bottleneck'
            current_t: int, current timestep

        Returns:
            fused_retrieved: [B, C, H, W] combined IR+VIS memory features
        """
        vis_feat = self.vis_memory.retrieve(
            query_features, scale_name, current_t
        )
        ir_feat = self.ir_memory.retrieve(
            query_features, scale_name, current_t
        )

        # If both are zero (nothing stored), return zero
        if (vis_feat == 0).all() and (ir_feat == 0).all():
            return torch.zeros_like(query_features)

        # Learned fusion gate: blend IR and VIS retrieved features
        combined = torch.cat([vis_feat, ir_feat], dim=1)
        gate = self.fusion_gates[scale_name](combined)
        fused = gate * vis_feat + (1 - gate) * ir_feat

        return fused

    def get_z_trajectory(self, modality, t):
        """Get stored z_t for a modality at timestep t."""
        return self._z_trajectories.get(modality, {}).get(t)

    @property
    def num_stored(self):
        return {
            'vis': self.vis_memory.num_stored,
            'ir': self.ir_memory.num_stored,
        }
