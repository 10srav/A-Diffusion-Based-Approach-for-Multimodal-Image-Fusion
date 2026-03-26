"""
Adaptive U-Net for Diffusion-Based Image Fusion
Lightweight denoising network (~12.5M params) with timestep embedding,
self-attention at low resolutions, and multi-scale condition injection.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def _num_groups(channels, max_groups=32):
    """Find largest group count <= max_groups that divides channels."""
    for g in range(min(max_groups, channels), 0, -1):
        if channels % g == 0:
            return g
    return 1


class SinusoidalTimestepEmbedding(nn.Module):
    """Sinusoidal timestep embedding following DDPM (Ho et al. 2020)."""

    def __init__(self, dim=128, time_embed_dim=256):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

    def forward(self, t):
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device, dtype=torch.float32) * -emb)
        emb = t.float()[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return self.mlp(emb)


class ResidualBlock(nn.Module):
    """Residual block with timestep embedding injection."""

    def __init__(self, in_channels, out_channels, time_embed_dim=256):
        super().__init__()
        self.norm1 = nn.GroupNorm(_num_groups(in_channels), in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.time_proj = nn.Linear(time_embed_dim, out_channels)
        self.norm2 = nn.GroupNorm(_num_groups(out_channels), out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.skip = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        self.act = nn.SiLU()

    def forward(self, x, t_emb):
        h = self.act(self.norm1(x))
        h = self.conv1(h)
        h = h + self.time_proj(t_emb)[:, :, None, None]
        h = self.act(self.norm2(h))
        h = self.conv2(h)
        return h + self.skip(x)


class SelfAttentionBlock(nn.Module):
    """Self-attention block for capturing global context."""

    def __init__(self, channels):
        super().__init__()
        self.norm = nn.GroupNorm(_num_groups(channels), channels)
        self.qkv = nn.Linear(channels, channels * 3)
        self.out_proj = nn.Linear(channels, channels)
        self.channels = channels

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x)
        h = h.reshape(B, C, H * W).permute(0, 2, 1)  # [B, HW, C]

        qkv = self.qkv(h).chunk(3, dim=-1)
        q, k, v = qkv

        attn = F.scaled_dot_product_attention(q, k, v)
        out = self.out_proj(attn)

        out = out.permute(0, 2, 1).reshape(B, C, H, W)
        return x + out


class Downsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        return self.conv(x)


class AdaptiveUNet(nn.Module):
    """
    Lightweight denoising U-Net for adaptive diffusion fusion.

    Architecture: [64, 128, 256, 512] channels across 4 resolution levels.
    Self-attention at 64x64 (level 2) and 32x32 (level 3) and bottleneck.
    Condition features injected at skip connections.

    Args:
        in_channels: Input channels (noisy image + scale-0 condition)
        out_channels: Output channels (predicted noise, typically 3)
        channel_mult: Channel counts at each level
        time_embed_dim: Timestep embedding dimension
        condition_channels: Channels of condition features at each scale [scale0, scale1, scale2, scale3]
    """

    def __init__(
        self,
        in_channels=19,       # 3 (noisy) + 16 (scale-0 condition)
        out_channels=3,
        channel_mult=(32, 64, 128, 256),
        time_embed_dim=256,
        condition_channels=(16, 32, 64, 128),
    ):
        super().__init__()
        self.time_embed = SinusoidalTimestepEmbedding(256, time_embed_dim)

        # ---- Encoder ----
        self.enc_conv_in = nn.Conv2d(in_channels, channel_mult[0], 3, padding=1)

        # Level 0: 256x256, 64ch
        self.enc_res0a = ResidualBlock(channel_mult[0], channel_mult[0], time_embed_dim)
        self.enc_res0b = ResidualBlock(channel_mult[0], channel_mult[0], time_embed_dim)
        self.down0 = Downsample(channel_mult[0])

        # Level 1: 128x128, 128ch
        self.enc_res1a = ResidualBlock(channel_mult[0], channel_mult[1], time_embed_dim)
        self.enc_res1b = ResidualBlock(channel_mult[1], channel_mult[1], time_embed_dim)
        self.down1 = Downsample(channel_mult[1])

        # Level 2: 64x64, 256ch (with attention)
        self.enc_res2a = ResidualBlock(channel_mult[1], channel_mult[2], time_embed_dim)
        self.enc_res2b = ResidualBlock(channel_mult[2], channel_mult[2], time_embed_dim)
        self.enc_attn2 = SelfAttentionBlock(channel_mult[2])
        self.down2 = Downsample(channel_mult[2])

        # Level 3: 32x32, 512ch (with attention)
        self.enc_res3a = ResidualBlock(channel_mult[2], channel_mult[3], time_embed_dim)
        self.enc_res3b = ResidualBlock(channel_mult[3], channel_mult[3], time_embed_dim)
        self.enc_attn3 = SelfAttentionBlock(channel_mult[3])
        self.down3 = Downsample(channel_mult[3])

        # ---- Bottleneck: 16x16 ----
        self.mid_res1 = ResidualBlock(channel_mult[3], channel_mult[3], time_embed_dim)
        self.mid_attn = SelfAttentionBlock(channel_mult[3])
        self.mid_res2 = ResidualBlock(channel_mult[3], channel_mult[3], time_embed_dim)

        # ---- Decoder ----
        # Level 3: 32x32 (skip from enc level 3 + condition[3])
        dec3_in = channel_mult[3] + channel_mult[3] + condition_channels[3]
        self.up3 = Upsample(channel_mult[3])
        self.dec_res3a = ResidualBlock(dec3_in, channel_mult[3], time_embed_dim)
        self.dec_res3b = ResidualBlock(channel_mult[3], channel_mult[3], time_embed_dim)
        self.dec_attn3 = SelfAttentionBlock(channel_mult[3])

        # Level 2: 64x64 (skip from enc level 2 + condition[2])
        dec2_in = channel_mult[3] + channel_mult[2] + condition_channels[2]
        self.up2 = Upsample(channel_mult[3])
        self.dec_res2a = ResidualBlock(dec2_in, channel_mult[2], time_embed_dim)
        self.dec_res2b = ResidualBlock(channel_mult[2], channel_mult[2], time_embed_dim)
        self.dec_attn2 = SelfAttentionBlock(channel_mult[2])

        # Level 1: 128x128 (skip from enc level 1 + condition[1])
        dec1_in = channel_mult[2] + channel_mult[1] + condition_channels[1]
        self.up1 = Upsample(channel_mult[2])
        self.dec_res1a = ResidualBlock(dec1_in, channel_mult[1], time_embed_dim)
        self.dec_res1b = ResidualBlock(channel_mult[1], channel_mult[1], time_embed_dim)

        # Level 0: 256x256 (skip from enc level 0 + condition[0])
        dec0_in = channel_mult[1] + channel_mult[0] + condition_channels[0]
        self.up0 = Upsample(channel_mult[1])
        self.dec_res0a = ResidualBlock(dec0_in, channel_mult[0], time_embed_dim)
        self.dec_res0b = ResidualBlock(channel_mult[0], channel_mult[0], time_embed_dim)

        # Output
        self.out_norm = nn.GroupNorm(_num_groups(channel_mult[0]), channel_mult[0])
        self.out_act = nn.SiLU()
        self.out_conv = nn.Conv2d(channel_mult[0], out_channels, 3, padding=1)

    def _run_encoder_and_bottleneck(self, x, t_emb):
        """Run encoder and bottleneck, returning skips and bottleneck feature."""
        h = self.enc_conv_in(x)

        # Level 0
        h = self.enc_res0a(h, t_emb)
        h = self.enc_res0b(h, t_emb)
        skip0 = h
        h = self.down0(h)

        # Level 1
        h = self.enc_res1a(h, t_emb)
        h = self.enc_res1b(h, t_emb)
        skip1 = h
        h = self.down1(h)

        # Level 2
        h = self.enc_res2a(h, t_emb)
        h = self.enc_res2b(h, t_emb)
        h = self.enc_attn2(h)
        skip2 = h
        h = self.down2(h)

        # Level 3
        h = self.enc_res3a(h, t_emb)
        h = self.enc_res3b(h, t_emb)
        h = self.enc_attn3(h)
        skip3 = h
        h = self.down3(h)

        # Bottleneck
        h = self.mid_res1(h, t_emb)
        h = self.mid_attn(h)
        h_bottleneck = h  # capture before second res block
        h = self.mid_res2(h, t_emb)

        return h, [skip0, skip1, skip2, skip3], h_bottleneck

    def _run_decoder(self, h, skips, t_emb, condition_features=None,
                     memory_bank=None, current_t=None):
        """Run decoder with optional memory injection."""
        skip0, skip1, skip2, skip3 = skips

        # Decoder Level 3
        h = self.up3(h)
        skip_cat = [h, skip3]
        if condition_features is not None and len(condition_features) > 3:
            skip_cat.append(condition_features[3])
        h = torch.cat(skip_cat, dim=1)
        h = self.dec_res3a(h, t_emb)
        h = self.dec_res3b(h, t_emb)
        # Memory injection at scale3 (32x32)
        if memory_bank is not None and current_t is not None:
            h = h + memory_bank.retrieve(h, 'scale3', current_t)
        h = self.dec_attn3(h)

        # Decoder Level 2
        h = self.up2(h)
        skip_cat = [h, skip2]
        if condition_features is not None and len(condition_features) > 2:
            skip_cat.append(condition_features[2])
        h = torch.cat(skip_cat, dim=1)
        h = self.dec_res2a(h, t_emb)
        h = self.dec_res2b(h, t_emb)
        # Memory injection at scale2 (64x64)
        if memory_bank is not None and current_t is not None:
            h = h + memory_bank.retrieve(h, 'scale2', current_t)
        h = self.dec_attn2(h)

        # Decoder Level 1
        h = self.up1(h)
        skip_cat = [h, skip1]
        if condition_features is not None and len(condition_features) > 1:
            skip_cat.append(condition_features[1])
        h = torch.cat(skip_cat, dim=1)
        h = self.dec_res1a(h, t_emb)
        h = self.dec_res1b(h, t_emb)

        # Decoder Level 0
        h = self.up0(h)
        skip_cat = [h, skip0]
        if condition_features is not None and len(condition_features) > 0:
            skip_cat.append(condition_features[0])
        h = torch.cat(skip_cat, dim=1)
        h = self.dec_res0a(h, t_emb)
        h = self.dec_res0b(h, t_emb)

        # Output
        h = self.out_act(self.out_norm(h))
        return self.out_conv(h)

    def forward(self, x, t, condition_features=None,
                memory_bank=None, current_t=None):
        """
        Forward pass.

        Args:
            x: [B, in_channels, 256, 256] noisy image concatenated with scale-0 condition
            t: [B] timesteps
            condition_features: list of [feat_0, feat_1, feat_2, feat_3] for skip injection
            memory_bank: optional FeatureMemoryBank for temporal feature retrieval
            current_t: optional int, current timestep for memory lookup

        Returns:
            [B, out_channels, 256, 256] predicted noise
        """
        t_emb = self.time_embed(t)
        h, skips, _ = self._run_encoder_and_bottleneck(x, t_emb)

        # Memory injection at bottleneck (16x16)
        if memory_bank is not None and current_t is not None:
            h = h + memory_bank.retrieve(h, 'bottleneck', current_t)

        return self._run_decoder(h, skips, t_emb, condition_features,
                                 memory_bank, current_t)

    def forward_with_intermediates(self, x, t, condition_features=None):
        """
        Same as forward() but also returns intermediate features
        for caching in FeatureMemoryBank during DDIM inversion.

        Returns:
            noise_pred: [B, out_channels, 256, 256]
            intermediates: dict with 'encoder_skips' and 'bottleneck'
        """
        t_emb = self.time_embed(t)
        h, skips, h_bottleneck = self._run_encoder_and_bottleneck(x, t_emb)

        noise_pred = self._run_decoder(h, skips, t_emb, condition_features)

        intermediates = {
            'encoder_skips': skips,
            'bottleneck': h_bottleneck,
        }
        return noise_pred, intermediates
