"""
Adaptive Fusion Network for IR-VIS Image Fusion
Dual modality encoder with content-aware adaptive fusion attention.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .adaptive_unet import AdaptiveUNet, _num_groups


class ConvFeatureBlock(nn.Module):
    """Two-layer convolution feature extraction block."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.GroupNorm(_num_groups(out_channels), out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.GroupNorm(_num_groups(out_channels), out_channels),
            nn.SiLU(),
        )

    def forward(self, x):
        return self.block(x)


class ModalityEncoder(nn.Module):
    """
    Multi-scale feature extractor for a single modality (IR or VIS).
    Extracts features at 4 resolution levels.
    """

    def __init__(self, in_channels=3, feature_channels=(16, 32, 64, 128)):
        super().__init__()
        self.level0 = ConvFeatureBlock(in_channels, feature_channels[0])      # 256x256
        self.level1 = ConvFeatureBlock(feature_channels[0], feature_channels[1])  # 128x128
        self.level2 = ConvFeatureBlock(feature_channels[1], feature_channels[2])  # 64x64
        self.level3 = ConvFeatureBlock(feature_channels[2], feature_channels[3])  # 32x32
        self.pool = nn.AvgPool2d(2)

    def forward(self, x):
        """
        Args:
            x: [B, 3, 256, 256] input image in [-1, 1]

        Returns:
            List of features at 4 scales:
            [B,32,256,256], [B,64,128,128], [B,128,64,64], [B,256,32,32]
        """
        f0 = self.level0(x)          # [B, 32, 256, 256]
        f1 = self.level1(self.pool(f0))  # [B, 64, 128, 128]
        f2 = self.level2(self.pool(f1))  # [B, 128, 64, 64]
        f3 = self.level3(self.pool(f2))  # [B, 256, 32, 32]
        return [f0, f1, f2, f3]


class AdaptiveFusionGate(nn.Module):
    """
    Convolutional content-aware gating for high-resolution scales.
    Used at 256x256 and 128x128 where cross-attention is too expensive.
    """

    def __init__(self, channels):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1),
            nn.Sigmoid()
        )

    def forward(self, ir_feat, vis_feat):
        """
        Args:
            ir_feat: [B, C, H, W] IR features
            vis_feat: [B, C, H, W] VIS features

        Returns:
            fused: [B, C, H, W] adaptively fused features
        """
        combined = torch.cat([ir_feat, vis_feat], dim=1)
        gate = self.gate(combined)
        return gate * ir_feat + (1 - gate) * vis_feat


class AdaptiveFusionAttention(nn.Module):
    """
    Cross-attention based adaptive fusion for low-resolution scales.
    IR attends to VIS and vice versa, then content-aware gating blends them.
    """

    def __init__(self, channels):
        super().__init__()
        self.channels = channels

        # IR -> VIS cross attention
        self.q_ir = nn.Linear(channels, channels)
        self.k_vis = nn.Linear(channels, channels)
        self.v_vis = nn.Linear(channels, channels)

        # VIS -> IR cross attention
        self.q_vis = nn.Linear(channels, channels)
        self.k_ir = nn.Linear(channels, channels)
        self.v_ir = nn.Linear(channels, channels)

        # Content-aware gating
        self.gate_net = nn.Sequential(
            nn.Linear(channels * 4, channels),
            nn.Sigmoid()
        )

        # Output projection
        self.out_proj = nn.Linear(channels, channels)

    def forward(self, ir_feat, vis_feat):
        """
        Args:
            ir_feat: [B, C, H, W]
            vis_feat: [B, C, H, W]

        Returns:
            fused: [B, C, H, W]
        """
        B, C, H, W = ir_feat.shape

        # Reshape to sequence: [B, HW, C]
        ir_seq = ir_feat.reshape(B, C, H * W).permute(0, 2, 1)
        vis_seq = vis_feat.reshape(B, C, H * W).permute(0, 2, 1)

        # IR queries VIS
        q_ir = self.q_ir(ir_seq)
        k_vis = self.k_vis(vis_seq)
        v_vis = self.v_vis(vis_seq)
        ir_cross = F.scaled_dot_product_attention(q_ir, k_vis, v_vis)

        # VIS queries IR
        q_vis = self.q_vis(vis_seq)
        k_ir = self.k_ir(ir_seq)
        v_ir = self.v_ir(ir_seq)
        vis_cross = F.scaled_dot_product_attention(q_vis, k_ir, v_ir)

        # Content-aware gating
        combined = torch.cat([ir_seq, vis_seq, ir_cross, vis_cross], dim=-1)
        gate = self.gate_net(combined)

        # Fuse: gate=1 favors IR, gate=0 favors VIS
        fused = gate * (ir_seq + ir_cross) + (1 - gate) * (vis_seq + vis_cross)
        fused = self.out_proj(fused)

        # Reshape back to spatial
        return fused.permute(0, 2, 1).reshape(B, C, H, W)


def create_pseudo_gt(ir_image, vis_image):
    """
    Create pseudo ground-truth fused image for training.
    Uses max-luminance strategy + visible color preservation.

    Args:
        ir_image: [B, 3, H, W] tensor in [-1, 1]
        vis_image: [B, 3, H, W] tensor in [-1, 1]

    Returns:
        fused: [B, 3, H, W] tensor in [-1, 1]
    """
    # Convert to [0, 1]
    ir_01 = (ir_image + 1) / 2
    vis_01 = (vis_image + 1) / 2

    # Luminance (Y channel)
    ir_y = 0.299 * ir_01[:, 0:1] + 0.587 * ir_01[:, 1:2] + 0.114 * ir_01[:, 2:3]
    vis_y = 0.299 * vis_01[:, 0:1] + 0.587 * vis_01[:, 1:2] + 0.114 * vis_01[:, 2:3]

    # Max luminance (structure from whichever source is stronger)
    fused_y = torch.max(ir_y, vis_y)

    # Chrominance from visible image (natural color)
    vis_cb = vis_01[:, 2:3] - vis_y
    vis_cr = vis_01[:, 0:1] - vis_y

    # Reconstruct RGB
    fused_r = fused_y + vis_cr
    fused_g = fused_y - 0.344136 * vis_cb - 0.714136 * vis_cr
    fused_b = fused_y + vis_cb

    fused = torch.cat([fused_r, fused_g, fused_b], dim=1)
    fused = torch.clamp(fused, 0, 1)

    # Convert back to [-1, 1]
    return fused * 2 - 1


class AdaptiveFusionModel(nn.Module):
    """
    Adaptive Diffusion Fusion Model.

    Complete model combining:
    - Dual modality encoders (IR + VIS)
    - Adaptive fusion attention (content-aware gating + cross-attention)
    - Denoising U-Net

    Total: ~17M parameters
    """

    def __init__(
        self,
        in_channels=3,
        feature_channels=(16, 32, 64, 128),
        unet_channels=(32, 64, 128, 256),
    ):
        super().__init__()

        # Dual encoders
        self.ir_encoder = ModalityEncoder(in_channels, feature_channels)
        self.vis_encoder = ModalityEncoder(in_channels, feature_channels)

        # Adaptive fusion at each scale
        self.fusion_blocks = nn.ModuleList([
            AdaptiveFusionGate(feature_channels[0]),       # 256x256: conv gating
            AdaptiveFusionGate(feature_channels[1]),       # 128x128: conv gating
            AdaptiveFusionAttention(feature_channels[2]),  # 64x64: cross-attention
            AdaptiveFusionAttention(feature_channels[3]),  # 32x32: cross-attention
        ])

        # Denoising U-Net
        self.unet = AdaptiveUNet(
            in_channels=in_channels + feature_channels[0],  # 3 + 32 = 35
            out_channels=in_channels,
            channel_mult=unet_channels,
            condition_channels=feature_channels,
        )

    def encode_conditions(self, ir_image, vis_image):
        """
        Extract and fuse multi-scale condition features.

        Args:
            ir_image: [B, 3, 256, 256] IR image in [-1, 1]
            vis_image: [B, 3, 256, 256] VIS image in [-1, 1]

        Returns:
            List of fused features at 4 scales
        """
        ir_feats = self.ir_encoder(ir_image)
        vis_feats = self.vis_encoder(vis_image)

        fused_feats = []
        for i, fusion_block in enumerate(self.fusion_blocks):
            fused = fusion_block(ir_feats[i], vis_feats[i])
            fused_feats.append(fused)

        return fused_feats

    def forward(self, x_noisy, t, ir_image, vis_image):
        """
        Forward pass: predict noise from noisy image conditioned on IR+VIS.

        Args:
            x_noisy: [B, 3, 256, 256] noisy fused image
            t: [B] timestep
            ir_image: [B, 3, 256, 256] clean IR input
            vis_image: [B, 3, 256, 256] clean VIS input

        Returns:
            predicted_noise: [B, 3, 256, 256]
        """
        cond_feats = self.encode_conditions(ir_image, vis_image)

        # Concatenate scale-0 features with noisy input
        x_in = torch.cat([x_noisy, cond_feats[0]], dim=1)

        # U-Net predicts noise, with multi-scale condition injection
        return self.unet(x_in, t, cond_feats)
