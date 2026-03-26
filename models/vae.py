"""
Lightweight VAE for Adaptive Diffusion Fusion
Maps RGB images [3,256,256] to latent space [4,32,32] (8x compression).
Separate encoders for IR and VIS ensure modality-specific encoding,
while a shared decoder reconstructs the fused output.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .adaptive_unet import _num_groups


class ResBlock(nn.Module):
    """Residual block for VAE encoder/decoder."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.norm1 = nn.GroupNorm(_num_groups(in_channels), in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(_num_groups(out_channels), out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.skip = (nn.Conv2d(in_channels, out_channels, 1)
                     if in_channels != out_channels else nn.Identity())
        self.act = nn.SiLU()

    def forward(self, x):
        h = self.act(self.norm1(x))
        h = self.conv1(h)
        h = self.act(self.norm2(h))
        h = self.conv2(h)
        return h + self.skip(x)


class VAEEncoder(nn.Module):
    """
    Encodes RGB image [B,3,256,256] -> latent [B,4,32,32].
    3 stages of downsampling (256->128->64->32) with residual blocks.
    Outputs mean and logvar for reparameterization.
    """

    def __init__(self, in_channels=3, latent_channels=4,
                 hidden_channels=(64, 128, 256)):
        super().__init__()

        # Initial convolution
        self.conv_in = nn.Conv2d(in_channels, hidden_channels[0], 3, padding=1)

        # Downsampling blocks: 256->128->64->32
        self.down_blocks = nn.ModuleList()
        ch_in = hidden_channels[0]
        for ch_out in hidden_channels:
            self.down_blocks.append(nn.ModuleList([
                ResBlock(ch_in, ch_out),
                ResBlock(ch_out, ch_out),
                nn.Conv2d(ch_out, ch_out, 3, stride=2, padding=1),  # downsample
            ]))
            ch_in = ch_out

        # Final blocks at 32x32
        self.mid_block = ResBlock(hidden_channels[-1], hidden_channels[-1])

        # Output: mean and logvar
        self.norm_out = nn.GroupNorm(
            _num_groups(hidden_channels[-1]), hidden_channels[-1]
        )
        self.conv_out = nn.Conv2d(
            hidden_channels[-1], latent_channels * 2, 3, padding=1
        )
        self.latent_channels = latent_channels

    def forward(self, x):
        """
        Args:
            x: [B, 3, 256, 256] in [-1, 1]
        Returns:
            mean: [B, 4, 32, 32]
            logvar: [B, 4, 32, 32]
        """
        h = self.conv_in(x)

        for res1, res2, down in self.down_blocks:
            h = res1(h)
            h = res2(h)
            h = down(h)

        h = self.mid_block(h)
        h = F.silu(self.norm_out(h))
        h = self.conv_out(h)

        mean, logvar = h.chunk(2, dim=1)
        logvar = torch.clamp(logvar, -30, 20)
        return mean, logvar

    def encode(self, x, sample=True):
        """Encode and optionally sample via reparameterization."""
        mean, logvar = self.forward(x)
        if sample:
            std = torch.exp(0.5 * logvar)
            z = mean + std * torch.randn_like(std)
            return z, mean, logvar
        return mean, mean, logvar


class VAEDecoder(nn.Module):
    """
    Decodes latent [B,4,32,32] -> RGB image [B,3,256,256].
    3 stages of upsampling (32->64->128->256) with residual blocks.
    """

    def __init__(self, latent_channels=4, out_channels=3,
                 hidden_channels=(256, 128, 64)):
        super().__init__()

        self.conv_in = nn.Conv2d(
            latent_channels, hidden_channels[0], 3, padding=1
        )

        # Mid block at 32x32
        self.mid_block = ResBlock(hidden_channels[0], hidden_channels[0])

        # Upsampling blocks: 32->64->128->256
        self.up_blocks = nn.ModuleList()
        ch_in = hidden_channels[0]
        for ch_out in hidden_channels:
            self.up_blocks.append(nn.ModuleList([
                ResBlock(ch_in, ch_out),
                ResBlock(ch_out, ch_out),
                nn.Sequential(  # upsample
                    nn.Upsample(scale_factor=2, mode='nearest'),
                    nn.Conv2d(ch_out, ch_out, 3, padding=1),
                ),
            ]))
            ch_in = ch_out

        # Output
        self.norm_out = nn.GroupNorm(
            _num_groups(hidden_channels[-1]), hidden_channels[-1]
        )
        self.conv_out = nn.Conv2d(
            hidden_channels[-1], out_channels, 3, padding=1
        )

    def forward(self, z):
        """
        Args:
            z: [B, 4, 32, 32] latent
        Returns:
            x: [B, 3, 256, 256] in [-1, 1]
        """
        h = self.conv_in(z)
        h = self.mid_block(h)

        for res1, res2, up in self.up_blocks:
            h = res1(h)
            h = res2(h)
            h = up(h)

        h = F.silu(self.norm_out(h))
        return torch.tanh(self.conv_out(h))


class FusionVAE(nn.Module):
    """
    Complete VAE for Adaptive Diffusion Fusion.

    Uses SEPARATE encoders for IR and VIS (modality-specific encoding)
    and a SHARED decoder for the fused output.

    Architecture:
        IR  [3,256,256] -> IR Encoder  -> z_ir  [4,32,32]
        VIS [3,256,256] -> VIS Encoder -> z_vis [4,32,32]
        Fused latent     -> Decoder    -> Fused [3,256,256]

    Total: ~2.5M params
    """

    def __init__(self, latent_channels=4):
        super().__init__()
        self.latent_channels = latent_channels

        # Separate encoders for each modality (lightweight)
        self.ir_encoder = VAEEncoder(
            in_channels=3, latent_channels=latent_channels,
            hidden_channels=(32, 64, 128),
        )
        self.vis_encoder = VAEEncoder(
            in_channels=3, latent_channels=latent_channels,
            hidden_channels=(32, 64, 128),
        )

        # Shared decoder for fused output (lightweight)
        self.decoder = VAEDecoder(
            latent_channels=latent_channels, out_channels=3,
            hidden_channels=(128, 64, 32),
        )

        # Scaling factor for latent normalization (learned)
        self.register_buffer(
            'latent_scale', torch.tensor(0.18215)  # similar to SD scaling
        )

    def encode_ir(self, ir_image, sample=True):
        """Encode IR image to latent space."""
        z, mean, logvar = self.ir_encoder.encode(ir_image, sample=sample)
        return z * self.latent_scale, mean, logvar

    def encode_vis(self, vis_image, sample=True):
        """Encode VIS image to latent space."""
        z, mean, logvar = self.vis_encoder.encode(vis_image, sample=sample)
        return z * self.latent_scale, mean, logvar

    def decode(self, z):
        """Decode latent to image."""
        return self.decoder(z / self.latent_scale)

    def forward(self, ir_image, vis_image):
        """
        Full forward: encode both, return latents and reconstructions.
        Used for VAE pretraining.
        """
        z_ir, mean_ir, logvar_ir = self.encode_ir(ir_image)
        z_vis, mean_vis, logvar_vis = self.encode_vis(vis_image)

        recon_ir = self.decode(z_ir)
        recon_vis = self.decode(z_vis)

        return {
            'z_ir': z_ir, 'z_vis': z_vis,
            'mean_ir': mean_ir, 'logvar_ir': logvar_ir,
            'mean_vis': mean_vis, 'logvar_vis': logvar_vis,
            'recon_ir': recon_ir, 'recon_vis': recon_vis,
        }

    @staticmethod
    def vae_loss(recon, target, mean, logvar, kl_weight=1e-6):
        """VAE reconstruction + KL divergence loss."""
        recon_loss = F.mse_loss(recon, target, reduction='mean')
        kl_loss = -0.5 * torch.mean(
            1 + logvar - mean.pow(2) - logvar.exp()
        )
        return recon_loss + kl_weight * kl_loss, recon_loss, kl_loss
