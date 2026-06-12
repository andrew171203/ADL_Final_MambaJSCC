import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers import UNet2DModel, DDPMScheduler, DPMSolverMultistepScheduler


class LightweightDiffusionRefiner(nn.Module):
    """
    Simplified lightweight diffusion refiner for VSSBJSCC.

    Receiver-side post-decoder refinement:
        decoder output x_hat -> diffusion refiner -> refined image

    This simplified version intentionally removes SNR/CBR maps from the UNet input.

    UNet input channels:
        noisy image      : 3 channels
        decoder output   : 3 channels
        total            : 6 channels

    UNet output:
        predicted noise  : 3 channels

    Notes:
        - training_loss(clean, cond, snr=None, cbr=None) keeps optional snr/cbr
          arguments only for backward compatibility. They are ignored.
        - refine(cond, snr=None, cbr=None, ...) also keeps optional snr/cbr
          arguments only for backward compatibility. They are ignored.
        - Default inference is low-noise one-step refinement controlled by
          strength and blend. This is much safer for PSNR than starting from
          a heavily noised sample.
    """

    def __init__(
        self,
        train_timesteps=256,
        inference_steps=1,
        blend=0.01,
        base_channels=16,
        strength=0.05,
    ):
        super().__init__()

        self.train_timesteps = int(train_timesteps)
        self.inference_steps = int(inference_steps)
        self.blend = float(blend)
        self.strength = float(strength)

        self.unet = UNet2DModel(
            sample_size=None,
            in_channels=6,
            out_channels=3,
            layers_per_block=1,
            block_out_channels=(
                base_channels,
                base_channels * 2,
                base_channels * 4,
            ),
            down_block_types=(
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
            ),
            up_block_types=(
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
            ),
            norm_num_groups=8,
        )

        # DDPM objective for training.
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=self.train_timesteps,
            beta_schedule="squaredcos_cap_v2",
            prediction_type="epsilon",
        )

        # Kept for compatibility with the DPM-Solver design, although the default
        # refine() path below uses a safer low-noise one-step reconstruction.
        self.sample_scheduler = DPMSolverMultistepScheduler.from_config(
            self.noise_scheduler.config,
            algorithm_type="dpmsolver++",
            solver_order=2,
        )

    @staticmethod
    def _to_diffusion_space(x):
        """Convert image from [0, 1] to [-1, 1]."""
        return x.clamp(0.0, 1.0) * 2.0 - 1.0

    @staticmethod
    def _to_image_space(x):
        """Convert image from [-1, 1] to [0, 1]."""
        return ((x + 1.0) / 2.0).clamp(0.0, 1.0)

    def training_loss(self, clean, cond, snr=None, cbr=None):
        """
        clean: original image, range [0, 1], shape [B, 3, H, W]
        cond : decoder output, range [0, 1], shape [B, 3, H, W]

        snr/cbr are accepted but ignored in this simplified 6-channel version.
        """
        clean = clean.clamp(0.0, 1.0)
        cond = cond.clamp(0.0, 1.0)

        clean_d = self._to_diffusion_space(clean)
        cond_d = self._to_diffusion_space(cond)

        B = clean_d.shape[0]
        device = clean_d.device

        noise = torch.randn_like(clean_d)
        timesteps = torch.randint(
            low=0,
            high=self.train_timesteps,
            size=(B,),
            device=device,
        ).long()

        # Train to denoise the clean target while conditioned on the decoder output.
        noisy = self.noise_scheduler.add_noise(clean_d, noise, timesteps)
        model_input = torch.cat([noisy, cond_d], dim=1)

        pred_noise = self.unet(model_input, timesteps).sample
        loss_noise = F.mse_loss(pred_noise, noise)

        # Image-space stabilizer.
        alphas_cumprod = self.noise_scheduler.alphas_cumprod.to(device)
        alpha = alphas_cumprod[timesteps].view(B, 1, 1, 1)

        pred_clean_d = (noisy - (1.0 - alpha).sqrt() * pred_noise) / alpha.sqrt()
        pred_clean = self._to_image_space(pred_clean_d)
        loss_img = F.mse_loss(pred_clean, clean)

        return loss_noise + 0.1 * loss_img

    @torch.no_grad()
    def refine(self, cond, snr=None, cbr=None, steps=None, blend=None, strength=None):
        """
        Safe low-noise refinement.

        cond: decoder output, range [0, 1], shape [B, 3, H, W]

        snr/cbr are accepted but ignored in this simplified 6-channel version.
        steps is accepted for CLI compatibility; this safe path uses one denoise step.
        """
        blend = float(blend) if blend is not None else self.blend
        strength = float(strength) if strength is not None else self.strength

        cond = cond.clamp(0.0, 1.0)
        cond_d = self._to_diffusion_space(cond)

        B = cond.shape[0]
        device = cond.device

        # Use a small noise level only. Do not start from near-pure noise.
        t_value = int(round((self.train_timesteps - 1) * strength))
        t_value = max(1, min(self.train_timesteps - 1, t_value))
        timesteps = torch.full((B,), t_value, device=device, dtype=torch.long)

        noise = torch.randn_like(cond_d)
        noisy = self.noise_scheduler.add_noise(cond_d, noise, timesteps)
        model_input = torch.cat([noisy, cond_d], dim=1)

        pred_noise = self.unet(model_input, timesteps).sample

        alphas_cumprod = self.noise_scheduler.alphas_cumprod.to(device)
        alpha = alphas_cumprod[timesteps].view(B, 1, 1, 1)

        pred_clean_d = (noisy - (1.0 - alpha).sqrt() * pred_noise) / alpha.sqrt()
        refined = self._to_image_space(pred_clean_d)

        # Conservative blend to protect the strong JSCC decoder output.
        out = torch.lerp(cond, refined, blend)
        return out.clamp(0.0, 1.0)
