from net.decoder import *
from net.encoder import *
from loss.distortion import Distortion
from net.channel import Channel
from random import choice
from net.model_names import normalize_model_name
from net.diffusion_refiner import LightweightDiffusionRefiner
import torch
import torch.nn as nn


class VSSBJSCC(nn.Module):
    def __init__(self, args, config):
        super(VSSBJSCC, self).__init__()
        self.config = config
        encoder_kwargs = config.encoder_kwargs
        decoder_kwargs = config.decoder_kwargs
        self.encoder = create_encoder(**encoder_kwargs)
        self.decoder = create_decoder(**decoder_kwargs)
        if config.logger is not None:
            config.logger.info("Network config: ")
            config.logger.info("Encoder: ")
            config.logger.info(encoder_kwargs)
            config.logger.info("Decoder: ")
            config.logger.info(decoder_kwargs)
        self.distortion_loss = Distortion(args)
        self.channel = Channel(args, config)
        self.pass_channel = config.pass_channel
        self.squared_difference = torch.nn.MSELoss(reduction='none')
        self.H = self.W = 0
        self.multiple_snr = args.multiple_snr.split(",")
        for i in range(len(self.multiple_snr)):
            self.multiple_snr[i] = int(self.multiple_snr[i])
        self.channel_number = args.C.split(",")
        for i in range(len(self.channel_number)):
            self.channel_number[i] = int(self.channel_number[i])
        self.downsample = config.downsample
        self.model = normalize_model_name(args.model)

        # Receiver-side lightweight diffusion refinement.
        # This is disabled by default and only created when --use-diffusion is used.
        self.use_diffusion = getattr(args, "use_diffusion", False)
        if self.use_diffusion:
            self.diffusion_refiner = LightweightDiffusionRefiner(
                train_timesteps=getattr(args, "diffusion_train_timesteps", 256),
                inference_steps=getattr(args, "diffusion_steps", 1),
                blend=getattr(args, "diffusion_blend", 0.01),
                base_channels=getattr(args, "diffusion_base_channels", 16),
                strength=getattr(args, "diffusion_strength", 0.05),
            )

    def distortion_loss_wrapper(self, x_gen, x_real):
        distortion_loss = self.distortion_loss.forward(x_gen, x_real, normalization=self.config.norm)
        return distortion_loss

    def feature_pass_channel(self, feature, chan_param, avg_pwr=False):
        noisy_feature = self.channel.forward(feature, chan_param, avg_pwr)
        return noisy_feature

    def forward_jscc(self, input_image, given_SNR=None, given_rate=None):
        """
        Original VSSBJSCC forward pass without diffusion.
        This keeps the stable encoder/channel/decoder pipeline unchanged.
        """
        B, _, H, W = input_image.shape

        if H != self.H or W != self.W:
            self.encoder.update_resolution(H, W)
            self.decoder.update_resolution(H // (2 ** self.downsample), W // (2 ** self.downsample))
            self.H = H
            self.W = W

        if given_SNR is None:
            SNR = choice(self.multiple_snr)
            chan_param = SNR
        else:
            chan_param = given_SNR

        if given_rate is None:
            channel_number = choice(self.channel_number)
        else:
            channel_number = given_rate

        if self.model == 'VSSBJSCC_w/o_SAandRA' or self.model == 'VSSBJSCC_w/_SA':
            feature = self.encoder(input_image, chan_param, channel_number, self.model)
            CBR = feature.numel() / 2 / input_image.numel()
            if self.pass_channel:
                noisy_feature = self.feature_pass_channel(feature, chan_param)
            else:
                noisy_feature = feature

        elif self.model == 'VSSBJSCC_w/_RA' or self.model == 'VSSBJSCC_w/_SAandRA':
            feature, mask = self.encoder(input_image, chan_param, channel_number, self.model)
            CBR = channel_number / (2 * 3 * 2 ** (self.downsample * 2))
            avg_pwr = torch.sum(feature ** 2) / mask.sum()
            if self.pass_channel:
                noisy_feature = self.feature_pass_channel(feature, chan_param, avg_pwr)
            else:
                noisy_feature = feature
            noisy_feature = noisy_feature * mask

        else:
            raise ValueError(f"Unsupported model name: {self.model}")

        recon_image = self.decoder(noisy_feature, chan_param, self.model)
        mse = self.squared_difference(input_image * 255., recon_image.clamp(0., 1.) * 255.)
        loss_G = self.distortion_loss.forward(input_image, recon_image.clamp(0., 1.))
        return recon_image, CBR, chan_param, mse.mean(), loss_G.mean()

    def forward(self, input_image, given_SNR=None, given_rate=None):
        """
        Training without diffusion keeps the original JSCC behavior.
        Evaluation with --use-diffusion applies post-decoder diffusion refinement.
        """
        recon_image, CBR, chan_param, mse, loss_G = self.forward_jscc(
            input_image,
            given_SNR,
            given_rate,
        )

        if self.use_diffusion and (not self.training):
            recon_image = self.diffusion_refiner.refine(
                cond=recon_image,
            )

            mse = self.squared_difference(
                input_image * 255.,
                recon_image.clamp(0., 1.) * 255.,
            )
            loss_G = self.distortion_loss.forward(
                input_image,
                recon_image.clamp(0., 1.),
            )

        return recon_image, CBR, chan_param, mse.mean(), loss_G.mean()

    def diffusion_training_loss(
        self,
        input_image,
        given_SNR=None,
        given_rate=None,
        detach_jscc=True,
    ):
        """
        Train the diffusion refiner using the original image as clean target and
        the VSSBJSCC decoder output as condition.
        """
        if not self.use_diffusion:
            raise RuntimeError("use_diffusion=False, but diffusion_training_loss() was called.")

        if detach_jscc:
            with torch.no_grad():
                recon_image, CBR, chan_param, mse, loss_G = self.forward_jscc(
                    input_image,
                    given_SNR,
                    given_rate,
                )
            recon_image = recon_image.detach()
        else:
            recon_image, CBR, chan_param, mse, loss_G = self.forward_jscc(
                input_image,
                given_SNR,
                given_rate,
            )

        loss_diff = self.diffusion_refiner.training_loss(
            clean=input_image,
            cond=recon_image.clamp(0., 1.),
        )

        return loss_diff, recon_image, CBR, chan_param, mse, loss_G

    @staticmethod
    def _set_requires_grad(module, requires_grad):
        for p in module.parameters():
            p.requires_grad = requires_grad

    def freeze_jscc(self):
        """Freeze the stable encoder/channel/decoder and train only diffusion."""
        self._set_requires_grad(self.encoder, False)
        self._set_requires_grad(self.decoder, False)
        self._set_requires_grad(self.channel, False)
        self.encoder.eval()
        self.decoder.eval()
        self.channel.eval()
        if self.use_diffusion:
            self._set_requires_grad(self.diffusion_refiner, True)
            self.diffusion_refiner.train()

    def unfreeze_decoder_only(self):
        """Freeze encoder/channel, train decoder and diffusion refiner."""
        self._set_requires_grad(self.encoder, False)
        self._set_requires_grad(self.channel, False)
        self._set_requires_grad(self.decoder, True)
        self.encoder.eval()
        self.channel.eval()
        self.decoder.train()
        if self.use_diffusion:
            self._set_requires_grad(self.diffusion_refiner, True)
            self.diffusion_refiner.train()

    def unfreeze_all_jscc(self):
        """Unfreeze encoder and decoder. Use carefully with a very small learning rate."""
        self._set_requires_grad(self.encoder, True)
        self._set_requires_grad(self.decoder, True)
        self.encoder.train()
        self.decoder.train()
        if self.use_diffusion:
            self._set_requires_grad(self.diffusion_refiner, True)
            self.diffusion_refiner.train()


# Backward-compatible alias for old scripts/checkpoints.
SwinJSCC = VSSBJSCC
