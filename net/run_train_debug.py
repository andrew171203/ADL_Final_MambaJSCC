import torch
import torch.nn as nn
import torch.optim as optim
from types import SimpleNamespace

from net.network import VSSBJSCC


args = SimpleNamespace(
    trainset="DIV2K",
    distortion_metric="MSE",
    channel_type="awgn",
    multiple_snr="10",
    C="96",
    model="VSSBJSCC_w/_SAandRA",
)


class config:
    pass_channel = True
    CUDA = True
    device = torch.device("cuda:0")
    norm = False
    logger = None
    downsample = 4

    encoder_kwargs = dict(
        model=args.model,
        img_size=(256, 256),
        patch_size=2,
        in_chans=3,
        embed_dims=[128, 192, 256, 320],
        depths=[2, 2, 2, 2],
        num_heads=[4, 6, 8, 10],
        C=None,
        window_size=8,
        mlp_ratio=4.,
        qkv_bias=True,
        qk_scale=None,
        norm_layer=nn.LayerNorm,
        patch_norm=True,
    )

    decoder_kwargs = dict(
        model=args.model,
        img_size=(256, 256),
        embed_dims=[320, 256, 192, 128],
        depths=[2, 2, 2, 2],
        num_heads=[10, 8, 6, 4],
        C=None,
        window_size=8,
        mlp_ratio=4.,
        qkv_bias=True,
        qk_scale=None,
        norm_layer=nn.LayerNorm,
        patch_norm=True,
    )


torch.manual_seed(42)

net = VSSBJSCC(args, config).cuda()
net.train()

optimizer = optim.Adam(net.parameters(), lr=1e-4)

for step in range(3):
    x = torch.rand(1, 3, 256, 256).cuda()

    recon, CBR, snr, mse, loss = net(x, given_SNR=10, given_rate=96)

    optimizer.zero_grad()
    loss.backward()

    torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)

    optimizer.step()

    print(
        f"Step {step + 1} | "
        f"loss={loss.item():.6f} | "
        f"mse={mse.item():.6f} | "
        f"CBR={CBR} | "
        f"SNR={snr} | "
        f"recon_shape={recon.shape}"
    )