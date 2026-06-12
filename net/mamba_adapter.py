import torch
import torch.nn as nn
from net.vmamba import VSSBlock


class VSSBlockAdapter(nn.Module):
    def __init__(self, dim, input_resolution, mlp_ratio=4.0):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution

        self.block = VSSBlock(
            hidden_dim=dim,
            norm_layer=nn.LayerNorm,
            ssm_d_state=16,
            ssm_ratio=2.0,
            ssm_rank_ratio=2.0,
            ssm_dt_rank="auto",
            ssm_act_layer=nn.SiLU,
            ssm_conv=3,
            ssm_conv_bias=True,
            ssm_drop_rate=0.0,
            forward_type="v3",
            mlp_ratio=mlp_ratio,
            mlp_act_layer=nn.GELU,
            mlp_drop_rate=0.0,
            scan="cross",
            scan_number=4,
            extent="MLP",
            channel_adaptive="no",
        )

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.reshape(B, H, W, C)
        x = self.block(x)
        x = x.reshape(B, H * W, C)

        return x

    def update_mask(self):
        pass

    def flops(self):
        return 0