import torch
import torch.nn as nn
import math
from models.modules.csm_triton import cross_scan_3d_fn, cross_merge_3d_fn
from models.modules.csms6s import selective_scan_fn
import torch.nn.functional as F

class Linear3d(nn.Linear):
    def forward(self, x: torch.Tensor):
        #print(f"Linear3d input shape: {x.shape}")
        # Convert linear to 3D conv
        out = nn.functional.conv3d(x, self.weight[:, :, None, None, None], self.bias)
        #print(f"Linear3d output shape: {out.shape}")
        return out

class LayerNorm3d(nn.LayerNorm):
    def forward(self, x: torch.Tensor):
        #print(f"LayerNorm3d input shape: {x.shape}")
        # B, C, D, H, W -> B, D, H, W, C
        x = x.permute(0, 2, 3, 4, 1)
        x = nn.functional.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        # B, D, H, W, C -> B, C, D, H, W
        x = x.permute(0, 4, 1, 2, 3)
        #print(f"LayerNorm3d output shape: {x.shape}")
        return x

class SS3D(nn.Module):
    def __init__(
        self,
        d_model=96,
        d_state=64,
        ssm_ratio=1.0,
        dt_rank="auto",
        act_layer=nn.GELU,
        d_conv=3,
        conv_bias=False,
        dropout=0.0,
        bias=False,
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        initialize="v2",
        forward_type="m0_noz",
        channel_first=False,
        disable_z=False,
        k_group = 6,
        dim_head =64, 
        **kwargs,
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.k_group = k_group  # Changed from 4 to 6 for 3D directions
        self.d_model = int(d_model)
        self.d_state = int(d_state)
        self.d_inner = int(ssm_ratio * d_model)
        self.dt_rank = int(math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank)
        self.channel_first = channel_first
        self.with_dconv = d_conv > 1
        self.disable_z = disable_z
        Linear = Linear3d if channel_first else nn.Linear
        self.forward = self.forwardv2

        self.in_proj = Linear(self.d_model, self.d_inner * 2, bias=bias)
        self.act = act_layer()
        if self.with_dconv:
            self.conv3d = nn.Conv3d(
                in_channels=self.d_inner,
                out_channels=self.d_inner,
                groups=self.d_inner,
                bias=conv_bias,
                kernel_size=d_conv,
                padding=(d_conv - 1) // 2,
            )

        self.x_proj = [
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False)
            for _ in range(self.k_group)
        ]
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))
        del self.x_proj

        self.out_proj = Linear(self.d_inner, self.d_model, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        self.A_logs = nn.Parameter(torch.zeros((self.k_group * self.d_inner, self.d_state)))
        self.Ds = nn.Parameter(torch.ones((self.k_group * self.d_inner)))
        self.dt_projs_weight = nn.Parameter(0.1 * torch.rand((self.k_group, self.d_inner, self.dt_rank)))
        self.dt_projs_bias = nn.Parameter(0.1 * torch.rand((self.k_group, self.d_inner)))

    def forwardv2(self, x: torch.Tensor, **kwargs):
        #print(f"SS3D forwardv2 input shape: {x.shape}")

        B, N, C = x.shape
        x = x.permute(0, 2, 1).reshape(B, C, 5, 8, 8)
        x = self.in_proj(x)
        #print(f"After in_proj shape: {x.shape}")
        if not self.disable_z:
            x, z = x.chunk(2, dim=(1 if self.channel_first else -1))
            z = self.act(z)
            #print(f"After chunking, x shape: {x.shape}, z shape: {z.shape}")
        else:
            z = None
        if not self.channel_first:
            x = x.permute(0, 4, 1, 2, 3).contiguous()
            #print(f"After permute shape: {x.shape}")
        if self.with_dconv:
            x = self.conv3d(x)
            #print(f"After conv3d shape: {x.shape}")
        x = self.act(x)
        y = self.forward_core(x)
        #print(f"After forward_core shape: {y.shape}")
        if z is not None:
            z = z.view(y.shape)  # Ensure z matches the shape of y
            y = y * z
            #print(f"After applying z, y shape: {y.shape}")
        # Only reshape if using channel-last format
        if not self.channel_first:
            y = y.view(y.size(0), y.size(1), -1).permute(0, 2, 1).contiguous()
            #print(f"After view and permute shape: {y.shape}")
        out = self.dropout(self.out_proj(y))
        #print(f"SS3D forwardv2 output shape: {out.shape}")
        out = out.permute(0, 2, 3, 4, 1).reshape(B, N, C)
        return out

    def forward_core(self, x: torch.Tensor):
        #print(f"SS3D forward_core input shape: {x.shape}")
        
        B, D_inner, D, H, W = x.shape
        D_total, N = self.A_logs.shape
        K, D_inner, R = self.dt_projs_weight.shape
        L = D * H * W

        # Six scanning directions for 3D
        x_dhwdhw = torch.stack([
            x.view(B, -1, L),  # Along D
            x.transpose(2, 3).contiguous().view(B, -1, L),  # Along H
            x.transpose(2, 4).contiguous().view(B, -1, L),  # Along W
        ], dim=1)
        xs = torch.cat([x_dhwdhw, torch.flip(x_dhwdhw, dims=[-1])], dim=1)  # (B, 6, D_inner, L)
        #print(f"After stacking and concatenating, xs shape: {xs.shape}")

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, self.x_proj_weight)
        #print(f"After einsum, x_dbl shape: {x_dbl.shape}")
        
        dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts, self.dt_projs_weight)
        #print(f"After splitting and einsum, dts shape: {dts.shape}, Bs shape: {Bs.shape}, Cs shape: {Cs.shape}")

        xs = xs.view(B, -1, L)
        dts = dts.contiguous().view(B, -1, L)
        Bs = Bs.contiguous()
        Cs = Cs.contiguous()
        
        As = -self.A_logs.float().exp()
        Ds = self.Ds.float()
        dt_projs_bias = torch.zeros_like(Ds).view(-1)
        #print(f"As shape: {As.shape}, Ds shape: {Ds.shape}, dt_projs_bias shape: {dt_projs_bias.shape}")

        to_fp32 = lambda *args: (_a.to(torch.float32) for _a in args)
        xs, dts, Bs, Cs = to_fp32(xs, dts, Bs, Cs)
        #print(f"xs shape: {xs.shape}, dts shape: {dts.shape}, Bs shape: {Bs.shape}, Cs shape: {Cs.shape}")

        out_y = selective_scan_fn(
            xs, dts, As, Bs, Cs, Ds,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
        ).view(B, K, -1, D, H, W)
        #print(f"SS3D forward_core selective_scan_fn output shape: {out_y.shape}")
        
        y = cross_merge_3d_fn(out_y, in_channel_first=True, out_channel_first=True, scans=0, force_torch=True)
        #print(f"SS3D forward_core output shape: {y.shape}")
        y = y.reshape(B, -1, D, H, W)
        return y

"""model = SS3D(d_model=192, channel_first=True).to('cuda:5')
x = torch.randn(16, 320, 192).to('cuda:5')  # (batch, channels, depth, height, width)
output = model(x)
print(output.shape)"""