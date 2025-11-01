import torch
import torch.nn as nn
import math
from models.modules.csm_triton import cross_scan_fn, cross_merge_fn
from models.modules.csms6s import selective_scan_fn
#device = 'cuda:1'
class mamba(nn.Module):
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
        disable_z=False,
        k_group = 4,
        dim_head =64, 
        **kwargs,
    ):
        super().__init__()
        self.k_group = k_group  # Using 4 directions for 1D (forward, backward, and their mirrors)
        self.scale = dim_head ** -0.5
        self.d_model = int(d_model)
        self.d_state = int(d_state)
        self.d_inner = int(ssm_ratio * d_model)
        self.dt_rank = int(math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank)
        self.with_dconv = d_conv > 1
        self.disable_z = disable_z
        self.forward = self.forwardv2

        # Linear layers for 1D
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias)
        self.act = act_layer()
        if self.with_dconv:
            self.conv1d = nn.Conv1d(
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

        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        self.A_logs = nn.Parameter(torch.zeros((self.k_group * self.d_inner, self.d_state)))
        self.Ds = nn.Parameter(torch.ones((self.k_group * self.d_inner)))
        self.dt_projs_weight = nn.Parameter(0.1 * torch.rand((self.k_group, self.d_inner, self.dt_rank)))
        self.dt_projs_bias = nn.Parameter(0.1 * torch.rand((self.k_group, self.d_inner)))

    def forwardv2(self, x: torch.Tensor, **kwargs):
        #print(f"mamba forwardv2 input shape: {x.shape}")
        if not self.disable_z:
            x = self.in_proj(x)
        #print(f"After in_proj shape: {x.shape}")
        
        if not self.disable_z:
            x, z = x.chunk(2, dim=-1)
            z = self.act(z)
            #print(f"After chunking, x shape: {x.shape}, z shape: {z.shape}")
        else:
            z = None
            
        if self.with_dconv:
            # For 1D conv, need to permute to (B, C, L) format
            x = x.permute(0, 2, 1)  # (B, L, C) -> (B, C, L)
            x = self.conv1d(x)
            x = x.permute(0, 2, 1)  # Back to (B, L, C)
            #print(f"After conv1d shape: {x.shape}")
            
        x = self.act(x)
        y = self.forward_core(x)
        #print(f"After forward_core shape: {y.shape}")
        
        if z is not None:
            # y is [B, C, L], permute to [B, L, C] to match z's shape [B, L, C]
            y = y.permute(0, 2, 1)  # [16, 192, 64] -> [16, 64, 192]
            y = y * z
            #print(f"After applying z, y shape: {y.shape}")
            # Do NOT permute back; keep [B, L, C] for out_proj
        else:
            y = y.permute(0, 2, 1)  # [16, 192, 64] -> [16, 64, 192]
           
        if not self.disable_z:
            out = self.dropout(self.out_proj(y))
        else:
            out = self.dropout(y)
        #print(f"mamba forwardv2 output shape: {out.shape}")
        return out

    def forward_core(self, x: torch.Tensor):
        #print(f"mamba forward_core input shape: {x.shape}")
        
        B, L, D_inner = x.shape
        
        # Four scanning directions for 1D
        x_lr = torch.stack([x, torch.flip(x, dims=[-2])], dim=1)  # (B, 2, L, D_inner)
        xs = torch.cat([x_lr, x_lr], dim=1)  # (B, 4, L, D_inner)
        #print(f"After stacking and concatenating, xs shape: {xs.shape}")

        x_dbl = torch.einsum("bkld,kcd->bkcl", xs, self.x_proj_weight)
        #print(f"After einsum, x_dbl shape: {x_dbl.shape}")
        
        R = self.dt_rank
        N = self.d_state
        dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
        dts = torch.einsum("bkrl,kdr->bkdl", dts, self.dt_projs_weight)
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
        ).view(B, self.k_group, -1, L)
        #print(f"mamba forward_core selective_scan_fn output shape: {out_y.shape}")
        
        y = cross_merge_fn(out_y, in_channel_first=True, out_channel_first=True, scans=0, force_torch=True)
        #print(f"mamba forward_core output shape: {y.shape}")
        return y

"""# Test
if __name__ == "__main__":
    model = mamba(d_model=192).to(device)
    x = torch.randn(16, 64, 192).to(device)  # (batch, length, channels)
    output = model(x)
    print(f"Final output shape: {output.shape}")"""