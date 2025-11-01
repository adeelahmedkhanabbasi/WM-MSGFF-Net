import torch
import warnings

WITH_TRITON = True
try:
    import triton
    import triton.language as tl
except:
    WITH_TRITON = False
    warnings.warn("Triton not installed, fall back to PyTorch implements.")

# Torch Implementation for 3D ========================================
def cross_scan_3d_fwd(x: torch.Tensor, in_channel_first=True, out_channel_first=True, scans=0):
    if in_channel_first:
        B, C, D, H, W = x.shape
        L = D * H * W
        if scans == 0:  # Six directions: D, H, W, and their reverses
            y = x.new_empty((B, 6, C, L))
            y[:, 0, :, :] = x.flatten(2, 4)  # Along D
            y[:, 1, :, :] = x.transpose(2, 3).flatten(2, 4)  # Along H
            y[:, 2, :, :] = x.transpose(2, 4).flatten(2, 4)  # Along W
            y[:, 3:6, :, :] = torch.flip(y[:, 0:3, :, :], dims=[-1])  # Reverse directions
        elif scans == 1:  # Unidirectional (repeat)
            y = x.view(B, 1, C, L).repeat(1, 6, 1, 1)
        elif scans == 2:  # Bidirectional (three directions + reverses)
            y = x.view(B, 1, C, L).repeat(1, 3, 1, 1)
            y = torch.cat([y, y.flip(dims=[-1])], dim=1)
    else:
        B, D, H, W, C = x.shape
        L = D * H * W
        if scans == 0:
            y = x.new_empty((B, L, 6, C))
            y[:, :, 0, :] = x.flatten(1, 3)
            y[:, :, 1, :] = x.transpose(1, 2).flatten(1, 3)
            y[:, :, 2, :] = x.transpose(1, 3).flatten(1, 3)
            y[:, :, 3:6, :] = torch.flip(y[:, :, 0:3, :], dims=[1])
        elif scans == 1:
            y = x.view(B, L, 1, C).repeat(1, 1, 6, 1)
        elif scans == 2:
            y = x.view(B, L, 1, C).repeat(1, 1, 3, 1)
            y = torch.cat([y, y.flip(dims=[1])], dim=2)

    if in_channel_first and not out_channel_first:
        y = y.permute(0, 3, 1, 2).contiguous()
    elif not in_channel_first and out_channel_first:
        y = y.permute(0, 2, 3, 1).contiguous()

    return y


def cross_merge_3d_fwd(y: torch.Tensor, in_channel_first=True, out_channel_first=True, scans=0):
    if out_channel_first:
        B, K, C, D, H, W = y.shape
        L = D * H * W
        y = y.view(B, K, C, L)
        if scans == 0:
            y_forward = y[:, 0:3]  # D, H, W directions
            y_reverse = y[:, 3:6].flip(dims=[-1]).view(B, 3, C, L)
            y = y_forward + y_reverse  # Shape: (B, 3, C, L)
            # Reshape and reorient each direction to (B, C, D, H, W)
            y_d = y[:, 0].view(B, C, D, H, W)  # Along D: (D, H, W)
            y_h = y[:, 1].view(B, C, H, D, W).permute(0, 1, 3, 2, 4).contiguous()  # (H, D, W) -> (D, H, W)
            y_w = y[:, 2].view(B, C, W, H, D).permute(0, 1, 4, 2, 3).contiguous()  # (W, H, D) -> (D, H, W)
            y = (y_d + y_h + y_w).view(B, C, L)
        elif scans == 1:
            y = y.sum(1)
        elif scans == 2:
            y_forward = y[:, 0:3]
            y_reverse = y[:, 3:6].flip(dims=[-1]).view(B, 3, C, L)
            y = (y_forward + y_reverse).sum(1)
    else:
        B, D, H, W, K, C = y.shape
        L = D * H * W
        y = y.view(B, L, K, C)
        if scans == 0:
            y_forward = y[:, :, 0:3]
            y_reverse = y[:, :, 3:6].flip(dims=[1]).view(B, L, 3, C)
            y = y_forward + y_reverse
            y_d = y[:, :, 0].view(B, D, H, W, C)  # Along D
            y_h = y[:, :, 1].view(B, H, D, W, C).permute(0, 2, 1, 3, 4).contiguous()  # (H, D, W) -> (D, H, W)
            y_w = y[:, :, 2].view(B, W, H, D, C).permute(0, 3, 2, 1, 4).contiguous()  # (W, H, D) -> (D, H, W)
            y = (y_d + y_h + y_w).view(B, L, C)
        elif scans == 1:
            y = y.sum(2)
        elif scans == 2:
            y_forward = y[:, :, 0:3]
            y_reverse = y[:, :, 3:6].flip(dims=[1]).view(B, L, 3, C)
            y = (y_forward + y_reverse).sum(2)

    if in_channel_first and not out_channel_first:
        y = y.permute(0, 2, 1).contiguous()
    elif not in_channel_first and out_channel_first:
        y = y.permute(0, 2, 1).contiguous()

    return y


class CrossScan3DF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, in_channel_first=True, out_channel_first=True, scans=0):
        ctx.in_channel_first = in_channel_first
        ctx.out_channel_first = out_channel_first
        ctx.scans = scans
        B, C, D, H, W = x.shape if in_channel_first else x.shape[:-1]
        ctx.shape = (B, C, D, H, W)
        y = cross_scan_3d_fwd(x, in_channel_first, out_channel_first, scans)
        return y

    @staticmethod
    def backward(ctx, ys: torch.Tensor):
        in_channel_first = ctx.in_channel_first
        out_channel_first = ctx.out_channel_first
        scans = ctx.scans
        B, C, D, H, W = ctx.shape
        ys = ys.view(B, -1, C, D, H, W) if out_channel_first else ys.view(B, D, H, W, -1, C)
        y = cross_merge_3d_fwd(ys, in_channel_first, out_channel_first, scans)
        return y.view(B, C, D, H, W) if in_channel_first else y.view(B, D, H, W, C), None, None, None


class CrossMerge3DF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ys: torch.Tensor, in_channel_first=True, out_channel_first=True, scans=0):
        ctx.in_channel_first = in_channel_first
        ctx.out_channel_first = out_channel_first
        ctx.scans = scans
        B, K, C, D, H, W = ys.shape if out_channel_first else (ys.shape[0], ys.shape[3], ys.shape[4], ys.shape[1], ys.shape[2], ys.shape[5])
        ctx.shape = (B, C, D, H, W)
        y = cross_merge_3d_fwd(ys, in_channel_first, out_channel_first, scans)
        return y

    @staticmethod
    def backward(ctx, x: torch.Tensor):
        in_channel_first = ctx.in_channel_first
        out_channel_first = ctx.out_channel_first
        scans = ctx.scans
        B, C, D, H, W = ctx.shape
        x = x.view(B, C, D, H, W) if in_channel_first else x.view(B, D, H, W, C)
        y = cross_scan_3d_fwd(x, in_channel_first, out_channel_first, scans)
        return y.view(B, 6, C, D, H, W) if out_channel_first else y.view(B, D, H, W, 6, C), None, None, None


def cross_scan_3d_fn(x: torch.Tensor, in_channel_first=True, out_channel_first=True, scans=0, force_torch=False):
    CSF = CrossScan3DF  # Triton version can be added similarly if needed
    with torch.cuda.device(x.device):
        return CSF.apply(x, in_channel_first, out_channel_first, scans)


def cross_merge_3d_fn(y: torch.Tensor, in_channel_first=True, out_channel_first=True, scans=0, force_torch=False):
    CMF = CrossMerge3DF
    with torch.cuda.device(y.device):
        return CMF.apply(y, in_channel_first, out_channel_first, scans)


# Note: Triton implementation for 3D would require updating triton_cross_scan_flex
# to handle 3D indices (D, H, W) and six scanning directions.