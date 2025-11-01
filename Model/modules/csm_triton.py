import torch
import warnings

WITH_TRITON = True
try:
    import triton
    import triton.language as tl
except:
    WITH_TRITON = False
    warnings.warn("Triton not installed, fall back to pytorch implements.")

if WITH_TRITON:
    try:
        from functools import cached_property
    except:
        warnings.warn("if you are using py37, add this line to functools.py: "
            "cached_property = lambda func: property(lru_cache()(func))")

def cross_scan_fwd(x: torch.Tensor, in_channel_first=True, out_channel_first=True, scans=0):
    if in_channel_first:
        B, C, L = x.shape
        if scans == 0:
            y = x.new_empty((B, 4, C, L))
            y[:, 0, :, :] = x
            y[:, 1, :, :] = x
            y[:, 2:4, :, :] = torch.flip(y[:, 0:2, :, :], dims=[-1])
        elif scans == 1:
            y = x.view(B, 1, C, L).repeat(1, 4, 1, 1)
        elif scans == 2:
            y = x.view(B, 1, C, L).repeat(1, 2, 1, 1)
            y = torch.cat([y, y.flip(dims=[-1])], dim=1)
    else:
        B, L, C = x.shape
        if scans == 0:
            y = x.new_empty((B, L, 4, C))
            y[:, :, 0, :] = x
            y[:, :, 1, :] = x
            y[:, :, 2:4, :] = torch.flip(y[:, :, 0:2, :], dims=[1])
        elif scans == 1:
            y = x.view(B, L, 1, C).repeat(1, 1, 4, 1)
        elif scans == 2:
            y = x.view(B, L, 1, C).repeat(1, 1, 2, 1)
            y = torch.cat([y, y.flip(dims=[1])], dim=2)

    if in_channel_first and (not out_channel_first):
        y = y.permute(0, 3, 1, 2).contiguous()
    elif (not in_channel_first) and out_channel_first:
        y = y.permute(0, 2, 3, 1).contiguous()
    return y

def cross_merge_fwd(y: torch.Tensor, in_channel_first=True, out_channel_first=True, scans=0):
    if out_channel_first:
        B, K, C, L = y.shape
        y = y.view(B, K, C, L)
        if scans == 0:
            y = y[:, 0:2] + y[:, 2:4].flip(dims=[-1]).view(B, 2, C, L)
            y = y[:, 0] + y[:, 1]
        elif scans == 1:
            y = y.sum(1)
        elif scans == 2:
            y = y[:, 0:2] + y[:, 2:4].flip(dims=[-1]).view(B, 2, C, L)
            y = y.sum(1)
    else:
        B, L, K, C = y.shape
        y = y.view(B, L, K, C)
        if scans == 0:
            y = y[:, :, 0:2] + y[:, :, 2:4].flip(dims=[1]).view(B, L, 2, C)
            y = y[:, :, 0] + y[:, :, 1]
        elif scans == 1:
            y = y.sum(2)
        elif scans == 2:
            y = y[:, :, 0:2] + y[:, :, 2:4].flip(dims=[1]).view(B, L, 2, C)
            y = y.sum(2)

    if in_channel_first and (not out_channel_first):
        y = y.permute(0, 2, 1).contiguous()
    elif (not in_channel_first) and out_channel_first:
        y = y.permute(0, 2, 1).contiguous()
    return y

def cross_scan1b1_fwd(x: torch.Tensor, in_channel_first=True, out_channel_first=True, scans=0):
    if in_channel_first:
        B, _, C, L = x.shape
        if scans == 0:
            y = torch.stack([
                x[:, 0],
                x[:, 1],
                torch.flip(x[:, 2], dims=[-1]),
                torch.flip(x[:, 3], dims=[-1]),
            ], dim=1)
        elif scans == 1:
            y = x
        elif scans == 2:
            y = torch.stack([
                x[:, 0],
                x[:, 1],
                torch.flip(x[:, 2], dims=[-1]),
                torch.flip(x[:, 3], dims=[-1]),
            ], dim=1)
    else:
        B, L, _, C = x.shape
        if scans == 0:
            y = torch.stack([
                x[:, :, 0],
                x[:, :, 1],
                torch.flip(x[:, :, 2], dims=[1]),
                torch.flip(x[:, :, 3], dims=[1]),
            ], dim=2)
        elif scans == 1:
            y = x
        elif scans == 2:
            y = torch.stack([
                x[:, :, 0],
                x[:, :, 1],
                torch.flip(x[:, :, 2], dims=[1]),
                torch.flip(x[:, :, 3], dims=[1]),
            ], dim=2)

    if in_channel_first and (not out_channel_first):
        y = y.permute(0, 3, 1, 2).contiguous()
    elif (not in_channel_first) and out_channel_first:
        y = y.permute(0, 2, 3, 1).contiguous()
    return y

def cross_merge1b1_fwd(y: torch.Tensor, in_channel_first=True, out_channel_first=True, scans=0):
    if out_channel_first:
        B, K, C, L = y.shape
        y = y.view(B, K, C, L)
        if scans == 0:
            y = torch.stack([
                y[:, 0],
                y[:, 1],
                torch.flip(y[:, 2], dims=[-1]),
                torch.flip(y[:, 3], dims=[-1]),
            ], dim=1)
        elif scans == 1:
            y = y
        elif scans == 2:
            y = torch.stack([
                y[:, 0],
                y[:, 1],
                torch.flip(y[:, 2], dims=[-1]),
                torch.flip(y[:, 3], dims=[-1]),
            ], dim=1)
    else:
        B, L, K, C = y.shape
        y = y.view(B, L, K, C)
        if scans == 0:
            y = torch.stack([
                y[:, :, 0],
                y[:, :, 1],
                torch.flip(y[:, :, 2], dims=[1]),
                torch.flip(y[:, :, 3], dims=[1]),
            ], dim=2)
        elif scans == 1:
            y = y
        elif scans == 2:
            y = torch.stack([
                y[:, :, 0],
                y[:, :, 1],
                torch.flip(y[:, :, 2], dims=[1]),
                torch.flip(y[:, :, 3], dims=[1]),
            ], dim=2)

    if out_channel_first and (not in_channel_first):
        y = y.permute(0, 3, 1, 2).contiguous()
    elif (not out_channel_first) and in_channel_first:
        y = y.permute(0, 2, 3, 1).contiguous()
    return y

class CrossScanF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, in_channel_first=True, out_channel_first=True, one_by_one=False, scans=0):
        ctx.in_channel_first = in_channel_first
        ctx.out_channel_first = out_channel_first
        ctx.one_by_one = one_by_one
        ctx.scans = scans

        if one_by_one:
            B, K, C, L = x.shape
            if not in_channel_first:
                B, L, K, C = x.shape
        else:
            B, C, L = x.shape
            if not in_channel_first:
                B, L, C = x.shape
        ctx.shape = (B, C, L)

        _fn = cross_scan1b1_fwd if one_by_one else cross_scan_fwd
        y = _fn(x, in_channel_first, out_channel_first, scans)
        return y
    
    @staticmethod
    def backward(ctx, ys: torch.Tensor):
        in_channel_first = ctx.in_channel_first
        out_channel_first = ctx.out_channel_first
        one_by_one = ctx.one_by_one
        scans = ctx.scans
        B, C, L = ctx.shape

        ys = ys.view(B, -1, C, L) if out_channel_first else ys.view(B, L, -1, C)
        _fn = cross_merge1b1_fwd if one_by_one else cross_merge_fwd
        y = _fn(ys, in_channel_first, out_channel_first, scans)
        
        if one_by_one:
            y = y.view(B, 4, -1, L) if in_channel_first else y.view(B, L, 4, -1)
        else:
            y = y.view(B, -1, L) if in_channel_first else y.view(B, L, -1)
        return y, None, None, None, None

class CrossMergeF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ys: torch.Tensor, in_channel_first=True, out_channel_first=True, one_by_one=False, scans=0):
        ctx.in_channel_first = in_channel_first
        ctx.out_channel_first = out_channel_first
        ctx.one_by_one = one_by_one
        ctx.scans = scans

        B, K, C, L = ys.shape
        if not out_channel_first:
            B, L, K, C = ys.shape
        ctx.shape = (B, C, L)
        
        _fn = cross_merge1b1_fwd if one_by_one else cross_merge_fwd
        y = _fn(ys, in_channel_first, out_channel_first, scans)
        return y
    
    @staticmethod
    def backward(ctx, x: torch.Tensor):
        in_channel_first = ctx.in_channel_first
        out_channel_first = ctx.out_channel_first
        one_by_one = ctx.one_by_one
        scans = ctx.scans
        B, C, L = ctx.shape
    
        if not one_by_one:
            if in_channel_first:
                x = x.view(B, C, L)
            else:
                x = x.view(B, L, C)
        else:
            if in_channel_first:
                x = x.view(B, 4, C, L)
            else:
                x = x.view(B, L, 4, C)   
                     
        _fn = cross_scan1b1_fwd if one_by_one else cross_scan_fwd
        x = _fn(x, in_channel_first, out_channel_first, scans)
        x = x.view(B, 4, C, L) if out_channel_first else x.view(B, L, 4, C)
        return x, None, None, None, None

@triton.jit
def triton_cross_scan_flex(
    x: tl.tensor,
    y: tl.tensor,
    x_layout: tl.constexpr,
    y_layout: tl.constexpr,
    operation: tl.constexpr,
    onebyone: tl.constexpr,
    scans: tl.constexpr,
    BC: tl.constexpr,
    BL: tl.constexpr,
    DC: tl.constexpr,
    DL: tl.constexpr,
    NL: tl.constexpr,
):
    i_l, i_c, i_b = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    _mask_l = (i_l * BL + tl.arange(0, BL)) < DL
    _for_C = min(DC - i_c * BC, BC)

    pos_l = (i_l * BL + tl.arange(0, BL))
    neg_l = (DL - i_l * BL - 1 - tl.arange(0, BL))
    if scans == 0:
        LRoute0 = pos_l
        LRoute1 = pos_l
        LRoute2 = neg_l
        LRoute3 = neg_l
    elif scans == 1:
        LRoute0 = pos_l
        LRoute1 = LRoute0
        LRoute2 = LRoute0
        LRoute3 = LRoute0
    elif scans == 2:
        LRoute0 = pos_l
        LRoute1 = LRoute0
        LRoute2 = neg_l
        LRoute3 = LRoute2      

    _tmp1 = DC * DL

    y_ptr_base = y + i_b * 4 * _tmp1 + (i_c * BC * DL if y_layout == 0 else i_c * BC)
    if y_layout == 0:
        p_y1 = y_ptr_base + LRoute0
        p_y2 = y_ptr_base + _tmp1 + LRoute1
        p_y3 = y_ptr_base + 2 * _tmp1 + LRoute2
        p_y4 = y_ptr_base + 3 * _tmp1 + LRoute3
    else:
        p_y1 = y_ptr_base + LRoute0 * 4 * DC
        p_y2 = y_ptr_base + DC + LRoute1 * 4 * DC
        p_y3 = y_ptr_base + 2 * DC + LRoute2 * 4 * DC
        p_y4 = y_ptr_base + 3 * DC + LRoute3 * 4 * DC       
    
    if onebyone == 0:
        x_ptr_base = x + i_b * _tmp1 + (i_c * BC * DL if x_layout == 0 else i_c * BC)
        if x_layout == 0:
            p_x = x_ptr_base + LRoute0
        else:
            p_x = x_ptr_base + LRoute0 * DC

        if operation == 0:
            for idxc in range(_for_C):
                _idx_x = idxc * DL if x_layout == 0 else idxc
                _idx_y = idxc * DL if y_layout == 0 else idxc
                _x = tl.load(p_x + _idx_x, mask=_mask_l)
                tl.store(p_y1 + _idx_y, _x, mask=_mask_l)
                tl.store(p_y2 + _idx_y, _x, mask=_mask_l)
                tl.store(p_y3 + _idx_y, _x, mask=_mask_l)
                tl.store(p_y4 + _idx_y, _x, mask=_mask_l)
        elif operation == 1:
            for idxc in range(_for_C):
                _idx_x = idxc * DL if x_layout == 0 else idxc
                _idx_y = idxc * DL if y_layout == 0 else idxc
                _y1 = tl.load(p_y1 + _idx_y, mask=_mask_l)
                _y2 = tl.load(p_y2 + _idx_y, mask=_mask_l)
                _y3 = tl.load(p_y3 + _idx_y, mask=_mask_l)
                _y4 = tl.load(p_y4 + _idx_y, mask=_mask_l)
                tl.store(p_x + _idx_x, _y1 + _y2 + _y3 + _y4, mask=_mask_l)
    else:
        x_ptr_base = x + i_b * 4 * _tmp1 + (i_c * BC * DL if x_layout == 0 else i_c * BC)
        if x_layout == 0:
            p_x1 = x_ptr_base + LRoute0
            p_x2 = p_x1 + _tmp1
            p_x3 = p_x2 + _tmp1
            p_x4 = p_x3 + _tmp1  
        else:
            p_x1 = x_ptr_base + LRoute0 * 4 * DC
            p_x2 = p_x1 + DC
            p_x3 = p_x2 + DC
            p_x4 = p_x3 + DC        
    
        if operation == 0:
            for idxc in range(_for_C):
                _idx_x = idxc * DL if x_layout == 0 else idxc
                _idx_y = idxc * DL if y_layout == 0 else idxc
                tl.store(p_y1 + _idx_y, tl.load(p_x1 + _idx_x, mask=_mask_l), mask=_mask_l)
                tl.store(p_y2 + _idx_y, tl.load(p_x2 + _idx_x, mask=_mask_l), mask=_mask_l)
                tl.store(p_y3 + _idx_y, tl.load(p_x3 + _idx_x, mask=_mask_l), mask=_mask_l)
                tl.store(p_y4 + _idx_y, tl.load(p_x4 + _idx_x, mask=_mask_l), mask=_mask_l)
        else:
            for idxc in range(_for_C):
                _idx_x = idxc * DL if x_layout == 0 else idxc
                _idx_y = idxc * DL if y_layout == 0 else idxc
                tl.store(p_x1 + _idx_x, tl.load(p_y1 + _idx_y), mask=_mask_l)
                tl.store(p_x2 + _idx_x, tl.load(p_y2 + _idx_y), mask=_mask_l)
                tl.store(p_x3 + _idx_x, tl.load(p_y3 + _idx_y), mask=_mask_l)
                tl.store(p_x4 + _idx_x, tl.load(p_y4 + _idx_y), mask=_mask_l)

class CrossScanTritonF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, in_channel_first=True, out_channel_first=True, one_by_one=False, scans=0):
        if one_by_one:
            if in_channel_first:
                B, _, C, L = x.shape
            else:
                B, L, _, C = x.shape
        else:
            if in_channel_first:
                B, C, L = x.shape
            else:
                B, L, C = x.shape
        B, C, L = int(B), int(C), int(L)
        BC, BL = 1, 32
        NL, NC = triton.cdiv(L, BL), triton.cdiv(C, BC)
        
        ctx.in_channel_first = in_channel_first
        ctx.out_channel_first = out_channel_first
        ctx.one_by_one = one_by_one
        ctx.scans = scans
        ctx.shape = (B, C, L)
        ctx.triton_shape = (BC, BL, NC, NL)

        y = x.new_empty((B, 4, C, L)) if out_channel_first else x.new_empty((B, L, 4, C))
        triton_cross_scan_flex[(NL, NC, B)](
            x.contiguous(), y, 
            (0 if in_channel_first else 1), (0 if out_channel_first else 1), 0, (0 if not one_by_one else 1), scans, 
            BC, BL, C, L, NL
        )
        return y
        
    @staticmethod
    def backward(ctx, y: torch.Tensor):
        in_channel_first = ctx.in_channel_first
        out_channel_first = ctx.out_channel_first
        one_by_one = ctx.one_by_one
        scans = ctx.scans
        B, C, L = ctx.shape
        BC, BL, NC, NL = ctx.triton_shape
        if one_by_one:
            x = y.new_empty((B, 4, C, L)) if in_channel_first else y.new_empty((B, L, 4, C))
        else:
            x = y.new_empty((B, C, L)) if in_channel_first else y.new_empty((B, L, C))
        
        triton_cross_scan_flex[(NL, NC, B)](
            x, y.contiguous(), 
            (0 if in_channel_first else 1), (0 if out_channel_first else 1), 1, (0 if not one_by_one else 1), scans,
            BC, BL, C, L, NL
        )
        return x, None, None, None, None

class CrossMergeTritonF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, y: torch.Tensor, in_channel_first=True, out_channel_first=True, one_by_one=False, scans=0):
        if out_channel_first:
            B, _, C, L = y.shape
        else:
            B, L, _, C = y.shape
        B, C, L = int(B), int(C), int(L)
        BC, BL = 1, 32
        NL, NC = triton.cdiv(L, BL), triton.cdiv(C, BC)
        ctx.in_channel_first = in_channel_first
        ctx.out_channel_first = out_channel_first
        ctx.one_by_one = one_by_one
        ctx.scans = scans
        ctx.shape = (B, C, L)
        ctx.triton_shape = (BC, BL, NC, NL)
        if one_by_one:
            x = y.new_empty((B, 4, C, L)) if in_channel_first else y.new_empty((B, L, 4, C))
        else:
            x = y.new_empty((B, C, L)) if in_channel_first else y.new_empty((B, L, C))
        triton_cross_scan_flex[(NL, NC, B)](
            x, y.contiguous(), 
            (0 if in_channel_first else 1), (0 if out_channel_first else 1), 1, (0 if not one_by_one else 1), scans,
            BC, BL, C, L, NL
        )
        return x
        
    @staticmethod
    def backward(ctx, x: torch.Tensor):
        in_channel_first = ctx.in_channel_first
        out_channel_first = ctx.out_channel_first
        one_by_one = ctx.one_by_one
        scans = ctx.scans
        B, C, L = ctx.shape
        BC, BL, NC, NL = ctx.triton_shape
        y = x.new_empty((B, 4, C, L)) if out_channel_first else x.new_empty((B, L, 4, C))
        triton_cross_scan_flex[(NL, NC, B)](
            x.contiguous(), y, 
            (0 if in_channel_first else 1), (0 if out_channel_first else 1), 0, (0 if not one_by_one else 1), scans,
            BC, BL, C, L, NL
        )
        return y, None, None, None, None, None

def cross_scan_fn(x: torch.Tensor, in_channel_first=True, out_channel_first=True, one_by_one=False, scans=0, force_torch=False):
    CSF = CrossScanTritonF if WITH_TRITON and x.is_cuda and (not force_torch) else CrossScanF
    with torch.cuda.device(x.device):
        return CSF.apply(x, in_channel_first, out_channel_first, one_by_one, scans)

def cross_merge_fn(y: torch.Tensor, in_channel_first=True, out_channel_first=True, one_by_one=False, scans=0, force_torch=False):
    CMF = CrossMergeTritonF if WITH_TRITON and y.is_cuda and (not force_torch) else CrossMergeF
    with torch.cuda.device(y.device):
        return CMF.apply(y, in_channel_first, out_channel_first, one_by_one, scans)

class CHECK:
    def check_csm_triton():
        B, C, L = 256, 192, 56 * 57
        dtype = torch.float32
        x = torch.randn((B, C, L), dtype=dtype, device=torch.device("cuda")).requires_grad_(True)
        y = torch.randn((B, 4, C, L), dtype=dtype, device=torch.device("cuda")).requires_grad_(True)
        x1 = x.clone().detach().requires_grad_(True)
        y1 = y.clone().detach().requires_grad_(True)

        def cross_scan(x: torch.Tensor):
            B, C, L = x.shape
            xs = torch.stack([
                x,
                x,
                torch.flip(x, dims=[-1]),
                torch.flip(x, dims=[-1]),
            ], dim=1).view(B, 4, C, L)
            return xs
        
        def cross_merge(out_y: torch.Tensor):
            B, K, C, L = out_y.shape
            inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, C, L)
            y = out_y[:, 0:2] + inv_y
            return y.sum(1)

        def cross_scan_1b1(x: torch.Tensor):
            B, K, C, L = x.shape
            xs = torch.stack([
                x[:, 0],
                x[:, 1],
                torch.flip(x[:, 2], dims=[-1]),
                torch.flip(x[:, 3], dims=[-1]),
            ], dim=1).view(B, 4, C, L)
            return xs
        
        def unidi_scan(x):
            B, C, L = x.shape
            x = x.view(B, 1, C, L).repeat(1, 4, 1, 1)
            return x
        
        def unidi_merge(ys):
            B, K, C, L = ys.shape
            return ys.view(B, 4, C, L).sum(1)

        def bidi_scan(x):
            B, C, L = x.shape
            x = x.view(B, 1, C, L).repeat(1, 2, 1, 1)
            x = torch.cat([x, x.flip(dims=[-1])], dim=1)
            return x
        
        def bidi_merge(ys):
            B, K, C, L = ys.shape
            ys = ys[:, 0:2] + ys[:, 2:4].flip(dims=[-1]).view(B, 2, C, L)
            return ys.sum(1)

        if True:
            res0 = triton.testing.do_bench(lambda :cross_scan(x))
            res1 = triton.testing.do_bench(lambda :cross_scan_fn(x, True, True, False))
            res3 = triton.testing.do_bench(lambda :cross_merge(y))
            res4 = triton.testing.do_bench(lambda :cross_merge_fn(y, True, True, False))
            print(res0, res1, res3, res4)
            res0 = triton.testing.do_bench(lambda :cross_scan(x).sum().backward())
            res1 = triton.testing.do_bench(lambda :cross_scan_fn(x, True, True, False).sum().backward())
            res3 = triton.testing.do_bench(lambda :cross_merge(y).sum().backward())
            res4 = triton.testing.do_bench(lambda :cross_merge_fn(y, True, True, False).sum().backward())
            print(res0, res1, res3, res4)

        print("test cross scan")
        for (cs0, cm0, cs1, cm1) in [
            (cross_scan, cross_merge, cross_scan_fn, cross_merge_fn),
            (unidi_scan, unidi_merge, lambda x: cross_scan_fn(x, scans=1), lambda x: cross_merge_fn(x, scans=1)),
            (bidi_scan, bidi_merge, lambda x: cross_scan_fn(x, scans=2), lambda x: cross_merge_fn(x, scans=2)),
            (cross_scan, cross_merge, lambda x: cross_scan_fn(x.permute(0, 2, 1), in_channel_first=False), lambda x: cross_merge_fn(x, in_channel_first=False).permute(0, 2, 1)),
            (cross_scan, cross_merge, lambda x: cross_scan_fn(x, out_channel_first=False).permute(0, 2, 3, 1), lambda x: cross_merge_fn(x.permute(0, 3, 1, 2), out_channel_first=False)),
            (cross_scan, cross_merge, lambda x: cross_scan_fn(x.permute(0, 2, 1), in_channel_first=False, out_channel_first=False).permute(0, 2, 3, 1), lambda x: cross_merge_fn(x.permute(0, 3, 1, 2), in_channel_first=False, out_channel_first=False).permute(0, 2, 1)),
        ]:
            x.grad, x1.grad, y.grad, y1.grad = None, None, None, None
            o0 = cs0(x)
            o1 = cs1(x1)
            o0.backward(y.view(B, 4, C, L))
            o1.backward(y.view(B, 4, C, L))
            print((o0 - o1).abs().max())
            print((x.grad - x1.grad).abs().max())
            o0 = cm0(y)
            o1 = cm1(y1)
            o0.backward(x.view(B, C, L))
            o1.backward(x.view(B, C, L))
            print((o0 - o1).abs().max())
            print((y.grad - y1.grad).abs().max())
            x.grad, x1.grad, y.grad, y1.grad = None, None, None, None
            print("===============", flush=True)

        print("test cross scan one by one")
        for (cs0, cs1) in [
            (cross_scan_1b1, lambda x: cross_scan_fn(x, one_by_one=True)),
        ]:
            o0 = cs0(y)
            o1 = cs1(y1)
            o0.backward(y.view(B, 4, C, L))
            o1.backward(y.view(B, 4, C, L))
            print((o0 - o1).abs().max())
            print((y.grad - y1.grad).abs().max())
            x.grad, x1.grad, y.grad, y1.grad = None, None, None, None
            print("===============", flush=True)

    def check_csm_scan3():
        B, C, L = 27, 253, 57 * 58
        x = torch.randn((B, C, L)).cuda()

        for scans in [0, 1, 2, 3]:
            o1 = cross_scan_fn(x, scans=scans, force_torch=True).view(B, 4, C, L)
            print((cross_scan_fn(x, scans=scans) == cross_scan_fn(x, scans=scans, force_torch=True)).all())
            print((cross_merge_fn(o1, scans=scans) == cross_merge_fn(o1, scans=scans, force_torch=True)).all())

            kwargs = dict(in_channel_first=False, out_channel_first=False)
            x2 = x.permute(0, 2, 1).contiguous()
            o2 = o1.permute(0, 3, 1, 2).contiguous()
            print((cross_scan_fn(x, scans=scans, **kwargs) == cross_scan_fn(x, scans=scans, force_torch=True, **kwargs)).all())
            print((cross_merge_fn(o2, scans=scans, **kwargs) == cross_merge_fn(o2, scans=scans, force_torch=True, **kwargs)).all())            

        breakpoint()

if __name__ == "__main__":
    CHECK.check_csm_scan3()
    CHECK.check_csm_triton()