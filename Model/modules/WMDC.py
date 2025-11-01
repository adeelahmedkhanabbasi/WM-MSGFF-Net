import torch
from torch import nn
import math
from models.wtcnn3D import WTConv3d
__all__ = ['WaveletMultiDimensionalCollaboration', 'MCAGate3D']

class StdPool3D(nn.Module):
    def __init__(self):
        super(StdPool3D, self).__init__()

    def forward(self, x):
        b, c, d, h, w = x.size()
        std = x.view(b, c, -1).std(dim=2, keepdim=True)
        return std.reshape(b, c, 1, 1, 1)

class MCAGate3D(nn.Module):
    def __init__(self, k_size, pool_types=['avg']):
        super(MCAGate3D, self).__init__()
        self.pools = nn.ModuleList()
        for pool_type in pool_types:
            if pool_type == 'avg':
                self.pools.append(nn.AdaptiveAvgPool3d(1))
            elif pool_type == 'max':
                self.pools.append(nn.AdaptiveMaxPool3d(1))
            elif pool_type == 'std':
                self.pools.append(StdPool3D())
            else:
                raise NotImplementedError

        self.conv= WTConv3d(in_channels = 1, out_channels=1, kernel_size=(1,1,k_size), wt_levels=3)
        self.sigmoid = nn.Sigmoid()
        self.weight = nn.Parameter(torch.rand(2))

    def forward(self, x):
        feats = [pool(x) for pool in self.pools]
        
        if len(feats) == 1:
            out = feats[0]
        elif len(feats) == 2:
            weight = torch.sigmoid(self.weight)
            out = (feats[0] + feats[1])/2 + weight[0]*feats[0] + weight[1]*feats[1]
        else:
            raise ValueError("Only 1 or 2 pool types supported")

        # Channel convolution
        out = out.permute(0,4,3,2,1).contiguous()  # [B,1,1,1,C]
        out = self.conv(out)
        out = out.permute(0,4,3,2,1).contiguous()  # [B,C,1,1,1]
        
        out = self.sigmoid(out)
        return x * out.expand_as(x)

class WaveletMultiDimensionalCollaboration(nn.Module):
    def __init__(self, inp, no_spatial=False):
        super(WaveletMultiDimensionalCollaboration, self).__init__()
        self.no_spatial = no_spatial

        # Kernel size calculation for channel gate
        lambd = 1.5
        gamma = 1
        temp = round(abs((math.log2(inp) - gamma) / lambd))
        kernel = temp if temp%2 else temp-1

        # Spatial gates
        self.d_gate = MCAGate3D(3)
        self.h_gate = MCAGate3D(3)
        self.w_gate = MCAGate3D(3)
        
        # Channel gate
        if not no_spatial:
            self.c_gate = MCAGate3D(kernel)

    def forward(self, x):
        # Depth gate
        x_d = x.permute(0,2,1,3,4).contiguous()  # [B,D,C,H,W]
        x_d = self.d_gate(x_d).permute(0,2,1,3,4).contiguous()

        # Height gate 
        x_h = x.permute(0,3,1,2,4).contiguous()  # [B,H,C,D,W]
        x_h = self.h_gate(x_h).permute(0,2,3,1,4).contiguous()

        # Width gate
        x_w = x.permute(0,4,1,2,3).contiguous()  # [B,W,C,D,H]
        x_w = self.w_gate(x_w).permute(0,2,3,4,1).contiguous()

        # Combine features
        if not self.no_spatial:
            x_c = self.c_gate(x)
            return (x_d + x_h + x_w + x_c) / 4
        else:
            return (x_d + x_h + x_w) / 3

"""# Example usage
if __name__ == "__main__":
    x = torch.randn(1, 256, 10, 16, 16)
    mca = WaveletMultiDimensionalCollaboration(inp=64, no_spatial=False)
    output = mca(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")  # Should be (1,64,32,32,32)"""