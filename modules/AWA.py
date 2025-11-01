import torch
import torch.nn as nn
import math

class Mix(nn.Module):
    def __init__(self, m=-0.80):
        super(Mix, self).__init__()
        w = torch.nn.Parameter(torch.FloatTensor([m]), requires_grad=True)
        self.w = torch.nn.Parameter(w, requires_grad=True)
        self.mix_block = nn.Sigmoid()

    def forward(self, fea1, fea2):
        mix_factor = self.mix_block(self.w)
        out = fea1 * mix_factor.expand_as(fea1) + fea2 * (1 - mix_factor.expand_as(fea2))
        return out

class AdaptiveWaveletAttention(nn.Module):
    def __init__(self, channel, b=1, gamma=2):
        super(AdaptiveWaveletAttention, self).__init__()
        # 3D adaptive average pooling
        self.avg_pool = nn.AdaptiveAvgPool3d(1)  # Pool over depth, height, width
        # 1D convolution for channel attention
        t = int(abs((math.log(channel, 2) + b) / gamma))
        k = t if t % 2 else t + 1
        self.conv1 = nn.Conv1d(1, 1, kernel_size=k, padding=int(k / 2), bias=False)
        # 3D convolution for fully connected layer
        self.fc = nn.Conv3d(channel, channel, 1, padding=0, bias=True)
        self.sigmoid = nn.Sigmoid()
        self.mix = Mix()

    def forward(self, input):
        # Input shape: (batch, channel, depth, height, width), e.g., (1, 64, 10, 16, 16)
        x = self.avg_pool(input)  # Shape: (batch, channel, 1, 1, 1), e.g., (1, 64, 1, 1, 1)
        
        # Compute x1: Reshape for Conv1d
        batch_size, channel = x.size(0), x.size(1)
        x1 = x.view(batch_size, channel, -1)  # Shape: (batch, channel, 1), e.g., (1, 64, 1)
        x1 = x1.transpose(1, 2)  # Shape: (batch, 1, channel), e.g., (1, 1, 64)
        x1 = self.conv1(x1)  # Shape: (batch, 1, channel), e.g., (1, 1, 64)
        
        # Compute x2: Apply fc and reshape
        x2 = self.fc(x)  # Shape: (batch, channel, 1, 1, 1), e.g., (1, 64, 1, 1, 1)
        x2 = x2.view(batch_size, channel, -1)  # Shape: (batch, channel, 1), e.g., (1, 64, 1)
        x2 = x2.transpose(1, 2)  # Shape: (batch, 1, channel), e.g., (1, 1, 64)
        
        # Compute out1: Attention weights via matrix multiplication
        out1 = torch.matmul(x1.transpose(-1, -2), x2)  # Shape:,,,,,,,,, (batch, channel, channel), e.g., (1, 64, 64)
        out1 = self.sigmoid(out1)  # Shape: (batch, channel, channel)
        out1 = out1.mean(dim=-1, keepdim=True)  # Shape: (batch, channel, 1), e.g., (1, 64, 1)
        out1 = out1.unsqueeze(-1).unsqueeze(-1)  # Shape: (batch, channel, 1, 1, 1), e.g., (1, 64, 1, 1, 1)
        
        # Compute out2: Second attention component
        out2 = torch.matmul(x2.transpose(-1, -2), x1)  # Shape: (batch, channel, channel), e.g., (1, 64, 64)
        out2 = self.sigmoid(out2)  # Shape: (batch, channel, channel)
        out2 = out2.mean(dim=-1, keepdim=True)  # Shape: (batch, channel, 1), e.g., (1, 64, 1)
        out2 = out2.unsqueeze(-1).unsqueeze(-1)  # Shape: (batch, channel, 1, 1, 1), e.g., (1, 64, 1, 1, 1)
        
        # Mix out1 and out2
        out = self.mix(out1, out2)  # Shape: (batch, channel, 1, 1, 1), e.g., (1, 64, 1, 1, 1)
        
        # Apply conv1 to reduce channel dimension
        out = out.view(batch_size, channel, -1)  # Shape: (batch, channel, 1), e.g., (1, 64, 1)
        out = out.transpose(1, 2)  # Shape: (batch, 1, channel), e.g., (1, 1, 64)
        out = self.conv1(out)  # Shape: (batch, 1, channel), e.g., (1, 1, 64)
        out = out.transpose(1, 2).unsqueeze(-1).unsqueeze(-1)  # Shape: (batch, channel, 1, 1, 1), e.g., (1, 64, 1, 1, 1)
        out = self.sigmoid(out)  # Shape: (batch, channel, 1, 1, 1)
        
        # Apply attention weights to input
        return input * out  # Broadcasting: (1, 64, 10, 16, 16) * (1, 64, 1, 1, 1) -> (1, 64, 10, 16, 16)

"""if __name__ == '__main__':
    input = torch.rand(1, 256, 5, 8, 8)  # 3D input
    A = AdaptiveWaveletAttention(channel=256)
    y = A(input)
    print(y.size())  # Expected output: torch.Size([1, 64, 10, 16, 16])"""