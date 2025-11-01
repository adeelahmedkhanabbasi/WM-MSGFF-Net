import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool as gap, GCNConv, AGNNConv
from modules.mamba import mamba
import torch_geometric
import unfoldNd
from modules.WMDC import WaveletMultiDimensionalCollaboration
from modules.AWA import AdaptiveWaveletAttention

class SelfAttention(nn.Module):
    def __init__(self, dim_q, dim_k, dim_v):
        super(SelfAttention, self).__init__()
        self.dim_q = dim_q
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.linear_q = nn.Linear(dim_q, dim_k, bias=False)
        self.linear_k = nn.Linear(dim_q, dim_k, bias=False)
        self.linear_v = nn.Linear(dim_v, dim_v, bias=False)
        self._norm_fact = 1 / math.sqrt(dim_k)

    def forward(self, x):
        batch, n, dim_q = x.shape
        assert dim_q == self.dim_q
        q = self.linear_q(x)
        k = self.linear_k(x)
        v = self.linear_v(x)
        dist = torch.bmm(q, k.transpose(1, 2)) * self._norm_fact
        dist = torch.softmax(dist, dim=-1)
        att = torch.bmm(dist, v)
        return att

class AWMDC(nn.Module):
    def __init__(self, inp=64, channel=256, no_spatial=False):
        """
        AWMDC Module 
        
        Args:
            inp: Input channels for Wavelet Multi-Dimensional Collaboration
            channel: Channel dimension for Adaptive Wavelet Attention  
            no_spatial: Whether to use spatial attention in Wavelet Multi-Dimensional Collaboration
        """
        super(AWMDC, self).__init__()
        
        self.wmc = WaveletMultiDimensionalCollaboration(inp=inp, no_spatial=no_spatial)
        self.awa = AdaptiveWaveletAttention(channel=channel)
        
    def forward(self, x):
        """
        Forward pass with residual connections
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor after Wavelet Multi-Dimensional Collaboration + Adaptive Wavelet Attention with residuals
        """
        # MCA with residual connection
        x = self.wmc(x) + x
        
        # ADA with residual connection  
        x = self.awa(x) + x
        
        return x

class AGNN(nn.Module):
    def __init__(self):
        super(AGNN, self).__init__()
        self.lin1 = torch.nn.Linear(512, 256)
        self.prop2 = AGNNConv(requires_grad=True)

    def forward(self, x, edge_index):
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.lin1(x))
        x = self.prop2(x, edge_index)
        return x

class WM_MSGFF(nn.Module):
    def __init__(self, channels=1, num_classes=2):
        super(WM_MSGFF, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.mamba = mamba(d_model=256,disable_z=False)
        
        # Stage 1
        self.conv_2x2x2_1 = nn.Sequential(
            nn.Conv3d(channels, 16, kernel_size=2, stride=2),
            nn.BatchNorm3d(16))
        self.conv_3x3x3_1 = nn.Sequential(
            nn.Conv3d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm3d(16))
        self.conv_3x3x3_2 = nn.Sequential(
            nn.Conv3d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm3d(16))

        # Stage 2
        self.pool_2x2x2_2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.conv_3x3x3_3 = nn.Sequential(
            nn.Conv3d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32))
        self.conv_3x3x3_4 = nn.Sequential(
            nn.Conv3d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32))

        # Stage 3
        self.pool_2x2x2_3 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.conv_3x3x3_5 = nn.Sequential(
            nn.Conv3d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64))
        self.conv_3x3x3_6 = nn.Sequential(
            nn.Conv3d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64))

        # Stage 4
        self.conv_3x3x3_7 = nn.Sequential(
            nn.Conv3d(128, 128, kernel_size=1),
            nn.BatchNorm3d(128))
        self.conv_3x3x3_8 = nn.Sequential(
            nn.Conv3d(128, 128, kernel_size=1),
            nn.BatchNorm3d(128))

        # Stage 5
        self.conv_3x3x3_9 = nn.Sequential(
            nn.Conv3d(256, 256, kernel_size=1),
            nn.BatchNorm3d(256))
        self.conv_3x3x3_10 = nn.Sequential(
            nn.Conv3d(256, 256, kernel_size=1),
            nn.BatchNorm3d(256))

        self.linear_1 = nn.Linear(1024, 256)  # Adjusted for unfold output (128 * 8)
        
        self.awmdc = AWMDC(inp=64, channel=256, no_spatial=False)

        self.self_attention = SelfAttention(dim_q=256, dim_k=256, dim_v=256)
        self.fc1 = nn.Linear(256, num_classes)
        self.fc2 = nn.Linear(256, num_classes)
        self.gcn_1 = GCNConv(256, 256)
        self.gcn_2 = GCNConv(256, 256)
        self.agnn1 = AGNN()
        self.agnn2 = AGNN()

    def forward(self, x):
        # Input: (b, 1, 80, 128, 128)
        # Stage 1
        out = self.relu(self.conv_2x2x2_1(x))  # (b, 16, 40, 64, 64)
        out_1 = self.relu(self.conv_3x3x3_1(out))
        out_2 = self.conv_3x3x3_2(out_1)
        temp_out = self.relu(out_1 + out_2)
        out_3 = torch.concat([out, temp_out], dim=1)  # (b, 32, 40, 64, 64)

        # Stage 2
        out_4 = self.pool_2x2x2_2(out_3)  # (b, 32, 20, 32, 32)
        out_5 = self.relu(self.conv_3x3x3_3(out_4))
        out_6 = self.conv_3x3x3_4(out_5)
        temp_out = self.relu(out_5 + out_6)
        out_7 = torch.concat([out_4, temp_out], dim=1)  # (b, 64, 20, 32, 32)

        # Stage 3
        out_8 = self.pool_2x2x2_3(out_7)  # (b, 64, 10, 16, 16)
        out_9 = self.relu(self.conv_3x3x3_5(out_8))
        out_10 = self.conv_3x3x3_6(out_9)
        temp_out = self.relu(out_9 + out_10)
        out_11 = torch.concat([out_8, temp_out], dim=1)  # (b, 128, 10, 16, 16)

        # Stage 4
        out_13 = self.relu(self.conv_3x3x3_7(out_11))  # (b, 128, 10, 16, 16)
        out_14 = self.conv_3x3x3_8(out_13)  # (b, 128, 10, 16, 16)
        temp_out = self.relu(out_13 + out_14)
        out_15 = torch.concat([out_11, temp_out], dim=1)  # (b, 256, 10, 16, 16)

        # Stage 5
        out_17 = self.relu(self.conv_3x3x3_9(out_15))  # (b, 256, 10, 16, 16)
        out_18 = self.relu(self.conv_3x3x3_10(out_17))  # (b, 256, 10, 16, 16)
        
        
        out_18 = self.awmdc(out_18)
        # Attention: 2560 patches (10 * 16 * 16)
        out_18_new = out_18.view(out_18.size(0), out_18.size(1), -1)  # (b, 256, 2560)
        out_18_new = out_18_new.transpose(2, 1)  # (b, 2560, 256)
        
        out = out_18_new
        out = self.self_attention(x=out) + out
        out_18_new  =  out 
        out = out.transpose(2, 1).mean(2)  # (b, 256)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)  # (b, 2)
        #########################################
        # SAMSFF 
        #########################################
        # Graph Construction
        # Unfold for graph: (b, 128, 10, 16, 16) -> 320 patches
        unfold1 = unfoldNd.UnfoldNd(kernel_size=2, dilation=1, padding=0, stride=2)
        out_14_unfolded = unfold1(out_14)  # (b, 128*8, 320) where 128*8=1024
        out_14 = torch.permute(out_14_unfolded, (0, 2, 1)).reshape(out_14.shape[0], 320, 128 * 8)  # (b, 320, 1024)
        out_14 = self.linear_1(out_14)  # (b, 320, 256)
    
        final_edge_index_1, final_edge_index_2 = 0, 0
        final_data_1, final_data_2 = 0, 0
        final_batch_1, final_batch_2 = 0, 0
        final_edge_weight_1, final_edge_weight_2 = 0, 0

        for index in range(x.shape[0]):

            patches_1, indices = torch.topk(out_18_new, k=20, dim=1)
            patches_2, indices = torch.topk(out_14, k=20, dim=1)

            patches_1 = patches_1[index]  # (2560, 256)
            patches_2 = patches_2[index]  # (320, 256)

            adj_matrix_1 = torch.cosine_similarity(patches_1.unsqueeze(1), patches_1.unsqueeze(0), dim=-1, eps=1e-08)
            adj_matrix_2 = torch.cosine_similarity(patches_2.unsqueeze(1), patches_2.unsqueeze(0), dim=-1, eps=1e-08)
            adj_matrix_1 = torch.where(adj_matrix_1 >= 0.4, adj_matrix_1, 0)
            adj_matrix_2 = torch.where(adj_matrix_2 >= 0, adj_matrix_2, 0)

            edge_index_1, edge_weight_1 = torch_geometric.utils.dense_to_sparse(adj_matrix_1)
            edge_index_2, edge_weight_2 = torch_geometric.utils.dense_to_sparse(adj_matrix_2)
            edge_index_1 = edge_index_1.to(x.device)
            edge_index_2 = edge_index_2.to(x.device)
            edge_weight_1 = edge_weight_1.to(x.device)
            edge_weight_2 = edge_weight_2.to(x.device)

            if isinstance(final_edge_index_1, int):
                final_edge_index_1 = edge_index_1
                final_edge_index_2 = edge_index_2
                final_edge_weight_1 = edge_weight_1
                final_edge_weight_2 = edge_weight_2
                final_data_1 = patches_1
                final_data_2 = patches_2
                final_batch_1 = edge_index_1.new_zeros(patches_1.shape[0])
                final_batch_2 = edge_index_2.new_zeros(patches_2.shape[0])
            else:
                edge_index_1 = edge_index_1 + final_data_1.shape[0]
                edge_index_2 = edge_index_2 + final_data_2.shape[0]
                final_edge_index_1 = torch.cat((final_edge_index_1, edge_index_1), dim=1)
                final_edge_index_2 = torch.cat((final_edge_index_2, edge_index_2), dim=1)
                final_edge_weight_1 = torch.cat((final_edge_weight_1, edge_weight_1))
                final_edge_weight_2 = torch.cat((final_edge_weight_2, edge_weight_2))
                final_data_1 = torch.cat((final_data_1, patches_1), dim=0)
                final_data_2 = torch.cat((final_data_2, patches_2), dim=0)
                batch_1 = edge_index_1.new_zeros(patches_1.shape[0]) + index
                batch_2 = edge_index_2.new_zeros(patches_2.shape[0]) + index
                final_batch_1 = torch.cat((final_batch_1, batch_1))
                final_batch_2 = torch.cat((final_batch_2, batch_2))

        final_data_1 = F.dropout(final_data_1, p=0.5, training=self.training)
        final_data_2 = F.dropout(final_data_2, p=0.5, training=self.training)
        graph_out_1 = F.relu(self.gcn_1(final_data_1, final_edge_index_1))
        graph_out_2 = F.relu(self.gcn_2(final_data_2, final_edge_index_2))

        graph_out_1 = torch.unsqueeze(graph_out_1, dim=1)
        graph_out_2 = torch.unsqueeze(graph_out_2, dim=1)
        fusion_graph_out = torch.concat([graph_out_1, graph_out_2], dim=1)
        fusion_graph_out = fusion_graph_out.permute(1, 0, 2)
        weighted_graph_out = self.mamba(x=fusion_graph_out)
        weighted_graph_out = weighted_graph_out.permute(1, 0, 2)
        graph_out_1 = weighted_graph_out[:, 0, :]
        graph_out_2 = weighted_graph_out[:, 1, :]

        graph_out = torch.concat([graph_out_1, graph_out_2], dim=-1)
        graph_out1 = self.agnn1(graph_out, final_edge_index_1)
        graph_out2 = self.agnn2(graph_out, final_edge_index_2)
        graph_out1 = gap(graph_out1, batch=final_batch_1)
        graph_out2 = gap(graph_out2, batch=final_batch_2)  # Fixed to use final_batch_2
        graph_out = graph_out1 + graph_out2

        graph_final_out = self.fc2(graph_out.view(x.shape[0], 256))
        return out, graph_final_out
        #return out

# Test the model
if __name__ == "__main__":
    criterion = nn.CrossEntropyLoss()
    model = WM_MSGFF(channels=4).to("cuda:2")
    batch_size = 4
    x = torch.randn(batch_size, 4, 80, 128, 128).to("cuda:2")
    outputs, graph_out = model(x)
    print('outputs:', outputs)
    print('graph_out:', graph_out)