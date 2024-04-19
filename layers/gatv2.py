import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATv2Conv

class GATv2(torch.nn.Module):
    '''
    `GATv2`: A simple Graph Attention Network model with two layers
        - `data`: PyG data object
        - `hidden_channels`: The number of hidden channels in the GAT layers
        - `dropout`: The dropout probability
        - `activation`: The activation function to use (e.g., `torch.nn.functional.relu`)
        - `heads`: The number of attention heads
    '''

    def __init__(self, data, hidden_channels, dropout=0.5, activation=F.relu, heads=8):
        super(GATv2, self).__init__()
        self.layer1 = GATv2Conv(data.num_features, hidden_channels, heads=heads, dropout=dropout)
        self.layer2 = GATv2Conv(hidden_channels * heads, data.num_classes, heads=1, dropout=dropout)
        self.dropout = dropout
        self.activation = activation

    def forward(self, data):
        # x: node features, edge_index: adjacency list
        x, edge_index = data.x, data.edge_index

        x = self.layer1(x, edge_index)
        x = self.activation(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.layer2(x, edge_index)

        return F.log_softmax(x, dim=1)