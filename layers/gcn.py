import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

class GCN(torch.nn.Module):
    '''
    `GCN`: A simple Graph Convolutional Network model with two layers
        - `data`: PyG data object
        - `hidden_channels`: The number of hidden channels in the GCN layers
        - `dropout`: The dropout probability
        - `activation`: The activation function to use (e.g., `torch.nn.functional.relu`)
    '''

    def __init__(self, data, hidden_channels, dropout=0.5, activation=F.relu):
        super(GCN, self).__init__()
        self.layer1 = GCNConv(data.num_features, hidden_channels)
        self.layer2 = GCNConv(hidden_channels, data.num_classes)
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