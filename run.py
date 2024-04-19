import random, os, time, logging
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.transforms import NormalizeFeatures
from utils.graph_loader import GraphLoader
from utils.graph_visualizer import GraphVisualizer
from training import train_k_fold

# Set random seed for reproducibility
def seed_everything(seed: int=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Set the device to GPU if available, otherwise to CPU
def set_device():
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    return device

# Return the activation function based on the input string
def activation_fn(activation: str='relu') -> torch.nn.functional:
    activation = activation.lower()
    if activation == 'relu':
        activation = F.relu
    elif activation == 'selu':
        activation = F.selu
    elif activation == 'elu':
        activation = F.elu
    elif activation == 'leaky_relu':
        activation = F.leaky_relu
    elif activation == 'prelu':
        activation = F.prelu
    else:
        raise ValueError(f"Activation function {activation} is not supported. Choose from [relu, selu, elu, leaky_relu, prelu]")
    return activation

class DataLoader:
    '''
    `DataLoader`: A class to load graph data from node and edge files
        - `file_nodes`: The path to the file containing the nodes data (e.g., cora.content)
        - `file_nodes`: The path to the file containing the nodes data (e.g., cora.content)
        - `undirected`: Whether to convert the graph to undirected
    '''

    def __init__(self, file_nodes: str, file_edges: str, undirected: bool=True):
        # Load the graph data from the files
        self.loader = GraphLoader(file_nodes=file_nodes, file_edges=file_edges)
        # Convert the graph to undirected if specified
        if undirected:
            self.loader.to_undirected()
        # Get the features (dict), labels (dict), and edge index (list)
        self.features, self.labels, self.edge_index = self.loader.get_data()
        # Print information about the graph
        self.loader.print_info()
        # A dictionary mapping node IDs to indices
        self.id_to_idx = None
        # A dictionary mapping indices to node IDs
        self.idx_to_id = None
        # A dictionary mapping labels to indices
        self.label_to_id = None
        # A dictionary mapping indices to labels
        self.id_to_label = None

    def get_data(self, row_normalize: bool=True, device: torch.device=torch.device('cpu')) -> Data:
        '''
        `get_data`: A function to convert the graph data to a PyG Data object
            - `row_normalize`: Whether to normalize the feature vectors to have sum of 1
            - `device`: The device to use (e.g., `torch.device('cuda')`)
        '''
        # Make a DataFrame from the node IDs and labels
        df = pd.DataFrame({'Node ID': list(self.features.keys()), 'Label': list(self.labels.values())})
        self.id_to_idx = {node_id: idx for idx, node_id in enumerate(df['Node ID'])}
        self.idx_to_id = {idx: node_id for idx, node_id in enumerate(df['Node ID'])}
        # Convert the node IDs in the edge list to indices
        edge_index = np.vectorize(lambda x: self.id_to_idx[x])(self.edge_index)
        # Convert features to tensor with shape (num_nodes, num_features)
        features_tensor = torch.tensor([self.features[key] for key in self.features.keys()], dtype=torch.float)
        # Find the unique labels
        unique_labels = set(self.labels.values())
        self.label_to_id = {label: i for i, label in enumerate(unique_labels)}
        self.id_to_label = {i: label for label, i in self.label_to_id.items()}
        # Convert labels to tensor with shape (num_nodes,)
        labels_tensor = torch.tensor([self.label_to_id[self.labels[key]] for key in self.labels.keys()], dtype=torch.long)
        # Convert edge list to tensor with shape (2, num_edges)
        edge_index_tensor = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        # Create a PyG Data object
        data = Data(x=features_tensor, y=labels_tensor, edge_index=edge_index_tensor)
        # Normalize the feature vectors to have sum of 1 if row_normalize is True
        if row_normalize:
            data = NormalizeFeatures()(data)
        # Add the number of classes to the data object
        data.num_classes = len(unique_labels)
        # Move the data object to the device (CPU or GPU)
        data.to(device)
        return data

# Main function to parse arguments and run the training
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Node Classification with GNN',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('--seed', type=int, default=42, help='random seed (default: 42)')
    parser.add_argument('--log_dir', type=str, default='logs', help='log directory')
    parser.add_argument('--file_nodes', type=str, default='./data/cora.content', help='node file (e.g., cora.content)')
    parser.add_argument('--file_edges', type=str, default='./data/cora.cites', help='edge file (e.g., cora.cites)')
    parser.add_argument('--undirected', type=bool, default=True, help='convert graph to undirected')
    parser.add_argument('--row_normalize', type=bool, default=True, help='normalizing feature vectors to have sum of 1')
    parser.add_argument('--folds', type=int, default=10, help='number of folds for cross-validation')
    parser.add_argument('--epochs', type=int, default=1000, help='number of epochs')
    parser.add_argument('--hidden_channels', type=int, default=16, help='number of hidden channels for the first layer')
    parser.add_argument('--activation', type=str, default='relu', help='activation function [relu, selu, elu, leaky_relu, prelu]')
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout rate')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--layer', type=str, default='gcn', help='layer type [gcn, gat, gatv2]')
    parser.add_argument('--heads', type=int, default=8, help='number of attention heads')
    parser.add_argument('--output', type=str, default='results.tsv', help='output file (e.g., results.tsv)')

    args = parser.parse_args()

    # Make log directory if it does not exist
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger('log' + time.strftime('%Y%m%d%H%M'))
    file_handler = logging.FileHandler(os.path.join(args.log_dir, 'log' + time.strftime('%Y%m%d%H%M') + '.log'))
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    
    logger.info(args)

    if args.layer not in ['gcn', 'gat', 'gatv2']:
        raise ValueError(f"Layer type {args.layer} is not supported. Choose from [gcn, gat, gatv2]")
    
    logger.info("Setting random seed equal to %d" % args.seed)
    seed_everything(args.seed)

    activation = activation_fn(args.activation)

    DEVICE = set_device()
    logger.info(f"Using device: {DEVICE}")

    logger.info("Loading data from %s and %s" % (args.file_nodes, args.file_edges))
    data_loader = DataLoader(file_nodes=args.file_nodes, file_edges=args.file_edges, undirected=args.undirected)

    logger.info("Getting PyG Data object")
    data = data_loader.get_data(row_normalize=args.row_normalize, device=DEVICE)
    logger.info("Successfully loaded data: %s" % data)

    logger.info("Starting training with %d folds" % args.folds)
    test_indices, predictions = train_k_fold(data, folds=args.folds,
                                             epochs=args.epochs, hidden_channels=args.hidden_channels,
                                             dropout=args.dropout, activation=activation, lr=args.lr,
                                             weight_decay=args.weight_decay, layer=args.layer,
                                             heads=args.heads, device=DEVICE, logger=logger)
    
    logger.info("Saving predictions to %s" % args.output)
    # Convert the predictions to the original node IDs
    preds = [data_loader.id_to_label[pred] for pred in predictions]
    # Convert the test indices to the original node IDs
    test_indices = [data_loader.idx_to_id[idx] for idx in test_indices]
    # Store the results in a DataFrame
    results = pd.DataFrame({'Node ID': test_indices, 'Prediction': preds})
    results.to_csv(args.output, sep='\t', index=False)
    logger.info("Successfully saved predictions!")
