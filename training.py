import numpy as np
import logging
import torch, torch_geometric
import torch.nn.functional as F
from torch_geometric.data import Data
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from layers.gcn import GCN
from layers.gat import GAT
from layers.gatv2 import GATv2

def train(model: torch.nn.Module,
          optimizer: torch.optim.Optimizer,
          data: torch_geometric.data.Data) -> torch.Tensor:
    '''
    `train`: A function to train the model for one epoch
        - `model`: The model to train
        - `optimizer`: The optimizer to use
        - `data`: A PyG `Data` object containing the graph data
    '''

    # Set the model to training mode
    model.train()

    optimizer.zero_grad()
    out = model(data)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss


def evaluate(model: torch.nn.Module,
             data: torch_geometric.data.Data) -> torch.Tensor:
    '''
    `evaluate`: A function to evaluate the model on the test set
        - `model`: The model to evaluate
        - `data`: A PyG `Data` object containing the graph data
    '''

    # Set the model to evaluation mode
    model.eval()

    out = model(data)
    pred = out.argmax(dim=1)
    pred_np = pred[data.test_mask].cpu().numpy()
    actual_np = data.y[data.test_mask].cpu().numpy()
    acc = accuracy_score(actual_np, pred_np)
    return pred_np, acc


def k_fold(data: torch_geometric.data.Data, folds: int=10, SEED: int=42) -> list:
    '''
    `k_fold`: A function to split the data into `folds` folds using KFold
        - `data`: PyG Data object
        - `folds`: The number of folds to split the data into
    '''
    kf = KFold(n_splits=folds, shuffle=True, random_state=SEED)
    # Store indices of train and test data for each fold
    train_indices = []
    test_indices = []
    for i, (train_index, test_index) in enumerate(kf.split(data.y)):
        train_indices.append(train_index)
        test_indices.append(test_index)
    return train_indices, test_indices

def train_k_fold(data, folds=10, epochs=100, hidden_channels=16, dropout=0.5,
                 activation=F.relu, lr=0.01, weight_decay=5e-4, layer='gcn',
                 heads=8, device=torch.device('cpu'), logger=logging.getLogger(__name__)):
    '''
    `train_k_fold`: A function to train the model using KFold cross validation
        - `data`: PyG Data object
        - `folds`: The number of folds to split the data into
        - `epochs`: The number of epochs to train the model
        - `hidden_channels`: The number of hidden channels in the GCN layers
        - `dropout`: The dropout probability
        - `activation`: The activation function to use (e.g., `torch.nn.functional.relu`)
        - `lr`: The learning rate
        - `weight_decay`: The weight decay
        - `layer`: The type of layer to use (e.g., `gcn`, `gat`, `gatv2`)
        - `heads`: The number of attention heads (only for GAT and GATv2)
        - `device`: The device to use (e.g., `torch.device('cuda')`)
        - `logger`: The logger to use (e.g., `logging.getLogger(__name__)`)
    '''

    # Split the data into folds
    train_indices, test_indices = k_fold(data, folds)

    # Store the results of each fold
    results = {'acc': []}
    predictions = []
    for i in range(folds):
        logger.info(f"Fold {i + 1}/{folds}")
        # Create the model
        if layer == 'gcn':
            model = GCN(data, hidden_channels, dropout, activation).to(device)
        elif layer == 'gat':
            model = GAT(data, hidden_channels, dropout, activation, heads).to(device)
        elif layer =='gatv2':
            model = GATv2(data, hidden_channels, dropout, activation, heads).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        # Set the train and test masks for the current fold
        data.train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        data.test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        data.train_mask[train_indices[i]] = 1
        data.test_mask[test_indices[i]] = 1

        # Train the model
        for epoch in range(epochs):
            loss = train(model, optimizer, data)
            if epoch > 0 and (epoch + 1) % 100 == 0:
                logger.info(f"Epoch {epoch + 1}/{epochs} | Loss: {loss:.4f}")

        # Evaluate the model
        pred, acc = evaluate(model, data)
        logger.info(f"Accuracy: {acc:.4f} | Loss: {loss:.4f}")
        print('-'*70)
        results['acc'].append(acc)
        predictions.append(pred)

    logger.info(f"Overall results:")
    logger.info(f"Average accuracy: {np.mean(results['acc']):.4f}")

    test_indices = np.concatenate(test_indices)
    predictions = np.concatenate(predictions)
    return test_indices, predictions