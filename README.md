# Node Classification

When it comes to graph data, the most common task is node classification. In this task, we are given a graph where each node has a label, and we are interested in predicting the label of the nodes for which the label is unknown. Representing our data as a graph allows us to leverage the relationships between nodes to make better predictions.

## Problem Statement

The objective is to develop a machine learning approach to predict the subjects of scientific papers.

## Dataset

### CORA

The Cora dataset consists of `2708` scientific publications classified into one of `7` classes:

- `Case_Based`
- `Genetic_Algorithms`
- `Neural_Networks`
- `Probabilistic_Methods`
- `Reinforcement_Learning`
- `Rule_Learning`
- `Theory`

Each publication in the dataset is described by a `0/1`-valued word vector indicating the absence/presence of the corresponding word from the dictionary. The dictionary consists of `1433` unique words. The citation network consists of `5429` links.

You can download the dataset from [here](https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz). However, the dataset is already available in the `data` directory.


## Methodology

We will use some of the most popular Graph Neural Networks (GNNs) to solve this node classification problem. The GNNs we will implement are:

1. Graph Convolutional Network (GCN) ([Semi-Supervised Classification with Graph Convolutional Networks](https://arxiv.org/abs/1609.02907))
2. Graph Attention Network (GAT) ([Graph Attention Networks](https://arxiv.org/abs/1710.10903))
3. Graph Attention Network v2 (GATv2) ([How Attentive are Graph Attention Networks?](https://arxiv.org/abs/2105.14491))

The full description of how a GNN works and what are the structures of these models are provided in the juptyer notebook.

The implementation of the GNNs is done using the [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/) library. The PyTorch Geometric library is a geometric deep learning extension library for PyTorch. It consists of various methods and utilities to ease the implementation of Graph Neural Networks.

The main steps involved in the implementation are:

1. Seed everything for reproducibility
2. Set the device to GPU if available
3. Load the data into a dictionary of features, a dictionary of labels, and a list of edges
4. Make the graph undirected by adding edges in both directions (if specified)
5. Mapping the node IDs to a range of `0` to `num_nodes`
6. Convert the data into PyTorch Geometric `Data` object
7. Normalize the node features to have a sum of `1` for each row (if specified)
8. Split the dataset into K-Folds for cross-validation
9. Train the model on each fold and evaluate the performance
10. Report the average performance across all folds
11. Concatenate the predictions of each fold to get the final predictions
12. Save the predictions to a TSV file


## Requirements

The code is written in Python `3.10` and is dependent on the following libraries:

- `numpy==1.25.2`
- `pandas==2.0.3`
- `networkx==3.3`
- `plotly==5.15.0`
- `matplotlib==3.7.1`
- `torch==2.2.1`
- `torch_geometric==2.5.2`

You can install the required libraries by running:

```bash
pip install -r requirements.txt
```

**Note:** The code is tested on [Google Colab](https://colab.research.google.com/) with GPU acceleration enabled. It has all the required libraries pre-installed, except for `torch_geometric`. You can install `torch_geometric` by running:

```bash
!pip install torch_geometric==2.5.2
```

## Usage

I highly recommend to use the juptyer notebook `NodeClassification_GNN.ipynb` (click [here](https://github.com/parsa-abbasi/NodeClassification/blob/main/NodeClassification_GNN.ipynb)) to run the code. The notebook is well-documented and provides a step-by-step guide to run the code.

However, you can also run the code using the `run.py` script. The script has the following arguments:

| Argument | Description | Default |
|----------|-------------|---------|
| `--seed` | Seed for reproducibility | `42` |
| `--log_dir` | Directory to save the logs | `logs` |
| `--file_nodes` | Path to the file containing the nodes | `./data/cora.content` |
| `--file_edges` | Path to the file containing the edges | `./data/cora.cites` |
| `--undirected` | Whether to make the graph undirected | `True` |
| `--row_normalize` | Whether to row normalize the node features | `True` |
| `--folds` | Number of folds for cross-validation | `10` |
| `--epochs` | Number of epochs to train the model | `1000` |
| `--hidden_channels` | Number of hidden channels in the GNN | `16` |
| `--activation` | Activation function for the GNN | `relu` |
| `--dropout` | Dropout rate for the GNN | `0.5` |
| `--lr` | Learning rate for the optimizer | `0.01` |
| `--weight_decay` | Weight decay for the optimizer | `5e-4` |
| `--layer` | Type of GNN to use (gcn, gat, gatv2) | `gcn` |
| `--heads` | Number of heads for the GAT/GATv2 | `8` |
| `--output` | Path to save the predictions | `results.tsv` |

```bash
!python run.py --seed 42 --log_dir logs --file_nodes ./data/cora.content --file_edges ./data/cora.cites --undirected True --row_normalize True --folds 3 --epochs 1000 --hidden_channels 8 --activation elu --dropout 0.6 --lr 0.01 --weight_decay 5e-4 --layer gatv2 --heads 8 --output results.tsv
```

## Repository Structure

| File/Folder | Description |
|-------------|-------------|
| `NodeClassification_GNN.ipynb` | A juptyer notebook containing all of the required code to train a GNN model on the CORA dataset |
| `run.py` | A script to run the code from the command line |
| `training.py` | A script containing the training loop for the GNNs using the K-Fold Cross-Validation technique |
| `utils/graph_loader.py` | A script containing a class to load the graph data from the files |
| `utils/graph_visualizer.py` | A script containing a class to visualize the graph using `plotly` and `networkx` |
| `layers/gcn.py` | The implementation of the Graph Convolutional Network (GCN) layer |
| `layers/gat.py` | The implementation of the Graph Attention Network (GAT) layer |
| `layers/gatv2.py` | The implementation of the Graph Attention Network v2 (GATv2) layer |
| `data` | A directory containing the CORA dataset |
| `predictions` | A directory containing the predictions of the GNNs in TSV format |
| `images` | A directory containing the images of the node embeddings visualization |

## Experiments

The designed GNNs consist of two layers of GCN/GAT/GATv2 and a dropout layer between the two layers. The GNNs are trained using the `Adam` optimizer and the Negative Log Likelihood (NLL) loss function. The hyperparameters which we set for the GNNs are based on the original papers, but they can be tuned to improve the performance. The hyperparameters are:

| Hyperparameter | GCN | GAT | GATv2 |
|----------------|-----|-----|-------|
| Epochs         | 1000 | 1000 | 1000 |
| Hidden Channels | 16 | 8 | 8 |
| Dropout Rate   | 0.5 | 0.6 | 0.6 |
| Learning Rate  | 0.01 | 0.01 | 0.01 |
| Weight Decay   | 5e-4 | 5e-4 | 5e-4 |
| Activation     | ReLU | ELU | ELU |
| Heads          | - | 8 | 8 |

The results of the GNNs on the CORA dataset are as follows:

| Model | 10-Fold Cross-Validation Accuracy |
|-------|-----------------------------------|
| GCN   | 88.55%                            |
| GAT   | 88.74%                            |
| GATv2 | 88.52%                            |

However, GATv2 is based on a stronger mathematical foundation and we can rely on its resulted attention scores to interpret the model's predictions. The attention scores can be used to understand which nodes are important for the model's predictions.

Based on this assumption, the GATv2 model was exprimented with different set of hyperparameters using the grid search technique. The hyperparameters which were tuned are:

| Hyperparameter | Search Space |
|----------------|--------------|
| `hidden_channels` | `[16, 32, 64]` |
| `dropout` | `[0.5, 0.6, 0.7]` |
| `activation` | `['relu', 'elu', 'leaky_relu']` |
| `heads` | `[8]` |

The best set of hyperparameters achieved an accuracy of `88.88%` on the CORA dataset. The best set of hyperparameters are:

| Hyperparameter | Value |
|----------------|-------|
| `hidden_channels` | `32` |
| `dropout` | `0.6` |
| `activation` | `'elu'` |
| `heads` | `8` |

<details>
  <summary>Code snippet to run the hyperparameter tuning</summary>

    I avoid putting the hyperparameter tuning code in the juptyer notebook to keep it clean. However, you can run the hyperparameter tuning using the following code snippet:

    ```python
    from itertools import product

    hyperparameters = {
        'hidden_channels': [16, 32, 64],
        'dropout': [0.5, 0.6, 0.7],
        'activation': ['relu', 'elu', 'leaky_relu'],
        'heads': [8],
        'layer': ['gatv2']
    }
    # Generate all combinations of hyperparameters
    combinations = list(product(*hyperparameters.values()))
    # Initialize an empty list to store the results
    all_results = []
    # Iterate over all combinations of hyperparameters
    for i, combination in enumerate(combinations):
        print('='*100)
        print(f"Combination {i + 1}/{len(combinations)}")
        hidden_channels, dropout, activation, heads, layer = combination
        print(f"Hidden Channels: {hidden_channels}, Dropout: {dropout}, Activation: {activation}, Heads: {heads}, Layer: {layer}")
        # Initialize the runner with the hyperparameters
        runner = Runner(folds=10, epochs=400, hidden_channels=hidden_channels, dropout=dropout,
                        activation=activation, lr=0.01, weight_decay=5e-4, layer=layer, heads=heads)
        # Run the model
        predictions = runner.run()
        # Compute the accuracy
        results_df = final_predictions(predictions)
        merged = pd.merge(df, results_df, on='Node ID')
        acc = accuracy_score(merged['Label'], merged['Prediction'])
        # Append the results to the list
        all_results.append({'Model': layer, 'Hidden Channels': hidden_channels, 'Dropout': dropout,
                            'Activation': activation, 'Heads': heads, 'Accuracy': acc})
    # Convert the results to a DataFrame and sort them by accuracy
    results = pd.DataFrame(all_results)
    results.sort_values(by='Accuracy', ascending=False, inplace=True)
    results.reset_index(drop=True, inplace=True)
    results
    ```
</details>

## Node Embeddings Visualization

We can visualize the node embeddings produced by the GNN's layers using the `t-SNE` algorithm. The `graph_visualizer.py` in the `utils` directory provides a class to interactively visualize the graph using `plotly` and `networkx`. At the end of the juptyer notebook, we visualize the node embeddings produced by the GATv2 model using this class. The visualization shows the nodes in the graph and their labels. The nodes are colored based on their labels. We can see that the GATv2 model has learned to cluster the nodes based on their labels.

### First Layer

![First Layer](https://github.com/parsa-abbasi/NodeClassification/blob/main/images/layer1.png?raw=true)

### Second Layer

![Second Layer](https://github.com/parsa-abbasi/NodeClassification/blob/main/images/layer2.png?raw=true)

## Future Work

The future work can be done to improve the performance of the GNNs on the CORA dataset. Some of the suggestions are:

- Try different number of layers for the GNNs
- Hyperparameter tuning to find the best set of hyperparameters for the GNNs
- Add skip connections to the GNNs to improve the flow of information
- Visualizing the attention scores to understand which nodes are important for the model's predictions
- Can feature engineering such as feature selection improve the performance of the GNNs?