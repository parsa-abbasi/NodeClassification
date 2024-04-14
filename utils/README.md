# Utils

This folder contains utility scripts that are used in the project.

## Graph Loader (`graph_loader.py`)

This script is used to load the graph from the given files. The graph should be stored in a format like the Cora dataset. It requires two files:

- `file_nodes`: a file where each line represents the node ID, its features, and its label
    - The format of each line is <node_id> <feature_0> <feature_1> ... <feature_n> <label>
- `file_edges`: a file where each line represents an edge between two nodes
    - The format of each line is <node_id_1> <node_id_2>

The script will load the graph and return the following:

| Variable | Description |
| --- | --- |
| `features` | a dictionary where the key is the node ID and the value is the features of the node |
| `labels` | a dictionary where the key is the node ID and the value is the label of the node |
| `edge_index` | a list of tuples where each tuple represents an edge between two nodes |

### Functions

| Function | Description |
| --- | --- |
| `load_data()` | Loads the graph from the given files |
| `get_data()` | Returns the loaded data (features, labels, and edge_index) |
| `print_info()` | Prints the information about the loaded graph (number of nodes, number of edges, number of labels, and number of features) |

### Usage

```python
from graph_loader import GraphLoader

graph_loader = GraphLoader(file_nodes='path/to/file_nodes', file_edges='path/to/file_edges')
features, labels, edge_index = graph_loader.load()
graph_loader.print_info()
```

## Graph Visualizer (`graph_visualizer.py`)

This script will make a networkx (`nx`) graph from the given graph data and visualize it using plotly. It requires the following data:

- `features`: a dictionary where the key is the node ID and the value is the features of the node
- `labels`: a dictionary where the key is the node ID and the value is the label of the node
- `edge_index`: a list of tuples where each tuple represents an edge between two nodes
- `directed`: a boolean that represents whether the graph is directed or not

### Functions

| Function | Description | Arguments |
| --- | --- | --- |
| `set_node_positions())` | Sets the positions of the nodes in the graph | `pos`: a dictionary where the key is the node ID and the value is the position of the node |
| `set_layout()` | Sets the layout of the graph | `layout`: a string that represents the layout of the graph (e.g., 'spring', 'kamada_kawai', 'circular', 'random') |
| `plot()` | Plots the graph | `title`: a string, width: an integer, height: an integer |

### Usage

```python
from graph_visualizer import GraphVisualizer

graph_visualizer = GraphVisualizer(features, labels, edge_index, directed=False)
graph_visualizer.set_layout('spring')
graph_visualizer.plot(title='Graph Visualization', width=800, height=800)
```