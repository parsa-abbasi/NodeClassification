import os    
import networkx as nx

class GraphLoader:
    '''
    `GraphLoader`: A class to load the raw graph data from the files
        - `file_nodes`: The path to the file containing the nodes data (e.g., cora.content)
            - The format of each line is <node_id> <feature_0> <feature_1> ... <feature_n> <label>
        - `file_edges`: The path to the file containing the edges data (e.g., cora.cites)
            - The format of each line is <ID of source node> <ID of target node>
    '''

    def __init__(self, file_nodes: str='', file_edges: str=''):

        # if files paths are not provided, use the default ones which is the cora dataset
        if file_nodes == '' or file_edges == '':
            data_dir = os.path.join(os.path.dirname("__file__"), 'data')
            file_nodes = os.path.join(data_dir, 'cora.content')
            file_edges = os.path.join(data_dir, 'cora.cites')
        self.file_nodes = file_nodes
        self.file_edges = file_edges

        # key: node ID, value: feature vector
        self.features = {}
        # key: node ID, value: label
        self.labels = {}
        # List of edges in the graph
        self.edge_index = []
        
        # Load the data from the files
        self.load_data()

    def load_data(self):

        # Load the features and labels of the nodes
        with open(self.file_nodes, 'r') as f:
            for line in f:
                line = line.strip().split()
                node = int(line[0])
                self.features[node] = list(map(int, line[1:-1]))
                self.labels[node] = line[-1]

        # Load the edges of the graph
        with open(self.file_edges, 'r') as f:
            for line in f:
                line = line.strip().split()
                self.edge_index.append([int(line[0]), int(line[1])])

    def get_data(self):
        return self.features, self.labels, self.edge_index
    
    def print_info(self):
        print('# Nodes:', len(self.features))
        print('# Edges:', len(self.edge_index))
        print('# Classes:', len(set(self.labels.values())))
        print('# Features:', len(self.features[list(self.features.keys())[0]]))