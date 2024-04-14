import networkx as nx
import numpy as np
from plotly import graph_objects as go

class GraphVisualizer:
    def __init__(self, node_features: dict, node_labels: dict, edge_index: list, directed: bool = True):
        '''
        `GraphVisualizer`: creates a directed networkx graph from the following inputs
            - `node_features`: a dictionary where the key is the node ID and the value is the feature vector of the node
            - `node_labels`: a dictionary where the key is the node ID and the value is the label of the node
            - `edge_index`: a list of edges where each edge is a list [source, target]
            - `directed`: a boolean indicating if the graph is directed or not
        '''
        # Create a networkx graph
        if directed:
            self.graph = nx.DiGraph()
        else:
            self.graph = nx.Graph()
        # Add nodes and edges to the graph
        for node, feature in node_features.items():
            self.graph.add_node(node, feature=feature, label=node_labels[node])
        for edge in edge_index:
            self.graph.add_edge(edge[0], edge[1])

    def set_node_positions(self, pos: dict = None):
        '''
        `set_node_positions`: sets the positions of the nodes in the graph
            - `pos`: a dictionary where the key is the node ID and the value is the position of the node
        '''
        for node in self.graph.nodes:
            self.graph.nodes[node]['pos'] = pos[node]


    def set_layout(self, layout: str = 'spring'):
        '''
        `set_layout`: sets the layout of the graph
            - `layout`: a string indicating the layout of the graph
        '''
        if layout == 'spring':
            pos = nx.spring_layout(self.graph)
        elif layout == 'kamada_kawai':
            pos = nx.kamada_kawai_layout(self.graph)
        elif layout == 'circular':
            pos = nx.circular_layout(self.graph)
        elif layout == 'random':
            pos = nx.random_layout(self.graph)
        elif layout == 'shell':
            pos = nx.shell_layout(self.graph)
        elif layout == 'spectral':
            pos = nx.spectral_layout(self.graph)
        elif layout == 'planar':
            pos = nx.planar_layout(self.graph)
        elif layout == 'fruchterman_reingold':
            pos = nx.fruchterman_reingold_layout(self.graph)
        else:
            raise ValueError('Invalid layout')

        self.set_node_positions(pos)

    def plot(self, title: str = 'Graph Visualization', width: int = 1000, height: int = 1000):
        '''
        `plot`: plots the graph using plotly
        '''

        # Set the node positions if they are not already set
        if 'pos' not in self.graph.nodes[list(self.graph.nodes.keys())[0]]:
            self.set_layout(layout='spring')
        
        # Get the node positions
        node_x = []
        node_y = []
        for node in self.graph.nodes():
            x, y = self.graph.nodes[node]['pos']
            node_x.append(x)
            node_y.append(y)
        
        # Get the node labels and set the colors
        labels = [self.graph.nodes[node]['label'] for node in self.graph.nodes()]
        idx = np.unique(labels, return_inverse=True)[1]
        colors = idx
        # Get the number of neighbors for each node
        num_neighbors = [len(list(nx.all_neighbors(self.graph, node))) for node in self.graph.nodes()]
        
        # Create the node trace
        node_trace = go.Scatter(
            x = node_x, y = node_y,
            mode = 'markers',
            hoverinfo = 'text',
            hovertext = [f'Label: {label}<br>Number of neighbors: {num}' for label, num in zip(labels, num_neighbors)],
            marker = dict(
                size = 12,
                line_width = 2,
                colorscale = 'Portland',
                color = colors,
            ),
        )

        # Get the edge positions
        edge_x = []
        edge_y = []
        for edge in self.graph.edges():
            x0, y0 = self.graph.nodes[edge[0]]['pos']
            x1, y1 = self.graph.nodes[edge[1]]['pos']
            edge_x.append(x0)
            edge_x.append(x1)
            edge_x.append(None)
            edge_y.append(y0)
            edge_y.append(y1)
            edge_y.append(None)
        
        # Create the edge trace
        edge_trace = go.Scatter(
            x = edge_x, y = edge_y,
            line = dict(width=0.5, color='#888'),
            hoverinfo = 'none',
            mode = 'lines')
        
        # Create the figure
        fig = go.Figure(data=[edge_trace, node_trace],
                        layout = go.Layout(
                        width = width,
                        height = height,
                        title = title,
                        titlefont_size = 16,
                        showlegend = False,
                        hovermode = 'closest',
                        margin = dict(b=20,l=5,r=5,t=40),
                        xaxis = dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis = dict(showgrid=False, zeroline=False, showticklabels=False))
                        )
        fig.show()