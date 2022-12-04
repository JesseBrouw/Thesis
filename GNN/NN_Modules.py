import torch
import numpy as np
import pandas as pd
import json

class GNN(torch.nn.Module):
    def __init__(self, dim, n_iter):
        """
            **V1**
            Pytorch module which contains a standard Graph Neural Network,
            which only performs simple message passing, without reduction
            of size. 

            - dim : size of the one_hot encoding vector
            - n_iter : number of iterations / message passings
        """

        super().__init__()

        # instantiate the linear layers which compute the message accumulation
        self.n_iter = n_iter
        self.linear_self = torch.nn.Linear(dim, dim)
        self.linear_neigh = torch.nn.Linear(dim, dim, bias=False)

    # forward pass that computes the message, and updates all the representations
    def forward(self, labels, adjacency):
        ReLu = torch.nn.ReLU()
        for _ in range(self.n_iter):
            # Accumulate messages from neighbors.
            accumulation = torch.matmul(adjacency.T, labels)
            # Compute update.
            labels = ReLu(self.linear_self(labels) + self.linear_neigh(accumulation))
        update = labels
        return update

class Predictor(torch.nn.Module):
    """
         **V1**
        Pytorch module which contains the Feedforward neural 
        network, which outputs a prediction given the computed
        graph representation of the GNN plus the hand-crafted
        graph-features.

        - input_shape : shape of the linear layer -> [GNN_out+len(features) x 1] 
    """
    def __init__(self,input_shape, classify=True):
        super().__init__()
        self.fc1 = torch.nn.Linear(input_shape,1)
        self.classify = classify
    
    # forward pass which computes the output of the linear layer.
    def forward(self, gnn_out, features):
        if self.classify:
            out = torch.sigmoid(self.fc1(torch.concat((gnn_out, features))))
        else:
            out = torch.relu(self.fc1(torch.concat((gnn_out, features))))
        return out

class GraphClassifier(torch.nn.Module):
    """
         **V1**
        Module which ties everything together.
        First computes the graph representation of the problem instance
        and feeds this input to the Predictor module together with the
        hand-crafted graph features. 

        - gnn_dim = Size of encoding vector of labels
        - nn_dim = Size of output_GNN + #hand-crafted-features
    """
    def __init__(self, gnn_dim, nn_dim, n_iter):
        super().__init__()
        self.gnn = GNN(dim=gnn_dim, n_iter=n_iter)
        self.predictor = Predictor(nn_dim)

    # Forward pass which encodes the graph, pools the entire graph
    # to one vector representing the entire graph, and passes this
    # to the predictor together with the hand crafted features. 
    def forward(self, labels, adj, features):
        labels_encoded = self.gnn(labels, adj)
        # pooled, _ = torch.max(labels_encoded, dim=-2)
        pooled = torch.mean(labels_encoded, dim=0) # yields best results
        return self.predictor(pooled, features)

class GraphRegressor(torch.nn.Module):
    """
         **V1**
        Module which ties everything together.
        First computes the graph representation of the problem instance
        and feeds this input to the Predictor module together with the
        hand-crafted graph features. 

        - gnn_dim = Size of encoding vector of labels
        - nn_dim = Size of output_GNN + #hand-crafted-features
    """
    def __init__(self, gnn_dim, nn_dim, n_iter):
        super().__init__()
        self.gnn = GNN(dim=gnn_dim, n_iter=n_iter)
        self.predictor = Predictor(nn_dim, False)

    # Forward pass which encodes the graph, pools the entire graph
    # to one vector representing the entire graph, and passes this
    # to the predictor together with the hand crafted features. 
    def forward(self, labels, adj, features):
        labels_encoded = self.gnn(labels, adj)
        # pooled, _ = torch.max(labels_encoded, dim=-2)
        pooled = torch.mean(labels_encoded, dim=0) # yields best results
        return self.predictor(pooled, features)


def parse_graph(graph_string:str, device):
    """
        Parse input json string to graph representation
        by returning the one hot encodings of all the 
        labels, and the adjacency matrix of the graph. 
    """
    graph_dict = json.loads(graph_string)
    labels = graph_dict['labels']
    edges = graph_dict['edges']

    adjacency = torch.zeros((len(labels), len(labels)))
    for index_node, edge_nodes in enumerate(edges):
        for neighbor in edge_nodes:
            # directed graph, so only from index node to neighbor
            adjacency[index_node, neighbor] = 1
    
    one_hot_encodings = torch.zeros((len(labels), 15))
    for idx, label in enumerate(labels):
        one_hot_encodings[idx, label] = 1

    return one_hot_encodings.to(device=device), adjacency.to(device=device)

def get_features(df:pd.DataFrame) -> torch.Tensor:
    """
        Get tensors out of the hand-crarfted graph features
    """
    feature_df = df.drop(columns=['graph', 'domain'])
    features = feature_df.to_numpy()

    return torch.from_numpy(features)


