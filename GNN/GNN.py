import tqdm
import torch
import pandas as pd
import sys
import json
from tqdm import tqdm
from sklearn.metrics import classification_report
import numpy as np

from load_data import load_data

# TODO: Make alterations such that it can run on a CUDA device when available.

class GNN(torch.nn.Module):
    def __init__(self, dim, n_iter):
        """
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
            accumulation = torch.matmul(adjacency, labels)
            # Compute update.
            labels = ReLu(self.linear_self(labels) + self.linear_neigh(accumulation))
        update = labels
        return update

class Predictor(torch.nn.Module):
    """
        Pytorch module which contains the Feedforward neural 
        network, which outputs a prediction given the computed
        graph representation of the GNN plus the hand-crafted
        graph-features.

        - input_shape : shape of the linear layer -> [GNN_out+len(features) x 1] 
    """
    def __init__(self,input_shape):
        super().__init__()
        self.fc1 = torch.nn.Linear(input_shape,1)
    
    # forward pass which computes the output of the linear layer.
    def forward(self, gnn_out, features):
        out = torch.sigmoid(self.fc1(torch.concat((gnn_out, features))))
        return out

class GraphPredictor(torch.nn.Module):
    """
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

def parse_graph(graph_string:str):
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

    return one_hot_encodings, adjacency

def get_features(df:pd.DataFrame) -> torch.Tensor:
    """
        Get tensors out of the hand-crarfted graph features
    """
    feature_df = df.drop(columns=['graph'])
    features = feature_df.to_numpy()

    return torch.from_numpy(features)


def main(argument:str):
    N_ITER = 2              # number of message passing iterations
    GNN_DIM = 15            # amount of node labels
    LEARNING_RATE = 0.001   # learning rate of the optimizer
    EPOCHS = 30             # number of epochs over the data

    # load the data
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_data(argument)

    # Get the labels and adjacency matrices from the json strings in the dataframe
    labels_train, adj_train = zip(*(parse_graph(json_str) for json_str in X_train['graph']))
    labels_val, adj_val = zip(*(parse_graph(json_str) for json_str in X_val['graph']))
    labels_test, adj_test = zip(*(parse_graph(json_str) for json_str in X_test['graph']))

    # Extract the hand-crafted features from the dataframe. 
    features_train = get_features(X_train)
    features_val = get_features(X_val)
    features_test = get_features(X_test)

    # Compute the dimension of the Predictor dimension
    nn_dim = features_train.shape[1] + GNN_DIM

    # instantiate model
    model = GraphPredictor(GNN_DIM, nn_dim, N_ITER)
    # instantiate optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    def train_pass(labels, adjacency, features, y):
        """
            computes one train pass over the data
        """
        # Set module to training mode
        model.train()
        loss_function = torch.nn.BCELoss()

        # permuate the training data randomly
        indices = torch.randperm(len(y))

        total_loss = 0
        for idx in tqdm(indices):
            labeli, adji, feati, yi = labels[idx], adjacency[idx], features[idx], y[idx]

            # compute output
            optimizer.zero_grad()
            out = model(labeli, adji, feati)
            out = out.squeeze()

            # compute loss
            loss = loss_function(out, torch.tensor(yi, dtype=torch.float32))
            total_loss += loss

            # perform backward pass
            loss.backward()
            optimizer.step()

        return total_loss / len(y)

    def report(labels, adjacency, features, y):
        """
        Compute classification report, and return predictions.
        """
        with torch.no_grad():
            model.eval()

            predictions = []
            for labeli, adji, feati, yi in tqdm(zip(labels, adjacency, features, y)):
                out = model(labeli, adji, feati)
                predictions.append(round(out.item()))

        return classification_report(y, np.array(predictions)), np.array(predictions)
    
    print("Before training:")
    print("Train report \n", report(labels_train, adj_train, features_train, y_train)[0])
    print("Valid report \n", report(labels_val, adj_val, features_val, y_val)[0])

    for epoch in range(EPOCHS):
        print(f"Epoch {epoch}")
        train_loss = train_pass(labels_train, adj_train, features_train, y_train)
        print(f"Training BCE loss \n {train_loss}")
        val_report = report(labels_val, adj_val, features_val, y_val)[0]
        print(f"Valid report \n {val_report}")
    
    test_report, predictions = report(labels_test, adj_test, features_test, y_test)
    print(f"Test report : \n {test_report}")
    print("actual / predictions : ")
    for y, pred in zip(y_test, predictions):
        print((y,pred))


if __name__ == '__main__':
    main(sys.argv[1])

