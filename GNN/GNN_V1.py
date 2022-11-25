import tqdm
import torch
import sys
import json
from tqdm import tqdm
from sklearn.metrics import classification_report
import numpy as np
from contextlib import redirect_stdout

import NN_Modules
from NN_Modules import parse_graph, get_features
from load_data import load_data


# TODO: Make alterations such that it can run on a CUDA device when available.

torch.manual_seed(1)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def main(argument:str):
    print(f'Device : {device}')
    N_ITER = 2              # number of message passing iterations
    GNN_DIM = 15            # amount of node labels
    LEARNING_RATE = 0.001   # learning rate of the optimizer
    EPOCHS = 30             # number of epochs over the data

    # load the data
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_data(argument)

    # Get the labels and adjacency matrices from the json strings in the dataframe
    labels_train, adj_train = zip(*(parse_graph(json_str, device) for json_str in X_train['graph']))
    labels_val, adj_val = zip(*(parse_graph(json_str, device) for json_str in X_val['graph']))
    labels_test, adj_test = zip(*(parse_graph(json_str, device) for json_str in X_test['graph']))

    # Extract the hand-crafted features from the dataframe. 
    features_train = get_features(X_train).to(device=device)
    features_val = get_features(X_val).to(device=device)
    features_test = get_features(X_test).to(device=device)

    # Compute the dimension of the Predictor dimension
    nn_dim = features_train.shape[1] + GNN_DIM

    # instantiate model
    model = NN_Modules.GraphPredictor(GNN_DIM, nn_dim, N_ITER).to(device=device)
    # instantiate optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    def train_pass(labels, adjacency, features, y):
        """
            computes a training pass over the data.
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
    arg = sys.argv[1]
    to_file = sys.argv[2]

    if to_file == 'True':
        orig_stdout = sys.stdout
        with open(f'./Results/exp_{sys.argv[0].split(".")[0]}_{arg.split(".")[0]}.txt', 'w') as wf:
            with redirect_stdout(wf):
                main(arg)
    else:
        main(arg)


