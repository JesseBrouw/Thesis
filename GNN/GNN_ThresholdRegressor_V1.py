import tqdm
import torch
import sys
from tqdm import tqdm
from sklearn.metrics import classification_report
import numpy as np
from contextlib import redirect_stdout

import NN_Modules
from NN_Modules import parse_graph, get_features
from load_data import load_data, load_threshold_data
import metrics


# TODO: Make alterations such that it can run on a CUDA device when available.

torch.manual_seed(1)
device = device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def main(argument:str):
    print(f'Device : {device}')
    N_ITER = 6              # number of message passing iterations
    GNN_DIM = 15            # amount of node labels
    LEARNING_RATE = 0.001   # learning rate of the optimizer
    EPOCHS = 300            # number of epochs over the data

    # load the data
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_threshold_data(argument)

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
    model = NN_Modules.GraphRegressorV1(GNN_DIM, nn_dim, N_ITER).to(device=device)
    # instantiate optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    def train_pass(labels, adjacency, features, y):
        """
            computes a training pass over the data.
        """
        # Set module to training mode
        model.train()
        def quadquad_loss(pred, y, p=2, alpha=0.7):
            """
                Estimation_of_flood_warning_runoff_thresholds_in_u.pdf

                Assymetric loss function, penalizes underestimating more when
                alpha > 0.5, less when alpha = 0.5, and for p = 2, and alpha = 0.5
                it is mean squared error. 
            """
            return 2*(alpha + (1-2*alpha)*((y-pred)<0)) * torch.abs((y-pred))**p

        loss_function = quadquad_loss

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

    for epoch in range(EPOCHS):
        print(f"Epoch {epoch}")
        train_loss = train_pass(labels_train, adj_train, features_train, y_train)
        print(f"Training MSE loss \n {train_loss}")
        print(f"Validation MSE loss \n {train_pass(labels_val, adj_val, features_val, y_val)}")
    
    test_loss = train_pass(labels_test, adj_test, features_test, y_test)
    print(f'The loss on the test set is : {test_loss}')
    model.eval()
    predictions = []
    for labeli, adji, feati, yi in tqdm(zip(labels_test, adj_test, features_test, y_test)):
        out = model(labeli, adji, feati)
        predictions.append(round(out.item()))
    print("actual / predictions : ")
    metrics.visualize_regr_results(list(y_test), list(predictions), list(X_test['domain']), f'V1_{argument.split(".")[0]}.png')

if __name__ == '__main__':
    arg = sys.argv[1]
    to_file = sys.argv[2]

    if to_file == 'True':
        orig_stdout = sys.stdout
        with open(f'./Results/Results_threshold/exp_{sys.argv[0].split(".")[0]}_{arg.split(".")[0]}.txt', 'w') as wf:
            with redirect_stdout(wf):
                main(arg)
    else:
        main(arg)