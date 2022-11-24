import pandas as pd
import numpy as np
import os
import json
from collections import Counter
import sys

def main(file):
    # read the file that is supposed to be transformed to a dataframe.
    df = pd.read_csv(os.path.join(os.getcwd(), 'solve_data', file))
    df['domain'] = df['domain'].str.strip()
    domains = list(df['domain'].unique())

    # create dictionary of dataframes that will contain all frames.
    dataframes = {}
    for domain in domains:
        print(domain)
        # create copy of dataframe containing only that specific domain
        copy = df.loc[df['domain'] == domain].copy()
        copy = copy[['domain', 'problem', 'horizon', 'coverage']]

        # remove trailing .pddl from the problem name
        copy['problem'] = copy['problem'].apply(lambda x: x.split(".")[0])
        files = [i for i in os.listdir(f'./graphs_domains/{domain}') if len(i.split(".")[0]) > 0]
        files = [i for i in files if i.split(".")[2] == 'txt']

        # for each file, read each problem instance graph
        # line by line and parse to list of integers per line.
        file_dict = {}
        for file in files:
            name = file.split(".")[0]
            with open(f'./graphs_domains/{domain}/{file}') as f:
                lines = f.readlines()
                lines = [line.strip() for line in lines]
                lines = [line.split(",") for line in lines]
                for i, line in enumerate(lines):
                    line = [int(i) for i in line]
                    lines[i] = line
                file_dict[name] = lines
        
        # save each graph as a dict containing the node label,
        # and edges to nodes this node connects to
        for key, value in file_dict.items():
            new = {}
            new['labels'] = [int(i[0]) for i in value]
            new['edges'] = [i[1:] for i in value]
            file_dict[key] = new

        # extract the counts of the node labels 
        counts = {}
        for key, value in file_dict.items():
            counter = dict(Counter(value['labels']))
            label_counts = {}
            for i in range(15):
                label_counts[i] = 0
            label_counts.update(counter)
            counts[key] = label_counts

        # dump the graph structure as json string in column 'graph'
        copy.problem = copy.problem.str.strip()
        copy.problem = copy.problem.str.lower()
        copy['graph'] = copy['problem'].apply(lambda x: json.dumps(file_dict[x]))

        # save the label count features in specific columns
        for i in range(15):
            copy[f'label_{i}'] = copy['problem'].apply(lambda x: json.dumps(counts[x][i]))

        # sort by problem, and horizon
        copy.sort_values(by=['problem', 'horizon'], inplace=True)

        # write to csv
        dataframes[domain] = copy
        copy.to_csv(f'./GNN_data/dataframe_{domain}.csv', index=False)

    # write csv of all domains in the file together
    concatenated = pd.concat(list(dataframes.values()))
    concatenated.to_csv('./GNN_data/complete_dataset.csv', index=False)

if __name__ == '__main__':
    file = sys.argv[1]
    main(file)