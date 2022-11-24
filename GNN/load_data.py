import os
import pandas as pd
import json

def load_data(target:str):
    """

    """
    file_path = os.path.join(os.getcwd(), 'GNN_data', target)

    df = pd.read_csv(file_path)
    problems = list(df['problem'].unique())

    # split the data grouped by problem instances
    train_split = int(0.7 * len(problems))
    valid_split = int(0.15 * len(problems))
    train = df[df['problem'].isin(problems[:train_split])]
    val = df[df['problem'].isin(problems[train_split:train_split+valid_split])]
    test = df[df['problem'].isin(problems[train_split+valid_split:])]

    def extract_xy(df:pd.DataFrame):
        y = df['coverage'].to_numpy()
        x = df.drop(columns=['coverage', 'domain', 'problem'])

        return x, y
    
    return extract_xy(train), extract_xy(val), extract_xy(test)



