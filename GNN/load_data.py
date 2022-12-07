import os
import pandas as pd
import numpy as np

def load_data(target:str, task:str):
    """
        Function which gets a target csv file as parameter
        and returns a train, val, test split. 
        
        There are two possibilities : Use one specific
        domain, or use multiple domains. 

        For the task there are also 2 possibilities. 
        either 'classification' or 'regression', where
        classification tackles the sol/usnol problem and
        regression, regresses the runtime
    """
    file_path = os.path.join(os.getcwd(), 'data','GNN_BC_data', target)

    df = pd.read_csv(file_path)
    if df['domain'].nunique() == 1:
        problems = list(df['problem'].unique())

        # split the data grouped by problem instances
        train_split = int(0.7 * len(problems))
        valid_split = int(0.15 * len(problems))
        train = df[df['problem'].isin(problems[:train_split])]
        val = df[df['problem'].isin(problems[train_split:train_split+valid_split])]
        test = df[df['problem'].isin(problems[train_split+valid_split:])]

        def extract_xy(df:pd.DataFrame, task):
            if task == 'classification':
                y = df['coverage'].to_numpy(dtype=float)
                x = df.drop(columns=['coverage', 'total_time', 'problem'])
            else:
                df = df[df['total_time'] != ' NA']
                y = df['total_time'].to_numpy(dtype=float)
                x = df.drop(columns=['coverage', 'total_time'])

            return x, y
        
        return extract_xy(train, task), extract_xy(val, task), extract_xy(test, task)
    
    else:
        domains = list(df['domain'].unique())
        print(domains)

        X_train = []
        y_train = []
        X_val = []
        y_val = []
        X_test = []
        y_test = []

        for domain in domains:
            domain_df = df[df['domain'] == domain].copy()
            problems = list(domain_df['problem'].unique())

            # split the data grouped by problem instances
            train_split = int(0.7 * len(problems))
            valid_split = int(0.15 * len(problems))
            train = domain_df[domain_df['problem'].isin(problems[:train_split])]
            val = domain_df[domain_df['problem'].isin(problems[train_split:train_split+valid_split])]
            test = domain_df[domain_df['problem'].isin(problems[train_split+valid_split:])]

            def extract_xy(df:pd.DataFrame):
                if task == 'classification':
                    y = df['coverage'].to_numpy(dtype=float)
                    x = df.drop(columns=['coverage', 'total_time', 'problem'])
                else:
                    df = df[df['total_time'] != ' NA']
                    y = df['total_time'].to_numpy(dtype=float)
                    x = df.drop(columns=['coverage', 'total_time'])

                return x, y
            
            xt, yt = extract_xy(train)
            xv, yv = extract_xy(val)
            xte, yte = extract_xy(test)

            X_train.append(xt)
            y_train.append(yt)
            X_val.append(xv)
            y_val.append(yv)
            X_test.append(xte)
            y_test.append(yte)

        X_train = pd.concat(X_train)
        y_train = np.concatenate(y_train)
        X_val = pd.concat(X_val)
        y_val = np.concatenate(y_val)
        X_test = pd.concat(X_test)
        y_test = np.concatenate(y_test)

        return (X_train, y_train), (X_val, y_val), (X_test, y_test)

def load_threshold_data(target:str):
    """
        Function which gets a target csv file as parameter
        and returns a train, val, test split. 
        
        There are two possibilities : Use one specific
        domain, or use multiple domains. 

        For the task there are also 2 possibilities. 
        either 'classification' or 'regression', where
        classification tackles the sol/usnol problem and
        regression, regresses the runtime
    """
    file_path = os.path.join(os.getcwd(), 'data','GNN_HorizonRegr_data', target)

    df = pd.read_csv(file_path)
    if df['domain'].nunique() == 1:
        problems = list(df['problem'].unique())

        # split the data grouped by problem instances
        train_split = int(0.7 * len(problems))
        valid_split = int(0.15 * len(problems))
        train = df[df['problem'].isin(problems[:train_split])]
        val = df[df['problem'].isin(problems[train_split:train_split+valid_split])]
        test = df[df['problem'].isin(problems[train_split+valid_split:])]

        def extract_xy(df:pd.DataFrame):
            y = df['horizon'].to_numpy(dtype=int)
            x = df.drop(columns=['coverage', 'total_time', 'problem', 'horizon'])

            return x, y
        
        return extract_xy(train), extract_xy(val), extract_xy(test)
    
    else:
        domains = list(df['domain'].unique())
        print(domains)

        X_train = []
        y_train = []
        X_val = []
        y_val = []
        X_test = []
        y_test = []

        for domain in domains:
            domain_df = df[df['domain'] == domain].copy()
            problems = list(domain_df['problem'].unique())

            # split the data grouped by problem instances
            train_split = int(0.7 * len(problems))
            valid_split = int(0.15 * len(problems))
            train = domain_df[domain_df['problem'].isin(problems[:train_split])]
            val = domain_df[domain_df['problem'].isin(problems[train_split:train_split+valid_split])]
            test = domain_df[domain_df['problem'].isin(problems[train_split+valid_split:])]

            def extract_xy(df:pd.DataFrame):
                y = df['horizon'].to_numpy(dtype=float)
                x = df.drop(columns=['coverage', 'total_time', 'problem', 'horizon'])

                return x, y
            
            xt, yt = extract_xy(train)
            xv, yv = extract_xy(val)
            xte, yte = extract_xy(test)

            X_train.append(xt)
            y_train.append(yt)
            X_val.append(xv)
            y_val.append(yv)
            X_test.append(xte)
            y_test.append(yte)

        X_train = pd.concat(X_train)
        y_train = np.concatenate(y_train)
        X_val = pd.concat(X_val)
        y_val = np.concatenate(y_val)
        X_test = pd.concat(X_test)
        y_test = np.concatenate(y_test)

        return (X_train, y_train), (X_val, y_val), (X_test, y_test)


