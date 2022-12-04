import matplotlib.pyplot as plt
import os
from collections import defaultdict

def ratio_OverThreshold(y, predictions):
    y = [y[i:i + 100] for i in range(0, len(y), 100)]
    predictions = [predictions[i:i + 100] for i in range(0, len(predictions), 100)]

    over_threshold = []
    for problem_y, problem_pred in zip(y, predictions):
        val1 = problem_y.index(1)
        val2 = problem_pred.index(1)
        print(val1, val2)
        over = 1 if val1 <= val2 else 0
        over_threshold.append(over)

    return sum(over_threshold)/len(over_threshold)

def visualize_results(y, predictions, domains, filename):
    y = [y[i:i + 100] for i in range(0, len(y), 100)]
    predictions = [predictions[i:i + 100] for i in range(0, len(predictions), 100)]

    domain_y = defaultdict(list)
    domain_pred = defaultdict(list)
    maximum = 0
    for i, (problem_y, problem_pred) in enumerate(zip(y, predictions)):
        domain_y[domains[i*100]].append(problem_y.index(1))
        domain_pred[domains[i*100]].append(problem_pred.index(1))
        maximum = max(domain_y[domains[i*100]]) if max(domain_y[domains[i*100]]) > maximum else maximum
        maximum = max(domain_pred[domains[i*100]]) if max(domain_pred[domains[i*100]]) > maximum else maximum


    fig, ax = plt.subplots()
    ax.plot([0,maximum],[0,maximum])
    ax.set_ylabel('Predicted threshold')
    ax.set_xlabel('True threshold')
    ax.set_title(f'Threshold predictions {filename.split(".")[0]}.')
    for key in domain_y.keys():
        ax.scatter(domain_y[key], domain_pred[key], label=key)
    ax.legend()
    path = os.path.join(os.getcwd(), 'Results', filename)
    fig.savefig(path, dpi=300)

