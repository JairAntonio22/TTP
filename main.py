import numpy as np
import pandas as pd

from ttp import *
from time import perf_counter
from tabulate import tabulate


def run_test(ttp_h, tsp_h, kp_h):
    error = []

    for instance, (_, _, best) in dataset.iterrows():
        ttp = load_ttp('instances/' + instance)
        ttp.solve(ttp_h + '/' + tsp_h + '/' + kp_h)
        score = ttp.eval()

        if score <= best:
            error.append(100 * np.abs(score - best) / best)

    error = np.array(error)
    mean = np.mean(error)
    std = np.std(error)

    print(tabulate([
        ['ttp', ttp_h], ['tsp', tsp_h], ['kp', kp_h],
        ['error mean', '%.2f%%' % mean], ['error std', '%.2f%%' % std],
    ], tablefmt='simple_grid'))


dataset = pd.read_csv('instance_results.csv', index_col='instance')

ttp_heuristics = ['default', 'closest', 'farthest']
tsp_heuristics = ['nearest_neighbor']
kp_heuristics = ['default', 'max_profit', 'max_density', 'min_weight']

for ttp_h in ttp_heuristics:
    for kp_h in kp_heuristics:
        for tsp_h in tsp_heuristics:
            run_test(ttp_h, tsp_h, kp_h)
