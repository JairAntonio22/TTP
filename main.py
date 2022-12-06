import numpy as np
import pandas as pd

from ttp import *
from tabulate import tabulate


instances = pd.read_csv('instance_results.csv', index_col='instance')

ttp_heuristics = ['default', 'closest', 'farthest']
tsp_heuristics = ['default', 'nearest_neighbor', 'greedy']
kp_heuristics = ['default', 'max_profit', 'max_density', 'min_weight']


def run_test(ttp_h, tsp_h, kp_h):
    error = []

    for instance, (_, _, best) in instances.iterrows():
        ttp = load_ttp('instances/' + instance)
        ttp.solve(ttp_h + '/' + tsp_h + '/' + kp_h)

        score = ttp.eval()

        error.append(100 * np.abs(score - best) / best)

    error = np.array(error)
    mean = np.mean(error)
    std = np.std(error)

    print(tabulate([
        ['ttp', ttp_h],
        ['tsp', tsp_h],
        ['kp', kp_h],
        ['error mean', '%.2f%%' % mean],
        ['error std', '%.2f%%' % std],
    ], tablefmt='simple_grid'))


def run_tests():
    for ttp_h in ttp_heuristics:
        for tsp_h in tsp_heuristics:
            for kp_h in kp_heuristics:
                run_test(ttp_h, tsp_h, kp_h)


def get_summary(array):
    summary = []
    summary.append(np.min(array))
    summary.append(np.max(array))
    summary.append(np.mean(array))
    summary.append(np.median(array))
    summary.append(np.std(array))
    return summary


def get_features(ttp):
    features = []

    features += [ttp.tsp.n_cities]
    features += get_summary(ttp.tsp.distance[np.nonzero(ttp.tsp.distance)])

    features += [ttp.kp.n_items]
    features += [ttp.kp.capacity]
    features += get_summary(ttp.kp.weight)
    features += get_summary(ttp.kp.weight)

    features += [ttp.speed_min]
    features += [ttp.speed_max]
    features += [ttp.rent]

    return features


def create_dataset():
    features = [
        'n_cities',
        'min_dist', 'max_dist', 'mean_dist', 'median_dist', 'std_dist',
        'n_items',
        'capacity',
        'min_weight', 'max_weight', 'mean_weight', 'median_weight', 'std_weight',
        'min_profit', 'max_profit', 'mean_profit', 'median_profit', 'std_profit',
        'speed_min', 'speed_max',
        'rent',
        'ttp_h', 'tsp_h', 'kp_h',
        'error'
    ]

    dataset = dict()

    for instance, (_, _, best) in instances.iterrows():
        ttp = load_ttp('instances/' + instance)

        best_heuristics = ('', '', '')
        lowest_error = np.inf

        for ttp_h in ttp_heuristics:
            for tsp_h in tsp_heuristics:
                for kp_h in kp_heuristics:
                    ttp.solve(ttp_h + '/' + tsp_h + '/' + kp_h)

                    score = ttp.eval()
                    error = np.abs(score - best) / best

                    print(
                        ttp_heuristics.index(ttp_h),
                        tsp_heuristics.index(tsp_h),
                        kp_heuristics.index(kp_h),
                        score, error
                    )
                            

                    if error < lowest_error:
                        lowest_error = error
                        best_heuristics = (
                            ttp_heuristics.index(ttp_h),
                            tsp_heuristics.index(tsp_h),
                            kp_heuristics.index(kp_h)
                        )

                    ttp.clear_solution()

        dataset[instance] = get_features(ttp) + list(best_heuristics) + [lowest_error]
        break

    df = pd.DataFrame.from_dict(dataset, orient='index', columns=features)
    # df.to_csv('dataset.csv')


if __name__ == '__main__':
    create_dataset()
