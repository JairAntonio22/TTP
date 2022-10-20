import numpy as np

from pyhermes import Problem


def next_item_default(kp):
    for candidate in range(kp.n_items):
        if kp.picked_item[candidate]:
            continue

        if kp.curr_weight + kp.weight[candidate] < kp.capacity:
            return candidate


def next_item_max_profit(kp):
    candidates = np.argsort(-kp.profit)

    for candidate in candidates:
        if kp.picked_item[candidate]:
            continue

        if kp.curr_weight + kp.weight[candidate] < kp.capacity:
            return candidate


def next_item_max_density(kp):
    candidates = np.argsort(-(kp.profit / kp.weight))

    for candidate in candidates:
        if kp.picked_item[candidate]:
            continue

        if kp.curr_weight + kp.weight[candidate] < kp.capacity:
            return candidate


def next_item_min_weight(kp):
    candidates = np.argsort(kp.weight)

    for candidate in candidates:
        if kp.picked_item[candidate]:
            continue

        if kp.curr_weight + kp.weight[candidate] < kp.capacity:
            return candidate


class KP(Problem):
    next_item = {
        'default'       : next_item_default,
        'max_profit'    : next_item_max_profit,
        'max_density'   : next_item_max_density,
        'min_weight'    : next_item_min_weight,
    }


    def __init__(self, items, capacity):
        self.n_items = len(items)
        self.profit = np.array([p for p, _ in items])
        self.weight = np.array([v for _, v in items])
        self.capacity = capacity
        self.clear_solution()


    def clear_solution(self):
        self.picked_item = np.array([False] * self.n_items)
        self.curr_weight = 0


    def eval(self):
        return np.sum(self.profit[self.picked_item])


    def show_results(self):
        print('=== KP ===')
        print(f'Total profit: {self.eval():,.2f}\n')


    def solve(self, heuristic):
        item = KP.next_item[heuristic](self)

        while item != None:
            self.picked_item[item] = True
            self.curr_weight += self.weight[item]
            item = KP.next_item[heuristic](self)


    def solveHH(self, hyperHeuristic):
        raise Exception("Method not implemented yet.")


    def getFeature(self, feature):
        raise Exception("Method not implemented yet.")


    def getObjValue(self):
        return -self.eval()


def load_kp(filename):
    raise Exception('Function not implemented yet')
