import numpy as np

from pyhermes import Problem
from tsp import TSP
from kp import KP


def get_city_default(ttp, item):
    for city, available_items in enumerate(ttp.availability):
        if len(available_items == item) > 0:
            return city


class TTP(Problem):
    get_city = {
        'default':  get_city_default
    }


    def __init__(self, tsp, kp, availability, speed, rent):
        self.tsp = tsp
        self.kp = kp
        self.availability = availability
        self.speed_max = speed[1]
        self.speed_factor = (speed[1] - speed[0]) / self.kp.capacity
        self.rent = rent
        self.clear_solution()


    def clear_solution(self):
        self.picking_plan = -np.ones(self.kp.n_items)


    def eval(self):
        items_to_pick = dict()

        for item, city in enumerate(self.picking_plan):
            if city == -1:
                continue

            if city in items_to_pick:
                items_to_pick[city].append(item)
            else:
                items_to_pick[city] = [item]

        profit, weight, time = 0, 0, 0
        edges = np.column_stack((self.tsp.tour, np.roll(self.tsp.tour, -1)))

        for edge in edges:
            if edge[0] not in items_to_pick:
                continue

            for item in items_to_pick[edge[0]]:
                profit += self.kp.profit[item]
                weight += self.kp.weight[item]

            velocity = self.speed_max - weight * self.speed_factor
            time += self.tsp.distance[edge[0], edge[1]] / velocity

        return profit - time * self.rent


    def show_results(self):
        print('===== TTP =====')
        print(f'Total profit: {self.eval():,.2f}\n')

        self.tsp.show_results()
        self.kp.show_results()


    def solve(self, heuristic):
        self.tsp.solve('nearest_neighbor')
        self.kp.solve('max_density')

        for item in range(self.kp.n_items):
            if self.kp.picked_item[item]:
                self.picking_plan[item] = TTP.get_city[heuristic](self, item)


    def solveHH(self, hyperHeuristic):
        raise Exception("Method not implemented yet.")


    def getFeature(self, feature):
        raise Exception("Method not implemented yet.")


    def getObjValue(self):
        return -self.eval()


def load_ttp(filename):
    with open(filename) as file:
        _ = file.readline()
        _ = file.readline()
        n_cities = int(file.readline().split()[-1])
        n_items = int(file.readline().split()[-1])
        capacity = int(file.readline().split()[-1])
        min_speed = float(file.readline().split()[-1])
        max_speed = float(file.readline().split()[-1])
        rent = float(file.readline().split()[-1])
        _ = file.readline()
        _ = file.readline()

        city_pos = np.empty([n_cities, 2])

        for i in range(n_cities):
            line = file.readline()
            line = line.split()
            city_pos[i] = np.array([float(coord) for coord in line[1:]])

        _ = file.readline()

        items = []
        availability = [[]] * n_cities

        for i in range(n_items):
            line = file.readline()
            line = line.split()
            line = [int(value) for value in line[1:]]

            items.append((line[0], line[1]))

            city = line[2] - 1
            availability[city].append(i)

        availability = np.array(availability, dtype=list)

    distance = np.zeros([n_cities, n_cities])

    for c1 in range(n_cities):
        for c2 in range(n_cities):
            c1_pos = city_pos[c1]
            c2_pos = city_pos[c2]
            distance[c1][c2] = np.linalg.norm(c1_pos - c2_pos)

    tsp = TSP(distance)
    kp = KP(items, capacity)
    speed = (min_speed, max_speed)

    return TTP(tsp, kp, availability, speed, rent)


if __name__ == '__main__':
    tsp = TSP(np.array([
        [0, 5, 6, 6],
        [5, 0, 5, 6],
        [6, 5, 0, 4],
        [6, 6, 4, 0]
    ]))

    kp = KP(
        items=[(100, 3), (40, 1), (40, 1), (20, 2), (30, 3)],
        capacity=3
    )

    availability = np.array([[3], [3], [3], [2, 4], [2]], dtype=list)
    speed = (0.1, 1)
    rent = 1

    ttp = TTP(tsp, kp, availability, speed, rent)

    ttp.tsp.tour = np.array([0, 2, 1, 3])
    ttp.picking_plan = np.array([-1, 2, -1, 1, -1])
    ttp.kp.picked_item = ttp.picking_plan >= 0

    ttp.show_results()
