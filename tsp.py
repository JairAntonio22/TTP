import numpy as np

from pyhermes import Problem


def next_city_default(tsp):
    for city in range(tsp.n_cities):
        if city not in tsp.visited:
            return city


def next_city_nearest_neighbor(tsp):
    if tsp.tour_len == 0:
        tsp.tour[0] = 0
        tsp.tour_len += 1
        tsp.visited.add(0)

    last_city = tsp.tour[tsp.tour_len - 1]
    min_dist = np.inf
    nearest_city = 0

    for city in range(tsp.n_cities):
        if city in tsp.visited:
            continue

        if tsp.distance[last_city, city] < min_dist:
            min_dist = tsp.distance[last_city, city]
            nearest_city = city

    return nearest_city


def next_city_greedy(tsp):
    if tsp.tour_len == 0:
        edges = np.argsort(tsp.distance.ravel())
        i = 0

        while edges[i] == 0:
            i += 1

        tsp.tour[0], tsp.tour[1] = np.unravel_index(i, tsp.distance.shape)
        tsp.tour_len += 2
        tsp.visited.add(tsp.tour[0])
        tsp.visited.add(tsp.tour[1])

    c1 = tsp.tour[0]
    c2 = tsp.tour[tsp.tour_len - 1]

    min_dist1 = np.inf
    nearest_city1 = 0

    min_dist2 = np.inf
    nearest_city2 = 0

    for city in range(tsp.n_cities):
        if city in tsp.visited:
            continue

        if tsp.distance[c1, city] < min_dist1:
            min_dist1 = tsp.distance[c1, city]
            nearest_city1 = city

        if tsp.distance[c2, city] < min_dist2:
            min_dist2 = tsp.distance[c2, city]
            nearest_city2 = city

    if min_dist1 < min_dist2:
        return nearest_city1
    else:
        return nearest_city2


class TSP(Problem):
    next_city = {
        'default':          next_city_default,
        'nearest_neighbor': next_city_nearest_neighbor,
        'greedy':           next_city_greedy,
    }


    def __init__(self, distance):
        self.n_cities = len(distance)
        self.distance = distance
        self.clear_solution()


    def clear_solution(self):
        self.tour = np.arange(self.n_cities)
        self.tour_len = 0
        self.visited = set()


    def eval(self):
        return np.sum([
            self.distance[edge[0], edge[1]]
            for edge in np.column_stack((self.tour, np.roll(self.tour, -1)))
        ])


    def show_results(self):
        print('=== TSP ===')
        print(f'Total distance: {self.eval():,.2f}\n')


    def solve(self, heuristic):
        while self.tour_len < self.n_cities:
            city = TSP.next_city[heuristic](self)
            self.tour[self.tour_len] = city
            self.tour_len += 1
            self.visited.add(city)

        assert len(np.unique(self.tour)) == len(self.tour)


    def solveHH(self, hyperHeuristic):
        raise Exception("Method not implemented yet.")


    def getFeature(self, feature):
        raise Exception("Method not implemented yet.")


    def getObjValue(self):
        return self.eval()


def load_self(filename):
    raise Exception('Function not implemented yet')
