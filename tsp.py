import numpy as np

from pyhermes import Problem


def solver_nearest_neighbor(tsp):
    cities = np.arange(tsp.n_cities)
    visited = np.array([False] * tsp.n_cities)
    last_city = 0

    tsp.tour[tsp.tour_len] = last_city
    tsp.tour_len += 1
    visited[last_city] = True

    while tsp.tour_len < tsp.n_cities:
        min_dist = np.inf

        for city, in_visited in zip(cities, visited):
            if in_visited:
                continue
            elif tsp.distance[last_city, city] < min_dist:
                min_dist = tsp.distance[last_city, city]
                nearest_city = city

        last_city = nearest_city

        tsp.tour[tsp.tour_len] = last_city
        tsp.tour_len += 1
        visited[last_city] = True


class TSP(Problem):
    solver = {
        'nearest_neighbor': solver_nearest_neighbor,
    }


    def __init__(self, distance):
        self.n_cities = len(distance)
        self.distance = distance
        self.clear_solution()


    def clear_solution(self):
        self.tour = np.arange(self.n_cities)
        self.tour_len = 0


    def eval(self):
        edges = np.column_stack((self.tour, np.roll(self.tour, -1)))
        distance = 0

        for edge in edges:
            distance += self.distance[edge[0], edge[1]]

        return distance


    def show_results(self):
        print('=== TSP ===')
        print(f'Total distance: {self.eval():,.2f}\n')


    def solve(self, heuristic):
        TSP.solver[heuristic](self)

        assert len(np.unique(self.tour)) == len(self.tour)


    def solveHH(self, hyperHeuristic):
        raise Exception("Method not implemented yet.")


    def getFeature(self, feature):
        raise Exception("Method not implemented yet.")


    def getObjValue(self):
        return self.eval()


def load_self(filename):
    raise Exception('Function not implemented yet')
