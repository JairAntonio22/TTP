import numpy as np
import pandas as pd


class TSP:
    def __init__(self, distance):
        self.n_cities = len(distance)
        self.distance = distance
        self.tour = None

    def eval(self, get_total_dist=True):
        if len(np.unique(self.tour)) != len(self.tour):
            return False, 0
        elif get_total_dist:
            total = 0

            for i in range(self.n_cities):
                c1 = self.tour[i]
                c2 = self.tour[(i + 1) % self.n_cities]
                total += self.distance[c1, c2]

            return True, total
        else:
            return True, 0

    def show_results(self):
        if self.tour is not None:
            valid, distance = self.eval()

            print('=== Tour ===')
            print(f'Valid solution: {valid}')

            if valid:
                print(f'Total distance: {distance:8.2f}')

            print()


class KP:
    def __init__(self, items, capacity):
        self.n_items = len(items)
        self.profit = np.array([p for p, _ in items])
        self.weight = np.array([v for _, v in items])
        self.capacity = capacity
        self.picked_items = None

    def eval(self):
        weight = np.sum(self.weight[self.picked_items])
        
        if weight <= self.capacity:
            return True, (weight, np.sum(self.profit[self.picked_items]))
        else:
            return False, (0, 0)

    def show_results(self):
        if self.picked_items is not None:
            data = np.transpose(np.array([self.profit, self.weight]))
            valid, (weight, profit) = self.eval()

            print('=== Picked items ===')
            print(f'Valid solution: {valid}')

            if valid:
                print(f'Total profit: {profit:8.2f}')
                print(f'Total weight: {weight:8.2f}')

            print()


class TTP:
    def __init__(self, tsp, kp, availability, speed, rent):
        self.tsp = tsp
        self.kp = kp
        self.availability = availability
        self.speed_max = speed[1]
        self.speed_factor = (speed[1] - speed[0]) / self.kp.capacity
        self.rent = rent
        self.picking_plan = None

    def eval(self):
        valid_kp, (weight, profit) = self.kp.eval()
        valid_tsp, _ = self.tsp.eval(get_total_dist=False)

        if valid_kp and valid_tsp:
            items_to_pick = dict()

            for item, city in enumerate(self.picking_plan):
                if city == -1:
                    continue

                if city in items_to_pick:
                    items_to_pick[city].append(item)
                else:
                    items_to_pick[city] = [item]

            weight = 0
            time = 0

            for i in range(self.tsp.n_cities):
                c1 = self.tsp.tour[i]
                c2 = self.tsp.tour[(i + 1) % self.tsp.n_cities]

                if c1 in items_to_pick:
                    for item in items_to_pick[c1]:
                        weight += self.kp.weight[item]

                velocity = self.speed_max - weight * self.speed_factor
                time += self.tsp.distance[c1, c2] / velocity

            return True, profit - time * self.rent
        else:
            return False, 0

    def show_results(self):
        valid, profit = self.eval()

        print('=== TTP ===')
        print(f'Valid solution: {valid}')

        if valid:
            print(f'Total profit: {profit:8.2f}\n')
            self.tsp.show_results()
            self.kp.show_results()


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
            city_pos[i] = np.array([int(coord) for coord in line[1:]])

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

    distance = np.empty([n_cities, n_cities])

    for c1 in range(n_cities):
        for c2 in range(n_cities):
            c1_pos = city_pos[c1]
            c2_pos = city_pos[c2]
            distance[c1][c2] = np.linalg.norm(c1_pos - c2_pos)

    tsp = TSP(distance)
    kp = KP(items, capacity)
    speed = (min_speed, max_speed)

    return TTP(tsp, kp, availability, speed, rent)


def example():
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
    ttp.kp.picked_items = ttp.picking_plan >= 0

    ttp.show_results()


if __name__ == '__main__':
    # example()
    ttp = load_ttp('a280_n279_bounded-strongly-corr_01.ttp')
