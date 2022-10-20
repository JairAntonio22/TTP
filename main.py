import numpy as np

from ttp import *
from glob import glob
from pprint import pprint


if __name__ == '__main__':
    filenames = glob('TTP Instances/berlin52-ttp/berlin52_n51_*')
    filenames.sort()

    for filename in filenames[:1]:
        ttp = load_ttp(filename)

        ttp.solve('default')
        ttp.show_results()
