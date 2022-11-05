import numpy as np
import pandas as pd

from ttp import *


instances = pd.read_csv('instance_results.csv', index_col='instance')
results = {'solver': [], 'best': []}

for name, row in instances.iterrows():
    ttp = load_ttp('instances/' + name)
    ttp.solve('default')
    score = ttp.eval()

    if score < row['profit']:
        results['solver'].append(score)
        results['best'].append(row['profit'])

df = pd.DataFrame(results)
df['error'] = 100 * np.abs(df['solver'] - df['best']) / df['best']
print(df['error'].describe().T)
