import os
from pathlib import Path
import sys

import pandas as pd
import numpy as np
rand = np.random

rand.seed(100)

DIR = Path('table_data') / 'sato_tables' / 'all'
INFILE = sys.argv[1]
OUTFILE = sys.argv[2]

LIMIT_ROWS = 10

def best_matches(arr1, arr2):
    if min(len(arr1), len(arr2)) == 0:
        return []
    if arr2[0].startswith(arr1[0]):
        matches = best_matches(arr1[1:], arr2[1:])
        return [(0, 0)] + [(m1+1, m2+1) for m1, m2 in matches]
    else:
        matches1 = best_matches(arr1, arr2[1:])
        matches1 = [(m1, m2+1) for m1, m2 in matches1]
        matches2 = best_matches(arr1[1:], arr2)
        matches2 = [(m1+1, m2) for m1, m2 in matches2]
        if len(matches2) > len(matches1):
            return matches2
        return matches1
    

df = pd.read_csv(INFILE)
prompt = []
completion = []
files = []
cols = []
good = 0
bad = 0
print(len(df['true_file'].unique()))
for fname, subset in df.groupby('true_file'):
    subset = subset.sort_values('col')
    fdf = pd.read_csv(DIR / fname)
    num_cols = min(len(fdf.columns), len(subset))
    best_match_idxs = best_matches(list(subset['label']), list(fdf.columns))
    subset_idxs = [i1 for i1, i2 in best_match_idxs]
    fdf_idxs = [i2 for i1, i2 in best_match_idxs]
    if len(best_match_idxs) < num_cols:
        #print(subset[['col', 'label']])
        #print(fdf)
        bad += 1
        continue
    subset = subset[subset['col'].isin(subset_idxs)]
    fdf = fdf[fdf.columns[fdf_idxs]]
    if len(fdf) > LIMIT_ROWS:
        ids = rand.choice(len(fdf), LIMIT_ROWS, replace=False)
        fdf = fdf.loc[ids]
    table_str = fdf.to_csv(index=False, header=False, sep='\t')
    ans_str =', '.join(list(subset['label']))
    prompt.append(f'{table_str}\n\n###\n\n')
    completion.append(f' {ans_str}\n')
    files.append(subset.iloc[0]['file'])
    cols.append(subset_idxs)
    good += 1
    print(good, bad)

out_df = pd.DataFrame({'file': files, 'cols': cols, 'prompt': prompt, 'completion': completion})

out_df.to_csv(OUTFILE, index=False)
