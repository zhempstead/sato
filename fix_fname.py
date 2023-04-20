import os
from pathlib import Path
import sys

import pandas as pd
import pygtrie as trie


DIR = Path('table_data') / 'sato_tables' / 'all'

df = pd.read_csv(sys.argv[1])


file2dir = trie.CharTrie()
for kdir in os.listdir(DIR):
    all_kdir_files = os.listdir(DIR / kdir)
    for f in all_kdir_files:
        file2dir[f] = kdir

badfile2file = {}
match = 0
nomatch = 0
manymatch = 0
def fix_fname(fname):
    global match
    global nomatch
    global manymatch
    global badfile2file
    global badfile2dir
    orig_fname = fname
    fname = fname.replace('/', ';')
    fname = fname.replace('+', '_')
    l, r = fname.split('.json.gz_')
    r = r.split('-')[0]
    glob = f'0_{l}.json.gz_{r}-'
    
    try:
        results = file2dir.items(glob)
    except KeyError:
        nomatch += 1
        return
    if len(results) > 1:
        manymatch += 1
    else:
        match += 1
        badfile2file[orig_fname] = f'{results[0][1]}/{results[0][0]}'
        
    print(match, nomatch, manymatch)
    #NAME_DICT['orig_fname'] = f'{results.iloc[0]["dir"]}/{results.iloc[0]["f"]}'

for f in df['file'].unique():
    fix_fname(f)

df['true_file'] = df['file'].apply(lambda x: badfile2file.get(x, None))

df.to_csv(sys.argv[1], index=False)
