import json
from pathlib import Path
import re

import pandas as pd

MATCH = re.compile(r'(.*)\+([0-9]+)-')

MULTI = [(p.name, p.parent.name) for p in Path('table_data/sato_tables/multionly').glob('*/*')]
ALL = [(p.name, p.parent.name) for p in Path('table_data/sato_tables/all').glob('*/*')]

df_multi = pd.DataFrame({"fname": [p[0] for p in MULTI], "k": [int(p[1][1]) for p in MULTI]})
df_all = pd.DataFrame({"fname": [p[0] for p in ALL], "k": [int(p[1][1]) for p in ALL]})

dfs_sato = []

SATO = {}

for corpus in ['webtables1-p1', 'webtables2-p1']:
    with open(f'extract/out/train_test_split/{corpus}_type78_multi-col.json') as f:
        loaded = json.load(f)
    for split in ['test', 'train']:
        df = pd.DataFrame({"entry": loaded[split]})
        df['corpus'] = corpus
        df['split'] = split
        dfs_sato.append(df)

df_sato = pd.concat(dfs_sato)
import pdb; pdb.set_trace()

def update_with_prefix(df):
    df['a'] = df['fname'].str.split('.json.gz_')
    df['b'] = df['a'].str[1].str.split('-')
    df['prefix'] = df['a'].str[0] + '.json.gz_' + df['b'].str[0] + '-'
    df.drop(columns=['a', 'b'], inplace=True)

def to_fname_prefix(df):
    df['a'] = '0_' + df['entry'].str.replace('/', ';')
    df['b'] = df['a'].str.split('+')
    df['c'] = df['b'].str[1].str.split('-')
    df['prefix'] = df['b'].str[0] + '_' + df['c'].str[0] + '-'
    df.drop(columns=['a', 'b', 'c'], inplace=True)

update_with_prefix(df_multi)
update_with_prefix(df_all)
to_fname_prefix(df_sato)

df = df_multi.merge(df_sato, on='prefix')
df.to_csv('full_df.csv', index=False)
out_struct = {}
for (corpus, k), k_df in df.groupby(['corpus', 'k']):
    if corpus not in out_struct:
        out_struct[corpus] = {}
    out_struct[corpus][f'K{k}'] = list(k_df['entry'])
import pdb; pdb.set_trace()
for corpus in ['webtables1-p1', 'webtables2-p1']:
    with open(f'extract/out/train_test_split/CV5_{corpus}_type78_multi-col.json', 'w') as f:
        f.write(json.dumps(out_struct[corpus]))
