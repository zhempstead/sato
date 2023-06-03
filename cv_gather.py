from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report


resultdir = Path('results/CRF_log/type78')

y_trues = []
y_preds = []
dfs = []

source_df = pd.read_csv('source_df.csv')

for k in range(5):
    kresultdir = resultdir / f'CRF+LDA_pathL_CV5-{k}_multi-col' / 'outputs'
    test_df = pd.read_csv(kresultdir / 'test_df.csv')
    test_df['num_cols'] = test_df['labels'].str.count(',') + 1
    test_df['pred_sato'] = None
    y_trues.append(np.load(kresultdir / 'y_true.npy'))
    y_pred = np.load(kresultdir / 'y_pred_epoch_3.npy')
    y_preds.append(y_pred.copy())
    for idx, row in test_df.iterrows():
        row_pred = y_pred[:row['num_cols']]
        y_pred = y_pred[row['num_cols']:]
        test_df.loc[idx, 'pred_sato'] = '[' + ', '.join(row_pred) + ']'
    assert(len(y_pred) == 0)
    dfs.append(test_df)

y_true = np.concatenate(y_trues)
y_pred = np.concatenate(y_preds)
df = pd.concat(dfs)
report = classification_report(y_true, y_pred, output_dict=True)
df = df.merge(source_df, left_on='table_id', right_on='entry')
df = df.drop(columns='entry')
good = 0
extra = 0
missing = 0
for idx, row in df.iterrows():
    table_df = pd.read_csv(f"table_data/sato_tables/multionly/K{row['k']}/{row['fname']}")
    if len(table_df.columns) == row['num_cols']:
        good += 1
    elif len(table_df.columns) < row['num_cols']:
        missing += 1
    else:
        extra += 1
print(f"good: {good}, extra: {extra}, missing: {missing}")
df.to_csv('full_testset.csv')
import pdb; pdb.set_trace()
