from pathlib import Path

import numpy as np
from sklearn.metrics import classification_report


resultdir = Path('results/CRF_log/type78')

y_trues = []
y_preds = []

for k in range(5):
    kresultdir = resultdir / f'CRF+LDA_pathL_CV5-{k}_multi-col' / 'outputs'
    y_trues.append(np.load(kresultdir / 'y_true.npy'))
    y_preds.append(np.load(kresultdir / 'y_pred_epoch_3.npy'))

y_true = np.concatenate(y_trues)
y_pred = np.concatenate(y_preds)
report = classification_report(y_true, y_pred, output_dict=True)
import pdb; pdb.set_trace()
