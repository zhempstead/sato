from analysis import data_gen_all
import os
basepath = os.environ['BASEPATH']

path = os.path.join(basepath, 'results/CRF_log/type78/CRF_path')
pathL = os.path.join(basepath, 'results/CRF_log/type78/CRF+LDA_pathL')

data_gen_all(path, path_L, 'multi-col', './output')