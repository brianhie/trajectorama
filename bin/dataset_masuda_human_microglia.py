from anndata import AnnData
import GEOparse
import gzip
import numpy as np
import os
from scanorama import *
import scanpy as sc
from scipy.sparse import vstack
from sklearn.preprocessing import normalize

from process import process, load_names, merge_datasets
from utils import *

NAMESPACE = 'masuda_human_microglia'
DIMRED = 100
DR_METHOD = 'svd'

data_names = [
    'data/microglia/masuda2019/human/GSM3529822_MS_case1_3.coutt',
    'data/microglia/masuda2019/human/GSM3529823_MS_case1_4.coutt',
    'data/microglia/masuda2019/human/GSM3529824_MS_case1_Pl1_1_9.coutt',
    'data/microglia/masuda2019/human/GSM3529825_MS_case1_Pl1_2_10.coutt',
    'data/microglia/masuda2019/human/GSM3529826_MS_case1_Pl2_1_9.coutt',
    'data/microglia/masuda2019/human/GSM3529827_MS_case1_Pl2_2_10.coutt',
    'data/microglia/masuda2019/human/GSM3529828_MS_case2_Plate1_1_3.coutt',
    'data/microglia/masuda2019/human/GSM3529829_MS_case2_Plate1_2_4.coutt',
    'data/microglia/masuda2019/human/GSM3529830_MS_case2_Plate2_1_7.coutt',
    'data/microglia/masuda2019/human/GSM3529831_MS_case2_Plate2_2_8.coutt',
    'data/microglia/masuda2019/human/GSM3529832_MS_case4_18_Plate1_2.coutt',
    'data/microglia/masuda2019/human/GSM3529833_MS_case4_18_Plate2_1.coutt',
    'data/microglia/masuda2019/human/GSM3529834_MS_case4_18_Plate2_2.coutt',
    'data/microglia/masuda2019/human/GSM3529835_MS_case5_18_Plate1_2.coutt',
    'data/microglia/masuda2019/human/GSM3529836_MS_case5_18_Plate2_1.coutt',
    'data/microglia/masuda2019/human/GSM3529837_MS_case5_18_Plate2_2.coutt',
    'data/microglia/masuda2019/human/GSM3529838_Pat1_GM_1_5.coutt',
    'data/microglia/masuda2019/human/GSM3529839_Pat1_GM_2_6.coutt',
    'data/microglia/masuda2019/human/GSM3529840_Pat1_WM_1_11.coutt',
    'data/microglia/masuda2019/human/GSM3529841_Pat1_WM_1_12.coutt',
    'data/microglia/masuda2019/human/GSM3529842_Pat2_GM_2_2.coutt',
    'data/microglia/masuda2019/human/GSM3529843_Pat2_WM_1_3.coutt',
    'data/microglia/masuda2019/human/GSM3529844_Pat2_WM_2_4.coutt',
    'data/microglia/masuda2019/human/GSM3529845_Pat3_GM_1_5.coutt',
    'data/microglia/masuda2019/human/GSM3529846_Pat3_GM_2_6.coutt',
    'data/microglia/masuda2019/human/GSM3529847_Pat3_WM_2_8.coutt',
    'data/microglia/masuda2019/human/GSM3529848_Pat4_Pl1_1_5.coutt',
    'data/microglia/masuda2019/human/GSM3529849_Pat4_Pl1_2_6.coutt',
    'data/microglia/masuda2019/human/GSM3529850_Pat4_Pl2_1_7.coutt',
    'data/microglia/masuda2019/human/GSM3529851_Pat4_Pl2_1_8.coutt',
    'data/microglia/masuda2019/human/GSM3529852_Pat5_1_1.coutt',
    'data/microglia/masuda2019/human/GSM3529853_Pat5_2_2.coutt',
]

def load_meta(datasets):
    injured = []
    cell_types = []

    for i in range(len(datasets)):
        if 'MS' in data_names[i]:
            injured += [ 'ms' ] * datasets[i].shape[0]
        else:
            injured += [ 'none' ] * datasets[i].shape[0]

    return np.array(injured), np.array(injured)

datasets, genes_list, n_cells = load_names(data_names, norm=False)
cell_types, injured = load_meta(datasets)
datasets, genes = merge_datasets(datasets, genes_list,
                                 union=True, verbose=False)

X = vstack(datasets)

qc_idx = [ i for i, s in enumerate(np.sum(X != 0, axis=1))
           if s >= 500 ]
tprint('Found {} valid cells among all datasets'.format(len(qc_idx)))
X = X[qc_idx]
cell_types = cell_types[qc_idx]
injured = injured[qc_idx]

if not os.path.isfile('data/dimred/{}_{}.txt'
                      .format(DR_METHOD, NAMESPACE)):
    mkdir_p('data/dimred')
    tprint('Dimension reduction with {}...'.format(DR_METHOD))
    X_dimred = reduce_dimensionality(normalize(X), dim_red_k=DIMRED)
    tprint('Dimensionality = {}'.format(X_dimred.shape[1]))
    np.savetxt('data/dimred/{}_{}.txt'
               .format(DR_METHOD, NAMESPACE), X_dimred)
else:
    X_dimred = np.loadtxt('data/dimred/{}_{}.txt'
                          .format(DR_METHOD, NAMESPACE))

dataset = AnnData(X)
dataset.var['gene_symbols'] = genes
dataset.obs['cell_types'] = [ NAMESPACE + '_' + l for l in cell_types ]
dataset.obs['ages'] = [ -1 for l in cell_types ]
dataset.obs['injured'] = injured
datasets = [ dataset ]
namespaces = [ NAMESPACE ]
