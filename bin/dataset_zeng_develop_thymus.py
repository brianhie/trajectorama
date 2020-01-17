from anndata import AnnData
import numpy as np
import os
from scanorama import *
import scanpy as sc
from scipy.sparse import vstack
from sklearn.preprocessing import normalize

from process import process, load_names, merge_datasets
from utils import *

NAMESPACE = 'zeng_develop_thymus'
DIMRED = 100
DR_METHOD = 'svd'

data_names = [
    'data/thymus/zeng2019/GSM3906003_w8_thymus_10x_rawdata',
    'data/thymus/zeng2019/GSM3906005_w9_thymus_10x_rawdata',
    'data/thymus/zeng2019/GSM3906004_w10_thymus_10x_rawdata',
]

datasets, genes_list, n_cells = load_names(data_names, norm=False)
datasets, genes = merge_datasets(datasets, genes_list)

X = vstack(datasets)

qc_idx = [ i for i, s in enumerate(np.sum(X != 0, axis=1))
           if s >= 500 ]
tprint('Found {} valid cells among all datasets'.format(len(qc_idx)))

X = X[qc_idx]

cell_types = np.array(
    open('data/thymus/zeng2019/zeng_develop_thymus_cluster.txt')
    .read().rstrip().split('\n')
)

hema_idx = cell_types == 'Hema'
X = X[hema_idx]
cell_types = cell_types[hema_idx]

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
datasets = [ dataset ]
namespaces = [ NAMESPACE ]
