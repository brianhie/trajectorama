import numpy as np
import os
from scanorama import *
from scipy.sparse import vstack
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import normalize, LabelEncoder

from process import process, load_names, merge_datasets
from utils import *

NAMESPACE = 'cortical_satija'
DIMRED = 100
DR_METHOD = 'svd'

data_names = [
    'data/mouse_brain/cortical/GSE103983_dropseq',
    'data/mouse_brain/cortical/GSE104156_digital_expression',
]

def keep_valid(datasets):
    ids = []
    for data_name in data_names:
        with open('{}.tsv'.format(data_name)) as f:
            ids += f.readline().rstrip().split('\t')[1:]

    valid_idx = []
    cell_types = []
    ages = []
    for j, cell_id in enumerate(ids):
        if not cell_id.startswith('CGE_E13.5_Lhx6neg') and \
           not cell_id.startswith('MGE_E13.5_Lhx6pos'):
            valid_idx.append(j)
            cell_types.append('neuron')
            if cell_id.startswith('CGE_'):
                ages.append(14.5)
            elif cell_id.startswith('MGE_'):
                ages.append(13.5)
            elif cell_id.startswith('LGE_'):
                ages.append(14.5)
            else:
                stage = cell_id.split('_')[1]
                if stage == 'P10':
                    ages.append(18)
                elif stage == 'E18.5':
                    ages.append(16)
                else:
                    ages.append(float(stage[1:]))

    return valid_idx, np.array(cell_types), np.array(ages)

datasets, genes_list, n_cells = load_names(data_names, norm=False)
qc_idx, cell_types, ages = keep_valid(datasets)
datasets, genes = merge_datasets(datasets, genes_list)

X = vstack(datasets)
X = X[qc_idx]

qc_idx = [ i for i, s in enumerate(np.sum(X != 0, axis=1))
           if s >= 500 ]
tprint('Found {} valid cells among all datasets'.format(len(qc_idx)))
X = X[qc_idx]
cell_types = cell_types[qc_idx]
ages = ages[qc_idx]

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
dataset.obs['ages'] = ages
datasets = [ dataset ]
namespaces = [ NAMESPACE ]
