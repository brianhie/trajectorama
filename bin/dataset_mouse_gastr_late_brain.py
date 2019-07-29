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

NAMESPACE = 'mouse_gastr_late_brain'
DIMRED = 100
DR_METHOD = 'svd'

data_names = [
    'data/mouse_gastr_late',
]

def keep_valid(datasets):
    valid_idx = []
    cell_types = []
    ages = []
    with open('{}/cell_annotate.csv'.format(data_names[0])) as f:

        f.readline()

        for j, line in enumerate(f):
            fields = line.rstrip().split(',')
            if fields[19] == 'FALSE' and fields[20] == 'FALSE' and \
               fields[9] != 'NA' and int(fields[9]) >= 400 and \
               fields[23] == 'Neural tube and notochord trajectory':
                valid_idx.append(j)
                cell_types.append(fields[22].replace(' ', '_'))
                ages.append(float(fields[8]))

    tprint('Found {} valid cells among all datasets'.format(len(valid_idx)))

    return valid_idx, np.array(cell_types), np.array(ages)

datasets, genes_list, n_cells = load_names(data_names, norm=False)
qc_idx, cell_types, ages = keep_valid(datasets)
datasets, genes = merge_datasets(datasets, genes_list)

X = vstack(datasets)
X = X[qc_idx]

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
