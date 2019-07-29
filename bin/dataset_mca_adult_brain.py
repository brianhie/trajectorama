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

NAMESPACE = 'mca_adult_brain'
DIMRED = 100
DR_METHOD = 'svd'

data_names = [
    'data/murine_atlases/mca/Brain1',
    'data/murine_atlases/mca/Brain2',
]

def keep_valid(datasets):
    id_to_type = {}
    with open('data/murine_atlases/mca/MCA_CellAssignments.csv') as f:
        f.readline()
        for line in f:
            fields = line.rstrip().split(',')
            id_to_type[fields[1]] = fields[6]

    ids = []
    for data_name in data_names:
        with open('{}.txt'.format(data_name)) as f:
            ids += f.readline().rstrip().split()[1:]

    valid_idx = []
    cell_types = []
    ages = []
    for j, cell_id in enumerate(ids):
        if cell_id in id_to_type:
            valid_idx.append(j)
            cell_types.append(id_to_type[cell_id])
            ages.append(20)

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
dataset.obs['cell_types'] = [ 'mca_han_etal_adult_' + l for l in cell_types ]
dataset.obs['ages'] = ages
datasets = [ dataset ]
namespaces = [ NAMESPACE ]
