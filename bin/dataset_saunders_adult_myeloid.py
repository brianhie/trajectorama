from anndata import AnnData
import numpy as np
import os
from scanorama import *
import scanpy as sc
from scipy.sparse import vstack
from sklearn.preprocessing import normalize

from process import process, load_names, merge_datasets
from utils import *

NAMESPACE = 'saunders_adult_myeloid'
DIMRED = 100
DR_METHOD = 'svd'

data_names = [
    'data/mouse_brain/dropviz/Cerebellum_ALT',
    'data/mouse_brain/dropviz/Cortex_noRep5_FRONTALonly',
    'data/mouse_brain/dropviz/Cortex_noRep5_POSTERIORonly',
    'data/mouse_brain/dropviz/EntoPeduncular',
    'data/mouse_brain/dropviz/GlobusPallidus',
    'data/mouse_brain/dropviz/Hippocampus',
    'data/mouse_brain/dropviz/Striatum',
    'data/mouse_brain/dropviz/SubstantiaNigra',
    'data/mouse_brain/dropviz/Thalamus',
]

data_name_abbrev = {
    'data/mouse_brain/dropviz/Cerebellum_ALT': 'CB',
    'data/mouse_brain/dropviz/Cortex_noRep5_FRONTALonly': 'FC',
    'data/mouse_brain/dropviz/Cortex_noRep5_POSTERIORonly': 'PC',
    'data/mouse_brain/dropviz/EntoPeduncular': 'ENT',
    'data/mouse_brain/dropviz/GlobusPallidus': 'GP',
    'data/mouse_brain/dropviz/Hippocampus': 'HC',
    'data/mouse_brain/dropviz/Striatum': 'STR',
    'data/mouse_brain/dropviz/SubstantiaNigra': 'SN',
    'data/mouse_brain/dropviz/Thalamus': 'TH',
}

def keep_valid(datasets):
    n_valid = 0
    qc_idx = []
    filter_labels = set([
        'doublet',
        'min_genes',
        'outlier',
        'small_cell',
    ])

    sub_type_common = {}
    with open('data/mouse_brain/dropviz/annotation.tsv') as f:
        f.readline() # Consume header.
        for line in f:
            fields = line.rstrip().split('\t')
            sub_type_common[fields[-1]] = fields[4].replace(' ', '_')

    sub_types = []
    injured = []

    for i in range(len(datasets)):
        valid_idx = []

        with open('{}/meta.txt'.format(data_names[i])) as f:
            n_lines = 0
            for j, line in enumerate(f):
                fields = line.rstrip().split()
                if fields[1] != 'NA':
                    valid_idx.append(j)
                    if fields[3] not in filter_labels:
                        sub_type = '{}_{}'.format(
                            data_name_abbrev[data_names[i]],
                            fields[2]
                        )
                        if sub_type in sub_type_common:
                            sub_types.append(sub_type_common[sub_type])
                        else:
                            sub_types.append('NA')
                            assert('-' not in sub_type)
                        injured.append(False)
                        qc_idx.append(n_valid)
                    n_valid += 1
                n_lines += 1

        assert(n_lines == datasets[i].shape[0])
        assert(len(qc_idx) <= n_valid)

        datasets[i] = datasets[i][valid_idx, :]
        tprint('{} has {} valid cells'
              .format(data_names[i], len(valid_idx)))

    tprint('Found {} cells among all datasets'.format(n_valid))
    tprint('Found {} valid cells among all datasets'.format(len(qc_idx)))

    return qc_idx, np.array(sub_types), np.array(injured)

datasets, genes_list, n_cells = load_names(data_names, norm=False)
qc_idx, sub_types, injured = keep_valid(datasets)
datasets, genes = merge_datasets(datasets, genes_list)

X = vstack(datasets)
X = X[qc_idx]

cell_types = np.array(
    open('data/mouse_brain/dropviz/mouse_brain_cluster.txt')
    .read().rstrip().split('\n')
)
cell_types = cell_types[qc_idx]

ages = np.array([ 21 ] * X.shape[0])

myeloid_idx = np.logical_or(
    cell_types == 'Macrophage', cell_types == 'Microglia'
)
X = X[myeloid_idx]
cell_types = cell_types[myeloid_idx]
sub_types = sub_types[myeloid_idx]
ages = ages[myeloid_idx]
injured = injured[myeloid_idx]

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
dataset.obs['sub_types'] = [ NAMESPACE + '_' + l for l in sub_types ]
dataset.obs['ages'] = ages
dataset.obs['injured'] = injured
datasets = [ dataset ]
namespaces = [ NAMESPACE ]
