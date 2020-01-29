from anndata import AnnData
import loompy
import numpy as np
import os
from scanorama import *
import scanpy as sc
from scipy.sparse import vstack
from sklearn.preprocessing import normalize

from process import process, load_names, merge_datasets
from utils import *

NAMESPACE = 'zeisel_adolescent_brain'
DIMRED = 100
DR_METHOD = 'svd'

data_names = [
    'data/mouse_brain/zeisel/amygdala',
    'data/mouse_brain/zeisel/cerebellum',
    'data/mouse_brain/zeisel/cortex1',
    'data/mouse_brain/zeisel/cortex2',
    'data/mouse_brain/zeisel/cortex3',
    'data/mouse_brain/zeisel/hippocampus',
    'data/mouse_brain/zeisel/hypothalamus',
    'data/mouse_brain/zeisel/medulla',
    'data/mouse_brain/zeisel/midbraindorsal',
    'data/mouse_brain/zeisel/midbrainventral',
    'data/mouse_brain/zeisel/olfactory',
    'data/mouse_brain/zeisel/pons',
    'data/mouse_brain/zeisel/striatumdorsal',
    'data/mouse_brain/zeisel/striatumventral',
    'data/mouse_brain/zeisel/thalamus',
]

def keep_valid(datasets):
    barcode_sub_type = {}
    with loompy.connect('data/mouse_brain/zeisel/l6_r1.loom') as ds:
        for barcode, sub_type in zip(ds.ca['CellID'], ds.ca['ClusterName']):
        #for barcode, sub_type in zip(ds.ca['CellID'], ds.ca['Taxonomy_group']):
            barcode_sub_type[barcode] = sub_type

    valid_idx = []
    cell_types = []
    sub_types = []
    ages = []
    for data_name in data_names:
        with open('{}/meta.tsv'.format(data_name)) as f:
            excluded = set([
                'Blood', 'Excluded', 'Immune', 'Vascular',
            ])
            for j, line in enumerate(f):
                fields = line.rstrip().split('\t')
                if fields[1] == 'Neurons' and fields[2] != '?':
                    valid_idx.append(j)
                    cell_types.append(fields[1])
                    if fields[0] in barcode_sub_type:
                        sub_types.append(barcode_sub_type[fields[0]])
                    else:
                        sub_types.append('NA')
                    try:
                        age = float(fields[2][1:])
                    except ValueError:
                        age = fields[2]
                        if age == 'p12, p35':
                            age = (12 + 35) / 2.
                        elif age == 'p16, p24':
                            age = (16 + 24) / 2.
                        elif age == 'p19, p21':
                            age = (19 + 21) / 2.
                        elif age == 'p21-23' or age == 'p21, p23':
                            age = (21 + 23) / 2.
                        elif age == 'p22-24':
                            age = (22 + 24) / 2.
                        elif age == 'p25-27':
                            age = (25 + 27) / 2.
                        elif age == '6w':
                            age = 7 * 6.
                        else:
                            continue
                    min_age = 19.
                    max_age = 60.
                    offset = (age - min_age) / (max_age - min_age) * 3
                    ages.append(19 + offset)

    return valid_idx, np.array(cell_types), np.array(ages), np.array(sub_types)

datasets, genes_list, n_cells = load_names(data_names, norm=False)
qc_idx, cell_types, ages, sub_types = keep_valid(datasets)
datasets, genes = merge_datasets(datasets, genes_list)

X = vstack(datasets)
X = X[qc_idx]

qc_idx = [ i for i, s in enumerate(np.sum(X != 0, axis=1))
           if s >= 500 ]
tprint('Found {} valid cells among all datasets'.format(len(qc_idx)))
X = X[qc_idx]
cell_types = cell_types[qc_idx]
sub_types = sub_types[qc_idx]
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
dataset.obs['sub_types'] = [ NAMESPACE + '_' + l for l in sub_types ]
dataset.obs['ages'] = ages
datasets = [ dataset ]
namespaces = [ NAMESPACE ]
