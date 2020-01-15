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

NAMESPACE = 'masuda_mouse_microglia'
DIMRED = 100
DR_METHOD = 'svd'

data_names = [
    'data/microglia/masuda2019/mouse/GSE120744_counts',
    'data/microglia/masuda2019/mouse/GSE120745_geosubmission_counts',
]

def load_meta(datasets):
    n_valid = 0
    qc_idx = []
    ages = []
    injured = []
    cell_types = []

    # Load metadata for each cell.

    id_to_meta = {}
    soft_fnames = [
        'data/microglia/masuda2019/mouse/GSE120744_family.soft.gz',
        'data/microglia/masuda2019/mouse/GSE120745_family.soft.gz',
    ]
    for fname in soft_fnames:
        gsms = GEOparse.get_GEO(filepath=fname, silent=True).gsms
        for geo_id in gsms:
            cell_id = gsms[geo_id].metadata['title'][0]
            meta = {
                attr.split(':')[0].strip(): attr.split(':')[1].strip()
                for attr in gsms[geo_id].metadata['characteristics_ch1']
            }
            id_to_meta[cell_id] = meta

    # Map cell ids to important attributes.

    for i in range(len(datasets)):
        with gzip.open(data_names[i] + '.tsv.gz') as f:
            cell_ids = f.readline().decode('utf-8').rstrip().split()[1:]

        for cell_id in cell_ids:
            meta = id_to_meta[cell_id]

            age_str = meta['age']
            if age_str == '16 weeks':
                age = 16 * 7
                age_str = 'P{}'.format(age)
            elif age_str == 'embryonal': # Sic.
                age = 16.5
                age_str = 'E{}'.format(age)
            elif age_str == '03_w':
                age = 3 * 7
                age_str = 'P{}'.format(age)
            elif age_str == '16_w':
                age = 16 * 7
                age_str = 'P{}'.format(age)
            else:
                continue
            if age_str.startswith('P'):
                min_age = 19.
                max_age = 60.
                age = 19 + ((age - min_age) / (max_age - min_age) * 3)
            ages.append(age)

            if 'treatment' in meta:
                if 'demyelination' in meta['treatment']:
                    inj = 'demyelination'
                elif 'remyelination' in meta['treatment']:
                    inj = 'remyelination'
                elif 'Facial_nerve_axotomy' in meta['treatment']:
                    inj = 'fxn'
                else:
                    inj = 'none'
            else:
                inj = 'none'
            injured.append(inj)

            cell_types.append('{}_{}'.format(age_str, inj))

            qc_idx.append(n_valid)
            n_valid += 1

    return qc_idx, np.array(cell_types), np.array(ages), np.array(injured)

datasets, genes_list, n_cells = load_names(data_names, norm=False)
qc_idx, cell_types, ages, injured = load_meta(datasets)
datasets, genes = merge_datasets(datasets, genes_list)

X = vstack(datasets)
X = X[qc_idx]

qc_idx = [ i for i, s in enumerate(np.sum(X != 0, axis=1))
           if s >= 500 ]
tprint('Found {} valid cells among all datasets'.format(len(qc_idx)))
X = X[qc_idx]
cell_types = cell_types[qc_idx]
ages = ages[qc_idx]
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
dataset.obs['ages'] = ages
dataset.obs['injured'] = injured
datasets = [ dataset ]
namespaces = [ NAMESPACE ]
