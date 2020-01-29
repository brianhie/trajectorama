from anndata import AnnData
import GEOparse
import numpy as np
import os
from scanorama import *
import scanpy as sc
from scipy.sparse import vstack
from sklearn.preprocessing import normalize

from process import process, load_names, merge_datasets
from utils import *

NAMESPACE = 'hammond_microglia'
DIMRED = 100
DR_METHOD = 'svd'

data_names = [
    'data/microglia/hammond2019/GSM3442006_E14_F_B10.dge',
    'data/microglia/hammond2019/GSM3442007_E14_M_B11.dge',
    'data/microglia/hammond2019/GSM3442008_E14_F_B12.dge',
    'data/microglia/hammond2019/GSM3442009_E14_F_C1.dge',
    'data/microglia/hammond2019/GSM3442010_E14_M_B9.dge',
    'data/microglia/hammond2019/GSM3442011_E14_F_B6.dge',
    'data/microglia/hammond2019/GSM3442012_E14_M_B7.dge',
    'data/microglia/hammond2019/GSM3442013_E14_M_B8.dge',
    'data/microglia/hammond2019/GSM3442014_P5_M_A1.dge',
    'data/microglia/hammond2019/GSM3442015_P5_F_A2.dge',
    'data/microglia/hammond2019/GSM3442016_P4_M_A4.dge',
    'data/microglia/hammond2019/GSM3442017_P4_M_A5.dge',
    'data/microglia/hammond2019/GSM3442018_P4_F_A6.dge',
    'data/microglia/hammond2019/GSM3442019_P4_F_B3.dge',
    'data/microglia/hammond2019/GSM3442020_P4_F_B4.dge',
    'data/microglia/hammond2019/GSM3442021_P4_M_B5.dge',
    'data/microglia/hammond2019/GSM3442022_P30_Male_1.dge',
    'data/microglia/hammond2019/GSM3442023_P30_Male_2.dge',
    'data/microglia/hammond2019/GSM3442024_P30_male_3.dge',
    'data/microglia/hammond2019/GSM3442025_P30_male_4.dge',
    'data/microglia/hammond2019/GSM3442026_P100_Male_1.dge',
    'data/microglia/hammond2019/GSM3442027_P100_Male_2.dge',
    'data/microglia/hammond2019/GSM3442028_P100_female_1.dge',
    'data/microglia/hammond2019/GSM3442029_P100_female_2.dge',
    'data/microglia/hammond2019/GSM3442030_P100_male_3.dge',
    'data/microglia/hammond2019/GSM3442031_P100_male_4.dge',
    'data/microglia/hammond2019/GSM3442032_P100_female_3.dge',
    'data/microglia/hammond2019/GSM3442033_P100_female_4.dge',
    'data/microglia/hammond2019/GSM3442034_Old_male_1.dge',
    'data/microglia/hammond2019/GSM3442035_Old_male_2.dge',
    'data/microglia/hammond2019/GSM3442036_Old_male_3.dge',
    'data/microglia/hammond2019/GSM3442037_Old_male_4.dge',
    'data/microglia/hammond2019/GSM3442038_P100_M_A1.dge',
    'data/microglia/hammond2019/GSM3442039_P100_M_A2.dge',
    'data/microglia/hammond2019/GSM3442040_P100_M_B5.dge',
    'data/microglia/hammond2019/GSM3442041_P100_M_SALINE_A3.dge',
    'data/microglia/hammond2019/GSM3442042_P100_M_SALINE_A5.dge',
    'data/microglia/hammond2019/GSM3442043_P100_M_SALINE_B9.dge',
    'data/microglia/hammond2019/GSM3442044_P100_M_LPC_A4.dge',
    'data/microglia/hammond2019/GSM3442045_P100_M_LPC_A6.dge',
    'data/microglia/hammond2019/GSM3442046_P100_M_LPC_B10.dge',
    'data/microglia/hammond2019/GSM3442047_P5_female_nopercoll_1.dge',
    'data/microglia/hammond2019/GSM3442048_P5_female_nopercoll_2.dge',
    'data/microglia/hammond2019/GSM3442049_P5_female_nopercoll_3.dge',
    'data/microglia/hammond2019/GSM3442050_P5_female_percoll_1.dge',
    'data/microglia/hammond2019/GSM3442051_P5_female_percoll_2.dge',
    'data/microglia/hammond2019/GSM3442052_P5_female_percoll_3.dge',
]

def load_meta(datasets):
    qc_idx = []
    ages = []
    injured = []
    cell_types = []

    gsms = GEOparse.get_GEO(
        filepath='data/microglia/hammond2019/GSE121654_family.soft.gz',
        silent=True,
    ).gsms

    for i in range(len(datasets)):
        geo_id = data_names[i].split('/')[-1].split('_')[0]
        meta = {
            attr.split(':')[0].strip(): attr.split(':')[1].strip()
            for attr in gsms[geo_id].metadata['characteristics_ch1']
        }

        age_str = meta['age']
        if age_str.startswith('E'):
            age = float(age_str[1:])
        elif age_str.startswith('P'):
            age = float(age_str[1:])
            age = min(age, 300)
            min_age = 19.
            max_age = 60.
            age = 19 + ((age - min_age) / (max_age - min_age) * 3)
        else:
            raise ValueError('Unhandled age {}'.format(age_str))
        ages += [ age ] * datasets[i].shape[0]

        inj = 'demyelination' if 'Lysolecithin' in meta['treatment'] else 'none'
        injured += [ inj ] * datasets[i].shape[0]

        cell_types += [ '{}_{}'.format(age_str, inj) ] * datasets[i].shape[0]

    return np.array(cell_types), np.array(ages), np.array(injured)

datasets, genes_list, n_cells = load_names(data_names, norm=False)
cell_types, ages, injured = load_meta(datasets)
datasets, genes = merge_datasets(datasets, genes_list,
                                 union=True, verbose=False)

X = vstack(datasets)

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
