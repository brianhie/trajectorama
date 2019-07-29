from anndata import AnnData
import gzip
import numpy as np
from process import load_names, merge_datasets
from utils import *

NAMESPACE = 'human_bonemarrow_ica'
DIMRED = 100
DR_METHOD = 'svd'

data_names = [
    'data/ica/ica_bone_marrow_h5',
]
namespaces = [
    'ica_bone_marrow',
]

[ X ], [ genes ], _ = load_names(data_names)

umi_sum = np.sum(X, axis=1)
gt_idx = [ i for i, s in enumerate(umi_sum)
           if s >= 500 ]
low_idx = [ idx for idx, gene in enumerate(genes)
            if gene.startswith('RPS') or gene.startswith('RPL') ]
lt_idx = [ i for i, s in enumerate(np.sum(X[:, low_idx], axis=1) / umi_sum)
           if s <= 0.5 ]

qc_idx = sorted(set(gt_idx) & set(lt_idx))
X = csr_matrix(X[qc_idx])

print('Found {} valid cells'.format(X.shape[0]))

if not os.path.isfile('data/dimred/{}_{}.txt'
                      .format(DR_METHOD, NAMESPACE)):
    mkdir_p('data/dimred')
    print('Dimension reduction with {}...'.format(DR_METHOD))
    X_dimred = reduce_dimensionality(normalize(X), dim_red_k=DIMRED)
    print('Dimensionality = {}'.format(X_dimred.shape[1]))
    np.savetxt('data/dimred/{}_{}.txt'
               .format(DR_METHOD, NAMESPACE), X_dimred)
else:
    X_dimred = np.loadtxt('data/dimred/{}_{}.txt'
                          .format(DR_METHOD, NAMESPACE))

dataset = AnnData(X)
dataset.var['gene_symbols'] = genes

if os.path.isfile('data/ica/ica_bone_marrow_cell_types.txt'):
    with open('data/ica/ica_bone_marrow_cell_types.txt') as f:
        cell_types = f.read().rstrip().split('\n')
else:
    cell_types = None

if cell_types is None or len(cell_types) != X.shape[0]:
    dataset.obsm['X_pca'] = X_dimred
    print('Neighbors...')
    sc.pp.neighbors(dataset, n_neighbors=15)
    print('Louvain...')
    sc.tl.louvain(dataset, resolution=1.)
    print('UMAP...')
    sc.tl.umap(dataset)
    sc.pl.umap(dataset, color='louvain', size=0.5,
               save='_ica_bonemarrow_louvain.png')
    print('Rank genes...')
    sc.tl.rank_genes_groups(dataset, 'louvain', n_genes=20,
                            corr_method='bonferroni')
    sc.pl.rank_genes_groups(dataset, save='_ica_bonemarrow.png',
                            gene_symbols='gene_symbols')

    louv2type = {
        '0': 'Macrophage-Monocyte_CD14-TYROBP',
        '1': 'T_CD3E',
        '2': 'NaturalKiller-CytotoxicT_NKG7-GNLY',
        '3': 'T_CD3E',
        '4': 'Monocyte_CD14-S100A8',
        '5': 'CytotoxicT_CD8-CD3E-CCL5',
        '6': 'B_CD79A-MS4A1',
        '7': 'B_CD79A-MS4A1',
        '8': 'NaturalKiller_NKG7-CCL5',
        '9': 'Quiescent',
        '10': 'Macrophage-Monocyte_LYZ-S100A8',
        '11': 'Erythroblast_HBB',
        '12': 'CellDivision',
        '13': 'B_CD79B',
        '14': 'Ambient_RPL-RPS',
        '15': 'Ambient-MT',
        '16': 'NaturalKiller-CytotoxicT_NKG7-GNLY',
        '17': 'Macrophage-Monocyte_LST1-TYROBP',
        '18': 'Macrophage-Monocyte_TMSB10-CD74',
        '19': 'Macrophage-Monocyte_CD74-ALOX5AP',
        '20': 'B_IGKC-MZB1',
        '21': 'B_MZB1-CD27',
        '22': 'Megakaryocyte_TUBB1',
        '23': 'Adipoctye_CFD-LEPR',
        '24': 'B_IGLC3-MZB1',
        '25': 'Macrophage_FTL-CD68',
    }

    cell_types = [ louv2type[louv] for louv in dataset.obs['louvain'] ]

    with open('data/ica/ica_bone_marrow_cell_types.txt', 'w') as of:
        [ of.write('{}\n'.format(label)) for label in cell_types ]

dataset.obs['cell_types'] = [ NAMESPACE + '_' + l for l in cell_types ]
datasets = [ dataset ]
namespaces = [ NAMESPACE ]
