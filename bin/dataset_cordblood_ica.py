from anndata import AnnData
import gzip
import numpy as np
from process import load_names, merge_datasets
from utils import *

NAMESPACE = 'human_cordblood_ica'
DIMRED = 100
DR_METHOD = 'svd'

data_names = [
    'data/ica/ica_cord_blood_h5',
]
namespaces = [
    'ica_cord_blood',
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

if os.path.isfile('data/ica/ica_cord_blood_cell_types.txt'):
    with open('data/ica/ica_cord_blood_cell_types.txt') as f:
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
               save='_ica_cordblood_louvain.png')
    print('Rank genes...')
    sc.tl.rank_genes_groups(dataset, 'louvain', n_genes=20,
                            corr_method='bonferroni')
    sc.pl.rank_genes_groups(dataset, save='_ica_cordblood.png',
                            gene_symbols='gene_symbols')

    louv2type = {
        '0': 'T_CD3E',
        '1': 'Macrophage-Monocyte_TYROBP-S100A8',
        '2': 'T_SARAF-CD6',
        '3': 'Ambient-RPL-RPS',
        '4': 'Macrophage-Monocyte_TYROBP-CD63',
        '5': 'Quiescent-MT',
        '6': 'NaturalKiller_NKG7',
        '7': 'B_CD79A-CD79B-CD74',
        '8': 'Quiescent-MT',
        '9': 'NaturalKiller-CytotoxicT_IL32-CD3E',
        '10': 'B_CD74-MS4A1',
        '11': 'T_SLC38A2',
        '12': 'Macrophage-Monocyte_NEAT1',
        '13': 'T_CD3E-IL32',
        '14': 'Erythroblast_HBA2',
        '15': 'Erythroblast_HBB',
        '16': 'Macrophage-Monocyte_CD14-LYZ',
        '17': 'Megakaryocyte_GP9-TREML1',
        '18': 'Erythroblast_HBA1',
        '19': 'Macrophage-Monocyte_NEAT1-CD63',
        '20': 'preB_LAPTM5-CD83',
        '21': 'Ambient_RPL-RPS',
        '22': 'Ambient_RPL-RPS',
        '23': 'Macrophage-Monocyte_TYROBP-S100A6',
        '24': 'Ambient_RPL-RPS',
    }

    cell_types = [ louv2type[louv] for louv in dataset.obs['louvain'] ]

    with open('data/ica/ica_cord_blood_cell_types.txt', 'w') as of:
        [ of.write('{}\n'.format(label)) for label in cell_types ]

dataset.obs['cell_types'] = [ NAMESPACE + '_' + l for l in cell_types ]
datasets = [ dataset ]
namespaces = [ NAMESPACE ]
