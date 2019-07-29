from process import process, load_names, merge_datasets
from utils import *

NAMESPACE = 'human_pbmc_zheng'
DR_METHOD = 'svd'
DIMRED = 100

data_names = [
    'data/pbmc/10x/b_cells',
    'data/pbmc/10x/cd14_monocytes',
    'data/pbmc/10x/cd34',
    'data/pbmc/10x/cd4_t_helper',
    'data/pbmc/10x/cd56_nk',
    'data/pbmc/10x/cytotoxic_t',
    'data/pbmc/10x/memory_t',
    'data/pbmc/10x/naive_cytotoxic_t',
    'data/pbmc/10x/naive_t',
    'data/pbmc/10x/regulatory_t',
    'data/pbmc/68k',
]

datasets, genes_list, n_cells = load_names(data_names)

Xs, genes = merge_datasets(datasets[:], genes_list, ds_names=data_names)

uniq_cell_types = [
    'CD19+_B',
    'CD14+_Monocyte',
    'CD34+',
    'CD4+_T_Helper2',
    'CD56+_NK',
    'CD8+_Cytotoxic_T',
    'CD4+/CD45RO+_Memory',
    'CD8+/CD45RA+_Naive_Cytotoxic',
    'CD4+/CD45RA+/CD25-_Naive_T',
    'CD4+/CD25_T_Reg',
]

cell_types = []
for i, a in enumerate(Xs):
    if i < len(Xs) - 1:
        cell_type = uniq_cell_types[i]
        cell_types += [ cell_type ] * a.shape[0]
    else:
        with open('data/pbmc/68k/pbmc_68k_cluster.txt') as f:
            cell_types += f.read().rstrip().split('\n')
            cell_types = np.array(cell_types)

X = vstack(Xs)

umi_sum = np.sum(X, axis=1)
gt_idx = [ i for i, s in enumerate(umi_sum)
           if s >= 500 ]
low_idx = [ idx for idx, gene in enumerate(genes)
            if gene.startswith('RPS') or gene.startswith('RPL') ]
lt_idx = [ i for i, s in enumerate(np.sum(X[:, low_idx], axis=1) / umi_sum)
           if s <= 0.5 ]

qc_idx = sorted(set(gt_idx) & set(lt_idx))
X = csr_matrix(X[qc_idx])
cell_types = cell_types[qc_idx]

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
dataset.obs['cell_types'] = [ NAMESPACE + '_' + l for l in cell_types ]
datasets = [ dataset ]
namespaces = [ NAMESPACE ]
