from anndata import AnnData
from joblib import Parallel, delayed
import numpy as np
from scanorama import plt, visualize
import scanpy as sc
import scipy.sparse as ss
from scipy.stats import pearsonr, spearmanr
import seaborn as sns
from sklearn.random_projection import GaussianRandomProjection
from sklearn.random_projection import SparseRandomProjection

from dict_learning import DictionaryLearning
from draw_graph import draw_graph
from utils import *

NAMESPACE = 'microglia_spearman_louvain_sparse0.1'

N_COMPONENTS = 20
INIT = 'eigen'

VIZ_AGE = True
VIZ_KNN = True
VIZ_SPARSITY = True
VIZ_STUDY = True
VIZ_CORR_PSEUDOTIME = True

def srp_worker(X, srp, triu_idx):
    return srp.transform(np.abs(X.toarray())[triu_idx].reshape(1, -1))[0]

def savefig(fname, ax):
    ratio = 1.
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    ax.set_aspect(abs((xmax - xmin) / (ymax - ymin)) * ratio)
    plt.savefig(fname)
    plt.close()

if __name__ == '__main__':

    mkdir_p('figures/')

    dirname = 'target/sparse_correlations/{}'.format(NAMESPACE)

    with open('{}/genes.txt'.format(dirname)) as f:
        genes = f.read().rstrip().split('\n')

    with open('{}/cluster_studies.txt'.format(dirname)) as f:
        studies = np.array(f.read().rstrip().split('\n'))

    tprint('Loading correlation matrices...')

    fnames = os.listdir(dirname)

    srp = None
    triu_idx = None

    Xs = []
    node_idxs = []
    ages = []
    node_sizes = []
    nonzero_idx = set()
    sparsities = []
    node_sizes = []
    for fname in fnames:
        fields = fname.split('_')
        if fields[0] != 'node' or fields[2] != 'at' or \
           fields[4] != 'has' or fields[6] != 'leaves.npz':
            continue

        X = ss.load_npz(dirname + '/' + fname)
        sparse_cutoff = 100000
        if len(X.data) > sparse_cutoff:
            cutoff = sorted(-abs(X.data))[sparse_cutoff - 1]
            X[abs(X) < abs(cutoff)] = 0
        Xs.append(X)

        nonzero_set = set([ (r, c) for r, c in zip(*X.nonzero()) ])
        nonzero_idx |= nonzero_set

        node_idxs.append(int(fields[1]))
        ages.append(float(fields[3]))
        node_sizes.append(int(fields[5]))

        sparsity = np.log10(X.count_nonzero())
        sparsities.append(sparsity)

    n_features = Xs[0].shape[0]
    n_correlations = int(comb(n_features, 2) + n_features)
    triu_idx = np.triu_indices(n_features)

    tprint(len(nonzero_idx))

    sorted_nonzero_idx = sorted(nonzero_idx)
    nonzero_tup = ([ ni[0] for ni in sorted_nonzero_idx ],
                   [ ni[1] for ni in sorted_nonzero_idx ])
    Xs_dimred = [
        X[nonzero_tup].A.flatten()
        for X in Xs
    ]

    # Change from lexicographic ordering to numeric.
    ordered = [ node_idxs.index(i) for i in sorted(node_idxs) ]
    Xs = [ Xs[o] for o in ordered  ]
    Xs_dimred = [ Xs_dimred[o] for o in ordered  ]
    ages = [ ages[o] for o in ordered ]
    node_sizes = [ node_sizes[o] for o in ordered ]
    sparsities = [ sparsities[o] for o in ordered ]

    gene_pairs = []
    with open('{}/gene_pairs.txt'.format(dirname), 'w') as of:
        for gidx_i, gidx_j in sorted(nonzero_idx):
            pair = (genes[gidx_i], genes[gidx_j])
            of.write('{}_{}\n'.format(*pair))
            gene_pairs.append(pair)

    X_dimred = np.vstack(Xs_dimred)

    tprint(X_dimred.shape)

    adata = AnnData(X=X_dimred)
    adata.obs['age'] = [ age if age >= 0 else 19 for age in ages ]
    sc.tl.pca(adata, n_comps=100, svd_solver='randomized')
    sc.pp.neighbors(adata, n_neighbors=20)

    draw_graph(adata, layout='fa')

    if VIZ_SPARSITY:
        tprint('Plot sparsity...')

        adata.obs['sparsity'] = sparsities
        adata.obs['sizes'] = np.log10(node_sizes)

        ax = sc.pl.draw_graph(
            adata, color='sparsity', edges=True, edges_color='#CCCCCC', show=False,
        )
        savefig('figures/draw_graph_fa_{}_cluster_trajectory_sparsity.png'
                .format(NAMESPACE), ax)

        ax = sc.pl.draw_graph(
            adata, color='sizes', edges=True, edges_color='#CCCCCC', show=False,
        )
        savefig('figures/draw_graph_fa_{}_cluster_trajectory_sizes.png'
                .format(NAMESPACE), ax)

    if VIZ_STUDY:
        tprint('Color by study...')

        adata.obs['study'] = studies

        ax = sc.pl.draw_graph(
            adata, color='study', edges=True, edges_color='#CCCCCC',
            show=False,
        )
        savefig('figures/draw_graph_fa_{}_study.png'.format(NAMESPACE), ax)

        for study in sorted(set(studies)):
            adata.obs[study] = studies == study
            ax = sc.pl.draw_graph(
                adata, color=study, edges=True, edges_color='#CCCCCC',
                show=False,
            )
            savefig('figures/draw_graph_fa_{}_cluster_trajectory_{}.png'
                    .format(NAMESPACE, study), ax)

    if VIZ_AGE:
        tprint('Visualize age...')

        ax = sc.pl.draw_graph(
            adata, color='age', edges=True, edges_color='#CCCCCC',
            show=False,
        )
        savefig('figures/draw_graph_fa_{}_cluster_trajectory_age.png'
                .format(NAMESPACE), ax)

        if VIZ_KNN:
            for knn in [ 8, 10, 13, 15, 20, 30, 40, 50 ]:
                sc.pp.neighbors(adata, n_neighbors=knn)
                draw_graph(adata, layout='fa')
                ax = sc.pl.draw_graph(
                    adata, color='age', edges=True, edges_color='#CCCCCC',
                    show=False,
                )
                savefig('figures/draw_graph_fa_{}_cluster_trajectory_age_k{}.png'
                        .format(NAMESPACE, knn), ax)

    if VIZ_CORR_PSEUDOTIME:
        sc.pp.neighbors(adata, n_neighbors=20)

        draw_graph(adata, layout='fa')

        tprint('Diffusion pseudotime analysis...')

        tprint('pseudotime')
        sc.tl.diffmap(adata)
        adata.uns['iroot'] = np.flatnonzero(adata.obs['age'] < 14.6)[0]
        sc.tl.dpt(adata)
        finite_idx = np.isfinite(adata.obs['dpt_pseudotime'])
        tprint(pearsonr(adata.obs['dpt_pseudotime'][finite_idx],
                        adata.obs['age'][finite_idx]))
        tprint(spearmanr(adata.obs['dpt_pseudotime'][finite_idx],
                         adata.obs['age'][finite_idx]))

        ax = sc.pl.draw_graph(
            adata, color='dpt_pseudotime', edges=True, edges_color='#CCCCCC',
            color_map='inferno', show=False,
        )
        savefig('figures/draw_graph_fa_{}_cluster_trajectory_dpt.png'
                .format(NAMESPACE), ax)

        pair2corr = {}
        assert(len(gene_pairs) == X_dimred.shape[1])
        for pair_idx, pair in enumerate(gene_pairs):
            pair2corr[pair] = pearsonr(
                X_dimred[finite_idx, pair_idx],
                adata.obs['dpt_pseudotime'][finite_idx]
            )[0]
        for pair, corr in sorted(
                pair2corr.items(), key=lambda kv: -abs(kv[1])
        ):
            print('{}\t{}\t{}'.format(pair[0], pair[1], corr))
