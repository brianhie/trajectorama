from anndata import AnnData
from joblib import Parallel, delayed
import numpy as np
from scanorama import plt, visualize
import scanpy as sc
import scipy.sparse as ss
import seaborn as sns
from sklearn.random_projection import GaussianRandomProjection
from sklearn.random_projection import SparseRandomProjection

from dict_learning import DictionaryLearning
from draw_graph import draw_graph
from utils import *

NAMESPACE = 'microglia_spearman_louvain_sparse0.3'

N_COMPONENTS = 20
INIT = 'eigen'

VIZ_AGE = True
VIZ_KNN = True
VIZ_SPARSITY = True
VIZ_STUDY = True
VIZ_DICT_LEARN = True

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

        #sparse_cutoff = 10000
        #if len(X.data) > sparse_cutoff:
        #    cutoff = sorted(-abs(X.data))[sparse_cutoff - 1]
        #    X[abs(X) < cutoff] = 0

        Xs.append(X)

        nonzero_idx |= set([ (r, c) for r, c in zip(*X.nonzero()) ])

        node_idxs.append(int(fields[1]))
        ages.append(float(fields[3]))
        node_sizes.append(int(fields[5]))

        sparsity = np.log10(X.count_nonzero())
        sparsities.append(sparsity)

    n_features = Xs[0].shape[0]
    n_correlations = int(comb(n_features, 2) + n_features)
    triu_idx = np.triu_indices(n_features)

    print(len(nonzero_idx))

    nonzero_tup = ([ ni[0] for ni in sorted(nonzero_idx) ],
                   [ ni[1] for ni in sorted(nonzero_idx) ])
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

    print(X_dimred.shape)

    adata = AnnData(X=X_dimred)
    adata.obs['age'] = [ age if age >= 0 else 19 for age in ages ]
    sc.pp.neighbors(adata, n_neighbors=20, use_rep='X')

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

    if VIZ_DICT_LEARN:
        tprint('Dictionary learning...')

        dl = DictionaryLearning(
            n_components=N_COMPONENTS,
            alpha=0.1,
            max_iter=100,
            tol=1e-8,
            fit_algorithm='lars',
            transform_algorithm='lasso_lars',
            n_jobs=20,
            verbose=2,
            split_sign=False,
            random_state=69,
            positive_code=True,
            positive_dict=True,
        )
        weights = dl.fit_transform(adata.X)

        for comp in range(N_COMPONENTS):
            comp_name = 'dict_entry_{}'.format(comp)
            adata.obs[comp_name] = weights[:, comp]
            ax = sc.pl.draw_graph(
                adata, color=comp_name, edges=True, edges_color='#CCCCCC',
                show=False,
            )
            savefig('figures/draw_graph_fa_{}_cluster_trajectory_dict{}.png'
                    .format(NAMESPACE, comp), ax)
            np.savetxt('{}/dictw{}.txt'.format(dirname, comp),
                       dl.components_[comp])

    if VIZ_AGE:
        tprint('Visualize age...')

        ax = sc.pl.draw_graph(
            adata, color='age', edges=True, edges_color='#CCCCCC',
            show=False,
        )
        savefig('figures/draw_graph_fa_{}_cluster_trajectory_age.png'
                .format(NAMESPACE), ax)

        if VIZ_KNN:
            for knn in [ 15, 20, 30, 40, 50 ]:
                sc.pp.neighbors(adata, n_neighbors=knn, use_rep='X')
                draw_graph(adata, layout='fa')
                ax = sc.pl.draw_graph(
                    adata, color='age', edges=True, edges_color='#CCCCCC',
                    show=False,
                )
                savefig('figures/draw_graph_fa_{}_cluster_trajectory_age_k{}.png'
                        .format(NAMESPACE, knn), ax)
