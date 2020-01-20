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

NAMESPACE = 'mouse_develop_spearman_louvain'

N_COMPONENTS = 15
INIT = 'eigen'

VIZ_AGE = True
VIZ_KNN = False
VIZ_SPARSITY = False
VIZ_STUDY = False
VIZ_DICT_LEARN = True
VIZ_CORR_PSEUDOTIME = False

def srp_worker(X, srp, triu_idx):
    return srp.transform(np.abs(X.toarray())[triu_idx].reshape(1, -1))[0]

def savefig(fname, ax):
    ratio = 2.
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

    #srp = SparseRandomProjection(
    #    eps=0.1, random_state=69
    #).fit(ss.csr_matrix((len(Xs), n_correlations)))
    #
    #Xs_dimred = Parallel(n_jobs=20, backend='multiprocessing') (
    #    delayed(srp_worker)(X, srp, triu_idx)
    #    for X in Xs
    #)

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
    adata.obs['age'] = ages
    sc.pp.neighbors(adata, n_neighbors=40, use_rep='X')

    draw_graph(adata, layout='fa')
    #argsort = np.argsort(adata.obsm['X_draw_graph_fa'][:, 0])[:6]
    #mean_pos = np.mean(adata.obsm['X_draw_graph_fa'][:, 0])
    #adata.obsm['X_draw_graph_fa'][argsort, 0] = mean_pos
    #argsort = np.argsort(-adata.obsm['X_draw_graph_fa'][:, 1])[:20]
    #adata.obsm['X_draw_graph_fa'][argsort, 0] = mean_pos
    #adata.obsm['X_draw_graph_fa'][argsort, 1] *= 0.7

    print('pseudotime')
    sc.tl.diffmap(adata)
    adata.uns['iroot'] = np.flatnonzero(adata.obs['age'] < 9.6)[0]
    sc.tl.dpt(adata)
    adata.obs['dpt_pseudotime'][adata.obs['dpt_pseudotime'] > 0.19] = 0.19
    #adata.obs['dpt_pseudotime'] /= 0.20
    from scipy.stats import pearsonr, spearmanr
    print(pearsonr(adata.obs['dpt_pseudotime'], adata.obs['age']))
    print(spearmanr(adata.obs['dpt_pseudotime'], adata.obs['age']))

    plt.figure()
    sns.lmplot('age', 'dpt_pseudotime', adata.obs, ci=99)
    plt.savefig('pseudo_age.svg')

    ax = sc.pl.draw_graph(
        adata, color='dpt_pseudotime', edges=True, edges_color='#CCCCCC',
        color_map='inferno', show=False,
    )
    savefig('figures/draw_graph_fa_{}_cluster_trajectory_dpt.png'
            .format(NAMESPACE), ax)

    if VIZ_CORR_PSEUDOTIME:
        tprint('Diffusion pseudotime analysis...')

        pair2corr = {}
        assert(len(gene_pairs) == X_dimred.shape[1])
        for pair_idx, pair in enumerate(gene_pairs):
            pair2corr[pair] = pearsonr(
                X_dimred[:, pair_idx], adata.obs['dpt_pseudotime']
            )[0]
        for pair, corr in sorted(
                pair2corr.items(), key=lambda kv: -abs(kv[1])
        ):
            print('{}\t{}\t{}'.format(pair[0], pair[1], corr))

            if pair == ('FOS', 'FOS') or pair == ('PTGDS', 'PTGDS') or \
               pair == ('LOXL2', 'LOXL2') or pair == ('LHX1', 'LHX1') or \
               pair == ('EOMES', 'EOMES'):
                pair_name = '_'.join(pair)
                pair_idx = gene_pairs.index(pair)
                adata.obs[pair_name] = X_dimred[:, pair_idx]
                ax = sc.pl.draw_graph(
                    adata, color=pair_name, edges=True, edges_color='#CCCCCC',
                    show=False, color_map='coolwarm',
                )
                savefig('figures/draw_graph_fa_{}_pair_{}.png'
                        .format(NAMESPACE, pair_name), ax)

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
                show=False, color_map='plasma',
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
