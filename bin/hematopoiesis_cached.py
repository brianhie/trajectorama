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
from spddl import CovarianceDictionary
from utils import *

NAMESPACE = 'hematopoiesis_pearson_louvain'

N_COMPONENTS = 15
INIT = 'eigen'

VIZ_CELL_TYPES = True
VIZ_LOUVAIN = False
VIZ_SPARSITY = False
VIZ_STUDY = False
VIZ_DICT_LEARN = False
VIZ_CORR_COMP = False

def srp_worker(X, srp, triu_idx):
    return srp.transform(np.abs(X.toarray())[triu_idx].reshape(1, -1))[0]

def savefig(fname, ax):
    ratio = 0.4
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    ax.set_aspect(abs((xmax - xmin) / (ymax - ymin)) * ratio)
    plt.savefig(fname)
    
if __name__ == '__main__':

    dirname = 'target/sparse_correlations/{}'.format(NAMESPACE)
    
    with open('{}/genes.txt'.format(dirname)) as f:
        genes = f.read().rstrip().split('\n')

    with open('{}/cluster_studies.txt'.format(dirname)) as f:
        studies = np.array(f.read().rstrip().split('\n'))

    fnames = os.listdir(dirname)

    srp = None
    triu_idx = None
            
    Xs = []
    node_idxs = []
    sparsities = []
    node_sizes = []
    nonzero_idx = set()
    dense_idx = set()
    for fname in fnames:
        fields = fname.split('_')
        if fields[0] != 'node' or fields[2] != 'has' or \
           fields[4] != 'leaves.npz':
            continue

        X = ss.load_npz(dirname + '/' + fname)

        sparse_cutoff = 10000
        if len(X.data) > sparse_cutoff:
            cutoff = sorted(-abs(X).data)[sparse_cutoff - 1]
            X[abs(X) < cutoff] = 0

        Xs.append(X)
        node_idxs.append(int(fields[1]))
        node_sizes.append(int(fields[3]))

        sparsity = np.log10(X.count_nonzero())
        sparsities.append(sparsity)

        if sparsity > np.log10(10000):
            continue

        #print('nonzero: {}'.format(X.count_nonzero()))

        #nonzero_idx = set([ (i, i) for i in range(X.shape[0]) ])

        nonzero_idx |= set([ (r, c) for r, c in zip(*X.nonzero()) ])

    print(10 ** np.mean(sparsities))
    print(10 ** np.median(sparsities))
    print(10 ** np.percentile(sparsities, 95))
    print(10 ** np.percentile(sparsities, 99))
    print(10 ** np.percentile(sparsities, 99.9))
    print(10 ** max(sparsities))

    #X_full = ss.load_npz(dirname + '/full_X.npz')
    #
    #highlight = np.zeros(X_full.shape[0])
    #highlight[sorted(dense_idx)] = 1
    #
    #adata = AnnData(X=X_full)
    #adata.obs['highlight'] = highlight
    #sc.pp.neighbors(adata)
    #sc.tl.umap(adata)
    #sc.pl.scatter(
    #    adata, color='highlight', basis='umap',
    #    save='_{}_highlight_dense_all.png'.format(NAMESPACE)
    #)
    #exit()
    
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
    
    #analyze_dense(Xs, Xs_dimred, sparsities, node_sizes)

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
    sparsities = [ sparsities[o] for o in ordered ]
    node_sizes = [ node_sizes[o] for o in ordered ]

    gene_pairs = []
    with open('{}/gene_pairs.txt'.format(dirname), 'w') as of:
        for gidx_i, gidx_j in sorted(nonzero_idx):
            pair = (genes[gidx_i], genes[gidx_j])
            of.write('{}_{}\n'.format(*pair))
            gene_pairs.append(pair)

    X_dimred = np.vstack(Xs_dimred)
    adata = AnnData(X=X_dimred)

    if VIZ_CELL_TYPES:
        with open('{}/cell_type_composition.txt'.format(dirname)) as f:
            uniq_cell_types = [
                type_.replace('/', '_')
                for type_ in f.readline().rstrip().split('\t')[1:]
            ]
            cd4_idx = [ idx for idx, ct in enumerate(uniq_cell_types)
                        if 'human_pbmc_zheng_CD4+' in ct ]
            cd8_idx = [ idx for idx, ct in enumerate(uniq_cell_types)
                        if 'human_pbmc_zheng_CD8+' in ct ]
            
            comps = []
            for line in f:
                fields = line.rstrip().split()
                pcts = [ float(field) for field in fields[1:] ]
                cd4 = sum([ pcts[idx] for idx in cd4_idx ])
                cd8 = sum([ pcts[idx] for idx in cd8_idx ])
                comps.append(pcts + [ cd4, cd8 ])
            comps = np.array(comps)

        sc.pp.neighbors(adata, n_neighbors=30, use_rep='X')
        draw_graph(adata, layout='fa')

        for idx, cell_type in enumerate(uniq_cell_types + [
                'human_pbmc_zheng_CD4+', 'human_pbmc_zheng_CD8+'
        ]):
            adata.obs[cell_type] = comps[:, idx]
            ax = sc.pl.draw_graph(
                adata, color=cell_type, edges=True, edges_color='#CCCCCC',
                show=False, vmin=0, vmax=1
            )
            savefig('figures/draw_graph_fa_{}_cell_type_{}.png'
                    .format(NAMESPACE, cell_type), ax)
        exit()
                
        max_cell_types = [
            uniq_cell_types[np.argmax(comps[i])]
            for i in range(adata.X.shape[0])
        ]
        adata.obs['max_cell_type'] = max_cell_types

        for knn in [ 15, 20, 30, 40 ]:
            sc.pp.neighbors(adata, n_neighbors=knn, use_rep='X')
            
            draw_graph(adata, layout='fa')
            ax = sc.pl.draw_graph(
                adata, color='max_cell_type', edges=True, edges_color='#CCCCCC',
                show=False,
            )
            savefig('figures/draw_graph_fa_{}_cell_type_max_k{}.png'
                    .format(NAMESPACE, knn), ax)

    if VIZ_STUDY:
        adata.obs['study'] = studies

        for knn in [ 15, 20, 30, 40 ]:
            sc.pp.neighbors(adata, n_neighbors=knn, use_rep='X')
            
            draw_graph(adata, layout='fa')
            ax = sc.pl.draw_graph(
                adata, color='study', edges=True, edges_color='#CCCCCC',
                show=False,
            )
            savefig('figures/draw_graph_fa_{}_cluster_trajectory_study_k{}.png'
                    .format(NAMESPACE, knn), ax)
            
            #sc.tl.umap(adata, min_dist=0.25)
            #sc.pl.scatter(
            #    adata, color='study', basis='umap',
            #    save='_{}_umap_study_k{}.png'.format(NAMESPACE, knn)
            #)
            
        sc.pp.neighbors(adata, n_neighbors=30, use_rep='X')
        draw_graph(adata, layout='fa')
        sc.tl.umap(adata)
        for study in sorted(set(studies)):
            adata.obs[study] = studies == study
            ax = sc.pl.draw_graph(
                adata, color=study, edges=True, edges_color='#CCCCCC',
                show=False,
            )
            savefig('figures/draw_graph_fa_{}_cluster_trajectory_{}.png'
                    .format(NAMESPACE, study), ax)
            
            sc.pl.scatter(
                adata, color=study, basis='umap',
                save='_{}_umap_{}.png'.format(NAMESPACE, study)
            )

    if VIZ_LOUVAIN:
        for knn in [ 15, 20, 30, 40 ]:
            sc.pp.neighbors(adata, n_neighbors=knn, use_rep='X')
            sc.tl.louvain(adata, resolution=1.)
            draw_graph(adata, layout='fa')
            ax = sc.pl.draw_graph(
                adata, color='louvain', edges=True, edges_color='#CCCCCC',
                show=False,
            )
            savefig('figures/draw_graph_fa_{}_cluster_trajectory_louvain_k{}.png'
                    .format(NAMESPACE, knn), ax)

            sc.tl.umap(adata, min_dist=0.25)
            sc.pl.scatter(
                adata, color='louvain', basis='umap',
                save='_{}_umap_louvain_k{}.png'.format(NAMESPACE, knn)
            )

    if VIZ_SPARSITY:
        adata.obs['sparsity'] = sparsities
        adata.obs['sizes'] = np.log10(node_sizes)
        
        sc.pp.neighbors(adata, n_neighbors=30, use_rep='X')
        draw_graph(adata, layout='fa')
        sc.tl.umap(adata)

        ax = sc.pl.draw_graph(
            adata, color='sparsity', edges=True, edges_color='#CCCCCC',
            show=False,
        )
        savefig('figures/draw_graph_fa_{}_cluster_trajectory_sparsity.png'
                .format(NAMESPACE), ax)
        sc.pl.scatter(
            adata, color='sparsity', basis='umap',
            save='_{}_umap_sparsity.png'.format(NAMESPACE)
        )        
        ax = sc.pl.draw_graph(
            adata, color='sizes', edges=True, edges_color='#CCCCCC',
            show=False,
        )
        savefig('figures/draw_graph_fa_{}_cluster_trajectory_sizes.png'
                .format(NAMESPACE), ax)
        sc.pl.scatter(
            adata, color='sizes', basis='umap',
            save='_{}_umap_sizes.png'.format(NAMESPACE)
        )        

    if VIZ_DICT_LEARN:
        sc.pp.neighbors(adata, n_neighbors=30, use_rep='X')
        draw_graph(adata, layout='fa')
        sc.tl.umap(adata)

        dl = DictionaryLearning(
            n_components=N_COMPONENTS,
            alpha=1e-3,
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
        
        np.savetxt('{}/weights.txt'.format(dirname), weights)

        for comp in range(N_COMPONENTS):
            comp_name = 'dict_entry_{}'.format(comp)
            adata.obs[comp_name] = weights[:, comp]
            np.savetxt('{}/dictw{}.txt'.format(dirname, comp),
                       dl.components_[comp])
            ax = sc.pl.draw_graph(
                adata, color=comp_name, edges=True, edges_color='#CCCCCC',
                show=False,
            )
            savefig('figures/draw_graph_fa_{}_cluster_trajectory_dict{}.png'
                    .format(NAMESPACE, comp), ax)
            sc.pl.scatter(
                adata, color=comp_name, basis='umap',
                save='_{}_umap_dict{}.png'.format(NAMESPACE, comp)
            )

        adata.obs['dict_entry_eryth'] = (
            weights[:, 1] +
            weights[:, 2] +
            weights[:, 6] +
            weights[:, 7] +
            weights[:, 8] +
            weights[:, 10]
        )
        ax = sc.pl.draw_graph(
            adata, color='dict_entry_eryth', edges=True,
            edges_color='#CCCCCC', show=False,
        )
        savefig('figures/draw_graph_fa_{}_cluster_trajectory_dict{}.png'
                .format(NAMESPACE, 'eryth'), ax)

    if VIZ_CORR_COMP:
        weights = np.loadtxt('{}/weights.txt'.format(dirname))

        eryth_comp = (
            weights[:, 1] +
            weights[:, 2] +
            weights[:, 6] +
            weights[:, 7] +
            weights[:, 8] +
            weights[:, 10]
        )
        comps = [
            weights[:, 3],
            weights[:, 9],
            weights[:, 11],
            eryth_comp
        ]
        comp_names = [
            'progenitor', 'lymphoid', 'myeloid', 'erythroid'
        ]
        
        pair2corr = {}
        assert(len(gene_pairs) == X_dimred.shape[1])
        for comp, name in zip(comps, comp_names):
            print('Component: {}'.format(name))
            for pair_idx, pair in enumerate(gene_pairs):
                pair2corr[pair] = pearsonr(
                    X_dimred[:, pair_idx], comp
                )[0]

            for pair, corr in sorted(
                    pair2corr.items(), key=lambda kv: -kv[1]
            ):
                print('{}\t{}\t{}'.format(pair[0], pair[1], corr))
                
                if pair == ('CD34', 'CD34') or pair == ('HLA-DRB1', 'HLA-DRB5') or \
                   pair == ('FTH1', 'MALAT1') or pair == ('GYPC', 'HBB'):
                    pair_name = '_'.join(pair)
                    pair_idx = gene_pairs.index(pair)
                    adata.obs[pair_name] = X_dimred[:, pair_idx]
                    ax = sc.pl.draw_graph(
                        adata, color=pair_name, edges=True, edges_color='#CCCCCC',
                        show=False,
                    )
                    savefig('figures/draw_graph_fa_{}_pair_{}.png'
                            .format(NAMESPACE, pair_name), ax)
