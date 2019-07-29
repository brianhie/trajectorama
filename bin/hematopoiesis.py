from anndata import AnnData
from matplotlib import cm
import numpy as np
import os
from scanorama import process_data, plt, reduce_dimensionality, visualize
import scanpy as sc
from scipy.sparse import vstack, save_npz
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, normalize
import sys

from draw_graph import draw_graph
from pan_corr import PanCorrelation
from pan_dag import PanDAG
from process import merge_datasets
from utils import *


CORR_METHOD = 'pearson'
DAG_METHOD = 'louvain'
DIMRED = 100
DR_METHOD = 'svd'

REASSEMBLE_METHOD = 'louvain'
REASSEMBLE_K = 15

CORR_CUTOFF = 0.7
RANDOM_PROJ = False

NAMESPACE = 'hematopoiesis_{}_{}'.format(CORR_METHOD, DAG_METHOD)

if RANDOM_PROJ:
    NAMESPACE += '_randproj'

all_datasets = []
all_namespaces = []
all_dimreds = []

from dataset_bonemarrow_ica import datasets, namespaces, X_dimred
all_datasets += datasets
all_namespaces += namespaces
all_dimreds.append(X_dimred)
from dataset_cordblood_ica import datasets, namespaces, X_dimred
all_datasets += datasets
all_namespaces += namespaces
all_dimreds.append(X_dimred)
from dataset_pbmc_10x import datasets, namespaces, X_dimred
all_datasets += datasets
all_namespaces += namespaces
all_dimreds.append(X_dimred)

def count_cell_types(cell_types):
    counter = Counter(cell_types)
    n_values = float(sum(counter.values()))
    return [
        float(counter[cell_type]) / n_values
        for cell_type in uniq_cell_types
    ]

if __name__ == '__main__':
    dirname = 'target/sparse_correlations/{}'.format(NAMESPACE)
    mkdir_p(dirname)

    hv_genes = None
    for i, dataset in enumerate(all_datasets):
        genes_hvg, _ = hvg([dataset.X], dataset.var['gene_symbols'], 'dispersion')
        genes_hvg = [ gene for gene in genes_hvg
                      if not gene.startswith('RPS') and
                      not gene.startswith('RPL') ]
        if hv_genes is None:
            hv_genes = set([ g for gene in genes_hvg[:5000]
                             for g in gene.split(';') ])
        else:
            hv_genes &= set([ g for gene in genes_hvg[:5000]
                              for g in gene.split(';') ])
        print('{}: {}'.format(all_namespaces[i], len(hv_genes)))
        sys.stdout.flush()

    # Keep only those highly variable genes.

    Xs, genes = merge_datasets([ dataset.X for dataset in all_datasets ],
                               [ dataset.var['gene_symbols'] for dataset in all_datasets ],
                               keep_genes=hv_genes, ds_names=all_namespaces,
                               verbose=True)

    [ print(X.shape[0]) for X in Xs ]

    X = vstack(Xs)
    X = X.log1p()

    #save_npz('{}/full_X.npz'.format(dirname), X)

    cell_types = np.concatenate(
        [ dataset.obs['cell_types'] for dataset in all_datasets ],
        axis=None
    )

    cds = [
        PanDAG(
            dag_method=DAG_METHOD,
            reduce_dim=all_dimreds[i],
            verbose=True,
        ).fit(all_dimreds[i])
        for i in range(len(all_dimreds))
    ]

    ct = PanCorrelation(
        n_components=25,
        min_modules=3,
        min_leaves=500,
        dag_method=DAG_METHOD,
        corr_method=CORR_METHOD,
        corr_cutoff=CORR_CUTOFF,
        reassemble_method=REASSEMBLE_METHOD,
        reassemble_K=REASSEMBLE_K,
        random_projection=RANDOM_PROJ,
        dictionary_learning=True,
        n_jobs=1,
        verbose=2,
    )

    studies = [ 'human_hematopoiesis' ]
    curr_idx = 0
    for i, cd in enumerate(cds):
        for node in cd.nodes:
            node.sample_idx = np.array(node.sample_idx) + curr_idx
            ct.nodes.append(node)
            if node.n_leaves >= ct.min_leaves:
                studies.append(all_namespaces[i])
        curr_idx += all_dimreds[i].shape[0]
        print(all_dimreds[i].shape[0])
    print(curr_idx)
    print(X.shape[0])
    assert(curr_idx == X.shape[0])


    with open('{}/genes.txt'.format(dirname), 'w') as of:
        [ of.write('{}\n'.format(gene)) for gene in genes ]

    with open('{}/cluster_studies.txt'.format(dirname), 'w') as of:
        [ of.write('{}\n'.format(study)) for study in studies ]

    ct.sample_idx = list(range(X.shape[0]))
    ct.n_leaves = X.shape[0]
    ct.fill_correlations(X)

    uniq_cell_types = sorted(set(cell_types))

    with open('{}/cell_type_composition.txt'.format(dirname), 'w') as of:
        of.write('node')
        for cell_type in uniq_cell_types:
            of.write('\t{}'.format(cell_type))
        of.write('\n')

        for node_idx, node in enumerate(ct.nodes):
            if node.n_leaves < ct.min_leaves:
                continue
            save_npz('{}/node_{}_has_{}_leaves.npz'.format(
                dirname, node_idx, node.n_leaves
            ), node.correlations)

            fractions = count_cell_types(cell_types[node.sample_idx])
            of.write('{}\t'.format(node_idx))
            of.write('\t'.join([ str(frac) for frac in fractions ]) + '\n')
    exit()

    from mouse_develop import correct_scanorama, correct_scvi
    #X = correct_scanorama(Xs, genes)
    X = correct_scvi(Xs, genes)

    C = np.vstack([
        #(np.exp(
        #    ((1. / node.n_leaves) * np.log1p(X[node.sample_idx]).sum(0)) - 1
        #) + 1) / (
        #    (1. / node.n_leaves) * X[node.sample_idx].sum(0) + 1
        #)
        X[node.sample_idx].mean(0)
        for node in ct.nodes
        if node.n_leaves >= ct.min_leaves
    ])
    adata = AnnData(X=C)
    adata.obs['study'] = studies

    for knn in [ 15, 20, 30, 40 ]:
        sc.pp.neighbors(adata, n_neighbors=knn, use_rep='X')
        draw_graph(adata, layout='fa')
        sc.pl.draw_graph(
            adata, color='study', edges=True, edges_color='#CCCCCC',
            save='_{}_expr_gmean_k{}.png'
            .format(NAMESPACE + '_scvi', knn)
        )
        sys.stdout.flush()

    adata = AnnData(X=X)
    adata.obs['study'] = [ '_'.join(ct.split('_')[:3]) for ct in cell_types ]
    sc.pp.neighbors(adata)#, use_rep='X')
    sc.tl.umap(adata, init_pos='random')
    sc.pl.scatter(
        adata, color='study', basis='umap',
        save='_{}_umap_study.png'.format(NAMESPACE + '_scvi')
    )
    np.save('data/hema_umap_coord_scvi.npy', adata.obsm['X_umap'])
