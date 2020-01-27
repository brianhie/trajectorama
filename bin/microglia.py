from anndata import AnnData
from matplotlib import cm
import numpy as np
import pandas as pd
import os
from scanorama import process_data, plt, reduce_dimensionality, visualize
import scanpy as sc
from scipy.stats import spearmanr
from scipy.sparse import vstack, save_npz
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, normalize
from sklearn.metrics import pairwise_distances
import sys

from draw_graph import draw_graph
from pan_corr import PanCorrelation
from pan_dag import PanDAG
from process import merge_datasets
from utils import *

CORR_METHOD = 'spearman'
DAG_METHOD = 'louvain'
DIMRED = 100
DR_METHOD = 'svd'

REASSEMBLE_METHOD = 'louvain'
REASSEMBLE_K = 15

CORR_CUTOFF = 0.1
RANDOM_PROJ = False

NAMESPACE = 'microglia_{}_{}'.format(CORR_METHOD, DAG_METHOD)

if RANDOM_PROJ:
    NAMESPACE += '_randproj'

def import_data():
    all_datasets, all_namespaces, all_dimreds = [], [], []

    from dataset_masuda_mouse_microglia import datasets, namespaces, X_dimred
    all_datasets += datasets
    all_namespaces += namespaces
    all_dimreds.append(X_dimred)
    from dataset_masuda_human_microglia import datasets, namespaces, X_dimred
    all_datasets += datasets
    all_namespaces += namespaces
    all_dimreds.append(X_dimred)
    from dataset_hammond_microglia import datasets, namespaces, X_dimred
    all_datasets += datasets
    all_namespaces += namespaces
    all_dimreds.append(X_dimred)
    from dataset_saunders_adult_myeloid import datasets, namespaces, X_dimred
    all_datasets += datasets
    all_namespaces += namespaces
    all_dimreds.append(X_dimred)

    return all_datasets, all_namespaces, all_dimreds

if __name__ == '__main__':
    dirname = 'target/sparse_correlations/{}'.format(NAMESPACE)
    mkdir_p(dirname)

    all_datasets, all_namespaces, all_dimreds = import_data()

    hv_genes = None
    for i, dataset in enumerate(all_datasets):
        genes_hvg, _ = hvg([dataset.X], dataset.var['gene_symbols'], 'dispersion')
        if hv_genes is None:
            hv_genes = set([ g for gene in genes_hvg[:8800]
                             for g in gene.split(';') ])
        else:
            hv_genes &= set([ g for gene in genes_hvg[:8800]
                              for g in gene.split(';') ])
        tprint('{}: {}'.format(all_namespaces[i], len(hv_genes)))
        sys.stdout.flush()

    # Keep only those highly variable genes.

    X_studies, genes = merge_datasets(
        [ dataset.X for dataset in all_datasets ],
        [ dataset.var['gene_symbols'] for dataset in all_datasets ],
        keep_genes=hv_genes, ds_names=all_namespaces,
        verbose=True
    )

    [ tprint(X.shape[0]) for X in X_studies ]

    cell_types = np.concatenate(
        [ dataset.obs['cell_types'] for dataset in all_datasets ],
        axis=None
    )

    X = vstack(X_studies)
    X = X.log1p()

    from subtypes_zeisel_saunders import sub_coexpr
    from subtypes_zeisel_saunders import plot_clustermap, interpret_clustermap
    coexpr, types, expr = sub_coexpr(
        X, cell_types, return_expr=True, min_samples=0,
        corr_cutoff=0., corr_method='spearman'
    )
    dist = pairwise_distances(coexpr)
    dist[np.isnan(dist)] = 0.
    df = pd.DataFrame(-dist, index=types, columns=types)
    linkage, _ = plot_clustermap(df, dist, 'coexpr', 'microglia')

    print(types)

    interpret_clustermap(coexpr, genes, types, linkage,
                         n_clusters=2, n_report=150)

    Xs, cds, cd_names, ages = [], [], [], []
    for i in range(len(all_dimreds)):
        cell_types_i = np.array(all_datasets[i].obs['cell_types'])
        for cell_type in sorted(set(cell_types_i)):
            type_idx = cell_types_i == cell_type
            Xs.append(X_studies[i][type_idx])
            X_dimred = all_dimreds[i][type_idx]
            cds.append(PanDAG(
                dag_method=DAG_METHOD,
                reduce_dim=X_dimred
            ).fit(X_dimred))
            cd_names.append(cell_type)
            ages.append(np.array(all_datasets[i].obs['ages'])[type_idx])
    ages = np.concatenate(ages)

    X = vstack(Xs)
    X = X.log1p()

    [ tprint(X.shape[0]) for X in Xs ]

    ct = PanCorrelation(
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

    studies = [ 'microglia' ]
    curr_idx = 0
    for i, cd in enumerate(cds):
        for node in cd.nodes:
            node.sample_idx = np.array(node.sample_idx) + curr_idx
            ct.nodes.append(node)
            if node.n_leaves >= ct.min_leaves:
                studies.append(cd_names[i])
        curr_idx += Xs[i].shape[0]

    with open('{}/genes.txt'.format(dirname), 'w') as of:
        [ of.write('{}\n'.format(gene)) for gene in genes ]

    with open('{}/cluster_studies.txt'.format(dirname), 'w') as of:
        [ of.write('{}\n'.format(study)) for study in studies ]

    ct.sample_idx = list(range(X.shape[0]))
    ct.n_leaves = X.shape[0]
    ct.fill_correlations(X)

    for node_idx, node in enumerate(ct.nodes):
        if node.n_leaves < ct.min_leaves:
            continue
        avg_age = np.mean(ages[node.sample_idx])
        save_npz('{}/node_{}_at_{}_has_{}_leaves.npz'.format(
            dirname, node_idx, avg_age, node.n_leaves
        ), node.correlations)
