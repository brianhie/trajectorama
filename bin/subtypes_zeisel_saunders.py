from anndata import AnnData
from matplotlib import cm
import numpy as np
import os
from scanorama import process_data, plt, reduce_dimensionality, visualize
import scanpy as sc
from scipy.sparse import vstack, save_npz
import seaborn as sns
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import LabelEncoder, normalize
import sys

from draw_graph import draw_graph
from pan_corr import pearson_multi, spearman_multi
from process import merge_datasets
from utils import *

CORR_METHOD = 'pearson'

CORR_CUTOFF = 0.
RANDOM_PROJ = False

all_datasets = []
all_namespaces = []
all_dimreds = []

from dataset_zeisel_adolescent_brain import datasets, namespaces, X_dimred
all_datasets += datasets
all_namespaces += namespaces
all_dimreds.append(X_dimred)
from dataset_saunders_adult_brain import datasets, namespaces, X_dimred
all_datasets += datasets
all_namespaces += namespaces
all_dimreds.append(X_dimred)

def sub_coexpr(X, sub_types):
    n_cells, n_genes = X.shape
    sub_types = np.array(sub_types)

    triu_idx = np.triu_indices(n_genes)

    coexpr = []
    sub_types_uniq = sorted(set(sub_types))
    sub_types_used = []
    for sub_type in sub_types_uniq:
        X_sub = X[sub_types == sub_type].todense()

        if X_sub.shape[0] < 500:
            continue

        if CORR_METHOD == 'pearson':
            corr = pearson_multi(X_sub)
        else:
            corr = spearman_multi(X_sub)
        corr[np.isnan(corr)] = 0.
        corr[np.abs(corr) < CORR_CUTOFF] = 0

        coexpr.append(corr[triu_idx])
        sub_types_used.append(sub_type)

    return np.array(coexpr), sub_types_used

if __name__ == '__main__':
    hv_genes = None
    for i, dataset in enumerate(all_datasets):
        genes_hvg, _ = hvg([dataset.X], dataset.var['gene_symbols'], 'dispersion')
        if hv_genes is None:
            hv_genes = set([ g for gene in genes_hvg[:5000]
                             for g in gene.split(';') ])
        else:
            hv_genes &= set([ g for gene in genes_hvg[:5000]
                              for g in gene.split(';') ])
        tprint('{}: {}'.format(all_namespaces[i], len(hv_genes)))
        sys.stdout.flush()

    # Keep only those highly variable genes.

    [ X_zeisel, X_saunders ], genes = merge_datasets(
        [ dataset.X for dataset in all_datasets ],
        [ dataset.var['gene_symbols'] for dataset in all_datasets ],
        keep_genes=hv_genes, ds_names=all_namespaces,
        verbose=True
    )

    coexpr_zeisel, types_zeisel = sub_coexpr(
        X_zeisel, all_datasets[0].obs['sub_types']
    )
    coexpr_saunders, types_saunders = sub_coexpr(
        X_saunders, all_datasets[1].obs['sub_types']
    )

    dist = pairwise_distances(coexpr_zeisel, coexpr_saunders)

    plt.figure()
    sns.clustermap(-dist)
    plt.savefig('figures/subtypes_zeisel_saunders_heatmap.png', dpi=300)
    plt.close()
