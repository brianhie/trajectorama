import numpy as np
import pandas as pd
from scipy.sparse import vstack, save_npz, csr_matrix
from sklearn.metrics import pairwise_distances
import sys
import trajectorama

from process import merge_datasets
from subtypes_zeisel_saunders import (
    interpret_clustermap, plot_clustermap, sub_coexpr,
)
from utils import *

NAMESPACE = 'microglia'

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

    # Determine highly variable genes.

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

    # Merge data and metadata.

    X_studies, genes = merge_datasets(
        [ dataset.X for dataset in all_datasets ],
        [ dataset.var['gene_symbols'] for dataset in all_datasets ],
        keep_genes=hv_genes, ds_names=all_namespaces,
        verbose=True
    )

    X = vstack(X_studies)
    X = X.log1p()

    cell_types = np.concatenate(
        [ dataset.obs['cell_types'] for dataset in all_datasets ],
        axis=None
    )
    ages = np.concatenate(
        [ dataset.obs['ages'] for dataset in all_datasets ],
        axis=None
    )

    # Look at cell types individually.

    coexpr, types, expr = sub_coexpr(
        X, cell_types, return_expr=True, min_samples=0,
        corr_cutoff=0., corr_method='spearman'
    )
    dist = pairwise_distances(coexpr)
    dist[np.isnan(dist)] = 0.
    df = pd.DataFrame(-dist, index=types, columns=types)
    linkage, _ = plot_clustermap(df, dist, 'coexpr', 'microglia')

    interpret_clustermap(coexpr, genes, types, linkage,
                         n_clusters=2, n_report=150)


    # Run Trajectorama clustering and featurization.

    Xs_coexpr, sample_idxs = trajectorama.transform(
        X, cell_types,
        X_dimred=np.concatenate(all_dimreds),
        log_transform=False,
        corr_cutoff=0.1,
        corr_method='spearman',
        cluster_method='louvain',
        min_cluster_samples=500,
        n_jobs=1,
        verbose=2,
    )

    # Save to files for additional analysis

    with open('{}/genes.txt'.format(dirname), 'w') as of:
        [ of.write('{}\n'.format(gene)) for gene in genes ]

    with open('{}/cluster_studies.txt'.format(dirname), 'w') as of:
        of.write('microglia\n')
        [ of.write('{}\n'.format(set(cell_types[sample_idx]).pop()))
          for sample_idx in sample_idxs[1:] ]

    tprint('Saving coexpression matrices to "{}" directory...'
           .format(dirname))
    for node_idx, (X_coexpr, sample_idx) in enumerate(zip(Xs_coexpr, sample_idxs)):
        age = np.mean(ages[sample_idx])
        n_cells = len(sample_idx)
        save_npz('{}/node_{}_at_{}_has_{}_leaves.npz'
                 .format(dirname, node_idx, age, n_cells),
                 csr_matrix(X_coexpr))
