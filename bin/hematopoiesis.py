from anndata import AnnData
import numpy as np
import scanpy as sc
from scipy.sparse import vstack, save_npz, csr_matrix
import sys
import trajectorama

from draw_graph import draw_graph
from process import merge_datasets
from utils import *

NAMESPACE = 'hematopoiesis'

def import_data():
    all_datasets, all_namespaces, all_dimreds = [], [], []

    from dataset_bonemarrow_ica import datasets, namespaces, X_dimred
    all_datasets += datasets
    all_namespaces += namespaces
    all_dimreds.append(X_dimred)
    from dataset_cordblood_ica import datasets, namespaces, X_dimred
    all_datasets += datasets
    all_namespaces += namespaces
    all_dimreds.append(X_dimred)
    from dataset_zeng_develop_thymus import datasets, namespaces, X_dimred
    all_datasets += datasets
    all_namespaces += namespaces
    all_dimreds.append(X_dimred)
    from dataset_pbmc_10x import datasets, namespaces, X_dimred
    all_datasets += datasets
    all_namespaces += namespaces
    all_dimreds.append(X_dimred)

    return all_datasets, all_namespaces, all_dimreds

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

    all_datasets, all_namespaces, all_dimreds = import_data()

    # Determine highly variable genes.

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

    # Merge data and metadata.

    Xs, genes = merge_datasets(
        [ dataset.X for dataset in all_datasets ],
        [ dataset.var['gene_symbols'] for dataset in all_datasets ],
        keep_genes=hv_genes, ds_names=all_namespaces,
        verbose=True
    )

    X = vstack(Xs)
    X = X.log1p()

    cell_types = np.concatenate(
        [ dataset.obs['cell_types'] for dataset in all_datasets ],
        axis=None
    )
    studies = np.concatenate([
        np.array([ all_namespaces[i] ] * Xs[i].shape[0] )
        for i in range(len(Xs))
    ])

    # Run Trajectorama clustering and featurization.

    Xs_coexpr, sample_idxs = trajectorama.transform(
        X, studies,
        X_dimred=np.concatenate(all_dimreds),
        log_transform=False,
        corr_cutoff=0.7,
        corr_method='spearman',
        cluster_method='louvain',
        min_cluster_samples=500,
        n_jobs=1,
        verbose=2,
    )

    # Save to files for additional analysis.

    with open('{}/genes.txt'.format(dirname), 'w') as of:
        [ of.write('{}\n'.format(gene)) for gene in genes ]

    with open('{}/cluster_studies.txt'.format(dirname), 'w') as of:
        of.write('hematopoiesis\n')
        [ of.write('{}\n'.format(set(studies[sample_idx]).pop()))
          for sample_idx in sample_idxs[1:] ]

    with open('{}/cell_type_composition.txt'.format(dirname), 'w') as of:
        of.write('node')
        for cell_type in uniq_cell_types:
            of.write('\t{}'.format(cell_type))
        of.write('\n')
        for node_idx, sample_idx in enumerate(sample_idxs):
            fractions = count_cell_types(cell_types[sample_idx])
            of.write('{}\t'.format(node_idx))
            of.write('\t'.join([ str(frac) for frac in fractions ]) + '\n')

    tprint('Saving coexpression matrices to "{}" directory...'
           .format(dirname))
    for node_idx, (X_coexpr, sample_idx) in enumerate(zip(Xs_coexpr, sample_idxs)):
        n_cells = len(sample_idx)
        save_npz('{}/node_{}_has_{}_leaves.npz'
                 .format(dirname, node_idx, n_cells),
                 csr_matrix(X_coexpr))

    exit()

    # Benchmarking code.

    from mouse_develop import (
        correct_harmony,
        correct_scanorama,
        correct_scvi,
    )

    expr_type = 'uncorrected'

    if expr_type == 'harmony':
        X = correct_harmony(all_dimreds)
    if expr_type == 'scanorama':
        X = correct_scanorama(Xs, genes)
    if expr_type == 'scvi':
        nonzero_idx = np.array(X.sum(1) > 0).flatten()
        X = np.zeros((X.shape[0], 30))
        X_scvi = correct_scvi(Xs, genes)
        X[nonzero_idx, :] = X_scvi
        X[np.isnan(X)] = 0
        X[np.isinf(X)] = 0

    C = np.vstack([
        X[sample_idx].mean(0)
        sample_idx in sample_idxs
    ])

    adata = AnnData(X=C)
    adata.obs['study'] = studies

    for knn in [ 15, 20, 30, 40 ]:
        sc.pp.neighbors(adata, n_neighbors=knn, use_rep='X')
        draw_graph(adata, layout='fa')
        sc.pl.draw_graph(
            adata, color='study', edges=True, edges_color='#CCCCCC',
            save='_{}_expr_gmean_k{}.png'
            .format(NAMESPACE + '_' + expr_type, knn)
        )
        sys.stdout.flush()

    adata = AnnData(X=X)
    adata.obs['study'] = [ '_'.join(ct.split('_')[:3]) for ct in cell_types ]
    sc.pp.neighbors(adata)#, use_rep='X')
    sc.tl.umap(adata, init_pos='random')
    sc.pl.scatter(
        adata, color='study', basis='umap',
        save='_{}_umap_study.png'.format(NAMESPACE + '_' + expr_type)
    )
    np.save('data/hema_umap_coord_{}.npy'.format(expr_type),
            adata.obsm['X_umap'])
