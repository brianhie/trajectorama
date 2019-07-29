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

NAMESPACE = 'mouse_develop_{}_{}'.format(CORR_METHOD, DAG_METHOD)

if RANDOM_PROJ:
    NAMESPACE += '_randproj'

all_datasets = []
all_namespaces = []
all_dimreds = []

from dataset_mouse_gastr_late_brain import datasets, namespaces, X_dimred
all_datasets += datasets
all_namespaces += namespaces
all_dimreds.append(X_dimred)
from dataset_mca_fetal_brain import datasets, namespaces, X_dimred
all_datasets += datasets
all_namespaces += namespaces
all_dimreds.append(X_dimred)
from dataset_cortical import datasets, namespaces, X_dimred
all_datasets += datasets
all_namespaces += namespaces
all_dimreds.append(X_dimred)
from dataset_mca_neonatal_brain import datasets, namespaces, X_dimred
all_datasets += datasets
all_namespaces += namespaces
all_dimreds.append(X_dimred)
from dataset_zeisel_adolescent_brain import datasets, namespaces, X_dimred
all_datasets += datasets
all_namespaces += namespaces
all_dimreds.append(X_dimred)
from dataset_saunders_adult_brain import datasets, namespaces, X_dimred
all_datasets += datasets
all_namespaces += namespaces
all_dimreds.append(X_dimred)
from dataset_mca_adult_brain import datasets, namespaces, X_dimred
all_datasets += datasets
all_namespaces += namespaces
all_dimreds.append(X_dimred)

def correct_scanorama(Xs, genes):
    from scanorama import correct
    Xs, genes = correct(
        Xs, [ genes for _ in Xs ], alpha=0, batch_size=10000
    )
    X = vstack(Xs)
    return X

def correct_scvi(Xs, genes):
    import torch
    use_cuda = True
    torch.cuda.set_device(1)

    from scvi.dataset.dataset import GeneExpressionDataset
    from scvi.inference import UnsupervisedTrainer
    from scvi.models import SCANVI, VAE
    from scvi.dataset.anndata import AnnDataset

    all_ann = [ AnnDataset(AnnData(X, var=genes)) for X in Xs ]

    all_dataset = GeneExpressionDataset.concat_datasets(*all_ann)

    vae = VAE(
        all_dataset.nb_genes,
        n_batch=all_dataset.n_batches,
        n_labels=all_dataset.n_labels,
        n_hidden=128,
        n_latent=30,
        n_layers=2,
        dispersion='gene'
    )
    trainer = UnsupervisedTrainer(vae, all_dataset, train_size=0.99999)
    n_epochs = 100
    #trainer.train(n_epochs=n_epochs)
    #torch.save(trainer.model.state_dict(),
    #           'data/harmonization.vae.pkl')
    trainer.model.load_state_dict(torch.load('data/harmonization.vae.pkl'))
    trainer.model.eval()

    full = trainer.create_posterior(
        trainer.model, all_dataset, indices=np.arange(len(all_dataset))
    )
    latent, batch_indices, labels = full.sequential().get_latent()

    return latent


if __name__ == '__main__':
    dirname = 'target/sparse_correlations/{}'.format(NAMESPACE)
    mkdir_p(dirname)

    hv_genes = None
    for i, dataset in enumerate(all_datasets):
        genes_hvg, _ = hvg([dataset.X], dataset.var['gene_symbols'], 'dispersion')
        if hv_genes is None:
            hv_genes = set([ g for gene in genes_hvg[:9690]
                             for g in gene.split(';') ])
        else:
            hv_genes &= set([ g for gene in genes_hvg[:9690]
                              for g in gene.split(';') ])
        tprint('{}: {}'.format(all_namespaces[i], len(hv_genes)))
        sys.stdout.flush()

    # Keep only those highly variable genes.

    Xs, genes = merge_datasets(
        [ dataset.X for dataset in all_datasets ],
        [ dataset.var['gene_symbols'] for dataset in all_datasets ],
        keep_genes=hv_genes, ds_names=all_namespaces,
        verbose=True
    )

    [ tprint(X.shape[0]) for X in Xs ]

    X = vstack(Xs)
    X = X.log1p()

    cell_types = np.concatenate(
        [ dataset.obs['cell_types'] for dataset in all_datasets ],
        axis=None
    )
    ages = np.concatenate(
        [ dataset.obs['ages'] for dataset in all_datasets ],
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
    )#.fit(X)

    studies = [ 'mouse_develop' ]
    curr_idx = 0
    for i, cd in enumerate(cds):
        for node in cd.nodes:
            node.sample_idx = np.array(node.sample_idx) + curr_idx
            ct.nodes.append(node)
            if node.n_leaves >= ct.min_leaves:
                studies.append(
                    all_namespaces[i]
                    if not all_namespaces[i].startswith('mca')
                    else 'mca_han'
                )
        curr_idx += all_dimreds[i].shape[0]
        tprint(all_dimreds[i].shape[0])

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

    exit()

    expr_type = 'scvi'

    if expr_type == 'scanorama':
        X = correct_scanorama(Xs, genes)
    if expr_type == 'scvi':
        X = correct_scvi(Xs, genes)
        X[np.isnan(X)] = 0
        X[np.isinf(X)] = 0

    if expr_type == 'uncorrected':
        # Geometric mean for nonnegative count data.
        C = np.vstack([
            (np.exp(
                ((1. / node.n_leaves) * np.log1p(X[node.sample_idx]).sum(0)) - 1
            ) + 1) / (
                (1. / node.n_leaves) * X[node.sample_idx].sum(0) + 1
            )
            for node in ct.nodes
            if node.n_leaves >= ct.min_leaves
        ])
    else:
        # Regular mean for continuous data.
        C = np.vstack([
            X[node.sample_idx].mean(0)
            for node in ct.nodes
            if node.n_leaves >= ct.min_leaves
        ])

    np.save('data/expression_cluster_{}.npy'.format(expr_type), C)

    adata = AnnData(X=C)
    adata.obs['age'] = [
        np.mean(ages[node.sample_idx])
        for node_idx, node in enumerate(ct.nodes)
        if node.n_leaves >= ct.min_leaves
    ]
    adata.obs['study'] = studies

    for knn in [ 15, 20, 30, 40 ]:
        sc.pp.neighbors(adata, n_neighbors=knn, use_rep='X')
        draw_graph(adata, layout='fa')
        sc.pl.draw_graph(
            adata, color='age', edges=True, edges_color='#CCCCCC',
            save='_{}_expr_gmean_k{}.png'
            .format(NAMESPACE + '_' + expr_type, knn)
        )
        sc.pl.draw_graph(
            adata, color='study', edges=True, edges_color='#CCCCCC',
            save='_{}_expr_gmean_study_k{}.png'
            .format(NAMESPACE + '_' + expr_type, knn)
        )
        #sc.tl.diffmap(adata)
        #adata.uns['iroot'] = np.flatnonzero(adata.obs['age'] < 9.6)[0]
        #sc.tl.dpt(adata)
        #from scipy.stats import pearsonr
        #print(pearsonr(adata.obs['dpt_pseudotime'], adata.obs['age']))
        #sys.stdout.flush()

    adata = AnnData(X=X)
    adata.obs['age'] = ages
    adata.obs['study'] = [ '_'.join(ct.split('_')[:3]) for ct in cell_types ]
    sc.pp.neighbors(adata)#, use_rep='X')
    sc.tl.umap(adata, init_pos='random')
    sc.pl.scatter(
        adata, color='study', basis='umap',
        save='_{}_umap_study.png'.format(NAMESPACE + '_' + expr_type)
    )
    sc.pl.scatter(
        adata, color='age', basis='umap',
        save='_{}_umap_age.png'.format(NAMESPACE + '_' + expr_type)
    )
    np.save('data/umap_coord_{}.npy'.format(expr_type), adata.obsm['X_umap'])
