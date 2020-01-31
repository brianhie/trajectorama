from anndata import AnnData
import numpy as np
import scanpy as sc
from scipy.sparse import vstack, save_npz, csr_matrix
import sys
import trajectorama

from draw_graph import draw_graph
from process import merge_datasets
from utils import *

NAMESPACE = 'mouse_develop'

def import_data():
    all_datasets, all_namespaces, all_dimreds = [], [], []

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

    return all_datasets, all_namespaces, all_dimreds

def correct_scanorama(Xs, genes):
    from scanorama import correct
    Xs, genes = correct(
        Xs, [ genes for _ in Xs ], alpha=0, batch_size=10000
    )
    X = vstack(Xs)
    return X

def correct_harmony(X_dimreds):
    from subprocess import Popen

    dirname = 'target/harmony'
    mkdir_p(dirname)

    embed_fname = '{}/embedding.txt'.format(dirname)
    label_fname = '{}/labels.txt'.format(dirname)

    X_dimred = np.concatenate(X_dimreds)
    np.savetxt(embed_fname, X_dimred)

    labels = []
    curr_label = 0
    for i, a in enumerate(X_dimreds):
        labels += list(np.zeros(a.shape[0]) + curr_label)
        curr_label += 1
    labels = np.array(labels, dtype=int)
    np.savetxt(label_fname, labels)

    tprint('Integrating with harmony...')
    rcode = Popen('Rscript bin/R/harmony.R {} {} > harmony.log 2>&1'
                  .format(embed_fname, label_fname), shell=True).wait()
    if rcode != 0:
        sys.stderr.write('ERROR: subprocess returned error code {}\n'
                         .format(rcode))
        exit(rcode)
    tprint('Done with harmony integration')

    integrated = np.loadtxt('{}/integrated.txt'.format(dirname))
    return integrated

def correct_scvi(Xs, genes):
    import torch
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    from scvi.dataset import AnnDatasetFromAnnData
    from scvi.dataset.dataset import GeneExpressionDataset
    from scvi.inference import UnsupervisedTrainer
    from scvi.models import VAE

    all_ann = [ AnnDatasetFromAnnData(AnnData(X, var=genes)) for X in Xs ]

    all_dataset = GeneExpressionDataset()
    all_dataset.populate_from_datasets(all_ann)

    vae = VAE(
        all_dataset.nb_genes,
        n_batch=all_dataset.n_batches,
        n_labels=all_dataset.n_labels,
        n_hidden=128,
        n_latent=30,
        n_layers=2,
        dispersion='gene'
    )
    trainer = UnsupervisedTrainer(
        vae, all_dataset, train_size=1., use_cuda=True,
    )
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

    all_datasets, all_namespaces, all_dimreds = import_data()

    # Determine highly variable genes.

    hv_genes = None
    for i, dataset in enumerate(all_datasets):
        genes_hvg, _ = hvg([dataset.X], dataset.var['gene_symbols'], 'dispersion')
        if hv_genes is None:
            hv_genes = set([ g for gene in genes_hvg[:12000]
                             for g in gene.split(';') ])
        else:
            hv_genes &= set([ g for gene in genes_hvg[:12000]
                              for g in gene.split(';') ])
        tprint('{}: {}'.format(all_namespaces[i], len(hv_genes)))
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
    ages = np.concatenate(
        [ dataset.obs['ages'] for dataset in all_datasets ],
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
        of.write('mouse_develop\n')
        [ of.write('{}\n'.format(set(studies[sample_idx]).pop()))
          for sample_idx in sample_idxs[1:] ]

    tprint('Saving coexpression matrices to "{}" directory...'
           .format(dirname))
    for node_idx, (X_coexpr, sample_idx) in enumerate(zip(Xs_coexpr, sample_idxs)):
        age = np.mean(ages[sample_idx])
        n_cells = len(sample_idx)
        save_npz('{}/node_{}_at_{}_has_{}_leaves.npz'
                 .format(dirname, node_idx, age, n_cells),
                 csr_matrix(X_coexpr))
    exit()

    # Benchmarking code.

    expr_type = 'uncorrected'

    if expr_type == 'harmony':
        X = correct_harmony(all_dimreds)
    if expr_type == 'scanorama':
        X = correct_scanorama(Xs, genes)
    if expr_type == 'scvi':
        X = correct_scvi(Xs, genes)
        X[np.isnan(X)] = 0
        X[np.isinf(X)] = 0

    C = np.vstack([
        X[sample_idx].mean(0) for sample_idx in sample_idxs
    ])

    np.save('data/expression_cluster_{}.npy'.format(expr_type), C)

    adata = AnnData(X=C)
    adata.obs['age'] = [
        np.mean(ages[sample_idx]) for sample_idx in sample_idxs
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

    adata = AnnData(X=X)
    adata.obs['age'] = ages
    adata.obs['study'] = [ '_'.join(ct.split('_')[:3]) for ct in cell_types ]
    sc.pp.neighbors(adata)
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
