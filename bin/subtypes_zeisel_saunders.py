import numpy as np
import pandas as pd
from scipy.spatial import distance
from scipy.cluster import hierarchy
import seaborn as sns
from sklearn.metrics import pairwise_distances
import sys

from pan_corr import pearson_multi, spearman_multi
from process import merge_datasets
from utils import *

CORR_METHOD = 'spearman'
CORR_CUTOFF = 0.

def import_data():
    all_datasets, all_namespaces, all_dimreds = [], [], []

    from dataset_zeisel_adolescent_brain import datasets, namespaces, X_dimred
    all_datasets += datasets
    all_namespaces += namespaces
    all_dimreds.append(X_dimred)
    from dataset_saunders_adult_brain import datasets, namespaces, X_dimred
    all_datasets += datasets
    all_namespaces += namespaces
    all_dimreds.append(X_dimred)

    return all_datasets, all_namespaces, all_dimreds

def sub_coexpr(X, sub_types, return_expr=False, min_samples=500,
               corr_cutoff=CORR_CUTOFF, corr_method=CORR_METHOD):
    n_cells, n_genes = X.shape
    sub_types = np.array(sub_types)

    triu_idx = np.triu_indices(n_genes)

    coexpr = []
    sub_types_uniq = sorted(set(sub_types))
    sub_types_used = []
    if return_expr:
        expr = []
    for sub_type in sub_types_uniq:
        X_sub = X[sub_types == sub_type].todense()

        if X_sub.shape[0] < min_samples or sub_type.endswith('_NA'):
            continue

        if corr_method == 'pearson':
            corr = pearson_multi(X_sub)
        else:
            corr = spearman_multi(X_sub)
        corr[np.isnan(corr)] = 0.
        corr[np.abs(corr) < corr_cutoff] = 0.

        coexpr.append(corr[triu_idx])
        sub_types_used.append(sub_type + '_' + str(X_sub.shape[0]))

        if return_expr:
            expr.append(X_sub.mean(0).flatten())

    if return_expr:
        expr = np.array(expr)
        if len(expr.shape) == 3:
            expr = np.reshape(expr, (expr.shape[0], expr.shape[2]))
        return np.array(coexpr), sub_types_used, expr

    return np.array(coexpr), sub_types_used

def plot_clustermap(df, dist, suffix, prefix='subtypes_zeisel_saunders'):
    row_linkage = hierarchy.linkage(dist, method='average')
    col_linkage = hierarchy.linkage(dist.T, method='average')

    plt.figure(figsize=(10, 10))
    sns.clustermap(
        df, xticklabels=True, yticklabels=True,
        row_linkage=row_linkage, col_linkage=col_linkage,
    )
    plt.gcf().set_size_inches(40, 40)
    plt.tight_layout()
    plt.savefig(
        'figures/{}_heatmap_{}.png'.format(prefix, suffix), dpi=300
    )
    plt.savefig(
        'figures/{}_heatmap_{}.svg'.format(prefix, suffix)
    )
    plt.close()

    return row_linkage, col_linkage

def interpret_clustermap(coexpr, genes, sub_types, linkage,
                         n_clusters=5, n_report=100):
    clusters = hierarchy.fcluster(linkage, n_clusters, 'maxclust')
    interpret_clusters(coexpr, clusters, genes, sub_types, n_report)

def interpret_clusters(coexpr, clusters, genes, sub_types, n_report=100):
    clusters = np.array(clusters)
    sub_types = np.array(sub_types)

    uniq_clusters = sorted(set(clusters))
    cluster_coexprs = []
    for i, cluster in enumerate(uniq_clusters):
        cluster_idx = clusters == cluster
        print('Cluster {}\n'.format(i) +
              '\n'.join(sub_types[cluster_idx]) +
              '\n')
        cluster_coexprs.append(coexpr[cluster_idx].mean(0))

    coexpr_mean = coexpr.mean(0)
    triu_idx = np.triu_indices(len(genes))
    gene_pairs = np.array([
        (genes[gidx_i], genes[gidx_j])
        for gidx_i, gidx_j in zip(triu_idx[0], triu_idx[1])
    ])

    for c_idx, cluster_coexpr in enumerate(cluster_coexprs):
        centroid_dist = cluster_coexpr - coexpr_mean # Sign is important.
        farthest_idx = np.argsort(-centroid_dist)
        ranked_pairs = gene_pairs[farthest_idx]
        ranked_dists = centroid_dist[farthest_idx]

        print('Cluster {}\n'.format(c_idx))
        for pair, weight in zip(ranked_pairs[:n_report],
                                ranked_dists[:n_report]):
            print('{}\t{}\t{}'.format(pair[0], pair[1], weight))

        ranked_genes = []
        used_genes = set()
        for gene1, gene2 in ranked_pairs:
            if gene1 not in used_genes:
                ranked_genes.append(gene1)
                used_genes.add(gene1)
            if gene2 not in used_genes:
                ranked_genes.append(gene2)
                used_genes.add(gene2)
        print('Cluster {}\n'.format(c_idx) +
              '\n'.join(ranked_genes) +
              '\n')

if __name__ == '__main__':
    all_datasets, all_namespaces, all_dimreds = import_data()

    hv_genes = None
    for i, dataset in enumerate(all_datasets):
        genes_hvg, _ = hvg([dataset.X], dataset.var['gene_symbols'], 'dispersion')
        if hv_genes is None:
            hv_genes = set([ g for gene in genes_hvg[:4000]
                             for g in gene.split(';') ])
        else:
            hv_genes &= set([ g for gene in genes_hvg[:4000]
                              for g in gene.split(';') ])
        tprint('{}: {}'.format(all_namespaces[i], len(hv_genes)))
        sys.stdout.flush()

    [ X_zeisel, X_saunders ], genes = merge_datasets(
        [ dataset.X for dataset in all_datasets ],
        [ dataset.var['gene_symbols'] for dataset in all_datasets ],
        keep_genes=hv_genes, ds_names=all_namespaces,
        verbose=True
    )

    with open('genes.txt', 'w') as of:
        of.write('\n'.join(genes) + '\n')

    coexpr_zeisel, types_zeisel, expr_zeisel = sub_coexpr(
        X_zeisel, all_datasets[0].obs['sub_types'], True
    )
    coexpr_saunders, types_saunders, expr_saunders = sub_coexpr(
        X_saunders, all_datasets[1].obs['sub_types'], True
    )

    print('Coexpression Zeisel: {}'.format(coexpr_zeisel.shape))
    print('Coexpression Saunders: {}'.format(coexpr_saunders.shape))
    print('Expression Zeisel: {}'.format(expr_zeisel.shape))
    print('Expression Saunders: {}'.format(expr_saunders.shape))

    dist = pairwise_distances(coexpr_zeisel, coexpr_saunders)
    df = pd.DataFrame(-dist, index=types_zeisel, columns=types_saunders)
    plot_clustermap(df, dist, 'coexpr')

    coexpr_cat = np.concatenate([ coexpr_zeisel, coexpr_saunders ])
    types_cat = types_zeisel + types_saunders

    dist = pairwise_distances(coexpr_cat)
    df = pd.DataFrame(-dist, index=types_cat, columns=types_cat)
    linkage, _ = plot_clustermap(df, dist, 'coexprcat')
    interpret_clustermap(coexpr_cat, genes, types_cat, linkage)

    expr_cat = np.concatenate([ expr_zeisel, expr_saunders ])

    dist = pairwise_distances(expr_cat)
    df = pd.DataFrame(-dist, index=types_cat, columns=types_cat)
    plot_clustermap(df, dist, 'exprcat')
