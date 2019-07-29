from anndata import AnnData
from collections import Counter
import datetime
import errno
import math
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
from scipy.special import comb
from scipy.sparse import csr_matrix, vstack
from sklearn.preprocessing import LabelEncoder, normalize
import sys
import os

COLORS = [
    '#377eb8', '#ff7f00', '#4daf4a',
    '#f781bf', '#a65628', '#984ea3',
    '#999999', '#e41a1c', '#dede00',
    '#ffe119', '#e6194b', '#ffbea3',
    '#911eb4', '#46f0f0', '#f032e6',
    '#d2f53c', '#008080', '#e6beff',
    '#aa6e28', '#800000', '#aaffc3',
    '#808000', '#ffd8b1', '#000080',
    '#808080', '#fabebe', '#a3f4ff'
]

def tprint(string):
    string = str(string)
    sys.stdout.write(str(datetime.datetime.now()) + ' | ')
    sys.stdout.write(string + '\n')
    sys.stdout.flush()
    
def dispersion(X, eps=1e-10):
    mean = X.mean(0)
    dispersion = np.zeros(mean.shape)
    nonzero_idx = np.nonzero(mean > eps)[1]
    X_nonzero = X[:, nonzero_idx]
    nonzero_mean = X_nonzero.mean(0)
    nonzero_var = (X_nonzero.multiply(X_nonzero)).mean(0)
    temp = (nonzero_var / nonzero_mean)
    dispersion[mean > eps] = temp.A1
    dispersion[mean <= eps] = 0
    return dispersion.flatten()

def hvg(datasets, genes, hvg_type='dispersion'):
    datasets = datasets[:]
    X = vstack(datasets)
    genes = np.array(genes)

    if hvg_type == 'dispersion':
        vary = dispersion(X.tocsc())
    elif hvg_type == 'pca':
        from fbpca import pca
        U, s, Vt = pca(X, k=10)
        vary = np.absolute(Vt[:1, :]).flatten()

    assert(vary.shape[0] == X.shape[1])

    highest_vary_idx = np.argsort(vary)[::-1]
    genes = genes[highest_vary_idx]
    
    return genes, list(reversed(list(np.sort(vary.flatten()))))

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

def reduce_dimensionality(X, dim_red_k=100):
    k = min((dim_red_k, X.shape[0], X.shape[1]))
    from fbpca import pca
    U, s, Vt = pca(X, k=k) # Automatically centers.
    return U[:, range(k)] * s[range(k)]

def nearest(X, sites, approx=True):
    if approx:
        return nearest_approx(X, sites)
    else:
        return nearest_exact(X, sites)

def nearest_exact(X, sites):
    if sites.shape[0] == 0:
        raise ValueError('Sketch cannot be empty')
    if X.shape[1] != sites.shape[1]:
        raise ValueError('Dataset and sketch must have same '
                         'feature dimension.')

    from sklearn.neighbors import NearestNeighbors as NN
    
    nn = NN(n_neighbors=1, metric='manhattan').fit(sites)
    ind = nn.kneighbors(X)[1]

    site_to_idx = { site_idx: []
                    for site_idx in range(sites.shape[0]) }

    for idx, site_idx in enumerate(ind.flatten()):
        site_to_idx[site_idx].append(idx)

    return site_to_idx

def nearest_approx(X, sites):
    from annoy import AnnoyIndex
    
    assert(X.shape[1] == sites.shape[1])

    # Build index over site points.
    aindex = AnnoyIndex(sites.shape[1], metric='manhattan')
    for i in range(sites.shape[0]):
        aindex.add_item(i, sites[i, :])
    aindex.build(max(10, int(np.log2(X.shape[0]))))

    site_to_idx = { site_idx: []
                    for site_idx in range(sites.shape[0]) }

    for idx in range(X.shape[0]):
        # Find nearest site point.
        nearest_sites = aindex.get_nns_by_vector(X[idx, :], 1)
        if len(nearest_sites) < 1:
            continue
        site_idx = nearest_sites[0]
        site_to_idx[site_idx].append(idx)

    return site_to_idx

def print_cell_types(cell_types, intensity):
    # Print most intense cell types in cluster.
    intense_qtile = np.percentile(intensity, 95)
    cluster_types = cell_types[intensity > intense_qtile]
    for cell_type, count in Counter(cluster_types).most_common():
        print('{} ({})'.format(cell_type, count))
    sys.stdout.flush()

def print_gene_modules(corr, genes):
    from sklearn.cluster import AffinityPropagation
    ap = AffinityPropagation(
        affinity='precomputed',
    ).fit(corr)
    
    modules = {}
    for feature_idx, label in enumerate(ap.labels_):
        if label not in modules:
            modules[label] = []
        modules[label].append(feature_idx)

    for label_idx, label in enumerate(sorted(modules.keys())):
        if len(modules[label]) < 5:
            continue

        print('Module {}'.format(label_idx))
        print('\n'.join([
            str(genes[f_idx]) for f_idx in modules[label]
        ]))
        
def visualize_dictionary(ct, X_dimred, genes, cell_types,
                         namespace, dag_method, verbose=True):
    from anndata import AnnData
    from scanorama import visualize
    import scanpy as sc
    import seaborn as sns

    # KNN and UMAP.
    
    if verbose:
        tprint('Constructing KNN graph...')
    adata = AnnData(X=X_dimred)
    sc.pp.neighbors(adata, use_rep='X')

    if verbose:
        tprint('Visualizing with UMAP...')
    sc.tl.umap(adata, min_dist=0.5)
    embedding = np.array(adata.obsm['X_umap'])
    embedding[embedding < -20] = -20
    embedding[embedding > 20] = 20

    # Visualize cell types.
    
    le = LabelEncoder().fit(cell_types)
    cell_types_int = le.transform(cell_types)
    visualize(
        None, cell_types_int,
        '{}_pan_umap_{}_type'.format(namespace, dag_method),
        np.array(sorted(set(cell_types))),
        embedding=embedding,
        image_suffix='.png'
    )

    #max_intensity = ct.labels_.max()

    for c_idx in range(ct.labels_.shape[1]):
        intensity = ct.labels_[:, c_idx]
        intensity /= intensity.max()

        print('\nCluster {}'.format(c_idx))

        print_cell_types(cell_types, intensity)
        
        # Visualize cluster in UMAP coordinates.

        plt.figure()
        plt.title('Cluster {}'.format(c_idx))
        plt.scatter(embedding[:, 0], embedding[:, 1],
                    c=intensity, cmap=cm.get_cmap('Blues'), s=1)
        plt.savefig('{}_pan_umap_{}_cluster{}.png'
                    .format(namespace, dag_method, c_idx), dpi=500)


        plt.figure()
        plt.title('Cluster {}'.format(c_idx))
        plt.hist(intensity.flatten(), bins=100)
        plt.savefig('{}_pan_umap_{}_intensehist{}.png'
                    .format(namespace, dag_method, c_idx), dpi=500)

        intensity = (intensity > 0.8) * 1
        
        plt.figure()
        plt.title('Cluster {}'.format(c_idx))
        plt.scatter(embedding[:, 0], embedding[:, 1],
                    c=intensity, cmap=cm.get_cmap('Blues'), s=1)
        plt.savefig('{}_pan_umap_{}_member{}.png'
                    .format(namespace, dag_method, c_idx), dpi=500)
        
    for c_idx in range(ct.labels_.shape[1]):

        # Visualize covariance matrix.

        corr = ct.dictionary_[:, :, c_idx]
        corr[np.isnan(corr)] = 0

        #print('\nCluster {}'.format(c_idx))

        #print_gene_modules(corr, genes)
    
        gene_idx = np.sum(np.abs(corr), axis=1) > 0
        if np.sum(gene_idx) == 0:
            continue
        corr = corr[gene_idx]
        corr = corr[:, gene_idx]

        plt.figure()
        plt.title('Cluster {}'.format(c_idx))
        plt.rcParams.update({'font.size': 5})
        cmap = sns.diverging_palette(220, 10, as_cmap=True)
        corr_max = max(corr.max(), abs(corr.min()))
        sns.clustermap(corr, xticklabels=genes[gene_idx],
                       yticklabels=genes[gene_idx], cmap=cmap,
                       vmin=-corr_max, vmax=corr_max)
        plt.xticks(rotation=90)
        plt.yticks(rotation=90)
        plt.savefig('{}_pan_cov_{}_cluster{}.png'
                    .format(namespace, dag_method, c_idx), dpi=500)

def save_mtx(dir_name, X, genes):
    X = X.tocoo()
    
    if not os.path.exists(dir_name):
        mkdir_p(dir_name)

    with open(dir_name + '/matrix.mtx', 'w') as f:
        f.write('%%MatrixMarket matrix coordinate integer general\n')
        
        f.write('{} {} {}\n'.format(X.shape[1], X.shape[0], X.nnz))

        try:
            from itertools import izip
        except ImportError:
            izip = zip
        
        for i, j, val in izip(X.row, X.col, X.data):
            f.write('{} {} {}\n'.format(j + 1, i + 1, int(val)))

    with open(dir_name + '/genes.tsv', 'w') as f:
        for idx, gene in enumerate(genes):
            f.write('{}\t{}\n'.format(idx + 1, gene))

    with open(dir_name + '/barcodes.tsv', 'w') as f:
        for idx in range(X.shape[0]):
            f.write('cell{}-1\n'.format(idx))
