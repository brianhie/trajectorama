import numpy as np
import os
import random
from sklearn.preprocessing import normalize
import warnings

from .pan_dag import PanDAG
from .utils import *

# Default parameters.
CORR_CUTOFF = 0.3
CORR_METHOD = 'spearman'
CLUSTER_METHOD = 'louvain'
LOG_TRANSFORM = False
MIN_CLUSTER_SAMPLES = 500
N_JOBS = 1
SEED = None
VERBOSE = 0

def transform(
        X,
        studies,
        X_dimred=None,
        log_transform=LOG_TRANSFORM,
        corr_method=CORR_METHOD,
        corr_cutoff=CORR_CUTOFF,
        cluster_method=CLUSTER_METHOD,
        min_cluster_samples=MIN_CLUSTER_SAMPLES,
        seed=SEED,
        n_jobs=N_JOBS,
        verbose=VERBOSE,
):
    """
    Transforms a data matrix from a set of studies by clustering the
    studies separately and computing a coexpression matrix for each of
    the clusters.

    Parameters
    ----------
    X: `numpy.ndarray` or `scipy.sparse.csr_matrix`
        Matrix with rows corresponding to all of the samples and columns
        corresponding to features that define the correlation matrices.
    studies: `list` or `np.ndarray`
        Metadata assigning corresponding samples in `X` to a different
        study, i.e., the sample `X[i]` belongs to the study `studies[i]`.
    X_dimred: `np.ndarray`, optional (default: `None`)
        Dense matrix with reduced dimensionality of `X`. If `None`, uses
        randomized SVD to reduce the dimensions of each study separately.
    log_transform: `bool`, optional (default: `False`)
        Before analysis, natural log transform `X` after adding a
        pseudocount of 1.
    corr_method: `str`, optional (default: 'spearman')
        Correlation measurement to use. Spearman is used as default to
        its robustness to many transformations of the underlying data,
        though Pearson correlation ('pearson') is available as well.
    corr_cutoff: `int`, optional (default: 0.3)
        Cutoff below which absolute correlations are set to zero. Used
        to sparsify the correlation matrix.
    cluster_method: `str`, optional (default: 'louvain')
        Method used to hierarchically cluster cells. Defaults to
        'louvain' clustering on the 15-nearest neighbors graph due to
        its close to linear-time asymptotics, though (quadratic)
        agglomerative cluster ('agg_ward') is available as well.
    min_cluster_samples: `int`, optional (default: 500)
        Minimum number of samples within a cluster for which the
        algorithm will compute a coexpression matrix. Values below 100
        may be sensitive to noise.
    seed: `int`, optional (default: None)
        Random seed used to enforce reproducibility of algorithm; if
        `None`, no random seed is used.
    n_jobs: `int`, optional (default: 1)
        Each jobs computes a single correlation matrix.
    verbose: `bool` or `int`, optional (default: 0)
        Log information if positive.

    Returns
    -------
    Xs_coexpr: `list` of `np.ndarray`
        A list of coexpression matrices, computed over the subset of
        samples defined in `sample_idxs` (see below).
    sample_idxs: `list` of `numpy.ndarray`
        Each element of this list is an array that indexes into `X`,
        thereby definining a subset of `X`. A coexpression matrix is
        computed for each of these subsets.
    """
    # Setup and error checking.

    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)

    _check_params(X, studies, X_dimred, corr_method, corr_cutoff,
                  cluster_method, min_cluster_samples, n_jobs)

    # Preprocessing.

    studies = np.array(studies)
    uniq_studies, seen_studies = [], set()
    for study in studies:
        if study not in seen_studies:
            uniq_studies.append(study)
            seen_studies.add(study)
    study_idxs = [
        np.where(studies == study)[0] for study in uniq_studies
    ]

    if X_dimred is None:
        X_dimred = np.concatenate([
            reduce_dimensionality(normalize(X[study_idx]))
            for study_idx in study_idxs
        ])

    if log_transform:
        X = X.log1p()

    # Perform panresolution clustering of studies separately.

    if verbose:
        tprint('Performing panresolution clustering...')

    cds = [
        PanDAG(
            cluster_method=cluster_method,
            verbose=verbose,
        ).fit(X_dimred[study_idx])
        for study_idx in study_idxs
    ]

    # Featurize each cluster by coexpression.

    if verbose:
        tprint('Featurizing clusters by coexpression...')

    sample_idxs = [ list(range(X.shape[0])) ]
    for study_idx, cd in zip(study_idxs, cds):
        for node in cd.nodes:
            if node.n_leaves >= min_cluster_samples:
                sample_idxs.append(study_idx[node.sample_idx])

    Xs_coexpr = fill_correlations(
        X, sample_idxs,
        corr_method=corr_method, corr_cutoff=corr_cutoff,
        n_jobs=n_jobs, verbose=verbose,
    )

    if verbose:
        tprint('Finished with Trajectorama transformation.')

    return Xs_coexpr, sample_idxs

def _check_params(X, studies, X_dimred, corr_method, corr_cutoff,
                 cluster_method, min_cluster_samples, n_jobs):
    """
    Basic parameter checking.
    """
    if X.shape[0] != len(studies):
        raise ValueError('Number of samples in X ({}) does not match '
                         'length of studies ({})'
                         .format(X.shape[0], len(studies)))
    if X_dimred is not None and X_dimred.shape[0] != X.shape[0]:
        raise ValueError('Number of samples in X ({}) does not match '
                         'number of samples in X_dimred ({})'
                         .format(X.shape[0], X_dimred.shape[0]))

    valid_corr_methods = { 'pearson', 'spearman' }
    if corr_method not in valid_corr_methods:
        raise ValueError('Invalid corr_method {}, must be one of {}'
                         .format(corr_method, valid_corr_methods))

    if corr_cutoff < -1. or corr_cutoff > 1.:
        raise ValueError('Invalid corr_cutoff {}, must be between -1 and 1'
                         .format(corr_cutoff))

    valid_cluster_methods = { 'louvain', 'agg_ward' }
    if cluster_method not in valid_cluster_methods:
        raise ValueError('Invalid cluster_method {}, must be one of {}'
                         .format(cluster_method, valid_cluster_methods))

    if min_cluster_samples < 0:
        raise ValueError('Invalid min_cluster_samples of {}, must be '
                         'nonnegative'.format(min_cluster_samples))
    if min_cluster_samples < 100:
        warnings.warn('Using less than 100 samples to estimate '
                      'correlations may be less robust to noise.')

    if n_jobs < 1:
        raise ValueError('Invalid n_jobs of {}, must be positive'
                         .format(n_jobs))

def pearson_multi(X):
    """
    Compute Pearson correlation across a set of samples.

    Parameters
    ----------
    X: `numpy.ndarray` or `scipy.sparse.csr_matrix`
        Matrix with rows corresponding to all of the samples and columns
        corresponding to features that define the correlation matrices.

    Returns
    -------
    corr: `np.ndarray`
        Dense Pearson coexpression matrix with dimension
        `(X.shape[1], X.shape[1])`.
    """
    from scipy.special import betainc
    from sklearn.covariance import EmpiricalCovariance
    from sklearn.preprocessing import StandardScaler

    X = StandardScaler().fit_transform(X)

    corr = EmpiricalCovariance(
        store_precision=False,
        assume_centered=True).fit(X).covariance_
    corr[corr > 1.] = 1.
    corr[corr < -1.] = -1.

    return corr

def spearman_multi(X):
    """
    Compute Spearman correlation across a set of samples.

    Parameters
    ----------
    X: `numpy.ndarray` or `scipy.sparse.csr_matrix`
        Matrix with rows corresponding to all of the samples and columns
        corresponding to features that define the correlation matrices.

    Returns
    -------
    corr: `np.ndarray`
        Dense Spearman coexpression matrix with dimension
        `(X.shape[1], X.shape[1])`.
    """
    from scipy.stats import spearmanr

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        corr, corr_scores = spearmanr(
            X, nan_policy='omit'
        )

    return corr

def _corr_worker(X, verbose, node_idx, n_nodes, corr_method,
                 cutoff):
    """
    joblib wrapper around coexpression matrix computation that sets
    nan values to zero and sparsifies the matrices.
    """
    n_samples, n_features = X.shape

    # Calculate correlation of positive cells.

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

        if corr_method == 'pearson':
            corr = pearson_multi(X)

        elif corr_method == 'spearman':
            corr = spearman_multi(X)

        else:
            raise ValueError('Invalid correlation method {}'.
                             format(corr_method))

    corr[np.isnan(corr)] = 0.
    corr[np.abs(corr) < cutoff] = 0

    if verbose > 1:
        tprint('Filled node {} out of {} with {} samples and '
               '{} nonzero correlations'
               .format(node_idx + 1, n_nodes, X.shape[0],
                       np.count_nonzero(corr)))

    return corr

def fill_correlations(
        X, sample_idxs,
        corr_method=CORR_METHOD,
        corr_cutoff=CORR_CUTOFF,
        n_jobs=N_JOBS,
        verbose=VERBOSE,
):
    """
    Compute correlations for subsets of data.

    Parameters
    ----------
    X: `numpy.ndarray` or `scipy.sparse.csr_matrix`
        Matrix with rows corresponding to all of the samples and columns
        corresponding to features that define the correlation matrices.
    sample_idxs: `list` of `numpy.ndarray`
        Each element of this list is an array that indexes into `X`,
        thereby definining a subset of `X`. A coexpression matrix is
        computed for each of these subsets.
    corr_method: `str`, optional (default: 'spearman')
        Correlation measurement to use.
    corr_cutoff: `int`, optional (default: 0.3)
        Cutoff below which absolute correlations are set to zero. Used
        to sparsify the correlation matrix.
    n_jobs: `int`, optional (default: 1)
        Each jobs computes a single correlation matrix.
    verbose: `bool` or `int`, optional (default: 0)
        Log information if positive.

    Returns
    -------
    Xs_coexpr: `list` of `np.ndarray`
        A list of coexpression matrices, computed over the subset of
        samples defined in `sample_idxs`.
    """
    from joblib import Parallel, delayed
    from scipy.special import comb

    n_samples, n_features = X.shape
    n_correlations = int(comb(n_features, 2) + n_features)

    n_nodes = len(sample_idxs)

    if verbose > 1:
        tprint('Found {} nodes'.format(n_nodes))

    results = Parallel(n_jobs=n_jobs) (
        delayed(_corr_worker)(
            X[sample_idx].toarray(), verbose, node_idx,
            n_nodes, corr_method, corr_cutoff,
        )
        for node_idx, sample_idx in enumerate(sample_idxs)
    )

    return results
