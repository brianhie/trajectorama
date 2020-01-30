import numpy as np
import os
import random
from sklearn.preprocessing import normalize

from .pan_corr import PanCorrelation
from .pan_dag import PanDAG
from .utils import *

def transform(
        X,
        studies,
        X_dimred=None,
        log_transform=False,
        corr_cutoff=0.7,
        corr_method='spearman',
        dag_method='louvain',
        min_cluster_cells=500,
        seed=None,
        n_jobs=1,
        verbose=2,
):
    # Error checking.

    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)

    if X.shape[0] != len(studies):
        raise ValueError('Number of samples in X ({}) does not match '
                         'length of studies ({})'
                         .format(X.shape[0], len(studies)))
    if X_dimred is not None and X_dimred.shape[0] != X.shape[0]:
        raise ValueError('Number of samples in X ({}) does not match '
                         'number of samples in X_dimred ({})'
                         .format(X.shape[0], X_dimred.shape[0]))

    # Preprocessing.

    uniq_studies = sorted(set(studies))
    studies = np.array(studies)
    study_idxs = [
        np.where(studies == study)[0] for study in uniq_studies
    ]

    if log_transform:
        X = X.log1p()

    # Perform panresolution clustering of studies separately.

    if verbose:
        tprint('Performing panresolution clustering...')

    cds = [
        PanDAG(
            dag_method=dag_method,
            verbose=verbose,
        ).fit(X_dimred[study_idx])
        for study_idx in study_idxs
    ]

    # Featurize each cluster by coexpression.

    if verbose:
        tprint('Featurizing clusters by coexpression...')

    ct = PanCorrelation(
        min_leaves=min_cluster_cells,
        dag_method=dag_method,
        corr_method=corr_method,
        corr_cutoff=corr_cutoff,
        n_jobs=n_jobs,
        verbose=verbose,
    )

    for study_idx, cd in zip(study_idxs, cds):
        for node in cd.nodes:
            node.sample_idx = study_idx[node.sample_idx]
            ct.nodes.append(node)

    ct.sample_idx = list(range(X.shape[0]))
    ct.n_leaves = X.shape[0]
    ct.fill_correlations(X)

    if verbose:
        tprint('Finished with Trajectorama transformation.')

    Xs_coexpr = [
        node.correlations for node in ct.nodes
    ]
    sample_idxs = [
        node.sample_idx for node in ct.nodes
    ]

    return Xs_coexpr, sample_idxs
