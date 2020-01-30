from scipy.special import comb
import warnings

from .pan_dag import *

def pearson_multi(X_dag):
    from scipy.special import betainc
    from sklearn.covariance import EmpiricalCovariance
    from sklearn.preprocessing import StandardScaler

    X_dag = np.log1p(X_dag)

    X_dag = StandardScaler().fit_transform(X_dag)

    corr = EmpiricalCovariance(
        store_precision=False,
        assume_centered=True).fit(X_dag).covariance_
    corr[np.isnan(corr)] = 0
    corr[corr > 1.] = 1.
    corr[corr < -1.] = -1.

    return corr

def spearman_multi(X_dag):
    from scipy.stats import spearmanr

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        corr, corr_scores = spearmanr(
            X_dag, nan_policy='omit'
        )

    corr[np.isnan(corr)] = 0
    corr[corr > 1.] = 1.
    corr[corr < -1.] = -1.

    return corr

def corr_worker(X_dag, verbose, node_idx, n_nodes, corr_method,
                cutoff):
    n_samples, n_features = X_dag.shape

    # Calculate correlation of positive cells.

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

        if corr_method == 'pearson':
            corr = pearson_multi(X_dag)

        elif corr_method == 'spearman':
            corr = spearman_multi(X_dag)

        else:
            raise ValueError('Invalid correlation method {}'.
                             format(corr_method))

    corr[np.isnan(corr)] = 0.
    corr[np.abs(corr) < cutoff] = 0

    if verbose > 1:
        tprint('Filled node {} out of {} with {} samples and '
               '{} nonzero correlations'
               .format(node_idx + 1, n_nodes, X_dag.shape[0],
                       np.count_nonzero(corr)))

class PanCorrelation(PanDAG):
    def __init__(
            self,
            dag_method='louvain',
            corr_method='spearman',
            corr_cutoff=0.7,
            min_leaves=100,
            max_leaves=1e10,
            sketch_size='auto',
            sketch_method='auto',
            reduce_dim=None,
            n_jobs=1,
            verbose=False,
    ):
        """
        Initializes correlation DAG object.
        """
        super(PanCorrelation, self).__init__(
            dag_method, sketch_size, sketch_method,
            reduce_dim, verbose
        )

        self.corr_method = corr_method
        self.corr_cutoff = corr_cutoff
        self.min_leaves = min_leaves
        self.max_leaves = max_leaves
        self.n_jobs = n_jobs
        self.verbose = verbose

        # Items that need to be populated in self.fill_correlations().
        self.correlations = None
        self.correlations_proj = None

    def fill_correlations(self, X):
        """
        Stack the correlation matrices across all nodes in DAG.

        Parameters
        ----------
        X: `numpy.ndarray` or `scipy.sparse.csr_matrix`
            Matrix with rows corresponding to all of the samples that
            define the DAG and columns corresponding to features that
            define the correlation matrices.
        """
        n_samples, n_features = X.shape
        n_correlations = int(comb(n_features, 2) + n_features)

        n_nodes = len([ node for node in self.nodes
                        if self.min_leaves <= node.n_leaves <= self.max_leaves ])

        if self.verbose > 1:
            tprint('Found {} nodes'.format(n_nodes))

        results = Parallel(n_jobs=self.n_jobs) (
            delayed(corr_worker)(
                X[node.sample_idx].toarray(), self.verbose, node_idx,
                len(self.nodes), self.corr_method, self.corr_cutoff,
            )
            for node_idx, node in enumerate(self.nodes)
            if self.min_leaves <= node.n_leaves <= self.max_leaves
        )

        result_idx = 0
        for node_idx, node in enumerate(self.nodes):
            if self.min_leaves <= node.n_leaves <= self.max_leaves:
                node.correlations = results[result_idx]
                result_idx += 1
            else:
                node.correlations = None
