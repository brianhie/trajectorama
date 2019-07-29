from pan_dag import *

from utils import *

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

    corr, corr_scores = spearmanr(
        X_dag, nan_policy='omit'
    )

    corr[np.isnan(corr)] = 0

    return corr

def corr_worker(X_dag, verbose, node_idx, n_nodes, corr_method,
                cutoff, srp):
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

    if verbose:
        tprint('Filled node {} out of {} with {} samples and '
               '{} nonzero correlations'
               .format(node_idx + 1, n_nodes, X_dag.shape[0],
                       np.count_nonzero(corr)))

    if srp is None:
        return csr_matrix(corr)
    else:
        triu_idx = np.triu_indices(n_features)
        try:
            return srp.transform(corr[triu_idx].reshape(1, -1))[0]
        except ValueError:
            return corr[triu_idx]

class PanCorrelation(PanDAG):
    def __init__(
            self,
            n_components='auto',
            dag_method='louvain',
            corr_method='spearman',
            corr_cutoff=0.7,
            reassemble_method='louvain',
            reassemble_K=15,
            dictionary_learning=False,
            min_leaves=100,
            max_leaves=1e10,
            min_modules=5,
            sketch_size='auto',
            sketch_method='auto',
            reduce_dim=None,
            random_projection=False,
            psd=False,
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

        self.n_components = n_components
        self.corr_method = corr_method
        self.corr_cutoff = corr_cutoff
        self.reassemble_method = reassemble_method
        self.reassemble_K = reassemble_K
        self.dictionary_learning = dictionary_learning
        self.random_projection = random_projection
        self.psd = psd
        self.min_leaves = min_leaves
        self.max_leaves = max_leaves
        self.min_modules = min_modules
        self.n_jobs = n_jobs
        self.verbose = verbose

        # Items that need to be populated in self.fill_correlations().
        self.correlations = None
        self.correlations_proj = None

        # Items that need to be populated in self.find_modules().
        self.modules = None

        self.dictionary_ = None
        self.modules_ = None

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

        n_jobs = max(self.n_jobs // 4, 1)

        n_nodes = len([ node for node in self.nodes
                        if self.min_leaves <= node.n_leaves <= self.max_leaves ])

        if self.verbose:
            tprint('Found {} nodes'.format(n_nodes))

        if self.random_projection:
            from sklearn.random_projection import SparseRandomProjection
            self.srp = SparseRandomProjection(
                eps=0.1, random_state=69
            ).fit(csr_matrix((n_nodes, n_correlations)))
        else:
            self.srp = None

        results = Parallel(n_jobs=n_jobs) (#, backend='multiprocessing') (
            delayed(corr_worker)(
                X[node.sample_idx].toarray(), self.verbose, node_idx,
                len(self.nodes), self.corr_method, self.corr_cutoff, self.srp
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
