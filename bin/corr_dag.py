from pan_dag import *

from scipy.special import comb

def corr_worker(X_dag, X_positive, corr_method, i, j):
    """
    Calculate the correlation between the i-th and j-th genes in
    X_dag using only the cells that have positive expression in
    both genes.
    """
    both_positive = np.logical_and(X_positive[:, i],
                                   X_positive[:, j])

    if corr_method == 'pearson':
        corr, corr_score = pearsonr(
            X_dag[both_positive, i], X_dag[both_positive, j]
        )

    elif corr_method == 'spearman':
        corr, corr_score = spearmanr(
            X_dag[both_positive, i], X_dag[both_positive, j],
            nan_policy='omit'
        )

    else:
        raise ValueError('Invalid correlation method {}'
                         .format(self.corr_method))

    if not np.isfinite(corr) or not np.isfinite(corr_score) or \
       abs(corr) == 1. :
        corr = 0.
        corr_score = 1.

    return i, j, corr, corr_score

class CorrelationDAG(PanDAG):
    def __init__(
            self,
            cluster_method='agg_ward',
            corr_method='pearson',
            min_leaves=100,
            sketch_size='auto',
            sketch_method='auto',
            reduce_dim=None,
            permute=False,
            n_jobs=1,
            verbose=False,
    ):
        """
        Initializes correlation DAG object.
        """
        super(CorrelationDAG, self).__init__(
            cluster_method, sketch_size, sketch_method,
            reduce_dim, verbose
        )

        self.corr_method = corr_method
        self.permute = permute
        self.min_leaves = min_leaves
        self.n_jobs = n_jobs
        self.null_scores = None
        self.real_scores = None
        self.verbose = verbose
        self.features = None

        # Items that need to be populated in self.fill_correlations().
        self.correlations = None
        self.corr_scores = None

    def fill_correlations(self, X, permute_step=False):
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

        triu_idx = np.triu_indices(X.shape[1], 1)
        tril_idx = np.tril_indices(X.shape[1], 0)

        if self.permute:
            dist_scores = []
            dist_sample_size = comb(n_features, 2) * len(self.nodes)
            dist_sample_p = 1e7 / dist_sample_size
            if dist_sample_p < 1. and self.verbose:
                print('Downsampling to {}% for distribution estimate'
                      .format(dist_sample_p * 100.))
                sys.stdout.flush()

        for node_idx, node in enumerate(self.nodes):

            if self.verbose:
                print('Filling node {} out of {}'
                      .format(node_idx, len(self.nodes)))
                sys.stdout.flush()

            if node.n_leaves < self.min_leaves:
                node.correlations = None
                node.corr_scores = None
                continue

            X_dag = X[node.sample_idx].toarray()
            X_positive = X_dag != 0
            X_binary = X_dag * 1

            if permute_step:
                for cell_idx in range(X_dag.shape[0]):
                    perm_idx = np.random.permutation(n_features)
                    X_dag[cell_idx] = X_dag[cell_idx][perm_idx]


            # Calculate correlation of positive cells.

            with warnings.catch_warnings():
                warnings.simplefilter('ignore')

                corr, corr_scores = spearmanr(
                    X_dag, nan_policy='omit'
                )

                corr_scores[np.isnan(corr)] = 1.
                corr_scores[np.isnan(corr_scores)] = 1.
                corr_scores[tril_idx] = 1.

                abs_corr = np.abs(corr)
                corr_scores[abs_corr == 1.] = 1.
                corr_scores[abs_corr < 0.25] = 1.
                del abs_corr

                cutoff = 0.01 / (len(tril_idx) * len(self.nodes))

                results = Parallel(n_jobs=self.n_jobs,
                                   backend='loky') (
                    delayed(corr_worker)(
                        X_dag, X_positive, self.corr_method, i, j
                    )
                    for i in range(n_features)
                    for j in range(n_features)
                    if corr_scores[i, j] < cutoff
                )

            for i, j, corr_ij, score_ij in results:
                corr[i, j] = corr_ij
                corr_scores[i, j] = score_ij

            if self.permute:
                # Add significance scores to the distribution.
                dist_samples = corr_scores[triu_idx].flatten()
                if dist_sample_p < 1.:
                    dist_sample_idx = np.random.choice(
                        len(dist_samples),
                        size=int(dist_sample_p * len(dist_samples)),
                        replace=False,
                    )
                    dist_samples = dist_samples[dist_sample_idx]
                dist_scores.append(dist_samples)

            if not permute_step:
                node.correlations = corr
                node.corr_scores = corr_scores

        if self.permute:
            if permute_step:
                self.null_scores = np.concatenate(dist_scores)
            else:
                self.real_scores = np.concatenate(dist_scores)

    def sig_bsearch(self, n_features):
        FLOAT_MIN = np.nextafter(0, 1)
        low_cutoff = FLOAT_MIN
        high_cutoff = 1
        cutoff = np.exp(np.log(FLOAT_MIN) / 2)

        n_iter = 0
        max_iter = 100000
        while True:
            n_real = float(np.sum(self.real_scores < cutoff))
            n_fake = float(np.sum(self.null_scores < cutoff))

            if n_real + n_fake == 0:
                pct_fake = 0
            else:
                pct_fake = n_fake / (n_real + n_fake)

            if pct_fake < 0.05:
                low_cutoff = cutoff

            elif pct_fake > 0.05:
                high_cutoff = cutoff

            else:
                return cutoff

            cutoff = (high_cutoff + low_cutoff) / 2.

            if low_cutoff >= high_cutoff:
                return cutoff

            n_iter += 1
            if n_iter >= max_iter:
                break

        if n_iter >= max_iter:
            warnings.warn('Exceeded {} iterations in FDR binary search'
                          .format(max_iter),
                          RuntimeWarning)
        return cutoff

    def significant(self, n_features):
        if self.permute:
            if self.null_scores is None:
                raise NotImplementedError(
                    'Need to perform permutation run before calling this'
                    ' method.'
                )
            cutoff = self.sig_bsearch(n_features)

        else:
            cutoff = 0.01 / (comb(n_features, 2) * len(self.nodes))

        if self.verbose:
            n_real = float(np.sum(self.real_scores < cutoff))
            n_fake = float(np.sum(self.null_scores < cutoff))

            if n_real + n_fake == 0:
                pct_fake = 0
            else:
                pct_fake = n_fake / (n_real + n_fake)

            print('Using {} as significance score cutoff, {}% FDR'
                  .format(cutoff, pct_fake * 100))
            sys.stdout.flush()

        for node_idx, node in enumerate(self.nodes):
            if node.corr_scores is not None:
                not_sig = node.corr_scores >= cutoff
                node.corr_scores[not_sig] = 1.

    def stack_correlations(self):
        """
        Stack the correlation matrices across all nodes in DAG.

        Returns
        -------
        stacked: numpy.ndarray
            A `(n_feature, n_feature, n_node)` array with the matrices
            stacked along the third dimension.
        """
        if self.correlations is None:
            raise NotImplementedError('Need to call fit() before calling this'
                                      ' method.')

        corr_list = []
        for i, node in enumerate(self.nodes):
            if node.n_leaves < self.min_leaves:
                continue
            corr_list.append(node.correlations)

        return np.stack(corr_list, axis=2)

    def stack_corr_scores(self):
        """
        Stack the significance score matrices across all nodes in DAG.

        Returns
        -------
        stacked: numpy.ndarray
            A `(n_feature, n_feature, n_node)` array with the matrices
            stacked along the third dimension.
        """
        if self.correlations is None:
            raise NotImplementedError('Need to call fit() before calling this'
                                      ' method.')

        score_list = []
        for i, node in enumerate(self.nodes):
            if node.corr_scores is not None:
                score_list.append(node.corr_scores)

        return np.stack(score_list, axis=2)

    def collapse_correlations(self):
        """
        Take strongest correlations across all nodes in DAG.

        Returns
        -------
        strongest: numpy.ndarray
            An upper triangular matrix with the stongest correlation values.
        """
        corr = self.stack_correlations()
        strongest = corr.max(axis=2)
        min_corr = corr.min(axis=2)
        abs_corr = np.abs(corr).max(axis=2)
        min_strongest = abs_corr == np.abs(min_corr)
        strongest[min_strongest] = min_corr[min_strongest]
        return strongest

    def collapse_corr_scores(self):
        """
        Take lowest significance score across all nodes in DAG.

        Returns
        -------
        most_sig: numpy.ndarray
            An upper triangular matrix with the log10 of the significance
            scores and zeros on the diagonal.
        """
        corr_scores = self.stack_corr_scores()
        most_sig = corr_scores.min(axis=2)

        FLOAT_MIN = np.nextafter(0, 1)
        positive_idx = most_sig >= FLOAT_MIN
        most_sig[most_sig < FLOAT_MIN] = np.log10(FLOAT_MIN)
        most_sig[positive_idx] = np.log10(most_sig[positive_idx])

        np.fill_diagonal(most_sig, 0)

        return most_sig

    def fit(self, X, y=None, features=None):
        """
        Constructs DAG according to `self.cluster_method` and populates
        the DAG with correlation matrices.

        Parameters
        ----------
        X: `numpy.ndarray` or `scipy.sparse.csr_matrix`
            Matrix with rows corresponding to all of the samples that
            define the DAG and columns corresponding to features that
            define the correlation matrices.
        y
            Ignored
        features: `numpy.ndarray` of `str`
            A list of strings with feature labels.
        """
        print(X.shape)
        super(CorrelationDAG, self).fit(X, y, features)

        if self.permute:
            if self.verbose:
                print('Calculating null distribution...')
                sys.stdout.flush()

            self.fill_correlations(X, permute_step=True)

        if self.verbose:
            print('Computing correlations and scoring significance...')
            sys.stdout.flush()

        self.fill_correlations(X)

        self.significant(X.shape[1])

        return self
