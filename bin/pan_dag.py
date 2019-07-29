from anndata import AnnData
from joblib import Parallel, delayed
import numpy as np
import scanpy as sc
from sklearn.preprocessing import normalize
import os
import sys
import uuid
import warnings

from _louvain import louvain
from utils import *

def louvain_worker(X, resolution):
    log_uuid = str(uuid.uuid4())
    tmp_log_fname = 'target/tmp/{}_louvain.log'.format(log_uuid)

    # Run Louvain and save results to log file.
    adata = AnnData(X=X)
    sc.pp.neighbors(adata, use_rep='X')
    louvain(adata, resolution=resolution, key_added='louvain',
            log_fname=tmp_log_fname)

    return tmp_log_fname


class PanDAG(object):
    def __init__(
            self,
            dag_method='agg_ward',
            sketch_size='auto',
            sketch_method='auto',
            reduce_dim=None,
            verbose=False,
    ):
        """
        Initializes pan clustering DAG object.
        """
        self.dag_method = dag_method
        self.sketch_size = sketch_size
        self.sketch_method = sketch_method
        self.sketch_neighbors = None
        self.reduce_dim = reduce_dim
        self.verbose = verbose
        self.features = None

        # Items that need to be populated in self.create_dag*().
        self.children = [] # Pointer to child nodes.
        self.sample_idx = [] # All samples in subdag.
        self.n_leaves = len(self.sample_idx) # Number of samples in subdag.
        self.nodes = [ self ] # Pointer to all nodes in subdag.


    def check_and_sketch(self, X):
        """
        Determines sketch size and method based on dataset size and
        underlying DAG construction method.

        Parameters
        ----------
        X: `numpy.ndarray` or `scipy.sparse.csr_matrix`
            Dataset tot be sketched.

        Returns
        -------
        X_sketch
            Sketched version of dataset `X`.
        """
        n_samples = X.shape[0]

        if self.sketch_size is None:
            return X

        if self.sketch_method not in set([ 'auto', 'geometric', 'uniform' ]):
            raise ValueError('Invalid sketching method {}'
                             .format(self.sketch_method))

        if self.sketch_size == 'auto':
            # Agglomerative clustering does not scale well beyond 5000
            # samples.
            if self.dag_method == 'agg_ward':
                if n_samples > 5000:
                    self.sketch_size = 5000
                    if self.sketch_method == 'auto':
                        self.sketch_method = 'geometric'
                else:
                    return X

            # Louvain clustering can tolerate much larger datasets.
            elif self.dag_method == 'louvain':
                if n_samples > 1000000:
                    self.sketch_size = 1000000
                    if self.sketch_method == 'auto':
                        self.sketch_method = 'uniform'
                else:
                    return X

            # Default to reasonably large geometric sketch.
            else:
                self.sketch_size = 20000
                if self.sketch_method == 'auto':
                    self.sketch_method = 'geometric'

        # If sketch size is provided but not sketch method.
        elif self.sketch_method == 'auto':
            self.sketch_method = 'geometric'

        # Geometric sketching requires data to be PCAed first.
        if self.sketch_method == 'geometric' and self.reduce_dim is None:
            X = reduce_dimensionality(normalize(X), dim_red_k=100)

        return self.sketch(X)

    def sketch(self, X):
        """
        Actually sketches the dataset and saves nearest neighbor mappings
        from sketch elements to sample observations in full dataset in
        the `self.sketch_neighbors` variable.

        Parameters
        ----------
        X: `numpy.ndarray` or `scipy.sparse.csr_matrix`
            Dataset tot be sketched.

        Returns
        -------
        X_sketch
            Sketched version of dataset `X`.
        """
        n_samples = X.shape[0]

        if self.verbose:
            tprint('Sketching...')

        if self.sketch_method == 'geometric':
            from geosketch import gs
            sketch_idx = gs(X, self.sketch_size, replace=False)
        elif self.sketch_method == 'uniform':
            sketch_idx = sorted(np.random.choice(
                n_samples, size=self.sketch_size, replace=False
            ))
        else:
            return X

        X_sketch = X[sketch_idx]

        self.sketch_neighbors = nearest_approx(X, X_sketch)

        return X[sketch_idx]

    def create_dag_agg(self, Z, n_samples):
        """
        Form hierarchical structure among `n_samples` observations
        according to a linkage matrix outputted by a hierarchical
        clustering algorithm.

        Parameters
        ----------
        Z: `numpy.ndarray`
            Linkage matrix outputted by agglomerative clustering.
        n_samples: `int`
            The number of samples in the dataset.
        """
        self.nodes = []

        for i in range(Z.shape[0]):
            if i == Z.shape[0] - 1:
                node = self
            else:
                node = PanDAG()

            # Left side of agglomerative merge.
            if Z[i, 0] < n_samples:
                if self.sketch_neighbors is None:
                    node.sample_idx.append(Z[i, 0])
                else:
                    [ node.sample_idx.append(idx)
                      for idx in self.sketch_neighbors[Z[i, 0]] ]
            else:
                prev_node = self.nodes[Z[i, 0] - n_samples]
                node.children.append(prev_node)
                node.sample_idx += prev_node.sample_idx

            # Right side of agglomerative merge.
            if Z[i, 1] < n_samples:
                if self.sketch_neighbors is None:
                    node.sample_idx.append(Z[i, 1])
                else:
                    [ node.sample_idx.append(idx)
                      for idx in self.sketch_neighbors[Z[i, 1]] ]
            else:
                prev_node = self.nodes[Z[i, 1] - n_samples]
                node.children.append(prev_node)
                node.sample_idx += prev_node.sample_idx

            node.n_leaves = len(node.sample_idx)
            node.features = self.features

            self.nodes.append(node)

        return self

    def create_dag_louvain(self, X):
        """
        Form hierarchical structure among observed samples in `X`
        according to the Louvain clustering algorithm that iteratively
        merges nodes into larger "communities."

        Parameters
        ----------
        X: `numpy.ndarray` or `scipy.sparse.csr_matrix`
            Matrix with rows corresponding to all of the samples that
            define the DAG and columns corresponding to features that
            define the correlation matrices.
        """
        mkdir_p('target/tmp')

        resolutions = [ 1., 0.1, 10. ]
        results = Parallel(n_jobs=3, backend='multiprocessing')(
            delayed(louvain_worker)(X, resolution)
            for resolution in resolutions
        )

        # Integrate multiple resolutions.
        for tmp_log_fname in results:

            nodes = {} # Map from (community, iter) to dag object.
            v_to_node = {} # Map from vertex to list of nodes.
            max_iter = 0
            with open(tmp_log_fname) as f:
                for line in f:
                    fields = line.rstrip().split()
                    sample = int(fields[0])
                    community, iter_ = fields[1], int(fields[2])
                    if iter_ > max_iter:
                        max_iter = iter_

                    node_id = (community, iter_)
                    if not node_id in nodes:
                        nodes[node_id] = PanDAG()

                    if not sample in v_to_node:
                        v_to_node[sample] = []
                    v_to_node[sample].append(node_id)

                    node = nodes[node_id]
                    if self.sketch_neighbors is None:
                        node.sample_idx.append(sample)
                    else:
                        [ node.sample_idx.append(idx)
                          for idx in self.sketch_neighbors[sample] ]

            # No need for log file anymore.
            os.remove(tmp_log_fname)

            # Create root node with pointers to top-level clusters.
            nodes[('0', max_iter + 1)] = self
            if self.sketch_neighbors is None:
                self.sample_idx = list(range(X.shape[0]))
            else:
                self.sample_idx = [
                    idx
                    for sample in self.sketch_neighbors
                    for idx in self.sketch_neighbors[sample]
                ]

            # Fill out DAG edges.
            for node_id in nodes:
                node = nodes[node_id]
                child_ids = set()
                subdag_ids = set()
                for sample in node.sample_idx:
                    if sample not in v_to_node:
                        continue
                    for sample_node_id in v_to_node[sample]:
                        if sample_node_id[1] == node_id[1] - 1:
                            child_ids.add(sample_node_id)
                        if sample_node_id[1] < node_id[1]:
                            subdag_ids.add(sample_node_id)
                for child_id in sorted(child_ids):
                    node.children.append(nodes[child_id])
                for subdag_id in sorted(subdag_ids):
                    node.nodes.append(nodes[subdag_id])
                node.n_leaves = len(node.sample_idx)

        return self

    def fit(self, X, y=None, features=None):
        """
        Constructs DAG according to `self.dag_method`.

        Parameters
        ----------
        X: `numpy.ndarray` or `scipy.sparse.csr_matrix`
            Matrix with rows corresponding to all of the samples that
            define the DAG and columns corresponding to features.
        y
            Ignored
        features: `numpy.ndarray` of `str`
            A list of strings with feature labels.
        """
        if features is None:
            self.features = np.array(range(X.shape[1]))

        if self.reduce_dim is not None:
            if issubclass(type(self.reduce_dim), np.ndarray):
                X_ = self.reduce_dim
            elif isinstance(self.reduce_dim, int):
                X_ = reduce_dimensionality(X, dim_red_k=self.reduce_dim)
            else:
                raise ValueError('`reduce_dim` has invalid type {}'
                                 .format(type(self.reduce_dim)))
        else:
            X_ = X

        X_ = self.check_and_sketch(X_)

        if self.verbose:
            tprint('Constructing DAG...')

        if self.dag_method == 'agg_ward':
            from sklearn.cluster.hierarchical import ward_tree

            ret = ward_tree(X_, n_clusters=None, return_distance=True)
            children, n_components, n_leaves, parent, distances = ret
            assert(n_components == 1)

            self.create_dag_agg(children, X_.shape[0])

        elif self.dag_method == 'louvain':
            self.create_dag_louvain(X_)

        else:
            raise ValueError('Invalid DAG construction method {}'
                             .format(self.dag_method))

        if len(self.sample_idx) != X.shape[0]:
            warnings.warn('Some samples have been orphaned during '
                          'DAG construction, {} orphans detected'
                          .format(X.shape[0] - len(self.sample_idx)),
                          RuntimeWarning)

        return self
