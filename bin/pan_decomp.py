from pan_dag import *

import scipy
from scipy.sparse import hstack
from sklearn.decomposition.nmf import _update_coordinate_descent

class DecomposeDAG(PanDAG):
    def __init__(
            self,
            dag_method='louvain',
            decomp_method='nmf',
            n_components=100,
            sketch_size='auto',
            sketch_method='auto',
            reduce_dim=None,
            n_jobs=1,
            verbose=False,
    ):
        """
        Initializes correlation DAG object.
        """
        super(DecomposeDAG, self).__init__(
            dag_method, sketch_size, sketch_method,
            reduce_dim, verbose
        )

        self.decomp_method = decomp_method
        self.n_components = n_components
        self.n_jobs = n_jobs

        self.cluster_components = None

    def multiresolution_stack(self, X):
        """
        Duplicates dataset horizontally, each corresponding to a
        subcluster of cells.

        Parameters
        ----------
        X: `scipy.sparse.csr_matrix`
            Matrix with rows corresponding to all of the samples and
            columns corresponding to features that define the
            correlation matrices.

        Returns
        -------
        stacked: `scipy.sparse.csr_matrix`
            Stacked matrix.
        """
        n_samples, n_features = X.shape

        sample_set = set(range(n_samples))

        stack = []
        for node_idx, node in enumerate(self.nodes):
            X_new = X.copy()
            other_samples = list(sample_set - set(node.sample_idx))
            if issubclass(type(X_new), np.ndarray):
                X_new[other_samples] = 0
            elif issubclass(type(X_new), scipy.sparse.csr.csr_matrix):
                for r in other_samples:
                    X_new.data[X_new.indptr[r]:X_new.indptr[r+1]] = 0
                X_new.eliminate_zeros()
            else:
                sys.stderr.write('ERROR: Dataset must be numpy array or '
                                 'scipy.sparse.csr_matrix, received type '
                                 '{}.\n'.format(type(ds)))
            
            stack.append(X_new)

        return hstack(stack)

    def fit(self, X, y=None, features=None):
        """
        Constructs DAG according to `self.dag_method` and learns
        coexpression modules across multiple resolutions.        

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
        super(DecomposeDAG, self).fit(X, y, features)

        n_samples, n_features = X.shape

        if self.verbose:
            print('Stacking...')
            sys.stdout.flush()
        X_multi = self.multiresolution_stack(X)

        if self.verbose:
            print('Decomposing...')
            sys.stdout.flush()

        if self.decomp_method == 'nmf':
            #from sklearn.decomposition import NMF
            from nmf import NMF
            decomp = NMF(
                n_components=self.n_components,
                init=None,
                solver='cd',
                beta_loss='frobenius',
                alpha=1e-3,
                l1_ratio=1,
                random_state=69,
                tol=1e-2,
                verbose=self.verbose,
            ).fit(X_multi)
            components = decomp.components_

        elif self.decomp_method == 'lda':
            from sklearn.decomposition import (
                LatentDirichletAllocation as LDA
            )
            decomp = LDA(
                n_components=self.n_components,
                learning_method='online',
                max_iter=20,
                mean_change_tol=1e-2,
                n_jobs=self.n_jobs,
                random_state=69,
                verbose=self.verbose,
            ).fit(X_multi)
            components = decomp.components_

        elif self.decomp_method == 'hdp':
            from bnp.online_hdp import (
                HierarchicalDirichletProcess as HDP
            )
            hdp = HDP(
                n_topic_truncate=self.n_components,
                n_doc_truncate=10000,
                learning_method='online',
                n_jobs=self.n_jobs,
                random_state=69,
                verbose=self.verbose,
            ).fit(X_multi)
            components = hdp.lambda_

        else:
            raise ValueError('Invalid decomposition method {}'.
                             format(self.decomp_method))

        n_components = components.shape[0]
        self.cluster_components = np.reshape(
            components,
            (n_components, n_features, len(self.nodes))
        )

        cc = np.sum(self.cluster_components, axis=1)
        cc /= cc.max()
        assert(cc.shape == (n_components, len(self.nodes)))

        for node_idx, node in enumerate(self.nodes):
            node.viz_value = list(cc[:, node_idx])

        return self

    def visualize(self, namespace='', comp_idx=-1, reduce_type='max'):

        # Breadth-first traversal.

        seen_cells = set()
        cell_order = []

        from queue import SimpleQueue
        queue = SimpleQueue()
        queue.put((self, 0))

        n_levels = 1
        levels = {}
        level_sizes = {}

        while not queue.empty():
            node, depth = queue.get()

            if depth + 1 > n_levels:
                n_levels = depth + 1

            if depth not in levels:
                levels[depth] = {}
            for si in node.sample_idx:
                if si not in levels[depth]:
                    levels[depth][si] = []
                levels[depth][si].append(node.viz_value[comp_idx])
                    
            if depth not in level_sizes:
                level_sizes[depth] = 0
            level_sizes[depth] += 1.

            if len(node.children) == 0:
                [
                    cell_order.append(si)
                    for si in node.sample_idx
                    if si not in seen_cells
                ]
                seen_cells.update(node.sample_idx)

            else:
                for child in node.children:
                    queue.put((child, depth + 1))

        # Draw the pyramid.
        from pyx import canvas, path, style, deco, color

        c = canvas.canvas()

        unit = 20.
        height = unit * np.sqrt(3) / 2.

        level_height = height / n_levels
        curr_height = height

        grad = color.lineargradient_rgb(color.rgb.white,
                                        color.rgb.red)

        for level in range(n_levels):
            top_width = height - curr_height
            bottom_width = height - curr_height + level_height

            n_in_level = len(levels[level])
            assert(n_in_level == len(self.sample_idx) == len(cell_order))

            x_top = (unit / 2.) - (top_width / 2.)
            x_top_curr = x_top
            x_top_inc = top_width / n_in_level

            x_bottom = (unit / 2.) - (bottom_width / 2.)
            x_bottom_curr = x_bottom
            x_bottom_inc = bottom_width / n_in_level

            if reduce_type == 'mean':
                reduce_fn = np.mean
            elif reduce_type == 'median':
                reduce_fn = np.median
            elif reduce_type == 'max':
                reduce_fn = np.max
            elif reduce_type == 'min':
                reduce_fn = np.min
            else:
                raise ValueError('Invalid reduce function {}'
                                 .format(reduce_type))
            
            curr_node = reduce_fn(levels[level][cell_order[0]])

            for sidx in cell_order:

                if reduce_fn(levels[level][sidx]) != curr_node:

                    brick = path.path(
                        path.moveto(x_top, curr_height),
                        path.lineto(x_top_curr, curr_height),
                        path.lineto(x_bottom_curr, curr_height - level_height),
                        path.lineto(x_bottom, curr_height - level_height),
                        path.closepath(),
                    )
                    c.stroke(
                        brick,
                        [ style.linewidth(1e-3),
                          deco.filled([grad.getcolor(curr_node)]) ]
                    )

                    curr_node = reduce_fn(levels[level][sidx])
                    x_top = x_top_curr
                    x_bottom = x_bottom_curr

                x_top_curr += x_top_inc
                x_bottom_curr += x_bottom_inc

            brick = path.path(
                path.moveto(x_top, curr_height),
                path.lineto(x_top_curr, curr_height),
                path.lineto(x_bottom_curr, curr_height - level_height),
                path.lineto(x_bottom, curr_height - level_height),
                path.closepath(),
            )
            c.stroke(
                brick,
                [ style.linewidth(1e-3),
                  deco.filled([grad.getcolor(curr_node)]) ]
            )

            curr_height -= level_height

        c.writeSVGfile('pyramid{}_{}_{}_{}_{}'
                       .format(namespace, self.dag_method,
                               self.decomp_method, reduce_type, comp_idx))
