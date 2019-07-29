from joblib import Parallel, delayed
import scipy
import scipy.sparse as ss
from sklearn.preprocessing import normalize
import numpy as np
from utils import tprint

from nls import nls_lbfgs_b

def obj_worker(Xi, Ci, D, beta):
    tonorm = Xi.toarray() - D.dot(np.multiply(Ci[:, None], D.T))
    reg = beta * Ci.sum()
    return (0.5 * (np.linalg.norm(tonorm) ** 2)) + reg
        
class CovarianceDictionary(object):

    """
    Learns a dictionary of covariance matrices

    Parameters
    ----------
    n_components : int, optional, default = 2
        Number of dictionary elements

    method : str in {'frob-sc'}, optional, default = 'frob-sc'
        Specifies which optimization algorithm to use. Frobenius norm sparse
        coding from Sra, & Cherian (2011) supported.
 
    init : str in {'eigen', 'rand'}, optional, default = 'eigen'
        Specifies how to initialize the dictionary and weights. 'eigen'
        initialized the data using the eigendecomposition of the average
        covariance matrix.
        'rand' initializes to random linear combinations of input.

    max_iter : int, optional, default = 1000
        Maximum number of iterations for main optimization step.

    tol : float, optional, default = 1e-3
        Stopping condition on raw objective value.

    step_size : float, optional, default = 1e-1
        Step size for stochastic gradient descent on dictionary value.

    minibatch_size : int, optional, default = 100
        Size of minibatches for SGD.

    n_jobs : int, optional, default = None
        Number of parallel jobs to use.

    nls_max_iter : int, optional, default = 2000
        Maximum number of iterations for the non-negative least-squares (NLS)
        subproblem.

    nls_beta : float > 0 and < 1, optional, default = 0.2
        L1 sparsity weight on NLS subproblem.

    nls_tol : float, optional, default = 1e-5
        Stopping condition on NLS subproblem.

    verbose : boolean, optional, default = False
        Whether to print algorithm progress.


    Attributes
    ----------
    dictionary_: array of size (n_components, n_features)
        Dictionary of rank 1 SPD matrices defined by outer product of each row.

    References
    ----------
    Sra, S. & Cherian, A. (2012). "Generalized dictionary learning for symmetric
        positive definite matrices with application to nearest neighbor retrieval."
    Wild, Curry, & Dougherty. (2004). "Improving non-negative matrix factorizations through
        structured initialization"

    """

    def __init__(
            self,
            n_components=2,
            method='frob-sc',
            init='eigen',
            max_iter=1000,
            tol=1e-3,
            step_size=1e-1,
            momentum=0.9,
            minibatch_size=100,
            n_jobs=None,
            nls_max_iter=2000,
            nls_beta=0.2,
            nls_tol=1e-5,
            random_state=None,
            verbose=False,
    ):

        possible = ('frob-sc', 'riemann-sc')
        if method not in possible:
            raise ValueError(
                'Invalid method: got %r instead of one of %r' %
                (method, possible)
            )
        self.method = method

        possible = ('eigen', 'rand')
        if init not in possible:
            raise ValueError(
                'Invalid initialization: got %r instead of one of %r' %
                (init, possible)
            )
        self.init = init

        self.max_iter = max_iter
        self.tol = tol
        self.momentum = momentum
        self.step_size = step_size
        self.minibatch_size = minibatch_size

        self.n_components = n_components
        self.nls_max_iter = nls_max_iter
        self.nls_beta = nls_beta
        self.nls_tol = nls_tol
        self.verbose = verbose
        self.dictionary_ = None
        self.random_state = random_state
        self.n_jobs = 1 if n_jobs is None else n_jobs


    def _initialize(self, X):
        n_samples = len(X)
        n_features = X[0].shape[0]

        if self.init == 'eigen':

            Xavg = ss.csr_matrix((n_features, n_features))
            for i in range(n_samples):
                Xavg += (1. / n_samples) * X[i]

            d, V = ss.linalg.eigsh(
                Xavg, k=min(self.n_components, n_features)
            )
            D_init = np.zeros((n_features, n_components))
            D_init[:, :self.n_components] = V * d

        elif self.init == 'rand':

            # Initialize modules to random linear combinations
            # of input covariances.
            D_init = np.random.uniform(0, 1, (n_features, n_components))

        D_init = normalize(D_init, norm='l2', axis=0)

        C_init = self._nls_subproblem(X, D_init)

        return C_init, D_init

    def _obj(self, X, C, D):
        results = Parallel(n_jobs=self.n_jobs, backend='multiprocessing')(
            delayed(obj_worker)(X[i], C[i], D, self.nls_beta)
            for i in range(len(X))
        )        
        return sum(results)

    def _nls_subproblem(self, X, D, C_init=None, minibatch=None):
        n_samples = len(X)
        n_features = X[0].shape[0]

        minibatch = range(n_samples) if minibatch is None else minibatch

        results = Parallel(n_jobs=self.n_jobs, backend='multiprocessing')(
            delayed(nls_lbfgs_b)(
                X[i], D, C_init=None if C_init is None else C_init[i],
                l1_reg=self.nls_beta, max_iter=self.nls_max_iter,
                tol=self.nls_tol,
            )
            for i in minibatch
        )
        C_batch = np.vstack([ result[0] for result in results ])

        #C_batch = normalize(C_batch, norm='l1', axis=1)

        return C_batch

    def _gradient_subproblem(self, X, C, D, velocity, minibatch=None):
        n_samples = len(X)
        n_features = X[0].shape[0]
        n_minibatch = C.shape[0]

        minibatch = range(n_samples) if minibatch is None else minibatch

        gradient = np.zeros(D.shape)
        for i in minibatch:
            gradient += (
                D.dot(np.multiply(C[i][:, None], D.T)) - X[i].toarray()
            ).dot(D * C[i])
        gradient *= 2

        velocity = (self.momentum * velocity) - (self.step_size * gradient)
        D = D + velocity

        #nrm2, = scipy.linalg.get_blas_funcs(('nrm2',), (D,))
        #for j in range(D.shape[1]):
        #    D[:, j] /= max(nrm2(D[:, j]), 1)

        return D, velocity

    def _frob_sc(self, X, C_init=None, D_init=None):
        n_samples = len(X)
        n_features = X[0].shape[0]
        
        if C_init is None or D_init is None:
            C, D = self._initialize(X)
        else:
            C, D = C_init, D_init

        unsampled = set(range(n_samples))
        if self.minibatch_size > n_samples:
            if self.verbose:
                tprint('Setting minibatch size to {}'.format(n_samples))
            self.minibatch_size = n_samples

        velocity = np.zeros(D.shape)

        cost = self._obj(X, C, D)

        for it in range(self.max_iter):

            try:
                minibatch = [
                    unsampled.pop() for mb_idx in range(self.minibatch_size)
                ]
            except KeyError:
                unsampled = set(range(n_samples))
                minibatch += [
                    unsampled.pop() for mb_idx in range(
                        self.minibatch_size - len(minibatch)
                    )
                ]

            # Fit coefficients to current dictionary.
            C[minibatch] = self._nls_subproblem(
                X, D, C_init=C, minibatch=minibatch
            )
            if self.verbose > 2:
                tprint(self._obj(X, C, D))
            
            # Optimize D with SGD.
            D, velocity = self._gradient_subproblem(
                X, C, D, velocity, minibatch=minibatch
            )
            if self.verbose > 2:
                tprint(self._obj(X, C, D))            

            curr_cost = self._obj(X, C, D)
            delta = abs(curr_cost - cost) / cost
            cost = curr_cost

            if self.verbose > 1:
                tprint('Iteration {}, cost: {}, delta: {}'.format(it, cost, delta))

            if delta < self.tol:
                if self.verbose:
                    tprint('Converged at iteration {}'.format(it))
                break

        if it == self.max_iter and self.verbose:
            tprint('Main optimization loop did not converge')

        C = self._nls_subproblem(X, D, C_init=C)

        return C, D


    def fit_transform(self, X, y=None):
        C, D = self._initialize(X)

        if self.method == 'frob-sc':
            C, D = self._frob_sc(X, C, D)

        self.dictionary_ = D.T

        return C


    def fit(self, X, y=None):
        self.fit_transform(X)
        return self


if __name__ == '__main__':
    n_samples = 10
    n_features = 100
    n_components = 10

    np.random.seed(69)

    X = []
    Xi_prev = None
    for i in range(n_samples):
        Xi = np.random.rand(n_features)
        #Xi /= max(np.linalg.norm(Xi), 1)
        Xi = np.outer(Xi, Xi)
        if Xi_prev is None:
            Xi_prev = Xi
        elif i == n_samples - 1:
            Xi += Xi_prev
            Xi += X[-1]
        else:
            Xi += Xi_prev
        X.append(ss.csr_matrix(Xi))

    cd = CovarianceDictionary(
        n_components=n_components,
        method='frob-sc',
        init='rand',
        max_iter=10000,
        tol=5e-5,
        step_size=1e-2,
        momentum=0.,
        minibatch_size=100,
        n_jobs=10,
        nls_max_iter=2000,
        nls_beta=0.2,
        nls_tol=1e-5,
        random_state=None,
        verbose=2,
    )

    weights = cd.fit_transform(X)

    print(weights)

    for i in range(n_samples):
        D = cd.dictionary_.T
        tprint('{}'.format(i))
        tprint(np.linalg.norm(
            np.abs(X[i].toarray() -
                   D.dot(np.multiply(weights[i][:, None], D.T)))
        ) / np.linalg.norm(X[i].toarray()))
