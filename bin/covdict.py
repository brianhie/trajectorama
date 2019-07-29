"""
Class and functions for learning dictionary of covariances matrices
"""

import warnings
import sys
import time
from itertools import permutations
from numpy import (arange, cov, diag, diag_indices, dot, dsplit, dstack, empty, finfo, fmax,
    hstack, identity, logical_or, max, mean, mod, sort, sqrt, where, triu_indices, vstack, zeros)
import numpy as np
from numpy import sum as npsum
from numpy.random import rand, randint, randn
from scipy.special import factorial
from scipy.linalg import eigh, inv, norm, qr, solve
from sklearn.preprocessing import normalize

from correlation_tools import cov_nearest
from fbpca import pca, eigens
from utils import tprint

def plot_element(element, coords, thresh=0.5, size=200, colors=None, diameter_scale=5,
    neg_color='PaleVioletRed', pos_color='YellowGreen', y_range=None, x_range=None, rand_order=False):

    # TODO: If covariance and not correlation element, need to normalize
    # line widths (commented out below).

    from numpy import array, ones, percentile
    from numpy.random import permutation
    import matplotlib.pyplot as plt

    n = element.shape[0]
    if colors is None:
        colors = 'DarkGray'

    # Positively correlated edges.
    edges_pos = where(element > percentile(element[element > 0], 100 * thresh))
    # alpha_pos = element[edges_pos] / max(element[element > 0])
    width_pos = element[edges_pos] * diameter_scale
    # Negatively correlated edges.
    if (element < 0).any():
        edges_neg = where(-element > percentile(-element[element < 0], 100 * thresh))
        # alpha_neg = -element[edges_neg] / max(-element[element < 0])
        width_neg = -element[edges_neg] * diameter_scale

    edges = hstack((edges_pos, edges_neg))
    width = hstack((width_pos, width_neg))
    is_neg = hstack((zeros(width_pos.size), ones(width_neg.size)))

    if rand_order:
        perm = permutation(width.size)
        edges = edges[:, perm]
        width = width[perm]
        is_neg = is_neg[perm]

    for i, edge in enumerate(array(edges).T):
        if is_neg[i]:
            plt.plot(coords[edge, 0], coords[edge, 1], neg_color, linewidth=width[i]);
        else:
            plt.plot(coords[edge, 0], coords[edge, 1], pos_color, linewidth=width[i]);

    plt.scatter(coords[:, 0], coords[:, 1], s=size, linewidth=0, c=colors);
    if y_range is None or x_range is None:
        plt.axis('scaled')
    else:
        plt.axis('scaled')
        plt.ylim(y_range)
        plt.xlim(x_range)
    plt.axis('off')



def norm_concat(x1, x2):

    # Fastest way to compute the norm of two concatenated vectors,
    # compared to:
    # np.linalg.norm(np.vstack((x1, x2)))
    # np.sqrt(np.sum(np.power(x1, 2)) + np.sum(np.power(x2, 2)))

    return sqrt(pow(norm(x1), 2) + pow(norm(x2), 2))



def pack(vec, n=None, half=False):

    # Reforms a vector into the upper triangle
    # of a symmetric matrix.

    if n is None:
        n = npair2n(vec.size)

    packed = zeros((n, n))
    packed[triu_indices(n)] = vec

    if not half:
        symm = packed + packed.T
        symm[diag_indices(n)] = packed[diag_indices(n)]
        return symm

    return packed



def pack_samples(unpacked, n=None, half=False):

    # Packs each column into the upper triangle of
    # a symmetric and returns 3D stack.

    n_pair, n_samp = unpacked.shape
    if n is None:
        n = npair2n(n_pair)
    return dstack([pack(col, n=n, half=half) for col in unpacked.T])



def unpack_samples(packed):

    # Unpacks the upper triangle of each n x n matrix in a 3D stack
    # into a column of the 2D output.

    n, _, n_samp = packed.shape
    return vstack([mat[triu_indices(n)] for mat in packed.T]).T



def proj_weights(W, correlation=False):

    # From Chen, Y. & Ye, X. (2011). Projection onto a simplex.

    if correlation:

        k, n = W.shape
        W_proj = empty((k, n))

        for col_idx in range(n):
            w = sort(W[:, col_idx])
            idx = k - 2

            while(True):
                t_idx = (sum(w[idx + 1 :]) - 1) / (k - idx - 1)
                if t_idx >= w[idx]:
                    W_proj[:, col_idx] = fmax(W[:, col_idx] - t_idx, 0)
                    break
                else:
                    idx = idx - 1
                    if idx < 0:
                        t_idx = (sum(w) - 1) / k
                        W_proj[:, col_idx] = fmax(W[:, col_idx] - t_idx, 0)
                        break
        return W_proj
    else:
        return fmax(W, 0)



def proj_psd(A, rand_on=False, rand_k=100):

    # Projects a symmetric matrix to nearest
    # positive semi-definite matrix.

    assert(A.shape[0] == A.shape[1])

    if rand_on:
        d, U = eigens(A, k=min(rand_k, A.shape[0]), n_iter=2)
        U = U[:, d > 0]
        d = d[d > 0]
    else:
        d, U = eigh(A, lower=False)
        U = U[:, d > 0]
        d = d[d > 0]

        Aproj = dot(dot(U, diag(d)), U.T)

    return Aproj



def proj_corr(A, max_iter=100, tol=1e-6, rand_on=False, rand_k=100):

    return cov_nearest(A, method='clipped', return_corr=True)

    # Projects a symmetric matrix to the nearest correlation
    # matrix (PSD matrix with equality constraints on
    # the diagonal and bound constraints on the off-diagonal)
    # using Dykstra's algorithm, as described in Higham (2002)
    # "Computing the nearest correlation matrix: A problem from finance".

    # How exactly is Dykstra's different from ADMM for two projections?

    # Is there a norm under which the closest correlation matrix to a
    # covariance matrix is its own correlation matrix? Should do projection
    # under this norm.

    n = A.shape[0]
    deltaS = zeros((n, n))
    Y = A
    X = zeros((n, n))

    #triu_idx = triu_indices(n)
    diag_idx = diag_indices(n)

    for n_iter in range(max_iter):

        Xprev = X
        Yprev = Y

        R = Y - deltaS

        # Project onto semidefinite cone.
        X = proj_psd(R, rand_on=rand_on, rand_k=rand_k)

        deltaS = X - R
        Y = X

        # Equality constraints.
        Y[diag_idx] = 1

        diffX = max(np.abs(X - Xprev)) / max(np.abs(X))
        diffY = max(np.abs(Y - Yprev)) / max(np.abs(Y))
        diffXY = max(np.abs(Y - X)) / max(np.abs(Y))

        if max([diffX, diffY, diffXY]) < tol:
            break

    if n_iter == max_iter - 1:
        warnings.warn("Max iterations reached in correlation matrix projection.")

    return Y



def proj_col_psd(A, correlation=False, rand_on=False, rand_k=100,
                 triu_idx=None):

    # Projects every column of a matrix to the upper-triangle
    # of the nearest positive semi-definite or correlation matrix.

    n_pair, n_col = A.shape
    n = npair2n(n_pair)

    Aproj = zeros((n_pair, n_col))
    mat_triu = zeros((n, n))
    if triu_idx is None:
        triu_idx = triu_indices(n)

    if correlation:
        for col_idx in range(n_col):

            # Reconstruct symmetric matrix.
            mat_triu[triu_idx] = A[:, col_idx]
            if rand_on:
                mat_triu = mat_triu + mat_triu.T - np.diag(mat_triu.diagonal())
            Aproj[:, col_idx] = proj_corr(
                mat_triu, rand_on=rand_on, rand_k=rand_k,
            )[triu_idx]
    else:
        for col_idx in range(n_col):

            # Reconstruct symmetric matrix.
            mat_triu[triu_idx] = A[:, col_idx]
            if rand_on:
                mat_triu = mat_triu + mat_triu.T - np.diag(mat_triu.diagonal())
            Aproj[:, col_idx] = proj_psd(
                mat_triu, rand_on=rand_on, rand_k=rand_k,
            )[triu_idx]

    return Aproj



def npair2n(n_pair):

    # Number of nodes from number of upper triangular entries.
    return int((sqrt(1 + 8 * n_pair) - 1) / 2)


def generate_psd(n, is_singular=False):

    # Generates a random positive semidefinite matrix
    # A = UDU^T, where U is a Haar-distributed random
    # orthogonal matrix and D is a diagonal matrix
    # of uniformly distributed positive entries in [0, 1].
    # (See http://epubs.siam.org/doi/pdf/10.1137/0717034.)
    # Distribution over faces (singular PSD) correct?

    # QR decomposition of random basis to get orthogonal matrix
    Q = zeros((n, n))
    while not Q.any() or Q.shape[1] < n:
        Q, _ = qr(randn(n, n)) # mode='reduced'

    if not is_singular:
        sing_vals = rand(n)
    else:
        nnz = randint(n)
        sing_vals = hstack((rand(nnz), zeros(n - nnz)))
    return dot(dot(Q, diag(sing_vals)), Q.T)

def generate_data(n, k, N, is_unpacked=True, is_singular=False):

    # Generates a dictionary of random positive semidefinite matrices
    # and non-negative weights. Default to unpacked since probably
    # going to run fit_transform() and unpack_samples().

    D_unpacked = vstack([generate_psd(n, is_singular)[triu_indices(n)] for _ in range(k)]).T
    W = rand(k, N)
    X = dot(D_unpacked, W)
    if not is_unpacked:
        X = pack_samples(X)
    return X, pack_samples(D_unpacked), W


def eval_necessary(D):

    # Checks necessary conditions for factorization uniqueness.
    pass


def norm_colwise(A):
    return sqrt(npsum(A ** 2, 0))


def eval_similarity(D, Dest):

    # Evaluates the cosine similarity between the true D and the estimated
    # Dest searching across all possible permutations of atoms.
    # normDW = norm_concat(D, W)
    # Technically, we can solve for the scaling constants that minimize
    # the combined error of D_i - c * Dest_i, W_i - (1 / c) * West_i...

    k = D.shape[2]
    D = unpack_samples(D)
    Dest = unpack_samples(Dest)
    D_norm_colwise = norm_colwise(D)
    similarities = empty(factorial(k))

    for i, perm in enumerate(permutations(list(range(k)))):
        Dest_perm = Dest[:, list(perm)]
        similarities[i] = mean(diag(dot(D.T, Dest_perm)) / (D_norm_colwise * norm_colwise(Dest_perm)))
    return max(similarities)



class CovarianceDictionary(object):

    """
    Learns a dictionary of covariance matrices

    Parameters
    ----------
    k : int, optional, default = 2
        Number of dictionary elements

    method : str in {'admm', 'als', 'pgm'}, optional, default = 'admm'
        Specifies which optimization algorithm to use. Alternating least-squares
        ('als'), alternating directions method of multipliers ('admm'), and
        projected gradient method ('pgm') supported

    init : str in {'kmeans', 'rand', 'svd', 'Dinit'}, optional, default = 'kmeans'
        Specifies how to initialize the dictionary and weights. 'k-means'
        clusters the input data to initialize as in Wild, Curry, & Dougherty (2004),
        'rand' initializes to random linear combinations of input, 'svd'
        uses the Singular Value Decomposition (SVD) initialization, and 'Dinit'
        allows the user to provide an initialization of the dictionary.
        Random initialization completely derails ADMM, takes forever to converge.

    max_iter : int, optional, default = None
        Maximum number of iterations. If None, 200 for ALS and 6000 for ADMM

    tol : float, optional, default = 1e-5 for ALS and ADMM, 1e-3 for PGM
        Stopping tolerance on projected gradient norm for ALS and PGM, and objective for ADMM.
        Taking it easy on PGM, poor thing tends to converge at much greater objective values,
        consistent with Lin (2007).

    n_jobs : int, optional, default = None
        Number of parallel jobs to use in k-means initialization step.

    rand : bool, optional, default = False
        Use randomized versions of matrix inverse and eigendecomposition.

    rand_k : int, optional, default = 100
        Number of components to use in randomized decomposition.

    nls_max_iter : int, optional, default = 2000
        Maximum number of iterations for the non-negative least-squares subproblem
        in ALS

    psdls_max_iter : int, optional, default = 2000
        Maximum number of iterations for the positive semidefinite least-squares
        subproblem in ALS

    nls_beta : float > 0 and < 1, optional, default = 0.2
        Step size search parameter for the non-negative least-squares subproblem
        in ALS, as in "Armijo rule along the projection arc" in Bertsekas (1999)
        Larger values mean larger jumps in searching for step size, so
        can speed up convergence but may be less accurate

    psdls_beta : float > 0 and < 1, optional, default = 0.2
        Step size search parameter for the positive-semidefinite least-squares subproblem
        in ALS, as in "Armijo rule along the projection arc" in Bertsekas (1999).
        Larger values mean larger jumps in searching for step size, so
        can speed up convergence but may be less accurate. Empirically
        larger psdls_beta affects accuracy more so than nls_beta

    pgm_beta : float > 0 and < 2, optional, default = 0.5
        Step size search parameter for the projected gradient method, as in
        "Armijo rule along the projection arc" in Bertsekas (1999).

    correlation : boolean, optional, default = False
        Whether to find dictionary of correlation matrices rather
        than covariance matrices. Supported for all algorithms,
        but takes long as chickens for ALS so only use ADMM or PGM

    admm_gamma : float, optional, default = 0.1
        Constant on step size rule for ADMM

    admm_alpha : float, optional, default = 47.75
        Scaling constant on penalty on proximal term ||U - D||_F^2 for ADMM

    verbose : boolean, optional, default = False
        Whether to print algorithm progress (projected gradient norm for
        ALS, objective for ADMM)

    time : boolean, optional, default = False
        Whether to time each iteration

    obj_tol : float, optional, default = None
        Stopping condition on raw objective value. If None, stopping rule is
        instead based on objective decrease for ADMM and projected gradient norms for ALS and PGM.
        Should only be used when true minimum objective value is known

    Attributes
    ----------
    dictionary: array, shape (n, n, k)
        Dictionary of covariance or correlation matrices where each column
        gives the upper triangle of a dictionary element

    objective: array, shape (n_iter)
        Value of objective ||X - DW||_F / ||X||_F at each iteration

    References
    ----------
    Bertsekas. 1976.
    Bertsekas. 1999.
    Calamai & More (1999)
    Lin, C.-J. & More (1999)
    Lin, C.-J. (2007). "Projected gradient methods for non-negative matrix factorization".
        Neural Computation.
    Wild, Curry, & Dougherty. (2004). "Improving non-negative matrix factorizations through
        structured initialization"
    Xu, Y., Yin, W., Wen, Z., & Zhang, Y. (2012). "An alternating direction algorithm for matrix
        completion with nonnegative factors".

    """

    SUFF_DECR = -1
    INSUFF_DECR = -2

    def __init__(
            self, k=2, method='admm', init='svd', max_iter=None, tol=None,
            verbose=False, obj_tol=None, time=False, n_jobs=None,
            rand_on=False, rand_k=100, random_state=None,
            nls_beta=0.2, psdls_beta=0.2, nls_max_iter=2000, psdls_max_iter=2000,
            pgm_beta=0.5, correlation=False, admm_gamma=0.1, admm_alpha=47.75,
    ):

        if method not in ('als', 'admm', 'pgm', 'dr'):
            raise ValueError(
                                'Invalid method: got %r instead of one of %r' %
                                (method, ('als', 'admm', 'pgm', 'dr')))
        self.method = method

        if init not in ('kmeans', 'rand', 'svd', 'Dinit'):
            raise ValueError(
                                'Invalid initialization: got %r instead of one of %r' %
                                (init, ('kmeans', 'rand', 'svd', 'Dinit')))
        self.init = init

        if max_iter is None:
            if self.method == 'als':
                self.max_iter = 200
            elif self.method == 'admm':
                self.max_iter = 6000
            elif self.method == 'pgm':
                self.max_iter = 10000
            elif self.method == 'dr':
                self.max_iter = 200
        else:
            self.max_iter = max_iter

        if tol is None:
            self.tol = 1e-3 if self.method == 'pgm' else 1e-5
        else:
            self.tol = tol

        self.k = k
        self.nls_max_iter = nls_max_iter
        self.psdls_max_iter = psdls_max_iter
        self.nls_beta = nls_beta
        self.psdls_beta = psdls_beta
        self.pgm_beta = pgm_beta
        self.correlation = correlation
        self.admm_gamma = admm_gamma
        self.admm_alpha = admm_alpha
        self.verbose = verbose
        self.obj_tol = obj_tol
        self.time = time
        self.times = None
        self.dictionary = None
        self.objective = None
        self.rand_on = rand_on
        self.rand_k = rand_k
        self.random_state = random_state
        self.n_jobs = 1 if n_jobs is None else n_jobs


    def _initialize(self, X, Dinit=None):

        # Initializes the dictionary D and weights W randomly or using k-means,
        # as in Wild, Curry, & Dougherty (2004) "Improving non-negative
        # matrix factorizations through structured initialization".

        n_pair, n_samp = X.shape
        n = npair2n(n_pair)

        if self.init == 'kmeans':

            # Initialize modules to k-means cluster centroids.
            from sklearn.cluster import MiniBatchKMeans

            Xnorm = normalize(X, norm='l2', axis=0)
            km = MiniBatchKMeans(
                n_clusters=self.k,
                compute_labels=False,
                batch_size=100,
                n_init=3,
            ).fit(Xnorm.T)
            centroids = km.cluster_centers_.T
            Dinit = proj_col_psd(centroids, self.correlation,
                                 rand_on=self.rand_on, rand_k=self.rand_k)
            Winit = rand(self.k, n_samp)

        elif self.init == 'rand':

            # Initialize modules to random linear combinations
            # of input covariances.
            Dinit = dot(X, rand(n_samp, self.k))
            Winit = rand(self.k, n_samp)

        elif self.init == 'svd':

            # Initialize modules based on the singular value decomposition
            # (SVD), better for sparseness.
            from fbpca import pca
            U, S, V = pca(X, min(self.k, X.shape[0], X.shape[1]))
            Dinit = U * S
            Dinit = proj_col_psd(Dinit, self.correlation,
                                 rand_on=self.rand_on, rand_k=self.rand_k)
            Winit = (V.T * S).T

        elif self.init == 'Dinit' and Dinit is not None:
            Dinit = proj_col_psd(Dinit, self.correlation,
                                 rand_on=self.rand_on, rand_k=self.rand_k)
            Winit = rand(self.k, n_samp)

        Winit, _, _ = self._nls_subproblem(X, Dinit, Winit, 1e-3)
        return Dinit, Winit



    def _admm(self, X, Dinit, Winit):

        # Solves for covariance module and weights using ADMM. Reformulate
        # original optimization problem

        # minimize ||X - DW||_F
        # subject to
        # each column of D is a PSD matrix
        # each element of W is non-negative

        # as

        # minimize ||X - DW||_F
        # subject to
        # D = U
        # W = V
        # each column of U is a PSD matrix
        # each element of V is non-negative

        # and sequentially minimize the augmented Lagrangian
        # w.r.t U, V, D, and W, then update dual variables

        # Can also solve problem under constraint of correlation
        # matrices rather than general PSD matrices.

        n_pair, n_samp = X.shape
        n = npair2n(n_pair)
        max_dim = max([n_pair, n_samp])

        D = Dinit
        W = Winit
        V = Winit
        Lambda = zeros((n_pair, self.k))
        Pi = zeros((self.k, n_samp))

        normX = norm(X)
        objective = empty(self.max_iter)
        obj_prev = finfo('d').max
        last_decrease = 0

        if self.time:
            times = empty(self.max_iter)
            t = time.time()
        else:
            times = None

        for n_iter in range(self.max_iter):

            # Record objective.
            obj = norm(X - dot(D, W)) / normX
            objective[n_iter] = obj

            if self.time:
                times[n_iter] = time.time() - t

            # Stopping condition.
            if self.obj_tol is None:
                if (abs(obj - obj_prev) / fmax(1, obj_prev) < self.tol or
                    obj < self.tol or
                    (obj >= obj_prev and last_decrease + 10 <= n_iter)):
                    break
            elif obj < self.obj_tol:
                break

            if obj < obj_prev:
                last_decrease = n_iter

            obj_prev = obj

            if self.verbose:
                if mod(n_iter, 20) == 0:
                    tprint('Iter: %i. Objective: %f.' % (n_iter, obj))

            # Step size rule
            # alpha = self.admm_alpha * normX * max_dim / (n_iter + 1)
            # beta = alpha * n_samp / n_pair
            alpha = self.admm_alpha * max_dim / (n_iter + 1)
            beta = alpha * n_samp / n_pair

            # Primal variable updates
            U = dot(dot(X, V.T) + alpha * D - Lambda,
                    inv(dot(V, V.T) + alpha * identity(self.k)))
            V = dot(inv(dot(U.T, U) + beta * identity(self.k)),
                    dot(U.T, X) + beta * W - Pi)

            D = proj_col_psd(U + Lambda / alpha, self.correlation,
                             rand_on=self.rand_on, rand_k=self.rand_k)
            W = proj_weights(V + Pi / beta, self.correlation)

            # Dual variable updates
            Lambda = Lambda + self.admm_gamma * alpha * (U - D)
            Pi = Pi + self.admm_gamma * beta * (V - W)


        if self.verbose:
            tprint('Final iter: %i. Objective: %f.' % (n_iter, obj))

        objective = objective[: n_iter + 1]
        if self.time:
            times = times[: n_iter + 1]

        return D, W, objective, times

    # "Warm-start" trick in Lin & More (1999) for searching step size.
    def _adjust_step(self, decr_alpha, suff_decr, alpha, beta):

        # 1(a) If initially not sufficient decrease...
        if decr_alpha:
            # 1(c) ...there is sufficient decrease.
            if suff_decr:
                return CovarianceDictionary.SUFF_DECR
            # 1(b) ...decrease alpha until...
            else:
                return alpha * beta

        # 2(b) ...there is not sufficient decrease.
        elif not suff_decr: # or (Wold == Wnew).all():
            return CovarianceDictionary.INSUFF_DECR

        # 2(a) If initially sufficient decrease, increase alpha until...
        else:
            return alpha / beta



    def _nls_subproblem(self, X, D, Winit, tol):

        # Update weights by solving non-negative least-squares
        # using projected gradient descent (basically a transposed
        # version of scikit-learn's NMF _nls_subproblem method).

        W = Winit
        DtX = dot(D.T, X)
        DtD = dot(D.T, D)

        pg_norm = empty(self.nls_max_iter)
        # in_iter = empty(self.max_iter)

        alpha = 1.

        for n_iter in range(self.nls_max_iter):

            grad = dot(DtD, W) - DtX

            #print(np.sum(np.isnan(grad)))
            #print(np.sum(np.isinf(grad)))

            # Stopping condition on projected gradient norm.
            # Multiplication with a boolean array is more than twice
            # as fast as indexing into grad.

            # Correct modification for simplex projection?
            pgn = norm(proj_weights(W - grad, self.correlation) - W)
            # pgn = norm(grad * logical_or(grad < 0, W > 0))
            pg_norm[n_iter] = pgn

            if pgn < tol:
                break

            Wold = W

            # Search for step size that produces sufficient decrease
            # ("Armijo rule along the projection arc" in Bertsekas (1999), using shortcut
            # condition in Lin (2007) Eq. (16)).
            for inner_iter in range(10):

                # Gradient and projection step.
                Wnew = W - alpha * grad
                # Wnew *= Wnew > 0
                Wnew = proj_weights(Wnew, self.correlation)

                #print(np.sum(np.isnan(Wnew)))
                #print(np.sum(np.isinf(Wnew)))

                # Check Lin (2007) Eq. (16) condition.
                d = Wnew - W
                gradd = dot(grad.ravel(), d.ravel())
                dQd = dot(dot(DtD, d).ravel(), d.ravel())
                suff_decr = 0.99 * gradd + 0.5 * dQd < 0

                if inner_iter == 0:
                    decr_alpha = not suff_decr

                status = self._adjust_step(decr_alpha, suff_decr, alpha, self.nls_beta)
                if (status == CovarianceDictionary.SUFF_DECR):
                    W = Wnew
                    break
                elif (status == CovarianceDictionary.INSUFF_DECR):
                    W = Wold
                    break
                else:
                    alpha = status
                    Wold = Wnew

            # in_iter[n_iter] = inner_iter

        if n_iter == self.nls_max_iter - 1:
            warnings.warn("Max iterations reached in NLS subproblem.")

        pg_norm = pg_norm[: n_iter + 1]
        # in_iter = in_iter[: n_iter]

        return W, grad, n_iter


    def _psdls_subproblem(self, X, Dinit, W, tol):

        # Update modules by solving column-wise positive-semidefinite (PSD)
        # constrained least-squares using projected gradient descent:

        # minimize ||X - D * W||_F
        # subject to the constraint that every column of D
        # corresponds to the upper triangle of a PSD matrix.

        n_pair, n_samp = X.shape
        n = npair2n(n_pair)

        D = Dinit
        WWt = dot(W, W.T)
        XWt = dot(X, W.T)
        pg_norm = empty(self.psdls_max_iter)
        # in_iter = empty(self.psdls_max_iter)

        alpha = 1

        for n_iter in range(self.psdls_max_iter):

            gradD = dot(D, WWt) - XWt

            # Stopping condition on projected gradient norm.
            pgn = norm(proj_col_psd(D - gradD, self.correlation,
                                    rand_on=self.rand_on, rand_k=self.rand_k)
                       - D)
            pg_norm[n_iter] = pgn

            if pgn < tol:
                break

            Dold = D

            # Search for step size that produces sufficient decrease
            # ("Armijo rule along the projection arc" in Bertsekas (1999), using shortcut
            # condition in Lin (2007) Eq. (16).)
            for inner_iter in range(20):

                # Gradient and projection steps.
                Dnew = proj_col_psd(D - alpha * gradD, self.correlation,
                                    rand_on=self.rand_on, rand_k=self.rand_k)

                d = Dnew - D
                gradd = dot(gradD.ravel(), d.ravel())
                dQd = dot(dot(d, WWt).ravel(), d.ravel())
                suff_decr = 0.99 * gradd + 0.5 * dQd < 0

                if inner_iter == 0:
                    decr_alpha = not suff_decr

                status = self._adjust_step(decr_alpha, suff_decr, alpha, self.psdls_beta)
                if (status == CovarianceDictionary.SUFF_DECR):
                    D = Dnew
                    break
                elif (status == CovarianceDictionary.INSUFF_DECR):
                    D = Dold
                    break
                else:
                    alpha = status
                    Dold = Dnew

            # in_iter[n_iter] = inner_iter

        if n_iter == self.psdls_max_iter - 1:
            warnings.warn("Max iterations reached in PSDLS subproblem.")

        pg_norm = pg_norm[: n_iter + 1]
        # in_iter = in_iter[: n_iter]

        return D, gradD, n_iter


    def _als(self, X, Dinit, Winit):

        # Solves for covariance module and weights using
        # alternating constrained least-squares. Same framework
        # as scikit-learn's ALS for NMF.

        n_pair, n_mat = X.shape
        n = npair2n(n_pair)

        D = Dinit
        W = Winit

        # Initial gradient.
        gradD = dot(D, dot(W, W.T)) - dot(X, W.T)
        gradW = dot(dot(D.T, D), W) - dot(D.T, X)
        init_grad_norm = norm_concat(gradD, gradW.T)

        if self.verbose:
            tprint("Initial gradient norm: %f." % init_grad_norm)

        # Stopping tolerances for constrained ALS subproblems.
        tolD = self.tol * init_grad_norm
        tolW = tolD

        normX = norm(X)
        objective = empty(self.max_iter)
        # pg_norm = empty(self.max_iter)

        if self.time:
            times = empty(self.max_iter)
            t = time.time()
        else:
            times = None

        for n_iter in range(self.max_iter):

            # Stopping criterion, based on Calamai & More (1987) Lemma 3.1(c)
            # (stationary point iff projected gradient norm = 0).
            pgradW = gradW * logical_or(gradW < 0, W > 0)
            pgradD = proj_col_psd(D - gradD, self.correlation,
                                  rand_on=self.rand_on, rand_k=self.rand_k) - D
            pgn = norm_concat(pgradD, pgradW)
            # pg_norm[n_iter] = pgn

            # Record objective.
            obj = norm(X - dot(D, W)) / normX
            objective[n_iter] = obj

            if self.time:
                times[n_iter] = time.time() - t

            # Stopping condition.
            if self.obj_tol is None:
                if pgn < self.tol * init_grad_norm:
                    break
            elif obj < self.obj_tol:
                break

            if self.verbose:
                tprint('Iter: %i. Projected gradient norm: %f. Objective: %f.' % (n_iter, pgn, obj))

            # Update modules.
            D, gradD, iterD = self._psdls_subproblem(X, D, W, tolD)
            if iterD == 0:
                tolD = 0.1 * tolD

            # Update weights.
            W, gradW, iterW = self._nls_subproblem(X, D, W, tolW)
            if iterW == 0:
                tolH = 0.1 * tolW

        if self.verbose:
            tprint('Iter: %i. Final projected gradient norm %f. Final objective %f.' % (n_iter, pgn, obj))

        objective = objective[: n_iter + 1]
        # pg_norm = pg_norm[: n_iter + 1]

        if self.time:
            times = times[: n_iter + 1]

        return D, W, objective, times



    def _pgm(self, X, Dinit, Winit):

        # Solves for covariance modules and weights using
        # a projected gradient method.

        n_pair, n_mat = X.shape
        n = npair2n(n_pair)

        D = Dinit
        W = Winit

        # Initial gradient.
        gradD = dot(D, dot(W, W.T)) - dot(X, W.T)
        gradW = dot(dot(D.T, D), W) - dot(D.T, X)
        init_grad_norm = norm_concat(gradD, gradW.T)

        if self.verbose:
            tprint("Initial gradient norm: %f." % init_grad_norm)

        normX = norm(X)
        objective = empty(self.max_iter)

        if self.time:
            times = empty(self.max_iter)
            t = time.time()
        else:
            times = None

        alpha = 1

        for n_iter in range(self.max_iter):

            # Compute projected gradient for stopping condition.
            pgradW = gradW * logical_or(gradW < 0, W > 0)
            pgradD = proj_col_psd(D - gradD, self.correlation,
                                  rand_on=self.rand_on, rand_k=self.rand_k) - D
            pgn = norm_concat(pgradD, pgradW)

            # Record objective.
            obj = norm(X - dot(D, W)) / normX
            objective[n_iter] = obj

            if self.time:
                times[n_iter] = time.time() - t

            # Stopping criterion, based on Calamai & More (1987) Lemma 3.1(c)
            # (stationary point iff projected gradient norm = 0).
            if self.obj_tol is None:
                if pgn < self.tol * init_grad_norm:
                    break
            elif obj < self.obj_tol:
                break

            if self.verbose:
                tprint('Iter: %i. Projected gradient norm: %f. Objective: %f.' % (n_iter, pgn, obj))

            Wold = W
            Dold = D

            # Search for step size that produces sufficient decrease
            # ("Armijo rule along the projection arc" in Bertsekas (1999)).
            for inner_iter in range(20):

                # Proposed updates of dictionary and weights
                # (gradient step and projection step)
                # Gradient step.
                # reg = 1.0 / (inner_iter + 1) # momentum-style regularization,
                                               # see Combettes & Wajs (2005)
                Wnew = W - alpha * gradW
                # Wnew = W + reg * (Wnew * (Wnew > 0) - W)
                Wnew *= Wnew > 0
                Dnew = proj_col_psd(D - alpha * gradD, self.correlation,
                                    rand_on=self.rand_on, rand_k=self.rand_k)
                # Dnew = D - alpha * gradD
                # Dnew = D + reg * (proj_col_psd(Dnew, self.correlation) - D)

                # Check for sufficient decrease.
                obj_old = pow(obj * normX, 2)
                obj_new = pow(norm(X - dot(Dnew, Wnew)), 2)
                W_thresh_decr = dot(gradW.ravel(), (Wnew - W).ravel())
                D_thresh_decr = dot(gradD.ravel(), (Dnew - D).ravel())
                suff_decr = obj_new - obj_old < 0.01 * (W_thresh_decr + D_thresh_decr)

                if inner_iter == 0:
                    decr_alpha = not suff_decr

                # status = self._adjust_step(decr_alpha, suff_decr, alpha, self.pgm_beta)
                # if (status == CovarianceDictionary.SUFF_DECR):
                #     W = Wnew
                #     D = Dnew
                #     break
                # elif (status == CovarianceDictionary.INSUFF_DECR):
                #     W = Wold
                #     D = Dold
                #     break
                # else:
                #     alpha = status
                #     W = Wold
                #     D = Dold

                if decr_alpha:
                    # 1.3 ...there is sufficient decrease.
                    if suff_decr:
                        W = Wnew
                        D = Dnew
                        break
                    # 1.2 ...decrease alpha until...
                    else:
                        alpha *= self.pgm_beta

                # 2.3 ...there is not sufficient decrease.
                elif not suff_decr or ((Wold == Wnew).all() and (Dold == Dnew).all()):
                    W = Wold
                    D = Dold
                    break

                # 2.2 ...increase alpha until...
                else:
                    alpha /= self.pgm_beta
                    Wold = Wnew
                    Dold = Dnew

            gradD = dot(D, dot(W, W.T)) - dot(X, W.T)
            gradW = dot(dot(D.T, D), W) - dot(D.T, X)

        if self.verbose:
            tprint('Iter: %i. Final projected gradient norm %f. Final objective %f.' % (n_iter, pgn, obj))

        if n_iter == self.max_iter - 1:
            warnings.warn("Max iterations reached in PSDLS subproblem.")

        objective = objective[: n_iter + 1]
        if self.time:
            times = times[: n_iter + 1]

        return D, W, objective, times



    def _prox_loss(self, D, W, X, gamma):

        # Alternating least-squares to solve for the proximal
        # operator of ||X - DW||_F^2.

        # Equivalent to initializing Y = Z = 0.
        n_pair = X.shape[0]
        Y = D
        Z = W

        for n_iter in range(20):

            Yold = Y
            Zold = Z

            Y = solve(gamma * dot(Zold, Zold.T) + identity(self.k), gamma * dot(Zold, X.T) + D.T).T
            Z = solve(gamma * dot(Y.T, Y) + identity(self.k), gamma * dot(Y.T, X) + W)

            norm_diff = norm_concat(Y - Yold, Z - Zold) # relative to norm_concat(Y, Z)?
            if norm_diff < self.tol:
                break

        return Y, Z



    def _dr(self, X, Dinit, Winit):

        # Douglas-Rachford, using alternating least-squares
        # to solve for the proximal operator of ||X - DW||_F^2.
        # Seems to get stuck suboptimally...

        n_pair, n_mat = X.shape
        n = npair2n(n_pair)

        D = Dinit
        W = Winit
        Dhalf = zeros((n_pair, self.k)) # better initialization?
        Whalf = zeros((self.k, n_mat))

        dr_gamma = 5.0
        epsilon = 0.2

        normX = norm(X)
        objective = empty(self.max_iter)

        if self.time:
            times = empty(self.max_iter)
            t = time.time()
        else:
            times = None

        for n_iter in range(self.max_iter):

            # Update using proximal operator.
            Dprox, Wprox = self._prox_loss(2 * D - Dhalf, 2 * W - Whalf, X, dr_gamma)
            reg = max([(2 - epsilon) / (n_iter + 1), epsilon])
            Dhalf = Dhalf + reg * (Dprox - D)
            Whalf = Whalf + reg * (Wprox - W)

            # Make feasible.
            Dold = D
            Wold = W
            D = proj_col_psd(Dhalf, self.correlation,
                             rand_on=self.rand_on, rand_k=self.rand_k)
            W *= Whalf > 0

            if self.time:
                times[n_iter] = time.time() - t

            # Record objective.
            obj = norm(X - dot(D, W)) / normX
            objective[n_iter] = obj

            # relative to norm_concat(D, W)?
            norm_diff = norm_concat(D - Dold, W - Wold)

            # Check for convergence.
            if self.obj_tol is None:
                if norm_diff < self.tol:
                    break
            else:
                if obj < self.obj_tol:
                    break

            if self.verbose:
                tprint('Iter: %i. Norm of iterate difference: %f. Objective: %f.'
                       % (n_iter, norm_diff, obj))

        objective = objective[: n_iter + 1]
        if self.time:
            times = times[: n_iter + 1]
        return D, W, objective, times



    def fit_transform(self, X, is_unpacked=False, Dinit=None):

        """Learns a dictionary from data X and returns dictionary weights


        Parameters
        ----------
        X : array, shape (n, n, n_samp)
            Input covariance data as a stack of n_samp covariance matrices

        Returns
        -------
        W : array, shape (k, n_samp)
            Weights of dictionary elements for input

        """

        if self.verbose:
            tprint('Unpacking samples...')
        if not is_unpacked:
            X = unpack_samples(X) # unpack into (n_pair, n_samp) array

        if self.verbose:
            tprint('Initializing matrices...')
        Dinit, Winit = self._initialize(X, Dinit=Dinit)

        if self.verbose:
            tprint('Optimizing with {}...'.format(self.method))

        if self.method == 'als':
            D, W, obj, times = self._als(X, Dinit, Winit)
        elif self.method == 'admm':
            D, W, obj, times = self._admm(X, Dinit, Winit)
        elif self.method == 'pgm':
            D, W, obj, times = self._pgm(X, Dinit, Winit)
        elif self.method == 'dr':
            D, W, obj, times = self._dr(X, Dinit, Winit)

        self.dictionary = pack_samples(D)
        self.D = D
        self.objective = obj
        self.times = times

        return W



    def fit(self, X, Dinit=None):

        """Learns a covariance dictionary from the covariance data X


        Parameters
        ----------
        X : array, shape (n, n, n_samp)
            Input covariance data as a stack of n_samp covariance matrices

        Returns
        -------
        self : object
            The instance itself

        """

        self.fit_transform(X, Dinit=Dinit)
        return self



    def transform(self, X):

        """
        Computes the dictionary weights for input covariance data

        Parameters
        ----------
        X : array, shape (n, n, n_samp)
            Input covariance data as a stack of n_samp covariance matrices

        Returns
        -------
        W : array, shape (k, n_samp)
            Weights of dictionary elements for each input

        """

        X = unpack_samples(X)
        if self.dictionary is not None:
            W = self._nls_subproblem(X, self.dictionary, rand(k, n_samp), 1e-3)
        else:
            W = self.fit_transform(X)

        return W
