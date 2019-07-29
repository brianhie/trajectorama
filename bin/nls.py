import numpy as np
from scipy.optimize.lbfgsb import fmin_l_bfgs_b
import scipy.sparse as ss
from sklearn.utils import check_array
from sklearn.utils.extmath import safe_sparse_dot

def nls_lbfgs_b(S, D, C_init=None,
                l1_reg=0.1, max_iter=1000,
                tol=1e-4, callback=None):
    """Non-negative least squares solver using L-BFGS-B.
    
    """
    S = ss.csr_matrix(check_array(S, accept_sparse='csr'))
    D = ss.csr_matrix(check_array(D, accept_sparse='csr'))
    n_features = S.shape
    n_components = D.shape[1]

    DtD = safe_sparse_dot(D.T, D)
    DtSD = safe_sparse_dot(D.T, safe_sparse_dot(S, D))

    def f(C, *args):
        C = ss.diags(C)
        tonorm = S - safe_sparse_dot(D, safe_sparse_dot(C, D.T))
        reg = l1_reg * C.diagonal().sum()
        return (0.5 * (ss.linalg.norm(tonorm) ** 2)) + reg

    def fprime(C, *args):
        C = ss.diags(C)
        DtDCDtD = safe_sparse_dot(DtD, safe_sparse_dot(C, DtD))
        reg = l1_reg * ss.eye(C.shape[0])
        full = DtDCDtD - DtSD + reg
        return full.diagonal()

    if C_init is None:
        C = np.zeros(n_components, dtype=np.float64)
    elif C_init.shape == (n_features, n_features):
        C = np.diag(C_init)
    else:
        C = C_init
        
    C, residual, d = fmin_l_bfgs_b(
        f, x0=C, fprime=fprime, pgtol=tol,
        bounds=[(0, None)] * n_components,
        maxiter=max_iter, callback=callback,
    )
    
    # testing reveals that sometimes, very small negative values occur
    C[C < 0] = 0

    if l1_reg:
        residual -= l1_reg * C.sum()
    residual = np.sqrt(2 * residual)
    if d['warnflag'] > 0:
        print("L-BFGS-B failed to converge")
    
    return C, residual

if __name__ == '__main__':
    n_features = 100

    from numpy.testing import assert_almost_equal
    from sklearn.utils.testing import assert_true

    S = np.random.rand(n_features, n_features)
    S = S.dot(S.T)

    for n_components in [ 1, 10, 25, 50, 75, 100, 200 ]:
        D = np.random.rand(n_features, n_components)

        C, resid = nls_lbfgs_b(S, D)
        true_resid = np.linalg.norm(S - D.dot(np.diag(C)).dot(D.T))

        assert_almost_equal(resid, true_resid)
        assert_true(np.all(C >= 0))

        print('n_features: {}, n_components: {}, resid: {}'
              .format(n_features, n_components, resid))

    
