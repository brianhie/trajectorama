import datetime
import errno
import fbpca
import os
import sys

def tprint(string):
    string = str(string)
    sys.stdout.write(str(datetime.datetime.now()) + ' | ')
    sys.stdout.write(string + '\n')
    sys.stdout.flush()

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

def reduce_dimensionality(X, dim_red_k=100):
    k = min((dim_red_k, X.shape[0], X.shape[1]))
    from fbpca import pca
    U, s, Vt = pca(X, k=k) # Automatically centers.
    return U[:, range(k)] * s[range(k)]
