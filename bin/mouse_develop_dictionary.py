import glob
import itertools
import networkx as nx
from networkx.drawing.nx_pylab import draw_networkx

from utils import *

NAMESPACE = 'mouse_develop_spearman_louvain'

if __name__ == '__main__':
    from hematopoiesis_dictionary import (
        construct_networks,
        interpret_networks,
        network_overlap,
    )

    dirname = 'target/sparse_correlations/{}'.format(NAMESPACE)

    with open('{}/genes.txt'.format(dirname)) as f:
        genes = f.read().rstrip().split('\n')

    of_interest = [ 1, 2, 4, 8, 13 ]

    networks, pair2comp = construct_networks(of_interest, dirname, genes)

    interpret_networks(networks, pair2comp, dirname, genes)
