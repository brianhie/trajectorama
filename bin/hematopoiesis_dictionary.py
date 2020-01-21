import glob
import gzip
import itertools
import networkx as nx
from networkx.drawing.nx_pylab import draw_networkx
import os
import scipy.sparse as ss

from utils import *

NAMESPACE = 'hematopoiesis_spearman_louvain'

def permute_and_save(edges, perm_fname):
    import random
    random.seed(perm_fname)
    edges_set = set([ (e[0], e[1]) for e in edges ]) | \
                set([ (e[1], e[0]) for e in edges ])

    for _ in range(10):
        for idx in range(len(edges) - 1):
            swap_idx = random.randint(idx + 1, len(edges) - 1)
            for tries in range(10):
                new_first = edges[idx][0], edges[swap_idx][1]
                new_last = edges[swap_idx][0], edges[idx][1]
                if new_first in edges_set or new_last in edges_set:
                    continue
                edges[idx], edges[swap_idx] = new_first, new_last

    with open(perm_fname, 'w') as of:
        [ of.write('\t'.join(edge) + '\n') for edge in edges ]

    return edges

def load_known_nets(n_permute=100):
    net_fnames = [
        'data/SCT-MoA/networks/HIPPIE/human.txt.gz',
        'data/SCT-MoA/networks/OmniPath/human.txt.gz',
        'data/SCT-MoA/networks/Reactome/human.txt.gz',
        'data/SCT-MoA/networks/STRING/human.txt.gz',
    ]

    with open('data/gene2ensid.tsv') as f:
        ensid2gene = { line.split()[0]: line.split()[1]
                       for line in f.read().splitlines() }

    for fname in net_fnames:
        net_name = fname.split('/')[3]

        with gzip.open(fname) as f:
            f.readline()
            edges_ensid = f.read().decode('utf-8').splitlines()
            edges = [
                (ensid2gene[ensids.split()[0]], ensid2gene[ensids.split()[1]])
                for ensids in edges_ensid
                if ensids.split()[0] in ensid2gene and
                ensids.split()[1] in ensid2gene
            ]

        yield net_name, edges

        net_dirname = '/'.join(fname.split('/')[:-1])
        for pidx in range(n_permute):
            perm_name = net_name + '_perm{}'.format(pidx)
            perm_fname = net_dirname + '/net_perm_{}.txt'.format(pidx)
            if not os.path.isfile(perm_fname):
                permute_and_save(edges[:], perm_fname)
            with open(perm_fname) as f:
                edges = [ (edge.split()[0], edge.split()[1])
                          for edge in f.read().splitlines() ]
            yield perm_name, edges

def zscore(val, distribution):
    return (val - np.mean(distribution)) / np.std(distribution)

def pval(val, distribution):
    return sum(np.array(distribution) >= val) / len(distribution)

def network_overlap(networks, genes, baseline_fname, n_top_edges=10000):

    # Construct baseline network.

    baseline = nx.Graph()
    baseline_corr = np.load(baseline_fname)
    for gidx1, gene1 in enumerate(genes):
        for gidx2, gene2 in enumerate(genes):
            if gidx1 > gidx2:
                continue
            if baseline_corr[gidx1, gidx2] != 0:
                baseline.add_edge(
                    gene1, gene2,
                    weight=baseline_corr[gidx1, gidx2]
                )
    baseline_net = nx.Graph()
    baseline_net.add_edges_from(
        sorted(baseline.edges(data=True),
               key=lambda x: -abs(x[2].get('weight', 1)))[:n_top_edges]
    )

    # Construct target network.

    target = nx.compose_all([
        networks[comp] for comp in networks
    ])
    target_net = nx.Graph()
    target_net.add_edges_from(
        sorted(target.edges(data=True),
               key=lambda x: -abs(x[2].get('weight', 1)))[:n_top_edges]
    )

    # Compare with permuted networks.

    known_networks = load_known_nets()

    target_ns, baseline_ns = {}, {}
    target_net_perms, baseline_net_perms = {}, {}
    for network_name, edges in known_networks:
        prefix = network_name.split('_')[0]

        n_overlap = sum([
            target_net.has_edge(*edge) for edge in edges
        ])
        if '_perm' in network_name:
            if not prefix in target_net_perms:
                target_net_perms[prefix] = []
            target_net_perms[prefix].append(n_overlap)
        else:
            target_ns[prefix] = n_overlap

        n_overlap = sum([
            baseline_net.has_edge(*edge) for edge in edges
        ])
        if '_perm' in network_name:
            if not prefix in baseline_net_perms:
                baseline_net_perms[prefix] = []
            baseline_net_perms[prefix].append(n_overlap)
        else:
            baseline_ns[prefix] = n_overlap

    # Compute z-scores and output results.

    for prefix in target_net_perms:
        fields = [
            prefix, 'target', target_ns[prefix],
            target_ns[prefix] / target_net.number_of_edges(),
            zscore(target_ns[prefix], target_net_perms[prefix]),
            pval(target_ns[prefix], target_net_perms[prefix])
        ]
        print('\t'.join([ str(field) for field in fields ]))
        fields = [
            prefix, 'baseline', baseline_ns[prefix],
            baseline_ns[prefix] / baseline_net.number_of_edges(),
            zscore(baseline_ns[prefix], baseline_net_perms[prefix]),
            pval(baseline_ns[prefix], baseline_net_perms[prefix])
        ]
        print('\t'.join([ str(field) for field in fields ]))

def construct_networks(of_interest, dirname, genes):
    gene2idx = { gene: idx for idx, gene in enumerate(genes) }

    with open('{}/gene_pairs.txt'.format(dirname)) as f:
        gene_pairs = [ tuple(pair.split('_'))
                       for pair in f.read().rstrip().split('\n') ]

    with open('{}/gene_indices.txt'.format(dirname), 'w') as of:
        [ of.write('{}\t{}\n'.format(idx + 1, gene))
          for idx, gene in enumerate(genes) ]

    n_features = len(genes)
    n_correlations = len(gene_pairs)

    components = np.zeros((len(of_interest), n_correlations))
    networks = {}
    pair2comp = {}
    for comp_idx, comp in enumerate(of_interest):
        networks[comp] = nx.Graph()
        networks[comp].add_nodes_from(genes)

        components[comp_idx, :] = np.loadtxt(
            '{}/dictw{}.txt'.format(dirname, comp)
        )

        adjacency = np.zeros((n_features, n_features))

        with open('{}/gene_adjacency_dict{}.txt'
                  .format(dirname, comp), 'w') as of:
            for nc in range(n_correlations):
                idx_i = gene2idx[gene_pairs[nc][0]]
                idx_j = gene2idx[gene_pairs[nc][1]]
                if idx_i == idx_j:
                    continue
                adjacency[idx_i, idx_j] = components[comp_idx, nc]
                adjacency[idx_j, idx_i] = components[comp_idx, nc]
                of.write('{}\t{}\n'.format(idx_i + 1, idx_j + 1))
                of.write('{}\t{}\n'.format(idx_j + 1, idx_i + 1))

                if components[comp_idx, nc] > 0:
                    if idx_i > idx_j:
                        idx_i, idx_j = idx_j, idx_i
                    if (idx_i, idx_j) not in pair2comp:
                        pair2comp[(idx_i, idx_j)] = set()
                    pair2comp[(idx_i, idx_j)].add(comp)

                    networks[comp].add_edge(
                        genes[idx_i], genes[idx_j],
                        weight=components[comp_idx, nc]
                    )
    return networks, pair2comp

def visualize_top_betweenness(genes, network, fname):
    small_net = nx.Graph()

    for gene1, gene2 in itertools.combinations(genes, 2):
        small_net.add_node(gene1)
        small_net.add_node(gene2)
        try:
            path = nx.shortest_path(network, source=gene1, target=gene2)
        except nx.exception.NetworkXNoPath:
            continue
        if 0 < len(path) <= 3:
            small_net.add_edge(gene1, gene2)

    pos = nx.drawing.layout.spring_layout(small_net)
    plt.figure()
    ax = plt.gca()
    draw_networkx(small_net, pos=pos, node_size=1400,
                  node_color='#bacbd3', edge_color='#cccccc',
                  font_size=35, font_weight='normal', font_family='Helvetica')
    ratio = 0.4
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    ax.set_aspect(abs((xmax - xmin) / (ymax - ymin)) * ratio)
    ax.margins(0.1)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(fname)

def interpret_networks(networks, pair2comp, dirname, genes):
    for comp in networks:
        print('\nRW Betweeness for component {}'.format(comp))
        node2central = nx.betweenness_centrality(networks[comp])
        top_genes = []
        for idx, (gene, central) in enumerate(sorted(
                node2central.items(), key=lambda kv: -kv[1]
        )):
            if central > 0:
                print('{}\t{}'.format(gene, central))
            if idx < 10:
                top_genes.append(gene)

        visualize_top_betweenness(top_genes, networks[comp],
                                  'small_net_{}.svg'.format(comp))

    uniq_links = { comp: set() for comp in networks }

    with open('{}/gene_pair_comp.txt'.format(dirname), 'w') as of:
        for idx_i, idx_j in pair2comp:
            comp = ','.join([ str(c) for c in pair2comp[(idx_i, idx_j)] ])
            fields = [ genes[idx_i], genes[idx_j], comp ]
            of.write('\t'.join([ str(field) for field in fields ]) + '\n')

            if len(pair2comp[(idx_i, idx_j)]) == 1:
                only_comp = list(pair2comp[(idx_i, idx_j)])[0]
                uniq_links[only_comp].add(genes[idx_i])
                uniq_links[only_comp].add(genes[idx_j])

    for comp in uniq_links:
        with open('{}/uniq_link_genes_{}.txt'
                  .format(dirname, comp), 'w') as of:
            of.write('\n'.join(sorted(uniq_links[comp])))

if __name__ == '__main__':
    dirname = 'target/sparse_correlations/{}'.format(NAMESPACE)

    with open('{}/genes.txt'.format(dirname)) as f:
        genes = f.read().rstrip().split('\n')

    of_interest = list(range(15))#[ 5, 6, 13, 3 ]

    networks, pair2comp = construct_networks(of_interest, dirname, genes)

    baseline_fname = dirname + '/full_corr.npy'

    n_tops = [ 5000, 10000, 15000, 20000 ]
    for n_top in n_tops:
        tprint('Top {} edges'.format(n_top))
        network_overlap(networks, genes, baseline_fname, n_top)

    #interpret_networks(networks, pair2comp, dirname, genes)
