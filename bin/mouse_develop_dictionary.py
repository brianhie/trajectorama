import itertools
import networkx as nx
from networkx.drawing.nx_pylab import draw_networkx

from mouse_develop_cached import *

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

if __name__ == '__main__':

    dirname = 'target/sparse_correlations/{}'.format(NAMESPACE)
    
    with open('{}/genes.txt'.format(dirname)) as f:
        genes = f.read().rstrip().split('\n')
        
    gene2idx = { gene: idx for idx, gene in enumerate(genes) }

    n_features = len(genes)

    with open('{}/gene_pairs.txt'.format(dirname)) as f:
        gene_pairs = [ tuple(pair.split('_'))
                       for pair in f.read().rstrip().split('\n') ]
    
    n_correlations = len(gene_pairs)

    components = np.zeros((N_COMPONENTS, n_correlations))

    with open('{}/gene_indices.txt'.format(dirname), 'w') as of:
        [ of.write('{}\t{}\n'.format(idx + 1, gene))
          for idx, gene in enumerate(genes) ]
        
    adata = AnnData(X=np.ones((n_features, 1)))
    adata.obs['names'] = genes

    of_interest = set([ 8, 3, 20, 14, 19, 17 ])
    pair2comp = {}
    
    networks = { comp: nx.Graph()
                 for comp in range(N_COMPONENTS) if comp in of_interest }
    [ networks[comp].add_nodes_from(genes) for comp in networks ]

    for comp in range(N_COMPONENTS):
        if comp not in of_interest:
            continue
        
        components[comp, :] = np.loadtxt(
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
                adjacency[idx_i, idx_j] = components[comp, nc]
                adjacency[idx_j, idx_i] = components[comp, nc]
                of.write('{}\t{}\n'.format(idx_i + 1, idx_j + 1))
                of.write('{}\t{}\n'.format(idx_j + 1, idx_i + 1))

                if components[comp, nc] > 0:
                    if idx_i > idx_j:
                        idx_i, idx_j = idx_j, idx_i
                    if (idx_i, idx_j) not in pair2comp:
                        pair2comp[(idx_i, idx_j)] = set()
                    pair2comp[(idx_i, idx_j)].add(comp)

                    networks[comp].add_edge(
                        genes[idx_i], genes[idx_j], weight=components[comp, nc]
                    )

    from hematopoiesis_dictionary import network_overlap
    baseline_fname = (
        dirname + '/node_0_at_14.656471893132178_has_932301_leaves.npz'
    )
    network_overlap(networks, genes, baseline_fname)
                    
    exit()
    
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

    uniq_links = { comp: set() for comp in range(N_COMPONENTS) }

    with open('{}/gene_pair_comp.txt'.format(dirname), 'w') as of:
        for idx_i, idx_j in pair2comp:
            comp = ','.join([ str(c) for c in pair2comp[(idx_i, idx_j)] ])
            fields = [ genes[idx_i], genes[idx_j], comp ]
            of.write('\t'.join([ str(field) for field in fields ]) + '\n')

            if len(pair2comp[(idx_i, idx_j)]) == 1:
                uniq_links[list(pair2comp[(idx_i, idx_j)])[0]].add(genes[idx_i])
                uniq_links[list(pair2comp[(idx_i, idx_j)])[0]].add(genes[idx_j])

    for comp in uniq_links:
        with open('{}/uniq_link_genes_{}.txt'
                  .format(dirname, comp), 'w') as of:
            of.write('\n'.join(sorted(uniq_links[comp])))

