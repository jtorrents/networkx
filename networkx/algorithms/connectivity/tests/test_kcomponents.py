# Test for Moody and White k-components algorithm
from nose.tools import assert_equal, assert_true
import networkx as nx
from networkx.algorithms.connectivity import build_k_number_dict

##
## Some nice synthetic graphs
##
def graph_example_1():
    G = nx.convert_node_labels_to_integers(nx.grid_graph([5,5]),
                                            label_attribute='labels')
    rlabels = nx.get_node_attributes(G, 'labels')
    labels = dict((v, k) for k, v in rlabels.items())

    for nodes in [(labels[(0,0)], labels[(1,0)]),
                    (labels[(0,4)], labels[(1,4)]),
                    (labels[(3,0)], labels[(4,0)]),
                    (labels[(3,4)], labels[(4,4)]) ]:
        new_node = G.order()+1
        # Petersen graph is triconnected
        P = nx.petersen_graph()
        G = nx.disjoint_union(G,P)
        # Add two edges between the grid and P
        G.add_edge(new_node+1, nodes[0])
        G.add_edge(new_node, nodes[1])
        # K5 is 4-connected
        K = nx.complete_graph(5)
        G = nx.disjoint_union(G,K)
        # Add three edges between P and K5
        G.add_edge(new_node+2,new_node+11)
        G.add_edge(new_node+3,new_node+12)
        G.add_edge(new_node+4,new_node+13)
        # Add another K5 sharing a node
        G = nx.disjoint_union(G,K)
        nbrs = G[new_node+10]
        G.remove_node(new_node+10)
        for nbr in nbrs:
            G.add_edge(new_node+17, nbr)
        G.add_edge(new_node+16, new_node+5)

    G.name = 'Example graph for connectivity'
    return G

def torrents_and_ferraro_graph():
    G = nx.convert_node_labels_to_integers(nx.grid_graph([5,5]),
                                            label_attribute='labels')
    rlabels = nx.get_node_attributes(G, 'labels')
    labels = dict((v, k) for k, v in rlabels.items())

    for nodes in [ (labels[(0,4)], labels[(1,4)]),
                    (labels[(3,4)], labels[(4,4)]) ]:
        new_node = G.order()+1
        # Petersen graph is triconnected
        P = nx.petersen_graph()
        G = nx.disjoint_union(G,P)
        # Add two edges between the grid and P
        G.add_edge(new_node+1, nodes[0])
        G.add_edge(new_node, nodes[1])
        # K5 is 4-connected
        K = nx.complete_graph(5)
        G = nx.disjoint_union(G,K)
        # Add three edges between P and K5
        G.add_edge(new_node+2,new_node+11)
        G.add_edge(new_node+3,new_node+12)
        G.add_edge(new_node+4,new_node+13)
        # Add another K5 sharing a node
        G = nx.disjoint_union(G,K)
        nbrs = G[new_node+10]
        G.remove_node(new_node+10)
        for nbr in nbrs:
            G.add_edge(new_node+17, nbr)
        # Commenting this makes the graph not biconnected !!
        # This stupid mistake make one reviewer very angry :P
        G.add_edge(new_node+16, new_node+8)

    for nodes in [(labels[(0,0)], labels[(1,0)]),
                    (labels[(3,0)], labels[(4,0)])]:
        new_node = G.order()+1
        # Petersen graph is triconnected
        P = nx.petersen_graph()
        G = nx.disjoint_union(G,P)
        # Add two edges between the grid and P
        G.add_edge(new_node+1, nodes[0])
        G.add_edge(new_node, nodes[1])
        # K5 is 4-connected
        K = nx.complete_graph(5)
        G = nx.disjoint_union(G,K)
        # Add three edges between P and K5
        G.add_edge(new_node+2,new_node+11)
        G.add_edge(new_node+3,new_node+12)
        G.add_edge(new_node+4,new_node+13)
        # Add another K5 sharing two nodes
        G = nx.disjoint_union(G,K)
        nbrs = G[new_node+10]
        G.remove_node(new_node+10)
        for nbr in nbrs:
            G.add_edge(new_node+17, nbr)
        nbrs2 = G[new_node+9]
        G.remove_node(new_node+9)
        for nbr in nbrs2:
            G.add_edge(new_node+18, nbr)

    G.name = 'Example graph for connectivity'
    return G

# Helper function
def _check_connectivity(G):
    result = nx.k_components(G)
    for k, components in result.items():
        if k < 3:
            continue
        for component in components:
            C = G.subgraph(component)
            K = nx.node_connectivity(C)
            assert_true(K >= k)

def test_torrents_and_ferraro_graph():
    G = torrents_and_ferraro_graph()
    _check_connectivity(G)

def test_example_1():
    G = graph_example_1()
    _check_connectivity(G)

def test_random_gnp():
    G = nx.gnp_random_graph(50, 0.2)
    _check_connectivity(G)

def test_shell():
    constructor=[(20,80,0.8),(80,180,0.6)]
    G = nx.random_shell_graph(constructor)
    _check_connectivity(G)

def test_configuration():
    deg_seq = nx.utils.create_degree_sequence(100,nx.utils.powerlaw_sequence)
    G = nx.Graph(nx.configuration_model(deg_seq))
    G.remove_edges_from(G.selfloop_edges())
    _check_connectivity(G)

def test_karate_0():
    G = nx.karate_club_graph()
    _check_connectivity(G)

def test_karate_1():
    karate_k_num = {0: 4, 1: 4, 2: 4, 3: 4, 4: 3, 5: 3, 6: 3, 7: 4, 8: 4, 9: 2,
                    10: 3, 11: 1, 12: 2, 13: 4, 14: 2, 15: 2, 16: 2, 17: 2, 18: 2,
                    19: 3, 20: 2, 21: 2, 22: 2, 23: 3, 24: 3, 25: 3, 26: 2, 27: 3,
                    28: 3, 29: 3, 30: 4, 31: 3, 32: 4, 33: 4}
    G = nx.karate_club_graph()
    k_components = nx.k_components(G)
    k_num = build_k_number_dict(k_components)
    assert_equal(karate_k_num, k_num)

def test_example_1_detail_3_and_4():
    solution = {
        3: [set([40, 41, 42, 43, 39]),
            set([32, 33, 34, 35, 36, 37, 38, 42, 25, 26, 27, 28, 29, 30, 31]),
            set([58, 59, 60, 61, 62]),
            set([44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 61]),
            set([80, 81, 77, 78, 79]),
            set([64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 80, 63]),
            set([97, 98, 99, 100, 101]),
            set([96, 100, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 94, 95])
        ],
        4: [set([40, 41, 42, 43, 39]),
            set([42, 35, 36, 37, 38]),
            set([58, 59, 60, 61, 62]),
            set([56, 57, 61, 54, 55]),
            set([80, 81, 77, 78, 79]),
            set([80, 73, 74, 75, 76]),
            set([97, 98, 99, 100, 101]),
            set([96, 100, 92, 94, 95])
        ],
    }
    G = graph_example_1()
    result = nx.k_components(G, pretty_output=True)
    for k, components in solution.items():
        for component in components:
            assert_true(component in result[k])

def test_davis_southern_women():
    solution = {
        3: [set(['Nora Fayette',
                'E10',
                'Myra Liddel',
                'E12',
                'E14',
                'Frances Anderson',
                'Evelyn Jefferson',
                'Ruth DeSand',
                'Helen Lloyd',
                'Eleanor Nye',
                'E9',
                'E8',
                'E5',
                'E4',
                'E7',
                'E6',
                'E1',
                'Verne Sanderson',
                'E3',
                'E2',
                'Theresa Anderson',
                'Pearl Oglethorpe',
                'Katherina Rogers',
                'Brenda Rogers',
                'E13',
                'Charlotte McDowd',
                'Sylvia Avondale',
                'Laura Mandeville'])
            ],
        4: [set(['Nora Fayette',
                'E10',
                'Verne Sanderson',
                'E12',
                'Frances Anderson',
                'Evelyn Jefferson',
                'Ruth DeSand',
                'Helen Lloyd',
                'Eleanor Nye',
                'E9',
                'E8',
                'E5',
                'E4',
                'E7',
                'E6',
                'Myra Liddel',
                'E3',
                'Theresa Anderson',
                'Katherina Rogers',
                'Brenda Rogers',
                'Charlotte McDowd',
                'Sylvia Avondale',
                'Laura Mandeville']),
        ]
    }
    G = nx.davis_southern_women_graph()
    result = nx.k_components(G)
    for k, components in solution.items():
        for component in components:
            assert_true(component in result[k])


