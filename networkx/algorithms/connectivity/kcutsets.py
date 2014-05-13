# -*- coding: utf-8 -*-
"""
Kanevsky all minimum k cutsets
"""
from operator import itemgetter
from copy import deepcopy
import networkx as nx
from .utils import build_auxiliary_node_connectivity
from networkx.algorithms.flow.utils import build_residual_network

__author__ = '\n'.join(['Jordi Torrents <jtorrents@milnou.net>'])

__all__ = ['k_cutsets']

def k_cutsets(G, k=None, flow_func=None):
    r"""Returns all minimum k cutsets of an undirected graph G. 

    This implementation is based on Kanevsky's algorithm [1] for finding all
    minimum-size node cut-sets of an undirected graph G; ie the set (or sets) 
    of nodes of cardinality equal to the node connectivity of G. Thus if 
    removed, would break G into two or more connected components.
   
    Parameters
    ----------
    G : NetworkX graph
        Undirected graph

    k : Integer (default=None)
        Node connectivity of the input graph. If k is None, then it is computed.

    Returns
    -------
    cuts : set of frozensets
        A set of node cutsets as frozensets. Each cutset has cardinality 
        equal to the node connectivity of the input graph.

    Examples
    --------
    >>> # A two-dimensional grid graph has 4 cutsets of cardinality 2
    >>> # Notice that nodes in a grid_2d_graph are tuples
    >>> G = nx.grid_2d_graph(5, 5)
    >>> nx.node_connectivity(G)
    2
    >>> # XXX If the output of nx.k_cutsets is formated nicly, the doctest fails!
    >>> nx.k_cutsets(G)
    set([frozenset([(0, 1), (1, 0)]), frozenset([(3, 0), (4, 1)]), frozenset([(3, 4), (4, 3)]), frozenset([(0, 3), (1, 4)])])

    Notes
    -----
    This implementation is based on the sequential algorithm for finding all
    minimum-size separating vertex sets in a graph [1]. The main idea is to
    compute minimum cuts using local maximum flow computations among a set 
    of nodes of highest degree and all other non-adjacent nodes in the Graph.
    Once we find a minimum cut, we add an edge between the high degree
    node and the target node of the local maximum flow computation to make 
    sure that we will not find that minimum cut again.

    The time complexity of this implmentation is not the same that the one
    reported in [1] because we always use the `ford-fulkerson` algorithm
    for st-maximum flow computations.

    See also
    --------
    node_connectivity
    k_components
    generate_partitions
    ford_fulkerson

    References
    ----------
    .. [1]  Kanevsky, A. (1993). Finding all minimum-size separating vertex 
            sets in a graph. Networks 23(6), 533--541.
            http://onlinelibrary.wiley.com/doi/10.1002/net.3230230604/abstract

    """
    # Some initial checks to save time when k = 0, 1 or n-1 
    if G.order() == 0 or not nx.is_connected(G):
        return set()
    #elif not nx.is_biconnected(G):
    #    return set(frozenset([a]) for a in nx.articulation_points(G))
    elif nx.density(G) == 1:
        node = next(G.nodes_iter())
        return set(frozenset(set(G)-set([node])))
    # Initialize data structures.
    directed = G.is_directed()
    # Even-Tarjan reduction is what we call auxiliary digraph 
    # for node connectivity.
    H = build_auxiliary_node_connectivity(G)
    mapping = H.graph['mapping']
    R = build_residual_network(H, 'capacity')
    # Define default flow function
    if flow_func is None:
        flow_func = nx.edmonds_karp
    # Begin the actual algorithm
    all_cuts = set()
    # step 1: Find node connectivity k of G
    if k is None:
        k = nx.node_connectivity(G)
    # step 2: 
    # Find k nodes with top degree, call it X:
    X = frozenset(n for n, deg in
            sorted(G.degree().items(), key=itemgetter(1), reverse=True)[:k])
    # Check if their are a k-cutset
    if is_separating_set(G, X):
        all_cuts.add(X)
    # For all x in X
    for x in X:
        # step 3: Compute local connectivity flow of x with all other
        # non adjacent nodes in G
        for v in (n for n in set(G) - set(X) if n not in G[x]):
            # step 4: compute maximum flow in an Even-Tarjan reduction H of G
            # and step:5 build the associated residual network R
            R = flow_func(H, '%sB' % mapping[x], '%sA' % mapping[v],
                          capacity="capacity", residual=R)
            flow_value = R.graph['flow_value']

            if flow_value == k:
                ## Remove saturated edges form the residual network
                saturated_edges = [(u, w, d) for u, w, d in R.edges(data=True)
                                   if d['capacity'] == d['flow']]
                R.remove_edges_from(saturated_edges)
                # step 6: shrink the strongly connected components of 
                # residual flow network R and call it L
                scc=nx.strongly_connected_components(R)
                L, cmap = my_condensation(R, scc, mapping=True)
                # step 7: Compute antichains of L; they map to closed sets in R
                # Any edge in R that links a closed set is part of a cutset
                antichains = compute_antichains(L)
                i = 0
                while i < k:
                    antichain = next(antichains, None)
                    if antichain is None:
                        break
                    for node in antichain:
                        this_cut = []
                        # For all nodes of each closed set of the residual graph
                        S = set(n for n, scc in cmap.items() if scc == node)
                        no_S = set(H) - set(S)
                        # Check if they have neighbors among other nodes in R
                        # XXX This is nicer and faster but reports wrong results
                        # for karate test and others. Not sure why.
                        #cutset = set()
                        #for u, nbrs in ((n, R[n]) for n in S):
                        #    cutset.update((u, w) for w in nbrs if w in no_S)
                        #for u, w in cutset:
                        #    # has to be internal edge in the ET-reduction
                        #    if H.node[w]['id'] == H.node[u]['id']:
                        #        this_cut.append(H.node[w]['id'])
                        for u in S:
                            for w in no_S:
                                if w in R[u] or u in R[w]:
                                    # has to be internal edge in the ET-reduction
                                    if H.node[w]['id'] == H.node[u]['id']:
                                        this_cut.append(H.node[w]['id'])

                        if len(this_cut) == k:
                            all_cuts.add(frozenset(this_cut))
                            # Add an edge (x, v) to make sure that we do not
                            # find this cutset again. This is equivalent
                            # of adding the edge in the input graph 
                            # G.add_edge(x, v) and then regenerate H and R.
                            H.add_edge('%sB' % mapping[x], '%sA' % mapping[v],
                                       capacity=1)
                            if not directed:
                                H.add_edge('%sB' % mapping[v], '%sA' % mapping[x],
                                           capacity=1)
                            # Add edge to the residual network.
                            R.add_edge('%sB' % mapping[x], '%sA' % mapping[v],
                                       capacity=1)
                            R.add_edge('%sA' % mapping[v], '%sB' % mapping[x],
                                       capacity=1)
                            i += 1
                # Add again the saturated edges to reuse the residual network
                R.add_edges_from(saturated_edges)
    return all_cuts


def transitive_closure(G):
    """Based on http://www.ics.uci.edu/~eppstein/PADS/PartialOrder.py"""
    TC = G.copy()
    for v in G:
        TC.add_edges_from((v, u) for u in nx.dfs_preorder_nodes(G, source=v)
                          if v != u)
    return TC

   
def compute_antichains(G):
    # Based on SAGE combinat.posets.hasse_diagram.py
    TC = transitive_closure(G)
    #print nx.to_numpy_matrix(TC)
    antichains_queues = [([], nx.topological_sort(G, reverse=True))]
    while antichains_queues:
        (antichain, queue) = antichains_queues.pop()
        # Invariant:
        #  - the elements of antichain are independent
        #  - the elements of queue are independent from those of antichain
        yield antichain
        while queue:
            x = queue.pop()
            new_antichain = antichain + [x]
            new_queue = [t for t in queue if not ((t in TC[x]) or (x in TC[t]))]
            antichains_queues.append((new_antichain, new_queue))


def my_condensation(G, scc, mapping=False):
    """Returns the condensation of G.

    The condensation of G is the graph with each of the strongly connected 
    components contracted into a single node.  

    Parameters
    ----------
    G : NetworkX DiGraph
       A directed graph.

    scc:  list
       A list of strongly connected components.  
       Use scc=nx.strongly_connected_components(G) to compute the components.

    Returns
    -------
    C : NetworkX DiGraph
       The condensation of G. The node labels are integers corresponding
       to the index of the component in the list of strongly connected 
       components.

    Notes
    -----
    After contracting all strongly connected components to a single node,
    the resulting graph is a directed acyclic graph.  
    """
    mapping = {}
    C = nx.DiGraph()
    scc = list(scc)
    for i,component in enumerate(scc):
        for n in component:
            mapping[n] = i
    C.add_nodes_from(range(len(scc)))
    for u,v in G.edges():
        if mapping[u] != mapping[v]:
            C.add_edge(mapping[u],mapping[v])
    if mapping:
        return C, mapping
    return C

def is_separating_set(G, cut):
    if not nx.is_connected(G):
        raise nx.NetworkXError('Input graph is disconnected')

    H = G.copy()
    H.remove_nodes_from(cut)
    if nx.is_connected(H):
        return False
    return True
    
def _is_trivial(G):
    if G.order() <= 2:
        return True
    else:
        return False
