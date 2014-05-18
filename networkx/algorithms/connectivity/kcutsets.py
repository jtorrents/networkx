# -*- coding: utf-8 -*-
"""
Kanevsky all minimum k cutsets
"""
from operator import itemgetter

import networkx as nx
from .utils import build_auxiliary_node_connectivity
# Define the default maximum flow function.
from networkx.algorithms.flow import edmonds_karp
from networkx.algorithms.flow import shortest_augmenting_path
from networkx.algorithms.flow.utils import build_residual_network
default_flow_func = edmonds_karp


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
    cuts : a generator of frozensets
        A set of node cutsets as frozensets. Each cutset has cardinality 
        equal to the node connectivity of the input graph.

    Examples
    --------
    >>> # A two-dimensional grid graph has 4 cutsets of cardinality 2
    >>> G = nx.grid_2d_graph(5, 5)
    >>> cutsets = set(nx.k_cutsets(G))
    >>> len(cutsets)
    4
    >>> all(2 == len(cutset) for cutset in cutsets)
    True
    >>> nx.node_connectivity(G)
    2

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
    # Initialize data structures.
    # Even-Tarjan reduction is what we call auxiliary digraph 
    # for node connectivity.
    H = build_auxiliary_node_connectivity(G)
    mapping = H.graph['mapping']
    R = build_residual_network(H, 'capacity')
    kwargs = dict(capacity='capacity', residual=R)
    # Define default flow function
    if flow_func is None:
        flow_func = default_flow_func
    if flow_func is shortest_augmenting_path:
        kwargs['two_phase'] = True
    # Begin the actual algorithm
    # step 1: Find node connectivity k of G
    if k is None:
        k = nx.node_connectivity(G, flow_func=flow_func)
    # step 2: 
    # Find k nodes with top degree, call it X:
    X = frozenset(n for n, deg in
            sorted(G.degree().items(), key=itemgetter(1), reverse=True)[:k])
    # Check if X is a k-node-cutset
    if is_separating_set(G, X):
        yield X

    for x in X:
        # step 3: Compute local connectivity flow of x with all other
        # non adjacent nodes in G
        non_adjacent = (n for n in set(G) - X if n not in G[x])
        for v in non_adjacent:
            # step 4: compute maximum flow in an Even-Tarjan reduction H of G
            # and step:5 build the associated residual network R
            R = flow_func(H, '%sB' % mapping[x], '%sA' % mapping[v], **kwargs)
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
                # step 7: Compute antichains of L; they map to closed sets in H
                # Any edge in H that links a closed set is part of a cutset
                antichains = antichain_generator(L)

                found = False
                while not found:
                    antichain = next(antichains, None)
                    if antichain is None:
                        break
                    # Nodes in an antichain of the condensation graph of
                    # the residual network map to a closed set of nodes that
                    # define a node partition of the auxiliary digraph H.
                    S = set(n for n, scc_id in cmap.items() if scc_id in antichain)
                    # Find the cutset that links the node partition (S,~S) in H
                    cutset = set()
                    for u, nbrs in ((n, H[n]) for n in S):
                        cutset.update((u, w) for w in nbrs if w not in S)
                    # The edges in H that form the cutset are internal edges
                    # (ie edges that represent a node of the original graph G)
                    node_cut = set(H.node[n]['id'] for edge in cutset for n in edge)

                    if len(node_cut) == k:
                        yield frozenset(node_cut)
                        # Add an edge (x, v) to make sure that we do not
                        # find this cutset again. This is equivalent
                        # of adding the edge in the input graph 
                        # G.add_edge(x, v) and then regenerate H and R:
                        # Add edges to the auxiliary digraph.
                        H.add_edge('%sB' % mapping[x], '%sA' % mapping[v],
                                   capacity=1)
                        H.add_edge('%sB' % mapping[v], '%sA' % mapping[x],
                                   capacity=1)
                        # Add edges to the residual network.
                        R.add_edge('%sB' % mapping[x], '%sA' % mapping[v],
                                   capacity=1)
                        R.add_edge('%sA' % mapping[v], '%sB' % mapping[x],
                                   capacity=1)
                        found = True
                # Add again the saturated edges to reuse the residual network
                R.add_edges_from(saturated_edges)


def transitive_closure(G):
    """Based on http://www.ics.uci.edu/~eppstein/PADS/PartialOrder.py"""
    TC = nx.DiGraph()
    TC.add_nodes_from(G.nodes_iter())
    TC.add_edges_from(G.edges_iter())
    for v in G:
        TC.add_edges_from((v, u) for u in nx.dfs_preorder_nodes(G, source=v)
                          if v != u)
    return TC

   
def antichain_generator(G):
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

    if len(cut) == len(G) - 1:
        return True

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
