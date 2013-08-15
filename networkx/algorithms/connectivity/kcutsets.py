# -*- coding: utf-8 -*-
"""
Kanevsky all minimum k cutsets
"""
from operator import itemgetter
from copy import deepcopy
import networkx as nx
from networkx.algorithms.connectivity.connectivity import \
    _aux_digraph_node_connectivity, node_connectivity

__author__ = '\n'.join(['Jordi Torrents <jtorrents@milnou.net>'])

__all__ = ['k_cutsets']

def k_cutsets(F, k=None):
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
    if F.order() == 0 or not nx.is_connected(F):
        return set()
    elif not nx.is_biconnected(F):
        return set(frozenset([a]) for a in nx.articulation_points(F))
    elif nx.density(F) == 1:
        node = next(F.nodes_iter())
        return set(frozenset(set(F)-set([node])))
    # Begin the actual algorithm
    # We have to modify the input graph, so we make a copy
    G = F.copy()
    all_cuts = set()
    # step 1: Find node connectivity k of G
    if k is None:
        k = node_connectivity(G)
    # step 2: 
    # Find k nodes with top degree, call it X:
    X = frozenset(n for n,deg in \
                sorted(G.degree().items(), key=itemgetter(1), reverse=True)[:k])
    # Check if their are a k-cutset
    if is_separating_set(G, X):
        all_cuts.add(X)
    # For all x in X
    for x in X:
        # step 3: Compute local connectivity flow of x with all other
        # non adjacent nodes in G
        for v in set(G) - set(X):
            if v in G[x]: continue
            # Even-Tarjan reduction
            R, mapping = _aux_digraph_node_connectivity(G)
            # step 4: compute maximum flow in an Even-Tarjan reduction R of G
            # and step:5 the associated residual network H
            #flow, H = nx.ford_fulkerson(R, '%sB'%mapping[x], '%sA'%mapping[v],
            #                                capacity="capacity", residual=True)
            flow, H = nx.ford_fulkerson_flow_and_auxiliary(R, '%sB'%mapping[x], 
                                                            '%sA'%mapping[v],
                                                            capacity="capacity")
            if flow == k:
                # step 6: shrink the strongly connected components of 
                # residual flow network H and call it L
                scc=nx.strongly_connected_components(H)
                L, cmap = my_condensation(H, scc, mapping=True)
                # step 7: Compute antichains of L; they map to closed sets in H
                # Any edge in H that links a closed set is part of a cutset
                for antichain in antichains(L):
                    for node in antichain:
                        this_cut = []
                        # For all nodes of each closed set of the residual graph
                        S = set(n for n, scc in cmap.items() if scc==node)
                        for u in S:
                            # Check if they have neighbors among other nodes in H
                            for w in set(H) - set(S):
                                if w in H[u] or u in H[w]:
                                    # has to be internal edge in the ET-reduction
                                    if R.node[w]['id'] == R.node[u]['id']:
                                        this_cut.append(R.node[w]['id'])
                        if len(this_cut) == k:
                            all_cuts.add(frozenset(this_cut))
                            # Add an edge to make sure that 
                            # we do not find this cutset again
                            G.add_edge(x, v)
    return all_cuts

def antichains(F):
    # Based on SAGE combinat.posets.hasse_diagram.py
    def _helper(A, node, T, antichains):
        Ap = A + [node]
        antichains.append(Ap)
        if T:
            _helper(A, T[0], T[1:], antichains)
            Tp = []
            for t in T:
                if not t in G[node] and not node in G[t]:
                    Tp.append(t)
            if Tp:
                _helper(Ap, Tp[0], Tp[1:], antichains)
        else:
            return antichains
    G = F.copy()
    # add edges from every node to each of its successors
    for node in G:
        for successor in nx.dfs_tree(G, node):
            G.add_edge(node, successor)
    #print nx.to_numpy_matrix(G)
    # initial antichains
    antichains = [[]]
    # initial starting point
    node = next(G.nodes_iter())
    # Other nodes in G
    T = list(set(G)-set([node]))
    _helper([], node, T, antichains)
    return antichains

def new_antichains(F):
    G = F.copy()
    # add edges from every node to each of its successors
    seen = set()
    for node in G:
        for successor in nx.bfs_tree(F, node):
            G.add_edge(node, successor)
    #print nx.to_numpy_matrix(G)
    antichains_queues = [([], list(reversed(G.nodes())))]
    while antichains_queues:
        (antichain, queue) = antichains_queues.pop()
        # Invariant:
        #  - the elements of antichain are independent
        #  - the elements of queue are independent from those of antichain
        yield antichain
        while queue:
            x = queue.pop()
            new_antichain = antichain + [x]
            new_queue = [t for t in queue if not ((t in G[x]) or (x in G[t]))]
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
