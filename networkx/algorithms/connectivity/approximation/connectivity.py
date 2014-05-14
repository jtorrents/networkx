""" Fast approximation for node independent paths and node connectivity
"""
#    Copyright (C) 2013 by 
#    Jordi Torrents <jtorrents@milnou.net>
#    All rights reserved.
#    BSD license.
import itertools
import collections
import networkx as nx

__author__ = """\n""".join(['Jordi Torrents <jtorrents@milnou.net>'])

__all__ = ['node_independent_paths',
           'local_node_connectivity',
           'node_connectivity',
           'single_source_node_connectivity',
           'all_pairs_node_connectivity']

def node_independent_paths(G, source, target, cutoff=None):
    """Return node independent paths between two nodes

    Node independent paths or disjoint node paths are paths between two nodes
    that that share no nodes in common other than their starting and ending
    nodes. 

    This algorithm is a fast approximation that gives an strict lower bound on
    the actual number of node independent paths between two nodes [1]. It works 
    for both directed and undirected graphs.

    Parameters
    ----------

    G : NetworkX graph

    source : node
       Starting node for node independent paths

    target : node
       Ending node for node independent paths

    cutoff : integer (optional)
        Maximum number of paths to consider

    Returns
    -------
    paths: list
       List of node independent paths between source and target

    Examples
    --------
    >>> # Platonic icosahedral graph has node connectivity 5 
    >>> # for each non adjacent node pair. Thus each pair has also
    >>> # 5 node independent paths among them
    >>> from networkx.algorithms.connectivity import approximation as approx
    >>> G = nx.icosahedral_graph()
    >>> print(approx.node_independent_paths(G,0,6))
    [[0, 1, 6], [0, 5, 6], [0, 8, 2, 6], [0, 11, 4, 6], [0, 7, 9, 3, 6]]

    Notes 
    -----
    This algorithm _[1] finds node independents paths between two nodes by 
    computing their shortest path using BFS, marking the nodes of the path 
    found as 'used' and then searching other shortest paths excluding the 
    nodes marked as used until no more paths exist. This method is a fast
    approximation for the actual number of node independent paths between
    two nodes.

    Note that the authors propose a further refinement, losing accuracy and 
    gaining speed, which is not implemented (yet).

    See also
    --------
    local_node_connectivity

    References
    ----------
    .. [1] White, Douglas R., and Mark Newman. 2001 A Fast Algorithm for 
        Node-Independent Paths. Santa Fe Institute Working Paper #01-07-035
        http://eclectic.ss.uci.edu/~drwhite/working.pdf
 
    """
    # Maximum possible node independent paths
    if G.is_directed():
        possible = min(G.out_degree(source), G.in_degree(target))
    else:
        possible = min(G.degree(source), G.degree(target))
    
    if cutoff is None:
        cutoff = float('inf')

    paths=[]
    
    if target == source:
        return [] # Maybe we should return something else here
    elif possible == 0:
        return []
    
    exclude = set()
    for i in range(min(possible, cutoff)):
        try:
            path = bidirectional_shortest_path(G, source, target, exclude)
            exclude.update(set(path))
            paths.append(path)
        except nx.NetworkXNoPath:
            break

    return paths

def local_node_connectivity(G, source, target, cutoff=None, strict=False):
    """Compute node connectivity between source and target.
    
    Pairwise or local node connectivity between two distinct and nonadjacent 
    nodes is the minimum number of nodes that must be removed (minimum 
    separating cutset) to disconnect them. By Merger's theorem, this is equal 
    to the number of node independent paths (paths that share no nodes other
    than source and target). Which is what we compute in this function.

    This algorithm is a fast approximation that gives an strict lower
    bound on the actual number of node independent paths between two nodes [1]. 
    It works for both directed and undirected graphs.

    For adjacent nodes, node connectivity is not defined, but if strict=False
    the number of node independent paths between them is returned.

    Parameters
    ----------

    G : NetworkX graph

    source : node
        Starting node for node connectivity

    target : node
        Ending node for node connectivity

    cutoff : integer, float (optional)
        Maximum number of paths to consider

    strict : bolean (default=False)
        If True return float('nan') for adjacent nodes

    Returns
    -------
    k: integer
       pairwise node connectivity

    Examples
    --------
    >>> # Platonic icosahedral graph has node connectivity 5 
    >>> # for each non adjacent node pair
    >>> from networkx.algorithms.connectivity import approximation as approx
    >>> G = nx.icosahedral_graph()
    >>> approx.local_node_connectivity(G,0,6)
    5

    Notes 
    -----
    This algorithm _[1] finds node independents paths between two nodes by 
    computing their shortest path using BFS, marking the nodes of the path 
    found as 'used' and then searching other shortest paths excluding the 
    nodes marked as used until no more paths exist.

    Note that the authors propose a further refinement, losing accuracy and 
    gaining speed, which is not implemented yet.

    See also
    --------
    node_independent_paths
    single_source_node_connectivity
    all_pairs_node_connectivity

    References
    ----------
    .. [1] White, Douglas R., and Mark Newman. 2001 A Fast Algorithm for 
        Node-Independent Paths. Santa Fe Institute Working Paper #01-07-035
        http://eclectic.ss.uci.edu/~drwhite/working.pdf
 
    """
    # Maximum possible node independent paths
    if G.is_directed():
        possible = min(G.out_degree(source), G.in_degree(target))
    else:
        possible = min(G.degree(source), G.degree(target))
    
    if cutoff is None:
        cutoff = float('Inf')

    K = 0
    
    if target == source:
        return None
    elif possible == 0:
        return 0
    elif strict and target in G[source]:
        return float('nan')
    
    exclude = set()
    for i in range(min(possible, cutoff)):
        try:
            path = bidirectional_shortest_path(G,source,target,exclude=exclude)
            exclude.update(set(path))
            K += 1
        except nx.NetworkXNoPath:
            break

    return K

def node_connectivity(G):
    r"""Returns an approximation for node connectivity for a graph or digraph G.

    Global node connectivity is the minimum number of nodes that 
    must be removed to disconnect G or render it trivial. By Merger's theorem, 
    this is equal to the number of node independent paths (paths that 
    share no nodes other than source and target). Which is what we compute 
    in this function.

    This algorithm is based on a fast approximation that gives an strict lower
    bound on the actual number of node independent paths between two nodes [1]. 
    It works for both directed and undirected graphs.
   
    Parameters
    ----------
    G : NetworkX graph
        Undirected graph

    Returns
    -------
    K : integer
        global node connectivity for G

    Examples
    --------
    >>> # Platonic icosahedral graph is 5-node-connected 
    >>> from networkx.algorithms.connectivity import approximation as approx
    >>> G = nx.icosahedral_graph()
    >>> approx.node_connectivity(G)
    5
    
    Notes
    -----
    This algorithm _[1] finds node independents paths between two nodes by 
    computing their shortest path using BFS, marking the nodes of the path 
    found as 'used' and then searching other shortest paths excluding the 
    nodes marked as used until no more paths exist.

    See also
    --------
    node_connectivity
    local_node_connectivity
    edge_connectivity
    local_edge_connectivity

    References
    ----------
    .. [1] White, Douglas R., and Mark Newman. 2001 A Fast Algorithm for 
        Node-Independent Paths. Santa Fe Institute Working Paper #01-07-035
        http://eclectic.ss.uci.edu/~drwhite/working.pdf

    """
    if G.is_directed():
        if not nx.is_weakly_connected(G):
            return 0
        iter_func = itertools.permutations
        def neighbors(v):
            return itertools.chain.from_iterable([G.predecessors_iter(v),
                                                  G.successors_iter(v)])
    else:
        if not nx.is_connected(G):
            return 0
        iter_func = itertools.combinations
        neighbors=G.neighbors_iter

    # Choose a node with minimum degree
    deg = G.degree()
    minimum_degree = min(deg.values())
    v = next(n for n,d in deg.items() if d==minimum_degree)
    # Node connectivity is bounded by minimum degree
    K = minimum_degree
    # compute local node connectivity with all non-neighbors nodes
    # and store the minimum
    for w in set(G)-set(neighbors(v))-set([v]):
        K = min(K, local_node_connectivity(G, v, w, cutoff=K))
    # Same for non adjacent pairs of neighbors of v
    for x,y in iter_func(neighbors(v), 2):
            if y not in G[x]:
                K = min(K, local_node_connectivity(G, x, y, cutoff=K))
    return K

def single_source_node_connectivity(G, source, cutoff=None, strict=False):
    """Compute pairwise node connectivity between source
    and all other nodes reachable from source.

    Pairwise or local node connectivity between two distinct and nonadjacent 
    nodes is the minimum number of nodes that must be removed (minimum 
    separating cutset) to disconnect them. By Merger's theorem, this is equal 
    to the number of node independent paths (paths that share no nodes other
    than source and target). Which is what we compute in this function.

    This algorithm is a fast approximation that gives an strict lower
    bound on the actual number of node independent paths between two nodes [1]. 
    It works for both directed and undirected graphs.

    For adjacent nodes, node connectivity is not defined, but if strict=False
    the number of node independent paths between them is returned, 
    following [1].

    Parameters
    ----------
    G : NetworkX graph

    source : node
       Starting node for node independent paths

    cutoff : integer (optional)
        Maximum number of paths to consider

    strict : bolean (default=False)
        If True return float('nan') for adjacent nodes

    Returns
    -------
    K : dictionary
        Dictionary, keyed by target, of pairwise node connectivity.

    See Also
    --------
    node_independent_paths
    local_node_connectivity
    all_pairs_node_connectivity
 
    References
    ----------
    .. [1] White, Douglas R., and Mark Newman. 2001 A Fast Algorithm for 
        Node-Independent Paths. Santa Fe Institute Working Paper #01-07-035
        http://eclectic.ss.uci.edu/~drwhite/working.pdf
    """
    K = {}
    for target in G:
        if target == source: continue
        K[target] = local_node_connectivity(G, source, target, 
                                                cutoff=cutoff, strict=strict)

    return K

def all_pairs_node_connectivity(G, cutoff=None, strict=False):
    """ Compute node connectivity between all pairs of nodes.

    Pairwise or local node connectivity between two distinct and nonadjacent 
    nodes is the minimum number of nodes that must be removed (minimum 
    separating cutset) to disconnect them. By Merger's theorem, this is equal 
    to the number of node independent paths (paths that share no nodes other
    than source and target). Which is what we compute in this function.

    This algorithm is a fast approximation that gives an strict lower
    bound on the actual number of node independent paths between two nodes [1]. 
    It works for both directed and undirected graphs.

    For adjacent nodes, node connectivity is not defined, but if strict=False
    the number of node independent paths between them is returned, 
    following [1].

    Parameters
    ----------
    G : NetworkX graph

    cutoff : integer (optional)
        Maximum number of paths to consider

    strict : bolean (default=False)
        If True return float('nan') for adjacent nodes

    Returns
    -------
    K : dictionary
        Dictionary, keyed by source and target, of pairwise node connectivity

    See Also
    --------
    local_node_connectivity
    all_pairs_node_connectivity
    node_independent_paths

    References
    ----------
    .. [1] White, Douglas R., and Mark Newman. 2001 A Fast Algorithm for 
        Node-Independent Paths. Santa Fe Institute Working Paper #01-07-035
        http://eclectic.ss.uci.edu/~drwhite/working.pdf
    """
    K = collections.defaultdict(dict)
    if G.is_directed():
        for n in G:
            K[n] = single_source_node_connectivity(G, n, cutoff=cutoff,
                                                    strict=strict)
    else:
        for u, v in itertools.combinations(G, 2):
            this_k = local_node_connectivity(G, u, v, cutoff=cutoff,
                                                strict=strict)
            K[u][v] = K[v][u] = this_k

    return dict(K)

def bidirectional_shortest_path(G, source, target, exclude=None):
    """Return shortest path between source and target ignoring nodes in the
    container 'exclude'.

    Parameters
    ----------

    G : NetworkX graph

    source : node
        Starting node for path

    target : node
        Ending node for path

    exclude: container, iterable (optional)
        Container for nodes to exclude from the search for shortest paths

    Returns
    -------
    path: list
        Shortest path between source and target ignoring nodes in 'exclude'

    Raises
    ------
    NetworkXNoPath: exception
        If there is no path or if nodes are adjacent and have only one path
        between them

    Notes
    -----
    This function and its helper are inspired in the algorithm proposed in [1]
    to find node independent paths. They are originaly from
    networkx.algorithms.shortest_paths.unweighted and are only modified to 
    accept the extra parameter 'exclude', which is a container for nodes 
    already used in other paths that should be ignored.

    References
    ----------
    .. [1] White, Douglas R., and Mark Newman. 2001 A Fast Algorithm for 
        Node-Independent Paths. Santa Fe Institute Working Paper #01-07-035
        http://eclectic.ss.uci.edu/~drwhite/working.pdf
    
    """
    # call helper to do the real work
    results=_bidirectional_pred_succ(G, source, target, exclude=exclude)
    pred,succ,w=results

    # build path from pred+w+succ
    path=[]
    # from source to w
    while w is not None:
        path.append(w)
        w = pred[w]
    path.reverse()
    # from w to target
    w = succ[path[-1]]
    while w is not None:
        path.append(w)
        w=succ[w]

    return path

def _bidirectional_pred_succ(G, source, target, exclude=None):
    # does BFS from both source and target and meets in the middle
    # excludes nodes in the container "exclude" from the search
    if source is None or target is None:
        raise nx.NetworkXException(\
            "Bidirectional shortest path called without source or target")
    if target == source:
        return ({target:None},{source:None},source)

    if exclude is None:
        exclude = set()
    
    # handle either directed or undirected
    if G.is_directed():
        Gpred=G.predecessors_iter
        Gsucc=G.successors_iter
    else:
        Gpred=G.neighbors_iter
        Gsucc=G.neighbors_iter

    # predecesssor and successors in search
    pred={source:None}
    succ={target:None}

    # initialize fringes, start with forward
    forward_fringe=[source]
    reverse_fringe=[target]

    level = 0
    
    while forward_fringe and reverse_fringe:
        # Make sure that we iterate one step forward and one step backwards
        # thus source and target will only tigger "found path" when they are
        # adjacent and then they can be safely included in the container 'exclude'
        level += 1
        if not level % 2 == 0:
            this_level=forward_fringe
            forward_fringe=[]
            for v in this_level:
                for w in Gsucc(v):
                    if w in exclude:
                        continue
                    if w not in pred:
                        forward_fringe.append(w)
                        pred[w]=v
                    if w in succ:
                        return pred,succ,w # found path
        else:
            this_level=reverse_fringe
            reverse_fringe=[]
            for v in this_level:
                for w in Gpred(v):
                    if w in exclude:
                        continue
                    if w not in succ:
                        succ[w]=v
                        reverse_fringe.append(w)
                    if w in pred: 
                        return pred,succ,w # found path

    raise nx.NetworkXNoPath("No path between %s and %s." % (source, target))
