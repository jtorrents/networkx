""" Fast approximation for k-component structure
"""
#    Copyright (C) 2013 by 
#    Jordi Torrents <jtorrents@milnou.net>
#    All rights reserved.
#    BSD license.
import itertools
import collections

import networkx as nx
from networkx.algorithms.connectivity.approximation import local_node_connectivity
from networkx.classes.antigraph import AntiGraph

__author__ = """\n""".join(['Jordi Torrents <jtorrents@milnou.net>'])

__all__ = ['k_components']

def k_components(G, average=True, 
                        exact=False,
                        store_comp=True, 
                        store_nip=True, 
                        return_nip=False,
                        min_density=0.95):
    r"""Returns the k-component structure of a graph G using a fast approximation.
    
    A `k`-component is a maximal subgraph of a graph G that has, at least, 
    node connectivity `k`: we need to remove at least `k` nodes to break it
    into more components. `k`-components have an inherent hierarchical
    structure because they are nested in terms of connectivity: a connected 
    graph can contain several 2-components, each of which can contain 
    one or more 3-components, and so forth.

    This implementation is based on a fast approximation algorithm to compute
    the `k`-component sturcture of a graph [1]. Which, in turn, it is based on
    a fast approximation algorithm for finding good lower bounds of the number 
    of node independent paths between two nodes [2]
    (see `approximation.local_node_connectivity`) for our implementation.
  
    Parameters
    ----------
    G : NetworkX graph
        Undirected graph

    average : Boolean (default=True)
        Compute also the average connectivity of each `k`-component

    store_comp : Boolean (default=True)
        If True return the sets of nodes that form each `k`-component. If False
        return only the order of each `k`-component

    store_nip : Boolean (default=True)
        If True use a dictionary to store the local node connectivity among all
        pairs of nodes. This speeds up the computation of average node
        connectivity but requires more memory

    return_nip : Boolean (default=False)
        If True return also the dictionary with the local node connectivity
        among all pairs of nodes.

    order : Integer (default=105)
        Order below which we consider node independent paths among adjacent
        nodes (see Notes below)

    min_density : Float (default=0.95)
        Density relaxation treshold.

    Returns
    -------
    k_components : dict
        Dictionary with connectivity level `k` as key and a list of
        sets of nodes that form a k-component of level `k` as values.
        
    k_number : dict
        Dictionary with nodes as keys with value of the maximum k of the 
        deepest k-component in which they are embedded.

    nip : dict (optional, only when return_nip is True)
        Dictionary with the local node connectivity (ie the number of node
        independent paths) among all pairs of nodes

    Examples
    --------
    >>> # Petersen graph has 10 nodes and it is triconnected, thus all 
    >>> # nodes are in a single component on all three connectivity levels
    >>> from networkx.algorithms.connectivity import approximation as approx
    >>> G = nx.petersen_graph()
    >>> k_components, k_number = approx.k_components(G)
    >>> for k, components in k_components.items():
    ...     print(k, [len(component) for avg_k, component in components])
    (1, [10])
    (2, [10])
    (3, [10])
    
    Notes
    -----
    The logic of the approximation algorithm for computing the `k`-component 
    structure [1] is based on repeatedly applying simple and fast algorithms 
    for `k`-cores and biconnected components in order to narrow down the 
    number of pairs of nodes over which we have to compute White and Newman's
    approximation algorithm for finding node independent paths [2]. More
    formally, this algorithm is based on Whitney's theorem, which states 
    an inclusion relation among node connectivity, edge connectivity, and 
    minimum degree for any graph G. This theorem implies that every 
    `k`-component is nested inside a `k`-edge-component, which in turn, 
    is contained in a `k`-core. Thus, this algorithm computes node independent
    paths among pairs of nodes in each biconnected part of each `k`-core,
    and repeats this procedure for each `k` from 3 to the maximal core number 
    of a node in the input graph.

    Because, in practice, many nodes of the core of level `k` inside a 
    bicomponent actually are part of a component of level k, the auxiliary 
    graph needed for the algorithm is likely to be very dense. Thus, we use 
    a complement graph data structure (see `AntiGraph`) to save memory. 
    AntiGraph only stores information of the edges that are *not* present 
    in the actual auxiliary graph. When applying algorithms to this 
    complement graph data structure, it behaves as if it were the dense 
    version.

    See also
    --------
    approximation.local_node_connectivity
    approximation.k_components_approx_accuracy
    cohesive_blocks
    k_components
    AntiGraph

    References
    ----------
    .. [1]  Torrents, J. and F. Ferraro. Structural Cohesion: a theoretical 
            extension and a fast approximation algorithm. Draft
            http://www.milnou.net/~jtorrents/structural_cohesion.pdf

    .. [2]  White, Douglas R., and Mark Newman. 2001 A Fast Algorithm for 
            Node-Independent Paths. Santa Fe Institute Working Paper #01-07-035
            http://eclectic.ss.uci.edu/~drwhite/working.pdf

    .. [3]  Moody, J. and D. White (2003). Social cohesion and embeddedness: 
            A hierarchical conception of social groups. 
            American Sociological Review 68(1), 103--28.
            http://www2.asanet.org/journals/ASRFeb03MoodyWhite.pdf

    """
    ## Data structures for results
    # Dictionary with connectivity level (k) as keys and a list of
    # sets of nodes that form a k-component as values
    k_components = collections.defaultdict(list)
    # Dictionary with nodes as keys and maximum k of the deepest 
    # k-component in which they are embedded
    k_number = dict.fromkeys(G, 0)
    # dict to store node independent paths
    if store_nip: nip = {} 
    #################
    def _update_results(k, components, avg_k=None):
        avg = True if avg_k is not None else False
        for component in components:
            if len(component) > k:
                if store_comp:
                    if avg: k_components[k].append((avg_k, set(component)))
                    else: k_components[k].append(set(component))
                else:
                    if avg: k_components[k].append((avg_k, len(component)))
                    else: k_components[k].append(len(component))
                for node in component:
                    if avg: k_number[node] = (k, avg_k)
                    else: k_number[node] = k
    # make a few functions local for speed
    if exact:
        node_connectivity = nx.local_node_connectivity
    else:
        node_connectivity = local_node_connectivity
    k_core = nx.k_core
    core_number = nx.core_number
    biconnected_components = nx.biconnected_components
    density = nx.density
    combinations = itertools.combinations
    # Exact solution for k = {1,2}
    # There is a linear time algorithm for triconnectivity, if we had an
    # implementation available we could start from k = 4.
    if average:
        _update_results(1, nx.connected_components(G), 1)
        _update_results(2, biconnected_components(G), 2)
    else:
        _update_results(1, nx.connected_components(G))
        _update_results(2, biconnected_components(G))
    # There is no k-component of k > maximum core number
    # \kappa(G) <= \lambda(G) <= \delta(G)
    g_cnum = core_number(G)
    max_core = max(g_cnum.values())
    for k in range(3, max_core + 1):
        C = k_core(G, k, core_number=g_cnum)
        for nodes in biconnected_components(C):
            # Build a subgraph SG induced by the nodes that are part of
            # each biconnected component of the k-core subgraph C.
            if len(nodes) < k:
                continue
            SG = G.subgraph(nodes)
            # Build auxiliary graph
            H = AntiGraph()
            H.add_nodes_from(SG.nodes_iter())
            for u,v in combinations(SG, 2):
                if average and store_nip:
                    K = node_connectivity(SG, u, v)
                    nip[(u,v)] = K
                elif not exact:
                    K = node_connectivity(SG, u, v, cutoff=k)
                else:
                    K = node_connectivity(SG, u, v)
                if k > K:
                    H.add_edge(u,v)
            for h_nodes in biconnected_components(H):
                if len(h_nodes) <= k:
                    continue
                HS = H.subgraph(h_nodes)
                h_cnum = core_number(HS)
                first = True
                for c_value in sorted(set(h_cnum.values()),reverse=True):
                    cands = set(n for n, cnum in h_cnum.items() if cnum == c_value)
                    # Skip checking for overlap for the highest core value
                    if first:
                        overlap = False
                        first = False
                    else:
                        overlap = set.intersection(*[
                                    set(x for x in HS[n] if x not in cands) 
                                    for n in cands])
                    if overlap and len(overlap) < k:
                        Hc = HS.subgraph(cands | overlap)
                    else:
                        Hc = HS.subgraph(cands)
                    if len(Hc) <= k:
                        continue
                    hc_core = core_number(Hc)
                    if _same(hc_core) and density(Hc) == 1.0:
                        Gc = k_core(SG.subgraph(Hc), k)
                    else:
                        while Hc:
                            Gc = k_core(SG.subgraph(Hc), k)
                            Hc = HS.subgraph(Gc)
                            if not Hc:
                                continue
                            hc_core = core_number(Hc)
                            if _same(hc_core) and density(Hc) >= min_density:
                                break
                            hc_deg = Hc.degree()
                            min_deg = min(hc_deg.values())
                            remove = [n for n, d in hc_deg.items() if d == min_deg]
                            Hc.remove_nodes_from(remove)
                    if not Hc or len(Gc) <= k:
                        continue
                    for k_component in biconnected_components(Gc):
                        if len(k_component) <= k:
                            continue
                        Gk = k_core(SG.subgraph(k_component), k)
                        if average:
                            num = 0.0
                            den = 0.0
                            for u,v in combinations(Gk, 2):
                                den += 1
                                if store_nip:
                                    num += (nip[(u,v)] if (u,v) in nip 
                                            else nip[(v,u)])
                                else:
                                    num += node_connectivity(Gk, u, v)
                            _update_results(k, [Gk.nodes()], (num/den))
                        else:
                            _update_results(k, [Gk.nodes()])
    if return_nip and store_nip:
        return k_components, k_number, nip
    return k_components, k_number

def _same(measure, tol=0):
    vals = set(measure.values())
    if (max(vals) - min(vals)) <= tol:
        return True
    return False

