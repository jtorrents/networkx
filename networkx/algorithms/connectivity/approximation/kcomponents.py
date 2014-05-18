""" Fast approximation for k-component structure
"""
#    Copyright (C) 2014 by 
#    Jordi Torrents <jtorrents@milnou.net>
#    All rights reserved.
#    BSD license.
import itertools
import collections

import networkx as nx
from networkx.algorithms.connectivity.approximation import local_node_connectivity
from networkx.algorithms.connectivity import local_node_connectivity as exact_local_node_connectivity
from networkx.algorithms.connectivity import build_auxiliary_node_connectivity
from networkx.algorithms.flow.utils import build_residual_network

from networkx.classes.antigraph import AntiGraph

__author__ = """\n""".join(['Jordi Torrents <jtorrents@milnou.net>'])

__all__ = ['k_components_average', 'k_components', 'build_k_number_dict']

def k_components_average(G, exact=False, store_nip=True, min_density=0.95):
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
    >>> from networkx.algorithms.connectivity import approximation as apxa
    >>> G = nx.petersen_graph()
    >>> k_components = apxa.k_components_average(G)
    
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
    # dict to store node independent paths
    if store_nip: nip = {}
    if exact:
        node_connectivity = exact_local_node_connectivity
        min_density = 1.0
        A = build_auxiliary_node_connectivity(G)
        R = build_residual_network(A, 'capacity')
    else:
        node_connectivity = local_node_connectivity
    # make a few functions local for speed
    k_core = nx.k_core
    core_number = nx.core_number
    biconnected_components = nx.biconnected_components
    density = nx.density
    combinations = itertools.combinations
    # Exact solution for k = {1,2}
    # There is a linear time algorithm for triconnectivity, if we had an
    # implementation available we could start from k = 4.
    for component in  nx.connected_components(G):
        # isolated nodes have connectivity 0
        comp = set(component)
        if len(comp) > 1:
            k_components[1].append((1, comp))
    for bicomponent in  nx.biconnected_components(G):
        # avoid considering dyads as bicomponents
        bicomp = set(bicomponent)
        if len(bicomp) > 2:
            k_components[2].append((2, bicomp))
    # There is no k-component of k > maximum core number
    # \kappa(G) <= \lambda(G) <= \delta(G)
    g_cnumber = core_number(G)
    max_core = max(g_cnumber.values())
    for k in range(3, max_core + 1):
        C = k_core(G, k, core_number=g_cnumber)
        for nodes in biconnected_components(C):
            # Build a subgraph SG induced by the nodes that are part of
            # each biconnected component of the k-core subgraph C.
            if len(nodes) < k:
                continue
            SG = G.subgraph(nodes)
            if exact:
                ar_nodes = [n for n, d in A.nodes(data=True) if d['id'] in nodes]
                SA = A.subgraph(ar_nodes)
                SR = R.subgraph(ar_nodes)
                kwargs = dict(auxiliary=SA, residual=SR)
            else:
                kwargs = dict()
            # Build auxiliary graph
            H = AntiGraph()
            H.add_nodes_from(SG.nodes_iter())
            for u,v in combinations(SG, 2):
                if store_nip:
                    K = node_connectivity(SG, u, v, **kwargs)
                    nip[(u,v)] = K
                else:
                    kwargs['cutoff'] = k
                    K = node_connectivity(SG, u, v, **kwargs)
                if k > K:
                    H.add_edge(u,v)
            for h_nodes in biconnected_components(H):
                if len(h_nodes) <= k:
                    continue
                SH = H.subgraph(h_nodes)
                for Gc in cliques_heuristic(SG, SH, k, min_density):
                    for k_nodes in biconnected_components(Gc):
                        Gk = nx.k_core(SG.subgraph(k_nodes), k)
                        if len(Gk) <= k:
                            continue
                        num = 0.0
                        den = 0.0
                        for u, v in combinations(Gk, 2):
                            den += 1
                            if store_nip:
                                num += nip.get((u, v), nip.get((v, u)))
                            else:
                                num += node_connectivity(Gk, u, v)
                        k_components[k].append((num/den, set(Gk)))
    return k_components


def k_components(G, exact=False, min_density=0.95):
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

    Examples
    --------
    >>> # Petersen graph has 10 nodes and it is triconnected, thus all 
    >>> # nodes are in a single component on all three connectivity levels
    >>> from networkx.algorithms.connectivity import approximation as apxa
    >>> G = nx.petersen_graph()
    >>> k_components = apxa.k_components(G)
    
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
    # Dictionary with connectivity level (k) as keys and a list of
    # sets of nodes that form a k-component as values
    k_components = collections.defaultdict(list)
    if exact:
        node_connectivity = exact_local_node_connectivity
        min_density = 1.0
        A = build_auxiliary_node_connectivity(G)
        R = build_residual_network(A, 'capacity')
    else:
        node_connectivity = local_node_connectivity
    # make a few functions local for speed
    k_core = nx.k_core
    core_number = nx.core_number
    biconnected_components = nx.biconnected_components
    density = nx.density
    combinations = itertools.combinations
    # Exact solution for k = {1,2}
    # There is a linear time algorithm for triconnectivity, if we had an
    # implementation available we could start from k = 4.
    for component in  nx.connected_components(G):
        # isolated nodes have connectivity 0
        comp = set(component)
        if len(comp) > 1:
            k_components[1].append(comp)
    for bicomponent in  nx.biconnected_components(G):
        # avoid considering dyads as bicomponents
        bicomp = set(bicomponent)
        if len(bicomp) > 2:
            k_components[2].append(bicomp)
    # There is no k-component of k > maximum core number
    # \kappa(G) <= \lambda(G) <= \delta(G)
    g_cnumber = core_number(G)
    max_core = max(g_cnumber.values())
    for k in range(3, max_core + 1):
        C = k_core(G, k, core_number=g_cnumber)
        for nodes in biconnected_components(C):
            # Build a subgraph SG induced by the nodes that are part of
            # each biconnected component of the k-core subgraph C.
            if len(nodes) < k:
                continue
            SG = G.subgraph(nodes)
            if exact:
                ar_nodes = [n for n, d in A.nodes(data=True) if d['id'] in nodes]
                SA = A.subgraph(ar_nodes)
                SR = R.subgraph(ar_nodes)
                kwargs = dict(auxiliary=SA, residual=SR)
            else:
                kwargs = dict()
            # Build auxiliary graph
            H = AntiGraph()
            H.add_nodes_from(SG.nodes_iter())
            for u,v in combinations(SG, 2):
                kwargs['cutoff'] = k
                K = node_connectivity(SG, u, v, **kwargs)
                if k > K:
                    H.add_edge(u,v)
            for h_nodes in biconnected_components(H):
                if len(h_nodes) <= k:
                    continue
                SH = H.subgraph(h_nodes)
                for Gc in cliques_heuristic(SG, SH, k, min_density):
                    for k_nodes in biconnected_components(Gc):
                        Gk = nx.k_core(SG.subgraph(k_nodes), k)
                        if len(Gk) <= k:
                            continue
                        k_components[k].append(set(Gk))
    return k_components


def cliques_heuristic(G, H, k, min_density):
    h_cnumber = nx.core_number(H)
    for i, c_value in enumerate(sorted(set(h_cnumber.values()), reverse=True)):
        cands = set(n for n, c in h_cnumber.items() if c == c_value)
        # Skip checking for overlap for the highest core value
        if i == 0:
            overlap = False
        else:
            overlap = set.intersection(*[
                        set(x for x in H[n] if x not in cands)
                        for n in cands])
        if overlap and len(overlap) < k:
            SH = H.subgraph(cands | overlap)
        else:
            SH = H.subgraph(cands)
        sh_cnumber = nx.core_number(SH)
        SG = nx.k_core(G.subgraph(SH), k)
        while not (_same(sh_cnumber) and nx.density(SH) >= min_density):
            SH = H.subgraph(SG)
            if len(SH) <= k:
                break
            sh_cnumber = nx.core_number(SH)
            sh_deg = SH.degree()
            min_deg = min(sh_deg.values())
            SH.remove_nodes_from(n for n, d in sh_deg.items() if d == min_deg)
            SG = nx.k_core(G.subgraph(SH), k)
        else:
            yield SG


def build_k_number_dict(k_components):
    k_num = {}
    for k, comps in sorted(k_components.items()):
        for comp in comps:
            for node in comp:
                k_num[node] = k
    return k_num


def _same(measure, tol=0):
    vals = set(measure.values())
    if (max(vals) - min(vals)) <= tol:
        return True
    return False

# Helper functions
def _check_connectivity(G, G_k):
    for k, components in G_k.items():
        if k < 3:
            continue
        for component in components:
            if isinstance(component, tuple):
                component = component[1]
            C = G.subgraph(component)
            K = nx.node_connectivity(C)
            try:
                assert K >= k
            except:
                msg = "error in {0}-components with {1} and node connectivity {2}"
                print(msg.format(k, len(component), K))

def print_connectivity(G_k):
    for k, comps in G_k.items():
        print("Connectivity {0}".format(k))
        for comp in comps:
            if isinstance(comp, tuple):
                print("    {0}-component ({1}): {2} nodes".format(k, comp[0], len(comp[1])))
            else:
                print("    {0}-component: {1} nodes".format(k, len(comp)))

