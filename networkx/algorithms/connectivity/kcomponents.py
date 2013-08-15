# -*- coding: utf-8 -*-
"""
Moody and White algorithm for k-components
"""
import collections
from operator import itemgetter
import networkx as nx

__author__ = '\n'.join(['Jordi Torrents <jtorrents@milnou.net>'])

__all__ = [ 'k_components',
            'cohesive_blocks']

def generate_partitions(G, cuts):
    def has_nbrs_in_partition(G, node, partition):
        for n in G[node]:
            if n in partition:
                return True
        else:
            return False
    if nx.density(G) == 1:
        # We cannot partition a complete subgraph
        yield nx.Graph()
    else:
        nodes = set(G) - set(n for cut in cuts for n in cut)
        H = G.subgraph(nodes)
        for cc in nx.connected_components(H):
            component = set(cc)
            for cut in cuts:
                for node in cut:
                    if has_nbrs_in_partition(G, node, cc):
                        component.add(node)
            if len(component) != G.order():
                yield component

def k_components(G, verbose=False):
    r"""Returns the k-component structure of a graph G.
    
    A `k`-component is a maximal subgraph of a graph G that has, at least, 
    node connectivity `k`: we need to remove at least `k` nodes to break it
    into more components. `k`-components have an inherent hierarchical
    structure because they are nested in terms of connectivity: a connected 
    graph can contain several 2-components, each of which can contain 
    one or more 3-components,  and so forth.

    This is an implementation of Moody and White [1] algorithm. 
   
    Parameters
    ----------
    G : NetworkX graph
        Undirected graph

    verbose: Bolean (default=False)
        Provide 

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
    >>> G = nx.petersen_graph()
    >>> k_components, k_number = nx.k_components(G)
    >>> for k, components in k_components.items():
    ...     print(k, [len(component) for component in components])
    (1, [10])
    (2, [10])
    (3, [10])
    
    Notes
    -----
    Moody and White [1] (appendix A) provide an algorithm for identifying 
    k-components in a graph, which is based on Kanevsky's algorithm [2] 
    for finding all minimum-size node cut-sets of a graph:

        1. Compute node connectivity, k, of the input graph G.

        2. Identify all k-cutsets at the current level of connectivity using  
            Kanevsky's algorithm. 

        3. Generate new graph components based on the removal of 
            these cutsets, Nodes in a cutset belong to both sides 
            of the induced cut.

        4. If the graph is neither complete nor trivial, return to 1; else end.

    See also
    --------
    node_connectivity
    k_cutsets
    generate_partitions
    cohesive_blocks
    approximation.k_components

    References
    ----------
    .. [1]  Moody, J. and D. White (2003). Social cohesion and embeddedness: 
            A hierarchical conception of social groups. 
            American Sociological Review 68(1), 103--28.
            http://www2.asanet.org/journals/ASRFeb03MoodyWhite.pdf

    .. [2]  Kanevsky, A. (1993). Finding all minimum-size separating vertex 
            sets in a graph. Networks 23(6), 533--541.
            http://onlinelibrary.wiley.com/doi/10.1002/net.3230230604/abstract

    """
    def _update_results(k, components):
        for component in components:
            if len(component) > k:
                my_comp = set(component)
                k_components[k].append(my_comp)
                for node in component:
                    if k_number[node] < k:
                        k_number[node] = k
    ## Data structures to return results
    # Dictionary with connectivity level (k) as keys and a list of
    # sets of nodes that form a k-component as values
    k_components = collections.defaultdict(list)
    # Dictionary with nodes as keys and maximum k of the deepest 
    # k-component in which they are embedded
    k_number = dict( ((n,0) for n in G) )
    # Bicomponents as a base to check for higher order k-components
    components = nx.connected_components(G)
    _update_results(1, components)
    bicomponents = nx.biconnected_components(G)
    _update_results(2, bicomponents)
    for bicomponent in bicomponents:
        B = G.subgraph(bicomponent)
        k = nx.node_connectivity(B)
        cuts = nx.k_cutsets(B, k)
        if k == len(B)-1:
            # complete subgraph
            _update_results(k, [B.nodes()])
            continue
        elif k > 2:
            _update_results(k, [B.nodes()])
        # Perform a DFS like  
        stack = [(k, generate_partitions(B, cuts))]
        while stack:
            (parent_k, partitions) = stack[-1]
            try:
                partition = next(partitions)
                C = nx.k_core(B.subgraph(partition), parent_k)
                if not _is_trivial(C):
                    this_k = nx.node_connectivity(C)
                    cuts = nx.k_cutsets(C, this_k)
                    if this_k > parent_k:
                        _update_results(this_k, [C.nodes()])
                    if cuts:
                        stack.append((  max(this_k, parent_k),
                                        generate_partitions(C, cuts)))
                    if verbose:
                        msg = "Subgraph: {0} nodes {1} edges, K={2} with {3} cuts"
                        print(msg.format(C.order(), C.size(), this_k, len(cuts)))
            except StopIteration:
                stack.pop()

    return k_components, k_number

def cohesive_blocks(k_components):
    r"""Returns a tree T representing the k-component structure for a graph G. 

    As proposed by White et al. [1], we can represent the `k`-component 
    structure of a graph by drawing a tree whose nodes are `k`-components, 
    and two nodes are linked if the `k`-component of higher level is 
    nested inside the `k`-component of lower level.
   
    Parameters
    ----------
    k_components : dict
        A dictionary with connectivity levels as keys representing the
        `k`-component structure of a Graph. This is the output of 
        `k_components` function.

    Returns
    -------
    T : NetworkX DiGraph
        A tree that represents the k-component structure of the input graph.

    Examples
    --------
    >>> G = nx.davis_southern_women_graph()
    >>> k_components, k_number = nx.k_components(G)
    >>> T = nx.cohesive_blocks(k_components)
    >>> # Cohesive block tree is well suited for the dot layout algorithm
    >>> pos = nx.graphviz_layout(T, prog='dot')
    
    Notes
    -----
    The root node of connectivity tree T contains all nodes in G. 
    If it is connected, the first level is k=1, if not the first level 
    is k=0. The nodes of the tree have an attribute `order` with the 
    nuber of nodes of the `k`-component that they represent. The arcs
    of T have an attribute `weight` with the order of the target node.
    This ensures a vertical alingment among giant k-components when 
    using the dot layout algrithm for drawing T.

    See also
    --------
    node_connectivity
    k_cutsets
    cohesive_blocks
    approximation.k_components

    References
    ----------
    .. [1]  White, D., J. Owen-Smith, J. Moody, and W. Powell (2004).
            Networks, fields and organizations: micro-dynamics,
            scale and cohesive embeddings. Computational & Mathematical
            Organization Theory 10(1), 95--117.
            http://www-personal.umich.edu/~jdos/pdfs/CMOT.pdf
    """
    T = nx.DiGraph()
    # The root node of connectivity tree contains all nodes in G. 
    # If it is connected, the first level is k=1, if not k=0
    if len(k_components[1]) > 1:
        T.add_node('0', k=0, order=sum([len(cc) for avg,cc in k_components[1]]))
        for i, (avg,component) in enumerate(k_components[1]):
            if len(component) < 5: continue
            node = '1_%s' % i
            T.add_node(node, k=1, order=len(component))
            T.add_edge('0', node, weight=len(component))
    else:
        T.add_node('1_0', k=1, order=len(k_components[1][0]))

    for k, components in k_components.items():
        if k == 1: continue
        for i, (avg,component) in enumerate(components):
            component = set(component)
            if len(component) < 5: continue
            node = '%s_%s' % (k, i)
            T.add_node(node, k=k, order=len(component))
            max_overlap = 0
            pred = None
            for j, (avg,predecessors) in enumerate(k_components[k-1]):
                intersection = len(component & set(predecessors))
                if intersection >= max_overlap:
                    max_overlap = intersection
                    pred = '%s_%s' % (k-1, j)
            if pred is not None and max_overlap >= k:
                if pred in T:
                    T.add_edge(pred, node, weight=len(component))
    return T

def _is_trivial(G):
    if G.order() <= 2:
        return True
    else:
        return False
