"""Community detection and quality functions for bipartite graphs."""

from collections import defaultdict

import networkx as nx
from networkx.algorithms.community.community_utils import is_partition
from networkx.algorithms.community.quality import NotAPartition
from networkx.utils.decorators import not_implemented_for, py_random_state

__all__ = ["modularity", "lpawb_plus_communities"]


@not_implemented_for("directed")
@nx._dispatchable(name="bipartite_modularity", edge_attrs="weight")
def modularity(G, communities, nodes, weight="weight", resolution=1):
    r"""Returns Barber's bipartite modularity of the given partition.

    Bipartite modularity [1]_ adapts Newman's modularity to bipartite
    networks by replacing the configuration-model null model with one
    that respects the bipartite structure: expected edges only occur
    between the two node sets. For a bipartite graph with "red" nodes
    (one set) and "blue" nodes (the other), it is defined as

    .. math::

        Q_B = \frac{1}{m} \sum_{v=1}^{r} \sum_{w=1}^{c}
              \left( \tilde{A}_{vw} - \gamma\frac{k_v d_w}{m} \right)
              \delta(c_v, c_w)

    where $m$ is the (weighted) number of edges, $\tilde{A}$ is the
    bipartite incidence matrix, $k_v$ is the (weighted) degree of red
    node $v$, $d_w$ is the (weighted) degree of blue node $w$, $\gamma$
    is the resolution parameter, and $\delta(c_v, c_w)$ is 1 if $v$ and
    $w$ are in the same community, else 0.

    Following Clauset, Newman and Moore [2]_, this can be rewritten as a
    sum over communities:

    .. math::

        Q_B = \sum_{c=1}^{n}
              \left[ \frac{L_c}{m} - \gamma\,\frac{k_c d_c}{m^2} \right]

    where $L_c$ is the (weighted) number of intra-community edges for
    community $c$, $k_c$ is the sum of degrees of the red nodes in $c$,
    and $d_c$ is the sum of degrees of the blue nodes in $c$. This is
    the form used in the implementation.

    Note the structural analogy with the standard (unipartite) sum
    formulation `L_c/m - gamma * (k_c / 2m)^2`: the null-model term is
    a product of two degree sums, and in the undirected unipartite case
    those two sums coincide so the product becomes a square. In the
    bipartite case the sums differ (red vs. blue), so the null model
    term remains a product of two distinct terms.

    Communities in a bipartite modularity partition may contain nodes
    from both bipartite sets; the bipartite structure enters only
    through the null model.

    Parameters
    ----------
    G : NetworkX Graph
        An undirected bipartite graph.

    communities : list or iterable of sets of nodes
        These node sets must represent a partition of `G`'s nodes.
        Communities may contain nodes from both bipartite sets.

    nodes : container of nodes
        A container with all nodes in **one** bipartite node set (the
        "red" nodes). The other set is inferred as ``set(G) - set(nodes)``.
        This follows the convention used throughout the bipartite
        subpackage.

    weight : string or None, optional (default="weight")
        The edge attribute that holds the numerical value used as the
        edge weight. If None or if an edge does not have the attribute,
        that edge is treated as having weight 1.

    resolution : float (default=1)
        Resolution parameter $\gamma$. Values smaller than 1 favor
        larger communities; values greater than 1 favor smaller
        communities. Not part of Barber's original definition, but a
        standard extension consistent with
        :func:`networkx.community.modularity`.

    Returns
    -------
    Q_B : float
        The bipartite modularity of the partition.

    Raises
    ------
    NotAPartition
        If `communities` is not a valid partition of the nodes of `G`.

    NetworkXNotImplemented
        If `G` is directed. Barber's formulation is for undirected
        bipartite graphs.

    Examples
    --------
    Two disconnected :math:`K_{2,2}` components form a strong bipartite
    community structure:

    >>> from networkx.algorithms import bipartite
    >>> G = nx.Graph([(0, 2), (1, 3)])
    >>> red = {0, 1}
    >>> bipartite.modularity(G, [{0, 2}, {1, 3}], red)
    0.5

    Notes
    -----
    The functions in the bipartite package do not check that `nodes` is
    actually one side of a bipartition of `G`. It is the caller's
    responsibility to provide a valid bipartite graph and a valid
    bipartite node set.

    See Also
    --------
    networkx.algorithms.community.quality.modularity

    References
    ----------
    .. [1] Barber, M. J. (2007). "Modularity and community detection in
       bipartite networks." Physical Review E, 76(6), 066102.
       https://doi.org/10.1103/PhysRevE.76.066102
    .. [2] Clauset, A., Newman, M. E. J., and Moore, C. (2004).
       "Finding community structure in very large networks."
       Physical Review E, 70(6), 066111.
    """
    if not isinstance(communities, list):
        communities = list(communities)
    if not is_partition(G, communities):
        raise NotAPartition(G, communities)

    red = set(nodes)
    blue = set(G) - red

    degree = dict(G.degree(weight=weight))
    # Every edge has exactly one red and one blue endpoint, so the sum of
    # red degrees equals the total (weighted) number of edges m.
    m = sum(degree[v] for v in red)

    if m == 0:
        return 0.0

    norm = 1 / m**2

    def community_contribution(community):
        comm = set(community)
        comm_red = comm & red
        comm_blue = comm & blue

        # L_c: (weighted) internal edges. Iterate over red side only so
        # each bipartite edge is visited exactly once.
        L_c = sum(
            wt
            for u in comm_red
            for _, v, wt in G.edges(u, data=weight, default=1)
            if v in comm_blue
        )

        k_c = sum(degree[u] for u in comm_red)
        d_c = sum(degree[u] for u in comm_blue)

        return L_c / m - resolution * k_c * d_c * norm

    return sum(community_contribution(c) for c in communities)


@not_implemented_for("directed")
@py_random_state("seed")
@nx._dispatchable(name="lpawb_plus_communities", edge_attrs="weight")
def lpawb_plus_communities(G, nodes, weight="weight", seed=None):
    r"""Find communities in a bipartite graph using LPAwb+.

    LPAwb+ (Beckett, 2016) [1]_ is a two-stage algorithm that maximizes
    Barber's weighted bipartite modularity $Q_B$ [2]_:

    .. math::

        Q_B = \sum_{c} \left[\frac{L_c}{m} - \frac{k_c d_c}{m^2}\right]

    where $m$ is the total (weighted) number of edges, $L_c$ is the
    (weighted) internal edge count of community $c$, and $k_c$, $d_c$
    are the degree sums of the two bipartite sets within $c$. See
    :func:`modularity` for details.

    The algorithm proceeds as follows. Each node starts in its own
    community.

    **Stage 1 (label propagation):** blue-side and red-side labels are
    alternately updated in random order; each node picks the community
    label that maximizes its local contribution to $Q_B$ using the
    update rule of Beckett Eq. 2.5. For a red node $x$ the best label
    $g$ is

    .. math::

        g_x^\text{new} = \operatorname*{argmax}_g
            \left( N_{xg} - \frac{y_x Z_g}{M} \right)

    where $N_{xg}$ is the sum of edge weights from $x$ to blue nodes
    with label $g$, $y_x$ is the strength of $x$, $Z_g$ is the total
    strength of blue nodes with label $g$, and $M$ is the total edge
    weight. Blue nodes use the symmetric rule. Ties (including ties
    with the node's current label) are broken in favor of keeping the
    current label, which guarantees monotonic improvement. Stage 1
    repeats until no label changes.

    **Stage 2 (agglomeration):** each existing module looks for its
    "best partner" --- the other module whose merger gives the largest
    positive increase in $Q_B$. If a pair of modules is mutually each
    other's best partner and merging them strictly increases $Q_B$,
    they are merged. Stage 1 is then re-run on the coarsened partition.
    The outer loop repeats until no improving merge is found.

    The merge delta for two modules $A$ and $B$ is

    .. math::

        \Delta Q_B = \frac{e_{AB}}{m}
            - \frac{k_A d_B + k_B d_A}{m^2}

    where $e_{AB}$ is the (weighted) number of edges between nodes
    labelled $A$ and nodes labelled $B$, and $k_\cdot$, $d_\cdot$ are
    per-module red/blue degree sums.

    Parameters
    ----------
    G : NetworkX Graph
        An undirected bipartite graph. May be weighted.

    nodes : container of nodes
        A container with all the nodes in one bipartite set. The other
        set is inferred as ``set(G) - set(nodes)``. Following the
        bipartite subpackage convention, the caller is responsible for
        providing a valid bipartite graph and a valid node set.

    weight : string or None, optional (default="weight")
        Edge attribute holding the numerical weight. If ``None`` or the
        attribute is missing, every edge is treated as having weight 1.

    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`. LPAwb+ is stochastic (random
        shuffle order and random tie-breaking at initialization); run
        with multiple seeds and keep the best :func:`modularity` value
        for small networks.

    Returns
    -------
    communities : list of sets
        A list of sets of nodes, one per community, sorted by size
        (largest first). Communities may contain nodes from both
        bipartite sets.

    Raises
    ------
    NetworkXNotImplemented
        If `G` is directed.

    Notes
    -----
    Per the paper, the smaller of the two bipartite sets is treated as
    the "red" side internally for computational efficiency; this does
    not affect the returned communities, which are unordered node
    sets.

    LPAwb+ is local and heuristic: it converges to a local optimum of
    $Q_B$ and is not guaranteed to find the global maximum. Running
    with several seeds and keeping the best-scoring partition is the
    standard practice.

    See Also
    --------
    modularity

    References
    ----------
    .. [1] Beckett, S. J. (2016). "Improved community detection in
       weighted bipartite networks." Royal Society open science, 3(1),
       140536. https://doi.org/10.1098/rsos.140536
    .. [2] Barber, M. J. (2007). "Modularity and community detection in
       bipartite networks." Physical Review E, 76(6), 066102.
    """
    red = set(nodes)
    blue = set(G) - red

    # Per paper, treat the smaller side as "red" internally.
    if len(red) > len(blue):
        red, blue = blue, red

    if G.number_of_edges() == 0 or not red or not blue:
        return [{n} for n in G]

    strength = dict(G.degree(weight=weight))
    # Total edge weight = sum of red-side strengths (each bipartite
    # edge has exactly one red endpoint).
    M = sum(strength[v] for v in red)

    if M == 0:
        return [{n} for n in G]

    # Start with every node in its own community.
    labels = {v: i for i, v in enumerate(G)}
    label_strength_red = defaultdict(float)
    label_strength_blue = defaultdict(float)
    for v in red:
        label_strength_red[labels[v]] = strength[v]
    for v in blue:
        label_strength_blue[labels[v]] = strength[v]

    red_list = list(red)
    blue_list = list(blue)

    def _best_label(x, Z_table):
        """Return the label maximizing x's Eq. 2.5 score.

        Ties including x's current label keep the current label.
        """
        y_x = strength[x]
        neighbor_weight = defaultdict(float)
        for v, data in G[x].items():
            wt = data.get(weight, 1) if weight is not None else 1
            neighbor_weight[labels[v]] += wt

        current = labels[x]
        current_score = neighbor_weight.get(current, 0.0) - (
            y_x * Z_table.get(current, 0.0) / M
        )

        best_score = current_score
        best = current
        tied = False
        for g, n_xg in neighbor_weight.items():
            if g == current:
                continue
            s = n_xg - y_x * Z_table.get(g, 0.0) / M
            if s > best_score:
                best_score = s
                best = g
                tied = False
            elif s == best_score and g != current:
                tied = True
                # Pick randomly between existing best and this one
                if seed.random() < 0.5:
                    best = g
        # If the best found still ties with current, keep current.
        if best_score == current_score:
            return current
        return best

    def _stage1():
        changed = True
        while changed:
            changed = False
            # Blue first, then red (paper convention).
            for side_nodes, is_red in ((blue_list, False), (red_list, True)):
                order = list(side_nodes)
                seed.shuffle(order)
                self_table = label_strength_red if is_red else label_strength_blue
                Z_table = label_strength_blue if is_red else label_strength_red
                for x in order:
                    old = labels[x]
                    new = _best_label(x, Z_table)
                    if new != old:
                        sx = strength[x]
                        self_table[old] -= sx
                        if self_table[old] <= 0:
                            self_table.pop(old, None)
                        self_table[new] = self_table.get(new, 0.0) + sx
                        labels[x] = new
                        changed = True

    def _stage2():
        """Perform one best mutual-best merge. Return True if merged."""
        inter = defaultdict(float)
        for u, v, data in G.edges(data=True):
            lu, lv = labels[u], labels[v]
            if lu != lv:
                wt = data.get(weight, 1) if weight is not None else 1
                key = (lu, lv) if lu < lv else (lv, lu)
                inter[key] += wt

        modules = sorted(set(labels.values()))
        if len(modules) < 2:
            return False

        def _delta(a, b):
            key = (a, b) if a < b else (b, a)
            e = inter.get(key, 0.0)
            cross = (
                label_strength_red.get(a, 0.0) * label_strength_blue.get(b, 0.0)
                + label_strength_red.get(b, 0.0) * label_strength_blue.get(a, 0.0)
            )
            return e / M - cross / (M * M)

        best_partner = {}
        for a in modules:
            best_d = float("-inf")
            best_b = None
            for b in modules:
                if b == a:
                    continue
                d = _delta(a, b)
                if d > best_d:
                    best_d = d
                    best_b = b
            best_partner[a] = (best_d, best_b)

        for a in modules:
            da, ba = best_partner[a]
            if da <= 0 or ba is None:
                continue
            db, bb = best_partner[ba]
            if bb == a:
                # Merge ba into a.
                for node, lab in labels.items():
                    if lab == ba:
                        labels[node] = a
                label_strength_red[a] = label_strength_red.get(
                    a, 0.0
                ) + label_strength_red.pop(ba, 0.0)
                label_strength_blue[a] = label_strength_blue.get(
                    a, 0.0
                ) + label_strength_blue.pop(ba, 0.0)
                if label_strength_red.get(a, 0.0) == 0:
                    label_strength_red.pop(a, None)
                if label_strength_blue.get(a, 0.0) == 0:
                    label_strength_blue.pop(a, None)
                return True
        return False

    _stage1()
    while _stage2():
        _stage1()

    communities = defaultdict(set)
    for v, lab in labels.items():
        communities[lab].add(v)
    return sorted(communities.values(), key=len, reverse=True)
