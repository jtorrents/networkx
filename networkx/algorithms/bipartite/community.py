"""Community detection and quality functions for bipartite graphs."""

from collections import defaultdict

import networkx as nx
from networkx.algorithms.community.community_utils import is_partition
from networkx.algorithms.community.quality import NotAPartition
from networkx.utils.decorators import not_implemented_for, py_random_state

__all__ = ["modularity", "condor_communities"]


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
@nx._dispatchable(name="bipartite_condor_communities", edge_attrs="weight")
def condor_communities(
    G, nodes, weight="weight", convergence_threshold=1e-4, seed=None
):
    r"""Find communities in a bipartite graph using CONDOR.

    CONDOR (COmplex Network Description Of Regulators) [1]_ detects
    communities in bipartite networks by maximizing Barber's bipartite
    modularity $Q_B$ [2]_.

    The algorithm has three steps:

    1. **Projection and initialization.** The bipartite graph is projected
       onto the smaller node set, producing a weighted unipartite graph.
       Louvain community detection is run on this projection to obtain an
       initial partition. The partition is then extended to the other node
       set by assigning each node to the community with the highest total
       edge weight.

    2. **BRIM refinement.** The partition is refined using BRIM (Bipartite,
       Recursively Induced Modules) [2]_. In each iteration, community
       assignments for one node set are fixed while the other set is
       updated by computing the product of the bimodularity matrix with
       the community membership matrix. Each node is assigned to the
       community that maximizes its contribution to $Q_B$. The two sides
       alternate until convergence.

       The bimodularity matrix $\tilde{B} = \tilde{A} - k d^T / m$ is
       never formed explicitly. Instead, the matrix--matrix product
       $\tilde{B} C$ is computed as $A C - k (d^T C) / m$, keeping
       $A$ sparse.

    3. **Output.** The converged partition is returned.

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

    convergence_threshold : float, optional (default=1e-4)
        BRIM stops when the increase in $Q_B$ between iterations falls
        below this value.

    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`. Controls the Louvain step
        used for initialization; the BRIM refinement itself is
        deterministic.

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
    CONDOR relies on a unipartite projection for initialization, which
    makes it particularly effective on large sparse bipartite networks
    where the projection is also sparse. The BRIM refinement uses sparse
    matrix operations (via SciPy), so the computational cost is dominated
    by the projection step and the Louvain initialization.

    CONDOR complements :func:`lpawb_plus_communities`: LPAwb+ is
    parameter-free and requires no dependencies beyond NetworkX, while
    CONDOR is faster on large networks thanks to matrix-based BRIM
    refinement.

    See Also
    --------
    modularity
    lpawb_plus_communities

    References
    ----------
    .. [1] Platig, J., Castaldi, P. J., DeMeo, D., and Quackenbush, J.
       (2016). "Bipartite Community Structure of eQTLs."
       PLoS Computational Biology, 12(9), e1005033.
       https://doi.org/10.1371/journal.pcbi.1005033
    .. [2] Barber, M. J. (2007). "Modularity and community detection in
       bipartite networks." Physical Review E, 76(6), 066102.
    """
    import numpy as np

    red = set(nodes)
    blue = set(G) - red

    if G.number_of_edges() == 0 or not red or not blue:
        return [{n} for n in G]

    # --- Step 1: Initial partition via unipartite projection ---

    # Project onto the smaller side for efficiency.
    if len(red) > len(blue):
        project_side, other_side = blue, red
    else:
        project_side, other_side = red, blue

    projection = nx.bipartite.weighted_projected_graph(G, project_side)
    initial_comms = nx.community.louvain_communities(
        projection, weight="weight", seed=seed
    )

    labels = {}
    for label, comm in enumerate(initial_comms):
        for node in comm:
            labels[node] = label

    # Extend to the other side: assign each node to the community
    # with the highest total edge weight from its neighbors.
    def _get_edge_weight(edge_data):
        return edge_data.get(weight, 1) if weight is not None else 1

    for node in other_side:
        weight_by_community = defaultdict(float)
        for neighbor, edge_data in G[node].items():
            if neighbor in labels:
                weight_by_community[labels[neighbor]] += _get_edge_weight(
                    edge_data
                )
        if weight_by_community:
            labels[node] = max(
                weight_by_community, key=weight_by_community.get
            )
        else:
            labels[node] = 0

    # --- Step 2: BRIM iterative refinement ---

    red_list = sorted(red)
    blue_list = sorted(blue)
    red_idx = {v: i for i, v in enumerate(red_list)}
    blue_idx = {v: i for i, v in enumerate(blue_list)}

    # Biadjacency matrix (red rows x blue cols).
    A = nx.bipartite.biadjacency_matrix(
        G, row_order=red_list, column_order=blue_list,
        weight=weight, format="csr",
    )
    red_degree = np.asarray(A.sum(axis=1)).ravel()
    blue_degree = np.asarray(A.sum(axis=0)).ravel()
    total_weight = red_degree.sum()

    if total_weight == 0:
        return [{n} for n in G]

    n_comms = max(labels.values()) + 1

    def _membership_matrix(node_list, idx_map, n_cols):
        """Build a dense |nodes| x n_cols membership matrix."""
        rows = np.array([idx_map[v] for v in node_list])
        cols = np.array([labels[v] for v in node_list])
        M = np.zeros((len(node_list), n_cols))
        M[rows, cols] = 1.0
        return M

    def _compute_modularity(R, C):
        """Compute Q_B = (1/m) tr(Rᵀ B̃ C) without forming B̃."""
        RtAC = (R.T @ A) @ C
        return (
            np.trace(RtAC) / total_weight
            - (red_degree @ R * (blue_degree @ C)).sum() / total_weight**2
        )

    def _update_labels(score_matrix, node_list):
        """Assign each node to the community with the highest score."""
        best = score_matrix.argmax(axis=1)
        for i, node in enumerate(node_list):
            labels[node] = int(best[i])

    prev_Q = -np.inf

    while True:
        # Update red labels: fix blue, optimize red.
        # T = B̃ @ C = A @ C - outer(red_degree, blue_degree @ C) / m
        C = _membership_matrix(blue_list, blue_idx, n_comms)
        T = A @ C - np.outer(red_degree, blue_degree @ C) / total_weight
        _update_labels(T, red_list)

        # Update blue labels: fix red, optimize blue.
        # S = B̃ᵀ @ R = Aᵀ @ R - outer(blue_degree, red_degree @ R) / m
        R = _membership_matrix(red_list, red_idx, n_comms)
        S = A.T @ R - np.outer(blue_degree, red_degree @ R) / total_weight
        _update_labels(S, blue_list)

        # Check convergence. Reuse R and rebuild C for the Q_B check.
        n_comms = max(labels.values()) + 1
        R = _membership_matrix(red_list, red_idx, n_comms)
        C = _membership_matrix(blue_list, blue_idx, n_comms)
        current_Q = _compute_modularity(R, C)

        if current_Q - prev_Q < convergence_threshold:
            break
        prev_Q = current_Q

    # --- Build output ---
    communities = {}
    for node, label in labels.items():
        communities.setdefault(label, set()).add(node)
    return sorted(communities.values(), key=len, reverse=True)
