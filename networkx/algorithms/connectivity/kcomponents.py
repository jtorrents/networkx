"""
Exact k-component decomposition.

Two algorithms are provided:

- ``method="moody-white"`` -- the original Moody and White [1]_ recursive cut
  procedure, built on top of Kanevsky's [2]_ all-minimum-size node-separator
  enumeration (``nx.all_node_cuts``). This has been the NetworkX exact
  algorithm since 2015; it is kept in-tree as a correctness oracle for the
  Sinkovits path and for differential testing.

- ``method="sinkovits"`` -- the reduction-based exact algorithm of Sinkovits,
  Moody, Oztan and White [3]_. Instead of running max-flow on every subgraph,
  it iteratively *shrinks* the input using clique, k-core and biconnected
  component properties, then calls specialized 2- and 3-separator searches
  (and Kanevsky's algorithm for larger separators) only on the reduced
  kernel. On sparse clique-rich networks such as co-authorship graphs this
  is more than an order of magnitude faster than Moody-White; on graphs
  without clique structure the gains are more modest (roughly 2x in our
  benchmarks).

Both methods return the same decomposition (up to set equality) on every
graph; they differ only in how they compute it.

Implementation notes
--------------------
The Sinkovits path is a **clean-room implementation from the 2016 paper**
(J. Comput. Sci. 17, 62-72). The authors' reference implementation is R +
igraph (GPL-2+) and cannot be translated into NetworkX (MIT); only the paper
and its Fig. 4 pseudocode are used here. Section numbers in the comments
refer to [3]_.

References
----------
.. [1] Moody, J. and D. White (2003). Social cohesion and embeddedness:
       A hierarchical conception of social groups.
       American Sociological Review 68(1), 103-128.
.. [2] Kanevsky, A. (1993). Finding all minimum-size separating vertex
       sets in a graph. Networks 23(6), 533-541.
.. [3] Sinkovits, R.S., Moody, J., Oztan, B.T. and White, D.R. (2016).
       Fast determination of structurally cohesive subgroups in large
       networks. J. Comput. Sci. 17, 62-72.
       https://doi.org/10.1016/j.jocs.2016.10.005
"""

from collections import defaultdict
from itertools import combinations
from operator import itemgetter

import networkx as nx

# Default maximum flow function used for the Moody-White path and for
# node_connectivity / local_node_connectivity fallbacks in the Sinkovits path.
from networkx.algorithms.flow import build_residual_network, edmonds_karp
from networkx.utils import not_implemented_for

from .utils import build_auxiliary_node_connectivity

default_flow_func = edmonds_karp

__all__ = ["k_components"]


@not_implemented_for("directed")
@nx._dispatchable
def k_components(G, flow_func=None, method="sinkovits"):
    r"""Returns the k-component structure of a graph G.

    A `k`-component is a maximal subgraph of a graph G that has, at least,
    node connectivity `k`: we need to remove at least `k` nodes to break it
    into more components. `k`-components have an inherent hierarchical
    structure because they are nested in terms of connectivity: a connected
    graph can contain several 2-components, each of which can contain
    one or more 3-components, and so forth.

    Parameters
    ----------
    G : NetworkX graph

    flow_func : function
        Function to perform the underlying flow computations. Default value
        :meth:`edmonds_karp`. This function performs better in sparse graphs
        with right tailed degree distributions. :meth:`shortest_augmenting_path`
        will perform better in denser graphs.

    method : {"sinkovits", "moody-white"}, default "sinkovits"
        Algorithm used for the exact k-component decomposition.

        - ``"sinkovits"`` : reduction-based procedure of Sinkovits et al.
          2016 [3]_. More than an order of magnitude faster on sparse
          clique-rich networks such as co-authorship graphs; more modest
          gains on graphs without clique structure.
        - ``"moody-white"`` : legacy Moody-White [1]_ recursive-cut
          procedure using ``all_node_cuts`` on every subgraph. Kept in
          tree as a correctness oracle.

    Returns
    -------
    k_components : dict
        Dictionary with all connectivity levels `k` in the input Graph as keys
        and a list of sets of nodes that form a k-component of level `k` as
        values.

    Raises
    ------
    NetworkXNotImplemented
        If the input graph is directed.
    ValueError
        If ``method`` is not one of the supported options.

    Examples
    --------
    >>> # Petersen graph has 10 nodes and it is triconnected, thus all
    >>> # nodes are in a single component on all three connectivity levels
    >>> G = nx.petersen_graph()
    >>> k_components = nx.k_components(G)

    Notes
    -----
    Two exact algorithms are available. Both produce identical decompositions
    (up to set equality); see the module docstring for discussion.

    See also
    --------
    node_connectivity
    all_node_cuts
    biconnected_components : special case of this function when k=2
    k_edge_components : similar to this function, but uses edge-connectivity
        instead of node-connectivity

    References
    ----------
    .. [1]  Moody, J. and D. White (2003). Social cohesion and embeddedness:
            A hierarchical conception of social groups.
            American Sociological Review 68(1), 103--28.

    .. [2]  Kanevsky, A. (1993). Finding all minimum-size separating vertex
            sets in a graph. Networks 23(6), 533--541.

    .. [3]  Sinkovits, R.S., Moody, J., Oztan, B.T. and White, D.R. (2016).
            Fast determination of structurally cohesive subgroups in large
            networks. J. Comput. Sci. 17, 62--72.

    .. [4]  Torrents, J. and F. Ferraro (2015). Structural Cohesion:
            Visualization and Heuristics for Fast Computation.
            https://arxiv.org/pdf/1503.04476v1

    """
    if flow_func is None:
        flow_func = default_flow_func

    if method == "sinkovits":
        return _k_components_sinkovits(G, flow_func=flow_func)
    elif method == "moody-white":
        return _k_components_moody_white(G, flow_func=flow_func)
    else:
        raise ValueError(f"method must be 'sinkovits' or 'moody-white', got {method!r}")


# ---------------------------------------------------------------------------
# Moody-White (2003) path -- legacy exact algorithm
# ---------------------------------------------------------------------------


def _k_components_moody_white(G, flow_func):
    """Moody-White recursive-cut algorithm using Kanevsky's all-cuts routine.

    Unchanged from the pre-2026 NetworkX implementation except for the rename
    and the explicit ``flow_func`` argument. See [1]_ appendix A for the
    algorithm; [4]_ describes the k-core / bicomponent preprocessing.
    """
    # Dictionary with connectivity level (k) as keys and a list of
    # sets of nodes that form a k-component as values. Note that
    # k-components can overlap (but only k - 1 nodes).
    k_components = defaultdict(list)
    # Bicomponents as a base to check for higher order k-components
    for component in nx.connected_components(G):
        # isolated nodes have connectivity 0
        comp = set(component)
        if len(comp) > 1:
            k_components[1].append(comp)
    bicomponents = [G.subgraph(c) for c in nx.biconnected_components(G)]
    for bicomponent in bicomponents:
        bicomp = set(bicomponent)
        # avoid considering dyads as bicomponents
        if len(bicomp) > 2:
            k_components[2].append(bicomp)
    for B in bicomponents:
        if len(B) <= 2:
            continue
        k = nx.node_connectivity(B, flow_func=flow_func)
        if k > 2:
            k_components[k].append(set(B))
        # Perform cuts in a DFS like order.
        cuts = list(nx.all_node_cuts(B, k=k, flow_func=flow_func))
        stack = [(k, _generate_partition(B, cuts, k))]
        while stack:
            (parent_k, partition) = stack[-1]
            try:
                nodes = next(partition)
                C = B.subgraph(nodes)
                this_k = nx.node_connectivity(C, flow_func=flow_func)
                if this_k > parent_k and this_k > 2:
                    k_components[this_k].append(set(C))
                cuts = list(nx.all_node_cuts(C, k=this_k, flow_func=flow_func))
                if cuts:
                    stack.append((this_k, _generate_partition(C, cuts, this_k)))
            except StopIteration:
                stack.pop()

    return _reconstruct_k_components(k_components)


# ---------------------------------------------------------------------------
# Sinkovits et al. (2016) path -- reduction-based exact algorithm
# ---------------------------------------------------------------------------
#
# Roadmap of this section, mapped to the paper:
#
#   _k_components_sinkovits      top-level driver: 1-comps, 2-comps, worklist
#   _sinkovits_find_components   worklist search over (subgraph, k) items
#   _reduce_graph                Fig. 4: iterative clique / k-core / bicomp
#       _clique_isolate_reduce       Sec. 2.1: worldly/isolated pruning
#       _kcore_peel                  Sec. 2.2: iteratively remove deg < k
#       _split_bicomponents          Sec. 2.3: keep largest bicomp, spin rest
#   _find_min_separators         escalating search: size 2, 3, 4, ...
#       _simplicial_nodes            Sec. 3.1: nodes excluded from separators
#       _find_2_separators           Sec. 3.2: articulation points after
#                                    node deletion
#       _find_3_separators_easy      Sec. 3.3.1: edges -> articulation points
#       _find_3_separators_hard      Sec. 3.3.2: local conn == 3 neighbor
#                                    tests
#
# The Sinkovits algorithm is iterative across levels (find tricomps, then
# 4-comps inside each tricomp, then 5-comps inside each 4-comp, ...). We
# express that pattern with an explicit worklist of (subgraph, k) pairs;
# functionally equivalent, and it bounds stack depth on graphs with long
# chains of separators.
#
# The separator search escalates by size instead of assuming the kernel has
# connectivity k - 1. Reduction (k-core peeling in particular) can lower the
# connectivity of the kernel below k - 1, and both the specialized searches
# and Kanevsky's algorithm are only complete for minimum-size separators.
# Escalation restores the invariant each stage needs: a stage for size s runs
# only after every stage for sizes < s came back empty, which proves the
# kernel is at least (s)-connected. The paper applies the same idea when it
# re-runs the 2-separator search inside the 4-component search (Sec. 5.1.2).
#
# ---------------------------------------------------------------------------


def _k_components_sinkovits(G, flow_func):
    """Top-level driver for the Sinkovits et al. 2016 algorithm.

    Produces the full nested decomposition of G: k=1 (connected components),
    k=2 (biconnected components), and every k >= 3 obtained by the reduction
    + separator pipeline of [3]_.
    """
    k_comps = defaultdict(list)

    # --- Level k = 1: connected components ---------------------------------
    # Trivial; every connected graph with at least one edge is 1-cohesive
    # ([3]_ Sec. 1).
    for cc in nx.connected_components(G):
        if len(cc) > 1:
            k_comps[1].append(set(cc))

    # --- Level k = 2: biconnected components -------------------------------
    # The nodes within a biconnected component form a 2-cohesive subgroup
    # ([3]_ Sec. 1). Dyads (single edges) are excluded by convention,
    # matching the Moody-White implementation.
    bicomponents = list(nx.biconnected_components(G))
    for bc in bicomponents:
        if len(bc) > 2:
            k_comps[2].append(set(bc))

    # --- Levels k >= 3: recurse on each non-trivial bicomponent ------------
    # Each bicomp is independent; higher-order k-components never cross a
    # biconnected-component boundary (an articulation point would violate
    # 2-connectivity). We process each one and accumulate results.
    for bc in bicomponents:
        if len(bc) <= 2:
            continue
        B = G.subgraph(bc).copy()
        _sinkovits_find_components(B, k=3, k_comps=k_comps, flow_func=flow_func)

    # The consolidation step is shared with the Moody-White path: merge
    # overlapping candidate sets and ensure every level k appears in the
    # returned dict even if only implied by a higher-level component.
    return _reconstruct_k_components(k_comps)


def _sinkovits_find_components(H, k, k_comps, flow_func):
    """Search ``H`` for k-components and for every j-component with j > k.

    ``H`` must be an owned, mutable graph (it is reduced in place). It is
    biconnected on entry for k == 3; for higher k it is a kernel or a
    partition part produced by a previous iteration. Work items are
    (subgraph, level) pairs on an explicit stack: each item is reduced via
    :func:`_reduce_graph`, searched for minimum-size separators, and either
    partitioned (separators found) or recorded as a component (none found,
    so the kernel is at least k-connected).
    """
    work = [(H, k)]
    while work:
        G, k = work.pop()
        if len(G) <= k:
            continue

        # --------------------------------------------------------------
        # (1) Reduce G. [3]_ Fig. 4 / Sec. 2.3
        # --------------------------------------------------------------
        # After _reduce_graph:
        #   kernel            -- the largest bicomp of the reduced graph
        #   clique_candidates -- list of (clique_nodes, n_t) recorded by
        #                        the worldly/isolated rule ([3]_ Sec. 2.1);
        #                        each is an (n_t - 1)-component candidate
        #   spinoffs          -- smaller bicomponents split off during the
        #                        reduction loop; they still need to be
        #                        searched at level k because reduction only
        #                        preserves k-components *within* the kernel
        kernel, clique_candidates, spinoffs = _reduce_graph(G, k)

        # Record each clique-derived candidate at its natural level
        # (n_t - 1). The consolidation pass at the end merges overlapping
        # cliques and reconciles with any kernel-derived components at the
        # same level.
        for clique_nodes, n_t in clique_candidates:
            level = n_t - 1
            if level >= 2 and len(clique_nodes) > 2:
                k_comps[level].append(set(clique_nodes))

        # Smaller bicomponents spun off during reduction are independent
        # sub-problems at the same level k.
        for sub in spinoffs:
            if len(sub) > k:
                work.append((sub, k))

        if len(kernel) <= k:
            continue

        # --------------------------------------------------------------
        # (2) Find the minimum-size separators of the kernel. [3]_ Sec. 3
        # --------------------------------------------------------------
        # The search escalates from size 2 upward and stops at the first
        # size with separators; see _find_min_separators for why the
        # kernel cannot be assumed to be (k-1)-connected.
        size, seps = _find_min_separators(kernel, k - 1, flow_func)

        if not seps:
            # The kernel has no separator of size k - 1 or smaller, and
            # ``size`` is its exact node connectivity (>= k, since the
            # empty result proves connectivity above k - 1). Record the
            # kernel at its connectivity level and keep searching for
            # higher-order components nested inside it.
            k_comps[size].append(set(kernel))
            work.append((kernel, size + 1))
            continue

        # --------------------------------------------------------------
        # (3) Partition the kernel along the separators. [3]_ Sec. 4
        # --------------------------------------------------------------
        # Every node in a separator belongs to *both* sides of the induced
        # cut. The ``_generate_partition`` helper implements the split used
        # by Moody-White and is reused here. Each part is processed at the
        # SAME level k -- it is only *candidate* k-connected and must be
        # re-reduced and re-separated to confirm.
        for part in _generate_partition(kernel, seps, size):
            if len(part) > k:
                work.append((kernel.subgraph(part).copy(), k))


# ---------------------------------------------------------------------------
# Graph reduction -- Fig. 4 of [3]_
# ---------------------------------------------------------------------------


def _reduce_graph(H, k):
    """Iterative reduction loop from [3]_ Fig. 4.

    Repeatedly apply, until a fixed point on the kernel size is reached:

    1. Clique worldly/isolated pruning ([3]_ Sec. 2.1).
       Only active for ``k >= 4``. [3]_ Sec. 4.1 notes that for
       tricomponent search the clique step is omitted because triangles
       dominate the clique enumeration and provide no useful reduction.

    2. k-core peeling: iteratively delete every node of degree < k
       ([3]_ Sec. 2.2).

    3. Biconnected-component decomposition: keep the largest bicomp as
       the kernel; any other bicomp is spun off for independent
       processing at the same level ([3]_ Sec. 2.3, Sec. 4.1).

    ``H`` is reduced destructively: the caller must own it.

    Returns
    -------
    kernel : Graph
        ``H`` itself, reduced in place (possibly empty).
    clique_candidates : list of (frozenset, int)
        Each entry is (clique_nodes, clique_size). The clique is an
        (n_t - 1)-component candidate. Reported so the driver can register
        the k-component implied by the clique itself.
    spinoffs : list of Graph
        Non-largest bicomponents encountered during iteration. Each is a
        subgraph copy that may still contain k-components and must be
        searched independently.
    """
    clique_candidates = []
    spinoffs = []

    while True:
        size_before = H.number_of_nodes()

        # --- 1. Clique worldly/isolated pruning -----------------------------
        # Only applied when searching for k-components with k >= 4 ([3]_
        # Sec. 4.1: for tricomponents the clique step "provides little
        # additional benefit, but at significant computational expense").
        if k >= 4:
            _clique_isolate_reduce(H, k, clique_candidates)

        # --- 2. k-core peeling ----------------------------------------------
        # Every k-component is a k-core ([3]_ Sec. 2.2), so nodes of degree
        # < k cannot belong to any k-component. The inline peel is cheaper
        # than repeated calls to nx.k_core because we are already mutating H.
        _kcore_peel(H, k)

        # --- 3. Biconnected-component retention -----------------------------
        # k-core peeling can introduce new articulation points ([3]_
        # Sec. 2.3, Fig. 3), which then get removed by restricting to the
        # largest biconnected component. Smaller bicomps are saved for
        # separate processing -- they may still contain k-components.
        _split_bicomponents(H, k, spinoffs)

        if H.number_of_nodes() == size_before:
            # Fixed point reached: neither cliques, nor k-core, nor bicomp
            # retention removed any node this iteration.
            break

    return H, clique_candidates, spinoffs


def _clique_isolate_reduce(G, k, clique_candidates):
    """Apply the clique worldly/isolated rule ([3]_ Sec. 2.1) in place.

    For every maximal clique in ``G``, partition its nodes into *worldly*
    (at least one neighbor outside the clique) and *isolated* (all
    neighbors inside). If the clique contains at most ``k - 1`` worldly
    nodes, the isolated nodes cannot belong to any k-component that
    extends outside the clique (all paths to the outside must pass through
    the <= k - 1 worldly gatekeepers, which is one short of the k
    node-disjoint paths required). Note that the prose in [3]_ Sec. 2.1
    says "n_w less than or equal to k", but with n_w == k an isolated node
    can still have k disjoint paths to the outside; the ``n_w == k - 1``
    rule of the Fig. 4 pseudocode is the safe one and is what we use, in
    the slightly more general ``n_w <= k - 1`` form.

    We remove those isolated nodes from ``G`` and record the clique itself
    as an (n_t - 1)-component candidate: a clique of size n_t has node
    connectivity n_t - 1 and is therefore a (n_t - 1)-component of the
    original graph.

    The loop is repeated until no more cliques qualify, as removing
    isolated nodes can change worldliness for other cliques ([3]_
    Sec. 2.3, Fig. 2).
    """
    while True:
        cliques_to_prune = []
        for C in nx.find_cliques(G):
            n_t = len(C)
            # A clique with fewer than k+1 nodes cannot give anything
            # useful at level k (the induced (n_t - 1)-component would
            # be below k), skip quickly. This is a cheap implementation
            # optimization; it does not alter the algorithm.
            if n_t < k + 1:
                continue
            clique_set = set(C)
            worldly = 0
            for v in C:
                # Early-exit as soon as worldly exceeds the threshold; we
                # only care whether n_w <= k - 1.
                for nbr in G[v]:
                    if nbr not in clique_set:
                        worldly += 1
                        break
                if worldly > k - 1:
                    break
            if worldly <= k - 1:
                cliques_to_prune.append((clique_set, n_t))

        if not cliques_to_prune:
            return

        removed_any = False
        for clique_set, n_t in cliques_to_prune:
            # Classify worldly vs. isolated for the final clique, since
            # node removals earlier in this batch may have changed the
            # graph (they should not touch these cliques -- an isolated
            # node belongs to exactly one maximal clique -- but be
            # defensive).
            isolated = {
                v
                for v in clique_set
                if v in G and all(nbr in clique_set for nbr in G[v])
            }
            if not isolated:
                continue
            # Record the clique BEFORE removing isolated nodes -- the
            # full clique is an (n_t - 1)-component candidate.
            clique_candidates.append((frozenset(clique_set), n_t))
            G.remove_nodes_from(isolated)
            removed_any = True

        if not removed_any:
            return


def _kcore_peel(G, k):
    """Iteratively remove nodes of degree < k from G, in place ([3]_ Sec. 2.2).

    Equivalent to replacing G by its k-core but done destructively on the
    graph owned by :func:`_reduce_graph`. Queue-based, so the peel runs in
    O(|V| + |E|) instead of rescanning all degrees on every round.
    """
    queue = [v for v, d in G.degree() if d < k]
    while queue:
        v = queue.pop()
        if v not in G:
            continue
        nbrs = list(G[v])
        G.remove_node(v)
        queue.extend(u for u in nbrs if G.degree(u) < k)


def _split_bicomponents(G, k, spinoffs):
    """Retain the largest bicomponent of G; spin off the rest ([3]_ Sec. 2.3).

    Mutates G in place so that only the nodes of the largest biconnected
    component remain. Every other bicomponent of more than k nodes becomes
    a separate subgraph copy appended to ``spinoffs`` for independent
    processing by the driver.
    """
    bcs = list(nx.biconnected_components(G))
    if not bcs:
        # Empty or single-vertex graph after k-core peeling.
        G.remove_nodes_from(list(G.nodes))
        return

    largest = max(bcs, key=len)
    for bc in bcs:
        if bc is largest:
            continue
        # Non-largest bicomps are split off for independent processing. We
        # only bother keeping ones that could still contain a k-component.
        if len(bc) > k:
            spinoffs.append(G.subgraph(bc).copy())

    # Restrict G to the largest bicomp, in place.
    to_remove = set(G.nodes) - set(largest)
    if to_remove:
        G.remove_nodes_from(to_remove)


# ---------------------------------------------------------------------------
# Separator search -- Sec. 3 of [3]_
# ---------------------------------------------------------------------------


def _find_min_separators(G, max_size, flow_func):
    """Find the minimum-size node separators of ``G``, up to ``max_size``.

    Returns a ``(size, separators)`` pair. When ``separators`` is empty,
    ``size`` is the exact node connectivity of ``G`` (necessarily larger
    than ``max_size``); otherwise ``separators`` holds every minimum-size
    separator and ``size`` their common size.

    The search is staged. The cheap articulation-point sweep for size 2
    runs first ([3]_ Sec. 3.2). If it comes back empty, the exact node
    connectivity ``kappa`` is computed once and dispatches the rest: the
    specialized easy/hard searches of [3]_ Sec. 3.3 when ``kappa == 3``
    (easy first, following the staged pipelines of [3]_ Figs. 11 and 12,
    since partitioning by easy separators and re-reducing shrinks the
    graph before the expensive hard search is needed), or Kanevsky's
    algorithm (:func:`nx.all_node_cuts`) at exactly ``kappa`` when
    ``4 <= kappa <= max_size``.

    The staging is required for correctness, not just speed: reduction can
    lower the connectivity of a kernel below k - 1 (k-core peeling can
    introduce new low-order separators, [3]_ Fig. 3), and both the hard
    3-separator search and Kanevsky's algorithm return incomplete results
    when the graph's connectivity does not match the requested size. The
    paper applies the same correction when it re-runs the 2-separator
    search inside the 4-component search ([3]_ Sec. 5.1.2). Knowing
    ``kappa`` exactly also avoids the very expensive alternative of
    proving "no hard 3-separators exist" by exhaustive pair testing on
    kernels that are in fact 4-connected or better.
    """
    # [3]_ Sec. 3.1: nodes whose neighbors all belong to one clique cannot
    # be members of any minimal separator; exclude them from the searches.
    simplicial = _simplicial_nodes(G)
    seps = list(_find_2_separators(G, simplicial))
    if seps:
        return 2, seps
    # G is at least 3-connected from here on. Its exact connectivity picks
    # the one enumeration that is complete for this graph; when it exceeds
    # max_size it is also the level the caller records the kernel at.
    kappa = nx.node_connectivity(G, flow_func=flow_func)
    if kappa > max_size:
        return kappa, []
    if kappa == 3:
        seps = list(_find_3_separators_easy(G, simplicial))
        if seps:
            return 3, seps
        # No 2-separators and no easy 3-separators, so every minimum cut
        # is a pairwise non-adjacent triple, which the hard search
        # enumerates completely on a 3-connected graph.
        return 3, list(_find_3_separators_hard(G, simplicial, flow_func))
    # 4 <= kappa <= max_size. [3]_ Sec. 5.1.3 adapts the Sec. 3.3.2 search
    # to higher orders; we use Kanevsky's algorithm on the (small) kernel
    # instead, which is complete when the requested size equals the exact
    # connectivity.
    return kappa, list(nx.all_node_cuts(G, k=kappa, flow_func=flow_func))


def _simplicial_nodes(G):
    """Return the set of simplicial nodes of ``G``.

    A node is simplicial when its neighborhood induces a clique; these are
    exactly the "isolated" clique members of [3]_ Sec. 3.1, which cannot
    belong to any minimal separator (two of their neighbors in different
    components of the separated graph would have to be adjacent). The
    paper reports that excluding them roughly halves separator search
    time on their co-authorship case study.
    """
    adj = G.adj
    simplicial = set()
    for v in G:
        nbrs = list(adj[v])
        if all(w in adj[u] for i, u in enumerate(nbrs) for w in nbrs[i + 1 :]):
            simplicial.add(v)
    return simplicial


def _find_2_separators(G, simplicial):
    """Enumerate all 2-node separators of ``G`` ([3]_ Sec. 3.2).

    Implementation: for every node ``v``, consider ``G - v``. Every
    articulation point ``u`` of that graph yields a 2-separator ``{u, v}``.
    Running time is O(|V| * (|V| + |E|)) since Tarjan's articulation-point
    algorithm is linear. A single working copy is mutated and restored per
    candidate, which is much cheaper than per-candidate subgraph views.

    Nodes in ``simplicial`` cannot belong to any minimal separator and are
    skipped as deletion candidates; their partners found as articulation
    points cannot be simplicial either, so the enumeration stays complete.
    """
    seen = set()
    H = G.copy()
    for v in list(G):
        if v in simplicial:
            continue
        nbrs = list(H[v])
        H.remove_node(v)
        if H.number_of_nodes() >= 2:
            for u in nx.articulation_points(H):
                sep = frozenset((u, v))
                if sep not in seen:
                    seen.add(sep)
                    yield set(sep)
        H.add_node(v)
        H.add_edges_from((v, w) for w in nbrs)


def _find_3_separators_easy(G, simplicial):
    """Enumerate easy-to-find 3-node separators ([3]_ Sec. 3.3.1).

    A 3-separator is "easy" when at least two of its three members are
    joined by an edge. For every edge ``(u, v)`` in G, compute the
    articulation points of ``G - {u, v}``; each such articulation point
    ``w`` yields the 3-separator ``{u, v, w}``. Worst-case
    O(|E| * (|V| + |E|)). Complete for minimal 3-separators containing an
    edge, provided G is 3-connected (the caller guarantees this by running
    the 2-separator search first).

    Edges with a simplicial endpoint are skipped: a minimal separator
    cannot contain a simplicial node.
    """
    seen = set()
    H = G.copy()
    for u, v in list(G.edges()):
        if u in simplicial or v in simplicial:
            continue
        if H.number_of_nodes() < 4:
            break
        u_nbrs = list(H[u])
        H.remove_node(u)
        v_nbrs = list(H[v])
        H.remove_node(v)
        for w in nx.articulation_points(H):
            sep = frozenset((u, v, w))
            if sep not in seen:
                seen.add(sep)
                yield set(sep)
        H.add_node(v)
        H.add_edges_from((v, w) for w in v_nbrs)
        H.add_node(u)
        H.add_edges_from((u, w) for w in u_nbrs)


def _find_3_separators_hard(G, simplicial, flow_func):
    """Enumerate hard-to-find 3-node separators ([3]_ Sec. 3.3.2).

    A 3-separator is "hard" when no two of its members are adjacent in G.
    The procedure has two phases:

    1. Identify the candidate set ``V_hard`` of nodes that *could*
       participate in some hard 3-separator. From [3]_ Sec. 3.3.2: for a
       candidate node ``v`` with neighbors ``{n_1, ..., n_N}``, if there
       exists a pair ``(n_i, n_j)`` of non-adjacent neighbors such that
       the number of node-disjoint paths between ``n_i`` and ``n_j`` in G
       is exactly 3, then v is in some 3-separator (every maximum family
       of disjoint paths must use the path through v, so every minimum
       cut for the pair contains v).

    2. Enumerate all 3-subsets of ``V_hard`` and keep the pairwise
       non-adjacent triples whose removal disconnects G.

    The Fig. 10 optimization -- skip neighbor pairs joined by an edge --
    is applied; a hard separator's witnessing pair straddles the cut and
    is never adjacent, so completeness is preserved. Simplicial nodes are
    skipped as candidates ([3]_ Sec. 3.1).

    G must be 3-connected for phase 1 to be complete (every member of a
    minimal hard 3-separator then has a witnessing pair with local
    connectivity exactly 3). The caller guarantees this by running the
    2-separator search first.
    """
    # --- Phase 1: collect V_hard ------------------------------------------
    # Build the Even-Tarjan auxiliary digraph and residual network once and
    # reuse them for every pair test; rebuilding them per pair dominates
    # the running time otherwise.
    aux = build_auxiliary_node_connectivity(G)
    residual = build_residual_network(aux, "capacity")
    local_node_connectivity = nx.connectivity.local_node_connectivity
    V_hard = set()
    # Cap local_node_connectivity at 4: we only need to distinguish
    # "== 3" from ">= 4".
    cutoff = 4
    for v in G.nodes:
        if v in simplicial:
            continue
        nbrs = list(G[v])
        if len(nbrs) < 2:
            continue
        for n_i, n_j in combinations(nbrs, 2):
            if G.has_edge(n_i, n_j):
                continue
            # If local connectivity between n_i and n_j is exactly 3, v is
            # in a hard 3-separator involving some triple of nodes.
            lnc = local_node_connectivity(
                G,
                n_i,
                n_j,
                flow_func=flow_func,
                auxiliary=aux,
                residual=residual,
                cutoff=cutoff,
            )
            if lnc == 3:
                V_hard.add(v)
                break

    # --- Phase 2: enumerate 3-subsets of V_hard and verify ----------------
    # For each triple {a, b, c} from V_hard: if {a, b, c} is pairwise
    # non-adjacent AND removing it disconnects G, it's a hard 3-separator.
    # Triples with an adjacent pair belong to the easy search.
    nodes = set(G.nodes)
    for a, b, c in combinations(list(V_hard), 3):
        if G.has_edge(a, b) or G.has_edge(a, c) or G.has_edge(b, c):
            continue
        rest = nodes - {a, b, c}
        if len(rest) < 2:
            continue
        if not nx.is_connected(G.subgraph(rest)):
            yield {a, b, c}


# ---------------------------------------------------------------------------
# Shared helpers (used by both Moody-White and Sinkovits paths)
# ---------------------------------------------------------------------------


def _consolidate(sets, k):
    """Merge sets that share k or more elements.

    See: http://rosettacode.org/wiki/Set_consolidation

    The iterative python implementation posted there is
    faster than this because of the overhead of building a
    Graph and calling nx.connected_components, but it's not
    clear for us if we can use it in NetworkX because there
    is no licence for the code.

    """
    G = nx.Graph()
    nodes = dict(enumerate(sets))
    G.add_nodes_from(nodes)
    G.add_edges_from(
        (u, v) for u, v in combinations(nodes, 2) if len(nodes[u] & nodes[v]) >= k
    )
    for component in nx.connected_components(G):
        yield set.union(*[nodes[n] for n in component])


def _generate_partition(G, cuts, k):
    """Generate parts of G induced by its minimum-size node cutsets.

    Following Moody and White (appendix A, step 3), cutsets are applied
    one at a time: removing a cutset splits the current set of nodes into
    two or more connected components, and each component plus the cutset
    that induced it becomes a new candidate part. Every j-component of
    ``G`` with ``j > k`` is preserved intact in at least one generated
    part, because removing ``k < j`` nodes cannot disconnect it.

    Applying cutsets one at a time is necessary for correctness: removing
    the union of all cutsets in a single pass can drop cutset nodes whose
    neighbors all belong to other cutsets, silently losing k-components.
    """
    order = G.order()
    cuts = [set(cut) for cut in cuts]
    parts = []
    seen = set()
    stack = [set(G)]
    while stack:
        nodes = stack.pop()
        for cut in cuts:
            if not cut < nodes:
                continue
            components = list(nx.connected_components(G.subgraph(nodes - cut)))
            if len(components) > 1:
                for component in components:
                    # Nodes of the cutset belong to both sides of the induced
                    # cut, but only if they have a neighbor in the component:
                    # a cutset node from a j-component (j > k) always has at
                    # least two neighbors in the component, while attaching a
                    # neighborless cutset node would disconnect the part.
                    child = frozenset(
                        component
                        | {n for n in cut if any(v in component for v in G[n])}
                    )
                    if child not in seen:
                        seen.add(child)
                        stack.append(set(child))
                break
        else:
            # No cutset splits this part any further.
            if len(nodes) < order:
                parts.append(nodes)
    # Merging parts that share at least k+1 nodes reduces the number of
    # subproblems that the loop in k_components examines. This is a
    # performance heuristic, not needed for correctness: no j-component
    # with j > k can span two parts. If merging rebuilds the whole graph,
    # fall back to the raw parts so the caller always recurses on
    # strictly smaller graphs.
    consolidated = list(_consolidate(parts, k + 1))
    if any(len(part) == order for part in consolidated):
        yield from parts
    else:
        yield from consolidated


def _reconstruct_k_components(k_comps):
    result = {}
    max_k = max(k_comps) if k_comps else 0
    for k in range(max_k, 0, -1):
        if k == max_k:
            result[k] = list(_consolidate(k_comps[k], k))
        elif k not in k_comps:
            result[k] = list(_consolidate(result[k + 1], k))
        else:
            # Propagate a component from level k+1 down to level k unless it
            # is already contained in a single component recorded at level k.
            # Checking against the union of all components recorded at k is
            # not enough: a (k+1)-level component can span several k-level
            # candidates without being a subset of any of them, and skipping
            # it would replace a genuine k-component by its fragments.
            to_add = [
                c for c in result[k + 1] if not any(c <= comp for comp in k_comps[k])
            ]
            if to_add:
                result[k] = list(_consolidate(k_comps[k] + to_add, k))
            else:
                result[k] = list(_consolidate(k_comps[k], k))
    return result


def build_k_number_dict(kcomps):
    return {
        node: k
        for k, comps in sorted(kcomps.items(), key=itemgetter(0))
        for comp in comps
        for node in comp
    }
