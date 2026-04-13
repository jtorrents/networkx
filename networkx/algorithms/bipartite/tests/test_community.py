"""Tests for bipartite community quality functions."""

import pytest

import networkx as nx
from networkx.algorithms import bipartite
from networkx.algorithms.community.quality import NotAPartition


class TestBipartiteModularity:
    def test_complete_bipartite_no_structure(self):
        # K_{2,2} has no community structure: every red connects to
        # every blue, so any split gives Q_B = 0.
        G = nx.complete_bipartite_graph(2, 2)
        red = {0, 1}
        communities = [{0, 2}, {1, 3}]
        assert bipartite.modularity(G, communities, red) == pytest.approx(0.0)

    def test_two_disjoint_edges(self):
        # Two disconnected edges form two perfect bipartite communities.
        G = nx.Graph([(0, 2), (1, 3)])
        red = {0, 1}
        communities = [{0, 2}, {1, 3}]
        # m = 2; each community: L_c=1, k_c=1, d_c=1
        # contribution = 1/2 - 1*1/4 = 1/4; total = 1/2
        assert bipartite.modularity(G, communities, red) == pytest.approx(0.5)

    def test_single_community(self):
        G = nx.complete_bipartite_graph(3, 4)
        red = set(range(3))
        communities = [set(G)]
        assert bipartite.modularity(G, communities, red) == pytest.approx(0.0)

    def test_singleton_communities(self):
        G = nx.complete_bipartite_graph(3, 4)
        red = set(range(3))
        communities = [{n} for n in G]
        assert bipartite.modularity(G, communities, red) == pytest.approx(0.0)

    def test_weighted(self):
        G = nx.Graph()
        G.add_edge(0, 2, weight=3)
        G.add_edge(1, 3, weight=1)
        red = {0, 1}
        communities = [{0, 2}, {1, 3}]
        # m = 4
        # {0,2}: L_c=3, k_c=3, d_c=3  -> 3/4 - 9/16 = 3/16
        # {1,3}: L_c=1, k_c=1, d_c=1  -> 1/4 - 1/16 = 3/16
        # total = 6/16 = 0.375
        assert bipartite.modularity(G, communities, red) == pytest.approx(0.375)

    def test_weight_none_ignores_attribute(self):
        G = nx.Graph()
        G.add_edge(0, 2, weight=3)
        G.add_edge(1, 3, weight=1)
        red = {0, 1}
        communities = [{0, 2}, {1, 3}]
        # With weight=None, both edges have weight 1. m = 2.
        # Each community contributes 1/2 - 1/4 = 1/4 -> total 0.5
        assert bipartite.modularity(G, communities, red, weight=None) == pytest.approx(
            0.5
        )

    def test_resolution_scales_null_model(self):
        G = nx.Graph([(0, 2), (1, 3)])
        red = {0, 1}
        communities = [{0, 2}, {1, 3}]
        q1 = bipartite.modularity(G, communities, red, resolution=1)
        q_low = bipartite.modularity(G, communities, red, resolution=0.5)
        q_high = bipartite.modularity(G, communities, red, resolution=2)
        assert q_low > q1 > q_high
        # Explicit values (m=2): per-community L_c/m - res*k_c*d_c/m^2
        # res=0.5: 2 * (1/2 - 0.5 * 1/4) = 0.75
        # res=2.0: 2 * (1/2 - 2   * 1/4) = 0
        assert q_low == pytest.approx(0.75)
        assert q_high == pytest.approx(0.0)

    def test_iterable_of_communities_is_accepted(self):
        G = nx.Graph([(0, 2), (1, 3)])
        red = {0, 1}
        communities = iter([{0, 2}, {1, 3}])
        assert bipartite.modularity(G, communities, red) == pytest.approx(0.5)

    def test_empty_graph(self):
        G = nx.Graph()
        assert bipartite.modularity(G, [], set()) == pytest.approx(0.0)

    def test_not_a_partition_raises(self):
        G = nx.complete_bipartite_graph(2, 2)
        red = {0, 1}
        # Missing node 3
        with pytest.raises(NotAPartition):
            bipartite.modularity(G, [{0, 2}, {1}], red)
        # Overlapping
        with pytest.raises(NotAPartition):
            bipartite.modularity(G, [{0, 1, 2}, {1, 3}], red)

    def test_directed_not_implemented(self):
        G = nx.DiGraph([(0, 2), (1, 3)])
        with pytest.raises(nx.NetworkXNotImplemented):
            bipartite.modularity(G, [{0, 2}, {1, 3}], {0, 1})

    def test_communities_with_single_side_contribute_zero(self):
        # A community containing only red (or only blue) nodes has L_c=0
        # and one of k_c, d_c = 0, so its contribution is 0.
        G = nx.complete_bipartite_graph(2, 2)
        red = {0, 1}
        communities = [{0, 1}, {2, 3}]
        assert bipartite.modularity(G, communities, red) == pytest.approx(0.0)

    def test_southern_women_trivial_partitions(self):
        G = nx.davis_southern_women_graph()
        women = {n for n, d in G.nodes(data=True) if d["bipartite"] == 0}
        assert bipartite.modularity(G, [set(G)], women) == pytest.approx(0.0)
        assert bipartite.modularity(G, [{n} for n in G], women) == pytest.approx(0.0)

    def test_disjoint_bipartite_components_positive(self):
        # Two disjoint K_{2,2} components; grouping each component as a
        # community gives strong bipartite modularity.
        G = nx.Graph()
        G.add_edges_from([(0, 2), (0, 3), (1, 2), (1, 3)])
        G.add_edges_from([(4, 6), (4, 7), (5, 6), (5, 7)])
        red = {0, 1, 4, 5}
        communities = [{0, 1, 2, 3}, {4, 5, 6, 7}]
        q = bipartite.modularity(G, communities, red)
        assert q > 0
        # Hand calc: m = 8. For each component L_c=4, k_c=4, d_c=4.
        # contribution = 4/8 - 16/64 = 0.5 - 0.25 = 0.25; total = 0.5
        assert q == pytest.approx(0.5)

    def test_random_bipartite_trivial_partitions(self):
        G = nx.bipartite.random_graph(6, 8, 0.5, seed=42)
        red = {n for n, d in G.nodes(data=True) if d["bipartite"] == 0}
        assert bipartite.modularity(G, [set(G)], red) == pytest.approx(0.0)
        assert bipartite.modularity(G, [{n} for n in G], red) == pytest.approx(0.0)

    def test_nodes_as_iterable(self):
        G = nx.complete_bipartite_graph(2, 2)
        # Pass nodes as list rather than set; function should accept any
        # iterable and convert internally.
        communities = [{0, 2}, {1, 3}]
        assert bipartite.modularity(G, communities, [0, 1]) == pytest.approx(0.0)


def _partition_covers_graph(communities, G):
    seen = set()
    for c in communities:
        assert seen.isdisjoint(c)
        seen.update(c)
    assert seen == set(G)


class TestLPAwbPlus:
    def test_two_disjoint_k22_components(self):
        # Two K_{2,2} components; LPAwb+ should recover them exactly,
        # giving Q_B = 0.5 (same as the hand-calc test).
        G = nx.Graph()
        G.add_edges_from([(0, 2), (0, 3), (1, 2), (1, 3)])
        G.add_edges_from([(4, 6), (4, 7), (5, 6), (5, 7)])
        red = {0, 1, 4, 5}
        comms = bipartite.lpawb_plus_communities(G, red, seed=42)
        _partition_covers_graph(comms, G)
        assert sorted(map(sorted, comms)) == [[0, 1, 2, 3], [4, 5, 6, 7]]
        assert bipartite.modularity(G, comms, red) == pytest.approx(0.5)

    def test_three_disjoint_k23_components(self):
        G = nx.Graph()
        for off in (0, 5, 10):
            for r in range(2):
                for b in range(3):
                    G.add_edge(off + r, off + 2 + b)
        red = {n for n in G if n % 5 < 2}
        comms = bipartite.lpawb_plus_communities(G, red, seed=1)
        _partition_covers_graph(comms, G)
        assert len(comms) == 3
        assert bipartite.modularity(G, comms, red) == pytest.approx(2 / 3)

    def test_weighted_disjoint_edges(self):
        # Same graph as the weighted modularity test: two disjoint
        # weighted edges. LPAwb+ should recover the two communities.
        G = nx.Graph()
        G.add_edge(0, 2, weight=3)
        G.add_edge(1, 3, weight=1)
        red = {0, 1}
        comms = bipartite.lpawb_plus_communities(G, red, seed=0)
        _partition_covers_graph(comms, G)
        assert sorted(map(sorted, comms)) == [[0, 2], [1, 3]]
        assert bipartite.modularity(G, comms, red) == pytest.approx(0.375)

    def test_complete_bipartite_no_structure(self):
        # K_{2,2} has Q_B = 0 for every partition. LPAwb+ must return
        # *some* valid partition covering the graph.
        G = nx.complete_bipartite_graph(2, 2)
        comms = bipartite.lpawb_plus_communities(G, {0, 1}, seed=0)
        _partition_covers_graph(comms, G)
        assert bipartite.modularity(G, comms, {0, 1}) == pytest.approx(0.0)

    def test_returns_list_of_sets(self):
        G = nx.complete_bipartite_graph(3, 4)
        comms = bipartite.lpawb_plus_communities(G, set(range(3)), seed=7)
        assert isinstance(comms, list)
        assert all(isinstance(c, set) for c in comms)
        # Sorted by decreasing size
        sizes = [len(c) for c in comms]
        assert sizes == sorted(sizes, reverse=True)

    def test_seed_reproducibility(self):
        G = nx.davis_southern_women_graph()
        women = {n for n, d in G.nodes(data=True) if d["bipartite"] == 0}
        c1 = bipartite.lpawb_plus_communities(G, women, seed=123)
        c2 = bipartite.lpawb_plus_communities(G, women, seed=123)
        assert sorted(map(sorted, c1)) == sorted(map(sorted, c2))

    def test_southern_women_matches_barber_brim(self):
        # Beckett's LPAwb+ is designed to maximize Barber's Q_B; the
        # BRIM algorithm in Barber (2007) reports Q = 0.34554 as the
        # best partition on the Davis Southern Women network (Table I).
        # LPAwb+ is stochastic, so we run multiple seeds and check that
        # the best is at least as good.
        G = nx.davis_southern_women_graph()
        women = {n for n, d in G.nodes(data=True) if d["bipartite"] == 0}
        best = -float("inf")
        for s in range(20):
            comms = bipartite.lpawb_plus_communities(G, women, seed=s)
            best = max(best, bipartite.modularity(G, comms, women))
        assert best >= 0.34554 - 1e-4

    def test_directed_not_implemented(self):
        G = nx.DiGraph([(0, 2), (1, 3)])
        with pytest.raises(nx.NetworkXNotImplemented):
            bipartite.lpawb_plus_communities(G, {0, 1}, seed=0)

    def test_empty_graph(self):
        G = nx.Graph()
        assert bipartite.lpawb_plus_communities(G, set(), seed=0) == []

    def test_graph_with_isolates(self):
        # Graph with one edge and one isolated node on each side.
        G = nx.Graph()
        G.add_edge(0, 2)
        G.add_node(1)  # isolated red
        G.add_node(3)  # isolated blue
        comms = bipartite.lpawb_plus_communities(G, {0, 1}, seed=0)
        _partition_covers_graph(comms, G)

    def test_nodes_on_larger_side(self):
        # Passing the larger bipartite set as `nodes` should give the
        # same communities as passing the smaller set (the algorithm
        # internally uses the smaller side as "red").
        G = nx.Graph()
        G.add_edges_from([(0, 3), (0, 4), (1, 3), (1, 4), (2, 5), (2, 6)])
        small = {0, 1, 2}
        large = {3, 4, 5, 6}
        c_small = bipartite.lpawb_plus_communities(G, small, seed=11)
        c_large = bipartite.lpawb_plus_communities(G, large, seed=11)
        assert sorted(map(sorted, c_small)) == sorted(map(sorted, c_large))
