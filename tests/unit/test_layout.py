"""Unit tests for NeuralMem V2.1 LayoutEngine — all mock-based.

Covers:
  • Vector2D arithmetic
  • Force simulation determinism (seeded RNG)
  • Energy monotonic decrease (loose check)
  • Circular and grid deterministic layouts
  • Fixed-node respect
"""
from __future__ import annotations

import math
from typing import Dict

import pytest

from neuralmem.visualization.canvas import CanvasGraph, GraphEdge, GraphNode
from neuralmem.visualization.layout import LayoutEngine, Vector2D


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def make_graph_triangle() -> CanvasGraph:
    """Return a graph with 3 nodes and 3 edges (triangle)."""
    g = CanvasGraph()
    for nid in ("a", "b", "c"):
        g.add_node(GraphNode(id=nid))
    g.add_edge(GraphEdge(id="e1", source_id="a", target_id="b"))
    g.add_edge(GraphEdge(id="e2", source_id="b", target_id="c"))
    g.add_edge(GraphEdge(id="e3", source_id="c", target_id="a"))
    return g


def make_graph_line(count: int = 4) -> CanvasGraph:
    """Return a simple chain graph."""
    g = CanvasGraph()
    for i in range(count):
        g.add_node(GraphNode(id=f"n{i}"))
    for i in range(count - 1):
        g.add_edge(GraphEdge(id=f"e{i}", source_id=f"n{i}", target_id=f"n{i+1}"))
    return g


# =============================================================================
# Vector2D
# =============================================================================

class TestVector2D:
    def test_add(self):
        a = Vector2D(1, 2)
        b = Vector2D(3, 4)
        assert (a + b).to_tuple() == (4, 6)

    def test_sub(self):
        a = Vector2D(5, 5)
        b = Vector2D(2, 1)
        assert (a - b).to_tuple() == (3, 4)

    def test_mul_scalar(self):
        v = Vector2D(2, 3)
        assert (v * 2).to_tuple() == (4, 6)

    def test_div_scalar(self):
        v = Vector2D(4, 6)
        assert (v / 2).to_tuple() == (2, 3)

    def test_length(self):
        assert Vector2D(3, 4).length() == pytest.approx(5.0)

    def test_normalize(self):
        v = Vector2D(3, 4).normalize()
        assert v.length() == pytest.approx(1.0)
        assert v.x == pytest.approx(0.6)
        assert v.y == pytest.approx(0.8)

    def test_normalize_zero(self):
        z = Vector2D(0, 0).normalize()
        assert z.length() == 0.0

    def test_from_tuple(self):
        assert Vector2D.from_tuple((7, 8)).to_tuple() == (7, 8)


# =============================================================================
# LayoutEngine — determinism
# =============================================================================

class TestLayoutEngineDeterminism:
    def test_same_seed_same_result(self):
        g = make_graph_triangle()
        engine_a = LayoutEngine(seed=42)
        engine_b = LayoutEngine(seed=42)
        pos_a = engine_a.simulate(g, iterations=50)
        pos_b = engine_b.simulate(g, iterations=50)
        for nid in ("a", "b", "c"):
            assert pos_a[nid].x == pytest.approx(pos_b[nid].x, abs=1e-9)
            assert pos_a[nid].y == pytest.approx(pos_b[nid].y, abs=1e-9)

    def test_different_seed_different_result(self):
        g = make_graph_triangle()
        engine_a = LayoutEngine(seed=1)
        engine_b = LayoutEngine(seed=999)
        pos_a = engine_a.simulate(g, iterations=50)
        pos_b = engine_b.simulate(g, iterations=50)
        # At least one coordinate should differ meaningfully
        diffs = [
            abs(pos_a[nid].x - pos_b[nid].x) + abs(pos_a[nid].y - pos_b[nid].y)
            for nid in ("a", "b", "c")
        ]
        assert max(diffs) > 1.0


# =============================================================================
# LayoutEngine — fixed nodes
# =============================================================================

class TestLayoutEngineFixedNodes:
    def test_fixed_node_unchanged(self):
        g = make_graph_triangle()
        g.nodes["a"].x = 50.0
        g.nodes["a"].y = 50.0
        g.nodes["a"].fixed = True
        engine = LayoutEngine(seed=42)
        pos = engine.simulate(g, iterations=20)
        assert pos["a"].x == pytest.approx(50.0)
        assert pos["a"].y == pytest.approx(50.0)


# =============================================================================
# LayoutEngine — energy
# =============================================================================

class TestLayoutEngineEnergy:
    def test_energy_decreases_over_iterations(self):
        g = make_graph_triangle()
        engine = LayoutEngine(seed=42)
        pos0 = engine.simulate(g, iterations=0)
        e0 = engine.energy(g, pos0)
        pos10 = engine.simulate(g, iterations=10)
        e10 = engine.energy(g, pos10)
        pos50 = engine.simulate(g, iterations=50)
        e50 = engine.energy(g, pos50)
        # Energy should generally drop as the system relaxes
        assert e50 < e10 < e0


# =============================================================================
# LayoutEngine — deterministic presets
# =============================================================================

class TestLayoutEnginePresets:
    def test_layout_circular(self):
        g = make_graph_triangle()
        engine = LayoutEngine()
        pos = engine.layout_circular(g, radius=100.0)
        assert len(pos) == 3
        for nid, vec in pos.items():
            assert vec.length() == pytest.approx(100.0)

    def test_layout_grid(self):
        g = make_graph_line(count=5)
        engine = LayoutEngine()
        pos = engine.layout_grid(g, spacing=50.0)
        assert len(pos) == 5
        # 5 nodes = 3 cols x 2 rows (ceil(sqrt(5)) = 3)
        xs = sorted({v.x for v in pos.values()})
        ys = sorted({v.y for v in pos.values()})
        assert xs == pytest.approx([0.0, 50.0, 100.0])
        assert ys == pytest.approx([0.0, 50.0])


# =============================================================================
# LayoutEngine — parameter tuning
# =============================================================================

class TestLayoutEngineParameters:
    def test_stronger_repulsion_spreads_nodes(self):
        g = make_graph_line(count=3)
        weak = LayoutEngine(seed=42, repulsion_strength=100)
        strong = LayoutEngine(seed=42, repulsion_strength=50000)
        pos_weak = weak.simulate(g, iterations=30)
        pos_strong = strong.simulate(g, iterations=30)
        spread_weak = max(
            (pos_weak["n0"] - pos_weak["n2"]).length(),
            (pos_weak["n1"] - pos_weak["n2"]).length(),
        )
        spread_strong = max(
            (pos_strong["n0"] - pos_strong["n2"]).length(),
            (pos_strong["n1"] - pos_strong["n2"]).length(),
        )
        assert spread_strong > spread_weak

    def test_shorter_spring_pulls_nodes_closer(self):
        g = make_graph_line(count=2)
        long_spring = LayoutEngine(seed=42, spring_length=200)
        short_spring = LayoutEngine(seed=42, spring_length=50)
        pos_long = long_spring.simulate(g, iterations=50)
        pos_short = short_spring.simulate(g, iterations=50)
        dist_long = (pos_long["n0"] - pos_long["n1"]).length()
        dist_short = (pos_short["n0"] - pos_short["n1"]).length()
        assert dist_short < dist_long
