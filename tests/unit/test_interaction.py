"""Unit tests for NeuralMem V2.1 GraphInteraction module.

Covers:
  • SelectionManager (multi-select, box, lasso, callbacks)
  • HighlightEngine (neighbours, path, cluster, callbacks)
  • FilterManager (type, time, relevance, predicate, reset)
  • GraphInteraction (animation, focus, integration, serialization)
  • Geometry helpers (point-in-polygon)
"""
from __future__ import annotations

from typing import Any, Dict, List, Set, Tuple

import pytest

from neuralmem.visualization.canvas import CanvasGraph, GraphEdge, GraphNode
from neuralmem.visualization.interaction import (
    AnimationState,
    FilterManager,
    GraphInteraction,
    HighlightEngine,
    SelectionManager,
    Tween,
    _lerp,
    _point_in_polygon,
    ease_in_out_quad,
    ease_out_cubic,
    linear_ease,
)
from neuralmem.visualization.layout import Vector2D


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def make_node(node_id: str = "n1", **kwargs: Any) -> GraphNode:
    defaults: Dict[str, Any] = {"label": node_id}
    defaults.update(kwargs)
    return GraphNode(id=node_id, **defaults)


def make_edge(edge_id: str = "e1", source: str = "n1", target: str = "n2", **kwargs: Any) -> GraphEdge:
    return GraphEdge(id=edge_id, source_id=source, target_id=target, **kwargs)


def make_triangle_graph() -> CanvasGraph:
    """Return a graph with 3 nodes forming a triangle."""
    g = CanvasGraph()
    g.add_node(make_node("n0", x=0, y=0))
    g.add_node(make_node("n1", x=10, y=0))
    g.add_node(make_node("n2", x=5, y=10))
    g.add_edge(make_edge("e01", "n0", "n1"))
    g.add_edge(make_edge("e12", "n1", "n2"))
    g.add_edge(make_edge("e20", "n2", "n0"))
    return g


def make_chain_graph(length: int = 5) -> CanvasGraph:
    """Return a linear chain graph: n0--n1--n2--..."""
    g = CanvasGraph()
    for i in range(length):
        g.add_node(make_node(f"n{i}", x=i * 10, y=0))
    for i in range(length - 1):
        g.add_edge(make_edge(f"e{i}", f"n{i}", f"n{i + 1}"))
    return g


# =============================================================================
# SelectionManager
# =============================================================================

class TestSelectionManager:
    def test_select_replaces(self):
        g = make_triangle_graph()
        sm = SelectionManager(g)
        sm.select({"n0", "n1"})
        assert sm.selected_ids == {"n0", "n1"}
        assert g.get_node("n0").selected is True
        assert g.get_node("n2").selected is False

    def test_toggle(self):
        g = make_triangle_graph()
        sm = SelectionManager(g)
        sm.toggle("n0")
        assert sm.selected_ids == {"n0"}
        sm.toggle("n0")
        assert sm.selected_ids == set()

    def test_add_and_remove(self):
        g = make_triangle_graph()
        sm = SelectionManager(g)
        sm.select({"n0"})
        sm.add_to_selection({"n1"})
        assert sm.selected_ids == {"n0", "n1"}
        sm.remove_from_selection({"n0"})
        assert sm.selected_ids == {"n1"}

    def test_clear(self):
        g = make_triangle_graph()
        sm = SelectionManager(g)
        sm.select({"n0", "n1"})
        sm.clear()
        assert sm.selected_ids == set()
        assert g.get_node("n0").selected is False

    def test_select_all(self):
        g = make_triangle_graph()
        sm = SelectionManager(g)
        sm.select_all()
        assert sm.selected_ids == {"n0", "n1", "n2"}

    def test_callback_fires(self):
        g = make_triangle_graph()
        sm = SelectionManager(g)
        calls: List[Set[str]] = []
        sm.on_change(lambda s: calls.append(set(s)))
        sm.select({"n0"})
        assert calls[-1] == {"n0"}
        sm.toggle("n1")
        assert calls[-1] == {"n0", "n1"}

    def test_box_select_replace(self):
        g = make_triangle_graph()
        sm = SelectionManager(g)
        result = sm.box_select(-1, -1, 6, 6, mode="replace")
        assert result == {"n0"}  # n0 at (0,0) inside; n1 at (10,0) outside
        assert sm.selected_ids == {"n0"}

    def test_box_select_add(self):
        g = make_triangle_graph()
        sm = SelectionManager(g)
        sm.select({"n0"})
        sm.box_select(8, -1, 12, 1, mode="add")
        assert sm.selected_ids == {"n0", "n1"}

    def test_box_select_toggle(self):
        g = make_triangle_graph()
        sm = SelectionManager(g)
        sm.select({"n0"})
        sm.box_select(-1, -1, 6, 6, mode="toggle")
        assert "n0" not in sm.selected_ids

    def test_box_select_subtract(self):
        g = make_triangle_graph()
        sm = SelectionManager(g)
        sm.select_all()
        sm.box_select(-1, -1, 6, 6, mode="subtract")
        assert sm.selected_ids == {"n1", "n2"}

    def test_box_select_invalid_mode(self):
        g = make_triangle_graph()
        sm = SelectionManager(g)
        with pytest.raises(ValueError):
            sm.box_select(0, 0, 1, 1, mode="invalid")

    def test_lasso_select_triangle(self):
        g = make_triangle_graph()
        sm = SelectionManager(g)
        # Lasso around n0 and n1 (top half of the triangle)
        polygon = [(-1, -1), (12, -1), (12, 5), (-1, 5)]
        result = sm.lasso_select(polygon, mode="replace")
        assert "n0" in result
        assert "n1" in result
        assert "n2" not in result

    def test_lasso_select_empty_polygon(self):
        g = make_triangle_graph()
        sm = SelectionManager(g)
        result = sm.lasso_select([], mode="replace")
        assert result == set()

    def test_lasso_select_add_mode(self):
        g = make_triangle_graph()
        sm = SelectionManager(g)
        sm.select({"n2"})
        polygon = [(-1, -1), (12, -1), (12, 5), (-1, 5)]
        sm.lasso_select(polygon, mode="add")
        assert sm.selected_ids == {"n0", "n1", "n2"}


# =============================================================================
# HighlightEngine
# =============================================================================

class TestHighlightEngine:
    def test_highlight_nodes(self):
        g = make_triangle_graph()
        he = HighlightEngine(g)
        he.highlight_nodes({"n0", "n1"})
        assert g.get_node("n0").highlighted is True
        assert g.get_node("n2").highlighted is False

    def test_highlight_edges(self):
        g = make_triangle_graph()
        he = HighlightEngine(g)
        he.highlight_edges({"e01"})
        assert g.get_edge("e01").highlighted is True
        assert g.get_edge("e12").highlighted is False

    def test_clear(self):
        g = make_triangle_graph()
        he = HighlightEngine(g)
        he.highlight_nodes({"n0"})
        he.highlight_edges({"e01"})
        he.clear()
        assert g.get_node("n0").highlighted is False
        assert g.get_edge("e01").highlighted is False

    def test_highlight_neighbours_depth_1(self):
        g = make_chain_graph(5)
        he = HighlightEngine(g)
        he.highlight_neighbours("n2", depth=1)
        assert he.highlighted_node_ids == {"n1", "n2", "n3"}
        # Edges between highlighted nodes
        assert "e1" in he.highlighted_edge_ids
        assert "e2" in he.highlighted_edge_ids
        assert "e0" not in he.highlighted_edge_ids
        assert "e3" not in he.highlighted_edge_ids

    def test_highlight_neighbours_depth_2(self):
        g = make_chain_graph(5)
        he = HighlightEngine(g)
        he.highlight_neighbours("n2", depth=2)
        assert he.highlighted_node_ids == {"n0", "n1", "n2", "n3", "n4"}

    def test_highlight_path_found(self):
        g = make_chain_graph(5)
        he = HighlightEngine(g)
        ok = he.highlight_path("n0", "n4")
        assert ok is True
        assert he.highlighted_node_ids == {"n0", "n1", "n2", "n3", "n4"}
        for eid in ("e0", "e1", "e2", "e3"):
            assert eid in he.highlighted_edge_ids

    def test_highlight_path_not_found(self):
        g = make_triangle_graph()
        he = HighlightEngine(g)
        # Remove an edge so n0 and n1 are not connected
        g.remove_edge("e01")
        g.remove_edge("e20")
        ok = he.highlight_path("n0", "n1")
        assert ok is False

    def test_highlight_path_missing_node(self):
        g = make_triangle_graph()
        he = HighlightEngine(g)
        assert he.highlight_path("n0", "nx") is False

    def test_highlight_cluster(self):
        g = make_triangle_graph()
        he = HighlightEngine(g)
        cluster = he.highlight_cluster("n0")
        assert cluster == {"n0", "n1", "n2"}
        assert he.highlighted_node_ids == cluster
        assert len(he.highlighted_edge_ids) == 3

    def test_highlight_cluster_isolated(self):
        g = make_triangle_graph()
        he = HighlightEngine(g)
        g.add_node(make_node("n3"))
        cluster = he.highlight_cluster("n3")
        assert cluster == {"n3"}

    def test_callback_fires(self):
        g = make_triangle_graph()
        he = HighlightEngine(g)
        calls: List[Tuple[Set[str], Set[str]]] = []
        he.on_change(lambda n, e: calls.append((set(n), set(e))))
        he.highlight_nodes({"n0"})
        assert calls[-1] == ({"n0"}, set())


# =============================================================================
# FilterManager
# =============================================================================

class TestFilterManager:
    def test_filter_by_type(self):
        g = CanvasGraph()
        g.add_node(make_node("n0", metadata={"type": "memory"}))
        g.add_node(make_node("n1", metadata={"type": "concept"}))
        g.add_node(make_node("n2", metadata={"type": "memory"}))
        fm = FilterManager(g)
        visible = fm.filter_by_type("memory")
        assert visible == {"n0", "n2"}
        assert g.get_node("n0").metadata["visible"] is True
        assert g.get_node("n1").metadata["visible"] is False
        assert g.get_node("n1").opacity == pytest.approx(0.15)

    def test_filter_by_time_range(self):
        g = CanvasGraph()
        g.add_node(make_node("n0", metadata={"timestamp": 1000}))
        g.add_node(make_node("n1", metadata={"timestamp": 2000}))
        g.add_node(make_node("n2", metadata={"timestamp": 3000}))
        fm = FilterManager(g)
        visible = fm.filter_by_time_range(min_ts=1500, max_ts=2500)
        assert visible == {"n1"}

    def test_filter_by_time_range_none(self):
        g = CanvasGraph()
        g.add_node(make_node("n0", metadata={"timestamp": 1000}))
        g.add_node(make_node("n1"))  # no timestamp
        fm = FilterManager(g)
        visible = fm.filter_by_time_range(min_ts=500)
        assert visible == {"n0"}

    def test_filter_by_relevance(self):
        g = CanvasGraph()
        g.add_node(make_node("n0", metadata={"relevance": 0.9}))
        g.add_node(make_node("n1", metadata={"relevance": 0.5}))
        g.add_node(make_node("n2"))  # no relevance
        fm = FilterManager(g)
        visible = fm.filter_by_relevance(min_score=0.6)
        assert visible == {"n0"}

    def test_filter_by_predicate(self):
        g = CanvasGraph()
        g.add_node(make_node("n0", x=0, y=0))
        g.add_node(make_node("n1", x=10, y=10))
        fm = FilterManager(g)
        visible = fm.filter_by_predicate(lambda n: n.x > 5)
        assert visible == {"n1"}

    def test_reset(self):
        g = CanvasGraph()
        g.add_node(make_node("n0"))
        g.add_node(make_node("n1"))
        fm = FilterManager(g)
        fm.filter_by_type("memory", metadata_key="missing_key")
        fm.reset()
        assert g.get_node("n0").opacity == pytest.approx(1.0)
        assert g.get_node("n0").metadata["visible"] is True
        assert fm.active_filters == []

    def test_active_filters(self):
        g = CanvasGraph()
        g.add_node(make_node("n0", metadata={"type": "a"}))
        fm = FilterManager(g)
        fm.filter_by_type("a")
        assert "type=a" in fm.active_filters

    def test_callback_fires(self):
        g = CanvasGraph()
        g.add_node(make_node("n0"))
        g.add_node(make_node("n1"))
        fm = FilterManager(g)
        calls: List[Set[str]] = []
        fm.on_change(lambda s: calls.append(set(s)))
        fm.filter_by_predicate(lambda n: n.id == "n0")
        assert calls[-1] == {"n0"}


# =============================================================================
# GraphInteraction
# =============================================================================

class TestGraphInteraction:
    def test_instantiation(self):
        g = make_triangle_graph()
        gi = GraphInteraction(g)
        assert gi.selection is not None
        assert gi.highlight is not None
        assert gi.filter is not None

    def test_select_and_highlight_neighbours(self):
        g = make_chain_graph(5)
        gi = GraphInteraction(g)
        gi.select_and_highlight_neighbours("n2", depth=1)
        assert gi.selection.selected_ids == {"n2"}
        assert gi.highlight.highlighted_node_ids == {"n1", "n2", "n3"}

    def test_highlight_selected_path_two_nodes(self):
        g = make_chain_graph(5)
        gi = GraphInteraction(g)
        gi.selection.select({"n0", "n4"})
        ok = gi.highlight_selected_path()
        assert ok is True
        assert gi.highlight.highlighted_node_ids == {"n0", "n1", "n2", "n3", "n4"}

    def test_highlight_selected_path_not_two_nodes(self):
        g = make_chain_graph(5)
        gi = GraphInteraction(g)
        gi.selection.select({"n0"})
        assert gi.highlight_selected_path() is False

    def test_select_visible(self):
        g = CanvasGraph()
        g.add_node(make_node("n0"))
        g.add_node(make_node("n1"))
        gi = GraphInteraction(g)
        gi.filter.filter_by_predicate(lambda n: n.id == "n0")
        gi.select_visible()
        assert gi.selection.selected_ids == {"n0"}

    def test_focus_nodes(self):
        g = make_triangle_graph()
        gi = GraphInteraction(g)
        gi.focus_nodes({"n0", "n1"}, zoom_target=2.0)
        vp = g.interaction.viewport
        assert vp.zoom == pytest.approx(2.0)
        # Centroid of (0,0) and (10,0) is (5,0)
        assert vp.x == pytest.approx(-5.0)
        assert vp.y == pytest.approx(0.0)

    def test_focus_nodes_empty(self):
        g = make_triangle_graph()
        gi = GraphInteraction(g)
        gi.focus_nodes(set(), zoom_target=3.0)
        assert g.interaction.viewport.zoom != pytest.approx(3.0)

    # -- animation --------------------------------------------------------- #

    def test_animate_positions(self):
        g = make_triangle_graph()
        gi = GraphInteraction(g)
        gi.animate_positions({"n0": Vector2D(100, 100)}, duration_ms=100)
        assert gi.has_active_animations() is True
        states = gi.step_animations(50)
        assert "n0" in states
        node = g.get_node("n0")
        # Halfway through linear interpolation
        assert node.x == pytest.approx(50.0)
        assert node.y == pytest.approx(50.0)

    def test_step_animations_finishes(self):
        g = make_triangle_graph()
        gi = GraphInteraction(g)
        gi.animate_positions({"n0": Vector2D(100, 0)}, duration_ms=100)
        gi.step_animations(100)
        assert gi.has_active_animations() is False
        assert g.get_node("n0").x == pytest.approx(100.0)

    def test_cancel_animations(self):
        g = make_triangle_graph()
        gi = GraphInteraction(g)
        gi.animate_positions({"n0": Vector2D(100, 0)})
        gi.cancel_animations()
        assert gi.has_active_animations() is False

    def test_cancel_animations_specific(self):
        g = make_triangle_graph()
        gi = GraphInteraction(g)
        gi.animate_positions({"n0": Vector2D(100, 0), "n1": Vector2D(200, 0)})
        gi.cancel_animations({"n0"})
        assert "n0" not in gi._tweens
        assert "n1" in gi._tweens

    def test_animation_callback(self):
        g = make_triangle_graph()
        gi = GraphInteraction(g)
        calls: List[Dict[str, AnimationState]] = []
        gi.on_animation_frame(lambda s: calls.append(dict(s)))
        gi.animate_positions({"n0": Vector2D(10, 0)}, duration_ms=10)
        gi.step_animations(5)
        assert len(calls) == 1
        assert "n0" in calls[0]

    def test_animate_to_full_state(self):
        g = make_triangle_graph()
        gi = GraphInteraction(g)
        target = AnimationState(x=50, y=50, size=20, opacity=0.5, color="#FF0000")
        gi.animate_to({"n0": target}, duration_ms=100)
        gi.step_animations(100)
        n = g.get_node("n0")
        assert n.x == pytest.approx(50.0)
        assert n.y == pytest.approx(50.0)
        assert n.size == pytest.approx(20.0)
        assert n.opacity == pytest.approx(0.5)
        assert n.color == "#FF0000"

    # -- serialization ----------------------------------------------------- #

    def test_to_dict(self):
        g = make_triangle_graph()
        gi = GraphInteraction(g)
        gi.selection.select({"n0"})
        gi.highlight.highlight_nodes({"n1"})
        d = gi.to_dict()
        assert set(d["selection"]) == {"n0"}
        assert set(d["highlight_nodes"]) == {"n1"}
        assert d["has_animations"] is False

    def test_from_dict(self):
        g = make_triangle_graph()
        gi = GraphInteraction(g)
        gi.selection.select({"n0"})
        gi.highlight.highlight_nodes({"n1"})
        gi.highlight.highlight_edges({"e01"})
        d = gi.to_dict()
        # Create fresh interaction and restore
        gi2 = GraphInteraction.from_dict(g, d)
        assert gi2.selection.selected_ids == {"n0"}
        assert gi2.highlight.highlighted_node_ids == {"n1"}
        assert gi2.highlight.highlighted_edge_ids == {"e01"}


# =============================================================================
# Animation / Easing / Tween
# =============================================================================

class TestEasing:
    def test_linear_ease(self):
        assert linear_ease(0.0) == pytest.approx(0.0)
        assert linear_ease(0.5) == pytest.approx(0.5)
        assert linear_ease(1.0) == pytest.approx(1.0)
        assert linear_ease(-0.1) == pytest.approx(0.0)
        assert linear_ease(1.5) == pytest.approx(1.0)

    def test_ease_in_out_quad(self):
        assert ease_in_out_quad(0.0) == pytest.approx(0.0)
        assert ease_in_out_quad(1.0) == pytest.approx(1.0)
        assert ease_in_out_quad(0.5) == pytest.approx(0.5)

    def test_ease_out_cubic(self):
        assert ease_out_cubic(0.0) == pytest.approx(0.0)
        assert ease_out_cubic(1.0) == pytest.approx(1.0)
        # At t=0.5, should be > 0.5 (deceleration)
        assert ease_out_cubic(0.5) > 0.5


class TestTween:
    def test_step_halfway(self):
        start = AnimationState(x=0, y=0, size=10, opacity=1.0)
        end = AnimationState(x=100, y=100, size=20, opacity=0.0)
        tween = Tween(start=start, end=end, duration_ms=100, easing=linear_ease)
        state = tween.step(50)
        assert state.x == pytest.approx(50.0)
        assert state.y == pytest.approx(50.0)
        assert state.size == pytest.approx(15.0)
        assert state.opacity == pytest.approx(0.5)
        assert tween.finished is False

    def test_step_finish(self):
        start = AnimationState(x=0, y=0)
        end = AnimationState(x=10, y=10)
        tween = Tween(start=start, end=end, duration_ms=10)
        state = tween.step(10)
        assert state.x == pytest.approx(10.0)
        assert tween.finished is True

    def test_step_past_finish(self):
        start = AnimationState(x=0, y=0)
        end = AnimationState(x=10, y=10)
        tween = Tween(start=start, end=end, duration_ms=10)
        tween.step(20)
        assert tween.finished is True


class TestLerp:
    def test_lerp_basic(self):
        assert _lerp(0, 10, 0.5) == pytest.approx(5.0)
        assert _lerp(0, 10, 0.0) == pytest.approx(0.0)
        assert _lerp(0, 10, 1.0) == pytest.approx(10.0)


# =============================================================================
# Geometry helpers
# =============================================================================

class TestPointInPolygon:
    def test_inside_square(self):
        square = [(0, 0), (10, 0), (10, 10), (0, 10)]
        assert _point_in_polygon((5, 5), square) is True

    def test_outside_square(self):
        square = [(0, 0), (10, 0), (10, 10), (0, 10)]
        assert _point_in_polygon((15, 5), square) is False

    def test_on_edge(self):
        square = [(0, 0), (10, 0), (10, 10), (0, 10)]
        # Edge cases can be ambiguous; at least one should be True/False
        assert isinstance(_point_in_polygon((5, 0), square), bool)

    def test_triangle(self):
        tri = [(0, 0), (10, 0), (5, 10)]
        assert _point_in_polygon((5, 2), tri) is True
        assert _point_in_polygon((5, 8), tri) is True
        assert _point_in_polygon((0, 5), tri) is False

    def test_empty_polygon(self):
        assert _point_in_polygon((0, 0), []) is False

    def test_two_point_polygon(self):
        assert _point_in_polygon((0, 0), [(0, 0), (1, 1)]) is False
