"""Unit tests for NeuralMem V2.1 CanvasGraph — all mock-based.

Covers:
  • GraphNode / GraphEdge dataclass behaviour
  • CanvasGraph CRUD (nodes, edges, auto-id)
  • Neighbourhood helpers
  • InteractionModel (pan, zoom, select, highlight)
  • Serialization (to_dict, from_dict, JSON round-trip)
"""
from __future__ import annotations

import json
import math
from typing import Any, Dict, List, Set

import pytest

from neuralmem.visualization.canvas import (
    CanvasGraph,
    GraphEdge,
    GraphNode,
    InteractionModel,
    Viewport,
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def make_node(node_id: str = "n1", **kwargs: Any) -> GraphNode:
    defaults: Dict[str, Any] = {"label": node_id}
    defaults.update(kwargs)
    return GraphNode(id=node_id, **defaults)


def make_edge(edge_id: str = "e1", source: str = "n1", target: str = "n2", **kwargs: Any) -> GraphEdge:
    return GraphEdge(id=edge_id, source_id=source, target_id=target, **kwargs)


# =============================================================================
# GraphNode
# =============================================================================

class TestGraphNode:
    def test_default_values(self):
        n = GraphNode(id="a")
        assert n.x == 0.0
        assert n.y == 0.0
        assert n.size == 10.0
        assert n.color == "#4A90D9"
        assert n.selected is False

    def test_distance_to(self):
        a = GraphNode(id="a", x=0, y=0)
        b = GraphNode(id="b", x=3, y=4)
        assert a.distance_to(b) == pytest.approx(5.0)

    def test_to_dict_roundtrip(self):
        n = GraphNode(id="a", label="Alpha", x=1, y=2, metadata={"foo": "bar"})
        d = n.to_dict()
        restored = GraphNode.from_dict(d)
        assert restored.id == "a"
        assert restored.label == "Alpha"
        assert restored.metadata == {"foo": "bar"}


# =============================================================================
# GraphEdge
# =============================================================================

class TestGraphEdge:
    def test_default_values(self):
        e = GraphEdge(id="e", source_id="a", target_id="b")
        assert e.weight == 1.0
        assert e.dashed is False

    def test_to_dict_roundtrip(self):
        e = GraphEdge(id="e", source_id="a", target_id="b", label="rel", metadata={"k": 1})
        d = e.to_dict()
        restored = GraphEdge.from_dict(d)
        assert restored.label == "rel"
        assert restored.metadata == {"k": 1}


# =============================================================================
# CanvasGraph — CRUD
# =============================================================================

class TestCanvasGraphCrud:
    def test_add_and_get_node(self):
        g = CanvasGraph()
        n = make_node("n1")
        g.add_node(n)
        assert g.get_node("n1") is n

    def test_remove_node_and_dangling_edges(self):
        g = CanvasGraph()
        g.add_node(make_node("n1"))
        g.add_node(make_node("n2"))
        g.add_edge(make_edge("e1", "n1", "n2"))
        removed = g.remove_node("n1")
        assert removed is not None
        assert "n1" not in g.nodes
        assert "e1" not in g.edges

    def test_add_edge_missing_endpoint_returns_none(self):
        g = CanvasGraph()
        g.add_node(make_node("n1"))
        e = make_edge("e1", "n1", "n2")
        assert g.add_edge(e) is None

    def test_auto_id_increments(self):
        g = CanvasGraph()
        assert g.auto_id() == "node-1"
        assert g.auto_id() == "node-2"
        assert g.auto_edge_id() == "edge-1"

    def test_degree(self):
        g = CanvasGraph()
        for i in range(3):
            g.add_node(make_node(f"n{i}"))
        g.add_edge(make_edge("e0", "n0", "n1"))
        g.add_edge(make_edge("e1", "n0", "n2"))
        assert g.degree("n0") == 2
        assert g.degree("n1") == 1


# =============================================================================
# CanvasGraph — Neighbourhood
# =============================================================================

class TestCanvasGraphNeighbourhood:
    def test_neighbours(self):
        g = CanvasGraph()
        for i in range(4):
            g.add_node(make_node(f"n{i}"))
        g.add_edge(make_edge("e01", "n0", "n1"))
        g.add_edge(make_edge("e02", "n0", "n2"))
        assert set(g.neighbours("n0")) == {"n1", "n2"}

    def test_incident_edges(self):
        g = CanvasGraph()
        for i in range(3):
            g.add_node(make_node(f"n{i}"))
        e1 = g.add_edge(make_edge("e01", "n0", "n1"))
        e2 = g.add_edge(make_edge("e02", "n0", "n2"))
        assert g.incident_edges("n0") == [e1, e2]


# =============================================================================
# CanvasGraph — Layout helpers
# =============================================================================

class TestCanvasGraphLayoutHelpers:
    def test_bounding_box(self):
        g = CanvasGraph()
        g.add_node(make_node("a", x=-10, y=-20))
        g.add_node(make_node("b", x=30, y=40))
        assert g.bounding_box() == (-10.0, -20.0, 30.0, 40.0)

    def test_center_graph(self):
        g = CanvasGraph()
        g.add_node(make_node("a", x=0, y=0))
        g.add_node(make_node("b", x=10, y=10))
        g.center_graph()
        assert g.get_node("a").x == pytest.approx(-5.0)
        assert g.get_node("a").y == pytest.approx(-5.0)
        assert g.get_node("b").x == pytest.approx(5.0)
        assert g.get_node("b").y == pytest.approx(5.0)


# =============================================================================
# InteractionModel
# =============================================================================

class TestInteractionModel:
    def test_pan(self):
        im = InteractionModel(viewport=Viewport(x=0, y=0, zoom=2.0))
        im.pan(20, 10)
        assert im.viewport.x == pytest.approx(10.0)
        assert im.viewport.y == pytest.approx(5.0)

    def test_zoom_at(self):
        im = InteractionModel(viewport=Viewport(zoom=1.0))
        im.zoom_at(2.0, cx=100, cy=100)
        assert im.viewport.zoom == 2.0

    def test_select_and_callback(self):
        im = InteractionModel()
        calls: List[Set[str]] = []
        im.on_select(lambda s: calls.append(set(s)))
        im.select({"a", "b"})
        assert calls[-1] == {"a", "b"}
        im.toggle_select("c")
        assert calls[-1] == {"a", "b", "c"}
        im.toggle_select("a")
        assert calls[-1] == {"b", "c"}

    def test_highlight(self):
        im = InteractionModel()
        im.highlight({"a"}, {"e1"})
        assert im.highlighted_node_ids == {"a"}
        assert im.highlighted_edge_ids == {"e1"}
        im.clear_highlight()
        assert im.highlighted_node_ids == set()

    def test_to_dict_roundtrip(self):
        im = InteractionModel(viewport=Viewport(x=10, y=20, zoom=2.0))
        im.select({"n1"})
        im.highlight({"n1", "n2"}, {"e1"})
        d = im.to_dict()
        restored = InteractionModel.from_dict(d)
        assert restored.viewport.x == 10.0
        assert restored.selected_node_ids == {"n1"}
        assert restored.highlighted_edge_ids == {"e1"}


# =============================================================================
# CanvasGraph — Interaction sync
# =============================================================================

class TestCanvasGraphInteractionSync:
    def test_select_node_syncs_flag(self):
        g = CanvasGraph()
        g.add_node(make_node("n1"))
        g.add_node(make_node("n2"))
        g.select_node("n1")
        assert g.get_node("n1").selected is True
        assert g.get_node("n2").selected is False

    def test_highlight_neighbours(self):
        g = CanvasGraph()
        for i in range(3):
            g.add_node(make_node(f"n{i}"))
        g.add_edge(make_edge("e01", "n0", "n1"))
        g.add_edge(make_edge("e12", "n1", "n2"))
        g.highlight_neighbours("n1")
        assert g.get_node("n1").highlighted is True
        assert g.get_node("n0").highlighted is True
        assert g.get_node("n2").highlighted is True
        assert g.get_edge("e01").highlighted is True
        assert g.get_edge("e12").highlighted is True


# =============================================================================
# Serialization
# =============================================================================

class TestCanvasGraphSerialization:
    def test_to_dict_contains_nodes_and_edges(self):
        g = CanvasGraph()
        g.add_node(make_node("n1"))
        g.add_node(make_node("n2"))
        g.add_edge(make_edge("e1", "n1", "n2"))
        d = g.to_dict()
        assert len(d["nodes"]) == 2
        assert len(d["edges"]) == 1

    def test_json_roundtrip(self):
        g = CanvasGraph()
        g.add_node(make_node("n1", x=10, y=20, metadata={"k": "v"}))
        g.add_node(make_node("n2", x=-5, y=5))
        g.add_edge(make_edge("e1", "n1", "n2", label="rel"))
        raw = g.to_json()
        restored = CanvasGraph.from_json(raw)
        assert restored.get_node("n1").x == pytest.approx(10.0)
        assert restored.get_node("n1").metadata == {"k": "v"}
        assert restored.get_edge("e1").label == "rel"

    def test_from_dict_restores_interaction(self):
        g = CanvasGraph()
        g.add_node(make_node("n1"))
        g.select_node("n1")
        d = g.to_dict()
        restored = CanvasGraph.from_dict(d)
        assert restored.interaction.selected_node_ids == {"n1"}
