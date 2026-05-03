"""Tests for GraphVisualizer export formats."""
from __future__ import annotations

import json
from unittest.mock import MagicMock

from neuralmem.core.types import Entity, Relation
from neuralmem.graph.knowledge_graph import KnowledgeGraph
from neuralmem.graph.visualization import GraphVisualizer

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_mock_storage(snapshot_data=None):
    storage = MagicMock()
    storage.load_graph_snapshot = MagicMock(return_value=snapshot_data)
    storage.save_graph_snapshot = MagicMock()
    storage.save_graph_nodes_incremental = MagicMock()
    storage.save_graph_edges_incremental = MagicMock()
    storage.load_graph_nodes = MagicMock(return_value=None)
    storage.load_graph_edges = MagicMock(return_value=None)
    return storage


def _sample_graph() -> KnowledgeGraph:
    """Create a small graph with two entities and one relation."""
    storage = _make_mock_storage()
    kg = KnowledgeGraph(storage)
    e1 = Entity(id="e1", name="Alice", entity_type="person")
    e2 = Entity(id="e2", name="Python", entity_type="technology")
    kg.upsert_entity(e1)
    kg.upsert_entity(e2)
    kg.add_relation(Relation(
        source_id="e1", target_id="e2",
        relation_type="uses", weight=0.9,
    ))
    kg.link_memory_to_entity("m1", "e1")
    kg.link_memory_to_entity("m2", "e1")
    return kg


# ---------------------------------------------------------------------------
# D3 JSON
# ---------------------------------------------------------------------------

class TestD3Json:
    def test_has_nodes_and_links_keys(self):
        vis = GraphVisualizer(_sample_graph())
        data = vis.to_d3_json()
        assert "nodes" in data
        assert "links" in data

    def test_node_count(self):
        vis = GraphVisualizer(_sample_graph())
        data = vis.to_d3_json()
        assert len(data["nodes"]) == 2

    def test_node_fields(self):
        vis = GraphVisualizer(_sample_graph())
        data = vis.to_d3_json()
        node = next(n for n in data["nodes"] if n["id"] == "e1")
        assert node["label"] == "Alice"
        assert node["type"] == "person"
        assert node["memory_count"] == 2
        assert isinstance(node["importance"], float)

    def test_link_fields(self):
        vis = GraphVisualizer(_sample_graph())
        data = vis.to_d3_json()
        assert len(data["links"]) == 1
        link = data["links"][0]
        assert link["source"] == "e1"
        assert link["target"] == "e2"
        assert link["relation_type"] == "uses"
        assert link["weight"] == 0.9

    def test_empty_graph(self):
        storage = _make_mock_storage()
        kg = KnowledgeGraph(storage)
        vis = GraphVisualizer(kg)
        data = vis.to_d3_json()
        assert data["nodes"] == []
        assert data["links"] == []

    def test_json_serializable(self):
        vis = GraphVisualizer(_sample_graph())
        data = vis.to_d3_json()
        text = json.dumps(data)
        assert isinstance(text, str)


# ---------------------------------------------------------------------------
# DOT
# ---------------------------------------------------------------------------

class TestDot:
    def test_starts_with_digraph(self):
        vis = GraphVisualizer(_sample_graph())
        dot = vis.to_dot()
        assert dot.startswith("digraph")

    def test_contains_node_declarations(self):
        vis = GraphVisualizer(_sample_graph())
        dot = vis.to_dot()
        assert '"e1"' in dot
        assert '"e2"' in dot

    def test_contains_edge(self):
        vis = GraphVisualizer(_sample_graph())
        dot = vis.to_dot()
        assert "->" in dot
        assert "uses" in dot

    def test_contains_node_attributes(self):
        vis = GraphVisualizer(_sample_graph())
        dot = vis.to_dot()
        assert "Alice" in dot
        assert "person" in dot

    def test_empty_graph_dot(self):
        storage = _make_mock_storage()
        kg = KnowledgeGraph(storage)
        vis = GraphVisualizer(kg)
        dot = vis.to_dot()
        assert "digraph" in dot
        assert dot.strip().endswith("}")


# ---------------------------------------------------------------------------
# Cytoscape JSON
# ---------------------------------------------------------------------------

class TestCytoscapeJson:
    def test_has_elements_key(self):
        vis = GraphVisualizer(_sample_graph())
        data = vis.to_cytoscape_json()
        assert "elements" in data

    def test_elements_has_nodes_and_edges(self):
        vis = GraphVisualizer(_sample_graph())
        data = vis.to_cytoscape_json()
        assert "nodes" in data["elements"]
        assert "edges" in data["elements"]

    def test_node_data_fields(self):
        vis = GraphVisualizer(_sample_graph())
        data = vis.to_cytoscape_json()
        node = next(
            n for n in data["elements"]["nodes"]
            if n["data"]["id"] == "e1"
        )
        assert node["data"]["label"] == "Alice"
        assert node["data"]["type"] == "person"
        assert node["data"]["memory_count"] == 2

    def test_edge_data_fields(self):
        vis = GraphVisualizer(_sample_graph())
        data = vis.to_cytoscape_json()
        edge = data["elements"]["edges"][0]
        assert edge["data"]["source"] == "e1"
        assert edge["data"]["target"] == "e2"
        assert edge["data"]["relation_type"] == "uses"
        assert edge["data"]["weight"] == 0.9

    def test_json_serializable(self):
        vis = GraphVisualizer(_sample_graph())
        data = vis.to_cytoscape_json()
        text = json.dumps(data)
        assert isinstance(text, str)


# ---------------------------------------------------------------------------
# Mermaid
# ---------------------------------------------------------------------------

class TestMermaid:
    def test_starts_with_graph(self):
        vis = GraphVisualizer(_sample_graph())
        mmd = vis.to_mermaid()
        assert mmd.startswith("graph LR")

    def test_contains_node_definitions(self):
        vis = GraphVisualizer(_sample_graph())
        mmd = vis.to_mermaid()
        assert "Alice" in mmd
        assert "Python" in mmd

    def test_contains_edge_arrow(self):
        vis = GraphVisualizer(_sample_graph())
        mmd = vis.to_mermaid()
        assert "-->" in mmd

    def test_contains_relation_type(self):
        vis = GraphVisualizer(_sample_graph())
        mmd = vis.to_mermaid()
        assert "uses" in mmd

    def test_empty_graph_mermaid(self):
        storage = _make_mock_storage()
        kg = KnowledgeGraph(storage)
        vis = GraphVisualizer(kg)
        mmd = vis.to_mermaid()
        assert mmd.startswith("graph LR")

    def test_safe_ids(self):
        """Ensure mermaid ids are safe (no special characters)."""
        vis = GraphVisualizer(_sample_graph())
        mmd = vis.to_mermaid()
        # Node ids should be N0, N1, etc.
        assert "N0" in mmd or "N1" in mmd


# ---------------------------------------------------------------------------
# Cross-format consistency
# ---------------------------------------------------------------------------

class TestCrossFormat:
    def test_all_formats_same_node_count(self):
        vis = GraphVisualizer(_sample_graph())
        d3 = vis.to_d3_json()
        cy = vis.to_cytoscape_json()
        assert len(d3["nodes"]) == len(cy["elements"]["nodes"])

    def test_all_formats_same_edge_count(self):
        vis = GraphVisualizer(_sample_graph())
        d3 = vis.to_d3_json()
        cy = vis.to_cytoscape_json()
        assert len(d3["links"]) == len(cy["elements"]["edges"])
