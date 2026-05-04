"""NeuralMem V2.1 Canvas Visualization — force-directed graph layout for memory visualization.

CanvasGraph
    Container for nodes and edges with a pluggable LayoutEngine.

GraphNode
    Memory node with position, size, color, label.

GraphEdge
    Relationship edge between memories.

InteractionModel
    Zoom, pan, select, highlight.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from neuralmem.visualization.layout import LayoutEngine, Vector2D


@dataclass
class GraphNode:
    """A visual node representing a memory item in the canvas graph."""

    id: str
    label: str = ""
    x: float = 0.0
    y: float = 0.0
    size: float = 10.0
    color: str = "#4A90D9"
    shape: str = "circle"
    opacity: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    selected: bool = False
    highlighted: bool = False
    fixed: bool = False

    def distance_to(self, other: "GraphNode") -> float:
        """Euclidean distance to another node."""
        return math.hypot(self.x - other.x, self.y - other.y)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a plain dict."""
        return {
            "id": self.id,
            "label": self.label,
            "x": self.x,
            "y": self.y,
            "size": self.size,
            "color": self.color,
            "shape": self.shape,
            "opacity": self.opacity,
            "metadata": dict(self.metadata),
            "selected": self.selected,
            "highlighted": self.highlighted,
            "fixed": self.fixed,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GraphNode":
        """Deserialize from a plain dict."""
        return cls(
            id=data["id"],
            label=data.get("label", ""),
            x=data.get("x", 0.0),
            y=data.get("y", 0.0),
            size=data.get("size", 10.0),
            color=data.get("color", "#4A90D9"),
            shape=data.get("shape", "circle"),
            opacity=data.get("opacity", 1.0),
            metadata=dict(data.get("metadata", {})),
            selected=data.get("selected", False),
            highlighted=data.get("highlighted", False),
            fixed=data.get("fixed", False),
        )


@dataclass
class GraphEdge:
    """A visual edge representing a relationship between memory nodes."""

    id: str
    source_id: str
    target_id: str
    label: str = ""
    weight: float = 1.0
    color: str = "#999999"
    width: float = 1.0
    dashed: bool = False
    opacity: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    highlighted: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a plain dict."""
        return {
            "id": self.id,
            "source_id": self.source_id,
            "target_id": self.target_id,
            "label": self.label,
            "weight": self.weight,
            "color": self.color,
            "width": self.width,
            "dashed": self.dashed,
            "opacity": self.opacity,
            "metadata": dict(self.metadata),
            "highlighted": self.highlighted,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GraphEdge":
        """Deserialize from a plain dict."""
        return cls(
            id=data["id"],
            source_id=data["source_id"],
            target_id=data["target_id"],
            label=data.get("label", ""),
            weight=data.get("weight", 1.0),
            color=data.get("color", "#999999"),
            width=data.get("width", 1.0),
            dashed=data.get("dashed", False),
            opacity=data.get("opacity", 1.0),
            metadata=dict(data.get("metadata", {})),
            highlighted=data.get("highlighted", False),
        )


@dataclass
class Viewport:
    """2-D viewport with zoom and pan."""

    x: float = 0.0
    y: float = 0.0
    zoom: float = 1.0
    width: float = 800.0
    height: float = 600.0

    def to_dict(self) -> Dict[str, Any]:
        return {"x": self.x, "y": self.y, "zoom": self.zoom, "width": self.width, "height": self.height}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Viewport":
        return cls(
            x=data.get("x", 0.0),
            y=data.get("y", 0.0),
            zoom=data.get("zoom", 1.0),
            width=data.get("width", 800.0),
            height=data.get("height", 600.0),
        )


class InteractionModel:
    """Zoom, pan, select, and highlight interactions.

    This is a pure-Python model that tracks state.  A real frontend would
    mirror these values and send events back via callbacks.
    """

    def __init__(self, viewport: Optional[Viewport] = None) -> None:
        self.viewport = viewport or Viewport()
        self.selected_node_ids: Set[str] = set()
        self.highlighted_node_ids: Set[str] = set()
        self.highlighted_edge_ids: Set[str] = set()
        self._on_select: List[Callable[[Set[str]], None]] = []
        self._on_highlight: List[Callable[[Set[str], Set[str]], None]] = []

    def pan(self, dx: float, dy: float) -> None:
        """Pan the viewport by (dx, dy) in screen pixels."""
        self.viewport.x += dx / self.viewport.zoom
        self.viewport.y += dy / self.viewport.zoom

    def zoom_at(self, factor: float, cx: float = 0.0, cy: float = 0.0) -> None:
        """Zoom by *factor* around canvas point (cx, cy)."""
        old_zoom = self.viewport.zoom
        new_zoom = max(0.1, min(10.0, old_zoom * factor))
        # Zoom toward the pointer so the point stays stable
        self.viewport.x += (cx / old_zoom - cx / new_zoom)
        self.viewport.y += (cy / old_zoom - cy / new_zoom)
        self.viewport.zoom = new_zoom

    def select(self, node_ids: Set[str]) -> None:
        """Replace the current selection."""
        self.selected_node_ids = set(node_ids)
        for cb in self._on_select:
            cb(self.selected_node_ids)

    def toggle_select(self, node_id: str) -> None:
        """Toggle a single node in the selection set."""
        if node_id in self.selected_node_ids:
            self.selected_node_ids.discard(node_id)
        else:
            self.selected_node_ids.add(node_id)
        for cb in self._on_select:
            cb(self.selected_node_ids)

    def highlight(
        self,
        node_ids: Optional[Set[str]] = None,
        edge_ids: Optional[Set[str]] = None,
    ) -> None:
        """Replace the current highlight."""
        self.highlighted_node_ids = set(node_ids) if node_ids else set()
        self.highlighted_edge_ids = set(edge_ids) if edge_ids else set()
        for cb in self._on_highlight:
            cb(self.highlighted_node_ids, self.highlighted_edge_ids)

    def clear_selection(self) -> None:
        self.selected_node_ids.clear()
        for cb in self._on_select:
            cb(self.selected_node_ids)

    def clear_highlight(self) -> None:
        self.highlighted_node_ids.clear()
        self.highlighted_edge_ids.clear()
        for cb in self._on_highlight:
            cb(self.highlighted_node_ids, self.highlighted_edge_ids)

    def on_select(self, callback: Callable[[Set[str]], None]) -> None:
        self._on_select.append(callback)

    def on_highlight(self, callback: Callable[[Set[str], Set[str]], None]) -> None:
        self._on_highlight.append(callback)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "viewport": self.viewport.to_dict(),
            "selected_node_ids": list(self.selected_node_ids),
            "highlighted_node_ids": list(self.highlighted_node_ids),
            "highlighted_edge_ids": list(self.highlighted_edge_ids),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "InteractionModel":
        inst = cls(viewport=Viewport.from_dict(data.get("viewport", {})))
        inst.selected_node_ids = set(data.get("selected_node_ids", []))
        inst.highlighted_node_ids = set(data.get("highlighted_node_ids", []))
        inst.highlighted_edge_ids = set(data.get("highlighted_edge_ids", []))
        return inst


class CanvasGraph:
    """Force-directed graph container for memory visualization.

    Similar to Supermemory's d3-force Canvas, but fully mock-testable
    with no real frontend dependencies.
    """

    def __init__(
        self,
        nodes: Optional[List[GraphNode]] = None,
        edges: Optional[List[GraphEdge]] = None,
        layout: Optional[LayoutEngine] = None,
        interaction: Optional[InteractionModel] = None,
    ) -> None:
        self.nodes: Dict[str, GraphNode] = {}
        self.edges: Dict[str, GraphEdge] = {}
        self.layout = layout or LayoutEngine()
        self.interaction = interaction or InteractionModel()
        self._node_id_counter = 0
        self._edge_id_counter = 0

        if nodes:
            for n in nodes:
                self.add_node(n)
        if edges:
            for e in edges:
                self.add_edge(e)

    # ------------------------------------------------------------------ #
    # Node / Edge CRUD
    # ------------------------------------------------------------------ #

    def add_node(self, node: GraphNode) -> GraphNode:
        """Add or replace a node."""
        self.nodes[node.id] = node
        return node

    def remove_node(self, node_id: str) -> Optional[GraphNode]:
        """Remove a node and any incident edges."""
        node = self.nodes.pop(node_id, None)
        if node is None:
            return None
        # Remove dangling edges
        to_remove = [eid for eid, e in self.edges.items() if e.source_id == node_id or e.target_id == node_id]
        for eid in to_remove:
            self.edges.pop(eid, None)
        return node

    def get_node(self, node_id: str) -> Optional[GraphNode]:
        return self.nodes.get(node_id)

    def add_edge(self, edge: GraphEdge) -> Optional[GraphEdge]:
        """Add an edge if both endpoints exist."""
        if edge.source_id not in self.nodes or edge.target_id not in self.nodes:
            return None
        self.edges[edge.id] = edge
        return edge

    def remove_edge(self, edge_id: str) -> Optional[GraphEdge]:
        return self.edges.pop(edge_id, None)

    def get_edge(self, edge_id: str) -> Optional[GraphEdge]:
        return self.edges.get(edge_id)

    def auto_id(self, prefix: str = "node") -> str:
        self._node_id_counter += 1
        return f"{prefix}-{self._node_id_counter}"

    def auto_edge_id(self) -> str:
        self._edge_id_counter += 1
        return f"edge-{self._edge_id_counter}"

    # ------------------------------------------------------------------ #
    # Neighbourhood helpers
    # ------------------------------------------------------------------ #

    def neighbours(self, node_id: str) -> List[str]:
        """Return IDs of nodes connected to *node_id* by an edge."""
        result: Set[str] = set()
        for edge in self.edges.values():
            if edge.source_id == node_id:
                result.add(edge.target_id)
            elif edge.target_id == node_id:
                result.add(edge.source_id)
        return list(result)

    def incident_edges(self, node_id: str) -> List[GraphEdge]:
        """Return edges touching *node_id*."""
        return [e for e in self.edges.values() if e.source_id == node_id or e.target_id == node_id]

    def degree(self, node_id: str) -> int:
        return len(self.incident_edges(node_id))

    # ------------------------------------------------------------------ #
    # Layout
    # ------------------------------------------------------------------ #

    def apply_layout(self, iterations: int = 100) -> None:
        """Run the force simulation and write positions back to nodes."""
        positions = self.layout.simulate(self, iterations=iterations)
        for nid, vec in positions.items():
            node = self.nodes.get(nid)
            if node and not node.fixed:
                node.x = vec.x
                node.y = vec.y

    def center_graph(self) -> None:
        """Translate all nodes so the centroid is at (0, 0)."""
        if not self.nodes:
            return
        cx = sum(n.x for n in self.nodes.values()) / len(self.nodes)
        cy = sum(n.y for n in self.nodes.values()) / len(self.nodes)
        for n in self.nodes.values():
            n.x -= cx
            n.y -= cy

    def bounding_box(self) -> Tuple[float, float, float, float]:
        """Return (min_x, min_y, max_x, max_y)."""
        if not self.nodes:
            return (0.0, 0.0, 0.0, 0.0)
        xs = [n.x for n in self.nodes.values()]
        ys = [n.y for n in self.nodes.values()]
        return (min(xs), min(ys), max(xs), max(ys))

    # ------------------------------------------------------------------ #
    # Interaction helpers
    # ------------------------------------------------------------------ #

    def select_node(self, node_id: str) -> None:
        """Select a single node and clear any previous selection."""
        self.interaction.select({node_id})
        self._sync_selection_state()

    def highlight_neighbours(self, node_id: str) -> None:
        """Highlight a node, its neighbours, and the connecting edges."""
        nbrs = self.neighbours(node_id)
        edge_ids = {e.id for e in self.incident_edges(node_id)}
        self.interaction.highlight(node_ids={node_id, *nbrs}, edge_ids=edge_ids)
        self._sync_highlight_state()

    def _sync_selection_state(self) -> None:
        for n in self.nodes.values():
            n.selected = n.id in self.interaction.selected_node_ids

    def _sync_highlight_state(self) -> None:
        for n in self.nodes.values():
            n.highlighted = n.id in self.interaction.highlighted_node_ids
        for e in self.edges.values():
            e.highlighted = e.id in self.interaction.highlighted_edge_ids

    # ------------------------------------------------------------------ #
    # Serialization
    # ------------------------------------------------------------------ #

    def to_dict(self) -> Dict[str, Any]:
        return {
            "nodes": [n.to_dict() for n in self.nodes.values()],
            "edges": [e.to_dict() for e in self.edges.values()],
            "interaction": self.interaction.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CanvasGraph":
        nodes = [GraphNode.from_dict(d) for d in data.get("nodes", [])]
        edges = [GraphEdge.from_dict(d) for d in data.get("edges", [])]
        inst = cls(nodes=nodes, edges=edges)
        inst.interaction = InteractionModel.from_dict(data.get("interaction", {}))
        return inst

    def to_json(self, indent: Optional[int] = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_json(cls, raw: str) -> "CanvasGraph":
        return cls.from_dict(json.loads(raw))
