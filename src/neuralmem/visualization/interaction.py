"""NeuralMem V2.1 Graph Interaction — advanced selection, highlighting, and filtering.

GraphInteraction
    Central orchestrator that wires SelectionManager, HighlightEngine,
    and FilterManager together and drives smooth animation state.

SelectionManager
    Multi-select, lasso (polygon), and axis-aligned box selection.

HighlightEngine
    Highlight connected neighbours, paths between nodes, and clusters.

FilterManager
    Filter nodes by type, time range, relevance score, or custom predicate.

Animation support
    Smooth transitions via tweened position / opacity / size interpolation.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Protocol, Set, Tuple

from neuralmem.visualization.canvas import CanvasGraph, GraphEdge, GraphNode
from neuralmem.visualization.layout import Vector2D


# --------------------------------------------------------------------------- #
# Animation helpers
# --------------------------------------------------------------------------- #

@dataclass
class AnimationState:
    """Snapshot of a node's animated properties at a specific time."""

    x: float = 0.0
    y: float = 0.0
    size: float = 10.0
    opacity: float = 1.0
    color: str = "#4A90D9"


class EasingFn(Protocol):
    def __call__(self, t: float) -> float:
        ...


def linear_ease(t: float) -> float:
    return max(0.0, min(1.0, t))


def ease_in_out_quad(t: float) -> float:
    t = max(0.0, min(1.0, t))
    return 2 * t * t if t < 0.5 else -1 + (4 - 2 * t) * t


def ease_out_cubic(t: float) -> float:
    t = max(0.0, min(1.0, t))
    return 1 - (1 - t) ** 3


@dataclass
class Tween:
    """A single tween between two AnimationStates over a duration."""

    start: AnimationState
    end: AnimationState
    duration_ms: float = 300.0
    easing: EasingFn = field(default=linear_ease)
    elapsed_ms: float = 0.0

    def step(self, dt_ms: float) -> AnimationState:
        """Advance the tween by *dt_ms* and return the interpolated state."""
        self.elapsed_ms += dt_ms
        t = self.easing(self.elapsed_ms / self.duration_ms) if self.duration_ms > 0 else 1.0
        return AnimationState(
            x=_lerp(self.start.x, self.end.x, t),
            y=_lerp(self.start.y, self.end.y, t),
            size=_lerp(self.start.size, self.end.size, t),
            opacity=_lerp(self.start.opacity, self.end.opacity, t),
            color=self.end.color if t >= 1.0 else self.start.color,
        )

    @property
    def finished(self) -> bool:
        return self.elapsed_ms >= self.duration_ms


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


# --------------------------------------------------------------------------- #
# SelectionManager
# --------------------------------------------------------------------------- #

class SelectionManager:
    """Multi-select, lasso (polygon), and box selection for a CanvasGraph.

    All coordinates are in **canvas space** (not screen space).
    """

    def __init__(self, graph: CanvasGraph) -> None:
        self.graph = graph
        self.selected_ids: Set[str] = set()
        self._on_change: List[Callable[[Set[str]], None]] = []

    # -- basic multi-select -------------------------------------------------- #

    def select(self, node_ids: Set[str]) -> None:
        """Replace the current selection."""
        self.selected_ids = set(node_ids)
        self._sync_graph_state()
        self._notify()

    def toggle(self, node_id: str) -> None:
        """Toggle a single node in the selection set."""
        if node_id in self.selected_ids:
            self.selected_ids.discard(node_id)
        else:
            self.selected_ids.add(node_id)
        self._sync_graph_state()
        self._notify()

    def add_to_selection(self, node_ids: Set[str]) -> None:
        """Add nodes without clearing existing selection."""
        self.selected_ids |= set(node_ids)
        self._sync_graph_state()
        self._notify()

    def remove_from_selection(self, node_ids: Set[str]) -> None:
        """Remove nodes from the current selection."""
        self.selected_ids -= set(node_ids)
        self._sync_graph_state()
        self._notify()

    def clear(self) -> None:
        """Clear all selection."""
        self.selected_ids.clear()
        self._sync_graph_state()
        self._notify()

    def select_all(self) -> None:
        """Select every node in the graph."""
        self.selected_ids = set(self.graph.nodes.keys())
        self._sync_graph_state()
        self._notify()

    # -- box select -------------------------------------------------------- #

    def box_select(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        mode: str = "replace",
    ) -> Set[str]:
        """Select nodes whose centre lies inside the axis-aligned box.

        Parameters
        ----------
        mode : {"replace", "add", "toggle", "subtract"}
            How to combine with the existing selection.
        """
        min_x, max_x = (x1, x2) if x1 < x2 else (x2, x1)
        min_y, max_y = (y1, y2) if y1 < y2 else (y2, y1)
        inside = {
            nid
            for nid, n in self.graph.nodes.items()
            if min_x <= n.x <= max_x and min_y <= n.y <= max_y
        }
        self._apply_mode(inside, mode)
        return inside

    # -- lasso (polygon) select ---------------------------------------------- #

    def lasso_select(
        self,
        polygon: List[Tuple[float, float]],
        mode: str = "replace",
    ) -> Set[str]:
        """Select nodes whose centre lies inside a closed polygon.

        Parameters
        ----------
        polygon : list of (x, y)
            Vertices in canvas space.  The polygon is implicitly closed.
        mode : {"replace", "add", "toggle", "subtract"}
        """
        inside = {nid for nid, n in self.graph.nodes.items() if _point_in_polygon((n.x, n.y), polygon)}
        self._apply_mode(inside, mode)
        return inside

    # -- callbacks --------------------------------------------------------- #

    def on_change(self, callback: Callable[[Set[str]], None]) -> None:
        self._on_change.append(callback)

    def _notify(self) -> None:
        for cb in self._on_change:
            cb(set(self.selected_ids))

    # -- internals --------------------------------------------------------- #

    def _sync_graph_state(self) -> None:
        for n in self.graph.nodes.values():
            n.selected = n.id in self.selected_ids

    def _apply_mode(self, ids: Set[str], mode: str) -> None:
        if mode == "replace":
            self.selected_ids = set(ids)
        elif mode == "add":
            self.selected_ids |= ids
        elif mode == "toggle":
            for nid in ids:
                if nid in self.selected_ids:
                    self.selected_ids.discard(nid)
                else:
                    self.selected_ids.add(nid)
        elif mode == "subtract":
            self.selected_ids -= ids
        else:
            raise ValueError(f"Unknown selection mode: {mode!r}")
        self._sync_graph_state()
        self._notify()


# --------------------------------------------------------------------------- #
# HighlightEngine
# --------------------------------------------------------------------------- #

class HighlightEngine:
    """Highlight connected neighbours, paths, and clusters."""

    def __init__(self, graph: CanvasGraph) -> None:
        self.graph = graph
        self.highlighted_node_ids: Set[str] = set()
        self.highlighted_edge_ids: Set[str] = set()
        self._on_change: List[Callable[[Set[str], Set[str]], None]] = []

    # -- basic highlight --------------------------------------------------- #

    def highlight_nodes(self, node_ids: Set[str]) -> None:
        """Highlight a set of nodes (edges untouched)."""
        self.highlighted_node_ids = set(node_ids)
        self._sync_graph_state()
        self._notify()

    def highlight_edges(self, edge_ids: Set[str]) -> None:
        """Highlight a set of edges (nodes untouched)."""
        self.highlighted_edge_ids = set(edge_ids)
        self._sync_graph_state()
        self._notify()

    def clear(self) -> None:
        """Remove all highlights."""
        self.highlighted_node_ids.clear()
        self.highlighted_edge_ids.clear()
        self._sync_graph_state()
        self._notify()

    # -- neighbour highlight ----------------------------------------------- #

    def highlight_neighbours(self, node_id: str, depth: int = 1) -> None:
        """Highlight *node_id* and all nodes within *depth* hops.

        Also highlights every edge that connects two highlighted nodes.
        """
        visited: Set[str] = set()
        frontier = {node_id}
        for _ in range(depth):
            if not frontier:
                break
            visited |= frontier
            next_frontier: Set[str] = set()
            for nid in frontier:
                for nbr in self.graph.neighbours(nid):
                    if nbr not in visited:
                        next_frontier.add(nbr)
            frontier = next_frontier
        visited |= frontier

        edge_ids = {
            e.id
            for e in self.graph.edges.values()
            if e.source_id in visited and e.target_id in visited
        }
        self.highlighted_node_ids = visited
        self.highlighted_edge_ids = edge_ids
        self._sync_graph_state()
        self._notify()

    # -- path highlight ---------------------------------------------------- #

    def highlight_path(self, source_id: str, target_id: str) -> bool:
        """Highlight the shortest path between two nodes (BFS).

        Returns ``True`` if a path was found, otherwise ``False``.
        """
        if source_id not in self.graph.nodes or target_id not in self.graph.nodes:
            return False

        # BFS with parent tracking
        queue = [source_id]
        parent: Dict[str, Optional[str]] = {source_id: None}
        found = False
        while queue:
            current = queue.pop(0)
            if current == target_id:
                found = True
                break
            for nbr in self.graph.neighbours(current):
                if nbr not in parent:
                    parent[nbr] = current
                    queue.append(nbr)

        if not found:
            return False

        # Reconstruct node path
        path_nodes: List[str] = []
        cur: Optional[str] = target_id
        while cur is not None:
            path_nodes.append(cur)
            cur = parent[cur]
        path_nodes.reverse()

        # Find edges that belong to the path
        path_edges: Set[str] = set()
        for i in range(len(path_nodes) - 1):
            a, b = path_nodes[i], path_nodes[i + 1]
            for e in self.graph.edges.values():
                if (e.source_id == a and e.target_id == b) or (e.source_id == b and e.target_id == a):
                    path_edges.add(e.id)
                    break

        self.highlighted_node_ids = set(path_nodes)
        self.highlighted_edge_ids = path_edges
        self._sync_graph_state()
        self._notify()
        return True

    # -- cluster highlight ------------------------------------------------- #

    def highlight_cluster(self, node_id: str) -> Set[str]:
        """Highlight the entire connected component containing *node_id*."""
        if node_id not in self.graph.nodes:
            return set()
        visited: Set[str] = set()
        stack = [node_id]
        while stack:
            cur = stack.pop()
            if cur in visited:
                continue
            visited.add(cur)
            for nbr in self.graph.neighbours(cur):
                if nbr not in visited:
                    stack.append(nbr)

        edge_ids = {
            e.id
            for e in self.graph.edges.values()
            if e.source_id in visited and e.target_id in visited
        }
        self.highlighted_node_ids = visited
        self.highlighted_edge_ids = edge_ids
        self._sync_graph_state()
        self._notify()
        return visited

    # -- callbacks --------------------------------------------------------- #

    def on_change(self, callback: Callable[[Set[str], Set[str]], None]) -> None:
        self._on_change.append(callback)

    def _notify(self) -> None:
        for cb in self._on_change:
            cb(set(self.highlighted_node_ids), set(self.highlighted_edge_ids))

    def _sync_graph_state(self) -> None:
        for n in self.graph.nodes.values():
            n.highlighted = n.id in self.highlighted_node_ids
        for e in self.graph.edges.values():
            e.highlighted = e.id in self.highlighted_edge_ids


# --------------------------------------------------------------------------- #
# FilterManager
# --------------------------------------------------------------------------- #

class FilterManager:
    """Filter nodes by type, time range, relevance score, or custom predicate.

    Filtering does **not** remove nodes from the graph; it only toggles the
    ``visible`` flag (stored in ``node.metadata``) and optionally adjusts
    ``node.opacity`` so the renderer can fade filtered-out items.
    """

    def __init__(self, graph: CanvasGraph) -> None:
        self.graph = graph
        self._active_filters: List[str] = []
        self._on_change: List[Callable[[Set[str]], None]] = []

    # -- predicate-based filtering ----------------------------------------- #

    def filter_by_predicate(
        self,
        predicate: Callable[[GraphNode], bool],
        name: str = "predicate",
        fade_opacity: float = 0.15,
    ) -> Set[str]:
        """Show only nodes where *predicate* returns ``True``.

        Returns the set of visible node IDs.
        """
        visible: Set[str] = set()
        for nid, node in self.graph.nodes.items():
            if predicate(node):
                visible.add(nid)
                node.opacity = 1.0
                node.metadata["visible"] = True
            else:
                node.opacity = fade_opacity
                node.metadata["visible"] = False
        self._active_filters.append(name)
        self._notify(visible)
        return visible

    def filter_by_type(
        self,
        node_type: str,
        metadata_key: str = "type",
        fade_opacity: float = 0.15,
    ) -> Set[str]:
        """Show only nodes whose ``metadata[metadata_key]`` equals *node_type*."""
        return self.filter_by_predicate(
            lambda n: n.metadata.get(metadata_key) == node_type,
            name=f"type={node_type}",
            fade_opacity=fade_opacity,
        )

    def filter_by_time_range(
        self,
        min_ts: Optional[float] = None,
        max_ts: Optional[float] = None,
        metadata_key: str = "timestamp",
        fade_opacity: float = 0.15,
    ) -> Set[str]:
        """Show only nodes whose timestamp metadata lies inside the range."""

        def _in_range(node: GraphNode) -> bool:
            ts = node.metadata.get(metadata_key)
            if ts is None:
                return False
            if min_ts is not None and ts < min_ts:
                return False
            if max_ts is not None and ts > max_ts:
                return False
            return True

        return self.filter_by_predicate(
            _in_range,
            name=f"time_range",
            fade_opacity=fade_opacity,
        )

    def filter_by_relevance(
        self,
        min_score: float = 0.0,
        metadata_key: str = "relevance",
        fade_opacity: float = 0.15,
    ) -> Set[str]:
        """Show only nodes with relevance >= *min_score*."""
        return self.filter_by_predicate(
            lambda n: n.metadata.get(metadata_key, 0.0) >= min_score,
            name=f"relevance>={min_score}",
            fade_opacity=fade_opacity,
        )

    # -- reset ------------------------------------------------------------- #

    def reset(self) -> None:
        """Clear all filters and restore full visibility."""
        for node in self.graph.nodes.values():
            node.opacity = 1.0
            node.metadata["visible"] = True
        self._active_filters.clear()
        self._notify(set(self.graph.nodes.keys()))

    @property
    def active_filters(self) -> List[str]:
        return list(self._active_filters)

    # -- callbacks --------------------------------------------------------- #

    def on_change(self, callback: Callable[[Set[str]], None]) -> None:
        self._on_change.append(callback)

    def _notify(self, visible: Set[str]) -> None:
        for cb in self._on_change:
            cb(set(visible))


# --------------------------------------------------------------------------- #
# GraphInteraction
# --------------------------------------------------------------------------- #

class GraphInteraction:
    """Central orchestrator that wires SelectionManager, HighlightEngine,
    FilterManager, and drives smooth animation transitions.
    """

    def __init__(self, graph: CanvasGraph) -> None:
        self.graph = graph
        self.selection = SelectionManager(graph)
        self.highlight = HighlightEngine(graph)
        self.filter = FilterManager(graph)

        # Animation bookkeeping
        self._tweens: Dict[str, Tween] = {}
        self._animation_callbacks: List[Callable[[Dict[str, AnimationState]], None]] = []
        self._default_duration_ms: float = 300.0
        self._default_easing: EasingFn = ease_in_out_quad

    # -- animation API ------------------------------------------------------- #

    def animate_to(
        self,
        targets: Dict[str, AnimationState],
        duration_ms: Optional[float] = None,
        easing: Optional[EasingFn] = None,
    ) -> None:
        """Start a tween for each node in *targets* toward the given state."""
        duration = duration_ms if duration_ms is not None else self._default_duration_ms
        ease = easing if easing is not None else self._default_easing
        for nid, end_state in targets.items():
            node = self.graph.nodes.get(nid)
            if node is None:
                continue
            start = AnimationState(
                x=node.x,
                y=node.y,
                size=node.size,
                opacity=node.opacity,
                color=node.color,
            )
            self._tweens[nid] = Tween(start=start, end=end_state, duration_ms=duration, easing=ease)

    def animate_positions(
        self,
        positions: Dict[str, Vector2D],
        duration_ms: Optional[float] = None,
        easing: Optional[EasingFn] = None,
    ) -> None:
        """Convenience wrapper to animate only node positions."""
        targets = {
            nid: AnimationState(x=vec.x, y=vec.y, size=self.graph.nodes[nid].size,
                                opacity=self.graph.nodes[nid].opacity,
                                color=self.graph.nodes[nid].color)
            for nid, vec in positions.items()
            if nid in self.graph.nodes
        }
        self.animate_to(targets, duration_ms=duration_ms, easing=easing)

    def step_animations(self, dt_ms: float) -> Dict[str, AnimationState]:
        """Advance all active tweens by *dt_ms* and apply results to the graph.

        Returns a dict of the current animated states.
        """
        finished: List[str] = []
        current_states: Dict[str, AnimationState] = {}
        for nid, tween in self._tweens.items():
            state = tween.step(dt_ms)
            current_states[nid] = state
            node = self.graph.nodes.get(nid)
            if node is not None:
                node.x = state.x
                node.y = state.y
                node.size = state.size
                node.opacity = state.opacity
                if state.color != node.color and tween.finished:
                    node.color = state.color
            if tween.finished:
                finished.append(nid)
        for nid in finished:
            self._tweens.pop(nid, None)
        for cb in self._animation_callbacks:
            cb(current_states)
        return current_states

    def cancel_animations(self, node_ids: Optional[Set[str]] = None) -> None:
        """Cancel active tweens.  If *node_ids* is ``None``, cancel all."""
        if node_ids is None:
            self._tweens.clear()
        else:
            for nid in node_ids:
                self._tweens.pop(nid, None)

    def has_active_animations(self) -> bool:
        return bool(self._tweens)

    def on_animation_frame(self, callback: Callable[[Dict[str, AnimationState]], None]) -> None:
        self._animation_callbacks.append(callback)

    # -- convenience: focus on a node / selection -------------------------- #

    def focus_nodes(
        self,
        node_ids: Set[str],
        zoom_target: float = 2.0,
        duration_ms: Optional[float] = None,
    ) -> None:
        """Animate the viewport so the centroid of *node_ids* is centred
        and zoomed to *zoom_target*.

        This updates the graph's underlying ``InteractionModel.viewport``
        directly (no tweening on the viewport itself — only the nodes).
        """
        if not node_ids:
            return
        xs = [self.graph.nodes[nid].x for nid in node_ids if nid in self.graph.nodes]
        ys = [self.graph.nodes[nid].y for nid in node_ids if nid in self.graph.nodes]
        if not xs:
            return
        cx = sum(xs) / len(xs)
        cy = sum(ys) / len(ys)
        vp = self.graph.interaction.viewport
        vp.zoom = zoom_target
        vp.x = -cx
        vp.y = -cy

    # -- selection + highlight integration --------------------------------- #

    def select_and_highlight_neighbours(self, node_id: str, depth: int = 1) -> None:
        """Select a single node and highlight its neighbourhood."""
        self.selection.select({node_id})
        self.highlight.highlight_neighbours(node_id, depth=depth)

    def highlight_selected_path(self) -> bool:
        """If exactly two nodes are selected, highlight the shortest path between them."""
        if len(self.selection.selected_ids) != 2:
            return False
        a, b = tuple(self.selection.selected_ids)
        return self.highlight.highlight_path(a, b)

    # -- filter + selection integration ------------------------------------ #

    def select_visible(self) -> None:
        """Select all currently visible (non-filtered) nodes."""
        visible = {
            nid for nid, n in self.graph.nodes.items()
            if n.metadata.get("visible", True)
        }
        self.selection.select(visible)

    # -- serialization ----------------------------------------------------- #

    def to_dict(self) -> Dict[str, Any]:
        return {
            "selection": list(self.selection.selected_ids),
            "highlight_nodes": list(self.highlight.highlighted_node_ids),
            "highlight_edges": list(self.highlight.highlighted_edge_ids),
            "active_filters": list(self.filter.active_filters),
            "has_animations": self.has_active_animations(),
        }

    @classmethod
    def from_dict(cls, graph: CanvasGraph, data: Dict[str, Any]) -> "GraphInteraction":
        """Restore interaction state onto an existing graph."""
        inst = cls(graph)
        inst.selection.select(set(data.get("selection", [])))
        inst.highlight.highlight_nodes(set(data.get("highlight_nodes", [])))
        inst.highlight.highlight_edges(set(data.get("highlight_edges", [])))
        return inst


# --------------------------------------------------------------------------- #
# Geometry helpers
# --------------------------------------------------------------------------- #

def _point_in_polygon(point: Tuple[float, float], polygon: List[Tuple[float, float]]) -> bool:
    """Ray-casting algorithm for point-in-polygon test."""
    x, y = point
    inside = False
    n = len(polygon)
    if n < 3:
        return False
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        # Check if edge intersects the ray to the right
        if ((y1 > y) != (y2 > y)) and (x < (x2 - x1) * (y - y1) / (y2 - y1 + 1e-9) + x1):
            inside = not inside
    return inside
