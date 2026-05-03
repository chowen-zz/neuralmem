"""Knowledge graph visualization export — D3, DOT, Cytoscape, Mermaid."""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from neuralmem.graph.knowledge_graph import KnowledgeGraph


class GraphVisualizer:
    """Export a KnowledgeGraph to various visualization formats.

    Supported formats:
    - D3.js JSON (nodes + links)
    - Graphviz DOT
    - Cytoscape.js JSON
    - Mermaid diagram syntax

    Each node includes: id, label, type, memory_count, importance.
    Each edge includes: source, target, relation_type, weight.
    """

    def __init__(self, graph: KnowledgeGraph) -> None:
        self._graph = graph

    # ------------------------------------------------------------------
    # D3.js
    # ------------------------------------------------------------------

    def to_d3_json(self) -> dict:
        """Export graph as D3.js compatible JSON with ``nodes`` and ``links`` arrays."""

        g = self._graph._graph
        nodes: list[dict] = []
        for nid, attrs in g.nodes(data=True):
            nodes.append({
                "id": nid,
                "label": attrs.get("name", nid),
                "type": attrs.get("entity_type", "unknown"),
                "memory_count": len(attrs.get("memory_ids", [])),
                "importance": attrs.get("attributes", {}).get(
                    "importance", 0.5
                ),
            })

        links: list[dict] = []
        for src, tgt, attrs in g.edges(data=True):
            links.append({
                "source": src,
                "target": tgt,
                "relation_type": attrs.get("relation_type", ""),
                "weight": attrs.get("weight", 1.0),
            })

        return {"nodes": nodes, "links": links}

    # ------------------------------------------------------------------
    # Graphviz DOT
    # ------------------------------------------------------------------

    def to_dot(self) -> str:
        """Export graph as Graphviz DOT format string."""
        g = self._graph._graph
        lines = ["digraph KnowledgeGraph {"]

        for nid, attrs in g.nodes(data=True):
            label = _escape_dot(attrs.get("name", nid))
            ntype = _escape_dot(attrs.get("entity_type", "unknown"))
            mem_count = len(attrs.get("memory_ids", []))
            importance = attrs.get("attributes", {}).get("importance", 0.5)
            lines.append(
                f'  "{nid}" [label="{label}" type="{ntype}" '
                f'memory_count={mem_count} importance={importance}];'
            )

        for src, tgt, attrs in g.edges(data=True):
            rtype = _escape_dot(attrs.get("relation_type", ""))
            weight = attrs.get("weight", 1.0)
            lines.append(
                f'  "{src}" -> "{tgt}" [label="{rtype}" weight={weight}];'
            )

        lines.append("}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Cytoscape.js
    # ------------------------------------------------------------------

    def to_cytoscape_json(self) -> dict:
        """Export graph as Cytoscape.js compatible JSON.

        Returns a dict with ``elements`` containing ``nodes`` and ``edges``.
        """
        g = self._graph._graph
        nodes: list[dict] = []
        for nid, attrs in g.nodes(data=True):
            nodes.append({
                "data": {
                    "id": nid,
                    "label": attrs.get("name", nid),
                    "type": attrs.get("entity_type", "unknown"),
                    "memory_count": len(attrs.get("memory_ids", [])),
                    "importance": attrs.get("attributes", {}).get(
                        "importance", 0.5
                    ),
                },
            })

        edges: list[dict] = []
        for idx, (src, tgt, attrs) in enumerate(g.edges(data=True)):
            edges.append({
                "data": {
                    "id": f"e{idx}",
                    "source": src,
                    "target": tgt,
                    "relation_type": attrs.get("relation_type", ""),
                    "weight": attrs.get("weight", 1.0),
                },
            })

        return {"elements": {"nodes": nodes, "edges": edges}}

    # ------------------------------------------------------------------
    # Mermaid
    # ------------------------------------------------------------------

    def to_mermaid(self) -> str:
        """Export graph as Mermaid diagram syntax.

        Uses ``graph LR`` (left-to-right) by default.
        """
        g = self._graph._graph
        lines = ["graph LR"]

        # Sanitise id/label for Mermaid (no special chars in node ids)
        id_map: dict[str, str] = {}
        for idx, (nid, attrs) in enumerate(g.nodes(data=True)):
            safe_id = f"N{idx}"
            id_map[nid] = safe_id
            label = _escape_mermaid(attrs.get("name", nid))
            ntype = attrs.get("entity_type", "unknown")
            mem_count = len(attrs.get("memory_ids", []))
            importance = attrs.get("attributes", {}).get("importance", 0.5)
            lines.append(
                f'    {safe_id}["{label}<br/>'
                f"type={ntype} mem={mem_count} imp={importance}\"]"
            )

        for src, tgt, attrs in g.edges(data=True):
            rtype = _escape_mermaid(attrs.get("relation_type", ""))
            weight = attrs.get("weight", 1.0)
            src_id = id_map.get(src, src)
            tgt_id = id_map.get(tgt, tgt)
            lines.append(
                f'    {src_id} -->|"{rtype} w={weight}"| {tgt_id}'
            )

        return "\n".join(lines)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _escape_dot(text: str) -> str:
    """Escape characters that are problematic in DOT labels."""
    return text.replace("\\", "\\\\").replace('"', '\\"')


def _escape_mermaid(text: str) -> str:
    """Escape characters that are problematic in Mermaid labels."""
    return text.replace('"', "'").replace("<", "").replace(">", "")
