"""NeuralMem V2.1 Renderer — export CanvasGraph to JSON, HTML, and SVG.

No real frontend dependencies; everything is pure string generation.
"""
from __future__ import annotations

import html
import json
from typing import Any, Dict, List, Optional

from neuralmem.visualization.canvas import CanvasGraph, GraphEdge, GraphNode


class Renderer:
    """Export a CanvasGraph to multiple frontend-friendly formats."""

    # ------------------------------------------------------------------ #
    # JSON
    # ------------------------------------------------------------------ #

    def to_json(self, graph: CanvasGraph, indent: Optional[int] = 2) -> str:
        """Full JSON dump of the graph (nodes + edges + interaction state)."""
        return graph.to_json(indent=indent)

    def to_json_dict(self, graph: CanvasGraph) -> Dict[str, Any]:
        """Return the serializable dict (useful for further processing)."""
        return graph.to_dict()

    # ------------------------------------------------------------------ #
    # HTML (embedded D3-like force graph)
    # ------------------------------------------------------------------ #

    def to_html(
        self,
        graph: CanvasGraph,
        title: str = "NeuralMem Canvas",
        width: int = 800,
        height: int = 600,
        background_color: str = "#1a1a2e",
    ) -> str:
        """Generate a self-contained HTML page with a simple SVG renderer.

        The page includes a minimal JavaScript pan/zoom engine so it works
        in a browser without external CDN dependencies.
        """
        data_json = json.dumps(graph.to_dict(), indent=2)
        # Escape for safe JS embedding
        data_js = data_json.replace("\u003c", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")

        lines: List[str] = [
            "<!DOCTYPE html>",
            "<html lang='en'>",
            "<head>",
            "  <meta charset='UTF-8'>",
            f"  <title>{html.escape(title)}</title>",
            "  <style>",
            f"    body {{ margin: 0; background: {background_color}; overflow: hidden; }}",
            f"    #canvas {{ width: {width}px; height: {height}px; }}",
            "    .node { cursor: pointer; }",
            "    .edge { stroke-linecap: round; }",
            "  </style>",
            "</head>",
            "<body>",
            f'  <svg id="canvas" viewBox="0 0 {width} {height}"></svg>',
            "  <script>",
            f"    const graphData = {data_js};",
            "    const svg = document.getElementById('canvas');",
            "    const ns = 'http://www.w3.org/2000/svg';",
            "    let zoom = graphData.interaction.viewport.zoom || 1;",
            "    let panX = graphData.interaction.viewport.x || 0;",
            "    let panY = graphData.interaction.viewport.y || 0;",
            "    let dragging = null, lastX = 0, lastY = 0;",
            "",
            "    function toScreen(x, y) {",
            "      return { x: (x + panX) * zoom + " + str(width // 2) + ", y: (y + panY) * zoom + " + str(height // 2) + " };",
            "    }",
            "",
            "    function render() {",
            "      svg.innerHTML = '';",
            "      const g = document.createElementNS(ns, 'g');",
            "      // Edges",
            "      graphData.edges.forEach(e => {",
            "        const s = graphData.nodes.find(n=>n.id===e.source_id);",
            "        const t = graphData.nodes.find(n=>n.id===e.target_id);",
            "        if(!s||!t) return;",
            "        const p1 = toScreen(s.x, s.y);",
            "        const p2 = toScreen(t.x, t.y);",
            "        const line = document.createElementNS(ns, 'line');",
            "        line.setAttribute('x1', p1.x); line.setAttribute('y1', p1.y);",
            "        line.setAttribute('x2', p2.x); line.setAttribute('y2', p2.y);",
            "        line.setAttribute('stroke', e.color);",
            "        line.setAttribute('stroke-width', e.width * zoom);",
            "        line.setAttribute('opacity', e.opacity);",
            "        if(e.dashed) line.setAttribute('stroke-dasharray', '4,4');",
            "        line.classList.add('edge');",
            "        g.appendChild(line);",
            "      });",
            "      // Nodes",
            "      graphData.nodes.forEach(n => {",
            "        const p = toScreen(n.x, n.y);",
            "        const circle = document.createElementNS(ns, 'circle');",
            "        circle.setAttribute('cx', p.x);",
            "        circle.setAttribute('cy', p.y);",
            "        circle.setAttribute('r', n.size * zoom);",
            "        circle.setAttribute('fill', n.color);",
            "        circle.setAttribute('opacity', n.opacity);",
            "        circle.classList.add('node');",
            "        g.appendChild(circle);",
            "        if(n.label) {",
            "          const text = document.createElementNS(ns, 'text');",
            "          text.setAttribute('x', p.x);",
            "          text.setAttribute('y', p.y + n.size * zoom + 12);",
            "          text.setAttribute('text-anchor', 'middle');",
            "          text.setAttribute('fill', '#eee');",
            "          text.setAttribute('font-size', 12 * zoom);",
            "          text.textContent = n.label;",
            "          g.appendChild(text);",
            "        }",
            "      });",
            "      svg.appendChild(g);",
            "    }",
            "",
            "    svg.addEventListener('wheel', e => {",
            "      e.preventDefault();",
            "      const factor = e.deltaY > 0 ? 0.9 : 1.1;",
            "      zoom *= factor;",
            "      render();",
            "    });",
            "",
            "    svg.addEventListener('mousedown', e => {",
            "      dragging = 'pan'; lastX = e.clientX; lastY = e.clientY;",
            "    });",
            "    window.addEventListener('mousemove', e => {",
            "      if(!dragging) return;",
            "      const dx = (e.clientX - lastX) / zoom;",
            "      const dy = (e.clientY - lastY) / zoom;",
            "      panX += dx; panY += dy;",
            "      lastX = e.clientX; lastY = e.clientY;",
            "      render();",
            "    });",
            "    window.addEventListener('mouseup', () => dragging = null);",
            "",
            "    render();",
            "  </script>",
            "</body>",
            "</html>",
        ]
        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    # SVG
    # ------------------------------------------------------------------ #

    def to_svg(
        self,
        graph: CanvasGraph,
        width: int = 800,
        height: int = 600,
        background_color: str = "#1a1a2e",
    ) -> str:
        """Generate a static SVG string.

        Pan / zoom state from the interaction model is applied so the
        exported image matches the current view.
        """
        vp = graph.interaction.viewport
        cx = width / 2
        cy = height / 2

        def tx(x: float, y: float) -> Tuple[float, float]:
            sx = (x + vp.x) * vp.zoom + cx
            sy = (y + vp.y) * vp.zoom + cy
            return (sx, sy)

        lines: List[str] = [
            f'\u003csvg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}"\u003e',
            f'  <rect width="100%" height="100%" fill="{html.escape(background_color)}" />',
            '  <g id="graph">',
        ]

        # Edges first so they render behind nodes
        for edge in graph.edges.values():
            src = graph.nodes.get(edge.source_id)
            tgt = graph.nodes.get(edge.target_id)
            if src is None or tgt is None:
                continue
            x1, y1 = tx(src.x, src.y)
            x2, y2 = tx(tgt.x, tgt.y)
            dash = ' stroke-dasharray="4,4"' if edge.dashed else ""
            lines.append(
                f'    <line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" '
                f'stroke="{html.escape(edge.color)}" stroke-width="{edge.width * vp.zoom:.2f}" '
                f'opacity="{edge.opacity}"{dash} />'
            )

        for node in graph.nodes.values():
            sx, sy = tx(node.x, node.y)
            r = node.size * vp.zoom
            lines.append(
                f'    <circle cx="{sx:.2f}" cy="{sy:.2f}" r="{r:.2f}" '
                f'fill="{html.escape(node.color)}" opacity="{node.opacity}" />'
            )
            if node.label:
                lines.append(
                    f'    <text x="{sx:.2f}" y="{sy + r + 12:.2f}" '
                    f'text-anchor="middle" fill="#eee" font-size="{12 * vp.zoom:.1f}px">'
                    f'{html.escape(node.label)}</text>'
                )

        lines.extend(["  </g>", "</svg>"])
        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    # D3-compatible JSON (simplified)
    # ------------------------------------------------------------------ #

    def to_d3_json(self, graph: CanvasGraph, indent: Optional[int] = 2) -> str:
        """Export in D3-force-compatible format {nodes:[{id,...}], links:[{source,target,...}]}."""
        payload = {
            "nodes": [n.to_dict() for n in graph.nodes.values()],
            "links": [
                {
                    "source": e.source_id,
                    "target": e.target_id,
                    "id": e.id,
                    "label": e.label,
                    "weight": e.weight,
                    "color": e.color,
                    "width": e.width,
                    "dashed": e.dashed,
                    "opacity": e.opacity,
                    "metadata": dict(e.metadata),
                    "highlighted": e.highlighted,
                }
                for e in graph.edges.values()
            ],
        }
        return json.dumps(payload, indent=indent)
