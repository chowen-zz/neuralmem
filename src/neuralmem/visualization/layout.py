"""NeuralMem V2.1 LayoutEngine — force simulation (spring, repulsion, centering).

Pure-Python implementation with no frontend dependencies.  Designed to be
mock-testable and deterministic when a fixed random seed is used.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from neuralmem.visualization.canvas import CanvasGraph


@dataclass
class Vector2D:
    """Simple 2-D vector for force calculations."""

    x: float = 0.0
    y: float = 0.0

    def __add__(self, other: "Vector2D") -> "Vector2D":
        return Vector2D(self.x + other.x, self.y + other.y)

    def __sub__(self, other: "Vector2D") -> "Vector2D":
        return Vector2D(self.x - other.x, self.y - other.y)

    def __mul__(self, scalar: float) -> "Vector2D":
        return Vector2D(self.x * scalar, self.y * scalar)

    def __truediv__(self, scalar: float) -> "Vector2D":
        return Vector2D(self.x / scalar, self.y / scalar)

    def length(self) -> float:
        return math.hypot(self.x, self.y)

    def normalize(self) -> "Vector2D":
        """Return a unit vector; zero vector if length is 0."""
        mag = self.length()
        if mag == 0:
            return Vector2D(0.0, 0.0)
        return self / mag

    def to_tuple(self) -> Tuple[float, float]:
        return (self.x, self.y)

    @classmethod
    def from_tuple(cls, t: Tuple[float, float]) -> "Vector2D":
        return cls(t[0], t[1])


class LayoutEngine:
    """Force-directed layout using spring attraction + repulsion + centering.

    Parameters
    ----------
    spring_length : float
        Ideal edge length (default 100).
    spring_strength : float
        Hooke's-law constant for edges (default 0.05).
    repulsion_strength : float
        Coulomb-like repulsion between all node pairs (default 5000).
    center_strength : float
        Pull toward origin (default 0.01).
    damping : float
        Velocity decay per iteration (default 0.9).
    max_velocity : float
        Cap on node displacement per step (default 50).
    seed : int | None
        RNG seed for reproducible initial placement.
    """

    def __init__(
        self,
        spring_length: float = 100.0,
        spring_strength: float = 0.05,
        repulsion_strength: float = 5000.0,
        center_strength: float = 0.01,
        damping: float = 0.9,
        max_velocity: float = 50.0,
        seed: Optional[int] = None,
    ) -> None:
        self.spring_length = spring_length
        self.spring_strength = spring_strength
        self.repulsion_strength = repulsion_strength
        self.center_strength = center_strength
        self.damping = damping
        self.max_velocity = max_velocity
        self.rng = random.Random(seed)
        self._velocities: Dict[str, Vector2D] = {}

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def simulate(
        self,
        graph: "CanvasGraph",
        iterations: int = 100,
        width: float = 800.0,
        height: float = 600.0,
    ) -> Dict[str, Vector2D]:
        """Run force simulation and return new positions.

        Nodes that are already placed (x, y != 0, 0) keep their positions
        as starting points; isolated nodes are scattered randomly inside
        the canvas bounds.
        """
        positions: Dict[str, Vector2D] = {}
        for nid, node in graph.nodes.items():
            if node.x == 0.0 and node.y == 0.0:
                positions[nid] = Vector2D(
                    self.rng.uniform(-width / 2, width / 2),
                    self.rng.uniform(-height / 2, height / 2),
                )
            else:
                positions[nid] = Vector2D(node.x, node.y)
            self._velocities.setdefault(nid, Vector2D(0.0, 0.0))

        for _ in range(iterations):
            forces = self._compute_forces(graph, positions)
            for nid in positions:
                if graph.nodes.get(nid) and graph.nodes[nid].fixed:
                    continue
                vel = self._velocities[nid] + forces[nid]
                # Damping + velocity cap
                speed = vel.length()
                if speed > self.max_velocity:
                    vel = vel.normalize() * self.max_velocity
                vel = vel * self.damping
                self._velocities[nid] = vel
                positions[nid] = positions[nid] + vel

        return positions

    def reset(self) -> None:
        """Clear internal velocities."""
        self._velocities.clear()

    # ------------------------------------------------------------------ #
    # Force calculations
    # ------------------------------------------------------------------ #

    def _compute_forces(
        self,
        graph: "CanvasGraph",
        positions: Dict[str, Vector2D],
    ) -> Dict[str, Vector2D]:
        forces: Dict[str, Vector2D] = {nid: Vector2D(0.0, 0.0) for nid in positions}

        # 1. Repulsion (Coulomb-like, all pairs)
        node_ids = list(positions.keys())
        for i in range(len(node_ids)):
            for j in range(i + 1, len(node_ids)):
                a, b = node_ids[i], node_ids[j]
                delta = positions[a] - positions[b]
                dist = delta.length()
                if dist == 0:
                    # Jitter to avoid division by zero
                    jitter = Vector2D(self.rng.uniform(-1, 1), self.rng.uniform(-1, 1))
                    delta = jitter
                    dist = delta.length() or 1e-6
                # F = k / d^2
                force_mag = self.repulsion_strength / (dist * dist)
                force = delta.normalize() * force_mag
                forces[a] = forces[a] + force
                forces[b] = forces[b] - force

        # 2. Spring attraction along edges
        for edge in graph.edges.values():
            a, b = edge.source_id, edge.target_id
            if a not in positions or b not in positions:
                continue
            delta = positions[b] - positions[a]
            dist = delta.length()
            if dist == 0:
                continue
            # Hooke: F = k * (d - ideal)  (attractive when stretched)
            displacement = dist - self.spring_length
            force_mag = self.spring_strength * displacement
            force = delta.normalize() * force_mag
            forces[a] = forces[a] + force
            forces[b] = forces[b] - force

        # 3. Centering force toward origin
        for nid, pos in positions.items():
            center_force = pos * (-self.center_strength)
            forces[nid] = forces[nid] + center_force

        return forces

    def energy(self, graph: "CanvasGraph", positions: Dict[str, Vector2D]) -> float:
        """Return a rough measure of layout energy (lower = more stable)."""
        total = 0.0
        # Repulsion energy ~ k / d
        node_ids = list(positions.keys())
        for i in range(len(node_ids)):
            for j in range(i + 1, len(node_ids)):
                a, b = node_ids[i], node_ids[j]
                dist = (positions[a] - positions[b]).length() or 1e-6
                total += self.repulsion_strength / dist
        # Spring energy ~ 0.5 * k * (d - ideal)^2
        for edge in graph.edges.values():
            a, b = edge.source_id, edge.target_id
            if a not in positions or b not in positions:
                continue
            dist = (positions[a] - positions[b]).length()
            total += 0.5 * self.spring_strength * (dist - self.spring_length) ** 2
        # Centering energy ~ 0.5 * c * d^2
        for pos in positions.values():
            total += 0.5 * self.center_strength * pos.length() ** 2
        return total

    def layout_circular(
        self,
        graph: "CanvasGraph",
        radius: float = 200.0,
    ) -> Dict[str, Vector2D]:
        """Place nodes evenly around a circle (deterministic, no forces)."""
        node_ids = list(graph.nodes.keys())
        count = len(node_ids)
        if count == 0:
            return {}
        positions: Dict[str, Vector2D] = {}
        for idx, nid in enumerate(node_ids):
            angle = 2 * math.pi * idx / count
            positions[nid] = Vector2D(
                radius * math.cos(angle),
                radius * math.sin(angle),
            )
        return positions

    def layout_grid(
        self,
        graph: "CanvasGraph",
        spacing: float = 100.0,
    ) -> Dict[str, Vector2D]:
        """Place nodes on a regular grid (deterministic, no forces)."""
        node_ids = list(graph.nodes.keys())
        count = len(node_ids)
        if count == 0:
            return {}
        cols = math.ceil(math.sqrt(count))
        positions: Dict[str, Vector2D] = {}
        for idx, nid in enumerate(node_ids):
            row = idx // cols
            col = idx % cols
            positions[nid] = Vector2D(col * spacing, row * spacing)
        return positions
