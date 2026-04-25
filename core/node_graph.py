from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class NodeDefinition:
    type_name: str
    title: str
    category: str
    color: str
    outputs: Tuple[str, ...] = ("frame",)
    inputs: Tuple[str, ...] = ("frame",)
    default_params: Dict[str, Any] = field(default_factory=dict)
    frame_multiplier: int = 1


@dataclass
class GraphNode:
    node_id: str
    definition: NodeDefinition
    params: Dict[str, Any] = field(default_factory=dict)
    pos: Tuple[float, float] = (0.0, 0.0)


@dataclass
class GraphEdge:
    src_id: str
    dst_id: str


class NodeGraphManager:
    """Scalable graph state container for handlers/nodes and IO linking."""

    def __init__(self):
        self.registry: Dict[str, NodeDefinition] = {}
        self.nodes: Dict[str, GraphNode] = {}
        self.edges: List[GraphEdge] = []
        self._next_id = 0
        self._register_defaults()
        self._bootstrap_graph()

    def _register_defaults(self):
        self.register(
            NodeDefinition("input", "INPUT", "Core", "#334155", inputs=(), outputs=("frame", "time"))
        )
        self.register(
            NodeDefinition("output", "OUTPUT", "Core", "#334155", inputs=("frame",), outputs=())
        )
        self.register(
            NodeDefinition(
                "cv", "CV Interpolate", "Interpolation", "#1d4ed8", frame_multiplier=2
            )
        )
        self.register(
            NodeDefinition(
                "film", "FILM Interpolate", "Interpolation", "#0f766e", frame_multiplier=2
            )
        )
        self.register(
            NodeDefinition(
                "pixel_sort",
                "Pixel Sort",
                "Experimental",
                "#7c3aed",
                default_params={"threshold": 0.45, "direction": "horizontal", "strength": 0.8},
            )
        )
        self.register(
            NodeDefinition("time_value", "Time Value", "Generators", "#a16207", inputs=(), outputs=("time",))
        )

    def _bootstrap_graph(self):
        input_id = self.add_node("input", (20, 40))
        output_id = self.add_node("output", (320, 40))
        self.add_edge(input_id, output_id)

    def register(self, definition: NodeDefinition):
        self.registry[definition.type_name] = definition

    def add_node(self, type_name: str, pos: Tuple[float, float] = (0.0, 0.0)) -> str:
        definition = self.registry[type_name]
        node_id = f"n{self._next_id}"
        self._next_id += 1
        self.nodes[node_id] = GraphNode(node_id, definition, dict(definition.default_params), pos)
        return node_id

    def remove_node(self, node_id: str):
        if node_id not in self.nodes:
            return
        if self.nodes[node_id].definition.type_name in ("input", "output"):
            return
        self.nodes.pop(node_id)
        self.edges = [e for e in self.edges if e.src_id != node_id and e.dst_id != node_id]

    def add_edge(self, src_id: str, dst_id: str):
        if src_id == dst_id:
            return
        if src_id not in self.nodes or dst_id not in self.nodes:
            return
        if any(e.src_id == src_id and e.dst_id == dst_id for e in self.edges):
            return
        self.edges.append(GraphEdge(src_id, dst_id))

    def remove_edge(self, src_id: str, dst_id: str):
        self.edges = [e for e in self.edges if not (e.src_id == src_id and e.dst_id == dst_id)]

    def update_node_position(self, node_id: str, x: float, y: float):
        if node_id in self.nodes:
            self.nodes[node_id].pos = (x, y)

    def topological_layers(self) -> Dict[str, int]:
        incoming = {n: 0 for n in self.nodes}
        for e in self.edges:
            incoming[e.dst_id] = incoming.get(e.dst_id, 0) + 1

        queue = [n for n, d in incoming.items() if d == 0]
        layer = {n: 0 for n in queue}
        while queue:
            cur = queue.pop(0)
            cur_layer = layer.get(cur, 0)
            for e in [edge for edge in self.edges if edge.src_id == cur]:
                nxt = e.dst_id
                layer[nxt] = max(layer.get(nxt, 0), cur_layer + 1)
                incoming[nxt] -= 1
                if incoming[nxt] == 0:
                    queue.append(nxt)
        for n in self.nodes:
            layer.setdefault(n, 0)
        return layer

    def auto_layout(self):
        layers = self.topological_layers()
        grouped: Dict[int, List[str]] = {}
        for nid, l in layers.items():
            grouped.setdefault(l, []).append(nid)

        x_gap, y_gap = 230, 100
        for l, node_ids in grouped.items():
            node_ids.sort()
            for i, nid in enumerate(node_ids):
                self.update_node_position(nid, 40 + l * x_gap, 40 + i * y_gap)

    def calculate_frame_multiplier(self) -> int:
        mult = 1
        for nid in self.ordered_pipeline_nodes():
            mult *= max(1, self.nodes[nid].definition.frame_multiplier)
        return mult

    def ordered_pipeline_nodes(self) -> List[str]:
        input_id = self.find_by_type("input")
        output_id = self.find_by_type("output")
        if not input_id or not output_id:
            return []

        ordered = []
        cur = input_id
        seen = set()
        while cur and cur not in seen:
            seen.add(cur)
            next_edges = [e for e in self.edges if e.src_id == cur]
            if not next_edges:
                break
            nxt = sorted(next_edges, key=lambda e: e.dst_id)[0].dst_id
            if nxt == output_id:
                break
            ordered.append(nxt)
            cur = nxt
        return ordered

    def find_by_type(self, type_name: str) -> Optional[str]:
        for nid, node in self.nodes.items():
            if node.definition.type_name == type_name:
                return nid
        return None

    def list_processing_steps(self) -> List[str]:
        return [self.nodes[nid].definition.type_name for nid in self.ordered_pipeline_nodes()]
