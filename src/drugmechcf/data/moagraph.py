"""
Mechanism-of-Action as a Graph
"""

from collections import defaultdict
import copy
from typing import Dict, List, Optional, Set, Union

import networkx as nx


# -----------------------------------------------------------------------------
#   Classes
# -----------------------------------------------------------------------------


class MoaGraph(nx.MultiDiGraph):
    """
    A graph representing a Mechanism-of-Action. More generally, a relatively small Multi-Di-Graph.
    This class just adds some convenience methods, and defines some conventions for Node, Edge.

    The graph may, optionally, define a name (str) with attribute 'graph_name'.

    The graph may, optionally, define a root-node and a sink-node.
        self.graph["root_node"]: Optional[str]
        self.graph["sink_node"]: Optional[str]

    Each Node has a unique Node-ID, which is used as the ref to the Node object in the graph.
    Each Node has attributes:
        EntityType: str. Same as NodeType
        name: str. Name for this node.
        ... may have other attributes ...

    Each edge has a Key = RelationName.
    """

    MERGED_EDGE_SEP = " :: "

    def __init__(self, graph_name: str = None, **attr):
        super().__init__(incoming_graph_data=None, multigraph_input=None, graph_name=graph_name, **attr)

        # EntityType => [node1, ...]
        self.entity_type_nodes: Dict[str, List[str]] = defaultdict(list)

        # Set of relations (relation-names)
        self.relations = set()

        return

    def copy(self, as_view=False):
        """
        Override this into a deepcopy, to make sure all internal structures are copied.
        Algorithms like `nx.transitive_closure_dag(G)` make a copy of the graph.
        """
        if as_view:
            return nx.graphviews.generic_graph_view(self)

        return copy.deepcopy(self)

    def reduce_graph(self, keep_nodes: Union[Set[str], Dict[str, str]]) -> "MoaGraph":
        """
        Returns a copy of the graph in which all nodes not in `keep_nodes` have been
        removed using `remove_node_merge_edges()`.
        """
        # noinspection PyTypeChecker
        moa_reduced: MoaGraph = self.copy()
        ref_nodes = list(self.nodes())
        for node in ref_nodes:
            if node not in keep_nodes:
                moa_reduced.remove_node_merge_edges(node)

        return moa_reduced

    def set_root_node(self, node: str):
        self.graph["root_node"] = node
        return

    def get_root_node(self) -> Optional[str]:
        return self.graph.get("root_node")

    def set_sink_node(self, node: str):
        self.graph["sink_node"] = node
        return

    def get_sink_node(self) -> Optional[str]:
        return self.graph.get("sink_node")

    def has_root_and_sink_nodes_defined(self):
        return self.get_root_node() is not None and self.get_sink_node() is not None

    # noinspection PyMethodOverriding,PyPep8Naming
    def add_node(self, node: str, *, EntityType: str, name: str, **attr):

        if node not in self.entity_type_nodes[EntityType]:
            self.entity_type_nodes[EntityType].append(node)
            super().add_node(node, EntityType=EntityType, name=name, **attr)

        return

    def remove_node(self, node: str):
        # Remove from `entity_type_nodes`
        try:
            node_type = self.get_node_entity_type(node)
            self.entity_type_nodes[node_type].remove(node)
            if not self.entity_type_nodes[node_type]:
                del self.entity_type_nodes[node_type]
        except KeyError:
            raise nx.NodeNotFound(node)

        super().remove_node(node)
        return

    def remove_node_merge_edges(self, node: str):
        """
        Remove the `node` and its edges, but create new edges by merging each in-edge with each out-edge
        s.t. node's predecessors are still connected to its successors.
        If `node` is a root-node or sink-node (leaf) then no new edges will get created.
        """

        # Merge edges
        node_name = self.get_node_name(node)

        for pred_node, _, in_reln in self.in_edges(node, keys=True):
            for _, succ_node, out_reln in self.out_edges(node, keys=True):
                new_reln = self.MERGED_EDGE_SEP.join([in_reln, node_name, out_reln])
                self.add_edge(pred_node, succ_node, key=new_reln)

        self.remove_node(node)
        return

    def change_node_entity_type(self, node: str, new_entity_type: str):
        """
        Needed to fix errors in DrugMechDB data.
        """
        old_entity_type = self.get_node_entity_type(node)
        self.entity_type_nodes[old_entity_type].remove(node)
        if not self.entity_type_nodes[old_entity_type]:
            del self.entity_type_nodes[old_entity_type]

        self.entity_type_nodes[new_entity_type].append(node)
        self.nodes[node]["EntityType"] = new_entity_type
        return

    def change_node_name(self, node: str, new_name: str):
        """
        Needed to fix errors in DrugMechDB data.
        """
        self.nodes[node]["name"] = new_name
        return

    # noinspection PyMethodOverriding
    def add_edge(self, head_node: str, tail_node: str, key: str, **attr):
        if self.has_edge(head_node, tail_node, key):
            return
        self.relations.add(key)
        return super().add_edge(head_node, tail_node, key=key, **attr)

    # -----------------------------------------------------------------------------
    #   Various feature getters
    # -----------------------------------------------------------------------------

    def get_graph_id(self) -> str:
        return self.graph.get("_id", self.graph.get("id"))

    def get_graph_name(self) -> str:
        return self.graph.get("graph_name")

    def set_graph_name(self, new_graph_name: str):
        self.graph["graph_name"] = new_graph_name
        return

    def get_node_name(self, node: str) -> str:
        return self.nodes[node]["name"]

    def get_node_entity_type(self, node: str) -> str:
        return self.nodes[node]["EntityType"]

    def get_entity_types(self) -> List[str]:
        # Only return etypes with non-empty nodes, in case `self.change_node_entity_type()` had been called.
        return [k for k, v in self.entity_type_nodes.items() if v]

    def get_nodes_for_type(self, node_type: str) -> List[str]:
        return self.entity_type_nodes.get(node_type, [])

    def get_node_names_for_type(self, node_type: str) -> List[str]:
        return [self.get_node_name(node) for node in self.get_nodes_for_type(node_type)]

    def get_relations(self) -> Set[str]:
        return self.relations

    def get_qualified_node_name(self, node) -> str:
        return f"{self.get_node_entity_type(node)}: {self.get_node_name(node)} ({node})"

    def get_edge_repr(self, head: str, reln: str, tail: str):
        return " | ".join([self.get_qualified_node_name(head), reln, self.get_qualified_node_name(tail)])

    def get_graph_heading(self):
        hdg = ""
        if graph_id := self.get_graph_id():
            hdg = f"{hdg}{graph_id}: "
        if graph_name := self.get_graph_name():
            hdg = f"{hdg}{graph_name}. "
        if node := self.get_root_node():
            hdg += f"Root = {node}: {self.get_node_name(node)}. "
        if node := self.get_sink_node():
            hdg += f"Sink = {node}: {self.get_node_name(node)}."

        return hdg

    def pprint(self, with_summary=False, stream=None):
        hdg = self.get_graph_heading()
        if hdg:
            print("---", hdg, "---", file=stream)
            print()

        if with_summary:
            root = self.get_root_node()
            sink = self.get_sink_node()
            print( "    Root node =", self.get_qualified_node_name(root) if root else None)
            print( "    Sink node =", self.get_qualified_node_name(sink) if sink else None)
            print(f"    nbr Nodes = {self.number_of_nodes()},  nbr Edges = {self.number_of_edges()}")
            print( "    Entity types =", ", ".join(sorted(self.entity_type_nodes.keys())))
            print()

        try:
            for node in nx.lexicographical_topological_sort(self):
                for head, tail, reln in self.out_edges(node, keys=True):
                    print(self.get_qualified_node_name(head), reln, self.get_qualified_node_name(tail),
                          sep=" | ", file=stream)
        except (nx.NetworkXError, nx.NetworkXUnfeasible):
            # IF graph is not acyclic
            print("    (graph may have cycles)")
            for head, tail, reln in self.edges(keys=True):
                print(self.get_edge_repr(head, reln, tail), file=stream)

        print()
        return

# /
