"""
Convert LLM output formatted text to a graph
"""

from collections import defaultdict
import re
from typing import Dict, List, Optional, Tuple
import warnings

from drugmechcf.data.moagraph import MoaGraph


# -----------------------------------------------------------------------------
#   Globals
# -----------------------------------------------------------------------------

EDGE_SPLIT = r"\s+\|\s+"
NODE_PATT = r"([^:]+):\s+(.+)"


# -----------------------------------------------------------------------------
#   Classes
# -----------------------------------------------------------------------------


class MoaFromText(MoaGraph):
    """
    A graph representing a Mechanism-of-Action. More generally, a relatively small Multi-Di-Graph.

    Builds a MoaGraph from graph_text as a series of lines (format below), each line representing a directed edge:
        <Src_NodeType>: <Src_NodeName> | <Relation> | <Target_NodeType>: <Target_NodeName>
    e.g.
        Drug: Aspirin | Decreases Activity Of | Protein: Prostaglandin G/H synthase 1

    Each Node has a unique Node-ID of the format: '{NodeType}.{Nt}',
        where 0 <= `Nt` < nbr Nodes of type NodeType.
    Each Node has attributes:
        EntityType: str. Same as NodeType
        name: str. Name for this node.

    Each edge has a Key = RelationName.

    The MoaGraph gets an optional attribute 'graph_name'.
    """

    UNK_ENTITY_TYPE = "UnknownEntityType"

    def __init__(self, graph_text_lines: List[str],
                 graph_name: str = None,
                 entity_type_map: Dict[str, str] = None,
                 preferred_root: Tuple[str, str] = None,
                 preferred_sink: Tuple[str, str] = ("Disease", "new disease"),
                 ):
        """
        Build the graph from text.

        :param graph_text_lines: Seq of lines, each line representing an edge in the format:
            <Src_NodeType>: <Src_NodeName> | <Relation> | <Target_NodeType>: <Target_NodeName>

        :param graph_name: (opt.) A name for the graph.

        :param entity_type_map: Map the EntityType names found in `graph_text_lines` to new names defined in this param.

        :param preferred_root: IF True THEN
            pick entity matching preferred_root as root

        :param preferred_sink: IF defined as (EntityType, NodeName) THEN
            - when multiple sink node candidates, pick entity type matching Disease
            - when no sink candidates matching Disease, add (EntityType, NodeName) as new sink node
        """
        super().__init__(graph_name=graph_name)

        # renaming of EntityType names
        self.entity_type_map = entity_type_map or dict()

        # Nodes in text are identified only by (NodeType, Name). This structure is used to define unique Node-IDs.
        self.node_type_names = defaultdict(list)

        for edge_txt in graph_text_lines:
            self._add_edge_from_text(edge_txt)

        # Set the root and sink nodes
        root_nodes = []
        sink_nodes = []
        for nd in self.nodes():
            # noinspection PyCallingNonCallable
            if self.in_degree(nd) == 0:
                root_nodes.append(nd)
            elif self.out_degree(nd) == 0:
                sink_nodes.append(nd)

        gname = "" if not graph_name else f" {graph_name}"

        # Hack to select root node
        if len(root_nodes) != 1 and preferred_root:
            preferred_root_type, preferred_root_name = preferred_root
            if len(root_nodes) == 0:
                drug_node = self.find_node_id(preferred_root_type, preferred_root_name)
                if drug_node is not None:
                    root_nodes = [drug_node]
            else:
                candidate_nodes = root_nodes
                if not candidate_nodes:
                    candidate_nodes = self.get_nodes_for_type(preferred_root_type)
                    if preferred_root_type == "Drug":
                        candidate_nodes += self.get_nodes_for_type("Chemical")
                root_nodes = [nd for nd in candidate_nodes
                              if preferred_root_name.casefold() == self.get_node_name(nd).casefold()]

        if len(root_nodes) == 1:
            self.set_root_node(root_nodes[0])
        else:
            warnings.warn(f"{len(root_nodes)} root nodes found in graph{gname}: "
                          + ", ".join([self.get_qualified_node_name(nd) for nd in root_nodes]),
                          category=UserWarning)

        # Hack to select sink node
        if len(sink_nodes) > 1 and preferred_sink:
            preferred_sink_type, preferred_sink_name = preferred_sink
            dis_sink_nodes = [nd for nd in sink_nodes
                              if preferred_sink_type.casefold() in self.get_node_entity_type(nd).casefold()
                              and preferred_sink_name.casefold() == self.get_node_name(nd).casefold()]
            if len(dis_sink_nodes) == 1:
                sink_nodes = dis_sink_nodes
            else:
                # Add new Disease node as Sink, and link all the sink_nodes to new Sink
                new_dis_node = self.get_node_id(preferred_sink_type, preferred_sink_name)
                for nd in sink_nodes:
                    self.add_edge(nd, new_dis_node, key="associated with")

                sink_nodes = [new_dis_node]

        if len(sink_nodes) == 1:
            self.set_sink_node(sink_nodes[0])
        else:
            warnings.warn(f"{len(sink_nodes)} sink nodes found in graph{gname}: "
                          + ", ".join([self.get_qualified_node_name(nd) for nd in sink_nodes]),
                          category=UserWarning)

        return

    @staticmethod
    def _make_node_id(node_type: str, nidx: int) -> str:
        return f"{node_type}.{nidx}"

    def find_node_id(self, node_type: str, node_name: str) -> Optional[str]:
        """
        IF node_name exists as type `node_type`, THEN return its node_id,
        ELSE return None
        """
        if names_for_type := self.node_type_names[node_type]:
            try:
                nidx = names_for_type.index(node_name)
            except ValueError:
                return None

            return self._make_node_id(node_type, nidx)

        return None

    def get_node_id(self, node_type: str, node_name: str) -> str:
        """
        IF node_name exists as type `node_type`, THEN return its node_id,
        ELSE add as new node and return its node_id
        """
        if nid := self.find_node_id(node_type, node_name):
            return nid

        nidx = len(self.node_type_names[node_type])
        self.node_type_names[node_type].append(node_name)

        nid = self._make_node_id(node_type, nidx)
        self.add_node(nid, EntityType=node_type, name=node_name)
        return nid

    def _add_edge_from_text(self, txt_edge: str):
        txt_edge = txt_edge.strip()
        if not txt_edge:
            return

        if (parts := self._parse_edge_text(txt_edge)) is None:
            return
        else:
            head_txt, reln, tail_txt = parts

        head = self._add_node_from_text(head_txt)
        tail = self._add_node_from_text(tail_txt)

        self.add_edge(head, tail, key=reln)
        return

    @staticmethod
    def _parse_edge_text(txt_edge: str) -> Tuple[str, str, str] | None:
        parts = re.split(EDGE_SPLIT, txt_edge)

        if len(parts) < 3:
            warnings.warn(f"Expected 'HEAD | RELN | TAIL' but only got {len(parts)} fields. Skipping: '{txt_edge}'")
            return None
        elif len(parts) > 3:
            warnings.warn(f"Too many fields in edge text: '{txt_edge}'")

        # Make best attempt to find where the Head and Tail nodes are.
        # ChatGPT sometimes puts them in the wrong order.
        # Sometimes non-binary relations are expressed as: "Head | converts | Entity-3 | Tail"
        # Also sometimes the Nodes are not qualified with "EntityType: ..."
        #
        # Extract Head, Tail, and everything else is joined to form the Relation

        node_part_idxs = [i for i in range(len(parts)) if re.match(NODE_PATT, parts[i])]

        if len(node_part_idxs) >= 2:
            head_idx = node_part_idxs[0]
            head_txt = parts[head_idx]
            tail_idx = node_part_idxs[-1]
            tail_txt = parts[tail_idx]

            reln = " + ".join([parts[i] for i in range(len(parts)) if i not in [head_idx, tail_idx]])

        elif len(node_part_idxs) == 1:
            idx = node_part_idxs[0]
            if idx in [0, 1]:
                head_idx = node_part_idxs[0]
                tail_idx = node_part_idxs[-1]
                head_txt, tail_txt = parts[head_idx], parts[tail_idx]
                # Handle > 3 fields
                reln = " + ".join([parts[i] for i in range(len(parts)) if i not in [head_idx, tail_idx]])
            else:
                head_txt, tail_txt = parts[0], parts[idx]
                reln = parts[1]

        else:
            # None of the parts match our Entity format, so assume "Head | reln | Tail"
            head_txt, tail_txt = parts[0], parts[-1]
            # Handle > 3 fields
            reln = " + ".join(parts[1 : -1])

        # Hack to fix LLM variability: "Disease: new disease" should only be as tail, so swap head, tail
        if head_txt == "Disease: new disease":
            tail_txt, head_txt = head_txt, tail_txt

        return head_txt, reln, tail_txt

    def _add_node_from_text(self, txt_node: str) -> str:
        m = re.match(NODE_PATT, txt_node)
        if not m:
            warnings.warn(f"Node does not specify Entity-Type: '{txt_node}'")
            node_type = self.UNK_ENTITY_TYPE
            node_name = txt_node

        else:
            try:
                node_type, node_name = m.groups()
            except ValueError:
                raise ValueError(f"Invalid node pattern in '{txt_node}'. Expected 'NodeType: NodeName'.")

        # Map to new name, if provided
        node_type = self.entity_type_map.get(node_type, node_type)

        return self.get_node_id(node_type, node_name)

    def change_node_entity_type(self, node: str, new_entity_type: str):
        # Changing entity-type would require changing the node-id (which is "{EType}.{Int}")
        raise NotImplementedError("`change_node_entity_type()` is not supported!")

# /
