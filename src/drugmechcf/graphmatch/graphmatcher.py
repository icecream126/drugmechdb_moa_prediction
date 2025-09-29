"""
basic graph matcher
"""

from collections import defaultdict
import dataclasses
from typing import Any, Dict, List, Sequence, Tuple, Union

import networkx as nx
import numpy as np

from drugmechcf.data.moagraph import MoaGraph
from drugmechcf.graphmatch.nodematcher import CharNodeMatcher

from drugmechcf.utils.misc import pp_underlined_hdg
from drugmechcf.utils.prettytable import PrettyTable


# -----------------------------------------------------------------------------
#   Classes
# -----------------------------------------------------------------------------


@dataclasses.dataclass
class GraphMatchScore:

    has_ref_graph: bool
    """
    Whether a reference graph was provided. Equivalent to label = True.
    """

    has_target_graph: bool
    """
    Whether a target graph was provided. Equivalent to Prediction = Yes.
    """

    # --- The following properties are non-zero only when both graphs are provided ---

    n_nodes_ref: int = 0
    """
    Total nbr nodes in the reference MoA graph. Nbr interior nodes is (n_nodes_* - 2).
    """

    n_nodes_target: int = 0
    """
    Total nbr nodes in the reference MoA graph. Nbr interior nodes is (n_nodes_* - 2).
    """

    node_ratio: float = 0
    """
    n_nodes_target / n_nodes_ref
    """

    n_nodes_matched_ref: int = 0
    """
    Total nbr ref nodes that match to target. Nbr matching interior nodes is (n_nodes_* - 2).
    """

    n_nodes_matched_target: int = 0
    """
    Total nbr target nodes that match to ref. Same as n_nodes_matched_ref for injective (1-to-1) mapping.
    """

    n_prompt_ref_nodes: int = 0
    """
    Nbr (internal) ref_moa nodes mentioned in prompt.
    """

    n_prompt_ref_nodes_matched: int = 0
    """
    Nbr of these nodes that matched.
    """

    node_match_score: float = 0
    """
    Proportion of reference nodes that matched to target = n_nodes_matched/n_nodes_ref.
    """

    interior_node_match_score: float = None
    """
    Proportion of interior reference nodes that matched to target.
    This score is defined only if both reference and target graphs have root and sink nodes defined.
    Then nbr ref interior nodes = n_nodes_ref - 2.
    """

    prompt_node_match_score: float = 0
    """
    n_prompt_ref_nodes_matched / n_prompt_ref_nodes
    """

    n_edges_ref: int = 0

    n_edges_ref_reduced: int = 0

    n_edges_target: int = 0

    n_edges_target_reduced: int = 0

    n_edges_consistent: int = 0
    """
    Nbr ref_moa_reduced edges that are consistent with target_moa.
    Consistency means target has a path from matching head to tail nodes.
    Graph ref_moa_reduced is obtained by dropping unmatched nodes but preserving connectivity between dropped node's
    predecessors and successors by adding new edges.
    """

    edge_match_score: float = 0
    """
    n_edges_consistent / n_edges_ref
    """

    edge_match_score_reduced: float = 0
    """
    n_edges_consistent / n_edges_ref_reduced
    """

    def to_serialized(self) -> Dict[str, Any]:
        """
        Make a serializable dict repr
        """
        # noinspection PyTypeChecker
        d = dataclasses.asdict(self)
        return d

    def pprint(self, stream=None, float_precision: int = 3, indent: int = 0):
        pfx = " " * indent
        # noinspection PyTypeChecker
        for fld in dataclasses.fields(self):
            fval = getattr(self, fld.name)
            if fval is not None and fld.type == float:
                fval = format(fval, f'.{float_precision}f')
            print(f"{pfx}{fld.name} = {fval}", file=stream)
        print(file=stream)

    def accumulate_metrics(self, cum_metrics: Dict[str, List[Union[bool, int, float]]] = None)\
            -> Dict[str, List[Union[bool, int, float]]]:

        if cum_metrics is None:
            cum_metrics = defaultdict(list)

        # noinspection PyTypeChecker
        for fld in dataclasses.fields(self):
            fval = getattr(self, fld.name)
            cum_metrics[fld.name].append(0 if fval is None else fval)

        return cum_metrics
# /


class BasicGraphMatcher:

    def __init__(self, verbosity: int = 0):
        self.verbosity = verbosity
        self.node_matcher = CharNodeMatcher()
        self.entity_type_map = None

        # Results of last node-match: (ref_node, target_node) => score
        self.last_node_match_scores: Dict[Tuple[str, str], float] | None = None

        return

    def set_entity_type_equivalences(self, etype_equivs: Sequence[Sequence[str]]):
        self.entity_type_map = dict()
        for etypes in etype_equivs:
            etypes = list(etypes)
            for i, et in enumerate(etypes):
                remaining_etypes = etypes[:i] + etypes[i + 1:]
                self.entity_type_map[et] = remaining_etypes

        return

    @staticmethod
    def pp_node_matches(ref_moa: MoaGraph, target_moa: MoaGraph,
                        ref_to_target_node_matches: Dict[str, str], tgt_to_ref_node_matches: Dict[str, str]):

        pp_underlined_hdg("Node matches")

        print("Nbr ref MoA nodes =", ref_moa.number_of_nodes())
        print("Nbr target nodes  =", target_moa.number_of_nodes())
        print("Nbr nodes matched =", len(ref_to_target_node_matches))
        print()

        ptbl = PrettyTable(for_md=True)
        ptbl.set_colnames(["ref MoA", "target MoA"], ["s", "s"])

        for rn, tn in ref_to_target_node_matches.items():
            ptbl.add_row_(ref_moa.get_qualified_node_name(rn), target_moa.get_qualified_node_name(tn))

        for rn in ref_moa:
            if rn not in ref_to_target_node_matches:
                ptbl.add_row_(ref_moa.get_qualified_node_name(rn), "")

        for tn in target_moa:
            if tn not in tgt_to_ref_node_matches:
                ptbl.add_row_("", target_moa.get_qualified_node_name(tn))

        print(ptbl)
        print("---\n")
        return

    def get_last_node_match_scores(self) -> Dict[Tuple[str, str], float] | None:
        """
        Returns Results of last node-match: (ref_node[str], target_node[str]) => score[float]
        """
        return self.last_node_match_scores

    def get_last_node_match_scores_simpledict(self, node_sep="||") -> Dict[str, float] | None:
        """
        Returns Results of last node-match as a 'simple' dict with str keys:
            f"{ref_node}{node_sep}{target_node}" => score[float]
        """
        if self.last_node_match_scores is None:
            return None

        scoredict = {f"{refnd}{node_sep}{tgtnd}": score
                     for (refnd, tgtnd), score in self.last_node_match_scores.items()}

        return scoredict

    def match_graphs(self, ref_moa: MoaGraph, target_moa: MoaGraph,
                     force_match_end_points = True,
                     match_entity_types = True,
                     ref_nodes_in_prompt: List[str] = None,
                     ) -> GraphMatchScore:
        """
        Match nodes, and then measure how conssitent is the partial order on nodes defined by directed edges.

        :param ref_moa:
        :param target_moa:

        :param force_match_end_points: IF True (default) THEN automatically pair root and sink nodes.

        :param match_entity_types: IF True (default) THEN nodes match only if Entity-Types also match.
            ELSE Entity-types are ignored and only node-names are matched.

        :param ref_nodes_in_prompt: List of nodes in ref_moa that were used in disease prompt.
            Ideally, all of these nodes would match to a target node,
            AND there would be additional internal nodes matched.

        :return: Match scores:
            - node_match_score: float
                Proportion of ref_moa nodes that matched with a tgt_moa node.
            - graph_match_score
        """

        if ref_nodes_in_prompt is None:
            ref_nodes_in_prompt = []

        metrics = GraphMatchScore(has_ref_graph=ref_moa is not None,
                                  has_target_graph=target_moa is not None,
                                  n_nodes_ref=ref_moa.number_of_nodes() if ref_moa else 0,
                                  n_prompt_ref_nodes=len(ref_nodes_in_prompt),
                                  n_nodes_target=target_moa.number_of_nodes() if target_moa else 0,
                                  n_edges_ref=ref_moa.number_of_edges() if ref_moa else 0,
                                  n_edges_target=target_moa.number_of_edges() if target_moa else 0
                                  )

        if ref_moa is None or target_moa is None:
            return metrics

        # --- 1: Match nodes

        metrics.node_ratio = target_moa.number_of_nodes() / ref_moa.number_of_nodes()

        self.node_matcher.set_reference_moa(ref_moa)
        ref_to_target_node_matches, tgt_to_ref_node_matches, self.last_node_match_scores = \
            self.node_matcher.get_node_matches(target_moa,
                                               force_match_end_points=force_match_end_points,
                                               match_entity_types=match_entity_types,
                                               ref2tgt_etypes=self.entity_type_map
                                               )

        if self.verbosity > 0:
            self.pp_node_matches(ref_moa, target_moa, ref_to_target_node_matches, tgt_to_ref_node_matches)

        metrics.n_nodes_matched_ref = len(ref_to_target_node_matches)
        metrics.n_nodes_matched_target = len(tgt_to_ref_node_matches)
        metrics.node_match_score = metrics.n_nodes_matched_ref / metrics.n_nodes_ref

        metrics.n_prompt_ref_nodes_matched = len([nd for nd in ref_nodes_in_prompt
                                                  if nd in ref_to_target_node_matches])
        metrics.prompt_node_match_score = (np.NAN if metrics.n_prompt_ref_nodes == 0
                                           else metrics.n_prompt_ref_nodes_matched / metrics.n_prompt_ref_nodes)

        both_have_end_points = False
        if ref_moa.has_root_and_sink_nodes_defined() and target_moa.has_root_and_sink_nodes_defined():
            both_have_end_points = True

            # Set default value
            metrics.interior_node_match_score = 0

            if metrics.n_nodes_ref > 2:
                ref_end_points = [ref_moa.get_root_node(), ref_moa.get_sink_node()]
                n_int_matches = 0
                for k in ref_to_target_node_matches.keys():
                    if k not in ref_end_points:
                        n_int_matches += 1

                metrics.interior_node_match_score = n_int_matches / (metrics.n_nodes_ref - 2)

        # IF at most 1 node matched THEN no edges to match
        if metrics.n_nodes_matched_ref <= 1:
            return metrics
        # IF ref_moa has interior nodes AND no interior nodes matched THEN no edges to measure
        if both_have_end_points and force_match_end_points and metrics.n_nodes_matched_ref <= 2 < metrics.n_nodes_ref:
            return metrics

        # --- 2: Measure consistency of target_moa with ref_moa edges

        return self.consistency_reduced_edge_to_path(ref_moa, target_moa, metrics,
                                                     ref_to_target_node_matches, tgt_to_ref_node_matches)

    # noinspection PyUnusedLocal
    def consistency_reduced_edge_to_path(self,
                                         ref_moa: MoaGraph,
                                         target_moa: MoaGraph,
                                         metrics: GraphMatchScore,
                                         ref_to_target_node_matches: Dict[str, str],
                                         tgt_to_ref_node_matches: Dict[str, str],
                                         ) -> GraphMatchScore:
        """
        Measures if for every edge (u, v) in reduced ref_moa, there is a path (u' -> v') in target_moa.

        Reduced-graph = Delete unmapped nodes, add edges

        :param ref_moa:
        :param target_moa:
        :param metrics:
        :param ref_to_target_node_matches:
        :param tgt_to_ref_node_matches:
        :return:
        """

        # Reduce ref_moa to only matched nodes
        ref_moa_reduced = ref_moa.reduce_graph(ref_to_target_node_matches)

        if self.verbosity > 1:
            pp_underlined_hdg("Reduced ref MoA")
            ref_moa_reduced.pprint()
            print("---\n")

        # Count only unique (ref_u, ref_v) edges in reduced `ref_moa`.
        ref_reduced_unique_edges = set(ref_moa_reduced.edges())

        metrics.n_edges_ref_reduced = len(ref_reduced_unique_edges)
        metrics.n_edges_target_reduced = target_moa.number_of_edges()

        if metrics.n_edges_ref_reduced == 0:
            return metrics

        # transitive closure of
        try:
            # First try the faster version. Should succeed in most cases.
            target_moa_closure = nx.transitive_closure_dag(target_moa)
        except nx.NetworkXUnfeasible:
            target_moa_closure = nx.transitive_closure(target_moa)

        # target_moa is consistent with an edge (ref_u, ref_v) in ref_moa
        #   IF there is a corresponding edge in target_moa_closure.
        n_edges_consistent = 0
        for ref_u, ref_v in ref_reduced_unique_edges:
            tgt_u = ref_to_target_node_matches.get(ref_u)
            tgt_v = ref_to_target_node_matches.get(ref_v)

            if tgt_u is None:
                if self.verbosity > 3:
                    print("No target node mapped to:", ref_u)
                continue
            if tgt_v is None:
                if self.verbosity > 3:
                    print("No target node mapped to:", ref_v)
                continue

            # Account for surjective (many-to-1) node mappings.
            if tgt_u == tgt_v or target_moa_closure.has_edge(tgt_u, tgt_v):
                n_edges_consistent += 1
            elif self.verbosity > 3:
                print("No target edge for:", (tgt_u, tgt_v))

        if self.verbosity > 3 and n_edges_consistent < metrics.n_edges_ref_reduced:
            print()

        metrics.n_edges_consistent = n_edges_consistent
        metrics.edge_match_score = n_edges_consistent / metrics.n_edges_ref
        metrics.edge_match_score_reduced = n_edges_consistent / metrics.n_edges_ref_reduced

        return metrics

    def reduced_graph_homomorphism(self,
                                   ref_moa: MoaGraph,
                                   target_moa: MoaGraph,
                                   metrics: GraphMatchScore,
                                   ref_to_target_node_matches: Dict[str, str],
                                   tgt_to_ref_node_matches: Dict[str, str],
                                   ) -> GraphMatchScore:
        """
        ref_moa is homomorphic to target_moa modulo the partial node-mapping if
            for every edge (u, v) in reduced-ref_moa,
                u', v' = node-mapping(u, v)
                u' == v' OR there is an edge (u', v') in reduced-target_moa.

        This means that for every edge (u, v) in reduced-ref_moa,
        there is a path (u', v') in target_moa (ignoring the case when u' == v')
            s.t. the path does not pass through any node w' in tgt_to_ref_node_matches.

        The node-mapping is defined in `ref_to_target_node_matches`, and its inverse `tgt_to_ref_node_matches`.
        """

        # Reduce both to only matched nodes
        ref_moa_reduced = ref_moa.reduce_graph(ref_to_target_node_matches)
        reduced_target_moa = target_moa.reduce_graph(tgt_to_ref_node_matches)

        if self.verbosity > 1:
            pp_underlined_hdg("Reduced ref MoA")
            ref_moa_reduced.pprint()
            print("---\n")
            pp_underlined_hdg("Reduced target MoA")
            reduced_target_moa.pprint()
            print("---\n")

        # Count only unique (ref_u, ref_v) edges in reduced `ref_moa`.
        ref_reduced_unique_edges = set(ref_moa_reduced.edges())

        metrics.n_edges_ref_reduced = len(ref_reduced_unique_edges)
        metrics.n_edges_target_reduced = len(set(reduced_target_moa.edges()))

        if metrics.n_edges_ref_reduced == 0 or metrics.n_edges_target_reduced == 0:
            return metrics

        # reduced_target_moa is consistent with an edge (ref_u, ref_v) in ref_moa_reduced
        #   IF there is a corresponding edge in reduced_target_moa.
        n_edges_consistent = 0
        for ref_u, ref_v in ref_reduced_unique_edges:
            tgt_u = ref_to_target_node_matches.get(ref_u)
            tgt_v = ref_to_target_node_matches.get(ref_v)

            if tgt_u is None or tgt_v is None:
                if self.verbosity > 3:
                    if tgt_u is None:
                        print("No target node mapped to:", ref_u)
                    else:
                        print("No target node mapped to:", ref_v)
                continue

            # Account for surjective (many-to-1) node mappings.
            if tgt_u == tgt_v or reduced_target_moa.has_edge(tgt_u, tgt_v):
                n_edges_consistent += 1
            elif self.verbosity > 3:
                print("No target edge for:", (tgt_u, tgt_v))

        if self.verbosity > 3 and n_edges_consistent < metrics.n_edges_ref_reduced:
            print()

        metrics.n_edges_consistent = n_edges_consistent
        metrics.edge_match_score = n_edges_consistent / metrics.n_edges_ref
        metrics.edge_match_score_reduced = n_edges_consistent / metrics.n_edges_ref_reduced

        return metrics

# /
