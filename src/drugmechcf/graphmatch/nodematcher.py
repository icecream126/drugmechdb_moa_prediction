"""

"""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from drugmechcf.text.tokenizer import BasicTokenizer
from drugmechcf.text.tfidfmatcher import TfdfParams, TfIdfMatchHelper
from drugmechcf.data.moagraph import MoaGraph

from drugmechcf.utils.misc import pp_underlined_hdg


# -----------------------------------------------------------------------------
#   Classes
# -----------------------------------------------------------------------------


class CharNodeMatcher:
    """
    Matches node names using TfIdfMatchHelper.
    """

    DEFAULT_TVZR_PARAMS = {
        "char_params": TfdfParams(ngram_widths_minmax=(2, 4)),
        "word_params": None,
    }

    FORCE_MATCH_SCORE = 999.0

    def __init__(self, tvzr_params: Optional[Union[str, Dict[str, TfdfParams]]] = None):
        if tvzr_params is None:
            tvzr_params = self.DEFAULT_TVZR_PARAMS

        self.tfdfhelper = TfIdfMatchHelper(self.__class__.__name__ + "_helper", tvzr_params)

        self.tokenizer = BasicTokenizer()
        self.ref_moa: Optional[MoaGraph] = None

        self.matches = None
        return

    def set_reference_moa(self, moa: MoaGraph):
        self.ref_moa = moa
        self._build_tfdfhelper()
        return

    def set_reference_nodes(self, etype: str, nodes_with_name: list[tuple[str, str]], do_build: bool = False):
        """
        Alternative method to set reference nodes.
        Call repeatedly, followed by `do_build=True`.

        :param etype: EntityType
        :param nodes_with_name: List[ (node, node_name), ...]
        :param do_build: IF True THEN do a build
        """
        for node, node_name in nodes_with_name:
            self.tfdfhelper.add_name(etype, node, node_name, self.normalize_name(node_name))

        if do_build:
            self.tfdfhelper.build()

        return

    def _build_tfdfhelper(self):
        for etype in self.ref_moa.get_entity_types():
            for node in self.ref_moa.get_nodes_for_type(etype):
                node_name = self.ref_moa.get_node_name(node)
                self.tfdfhelper.add_name(etype, node, node_name, self.normalize_name(node_name))

        self.tfdfhelper.build()
        return

    def get_reference_node_types_ids(self) -> List[Tuple[str, str]]:
        """
        Get list of nodes in the reference MoA.

        :return: List of Tuple: (entity_type: str, node: str)
        """
        return self.tfdfhelper.concept_type_and_id.copy()

    def normalize_name(self, name: str) -> str:
        return " ".join(self.tokenizer.tokenize(name))

    def get_common_entity_types(self, target_moa: MoaGraph) -> List[str]:
        """
        Return list of entity-types common to ref_moa and target_moa.
        """
        common_types = [e for e in target_moa.get_entity_types() if e in self.ref_moa.get_entity_types()]
        return common_types

    def all_entity_types_valid(self, target_moa: MoaGraph) -> bool:
        """
        Whether all the entity-types in `target_moa` are also in `ref_moa`
        """
        return len(self.get_common_entity_types(target_moa)) == len(target_moa.get_entity_types())

    def get_node_matches(self,
                         target_moa: MoaGraph,
                         force_match_end_points: bool = True,
                         match_entity_types: bool = True,
                         ref2tgt_etypes: Dict[str, List[str]] = None,
                         min_score: float = 0.5)\
            -> Tuple[Dict[str, str], Dict[str, str], Dict[Tuple[str, str], float]]:
        """
        Get the best matches between target_moa and reference_moa.
        Matches are found in a greedy manner by always picking the next highest scoring match
        between a new target-node and new ref-node.
        The resulting matches form an injective (one-to-one) partial mapping.

        :param target_moa:

        :param force_match_end_points:
            IF True THEN the following end-points are automatically paired without name checks:
                {ref_moa, target_moa}.get_root_node(),
                {ref_moa, target_moa}.get_sink_node().
                IF end-points are not defined, then this has no effect.

        :param match_entity_types: IF True THEN Entity-types should also match
            ELSE Ignore Entity-types and only match names

        :param ref2tgt_etypes: ref EntityType [str] => List[ target EntityType [str], ... ]
            IF match_entity_types is True THEN
                this defines alternative target EntityTypes that a ref EntityType can map to,
                in addition to the equality check.

            e.g. "GeneFamily" => ["Protein"], "Protein" => ["GeneFamily"]

        :param min_score: Match scores below this threshold are discarded

        :return:
            - ref_to_target_matches: Dict[ref_node => target_node]
            - tgt_to_ref_matches: Dict[target_node => ref_node]
            - ref_to_target_match_scores: Dict[(ref_nd, tgt_nd) => match-score]
        """

        if ref2tgt_etypes is None:
            ref2tgt_etypes = dict()

        # assemble nodes and names
        tgt_nodes = list(target_moa)
        tgt_node_names = [self.normalize_name(target_moa.get_node_name(node_)) for node_ in tgt_nodes]

        self.matches = self.tfdfhelper.get_matching_concepts_batched(tgt_node_names, min_score=min_score)

        # Flatten the matches
        matched_tgt_nodes = [tgt_nd for tgt_nd, node_matches in zip(tgt_nodes, self.matches)
                                    for _ in node_matches[0]]
        matched_ref_nodes = [etype_node_id[1] for node_matches in self.matches for etype_node_id in node_matches[0]]
        matched_scores = np.asarray([score_ for node_matches in self.matches for score_ in node_matches[1]])

        # Entity types
        matched_tgt_etypes = None
        matched_ref_etypes = None
        if match_entity_types:
            tgt_etypes = [target_moa.get_node_entity_type(nd) for nd in tgt_nodes]
            matched_tgt_etypes = [tgt_etype for tgt_etype, node_matches in zip(tgt_etypes, self.matches)
                                  for _ in node_matches[0]]
            matched_ref_etypes = [etype_node_id[0] for node_matches in self.matches
                                                   for etype_node_id in node_matches[0]]

        # Idxs that sort on score, descending
        sort_idxs = np.argsort(- matched_scores)

        # Select matches with highest score to a new ref_node
        ref_to_target_matches = dict()
        tgt_to_ref_matches = dict()
        ref_tgt_match_scores = dict()

        # ---
        def map_nodes(ref_nd_, tgt_nd_, score):
            if ref_nd_ is not None and tgt_nd_ is not None:
                ref_to_target_matches[ref_nd_] = tgt_nd_
                tgt_to_ref_matches[tgt_nd_] = ref_nd_
                ref_tgt_match_scores[(ref_nd_, tgt_nd_)] = score
            return
        # ---

        # Match end-point nodes
        if force_match_end_points:
            map_nodes(self.ref_moa.get_root_node(), target_moa.get_root_node(), self.FORCE_MATCH_SCORE)
            map_nodes(self.ref_moa.get_sink_node(), target_moa.get_sink_node(), self.FORCE_MATCH_SCORE)

        # Greedy match
        for i in sort_idxs:
            tgt_nd = matched_tgt_nodes[i]
            if tgt_nd in tgt_to_ref_matches:
                continue

            ref_nd = matched_ref_nodes[i]
            if ref_nd in ref_to_target_matches:
                continue

            # Match entity-types
            if match_entity_types:
                if (matched_tgt_etypes[i] != matched_ref_etypes[i] and
                        matched_tgt_etypes[i] not in ref2tgt_etypes.get(matched_ref_etypes[i], [])):
                    continue

            # New match
            map_nodes(ref_nd, tgt_nd, matched_scores[i])
            if len(tgt_to_ref_matches) == target_moa.number_of_nodes():
                break

        return ref_to_target_matches, tgt_to_ref_matches, ref_tgt_match_scores

    def get_matches_to_target_nodes(self,
                                    tgt_etype: str,
                                    target_nodes_with_name: list[tuple[str, str]],
                                    tgt2ref_etypes: Dict[str, List[str]] = None,
                                    min_score: float = 0.5):
        """
        Get the best matches of target_nodes to reference.
        Matches are found in a greedy manner by always picking the next highest scoring match
        between a new target-node and new ref-node FOR THE SAME ENTITY-TYPE `tgt_etype`.
        The resulting matches form an injective (one-to-one) partial mapping.

        :param tgt_etype: Entity type of all the target nodes

        :param target_nodes_with_name: List[ (nd, nd-name), ... ]

        :param tgt2ref_etypes: ref EntityType [str] => List[ reference-EntityType [str], ... ]
            IF match_entity_types is True THEN
                this defines alternative reference EntityTypes that a target EntityType can map to,
                in addition to the equality check.

            e.g. "GeneFamily" => ["Protein"], "Protein" => ["GeneFamily"]

        :param min_score: Match scores below this threshold are discarded

        :return:
            - ref_to_target_matches: Dict[ref_node => target_node]
            - tgt_to_ref_matches: Dict[target_node => ref_node]
            - ref_to_target_match_scores: Dict[(ref_nd, tgt_nd) => match-score]
        """

        if tgt2ref_etypes is None:
            tgt2ref_etypes = dict()

        tgt_nodes = [nd for nd, nm in target_nodes_with_name]
        tgt_node_names = [self.normalize_name(nm) for nd, nm in target_nodes_with_name]
        n_tgt_nodes = len(target_nodes_with_name)

        self.matches = self.tfdfhelper.get_matching_concepts_batched(tgt_node_names, min_score=min_score)

        # Flatten the matches
        matched_tgt_nodes = [tgt_nd for tgt_nd, node_matches in zip(tgt_nodes, self.matches)
                             for _ in node_matches[0]]
        matched_ref_nodes = [etype_node_id[1] for node_matches in self.matches for etype_node_id in node_matches[0]]
        matched_scores = np.asarray([score_ for node_matches in self.matches for score_ in node_matches[1]])

        # Get entity type of ref-nodes
        matched_ref_etypes = [etype_node_id[0] for node_matches in self.matches
                              for etype_node_id in node_matches[0]]

        # Select matches with highest score to a new ref_node
        ref_to_target_matches = dict()
        tgt_to_ref_matches = dict()
        ref_tgt_match_scores = dict()

        # ---
        def map_nodes(ref_nd_, tgt_nd_, score):
            if ref_nd_ is not None and tgt_nd_ is not None:
                ref_to_target_matches[ref_nd_] = tgt_nd_
                tgt_to_ref_matches[tgt_nd_] = ref_nd_
                ref_tgt_match_scores[(ref_nd_, tgt_nd_)] = score
            return
        # ---

        # Idxs that sort on score, descending
        sort_idxs = np.argsort(- matched_scores)

        # Greedy match
        for i in sort_idxs:
            tgt_nd = matched_tgt_nodes[i]
            if tgt_nd in tgt_to_ref_matches:
                continue

            ref_nd = matched_ref_nodes[i]
            if ref_nd in ref_to_target_matches:
                continue

            # Match entity-types
            if (tgt_etype != matched_ref_etypes[i] and
                    matched_ref_etypes[i] not in tgt2ref_etypes.get(tgt_etype, [])):
                continue

            # New match
            map_nodes(ref_nd, tgt_nd, matched_scores[i])
            if len(tgt_to_ref_matches) == n_tgt_nodes:
                break

        return ref_to_target_matches, tgt_to_ref_matches, ref_tgt_match_scores

    def pprint_matches(self, target_moa: MoaGraph, ref_to_target_matches: Dict[str, str]):
        pp_underlined_hdg("Reference MoA to Target MoA node matches")

        print(f"Nbr nodes in Reference MoA = {self.ref_moa.number_of_nodes():2d}")
        print(f"Nbr nodes in Target MoA    = {target_moa.number_of_nodes():2d}")
        print("Nbr nodes matched =", len(ref_to_target_matches))
        print()
        ref_etypes = set(self.ref_moa.get_entity_types())
        tgt_etypes = set(target_moa.get_entity_types())
        common_etypes = ref_etypes & tgt_etypes
        print("Reference MoA Entity-types =", ", ".join(sorted(ref_etypes)))
        print("Target MoA Entity-types    =", ", ".join(sorted(tgt_etypes)))
        print("Common Entity-types        =", ", ".join(sorted(common_etypes)) if common_etypes else "None")
        print()

        if len(ref_to_target_matches) == 0:
            print("No node matches found.")

        # ---
        def pp_node(moa, node) -> str:
            return f"{moa.get_node_entity_type(node)}: {node}"
        # ---

        for ref_nd, tgt_nd in ref_to_target_matches.items():
            print(pp_node(self.ref_moa, ref_nd), "->", pp_node(target_moa, tgt_nd))

        print()
        return

# /
