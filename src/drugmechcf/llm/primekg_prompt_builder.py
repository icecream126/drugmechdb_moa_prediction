"""
Builder for LLM prompts from PrimeKG
"""

from typing import Dict, List, Set, Tuple

import numpy as np

from drugmechcf.data.moagraph import MoaGraph
from drugmechcf.data.primekg import PHENOTYPE_ENTITY_TYPE, load_primekg
from drugmechcf.llm.prompt_templates import *
from drugmechcf.llm.prompt_builder import (PromptBuilder,
                                compute_string_similarities)
from drugmechcf.llm.prompt_types import DrugDiseasePromptInfo, EditLinkInfo, PromptSource, QueryType, PromptStyle
from drugmechcf.utils.misc import capitalize_words


# -----------------------------------------------------------------------------
#   Globals
# -----------------------------------------------------------------------------

PRIMEKG_DISEASE_NEIGHBORS_IN_PROMPT = ["effect/phenotype", "exposure", "gene/protein"]

# EntityType => (Singular, Plural)
PRIMEKG_ENTITY_TYPE_TO_PROMPT = {
    "effect/phenotype": ("the phenotype", "the phenotypes"),
    "exposure": ("exposure to", "exposure to"),
    "gene/protein": ("the protein", "the proteins")
}

# Dont change case of the names of entities from these types
DONT_CAPITALIZE_TYPES = ["gene/protein"]

PRIMEKG_ENTITY_TYPE_TO_FORMAL_NAME = {
    "anatomy": "Anatomy",
    "biological_process": "Biological Process",
    "cellular_component": "Cellular Component",
    "disease": "Disease",
    "drug": "Drug",
    "effect/phenotype": "Phenotype",
    "exposure": "Exposure",
    "gene/protein": "Protein",
    "molecular_function": "Molecular Function",
    "pathway": "Pathway",
}

# -----------------------------------------------------------------------------
#   Classes
# -----------------------------------------------------------------------------


class PrimeKGPromptBuilder(PromptBuilder):

    def __init__(self, opts: str = None):
        super().__init__(source_kg=PromptSource.PRIME_KG,
                         entity_type_translations=PRIMEKG_ENTITY_TYPE_TO_PROMPT,
                         capitalize_type_names=True,
                         )

        self.primekg = load_primekg(opts=opts)

        # Customized behavior

        # EntityType's whose entity-names do not get capitalized
        self.dont_capitalize_types: List[str] = DONT_CAPITALIZE_TYPES

        # Translating KG EntityType into Formal Name listed in prompt
        self.entity_type_to_formal_name: Dict[str, str] = PRIMEKG_ENTITY_TYPE_TO_FORMAL_NAME

        return

    # noinspection PyUnusedLocal
    def build_prompt_info(self,
                          drug_id: str,
                          disease_id: str,
                          prompt_style: PromptStyle,
                          prompt_version: int,
                          *,
                          is_negative_sample: bool = False,
                          moa: MoaGraph = None,
                          query_type: QueryType = QueryType.KNOWN_MOA,
                          edit_link_info: EditLinkInfo = None,
                          ) -> DrugDiseasePromptInfo:

        prompt_info = super().build_prompt_info(drug_id, disease_id, prompt_style, prompt_version,
                                                is_negative_sample=is_negative_sample,
                                                query_type=query_type,
                                                edit_link_info=edit_link_info,
                                                )

        prompt_info.disease_name = self.primekg.get_node_name(disease_id)
        prompt_info.drug_name = self.primekg.get_node_name(drug_id)

        return prompt_info

    def get_full_drug_disease_prompt(self,
                                     drug_id: str,
                                     disease_id: str,
                                     prompt_style: PromptStyle,
                                     *,
                                     is_negative_sample: bool = False,
                                     prompt_version: int = 0,
                                     query_type: QueryType = QueryType.KNOWN_MOA,
                                     moa: MoaGraph = None,
                                     edit_link_info: EditLinkInfo = None,
                                     verbose: bool = False
                                     ) -> DrugDiseasePromptInfo | None:
        """
        Build a prompt from PrimeKG ...
        """

        prompt_info = self.build_prompt_info(drug_id, disease_id, prompt_style, prompt_version,
                                             is_negative_sample=is_negative_sample,
                                             query_type=query_type,
                                             edit_link_info=edit_link_info,
                                             )

        disease_subprompt, prompt_nodes = self.get_disease_prompt_data(prompt_info, verbose=verbose)

        if not disease_subprompt:
            return None

        prompt_info.disease_subprompt = disease_subprompt
        prompt_info.disease_prompt_nodes = prompt_nodes

        llm_full_prompt, drug_disease_prompt = self.build_full_prompt(prompt_info)

        prompt_info.full_prompt = llm_full_prompt
        prompt_info.drug_disease_subprompt = drug_disease_prompt

        return prompt_info

    def get_disease_prompt_data(self,
                                prompt_info: DrugDiseasePromptInfo,
                                phenotype_similarity_max: float = 0.2,
                                verbose=False
                                ) -> Tuple[str, List[str]]:

        subprompt_template = DISEASE_SUBPROMPTS[prompt_info.prompt_style]

        if prompt_info.prompt_style is PromptStyle.NAMED_DISEASE:
            disease_subprompt = subprompt_template.format(disease_name=prompt_info.disease_name)
            return disease_subprompt, []

        dis_neighbors, drug_neighbors = self.primekg.get_disease_neighbors(prompt_info.disease_id,
                                                                           PRIMEKG_DISEASE_NEIGHBORS_IN_PROMPT)

        if prompt_info.prompt_style is PromptStyle.NAMED_DISEASE_WITH_ALL_ASSOCIATIONS:
            # Ensure ALL associations are mentioned -- no filtering by name
            phenotype_similarity_max = 100

        associations, nodes_used_in_assocs = self.describe_disease_associations(prompt_info.disease_name,
                                                                dis_neighbors,
                                                                phenotype_similarity_max=phenotype_similarity_max,
                                                                verbose=verbose)

        if not associations.strip():
            prompt = None
        else:
            prompt = subprompt_template.format(disease_name=capitalize_words(prompt_info.disease_name),
                                               associations=associations)

        return prompt, nodes_used_in_assocs

    def describe_disease_associations(self,
                                      dis_name: str,
                                      dis_neighbors: Dict[str, Set[str]],
                                      phenotype_similarity_max: float = 0.2,
                                      capitalize: bool = True,
                                      verbose=False
                                      ) -> Tuple[str, List[str]]:
        """
        Make disease sub-prompt with a disease described through its associations, and optionally its name.

        :param dis_name: English name for disease
        :param dis_neighbors: Dict{ EntityType => List[Node, ...] }
            Dict of direct neighbors of the Disease, of types allowed in prompts.
        :param phenotype_similarity_max: Phenotypes with similarity higher than this value are ignored
        :param capitalize: Whether names of EntityType and Node are capitalized
        :param verbose:

        :return:
            - all associations, as a str | None if no valid disease-associations
            - MoA nodes used in the associations to describe the disease (not incl. the disease node)
        """

        associations = ""
        nodes_used_in_assocs = []

        # Entity types have already been restricted to those allowed
        for ntype, nodes in dis_neighbors.items():
            nodes = dis_neighbors.get(ntype)
            if not nodes:
                continue

            node_names = [self.primekg.get_node_name(nd) for nd in nodes]
            # Should some nodes be skipped (nodes_mask[i] = False) ?
            if ntype == PHENOTYPE_ENTITY_TYPE:
                sims = compute_string_similarities(dis_name, node_names)
                nodes_mask = sims <= phenotype_similarity_max
            else:
                nodes_mask = np.ones(len(nodes), dtype=bool)

            if verbose and np.any(~nodes_mask):
                print(f"      Rejecting {ntype}'s as too similar to disease name:")
                print("         ", ", ".join(name for name, mask in zip(node_names, nodes_mask) if not mask))
                print()

            if not np.any(nodes_mask):
                continue

            if capitalize:
                node_names = [self.capitalize_node_name(nm, ntype) for nm in node_names]

            if associations:
                associations += "; and "

            typename = self.get_translated_entity_type(ntype, plural=len(nodes) > 1)

            nodes_used_in_assocs.extend([nd for nd, mask in zip(nodes, nodes_mask) if mask])

            associations += f"{typename}: " + \
                            ", ".join(name for name, mask in zip(node_names, nodes_mask) if mask)

        return associations, nodes_used_in_assocs

    def build_full_prompt(self,
                          prompt_info: DrugDiseasePromptInfo,
                          ) -> Tuple[str, str]:
        raise NotImplementedError

    def capitalize_type_name(self, name: str):
        # Don't capitalize the words: the, to
        dont_capitalize = ["the", "to"]

        return " ".join([w if w in dont_capitalize else w.capitalize() for w in name.split()])

# /
