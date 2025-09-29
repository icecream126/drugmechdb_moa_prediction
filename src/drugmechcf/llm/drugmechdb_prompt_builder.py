"""
Prompt-builder for DrugMechDB
"""

import dataclasses
import re
import warnings
from typing import Dict, List, Set, Tuple

import numpy as np

from drugmechcf.data.moagraph import MoaGraph
from drugmechcf.data.drugmechdb import PHENOTYPE_ENTITY_TYPE, DRUG_ENTITY_TYPE, DrugMechDB, load_drugmechdb
from drugmechcf.llm.prompt_templates import *
from drugmechcf.llm.prmpt_instructions_basic import DRUGMECHDB_PROMPT_VERSIONS
from drugmechcf.llm.prompt_builder import PromptBuilder, compute_string_similarities
from drugmechcf.llm.prompt_types import DrugDiseasePromptInfo, PromptSource, QueryType, PromptStyle, EditLinkInfo
from drugmechcf.utils.misc import capitalize_words, english_join


# -----------------------------------------------------------------------------
#   Globals
# -----------------------------------------------------------------------------

# DrugMechDB:EntityType => (singular, plural) for use in disease prompt.
DRUGMECHDB_ENTITY_TYPE_TO_PROMPT = {
    'Protein': ('protein', 'proteins'),
    'Drug': ('drug', 'drugs'),
    'Disease': ('disease', 'diseases'),
    'BiologicalProcess': ('biological process', 'biological processes'),
    'ChemicalSubstance': ('chemical', 'chemicals'),
    'GrossAnatomicalStructure': ('anatomical structure', 'anatomical structures'),
    'Pathway': ('pathway', 'pathways'),
    'OrganismTaxon': ('organism', 'organisms'),
    'MolecularActivity': ('molecular activity', 'molecular activities'),
    'GeneFamily': ('gene', 'genes'),
    'CellularComponent': ('cellular component', 'cellular components'),
    'PhenotypicFeature': ('phenotype', 'phenotypes'),
    'Cell': ('cell', 'cells'),
    'MacromolecularComplex': ('macromolecular complex', 'macromolecular complexes')
}

# Dont change case of the names of entities from these types
DONT_CAPITALIZE_TYPES = ["Protein", "GeneFamily"]

# Include all Entity Types as Neighbors
DRUGMECHDB_DISEASE_NEIGHBORS_IN_PROMPT = list(set(DRUGMECHDB_ENTITY_TYPE_TO_PROMPT.keys()) - {"Drug", "Disease"})


# -----------------------------------------------------------------------------
#   Classes
# -----------------------------------------------------------------------------


class DrugMechPromptBuilder(PromptBuilder):

    def __init__(self,
                 drugmechdb: DrugMechDB = None,
                 src_dirname: str = "DrugMechDB",
                 cache_file_name: str = "DrugMechDB.pkl"):

        super().__init__(source_kg=PromptSource.DRUG_MECH_DB,
                         entity_type_translations=DRUGMECHDB_ENTITY_TYPE_TO_PROMPT,
                         capitalize_type_names=True,
                         )

        if drugmechdb is not None:
            self.drugmechdb = drugmechdb
        else:
            self.drugmechdb: DrugMechDB = load_drugmechdb(src_dirname=src_dirname, cache_file_name=cache_file_name,
                                                          verbose=True)

        # --- Customized behavior

        # EntityType's whose entity-names do not get capitalized
        self.dont_capitalize_types: List[str] = DONT_CAPITALIZE_TYPES

        return

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
        Builds and returns the entire prompt for a Drug-Disease query.

        :param drug_id: Identifies the Drug, typically used as Node id in MoA
        :param disease_id: Identifies the Disease, typically used as Node id in MoA
        :param prompt_style: Prompt style
        :param prompt_version: Prompt version to use, if applicable

        :param query_type:
        :param moa: The MoA graph from which to build a prompt, if available.

        :param edit_link_info: additional info for `query_type` = QueryType.ADD_LINK, .DELETE_LINK, .INVERT_LINK

        :param is_negative_sample: Is this a negative example?
        :param verbose:

        :return: None if no LLM prompt possible, else DrugDiseasePromptInfo
        """
        if is_negative_sample:
            raise NotImplementedError("Negative examples not yet supported.")

        if query_type in [QueryType.ADD_LINK, QueryType.DELETE_LINK, QueryType.CHANGE_LINK]:
            assert moa is not None, f"The arg `moa` must be supplied when {query_type = }"

        if moa is None:
            indication_graphs = self.drugmechdb.get_indication_graphs(drug_id, disease_id)
            if len(indication_graphs) != 1:
                warnings.warn(f"Unexpected nbr indications graphs ({len(indication_graphs)}) found for "
                              f"{drug_id=}, {disease_id=}")

                if len(indication_graphs) == 0:
                    return None

            moa = indication_graphs[0]

        return self.get_full_drug_disease_prompt_from_moa(moa, drug_id, disease_id, prompt_style,
                                                          prompt_version=prompt_version,
                                                          query_type=query_type,
                                                          edit_link_info=edit_link_info,
                                                          verbose=verbose)

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

        assert moa is not None, "`moa` cannot be `None`."

        moa_id = self.drugmechdb.get_moa_id(moa)

        prompt_info = super().build_prompt_info(drug_id, disease_id, prompt_style, prompt_version,
                                                is_negative_sample=is_negative_sample,
                                                moa_id=moa_id,
                                                query_type=query_type,
                                                edit_link_info=edit_link_info,
                                                )

        prompt_info.disease_name = moa.get_node_name(disease_id)
        prompt_info.drug_name = moa.get_node_name(drug_id)

        return prompt_info

    def get_full_drug_disease_prompt_from_moa(self,
                                              moa: MoaGraph,
                                              drug_id: str,
                                              disease_id: str,
                                              prompt_style: PromptStyle,
                                              *,
                                              prompt_version: int = 0,
                                              query_type: QueryType = QueryType.KNOWN_MOA,
                                              edit_link_info: EditLinkInfo = None,
                                              verbose: bool = False
                                              ) -> DrugDiseasePromptInfo | None:

        prompt_info = self.build_prompt_info(drug_id, disease_id, prompt_style, prompt_version,
                                             moa=moa,
                                             query_type=query_type,
                                             edit_link_info=edit_link_info,
                                             )

        disease_subprompt, prompt_nodes = self.get_moa_disease_prompt_data(moa, prompt_info, verbose=verbose)

        if query_type is QueryType.KNOWN_MOA:
            if not disease_subprompt:
                return None

        prompt_info.disease_subprompt = disease_subprompt
        prompt_info.disease_prompt_nodes = prompt_nodes

        llm_full_prompt, drug_disease_prompt = self.build_full_prompt(prompt_info)

        prompt_info.full_prompt = llm_full_prompt
        prompt_info.drug_disease_subprompt = drug_disease_prompt

        return prompt_info

    def get_moa_disease_prompt_data(self,
                                    moa: MoaGraph | None,
                                    prompt_info: DrugDiseasePromptInfo,
                                    is_negative_sample: bool = False,
                                    disease_associations: str = None,
                                    phenotype_similarity_max: float = 0.2,
                                    verbose: bool = False
                                    ) \
            -> Tuple[str | None, List[str]]:
        """
        Builds the {disease_subprompt} used in a {drug_disease_prompt}

        :param moa: A MoA from DrugMechDB.
            For QueryType.KNOWN_MOA: this is the MoA for which the prompt is being built.
                QueryType.ADD_LINK: this is the source-MoA (containing the source-node).
        :param prompt_info:
        :param is_negative_sample:
        :param phenotype_similarity_max:
        :param disease_associations: pre-computed disease associations, if available
        :param verbose:
        :return:
            - (str) Disease sub-prompt
            - disease_prompt_nodes (Nodes used in the Disease sub-prompt, other than Drug, Disease)
        """

        if prompt_info.query_type in [QueryType.ADD_LINK, QueryType.DELETE_LINK, QueryType.CHANGE_LINK]:
            # Prompt nodes are: source-node (if not Drug), target-node.
            prompt_nodes = []

            edit_link_info = prompt_info.edit_link_info

            assert edit_link_info.source_node is not None and edit_link_info.target_node is not None, \
                f"For {prompt_info.query_type=}, `source_node` and `target_node` are both required."

            if edit_link_info.source_node_type != DRUG_ENTITY_TYPE:
                prompt_nodes.append(edit_link_info.source_node)

            prompt_nodes.append(edit_link_info.target_node)

            # No Disease sub-prompt for this query-type
            return None, prompt_nodes

        # ... for QueryType.KNOWN_MOA ...

        subprompt_template = DISEASE_SUBPROMPTS[prompt_info.prompt_style]

        if prompt_info.prompt_style is PromptStyle.NAMED_DISEASE or is_negative_sample:
            disease_subprompt = subprompt_template.format(disease_name=prompt_info.disease_name,
                                                          associations=disease_associations)
            return disease_subprompt, []

        # ... Not a -ive sample

        nodes_used_in_assocs = []

        if not disease_associations:

            nghbrs, drugs = self.drugmechdb.get_disease_neighbors(moa, prompt_info.disease_id,
                                                                  neighbor_types=DRUGMECHDB_DISEASE_NEIGHBORS_IN_PROMPT)

            if prompt_info.prompt_style is PromptStyle.NAMED_DISEASE_WITH_ALL_ASSOCIATIONS:
                # Ensure ALL associations are mentioned -- no filtering by name
                phenotype_similarity_max = 100

            disease_associations, nodes_used_in_assocs = \
                self.describe_disease_associations(moa, prompt_info.disease_name, nghbrs,
                                                   phenotype_similarity_max=phenotype_similarity_max,
                                                   verbose=verbose
                                                   )

        if not disease_associations.strip():
            prompt = None
        else:
            prompt = subprompt_template.format(disease_name=capitalize_words(prompt_info.disease_name),
                                               associations=disease_associations)

        return prompt, nodes_used_in_assocs

    def describe_disease_associations(self,
                                      moa: MoaGraph,
                                      dis_name: str,
                                      dis_neighbors: Dict[str, Set[str]],
                                      phenotype_similarity_max: float = 0.2,
                                      capitalize: bool = True,
                                      verbose=False
                                      ) -> Tuple[str, List[str]]:
        """
        Make disease sub-prompt with a disease described through its associations, and optionally its name.

        :return:
            - all associations, as a str | None if no valid disease-associations
            - MoA nodes used in the associations to describe the disease (not incl. the disease node)
        """

        associations = ""

        # Special handling of GrossAnatomicalStructure

        anatom_type = "GrossAnatomicalStructure"

        nodes_used_in_assocs = []

        if anatoms := dis_neighbors.get(anatom_type):

            for anatom in anatoms:
                nodes_used_in_assocs.append(anatom)

                anatom_typename = self.get_translated_entity_type(anatom_type)
                anatom_name = moa.get_node_name(anatom)
                if capitalize:
                    anatom_name = self.capitalize_node_name(anatom_name, anatom_type)

                anatom_assocs = []
                for h in moa.predecessors(anatom):
                    h_type = moa.get_node_entity_type(h)
                    if h_type in ["BiologicalProcess", "PhenotypicFeature"]:
                        nodes_used_in_assocs.append(h)
                        h_name = self.capitalize_node_name(moa.get_node_name(h), h_type)
                        anatom_assocs.append(h_name)

                if associations:
                    associations += "; and "

                if anatom_assocs:
                    associations += f"{english_join(anatom_assocs)} in the {anatom_typename}: {anatom_name}"
                else:
                    associations += f"the {anatom_typename}: {anatom_name}"

        for ntype, nodes in dis_neighbors.items():
            if not nodes or ntype == anatom_type:
                continue

            node_names = [moa.get_node_name(nd) for nd in nodes]
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

            associations += f"the {typename}: " + \
                            ", ".join(name for name, mask in zip(node_names, nodes_mask) if mask)

        return associations, nodes_used_in_assocs

    def build_full_prompt(self,
                          prompt_info: DrugDiseasePromptInfo
                          ) -> Tuple[str, str]:

        # [1] Build the drug_disease_prompt

        drug_disease_prompt = self.get_drug_disease_prompt(prompt_info)

        # [2] Build the examples

        ex_drugmechdb = self.drugmechdb

        prompt_examples = ""

        disease_association_reln_default = DISEASE_ASSOCIATION_RELATIONS.get(prompt_info.prompt_version)

        prompt_data = self.get_prompt_instructions_data(prompt_info.query_type, prompt_info.prompt_version)

        for i, example_data in enumerate(prompt_data["prompt_examples"], start=1):

            # -- (i) Build the example's {disease_subprompt}

            associations = None
            edit_link_info = None
            ex_moa = None

            if example_data.get("is_negative", False):

                # Since there is no MoA for a -ive example, the disease-associations are in the `example_data`.

                if prompt_info.prompt_style is PromptStyle.ANONYMIZED_DISEASE:
                    associations = example_data["filtered_disease_associations"]
                elif prompt_info.prompt_style is PromptStyle.NAMED_DISEASE_WITH_ALL_ASSOCIATIONS:
                    associations = example_data["all_disease_associations"]

                edit_link_info = self.get_edit_link_info(example_data, "edit_link_info")

            else:

                if prompt_info.query_type is QueryType.KNOWN_MOA:
                    ex_moa = ex_drugmechdb.get_indication_graph_with_id(example_data["moa_id"])
                else:
                    edit_link_info = self.get_edit_link_info(example_data, "edit_link_info")

            ex_prompt_info = dataclasses.replace(prompt_info,
                                                 drug_id=example_data["drug_node"],
                                                 drug_name=example_data["drug_name"],
                                                 disease_id=example_data["disease_node"],
                                                 disease_name=example_data["disease_name"],
                                                 edit_link_info=edit_link_info
                                                 )

            # Capture the disease_subprompt
            ex_prompt_info.disease_subprompt, _ = \
                self.get_moa_disease_prompt_data(ex_moa, ex_prompt_info,
                                                 is_negative_sample=example_data.get("is_negative", False),
                                                 disease_associations=associations
                                                 )

            # -- (ii) Build the example's {drug_disease_prompt}

            ex_drug_disease_prompt = self.get_drug_disease_prompt(ex_prompt_info)

            # -- (iii) Build the example's expected LLM response
            #   ... vars: {disease_association_reln}, {example_disease_name}

            if prompt_info.prompt_style is PromptStyle.ANONYMIZED_DISEASE:
                example_disease_name = ANONYMIZED_DISEASE_NAME
            else:
                example_disease_name = example_data["disease_name"]

            # Whether default 'associated with' or actual relation to Disease.
            if disease_association_reln_default is None:
                disease_association_reln = example_data.get("disease_association_relation", "NONE-999-WARNING")
            else:
                disease_association_reln = disease_association_reln_default

            example_llm_response = (example_data["example_llm_response"]
                                    .format(example_disease_name=example_disease_name,
                                            disease_association_reln=disease_association_reln,
                                            ))

            # -- (iv) Build the complete example

            ex_prompt = EXAMPLE_TEMPLATE.format(example_nbr=i,
                                                example_drug_disease_prompt=ex_drug_disease_prompt,
                                                example_llm_response=example_llm_response)

            # ... and add it to the prompt_examples
            prompt_examples += ex_prompt

        # [3] Build the full prompt

        all_entity_types = self.get_all_entity_types_for_llm()
        prompt_instructions = prompt_data["prompt_instructions"].format(all_entity_types=all_entity_types)

        persona = prompt_data["prompt_persona"] + "\n" if prompt_data["prompt_persona"] else ""

        full_prompt = \
            PROMPT_TEMPLATE.format(prompt_persona=persona,
                                   prompt_goal=prompt_data["prompt_goal"],
                                   prompt_instructions=prompt_instructions,
                                   prompt_examples=prompt_examples,
                                   drug_disease_prompt=drug_disease_prompt
                                   )

        return full_prompt, drug_disease_prompt

    @staticmethod
    def get_prompt_instructions_data(query_type: QueryType, prompt_version: int):
        """
        Get data for prompt instructions associated with `query_type` and `prompt_version`.

        :param query_type:
        :param prompt_version: 1-based version of the prompt-instructions to use.
        :return: requested prompt instructions data.
        """
        if query_type is QueryType.KNOWN_MOA:
            prompt_versions = DRUGMECHDB_PROMPT_VERSIONS
        else:
            raise NotImplementedError(f"{query_type = } is not supported!")

        assert 0 < prompt_version <= len(prompt_versions), NotImplementedError(
            f"Prompt version = {prompt_version} is not implemented.")

        prompt_data = prompt_versions[prompt_version - 1]

        return prompt_data

    @classmethod
    def get_llm_prompt_example_nodes(cls,
                                     query_type: QueryType,
                                     prompt_version: int) -> List[Tuple[str, str]]:
        """
        Returns List[ (drug_node, disease_node), ... ]
        for all the examples used in prompt version = `prompt_version`.
        prompt_version is 1-based.
        """
        prompt_data = cls.get_prompt_instructions_data(query_type, prompt_version)

        nodes_in_examples = [(example_data["drug_node"], example_data["disease_node"])
                             for example_data in prompt_data["prompt_examples"]]

        return nodes_in_examples

    @classmethod
    def get_llm_prompt_example_moa_ids(cls,
                                     query_type: QueryType,
                                     prompt_version: int) -> List[str | None]:
        """
        Returns List[ MoA-ID[str] | None, ... ]
            for all the examples used in prompt version = `prompt_version`.
        `None` is the MoA-ID for a -ive example.
        prompt_version is 1-based.
        """
        prompt_data = cls.get_prompt_instructions_data(query_type, prompt_version)

        moaids_in_examples = [example_data.get("moa_id")
                             for example_data in prompt_data["prompt_examples"]]

        return moaids_in_examples

    @staticmethod
    def moa_pprint_to_prompt_example(pp_txt: str):
        """
        Converts text output by `MoaGraph.pprint()` to format used in Prompt Examples.
        :param pp_txt:
        """

        for line in pp_txt.splitlines():
            # Remove the Node-IDs
            line = re.sub(r' \([^)]+\)', "", line)
            # Replace with friendlier Entity Types
            for k, vv in DRUGMECHDB_ENTITY_TYPE_TO_PROMPT.items():
                line = line.replace(k, capitalize_words(vv[0]))
            flds = line.split(" | ")
            print(capitalize_words(flds[0]), flds[1], capitalize_words(flds[2]), sep=" | ")
# /
