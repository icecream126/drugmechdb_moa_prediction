"""
Prompt templates for Add-Link,
used in retrieving MoA's created by adding a link (see kgproc.addlink)
"""

import dataclasses
from typing import Any, Dict, Tuple

from drugmechcf.data.moagraph import MoaGraph
from drugmechcf.data.drugmechdb import DRUG_ENTITY_TYPE, DrugMechDB
from drugmechcf.llm.prompt_types import PromptSource, PromptStyle, QueryType, DrugDiseasePromptInfo, EditLinkInfo
from drugmechcf.llmx.prompts_common import (DRUGMECHDB_ENTITY_TYPE_TO_PROMPT, capitalize_node_name,
                                            get_drugmechdb_moa_for_prompt, get_translated_entity_type_for_prompt)
from drugmechcf.utils.misc import capitalize_words


# -----------------------------------------------------------------------------
#   Globals: Prompt ... General
# -----------------------------------------------------------------------------


# Vars are: prompt_persona, prompt_goal, prompt_instructions, examples_subprompt, drug_disease_prompt
PROMPT_TEMPLATE = """
{prompt_persona}
-Goal-
{prompt_goal}

-Steps-
{prompt_instructions}

{examples_subprompt}
######################
-Real Query-

{drug_disease_prompt}
######################
Output:
"""


# Vars are: prompt_examples
EXAMPLES_SUBPROMPT_TEMPLATE = """
######################
-Examples-
######################
{prompt_examples}
"""


# This is the template for each Example in the full prompt.
# Vars are: example_nbr, example_drug_disease_prompt, example_llm_response
EXAMPLE_TEMPLATE = """
Example {example_nbr}:

{example_drug_disease_prompt}
################
Output:
{example_llm_response}

#############################
"""


# -----------------------------------------------------------------------------
#   Globals: Prompt ... Add-Link
# -----------------------------------------------------------------------------


# Vars are:
#   target_disease_name, source_drug_name,
#   source_node_type, source_node_name,
#   new_relation,
#   target_node_type, target_node_name
ADDLINK_DRUG_DISEASE_PROMPT = \
    """We have some patients with an unusual combination of symptoms that are being treated with
an experimental drug. The drug is helping with some of the symptoms.
However some patients are showing signs of {target_disease_name}, but the common drugs for
{target_disease_name} are contraindicated for these patients.
We have also observed a novel interaction in these patients: in the presence of the experimental drug,
the {source_node_type} {source_node_name} {new_relation} the {target_node_type} {target_node_name}.
Given this observation, for these patients, could the drug {source_drug_name} be useful
in treating {target_disease_name}?
If so, please provide a potential Mechanism of Action for the drug.
"""

# Vars are: drug_name, disease_name, moa_repr
INSERT_KNOWN_MOA_PROMPT = \
    """
We know the Mechanism of Action of the drug {drug_name} on the disease {disease_name} is:
{moa_repr}
"""


ADDLINK_PROMPT_VERSIONS = [

    # --- Version 0 ---
    # 8 examples: 5 +ive, 3 -ive
    {
        "version": 0,
        "prompt_persona": ("You are a biomedical research assistant at a research hospital, assisting "
                           "in the evaluation of drugs for treating diseases."),
        "prompt_goal": ("You will be presented with some observations about a group of patients undergoing\n"
                        "treatment with an experimental drug. These patients have some additional untreated\n"
                        "symptoms related to a known disease. You will be asked to determine, given some\n"
                        "observations, whether a proposed known drug could be useful in treating that disease, and\n"
                        "if it can be useful, to provide a potential Mechanism of Action for that drug on that disease."
                        ),
        "negative_response_text": "NO supporting beneficial Mechanism of Action found.",
        "prompt_instructions": """
1. Say "YES." if the drug could be useful in treating the disease in the described situation.
If on the other hand the drug could not be used in treating the disease despite the novel
interactions observed in these patients, then your final output should be
"{negative_response_text}"

2. If the response in step 1 is "YES", provide a Mechanism of Action describing how the drug will affect the disease.
Present the Mechanism of Action as a series of interaction steps, one step per line.
Each interaction step should mention a direct interaction relationship between two entities, containing 
the following information:
- source_entity_type: The type of the source entity. One of the following types: [{all_entity_types}]
- source_entity_name: The name of the source entity
- target_entity_type: The type of the target entity. One of the following types: [{all_entity_types}]
- target_entity_name: The name of the target entity
- interaction_relationship: How the source entity interacts with the target entity

Please use as many interaction steps as are needed for a complete detailed response, 
using specific entities instead of abstract entities wherever possible.
Please use the same formal entity name each time the entity is mentioned in your response.

Format each interaction step as a triple, using the format:
<source_entity_type>: <source_entity_name> | <interaction_relationship> | <target_entity_type>: <target_entity_name>
""",
        "prompt_examples": [
            # Example 1: +ive, PPI
            # Drug: prednisolone (MESH:D011239)
            #   --> Protein: Prostaglandin G/H synthase 2 (UniProt:P35354)
            #   ==> Protein: Histamine H2 receptor (UniProt:P25021)
            #   --> Disease: Gastroesophageal Reflux (MESH:D005764)
            {
                "drug_node": "MESH:D011239",
                "drug_name": "Prednisolone",
                "disease_node": "MESH:D005764",
                "disease_name": "Gastroesophageal Reflux",
                "edit_link_info": {
                    "source_moa_id": "DB00860_MESH_D008224_1",
                    "source_node": "UniProt:P35354",
                    "source_node_name": "Prostaglandin G/H synthase 2",
                    "source_node_type": "Protein",
                    "target_moa_id": "DB00863_MESH_D005764_1",
                    "target_node": "UniProt:P25021",
                    "target_node_name": "Histamine H2 receptor",
                    "target_node_type": "Protein",
                    "new_relation": "increases activity of",
                },
                "example_llm_response": """
YES.

Drug: Prednisolone | increases activity of | Protein: Glucocorticoid receptor
Protein: Glucocorticoid receptor | decreases activity of | Protein: Prostaglandin G/H Synthase 2
Protein: Prostaglandin G/H Synthase 2 | increases activity of | Protein: Histamine H2 Receptor
Protein: Histamine H2 receptor | positively regulates | Biological Process: Gastric Acid Secretion
Biological Process: Gastric Acid Secretion | increases abundance of | Chemical: Gastric Acid
Chemical: Gastric Acid | contributes to | Disease: Gastroesophageal Reflux
"""
            },
            # Example 2: +ive, Drug-PI
            # Drug: Cerivastatin (MESH:C086276)
            #   ==> Protein: Histamine H2 receptor (UniProt:P25021)
            #   --> Disease: Gastric ulcer (MESH:D013276)
            {
                "drug_node": "MESH:C086276",
                "drug_name": "Cerivastatin",
                "disease_node": "MESH:D013276",
                "disease_name": "Gastric Ulcer",
                "edit_link_info": {
                    "source_moa_id": "DB00439_MESH_D006937_1",
                    "source_node": "MESH:C086276",
                    "source_node_name": "Cerivastatin",
                    "source_node_type": "Drug",
                    "target_moa_id": "DB00585_MESH_D013276_1",
                    "target_node": "UniProt:P25021",
                    "target_node_name": "Histamine H2 receptor",
                    "target_node_type": "Protein",
                    "new_relation": "decreases activity of",
                },
                "example_llm_response": """
YES.

Drug: Cerivastatin | decreases activity of | Protein: Histamine H2 Receptor
Protein: Histamine H2 Receptor | positively regulates | Biological Process: Gastric Acid Secretion
Biological Process: Gastric Acid Secretion | increases abundance of | Chemical: Gastric Acid
Chemical: Gastric Acid | contributes to | Disease: Gastric Ulcer
"""
            },
            # Example 3: -ive example ... new link to uninvolved protein
            #   DrugMechDB has MoA's for 42 drugs for treating "MESH:D013290" (Streptococcus Pyogenes Infection)
            #   These collectively involve 12 Proteins.
            #   Protein: Muscarinic acetylcholine receptor M1 (UniProt:P11229) is not one of them.
            #   The Drug Pinacidil (MESH:D020110) has one MoA, for the Disease: Hypertension,
            #       which also does not contain the above Protein.
            #   We will introduce a new Drug-Protein link.
            {
                "drug_node": "MESH:D020110",
                "drug_name": "Pinacidil",
                "disease_node": "MESH:D013290",
                "disease_name": "Streptococcus Pyogenes Infection",
                # Special attributes for -ive examples
                "is_negative": True,
                "edit_link_info": {
                    "source_moa_id": "DB01212_MESH_D013290_1",
                    "source_node": "MESH:D020110",
                    "source_node_name": "Pinacidil",
                    "source_node_type": "Drug",
                    "target_moa_id": "DB06762_MESH_D006973_1",
                    "target_node": "UniProt:P11229",
                    "target_node_name": "Muscarinic Acetylcholine Receptor M1",
                    "target_node_type": "Protein",
                    "new_relation": "positively regulates",
                },
                "example_llm_response": """
NO supporting beneficial Mechanism of Action found.
"""
                # Reasons:
                # * The protein Muscarinic Acetylcholine Receptor M1 has no established relationship with
                # the disease Streptococcus Pyogenes Infection.
                # * Pinacidil does not have any other known effects against Streptococcus Pyogenes Infection.
            },

            # Example 4: -ive example (from example 1) ... inverted PPI
            # Drug: prednisolone (MESH:D011239)
            #   --> Protein: Prostaglandin G/H synthase 2 (UniProt:P35354)
            #   ==>- Protein: Histamine H2 receptor (UniProt:P25021)
            #   --> Disease: Gastroesophageal Reflux (MESH:D005764)
            {
                "drug_node": "MESH:D011239",
                "drug_name": "Prednisolone",
                "disease_node": "MESH:D005764",
                "disease_name": "Gastroesophageal Reflux",
                "edit_link_info": {
                    "source_moa_id": "DB00860_MESH_D008224_1",
                    "source_node": "UniProt:P35354",
                    "source_node_name": "Prostaglandin G/H synthase 2",
                    "source_node_type": "Protein",
                    "target_moa_id": "DB00863_MESH_D005764_1",
                    "target_node": "UniProt:P25021",
                    "target_node_name": "Histamine H2 receptor",
                    "target_node_type": "Protein",
                    "new_relation": "decreases activity of",
                },
                "example_llm_response": """
NO supporting beneficial Mechanism of Action found.
"""
                # Reasons (4o):
                # Given that Prostaglandin G/H synthase 2 is already acting beneficially by suppressing
                # Histamine H2 receptor activity, reducing its levels via Prednisolone may remove this
                # beneficial effect, possibly exacerbating reflux symptoms.
            },

            # Example 5: -ive example (from example 2) ... inverted Drug-PI
            # Drug: Cerivastatin (MESH:C086276)
            #   ==>+ Protein: Histamine H2 receptor (UniProt:P25021)
            #   --> Disease: Gastric ulcer (MESH:D013276)
            {
                "drug_node": "MESH:C086276",
                "drug_name": "Cerivastatin",
                "disease_node": "MESH:D013276",
                "disease_name": "Gastric Ulcer",
                "edit_link_info": {
                    "source_moa_id": "DB00439_MESH_D006937_1",
                    "source_node": "MESH:C086276",
                    "source_node_name": "Cerivastatin",
                    "source_node_type": "Drug",
                    "target_moa_id": "DB00585_MESH_D013276_1",
                    "target_node": "UniProt:P25021",
                    "target_node_name": "Histamine H2 receptor",
                    "target_node_type": "Protein",
                    "new_relation": "increases activity of",
                },
                "example_llm_response": """
NO supporting beneficial Mechanism of Action found.
"""
                # Reasons (4o):
                # In the novel observation, Cerivastatin increases activity of the Histamine H2 receptor.
                # This would increase acid secretion, potentially worsening the condition of a Gastric Ulcer,
                # rather than alleviating it.
            },


            # Example 6: +ive, PPI
            # Drug: Scopolamine (MESH:D012601)
            #   --> Protein: Muscarinic acetylcholine receptor M1 (UniProt:P11229)
            #   ==> Protein: D(2) dopamine receptor (UniProt:P14416)
            #   --> Disease: Schizophrenia (MESH:D012559)
            # Nbr extra nodes = 10
            # Entity types in graph = Anatomical Structure, Biological Process, Chemical, Disease, Drug,
            #                         Phenotype
            {
                "drug_node": "MESH:D012601",
                "drug_name": "Scopolamine",
                "disease_node": "MESH:D012559",
                "disease_name": "Schizophrenia",
                "edit_link_info": {
                    "source_moa_id": "DB00747_MESH_D015863_1",
                    "source_node": "UniProt:P11229",
                    "source_node_name": "Muscarinic Acetylcholine Receptor M1",
                    "source_node_type": "Protein",
                    "target_moa_id": "DB04842_MESH_D012559_1",
                    "target_node": "UniProt:P14416",
                    "target_node_name": "D(2) Dopamine Receptor",
                    "target_node_type": "Protein",
                    "new_relation": "increases activity of",
                },
                "example_llm_response": """
YES.

Drug: Scopolamine | decreases activity of | Protein: Muscarinic Acetylcholine Receptor M1
Protein: Muscarinic Acetylcholine Receptor M1 | increases activity of | Protein: D(2) Dopamine Receptor
Protein: D(2) Dopamine Receptor | positively regulates | Biological Process: Dopamine Secretion
Biological Process: Dopamine Secretion | increases abundance of | Chemical: Dopamine
Chemical: Dopamine | located in | Anatomical Structure: Dorsolateral Prefrontal Cortex
Chemical: Dopamine | located in | Anatomical Structure: Medial Forebrain Bundle
Anatomical Structure: Medial Forebrain Bundle | correlated with | Phenotype: Psychosis
Phenotype: Psychosis | positively correlated with | Disease: Schizophrenia
Anatomical Structure: Dorsolateral Prefrontal Cortex | correlated with | Phenotype: Anhedonia
Anatomical Structure: Dorsolateral Prefrontal Cortex | correlated with | Phenotype: Emotional Blunting
Anatomical Structure: Dorsolateral Prefrontal Cortex | correlated with | Phenotype: Cognitive Impairment
Anatomical Structure: Dorsolateral Prefrontal Cortex | correlated with | Phenotype: Apathy
Anatomical Structure: Dorsolateral Prefrontal Cortex | correlated with | Phenotype: Poor Speech
Phenotype: Apathy | positively correlated with | Disease: Schizophrenia
Phenotype: Poor Speech | positively correlated with | Disease: Schizophrenia
Phenotype: Anhedonia | positively correlated with | Disease: Schizophrenia
Phenotype: Emotional Blunting | positively correlated with | Disease: Schizophrenia
Phenotype: Cognitive Impairment | positively correlated with | Disease: Schizophrenia
"""
            },

            # Example 7: +ive, PPI
            # Drug: Drug: cortisone acetate (MESH:D003348)
            #   --> Protein: COX Genes (UniProt:P23219)
            #   ==> Protein: BCR/ABL (UniProt:P00519)
            #   --> Disease: CML (ph+) (MESH:D015464)
            # Nbr extra nodes = 1
            # Entity types in graph = Disease, Drug, Protein
            {
                "drug_node": "MESH:D003348",
                "drug_name": "Cortisone Acetate",
                "disease_node": "MESH:D015464",
                "disease_name": "CML (ph+)",
                "edit_link_info": {
                    "source_moa_id": "DB01380_MESH_D007634_1",
                    "source_node": "UniProt:P23219",
                    "source_node_name": "COX Genes",
                    "source_node_type": "Protein",
                    "target_moa_id": "DB00619_MESH_D015464_1",
                    "target_node": "UniProt:P00519",
                    "target_node_name": "BCR/ABL",
                    "target_node_type": "Protein",
                    "new_relation": "increases activity of",
                },
                "example_llm_response": """
YES.

Drug: Cortisone Acetate | increases activity of | Protein: Glucocorticoid Receptor
Drug: Cortisone Acetate | decreases abundance of | Protein: COX Genes
Protein: Glucocorticoid Receptor | decreases abundance of | Protein: COX Genes
Protein: COX Genes | increases activity of | Protein: BCR/ABL
Protein: BCR/ABL | causes | Disease: CML (ph+)
"""
            },

            # Example 8: +ive, PPI
            # Drug: Drug: imatinib (MESH:D000068877)
            #   --> Protein: BCR/ABL (UniProt:P00519)
            #   ==> Protein: Glucocorticoid receptor (UniProt:P04150)
            #   --> Disease: Keratitis (MESH:D007634)
            # Nbr extra nodes = 6
            # Entity types in graph = Disease, Drug, Protein
            {
                "drug_node": "MESH:D000068877",
                "drug_name": "Imatinib",
                "disease_node": "MESH:D007634",
                "disease_name": "Keratitis",
                "edit_link_info": {
                    "source_moa_id": "DB00619_MESH_D015464_1",
                    "source_node": "UniProt:P00519",
                    "source_node_name": "BCR/ABL",
                    "source_node_type": "Protein",
                    "target_moa_id": "DB01380_MESH_D007634_1",
                    "target_node": "UniProt:P04150",
                    "target_node_name": "Glucocorticoid receptor",
                    "target_node_type": "Protein",
                    "new_relation": "decreases activity of",
                },
                "example_llm_response": """
YES.

Drug: Imatinib | decreases activity of | Protein: BCR/ABL
Protein: BCR/ABL | decreases activity of | Protein: Glucocorticoid Receptor
Protein: Glucocorticoid Receptor | increases abundance of | Protein: lipocortin-1
Protein: Glucocorticoid Receptor | decreases abundance of | Protein: COX Genes
Protein: lipocortin-1 | negatively regulates | Pathway: Prostaglandin Synthesis
Protein: lipocortin-1 | negatively regulates | Biological Process: Immune Cell Funciton
Biological Process: Immune Cell Funciton | positively regulates | Biological Process: Inflammation
Protein: COX Genes | increases abundance of | Chemical: Prostaglandins
Chemical: Prostaglandins | participates in | Biological Process: Inflammation
Pathway: Prostaglandin Synthesis | participates in | Biological Process: Inflammation
Pathway: Prostaglandin Synthesis | positively regulates | Biological Process: Inflammation
Biological Process: Inflammation | causes | Disease: Keratitis
"""
            },
        ]
    },
]


# -----------------------------------------------------------------------------
#   Classes
# -----------------------------------------------------------------------------

class PromptBuilder:

    def __init__(self,
                 drugmechdb: DrugMechDB,
                 prompt_version: int = 0,
                 include_examples: bool = True,
                 insert_known_moas: bool = False,
                 ):
        """

        :param drugmechdb:
        :param prompt_version: This is a 0-based index of prompt versions.
        :param include_examples: Whether to include examples in the prompt.
        :param insert_known_moas: Whether to insert the known MoAs into the prompt.
        """

        self.drugmechdb = drugmechdb
        self.prompt_version = prompt_version
        self.include_examples = include_examples
        self.insert_known_moas = insert_known_moas

        self.entity_type_translations = DRUGMECHDB_ENTITY_TYPE_TO_PROMPT

        # --- derived fields ---

        # Translating KG EntityType into Formal Name listed in prompt
        self.entity_type_to_formal_name: Dict[str, str] = {
            kg_etype: get_translated_entity_type_for_prompt(kg_etype)
            for kg_etype in self.entity_type_translations.keys()
        }

        return

    def build_prompt_info(self,
                          drug_id: str,
                          drug_name: str,
                          disease_id: str,
                          disease_name: str,
                          *,
                          edit_link_info: EditLinkInfo,
                          is_negative_sample: bool,
                          moa: MoaGraph = None,
                          ) -> DrugDiseasePromptInfo:

        if moa is None:
            moa_id = None
        else:
            moa_id = self.drugmechdb.get_moa_id(moa)

        prompt_info = DrugDiseasePromptInfo(drug_id=drug_id,
                                            disease_id=disease_id,
                                            drug_name=drug_name,
                                            disease_name=disease_name,
                                            prompt_style=PromptStyle.NAMED_DISEASE,
                                            prompt_version=self.prompt_version,
                                            source_kg=PromptSource.DRUG_MECH_DB,
                                            is_negative_sample=is_negative_sample,
                                            query_type=QueryType.ADD_LINK,
                                            moa_id=moa_id,
                                            edit_link_info=edit_link_info,
                                            )
        return prompt_info

    def get_full_prompt(self,
                        drug_id: str,
                        drug_name: str,
                        disease_id: str,
                        disease_name: str,
                        *,
                        edit_link_info: EditLinkInfo,
                        is_negative_sample: bool = False,
                        moa: MoaGraph = None,
                        ) -> DrugDiseasePromptInfo:

        if not is_negative_sample:
            assert moa is not None, "`moa` cannot be `None` for positive samples."

        # [1] prompt_info

        prompt_info = self.build_prompt_info(drug_id, drug_name,
                                             disease_id, disease_name,
                                             edit_link_info=edit_link_info,
                                             is_negative_sample=is_negative_sample, moa=moa)

        # Fill in the disease_prompt_nodes: Nodes other than source.Drug mentioned in the new link

        prompt_nodes = []

        if edit_link_info.source_node_type != DRUG_ENTITY_TYPE:
            prompt_nodes.append(edit_link_info.source_node)

        prompt_nodes.append(edit_link_info.target_node)

        prompt_info.disease_prompt_nodes = prompt_nodes

        # Get the prompt

        full_prompt, drug_disease_subprompt = self.build_full_prompt(prompt_info)

        prompt_info.full_prompt = full_prompt
        prompt_info.drug_disease_subprompt = drug_disease_subprompt

        return prompt_info

    def build_full_prompt(self,
                          prompt_info: DrugDiseasePromptInfo
                          ) -> Tuple[str, str]:

        # [1] Build the drug_disease_prompt

        drug_disease_prompt = self.get_drug_disease_prompt(prompt_info)

        prompt_data = ADDLINK_PROMPT_VERSIONS[prompt_info.prompt_version]

        # [2] Build the examples

        examples_subprompt = ""

        if self.include_examples:

            prompt_examples = ""

            for i, example_data in enumerate(prompt_data["prompt_examples"], start=1):

                # -- (i) Build the example's {disease_subprompt}

                edit_link_info = self.get_edit_link_info(example_data, "edit_link_info")

                ex_prompt_info = dataclasses.replace(prompt_info,
                                                     drug_id=example_data["drug_node"],
                                                     drug_name=example_data["drug_name"],
                                                     disease_id=example_data["disease_node"],
                                                     disease_name=example_data["disease_name"],
                                                     edit_link_info=edit_link_info
                                                     )

                # -- (ii) Build the example's {drug_disease_prompt}

                ex_drug_disease_prompt = self.get_drug_disease_prompt(ex_prompt_info)

                # -- (iii) Get the example's expected LLM response

                example_llm_response = example_data["example_llm_response"]

                # -- (iv) Build the complete example

                ex_prompt = EXAMPLE_TEMPLATE.format(example_nbr=i,
                                                    example_drug_disease_prompt=ex_drug_disease_prompt,
                                                    example_llm_response=example_llm_response)

                # ... and add it to the prompt_examples
                prompt_examples += ex_prompt

            examples_subprompt = EXAMPLES_SUBPROMPT_TEMPLATE.format(prompt_examples=prompt_examples)

        # [4] Build the full prompt

        all_entity_types = self.get_all_entity_types_for_llm()

        prompt_instructions = prompt_data["prompt_instructions"].format(
                                                            negative_response_text=self.get_negative_response_text(),
                                                            all_entity_types=all_entity_types)

        persona = prompt_data["prompt_persona"] + "\n" if prompt_data["prompt_persona"] else ""

        full_prompt = \
            PROMPT_TEMPLATE.format(prompt_persona=persona,
                                   prompt_goal=prompt_data["prompt_goal"],
                                   prompt_instructions=prompt_instructions,
                                   examples_subprompt=examples_subprompt,
                                   drug_disease_prompt=drug_disease_prompt
                                   )

        return full_prompt, drug_disease_prompt

    def get_negative_response_text(self) -> str:
        """
        Return what LLM is instructed to say on a -ive sample, when there is no valid MoA
        """
        prompt_data = ADDLINK_PROMPT_VERSIONS[self.prompt_version]
        return prompt_data["negative_response_text"]

    def get_drug_disease_prompt(self, prompt_info: DrugDiseasePromptInfo) -> str:

        drug_name = capitalize_words(prompt_info.drug_name)
        disease_name = capitalize_words(prompt_info.disease_name)

        dd_template = ADDLINK_DRUG_DISEASE_PROMPT
        edit_link_info = prompt_info.edit_link_info

        drug_disease_prompt = dd_template.format(
            target_disease_name=disease_name,
            source_drug_name=drug_name,

            source_node_type=edit_link_info.source_node_type,
            source_node_name=capitalize_node_name(edit_link_info.source_node_name,
                                                  edit_link_info.source_node_type),

            new_relation=edit_link_info.new_relation,

            target_node_type=edit_link_info.target_node_type,
            target_node_name=capitalize_node_name(edit_link_info.target_node_name,
                                                  edit_link_info.target_node_type),
        )

        if self.insert_known_moas:
            known_moas_prompt = self.build_known_moas_subprompt(prompt_info)
            drug_disease_prompt += "\n" + known_moas_prompt

        return drug_disease_prompt

    def build_known_moas_subprompt(self, prompt_info: DrugDiseasePromptInfo) -> str:

        edit_link_info = prompt_info.edit_link_info

        known_moas_prompt = ""

        if edit_link_info.source_moa_id:
            moa = self.drugmechdb.get_indication_graph_with_id(edit_link_info.source_moa_id)
            drug_node = moa.get_root_node()
            drug_name = capitalize_node_name(moa.get_node_name(drug_node), moa.get_node_entity_type(drug_node))
            disease_node = moa.get_sink_node()
            disease_name = capitalize_node_name(moa.get_node_name(disease_node), moa.get_node_entity_type(disease_node))

            known_moas_prompt += INSERT_KNOWN_MOA_PROMPT.format(drug_name=drug_name, disease_name=disease_name,
                                                                moa_repr=get_drugmechdb_moa_for_prompt(moa))

        if edit_link_info.target_moa_id:
            moa = self.drugmechdb.get_indication_graph_with_id(edit_link_info.target_moa_id)
            drug_node = moa.get_root_node()
            drug_name = capitalize_node_name(moa.get_node_name(drug_node), moa.get_node_entity_type(drug_node))
            disease_node = moa.get_sink_node()
            disease_name = capitalize_node_name(moa.get_node_name(disease_node), moa.get_node_entity_type(disease_node))

            if known_moas_prompt:
                known_moas_prompt += "\n"

            known_moas_prompt += INSERT_KNOWN_MOA_PROMPT.format(drug_name=drug_name, disease_name=disease_name,
                                                                moa_repr=get_drugmechdb_moa_for_prompt(moa))

        return known_moas_prompt

    @staticmethod
    def get_edit_link_info(ddict: Dict[str, Any], field_name: str = "edit_link_info") \
            -> EditLinkInfo | None:

        if (eli_dict := ddict.get(field_name)) is None:
            return None

        return EditLinkInfo(**eli_dict)

    def get_all_entity_types_for_llm(self) -> str:
        all_entity_types = ", ".join(sorted(self.entity_type_to_formal_name.values()))
        return all_entity_types

    def formal_to_kg_entity_type_names(self) -> Dict[str, str]:
        return dict((v, k) for k, v in self.entity_type_to_formal_name.items())

# /
