"""
Prompt templates for Edit-Link: Change, Delete.
(see kgproc.editlink)
"""

import textwrap

from drugmechcf.data.moagraph import MoaGraph
from drugmechcf.data.drugmechdb import DrugMechDB
from drugmechcf.llm.prompt_types import PromptSource, PromptStyle, QueryType, DrugDiseasePromptInfo, EditLinkInfo
from drugmechcf.llmx.prompts_common import (DRUGMECHDB_ENTITY_TYPE_TO_PROMPT, capitalize_node_name,
                                            get_drugmechdb_moa_for_prompt, get_translated_entity_type_for_prompt)
from drugmechcf.utils.misc import capitalize_words


# -----------------------------------------------------------------------------
#   Globals: Prompt versions
# -----------------------------------------------------------------------------

# ** The overall Prompt template, versioned.
# Vars: drug_name, disease_name, observation_dsecription, response_options
#       {known_moa_insert}
PROMPT_TEMPLATE = [
    # --- version 0
    """
You are a biomedical research assistant at a research hospital, assisting
in the evaluation of drugs for treating diseases.

We have some patients with an unusual combination of symptoms that are 
being treated with an experimental drug. The drug is helping with some of
the symptoms. However some patients are showing signs of {disease_name}.
{observation_dsecription}

Given this observation, for these patients, could the drug {drug_name}
be useful in treating {disease_name}?
{known_moa_insert}

Please provide a detailed explanation, and then output the heading "Summary", followed by
one of the following options that best answers the query:

{response_options}
""",

    # --- version 1
    # Added:
    # these symptoms were observed both before and after the experimental drug was administered.
    # so the experimental drug is neither causing nor curing the observed {disease_name}.
    """
You are a biomedical research assistant at a research hospital, assisting
in the evaluation of drugs for treating diseases.

We have some patients with an unusual combination of symptoms that are 
being treated with an experimental drug. The drug is helping with some of
the symptoms. However some patients are showing signs of {disease_name}.
These symptoms were observed both before and after the experimental drug was administered,
so the experimental drug is neither causing nor curing the observed {disease_name}.

{observation_dsecription}

Given this observation, for these patients, could the drug {drug_name}
be useful in treating {disease_name}?
{known_moa_insert}

Please provide a detailed explanation, and then output the heading "Summary", followed by
one of the following options that best answers the query:

{response_options}
""",

    # --- version 2
    # Moved:
    # We know that the pharmacological action of the experimental drug does not rely on this interaction,
    # so the question is only about the effectiveness of {drug_name} on {disease_name} in these patients.
    """
You are a biomedical research assistant at a research hospital, assisting
in the evaluation of drugs for treating diseases.

We have some patients with an unusual combination of symptoms that are 
being treated with an experimental drug. The drug is helping with some of
the symptoms. However some patients are showing signs of {disease_name}.
These symptoms were observed both before and after the experimental drug was administered,
so the experimental drug is neither causing nor curing the observed {disease_name}.

{observation_dsecription}

We know that the pharmacological action of the experimental drug does not rely on this interaction,
so the question is only about the effectiveness of {drug_name} on {disease_name} in these patients.
{known_moa_insert}
Given this observation, for these patients, could the drug {drug_name}
be useful in treating {disease_name}?

Please provide a detailed explanation, and then output the heading "Summary", followed by
one of the following options that best addresses the effectiveness of {drug_name} on {disease_name}:

{response_options}
""",
]


# ** For `{observation_dsecription}, versioned.
# Vars: source_node_type, source_node_name, target_node_type, target_node_name, drug_name,
#       source_node_level_change, inverted_relation
PROMPT_OBSERVATION = [
    # --- version 0
    {
        QueryType.DELETE_LINK: "\n".join(
            ["We have also observed in these patients on the experimental drug, ",
             "the {source_node_type} {source_node_name} has no effect on ",
             "the {target_node_type} {target_node_name}."]),

        QueryType.CHANGE_LINK: "\n".join(
            ["We have also observed in these patients on the experimental drug, ",
             "when also given the the drug {drug_name}, ",
             "{source_node_level_change} the {source_node_type} {source_node_name} ",
             "{inverted_relation} the {target_node_type} {target_node_name}."]),
    },
    # --- version 1
    {
        QueryType.DELETE_LINK: "\n".join(
            ["We have also observed in these patients on the experimental drug, ",
             "the {source_node_type} {source_node_name} has no effect on ",
             "the {target_node_type} {target_node_name}.",
             "We know that the pharmacological action of the experimental drug does not rely on this interaction."
             ]),

        QueryType.CHANGE_LINK: "\n".join(
            ["We have also observed in these patients on the experimental drug, ",
             "when also given the the drug {drug_name}, ",
             "{source_node_level_change} the {source_node_type} {source_node_name} ",
             "{inverted_relation} the {target_node_type} {target_node_name}.",
             "We know that the pharmacological action of the experimental drug does not rely on this interaction."
             ]),
    },
    # --- version 2
    # {source_change_descr} = "the drug {drug_name}" when drug,
    #                    else "{source_node_level_change} the {source_node_type} {source_node_name}"
    {
        QueryType.DELETE_LINK: "\n".join(
            ["We have also observed in these patients,",
             "when the drug {drug_name} is combined with the experimental drug,",
             "the {source_node_type} {source_node_name} has no effect on ",
             "the {target_node_type} {target_node_name}.",
             ]),

        QueryType.CHANGE_LINK: "\n".join(
            ["We have also observed in these patients,",
             "when the drug {drug_name} is combined with the experimental drug,",
             "{source_change_descr} {inverted_relation} the {target_node_type} {target_node_name}.",
             ]),
    },
]


# ** When inserting the known MoA of the drug-disease, into the var `{known_moa_insert}`.
# Vars are: drug_name, disease_name, moa_repr
INSERT_KNOWN_MOA_PROMPT = \
    """
We know the Mechanism of Action of the drug {drug_name} on the disease {disease_name} is:
{moa_repr}
"""


# -----------------------------------------------------------------------------
#   Globals: Response options
# -----------------------------------------------------------------------------


RESPONSE_OPTIONS = {
    "No Effect":
        ("The observed behavior will not affect the drug's mechanism of action on the disease. The drug's intended"
         " effect is still active, so the drug can be used to treat the disease in these patients."),

    "Partially Blocked":
        ("The observed behavior affects only one of the drug's mechanisms of action."
         " However the drug has other mechanisms that are unaffected, so the drug could still provide"
         " therapeutic benefits in treating the disease in these patients."),

    "Fully Blocked":
        ("The observed behavior suggests that the drug's mechanism of action is completely blocked in these patients."
         " So the drug will not be effective in treating the disease in these patients."),

    "Contra-indicated":
        ("Given the observed behavior, the drug could worsen the disease in these patients and potentially exacerbate"
         " its symptoms. So the drug should not be used on these patients."),
}

UNK_OPT = {
    "Unknown": "The effect of the observed behavior on the drug's mechanism of action could not be determined.",
}


# Response Options for each QueryType, for Positive samples.
POSITIVE_RESPONSE_OPTIONS_STRICT = {
    QueryType.DELETE_LINK: ["Fully Blocked"],

    QueryType.CHANGE_LINK: ["Contra-indicated", "Fully Blocked"],
}

# Response Options for each QueryType, for Positive samples.
POSITIVE_RESPONSE_OPTIONS_RELAXED = {
    QueryType.DELETE_LINK: ["Fully Blocked", "Partially Blocked"],

    QueryType.CHANGE_LINK: ["Contra-indicated", "Fully Blocked", "Partially Blocked"],
}

NEGATIVE_RESPONSE_OPTIONS = {
    QueryType.DELETE_LINK: ["No Effect"],

    QueryType.CHANGE_LINK: ["No Effect"],
}

# Ensure correctness of POSITIVE_RESPONSE_OPTIONS
assert all(k in RESPONSE_OPTIONS for pos_opts in POSITIVE_RESPONSE_OPTIONS_STRICT.values() for k in pos_opts)
assert all(k in RESPONSE_OPTIONS for pos_opts in POSITIVE_RESPONSE_OPTIONS_RELAXED.values() for k in pos_opts)


# -----------------------------------------------------------------------------
#   Classes
# -----------------------------------------------------------------------------


class PromptBuilder:

    def __init__(self,
                 drugmechdb: DrugMechDB,
                 query_type: QueryType,
                 prompt_version: int = 2,
                 insert_known_moas: bool = False,
                 add_unknown_response_opt: bool = False,
                 ):
        """

        :param drugmechdb:
        :param prompt_version: This is a 0-based index of prompt versions.
        :param insert_known_moas: Whether to insert the known MoAs into the prompt.
        """

        assert query_type in [QueryType.CHANGE_LINK, QueryType.DELETE_LINK], f"Unsupported {query_type = }."

        self.drugmechdb = drugmechdb
        self.query_type = query_type
        self.prompt_version = prompt_version
        self.insert_known_moas = insert_known_moas
        self.add_unknown_response_opt = add_unknown_response_opt

        self.entity_type_translations = DRUGMECHDB_ENTITY_TYPE_TO_PROMPT

        # --- derived fields ---

        # Translating KG EntityType into Formal Name listed in prompt
        self.entity_type_to_formal_name: dict[str, str] = {
            kg_etype: get_translated_entity_type_for_prompt(kg_etype)
            for kg_etype in self.entity_type_translations.keys()
        }

        self.response_options = RESPONSE_OPTIONS
        if self.add_unknown_response_opt:
            self.response_options = self.response_options | UNK_OPT

        return

    def build_prompt_info(self,
                          drug_id: str,
                          drug_name: str,
                          disease_id: str,
                          disease_name: str,
                          *,
                          edit_link_info: EditLinkInfo,
                          is_negative_sample: bool,
                          moa: MoaGraph,
                          ) -> DrugDiseasePromptInfo:

        assert moa is not None

        moa_id = self.drugmechdb.get_moa_id(moa)

        prompt_info = DrugDiseasePromptInfo(drug_id=drug_id,
                                            disease_id=disease_id,
                                            drug_name=drug_name,
                                            disease_name=disease_name,
                                            prompt_style=PromptStyle.NAMED_DISEASE,
                                            prompt_version=self.prompt_version,
                                            source_kg=PromptSource.DRUG_MECH_DB,
                                            is_negative_sample=is_negative_sample,
                                            query_type=self.query_type,
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
                        is_negative_sample: bool,
                        moa: MoaGraph,
                        ) -> DrugDiseasePromptInfo:

        assert moa is not None, "`moa` cannot be `None` for EditLink samples."

        # [1] prompt_info

        prompt_info = self.build_prompt_info(drug_id, drug_name,
                                             disease_id, disease_name,
                                             edit_link_info=edit_link_info,
                                             is_negative_sample=is_negative_sample,
                                             moa=moa)

        # [2] Get the prompt

        prompt_info.full_prompt = self.build_full_prompt(prompt_info)

        return prompt_info

    def build_full_prompt(self,
                          prompt_info: DrugDiseasePromptInfo
                          ) -> str:

        eli = prompt_info.edit_link_info

        # [1] Build the drug_disease_prompt

        prompt_observation = PROMPT_OBSERVATION[self.prompt_version][self.query_type]

        drug_name = capitalize_words(prompt_info.drug_name)
        disease_name = capitalize_words(prompt_info.disease_name)

        observation_dsecription = prompt_observation.format(
            drug_name=drug_name,
            source_node_type=get_translated_entity_type_for_prompt(eli.source_node_type),
            source_node_name=capitalize_node_name(eli.source_node_name, eli.source_node_type),
            target_node_type=get_translated_entity_type_for_prompt(eli.target_node_type),
            target_node_name=capitalize_node_name(eli.target_node_name, eli.target_node_type),
            source_change_descr=eli.source_change_descr,
            source_node_level_change=eli.source_node_level_change,
            inverted_relation=eli.new_relation,
        )

        response_options = "\n\n".join("* " + opt for opt in self.get_response_options_dict().values())

        known_moa_insert = ""
        if self.insert_known_moas:
            known_moa_insert = self.build_known_moas_subprompt(prompt_info)

        prompt = PROMPT_TEMPLATE[self.prompt_version].format(
            drug_name=drug_name,
            disease_name=disease_name,
            observation_dsecription=observation_dsecription,
            response_options=response_options,
            known_moa_insert=known_moa_insert,
        )

        return prompt

    def get_response_options_dict(self) -> dict[str, str]:
        """The options have been text-wrapped for presentation in the prompt."""
        response_options = {key: "\n".join(textwrap.wrap(opt, width=90))
                            for key, opt in self.response_options.items()}
        return response_options

    @staticmethod
    def get_summary_heading() -> str:
        """The prompt asks the LLM to output this heading, followed by one of the provided options."""
        return "Summary"

    def build_known_moas_subprompt(self, prompt_info: DrugDiseasePromptInfo) -> str:

        edit_link_info = prompt_info.edit_link_info

        known_moas_prompt = ""

        moa = self.drugmechdb.get_indication_graph_with_id(edit_link_info.source_moa_id)

        drug_name = capitalize_node_name(prompt_info.drug_name, moa.get_node_entity_type(prompt_info.drug_id))
        disease_name = capitalize_node_name(prompt_info.disease_name, moa.get_node_entity_type(prompt_info.disease_id))

        known_moas_prompt += INSERT_KNOWN_MOA_PROMPT.format(drug_name=drug_name, disease_name=disease_name,
                                                            moa_repr=get_drugmechdb_moa_for_prompt(moa))

        return known_moas_prompt

# /
