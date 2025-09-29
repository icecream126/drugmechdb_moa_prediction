"""
Common prompt templates for Drug-Disease Queries
"""

from drugmechcf.llm.prompt_types import QueryType, PromptStyle


# Vars are: prompt_persona, prompt_goal, prompt_instructions, prompt_examples, drug_disease_prompt
PROMPT_TEMPLATE = """
{prompt_persona}
-Goal-
{prompt_goal}

-Steps-
{prompt_instructions}

######################
-Examples-
######################
{prompt_examples}
-Real Data-
######################
{drug_disease_prompt}
######################
Output:
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


ANONYMIZED_DISEASE_NAME = "new disease"


# Alternatives for {disease_subprompt} in DISEASE_NAMED_DRUG_PROMPT
# Vars are: disease_name, associations
DISEASE_SUBPROMPTS = {
    PromptStyle.ANONYMIZED_DISEASE: "A new disease has been identified which is associated with {associations}.",

    PromptStyle.NAMED_DISEASE: "I would like to find a treatment for the disease {disease_name}.",

    PromptStyle.NAMED_DISEASE_WITH_ALL_ASSOCIATIONS:
        ("I would like to find a treatment for the disease {disease_name},\n"
         "which is associated with {associations}."),
}


# PromptStyle => Relation. IF not provided THEN use original 'disease_association_relation'
# For use in: example_llm_response
DISEASE_ASSOCIATION_RELATIONS = {
    PromptStyle.ANONYMIZED_DISEASE: "associated with",
    PromptStyle.NAMED_DISEASE_WITH_ALL_ASSOCIATIONS: "associated with",
}


# {drug_disease_prompt} Prompts
# Dict: QueryType => str
DRUG_DISEASE_PROMPTS = {

    # Vars are: disease_subprompt, drug_name
    QueryType.KNOWN_MOA: """{disease_subprompt}
Will the drug {drug_name} be useful in treating this disease?
If so, please provide a Mechanism of Action for the drug.
""",

    # Vars: target_disease_name, source_drug_name,
    #       source_node_type, source_node_name, new_relation, target_node_type, target_node_name
    QueryType.ADD_LINK:
"""We have some patients with an unusual combination of symptoms that are being treated with
an experimental drug. The drug is helping with some of the symptoms.
However some patients are showing signs of {target_disease_name}, but the common drugs for
{target_disease_name} are contraindicated for these patients.
We have also observed a novel interaction in these patients: in the presence of the experimental drug,
the {source_node_type} {source_node_name} {new_relation} the {target_node_type} {target_node_name}.
Given this observation, for these patients, could the drug {source_drug_name} be useful
in treating {target_disease_name}?
If so, please provide a potential Mechanism of Action for the drug.
        """,

    # Vars: target_disease_name, source_drug_name,
    #       source_node_type, source_node_name, original_relation, target_node_type, target_node_name
    QueryType.DELETE_LINK:
"""We have some patients with an unusual combination of symptoms that are being treated with
an experimental drug. The drug is helping with some of the symptoms.
However some patients are showing signs of {target_disease_name}.
We have also observed in these patients that the experimental drug inhibits the following interaction:
the {source_node_type} {source_node_name} {original_relation} the {target_node_type} {target_node_name}.
Given this observation, for these patients, could the drug {source_drug_name} be useful
in treating {target_disease_name}?
If so, please provide a potential Mechanism of Action for the drug.
""",

    # Vars: target_disease_name, source_drug_name,
    #       source_node_type, source_node_name, new_relation, target_node_type, target_node_name
    QueryType.CHANGE_LINK:
"""We have some patients with an unusual combination of symptoms that are being treated with
an experimental drug. The drug is helping with some of the symptoms.
However some patients are showing signs of {target_disease_name}.
We have also observed a novel interaction in these patients: in the presence of the experimental drug,
the {source_node_type} {source_node_name} {new_relation} the {target_node_type} {target_node_name}.
Given this observation, for these patients, could the drug {source_drug_name} be useful
in treating {target_disease_name}?
If so, please provide a potential Mechanism of Action for the drug.
""",

}
