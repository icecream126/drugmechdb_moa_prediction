"""
Data returned by prompt builders
"""

import abc
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from drugmechcf.data.moagraph import MoaGraph
from drugmechcf.llm.prompt_templates import ANONYMIZED_DISEASE_NAME, DRUG_DISEASE_PROMPTS
from drugmechcf.llm.prompt_types import PromptSource, PromptStyle, QueryType, EditLinkInfo, DrugDiseasePromptInfo
from drugmechcf.utils.misc import capitalize_words


# -----------------------------------------------------------------------------
#   Classes
# -----------------------------------------------------------------------------


class PromptBuilder(abc.ABC):

    def __init__(self,
                 source_kg: PromptSource,
                 entity_type_translations: Dict[str, Tuple[str, str]] = None,
                 capitalize_type_names: bool = True
                 ):
        """

        :param source_kg: Indicates source KG
        :param entity_type_translations: For translating KG's Entity-Type name to one for use in Prompt.
            Dict: KG-Entity-Type => (singular type name, plural type name)
        :param capitalize_type_names: Whether Entity-Type names in prompt are capitalized.
        """

        self.source_kg = source_kg

        if entity_type_translations is None:
            self.entity_type_translations = dict()
        else:
            self.entity_type_translations = entity_type_translations

        self.capitalize_type_names = capitalize_type_names

        # --- derived fields ---

        # Translating KG EntityType into Formal Name listed in prompt
        self.entity_type_to_formal_name: Dict[str, str] = {
            kg_etype: self.get_translated_entity_type(kg_etype)
            for kg_etype in self.entity_type_translations.keys()
        }

        # --- Customized behavior -- Set in `init()`

        # EntityType's whose entity-names do not get capitalized
        self.dont_capitalize_types: List[str] = []

        return

    def get_translated_entity_type(self, entity_type: str, plural=False):
        idx = 1 if plural else 0
        translated_type_name = self.entity_type_translations.get(entity_type, [entity_type, entity_type])[idx]
        if self.capitalize_type_names:
            translated_type_name = self.capitalize_type_name(translated_type_name)

        return translated_type_name

    @abc.abstractmethod
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
                                     ) \
            -> DrugDiseasePromptInfo | None:
        """
        Build a full prompt for a LLM, given drug-node `drug_id` and disease-node `disease_id`.

        :param drug_id: Identifies the Drug, typically used as Node id in MoA
        :param disease_id: Identifies the Disease, typically used as Node id in MoA
        :param prompt_style: Prompt style
        :param prompt_version: Prompt version to use, if applicable

        :param query_type: Type of query
        :param moa: The MoA graph from which to build a prompt, if available.

        :param edit_link_info: Additional info relevant only for
            `query_type` = QueryType.ADD_LINK, *.DELETE_LINK, *.CHANGE_LINK.

        :param is_negative_sample: Is this a negative example?
        :param verbose:

        :return: DrugDiseasePromptData if prompt built, or None if prompt not possible
        """
        raise NotImplementedError

    def get_drug_disease_prompt(self, prompt_info: DrugDiseasePromptInfo) -> str:
        """
        Build the {drug_disease_prompt} needed in PROMPT_TEMPLATE.
        It describes the main Drug-Disease query (w/o persona, instructions and examples).
        """

        dd_template = DRUG_DISEASE_PROMPTS[prompt_info.query_type]

        drug_name = capitalize_words(prompt_info.drug_name)
        disease_name = capitalize_words(prompt_info.disease_name)

        if prompt_info.query_type is QueryType.KNOWN_MOA:

            drug_disease_prompt = dd_template.format(disease_subprompt=prompt_info.disease_subprompt,
                                                     drug_name=drug_name)

        elif prompt_info.query_type in [QueryType.ADD_LINK, QueryType.DELETE_LINK, QueryType.CHANGE_LINK]:

            assert prompt_info.edit_link_info is not None, \
                f"`prompt_info.add_link_info`  is required when query_type is '{prompt_info.query_type}'."

            edit_link_info = prompt_info.edit_link_info

            drug_disease_prompt = dd_template.format(
                target_disease_name=disease_name,
                source_drug_name=drug_name,

                source_node_type=edit_link_info.source_node_type,
                source_node_name=self.capitalize_node_name(edit_link_info.source_node_name,
                                                           edit_link_info.source_node_type),

                original_relation=edit_link_info.original_relation,
                new_relation=edit_link_info.new_relation,

                target_node_type=edit_link_info.target_node_type,
                target_node_name=self.capitalize_node_name(edit_link_info.target_node_name,
                                                           edit_link_info.target_node_type),
            )

        else:
            raise NotImplementedError(f"{prompt_info.query_type = } not supported.")

        return drug_disease_prompt

    @abc.abstractmethod
    def build_prompt_info(self,
                          drug_id: str,
                          disease_id: str,
                          prompt_style: PromptStyle,
                          prompt_version: int,
                          *,
                          is_negative_sample: bool = False,
                          moa_id: str = None,
                          query_type: QueryType = QueryType.KNOWN_MOA,
                          edit_link_info: EditLinkInfo = None,
                          ) -> DrugDiseasePromptInfo:

        prompt_info = DrugDiseasePromptInfo(drug_id=drug_id,
                                            disease_id=disease_id,
                                            moa_id=moa_id,
                                            prompt_style=prompt_style,
                                            prompt_version=prompt_version,
                                            source_kg=self.source_kg,
                                            is_negative_sample=is_negative_sample,
                                            query_type=query_type,
                                            edit_link_info=edit_link_info,
                                            )
        return prompt_info

    # -----------------------------------------------------------------------------
    #   Methods - Convenience
    # -----------------------------------------------------------------------------

    @staticmethod
    def get_edit_link_info(ddict: Dict[str, Any], field_name: str = "edit_link_info") \
            -> Optional[EditLinkInfo]:

        if (eli_dict := ddict.get(field_name)) is None:
            return None

        return EditLinkInfo(**eli_dict)

    # -----------------------------------------------------------------------------
    #   Methods - Name Capitalization
    # -----------------------------------------------------------------------------

    def capitalize_node_name(self, name: str, node_type: str) -> str:
        if node_type in self.dont_capitalize_types:
            return name
        else:
            return capitalize_words(name)

    # noinspection PyMethodMayBeStatic
    def capitalize_type_name(self, name: str):
        return capitalize_words(name)

    def get_all_entity_types_for_llm(self) -> str:
        assert self.entity_type_to_formal_name, \
            "`self.entity_type_to_formal_name` is Empty! Did you forget to set it in __init__()?"
        all_entity_types = ", ".join(sorted(self.entity_type_to_formal_name.values()))
        return all_entity_types

    def formal_to_kg_entity_type_names(self) -> Dict[str, str]:
        return dict((v, k) for k, v in self.entity_type_to_formal_name.items())

    @staticmethod
    def get_dis_name_in_llm_response(prompt_style: PromptStyle, disease_name: str) -> str:
        if prompt_style is PromptStyle.ANONYMIZED_DISEASE:
            dis_name_in_llm_response = ANONYMIZED_DISEASE_NAME
        else:
            dis_name_in_llm_response = disease_name

        return dis_name_in_llm_response

# /


# -----------------------------------------------------------------------------
#   Functions - Misc
# -----------------------------------------------------------------------------


def compute_string_similarities(dis_name: str, test_names: List[str]) -> np.ndarray:
    """
    Returns similarity scores of `dis_name` to names in `test_names`.
    :param dis_name:
    :param test_names:
    :return: float vector of length `len(test_names)`, whose contents are similarity scores in [0, 1.0]
    """
    corpus = [dis_name] + test_names
    tvec = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4))
    x = tvec.fit_transform(corpus)
    sims = cosine_similarity(x[0], x[1:])[0]
    return sims
