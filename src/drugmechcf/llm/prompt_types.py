import dataclasses
from enum import Enum, StrEnum
from typing import List, Dict, Any


# -----------------------------------------------------------------------------
#   Classes
# -----------------------------------------------------------------------------


class PromptSource(Enum):
    DRUG_MECH_DB = 1

    PRIME_KG = 2
# /


class QueryType(StrEnum):
    # Keep values same as the names, for serializing/de-serializing convenience.

    KNOWN_MOA = "KNOWN_MOA"
    """For retrieving Known MoAs."""

    ADD_LINK = "ADD_LINK"
    """For new MoA created by adding a new interaction."""

    DELETE_LINK = "DELETE_LINK"
    """For making a known MoA invalid by indicating that an interaction edge is invalid."""

    CHANGE_LINK = "CHANGE_LINK"
    """For making a known MoA invalid by changing an interaction edge."""
# /


class PromptStyle(Enum):
    ANONYMIZED_DISEASE = 0
    """No disease name, only disease associations."""

    NAMED_DISEASE = 1
    """Disease name only."""

    NAMED_DISEASE_WITH_ALL_ASSOCIATIONS = 2
    """Disease name, with associations."""
# /


@dataclasses.dataclass
class EditLinkInfo:
    """
    Data for `query_type` = QueryType.ADD_LINK, *.DELETE_LINK, *.CHANGE_LINK.
    """

    source_moa_id: str
    source_node: str
    source_node_name: str
    source_node_type: str

    target_moa_id: str
    target_node: str
    target_node_name: str
    target_node_type: str

    new_relation: str = None
    """
    New relation between source_node, target_node, if any.
    Needed for QueryType.ADD_LINK, *.CHANGE_LINK.
    """

    original_relation: str = None
    """
    Original relation between source_node, target_node, if any.
    Needed for QueryType.DELETE_LINK.
    """

    source_change_descr: str = None
    """
    For Edit-Link prompts
    """

    source_node_level_change: str = None
    """
    For Edit-Link prompts
    """

    def to_serialized(self) -> Dict[str, Any]:
        """
        Make a serializable dict repr
        """
        # noinspection PyTypeChecker
        d = dataclasses.asdict(self)
        return d

# /


@dataclasses.dataclass
class DrugDiseasePromptInfo:
    drug_id: str
    """
    Drug ID, typically also the Drug-Node in a MoA
    """

    disease_id: str
    """
    Disease ID, typically also the Disease-Node in a MoA
    """

    prompt_style: PromptStyle

    prompt_version: int

    source_kg: PromptSource
    """
    Which source KG was used. Currently only 2: DrugMechDB, PrimeKG
    """

    is_negative_sample: bool
    """
    Whether this sample is a negative sample: i.e. no MoA is expected.
    """

    query_type: QueryType = QueryType.KNOWN_MOA
    """
    The type of query this prompt is for.
    """

    moa_id: str = None
    """
    An ID for the MoA for (drug_id, disease_id), if there is one.
    """

    edit_link_info: EditLinkInfo = None
    """
    Additional information when query_type is ADD_LINK / DELETE_LINK / CHANGE_LINK.
    """

    # Prompt and components

    full_prompt: str = None
    """
    The complete prompt to use with a LLM.
    """
    # TODO: Split into (system, user)?

    disease_subprompt: str = None
    """
    The sub-prompt that describes and/or names the Disease.
    """

    drug_disease_subprompt: str = None
    """
    The subprompt that describes the Drug and the Disease.
    Includes the `disease_subprompt`.
    """

    # Drug and Disease names

    disease_name: str = None

    drug_name: str = None

    disease_prompt_nodes: List[str] = dataclasses.field(default_factory=list)
    """
    Additional nodes used to describe the disease, if any.
    """

    def to_serialized(self) -> Dict[str, Any]:
        """
        Make a serializable dict repr
        """
        # noinspection PyTypeChecker
        d = dataclasses.asdict(self)
        d["prompt_style"] = self.prompt_style.name
        d["source_kg"] = self.source_kg.name
        return d

    @staticmethod
    def from_serialized(d: Dict[str, Any]) -> "DrugDiseasePromptInfo":
        """
        Inverse of `to_dict()`
        """
        if isinstance(d["prompt_style"], str):
            d["prompt_style"] = PromptStyle[d["prompt_style"]]

            # Backward compatibility
            map_old_names = dict(DrugMechDB=PromptSource.DRUG_MECH_DB.name,
                                 PrimeKG=PromptSource.PRIME_KG.name)
            d["source_kg"] = PromptSource[ map_old_names.get(d["source_kg"], d["source_kg"]) ]

        if d["add_link_info"] is not None:
            d["add_link_info"] = EditLinkInfo(**d["add_link_info"])

        # Ensure value is StrEnum
        d["query_type"] = QueryType[d["query_type"]]

        return DrugDiseasePromptInfo(**d)
# /
