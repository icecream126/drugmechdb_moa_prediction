"""
Shared items for prompts, esp. DrugMechDB related
"""


import networkx as nx

from drugmechcf.data.moagraph import MoaGraph
from drugmechcf.utils.misc import capitalize_words


# -----------------------------------------------------------------------------
#   Globals: Misc
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


# -----------------------------------------------------------------------------
#   Functions
# -----------------------------------------------------------------------------

def capitalize_node_name(name: str, node_type: str) -> str:
    if node_type in DONT_CAPITALIZE_TYPES:
        return name
    else:
        return capitalize_words(name)


def get_all_entity_types_for_prompt() -> str:
    all_e_types = sorted([et[0] for et in DRUGMECHDB_ENTITY_TYPE_TO_PROMPT.values()])
    return ", ".join([capitalize_words(et) for et in all_e_types])


def get_translated_entity_type_for_prompt(entity_type: str, plural=False) -> str:
    idx = 1 if plural else 0
    translated_type_name = DRUGMECHDB_ENTITY_TYPE_TO_PROMPT.get(entity_type, [entity_type, entity_type])[idx]
    return capitalize_words(translated_type_name)


def get_qualified_node_name_for_prompt(node: str, moa: MoaGraph) -> str:
    n_type = moa.get_node_entity_type(node)
    n_name = capitalize_node_name(moa.get_node_name(node), n_type)
    n_type_name = get_translated_entity_type_for_prompt(n_type)
    node_name = f"{n_type_name}: {n_name}"
    return node_name


def get_drugmechdb_moa_for_prompt(moa: MoaGraph) -> str:
    triples = []
    try:
        for node in nx.lexicographical_topological_sort(moa):
            # noinspection PyArgumentList
            for head, tail, reln in moa.out_edges(node, keys=True):
                triples.append([head, reln, tail])

    except (nx.NetworkXError, nx.NetworkXUnfeasible):
        # IF graph is not acyclic
        print(f"WARNING (in get_drugmechdb_moa_for_prompt): MoA {moa.get_graph_id()}  may have cycles!")
        # noinspection PyArgumentList
        for head, tail, reln in moa.edges(keys=True):
            triples.append([head, reln, tail])

    interactions = [" | ".join([get_qualified_node_name_for_prompt(head, moa),
                                reln,
                                get_qualified_node_name_for_prompt(tail, moa)])
                    for head, reln, tail in triples]

    return "\n".join(interactions)
