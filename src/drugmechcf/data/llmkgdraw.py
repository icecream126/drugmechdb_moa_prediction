"""
Drawing the LLMKG graphs
"""

import itertools
from typing import Union

import networkx as nx
# import matplotlib.pyplot as plt

from drugmechcf.data.llmkg import LLMKG, load_llmkg


# -----------------------------------------------------------------------------
#   Globals
# -----------------------------------------------------------------------------

ENTITY_TYPE_COLORS = {
    'drug': 'royalblue',
    'chemical': 'cornflowerblue',
    'protein': 'tab:orange',
    'gene': 'goldenrod',
    'biological process': 'paleturquoise',
    'phenotype': 'lightcoral',
    'disease': 'indianred',
}

# -----------------------------------------------------------------------------
#   Functions
# -----------------------------------------------------------------------------

#
# See also: multipartite_layout()
# https://networkx.org/documentation/stable/reference/generated/networkx.drawing.layout.multipartite_layout.html
#


def draw_llmkg(kg_or_file: Union[str, LLMKG], font_size=6, node_size=20):

    if isinstance(kg_or_file, str):
        kg = load_llmkg(kg_or_file)
    else:
        kg = kg_or_file

    nodelist = list(itertools.chain.from_iterable(kg.get_nodes_for_types(et) for et in kg.get_entity_types()))
    node_labels = {nd: kg.get_node_name(nd) for nd in nodelist}
    node_color = [ENTITY_TYPE_COLORS.get(kg.get_entity_type(nd), 'gray') for nd in nodelist]

    edge_color = 'darkgray'

    # noinspection PyUnresolvedReferences
    nx.draw_networkx(kg, nodelist=nodelist, labels=node_labels, node_color=node_color,
                     node_size=node_size, font_size=font_size,
                     edge_color=edge_color,
                     pos=nx.bfs_layout(kg, nodelist[0])
                     )

    return kg
