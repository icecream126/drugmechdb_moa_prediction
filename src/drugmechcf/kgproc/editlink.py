"""
Deleting a Link
"""

from collections import Counter
from typing import List, NamedTuple, Tuple

import networkx as nx

from drugmechcf.data.drugmechdb import (CHEMICAL_ENTITY_TYPE, DRUG_ENTITY_TYPE, GENE_ENTITY_TYPE, PROTEIN_ENTITY_TYPE,
                             DrugMechDB, load_drugmechdb)
from drugmechcf.data.moagraph import MoaGraph

from drugmechcf.kgproc.simulate import get_valid_simulatable_moas

from drugmechcf.utils.misc import pp_underlined_hdg
from drugmechcf.utils.prettytable import pp_seq_key_count


# -----------------------------------------------------------------------------
#   Globals
# -----------------------------------------------------------------------------

EDITABLE_LINK_ENTITY_TYPES = [CHEMICAL_ENTITY_TYPE, DRUG_ENTITY_TYPE, GENE_ENTITY_TYPE, PROTEIN_ENTITY_TYPE]


# Dict: Relation => Inverse-relation. Target-in-reln must be one of these keys.
REQD_RELN_INVERSE = {"decreases activity of": "increases activity of",
                     "increases activity of": "decreases activity of",
                     "negatively regulates": "positively regulates",
                     "positively regulates": "negatively regulates",
                     "decreases abundance of": "increases abundance of",
                     "increases abundance of": "decreases abundance of",
                     # The following are non-causal
                     # ... Adding the next 2 increases the nbr editable links by only 136.
                     # "positively correlated with": "negatively correlated with",
                     # "negatively correlated with": "positively correlated with",
                     }


# -----------------------------------------------------------------------------
#   Classes
# -----------------------------------------------------------------------------


class EditableMoaLink(NamedTuple):
    moa: MoaGraph
    source_node: str
    target_node: str
    relation: str
    inverse_relation: str
# /


# -----------------------------------------------------------------------------
#   Functions
# -----------------------------------------------------------------------------


def cut_edges(moa: MoaGraph, root_node: str = None, sink_node: str = None) -> List[Tuple[str, str]]:
    """
    Return list of individual (s, t)-path cut-edges in `moa`.
    A cut-edge here is a bottleneck in the path from the root (Drug) to the sink (Disease):
        IF the edge is removed,
        THEN the root (Drug) will no longer be connected to the sink (Disease).

    :return: [(u, v), ...]
        Empty list is returned if no cut-edges.
    """

    # The algorithm is:
    # In any top-sort, let top-sort-index(u) = position (index) of node u in the top-sort.
    # Then, any consecutive pair of nodes (u, v), at positions (u.idx, u.idx + 1) in the top-sort, is a cut-edge iff:
    #   - graph has an edge (u, v)
    #   - There does not exist an edge (u', v') != (u, v) s.t.
    #       + top-sort-index(u') <= u.idx AND top-sort-index(v') >= u.idx + 1
    #
    # Note: This algorithm only finds cut-edges for (s, t)-paths, and not for all DAGs.

    edge_cuts = []

    try:
        top_sort = list(nx.topological_sort(moa))
    except nx.NetworkXUnfeasible:
        return edge_cuts

    # Note: For a general DAG, we would remove all nodes not reachable from root, and all nodes that do not reach sink.
    # The following is a short-cut, with the expectation that all nodes are reachable to-sink / from-root.

    # Just in case the expected node (e.g. Drug) is not the root
    root_idx = 0
    if root_node is not None and root_node != top_sort[root_idx]:
        root_idx = top_sort.index(root_node)

    # Just in case the expected node (e.g. Disease) is not the sink
    sink_idx = len(top_sort) - 1
    if sink_node is not None and sink_node != top_sort[sink_idx]:
        sink_idx = top_sort.index(sink_node)

    top_sort = top_sort[root_idx : sink_idx + 1]

    # {node => index}: Each node's index in the top_sort
    topsort_node_idx = dict((nd, idx) for idx, nd in enumerate(top_sort))

    # {node-index => max_index}: For each node, the highest top_sort-Index among its successors
    node_max_successor_idx = dict()
    for uidx, u in enumerate(top_sort):
        node_max_successor_idx[uidx] = max([topsort_node_idx[v] for v in moa.successors(u)], default=0)

    uidx = 0
    while uidx < len(top_sort) - 1:
        u = top_sort[uidx]

        if (max_idx := node_max_successor_idx[uidx]) > uidx + 1:
            # Jump to position of the highest successor to nodes in range [uidx, max_idx - 1]
            uidx = max(node_max_successor_idx[i] for i in range(uidx, max_idx))
            continue

        v = top_sort[uidx + 1]
        if moa.has_edge(u, v):
            edge_cuts.append((u, v))

        uidx += 1

    return edge_cuts


def get_editable_edges(moa: MoaGraph, root_node: str = None, sink_node: str = None) -> List[EditableMoaLink]:
    """
    Return edges valid for Deletion or Inversion.
    Edges must be for one of the relations in REQD_RELN_INVERSE, between nodes of EDITABLE_LINK_ENTITY_TYPES.
    Target node cannot be a Drug.

    :return: [(u, relation, v), ...]
        Empty list is returned if no cut-edges.
    """

    editable_edges = []
    for u, v in cut_edges(moa, root_node, sink_node):

        if (moa.get_node_entity_type(u) not in EDITABLE_LINK_ENTITY_TYPES or
                moa.get_node_entity_type(v) not in EDITABLE_LINK_ENTITY_TYPES):
            continue

        # Target cannot be Drug
        if moa.get_node_entity_type(v) == DRUG_ENTITY_TYPE:
            continue

        uv_relns = list(moa[u][v].keys())
        # Only if there is a single edge of one of the required relations
        if len(uv_relns) == 1 and uv_relns[0] in REQD_RELN_INVERSE:
            editable_edges.append(EditableMoaLink(moa=moa, source_node=u, target_node=v,
                                                  relation=uv_relns[0],
                                                  inverse_relation=REQD_RELN_INVERSE[uv_relns[0]]))

    return editable_edges


def select_editable_moa_links(drugmechdb: DrugMechDB) -> List[EditableMoaLink]:
    """
    Candidate Editable-MoA-Link must be:
        - Cut-edge
        - between nodes of EDITABLE_LINK_ENTITY_TYPES, Target node cannot be a Drug.
        - For reln one of REQD_RELN_INVERSE

    :returns: List: [EditableLink, ...]

        See `DrugMechDB/report_editlink_stats.txt` for stats.

        Full DrugMechDB
        ---------------

        Nbr Editable MoA-Links                 = 3,310
        Nbr unique Editable MoA's              = 2,847
        Nbr unique Editable Drug-Disease pairs = 2,847

        Nbr editable Links source node is the Drug = 2,575 ... 77.8%
        Nbr editable Links source node is Not Drug =   735 ... 22.2%

        Simulatable DrugMechDB
        ----------------------

        Nbr Editable MoA-Links                 = 2,499
        Nbr unique Editable MoA's              = 2,165
        Nbr unique Editable Drug-Disease pairs = 2,165

        Nbr editable Links source node is the Drug = 2,151 ... 86.1%
        Nbr editable Links source node is Not Drug =   348 ... 13.9%

    """

    editable_moa_links = []
    for moa in drugmechdb.indication_graphs:

        drug_node, disease_node = drugmechdb.get_moa_drug_disease_nodes(moa)

        # Skip because this (drug_id, disease_id) has multiple MoA's,
        # which will prevent any editable link from being a true cut-edge between that Drug-Disease pair.
        # This eliminates 237 MoAs, 310 (MoA, Link) pairs.
        if len(drugmechdb.get_indication_graphs(drug_node, disease_node)) > 1:
            continue

        editable_moa_links.extend(get_editable_edges(moa, drug_node, disease_node))

    return editable_moa_links


def pp_editable_moa_links(drugmechdb: DrugMechDB = None, count: int = 3):
    if drugmechdb is None:
        drugmechdb = load_drugmechdb()

    editable_moa_links = select_editable_moa_links(drugmechdb)

    print("\n")
    for i, editable_link in enumerate(editable_moa_links[: count], start=1):
        pp_editable_link(drugmechdb, editable_link, sample_nbr = i)

    return


def pp_editable_link(drugmechdb: DrugMechDB, editable_link: EditableMoaLink, sample_nbr=0):

    moa = editable_link.moa
    drug_node = drugmechdb.get_moa_source_drug_node(moa)
    disease_node = drugmechdb.get_moa_target_disease_node(moa)

    hdg = (("" if drug_node == editable_link.source_node else f"{moa.get_node_name(drug_node)} --> ")
           +
           moa.get_qualified_node_name(editable_link.source_node) + " ==> " +
           editable_link.relation + " ==> " +
           moa.get_qualified_node_name(editable_link.target_node)
           +
           ("" if editable_link.target_node == disease_node else f" --> {moa.get_qualified_node_name(disease_node)}")
           )

    pp_underlined_hdg(f"[{sample_nbr}] {hdg}", overline=True)
    moa.pprint(with_summary=True)
    print("\n", flush=True)
    return


def analyze_stats():
    drugmechdb = load_drugmechdb()
    print()
    pp_underlined_hdg("Stats for Full DrugMechDB", linechar='=', overline=True)
    analyze_stats_drugmechdb(drugmechdb)
    print()

    reduced_dmdb = get_valid_simulatable_moas(drugmechdb)
    print()
    pp_underlined_hdg("Stats for Simulatable MoAs in DrugMechDB", linechar='=', overline=True)
    analyze_stats_drugmechdb(reduced_dmdb)
    print()
    return


def analyze_stats_drugmechdb(drugmechdb):
    editable_moa_links = select_editable_moa_links(drugmechdb)

    n_drug_dis = len(set(drugmechdb.get_moa_drug_disease_nodes(moa)
                         for moa in drugmechdb.get_indication_graphs()))

    n_drug_dis_editable = len(set(drugmechdb.get_moa_drug_disease_nodes(x.moa)
                                  for x in editable_moa_links))

    n_editable_moa_links = len(editable_moa_links)
    n_source_is_drug = sum([drugmechdb.get_moa_source_drug_node(x.moa) == x.source_node
                            for x in editable_moa_links])

    print(f"Nbr MoA's in full DrugMechDB  = {drugmechdb.nbr_indications():5,d}")
    print(f"Nbr unique Drug-Disease pairs = {n_drug_dis:5,d}")
    print()
    print(f"Nbr Editable MoA-Links                 = {len(editable_moa_links):5,d}")
    print(f"Nbr unique Editable MoA's              = {len(set(x.moa for x in editable_moa_links)):5,d}")
    print(f"Nbr unique Editable Drug-Disease pairs = {n_drug_dis_editable:5,d}")
    print()
    print(f"Nbr editable Links source node is the Drug = {n_source_is_drug:5,d}",
          f"... {n_source_is_drug/n_editable_moa_links:.1%}")
    print(f"Nbr editable Links source node is Not Drug = {n_editable_moa_links - n_source_is_drug:5,d}",
          f"... {(n_editable_moa_links - n_source_is_drug)/n_editable_moa_links:.1%}")
    print()

    pp_underlined_hdg("Frequency of Source-Node-Type, Target-Node-Type in Editable Edges:")
    ctr = Counter(f"{x.moa.get_node_entity_type(x.source_node)}, {x.moa.get_node_entity_type(x.target_node)}"
                  for x in editable_moa_links)
    pp_seq_key_count(ctr.most_common(), total_count=ctr.total(), add_total=True)
    return


# ======================================================================================================
#   Main
# ======================================================================================================

# To run
# ------
#
# [Python]$ python -m drugmechcf.kgproc.editlink stats
#

if __name__ == "__main__":

    import argparse
    from datetime import datetime
    from drugmechcf.utils.misc import print_cmd

    _argparser = argparse.ArgumentParser(
        description='Valid Edit-Link MoAs in DrugMechDB.',
    )

    _subparsers = _argparser.add_subparsers(dest='subcmd',
                                            title='Available commands',
                                            )
    # Make the sub-commands required
    _subparsers.required = True

    # ... batch
    _sub_cmd_parser = _subparsers.add_parser('stats',
                                             help="Analyze +ive Edit-Link samples from DrugMechDB.")

    # ...

    _args = _argparser.parse_args()
    # .................................................................................................

    start_time = datetime.now()

    print("---------------------------------------------------------------------")
    print_cmd()

    if _args.subcmd == 'stats':

        analyze_stats()

    else:

        raise NotImplementedError(f"Command not implemented: {_args.subcmd}")

    # /

    print('\nTotal Run time =', datetime.now() - start_time)
    print()
