"""
Some PrimeKG stats
"""

from collections import defaultdict, Counter
from typing import List, Sequence, Set, Tuple, Union

import numpy as np

from drugmechcf.data.primekg import (PrimeKG, load_primekg,
                          DRUG_ENTITY_TYPE, DISEASE_ENTITY_TYPE,
                          RELATION_INDICATION, RELATION_OFF_LABEL, RELATION_CONTRA_INDICATION, DRUG_DISEASE_RELATIONS)
from drugmechcf.data.mondo import load_mondo

from drugmechcf.utils.misc import pp_underlined_hdg, pp_funcargs, reset_df_index
from drugmechcf.utils.prettytable import PrettyTable, pp_counts


# -----------------------------------------------------------------------------
#   Globals
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
#   Functions
# -----------------------------------------------------------------------------


def basic_stats():
    kg = load_primekg()
    opts = kg.opts
    df_kg = PrimeKG.read_kg(opts.srcdir)
    mondo = load_mondo(opts.mondo_opts_file)
    print()

    pp_underlined_hdg("Entities Report", linechar="=", overline=True)
    print()

    pp_underlined_hdg("Entity types and source in PrimeKG")
    # It is a bi-directional kg, so we only need to look at head entities 'x_*'
    counts = df_kg[['x_type', 'x_source', 'x_id']].drop_duplicates().value_counts(['x_type', 'x_source'], sort=False)
    crows = [[*k, cnt] for k, cnt in counts.items()]
    pp_counts(["Ent-type", "Ent-source", "Count"], ["s", "s", ",d"],
              rows=crows, count_col_idx=2)
    print()
    print("Note: 'MONDO_grouped' refers to several MONDO diseases grouped into one entity, with a combined id.")
    print("Examples:")
    dis_examples = df_kg[df_kg.x_source == 'MONDO_grouped'][['x_id', 'x_name']].drop_duplicates()[:3]
    for i, dis in enumerate(dis_examples.itertuples(), start=1):
        print(f"  [{i}] id = {dis.x_id}",
              f"      name = {dis.x_name}", "", sep="\n")
    print()

    print("Examples of Drug nodes:")
    for i, drug_nd in enumerate(kg.get_nodes_for_types(DRUG_ENTITY_TYPE)[:3], start=1):
        drug_data = kg.nodes[drug_nd]
        print(f"  [{i}] node = {drug_nd}, names = {', '.join(drug_data['names'])}")
    print("\n")

    all_dis_nodes = kg.get_nodes_for_types(DISEASE_ENTITY_TYPE)
    print(f"nbr Disease nodes = {len(all_dis_nodes):,d}")
    print(f"nbr Grouped Disease nodes = {len([nd for nd in all_dis_nodes if nd.startswith('MONDO_grouped')]):,d}")
    n_obsolete_diseases = sum([any(mondo.nodes[mondoid]["is_obsolete"]
                                   for mondoid in kg.build_mondo_ids(nd))
                               for nd in all_dis_nodes
                               ])
    print(f"nbr Disease nodes that are obsolete in Mondo = {n_obsolete_diseases:,d}")
    n_dis = len([nd for nd in all_dis_nodes
                 if kg.nodes[nd]["mesh_xrefs"]])
    print(f"nbr Disease nodes with MeSH xrefs = {n_dis:,d}")
    print("\n")

    pp_underlined_hdg("Relations Report", linechar="=", overline=True)
    print()

    print("Note: Homogeneous relations, those whose head and tail entities have the same type,",
          "      are considered symmetric, and have a parallel reverse edge with the same relation name.",
          "      However, many of these (except 'protein_protein') actually describe the ordered 'parent-child'",
          "      relation, but because each also has a reverse edge, the canonical order is lost.",
          "",
          "      Non-homogeneous relations have a reverse edge with prefx 'rev-' added to the relation name.",
          sep="\n")
    print("\n")

    pp_underlined_hdg("Relations and head/tail Entity types in PrimeKG, sorted on decreasing count")
    counts = kg.get_edge_counts()
    colnames = ["#"] + counts.columns.tolist()
    colfmts = ["d", "s", "s", "s", ",d"]
    pp_counts(colnames, colfmts, rows=list(counts.itertuples()), count_col_idx=4, add_index_col=False)
    print("\n")

    pp_underlined_hdg("Relations and head/tail Entity types in PrimeKG, sorted on arg entity types")
    counts = reset_df_index(counts.sort_values(["htype", "ttype", "relation"]))
    pp_counts(colnames, colfmts, rows=list(counts.itertuples()), count_col_idx=4, add_index_col=False)
    print()

    pp_underlined_hdg("Relations and head/tail Entity types in PrimeKG, sorted on name, display_name")
    counts = reset_df_index(kg.get_edge_counts(with_display_name=True).sort_values(["relation", "display_name"]))
    colnames = ["#"] + counts.columns.tolist()
    colfmts = ["d", "s", "s", "s", "s", ",d"]
    pp_counts(colnames, colfmts, rows=list(counts.itertuples()), count_col_idx=5, add_index_col=False)
    print()

    pp_underlined_hdg("Relations depicting 'parent-child' relationships")
    df = reset_df_index(counts[counts.display_name == "parent-child"][["relation", "display_name"]])
    ptbl = PrettyTable(for_md=True)
    ptbl.set_colnames(["#", "relation", "display_name"], ["d", "s", "s"])
    for n, row in enumerate(df.itertuples(), start=1):
        ptbl.add_row_(n, row[1], row[2])
    print(ptbl)
    print()

    pp_underlined_hdg("Drug-Protein relations with 'display_name' subtypes")
    counts = kg.get_edge_counts(relation="drug_protein", with_display_name=True)
    colnames = ["#"] + counts.columns.tolist()
    colfmts = ["d", "s", "s", "s", "s", ",d"]
    pp_counts(colnames, colfmts, rows=list(counts.itertuples()), count_col_idx=5, add_index_col=False)
    print("Note: The 'rev-drug_protein' relation has the same breakdown.")
    print("\n")

    return


def pp_entity_entity_degrees(tgt_entity_type_name: str,
                             all_entity_types: List[str],
                             tgt_to_etype_conns: np.ndarray,
                             tgt_from_etype_conns: np.ndarray):
    """

    :param tgt_entity_type_name:
    :param all_entity_types: List of all entity-types
    :param tgt_to_etype_conns: array with shape (n_tgts, n_ent_types)
    :param tgt_from_etype_conns: array with shape (n_tgts, n_ent_types)
    """

    ent_type_idx = dict((t, i) for i, t in enumerate(all_entity_types))

    tot_ent_conns = np.sum(tgt_to_etype_conns, axis=0) + np.sum(tgt_from_etype_conns, axis=0)

    ent_types = [all_entity_types[i] for i in np.nonzero(tot_ent_conns)[0]]

    pp_underlined_hdg(f"In/Out degrees for nodes of type: '{tgt_entity_type_name}'")

    ptbl = PrettyTable(for_md=True)

    ptbl.set_colnames(["Neighbor-Type",
                       "min(out)", "median(out)", "avg(out)", "max(out)",
                       "min(in)",  "median(in)",  "avg(in)",  "max(in)"],
                      ["s",
                       ",d", ".2f", ".2f", ",d",
                       ",d", ".2f", ".2f", ",d"])

    for etype in sorted(ent_types):
        e_idx = ent_type_idx[etype]
        tgt_to_e = tgt_to_etype_conns[:, e_idx]
        e_to_tgt = tgt_from_etype_conns[:, e_idx]

        out_max = tgt_to_e.max()
        in_max = e_to_tgt.max()

        row = ([etype] +
               (["-", "-", "-", "-"] if out_max == 0
                else [tgt_to_e.min(), np.median(tgt_to_e), tgt_to_e.mean(), out_max])
               +
               (["-", "-", "-", "-"] if in_max == 0
                else [e_to_tgt.min(), np.median(e_to_tgt), e_to_tgt.mean(), in_max])
               )

        ptbl.add_row(row)

    out_degrees = np.sum(tgt_to_etype_conns, axis=1)
    in_degrees = np.sum(tgt_from_etype_conns, axis=1)
    ptbl.add_row_("ALL",
                  out_degrees.min(), np.median(out_degrees), out_degrees.mean(), out_degrees.max(),
                  in_degrees.min(), np.median(in_degrees), in_degrees.mean(), in_degrees.max()
                  )
    print(ptbl)
    return


def pp_connectivity_counts(tgt_entity_type_name: str,
                           all_entity_types: List[str],
                           tgt_to_etype_conns: np.ndarray,
                           tgt_from_etype_conns: np.ndarray,
                           entity_group: List[str] = None):
    """

    :param tgt_entity_type_name:
    :param all_entity_types: List of all entity-types
    :param tgt_to_etype_conns: array with shape (n_tgts, n_ent_types)
    :param tgt_from_etype_conns: array with shape (n_tgts, n_ent_types)
    :param entity_group: List of entities to provide grouped connectivity counts for
    """

    ent_type_idx = dict((t, i) for i, t in enumerate(all_entity_types))

    tot_ent_conns = np.sum(tgt_to_etype_conns, axis=0) + np.sum(tgt_from_etype_conns, axis=0)

    ent_types = [all_entity_types[i] for i in np.nonzero(tot_ent_conns)[0]]

    n_trgts = tgt_to_etype_conns.shape[0]

    pp_underlined_hdg(f"Nbr target type Nodes by Connectivity, for nodes of type: '{tgt_entity_type_name}'")

    print(f"Total nbr nodes for type '{tgt_entity_type_name}' = {n_trgts:,d}")
    print()

    ptbl = PrettyTable(for_md=True)

    ptbl.set_colnames(["Neighbor-Type",
                       "0 nghbrs", "(0 nghbrs)%", "1 nghbr", "(1 nghbr)%", "2+ nghbrs", "(2+ nghbrs)%"],
                      ["s"] + ([",d", ".2%"] * 3))

    # ---
    def add_row(row_name, nghbr_counts):
        n_zero = np.sum(nghbr_counts == 0)
        n_one = np.sum(nghbr_counts == 1)
        n_2plus = np.sum(nghbr_counts > 1)
        ptbl.add_row_(row_name, n_zero, n_zero / n_trgts, n_one, n_one / n_trgts, n_2plus, n_2plus / n_trgts)
    # ---

    for etype in sorted(ent_types):
        e_idx = ent_type_idx[etype]
        node_nghbrs = tgt_to_etype_conns[:, e_idx] + tgt_from_etype_conns[:, e_idx]
        add_row(etype, node_nghbrs)

    if entity_group:
        grp_degrees = np.zeros(n_trgts, dtype=np.int32)
        for etype in entity_group:
            e_idx = ent_type_idx[etype]
            grp_degrees += tgt_to_etype_conns[:, e_idx] + tgt_from_etype_conns[:, e_idx]
        add_row("Group (*)", grp_degrees)

    out_degrees = np.sum(tgt_to_etype_conns, axis=1)
    in_degrees = np.sum(tgt_from_etype_conns, axis=1)
    all_degrees = out_degrees + in_degrees
    add_row("ALL", all_degrees)

    print(ptbl)

    if entity_group:
        print("   (*) Group =", " + ".join(entity_group))
        print()

    return


def connectivity():
    kg = load_primekg()

    print()
    print("",
          "NOTE: Reverse edges ignored for heterogeneous relations",
          "      (where head and tail entities have different types)",
          "", sep="\n")
    print()

    for tgt_type in sorted(kg.get_entity_types()):
        tgt_to_etype_conns, tgt_from_etype_conns, _ =\
            get_connectivity_for(kg.get_nodes_for_types(tgt_type), kg)

        pp_entity_entity_degrees(tgt_type, kg.get_entity_types(), tgt_to_etype_conns, tgt_from_etype_conns)
        print()

        if tgt_type == DISEASE_ENTITY_TYPE:
            pp_connectivity_counts(tgt_type, kg.get_entity_types(), tgt_to_etype_conns, tgt_from_etype_conns,
                                   entity_group=["effect/phenotype", "exposure", "gene/protein"])
            print()

    return


def node_repr(kg, node):
    ndata = kg.nodes[node]
    return f"({ndata['EntityType']}) {node} - {', '.join(ndata['names'])}"


def edge_repr(kg, head, tail, rdata, in_out: str = None):
    rname = rdata["Relation"]
    if (displ_nm := rdata["name"]) != rname:
        rname += " - " + displ_nm

    if in_out == "in":
        return f"{{{rname}}} <= {node_repr(kg, head)}"
    elif in_out == "out":
        return f"{{{rname}}} => {node_repr(kg, tail)}"
    else:
        return f"{node_repr(kg, head)} => {{{rname}}} => {node_repr(kg, tail)}"


def pp_node_neighborhood(kg: PrimeKG, node: str, uni_directional: bool = True, nbr: int = None,
                         neighbor_types: List[str] = None):

    prefix = f"[{nbr:3d}] " if nbr is not None else ""
    indent = " " * len(prefix)

    print()
    pp_underlined_hdg(f"{prefix}{node_repr(kg, node)}")

    node_is_disease = kg.get_entity_type(node) == "disease"
    drug_connections = defaultdict(list)

    print(f"{indent}In-edges:")
    ne = 0
    for head, tail, rdata in kg.in_edges(node, data=True):
        reln = rdata["Relation"]
        if uni_directional and reln.startswith("rev-"):
            continue

        if node_is_disease and reln in DRUG_DISEASE_RELATIONS:
            drug_connections[reln].append((head, tail, rdata))

        if neighbor_types and kg.get_entity_type(head) not in neighbor_types:
            continue

        ne += 1
        if not (node_is_disease and reln in DRUG_DISEASE_RELATIONS):
            print(f'{indent}  {ne:2d}. {edge_repr(kg, head, tail, rdata, "in")}')

    if drug_connections:
        print()
        print(f"{indent}Associated Drugs:")
        ne = 0
        for reln in sorted(drug_connections.keys()):
            for head, tail, rdata in drug_connections[reln]:
                ne += 1
                print(f'{indent}  {ne:2d}. {edge_repr(kg, head, tail, rdata, "in")}')

    print()
    print(f"{indent}Out-edges:")
    ne = 0
    for head, tail, rdata in kg.out_edges(node, data=True):
        if uni_directional and rdata["Relation"].startswith("rev-"):
            continue
        if neighbor_types and kg.get_entity_type(tail) not in neighbor_types:
            continue
        ne += 1
        print(f'{indent}  {ne:2d}. {edge_repr(kg, head, tail, rdata, "out")}')

    print()
    return


def compare_diseases_on_treatable():
    kg = load_primekg()
    print()

    dis_nodes = kg.get_nodes_for_types("disease")
    n_dis = len(dis_nodes)
    print()
    print(f"Nbr disease nodes = {n_dis:,d}")
    print()

    in_relations = DRUG_DISEASE_RELATIONS
    tgt_to_etype_conns, tgt_from_etype_conns, nodes_with_in_relns =\
        get_connectivity_for(dis_nodes, kg, in_relations)

    pp_entity_entity_degrees("Disease", kg.get_entity_types(),
                             tgt_to_etype_conns, tgt_from_etype_conns)

    print()
    pp_underlined_hdg("Comparing Diseases with / without any `indication` or `off-label use`",
                      linechar="=", overline=True)

    dis_nodes_ind_or_off_label = nodes_with_in_relns[RELATION_INDICATION] | nodes_with_in_relns[RELATION_OFF_LABEL]

    dis_nodes_zi = set(dis_nodes) - dis_nodes_ind_or_off_label

    tgt_to_etype_conns, tgt_from_etype_conns, _ \
        = get_connectivity_for(list(dis_nodes_ind_or_off_label), kg)

    tgt_to_etype_conns_zi, tgt_from_etype_conns_zi, _ \
        = get_connectivity_for(list(dis_nodes_zi), kg)

    # pp_entity_in_out_degrees("Disease", n_dis, zi_n_out_counts, zi_n_in_counts)

    pp_cmp_entity_entity_degrees("Diseases With Indications/Off-label", len(dis_nodes_ind_or_off_label),
                                 tgt_to_etype_conns, tgt_from_etype_conns,
                                 "Diseases withOUT Indications/Off-label", len(dis_nodes_zi),
                                 tgt_to_etype_conns_zi, tgt_from_etype_conns_zi,
                                 kg.get_entity_types()
                                 )

    return


def pp_cmp_entity_entity_degrees(tgt_entity_type_1: str,
                                 n_tgt_ent_nodes_1: int,
                                 tgt_to_etype_conns_1, tgt_from_etype_conns_1,
                                 tgt_entity_type_2: str,
                                 n_tgt_ent_nodes_2: int,
                                 tgt_to_etype_conns_2, tgt_from_etype_conns_2,
                                 all_entity_types: List[str],
                                 ):

    hdg1 = "Comparing out/in edges to other Entity Types:"
    hdg2 = f"    '{tgt_entity_type_1}'  - v/s -  '{tgt_entity_type_2}'"
    print(hdg1)
    pp_underlined_hdg(hdg2)

    maxw = max(len(tgt_entity_type_1), len(tgt_entity_type_2))
    print(f"E1: nbr. {tgt_entity_type_1:{maxw}s} = {n_tgt_ent_nodes_1:6,d}")
    print(f"E2: nbr. {tgt_entity_type_2:{maxw}s} = {n_tgt_ent_nodes_2:6,d}")
    print()

    tot_ent_conns = np.sum(tgt_to_etype_conns_1, axis=0) + np.sum(tgt_from_etype_conns_1, axis=0) + \
                    np.sum(tgt_to_etype_conns_2, axis=0) + np.sum(tgt_from_etype_conns_2, axis=0)
    common_ent_types = [all_entity_types[i] for i in np.nonzero(tot_ent_conns)[0]]

    ptbl = PrettyTable(for_md=True)

    ptbl.set_colnames(["Node Type",
                       "cnt E1->", "med E1->", "avg E1->", "cnt ->E1", "med ->E1", "avg ->E1",
                       "cnt E2->", "med E2->", "avg E2->", "cnt ->E2", "med ->E2", "avg ->E2",
                       ],
                      ["s",
                       ",d", ".2f", ".2f", ",d", ".2f", ".2f",
                       ",d", ".2f", ".2f", ",d", ".2f", ".2f"
                       ])

    ent_type_idx = dict((t, i) for i, t in enumerate(all_entity_types))
    for etype in sorted(common_ent_types):
        e_idx = ent_type_idx[etype]

        tgt_to_e_1 = tgt_to_etype_conns_1[:, e_idx]
        e_to_tgt_1 = tgt_from_etype_conns_1[:, e_idx]

        tgt_to_e_2 = tgt_to_etype_conns_2[:, e_idx]
        e_to_tgt_2 = tgt_from_etype_conns_2[:, e_idx]

        out_1_sum = tgt_to_e_1.sum()
        in_1_sum = e_to_tgt_1.sum()
        out_2_sum = tgt_to_e_2.sum()
        in_2_sum = e_to_tgt_2.sum()

        row = ([etype] +
               (["-", "-", "-"] if out_1_sum == 0
                else [out_1_sum, np.median(tgt_to_e_1), tgt_to_e_1.mean()])
               +
               (["-", "-", "-"] if in_1_sum == 0
                else [in_1_sum, np.median(e_to_tgt_1), e_to_tgt_1.mean()])
               +
               (["-", "-", "-"] if out_2_sum == 0
                else [out_2_sum, np.median(tgt_to_e_2), tgt_to_e_2.mean()])
               +
               (["-", "-", "-"] if in_2_sum == 0
                else [in_2_sum, np.median(e_to_tgt_2), e_to_tgt_2.mean()])
               )

        ptbl.add_row(row)

    out_degrees_1 = np.sum(tgt_to_etype_conns_1, axis=1)
    in_degrees_1 = np.sum(tgt_from_etype_conns_1, axis=1)
    out_degrees_2 = np.sum(tgt_to_etype_conns_2, axis=1)
    in_degrees_2 = np.sum(tgt_from_etype_conns_2, axis=1)

    ptbl.add_row_("ALL",
                  out_degrees_1.sum(), np.median(out_degrees_1), out_degrees_1.mean(),
                  in_degrees_1.sum(), np.median(in_degrees_1), in_degrees_1.mean(),
                  out_degrees_2.sum(), np.median(out_degrees_2), out_degrees_2.mean(),
                  in_degrees_2.sum(), np.median(in_degrees_2), in_degrees_2.mean()
                  )
    print(ptbl)
    return


def disease_examples(count: int = 5,
                     max_degree: int = 20,
                     with_indication: bool = True,
                     neighbor_types: Union[str, List[str]] = None):
    """

    :param count:
    :param max_degree: Max connectivity of disease nodes.
        IF `neighbor_types` THEN connectivity measured only on the restricted types.
    :param with_indication: Should the diseases have an `indication`?
    :param neighbor_types: IF provided THEN ensures a min-degree representation of each Entity-Type.
        Example: "eff.3+, gen.1"
    """

    pp_funcargs(disease_examples)

    kg = load_primekg()
    print()

    min_degrees = None
    if isinstance(neighbor_types, str):
        neighbor_types, min_degrees = expand_abbr_entypes(neighbor_types, kg.get_entity_types())
    elif neighbor_types:
        min_degrees = [1] * len(neighbor_types)

    if min_degrees:
        print("Using neighbor_types and min_degrees =",
              ", ".join(f"{t}={d}" for t, d in zip(neighbor_types, min_degrees)))
        print()

    ndis = 0
    for dis_node in kg.get_nodes_for_types(DISEASE_ENTITY_TYPE):
        in_nghbrs, out_nghbrs = get_neighbors_for(dis_node, kg, uni_directional=True,
                                                  has_in_relations=["indication"] if with_indication else None,
                                                  neighbor_types=neighbor_types)
        all_nghbrs = in_nghbrs | out_nghbrs
        if not all_nghbrs:
            continue
        if 0 < max_degree < len(all_nghbrs):
            continue

        if min_degrees:
            nghbr_degrees = Counter(kg.get_entity_type(nd) for nd in all_nghbrs)
            min_degr_match = True
            for entype, min_degr in zip(neighbor_types, min_degrees):
                if nghbr_degrees[entype] < min_degr:
                    min_degr_match = False
                    break
            if not min_degr_match:
                continue

        ndis += 1
        pp_node_neighborhood(kg, dis_node, nbr=ndis, neighbor_types=neighbor_types)

        if 0 < count <= ndis:
            break

    print()
    return


def disease_drugs_report(indications_range: tuple[int, int] = (0, 0),
                         offlabel_range: tuple[int, int] = (1, 10),
                         contra_range: tuple[int, int] = (1, 10),
                         ):
    """
    Will print diseases and related drugs when counts are within the range.

    Ranges are (min, max].
    """

    pp_underlined_hdg("Diseases with constraints on counts of related Drugs")
    pp_funcargs(disease_drugs_report)

    reln_ranges = {
        RELATION_INDICATION: indications_range,
        RELATION_OFF_LABEL: offlabel_range,
        RELATION_CONTRA_INDICATION: contra_range,
    }

    kg = load_primekg()
    print()
    print("Total nbr Diseases in PrimeKG =", f"{len(kg.get_nodes_for_types(DISEASE_ENTITY_TYPE)):,d}")
    print()

    # ---
    def get_node_names(nd: str):
        all_names = kg.get_all_node_names(nd)
        txt = f"({nd}) {all_names[0]}"
        if len(all_names) > 1:
            txt += f"... aka: {', '.join(all_names[1:])}"
        return txt
    # ---

    ndis = 0
    for dis_node in kg.get_nodes_for_types(DISEASE_ENTITY_TYPE):
        _, drug_neighbors = kg.get_disease_neighbors(dis_node, neighbor_types=[DRUG_ENTITY_TYPE])
        ddreln_counts = {reln: len(drug_neighbors.get(reln, [])) for reln in DRUG_DISEASE_RELATIONS}

        qualified = True
        for reln, rng in reln_ranges.items():
            rcount = ddreln_counts[reln]
            if rng[0] == rng[1] and rcount == rng[0]:
                continue
            if rcount <= rng[0] or rcount > rng[1]:
                qualified = False
                break

        if not qualified:
            continue

        ndis += 1

        print(f"[{ndis}] {get_node_names(dis_node)}")
        print()
        for reln, cnt in ddreln_counts.items():
            print(f"    {reln:>16s}: {cnt:3d} drugs")
        print()

        for reln in DRUG_DISEASE_RELATIONS:
            pkg_drugs = drug_neighbors.get(reln, [])
            if not pkg_drugs:
                continue

            print(f"    --- {reln} ... {len(pkg_drugs)} drugs")
            print()
            for n, drug in enumerate(pkg_drugs, start=1):
                print(f"      {n:2d}:  {get_node_names(drug)}")
            print()

    print()
    print("Nbr qualifying diseases found =", ndis)
    print()

    return


def expand_abbr_entypes(abbr: str, all_entypes: List[str]) -> Tuple[List[str], List[int]]:
    """

    :param abbr: e.g. "eff.3+, gen.1"
    :param all_entypes:
    :return:
        - List of Entt-Type[str]
        - List of min-degree[int]
    """
    entypes = []
    min_degrees = []

    if not abbr:
        return entypes, min_degrees

    for xabbr in abbr.split(", "):
        abbr_parts = xabbr.split(".")

        if len(abbr_parts) > 1:
            degr = abbr_parts[1]
            if degr.endswith("+"):
                degr = degr[:-1]
            min_degrees.append(int(degr))
        else:
            min_degrees.append(1)

        xa = abbr_parts[0]
        for entype in all_entypes:
            if entype.startswith(xa):
                entypes.append(entype)
                break

    return entypes, min_degrees


# -----------------------------------------------------------------------------
#   Functions - neighborhood
# -----------------------------------------------------------------------------


def get_connectivity_for(nodes: Sequence[str],
                         kg: PrimeKG,
                         in_relations: List[str] = None,
                         uni_directional: bool = True):
    """

    :param nodes:
    :param kg:
    :param in_relations:
    :param uni_directional:
    :return:
         tgt_to_etype_conns: array with shape (n_tgts, n_ent_types)
         tgt_from_etype_conns: array with shape (n_tgts, n_ent_types)
                ... entity-types in same order as retd. by `kg.get_entity_types()`
        nodes_with_in_relns: Dict:{ Relation => Set[node] }
    """

    n_tgts = len(nodes)

    ent_type_idx = dict((t, i) for i, t in enumerate(kg.get_entity_types()))
    n_ent_types = len(ent_type_idx)

    tgt_to_etype_conns = np.zeros((n_tgts, n_ent_types), dtype=np.int32)
    tgt_from_etype_conns = np.zeros((n_tgts, n_ent_types), dtype=np.int32)

    if in_relations is None:
        in_relations = []
    nodes_with_in_relns = defaultdict(set)

    for ni, node in enumerate(nodes):

        for head, tail, rdata in kg.out_edges(node, data=True):
            if uni_directional and rdata["Relation"].startswith("rev-"):
                continue
            # noinspection PyTypeChecker
            tgt_to_etype_conns[ni, ent_type_idx[kg.nodes[tail]['EntityType']]] += 1

        for head, tail, rdata in kg.in_edges(node, data=True):
            if uni_directional and rdata["Relation"].startswith("rev-"):
                continue
            # noinspection PyTypeChecker
            tgt_from_etype_conns[ni, ent_type_idx[kg.nodes[head]['EntityType']]] += 1
            if rdata["Relation"] in in_relations:
                nodes_with_in_relns[rdata["Relation"]].add(node)

    return tgt_to_etype_conns, tgt_from_etype_conns, nodes_with_in_relns


def get_neighbors_for(node: str,
                      kg: PrimeKG,
                      has_in_relations: List[str] = None,
                      uni_directional: bool = True,
                      neighbor_types: List[str] = None) -> Tuple[Set[str], Set[str]]:
    """

    :param node:
    :param kg:
    :param has_in_relations: Node must have an in-edge for one of these relations
    :param uni_directional: Skip "rev-" relations
    :param neighbor_types: Return neighbors only in these entity-types.
    :return: Set[in-neighbor-nodes], Set[out-neighbor-nodes]    ... sets may be empty
         IF has_in_relations specified and `node` does not occur in any of these in-relations,
         THEN {}, {}
    """

    in_nghbrs = set()
    out_nghbrs = set()

    is_in_relations = has_in_relations is None

    # noinspection PyArgumentList
    for head, tail, reln in kg.in_edges(node, keys=True):
        if uni_directional and reln.startswith("rev-"):
            continue
        if has_in_relations and reln in has_in_relations:
            is_in_relations = True
        if neighbor_types and kg.nodes[head]['EntityType'] not in neighbor_types:
            continue
        in_nghbrs.add(head)

    # noinspection PyArgumentList
    for head, tail, reln in kg.out_edges(node, keys=True):
        if uni_directional and reln.startswith("rev-"):
            continue
        if neighbor_types and kg.nodes[tail]['EntityType'] not in neighbor_types:
            continue
        out_nghbrs.add(tail)

    if not is_in_relations:
        return set(), set()

    return in_nghbrs, out_nghbrs


def pp_drugs_for_disease(disease_node: str, pkg: PrimeKG = None):
    if pkg is None:
        pkg = load_primekg()

    _, drug_neighbors = pkg.get_disease_neighbors(disease_node)

    pp_underlined_hdg(f"Drugs for the {pkg.get_qualified_node_name(disease_node)}")

    for reln in DRUG_DISEASE_RELATIONS:
        drugs = drug_neighbors.get(reln, {})
        print(f"{reln} ... {len(drugs)} drugs:")
        print()
        for n, drug in enumerate(sorted(drugs), start=1):
            names = pkg.get_all_node_names(drug)
            print(f"  {reln}.{n}: ({drug}) {names[0]}")
            if len(names) > 1:
                print(f"\t\taka: {', '.join(names[1:])}")
            print()

    print()
    return


# ======================================================================================================
#   Main
# ======================================================================================================

# To run
# ------
# [Python]$ python -m drugmechcf.analyze.primekgstats {basic | connectivity | disease_examples}
#

if __name__ == "__main__":

    import argparse
    from datetime import datetime
    from drugmechcf.utils.misc import print_cmd

    _argparser = argparse.ArgumentParser(
        description='PrimeKG analysis',
    )

    _subparsers = _argparser.add_subparsers(dest='subcmd',
                                            title='Available commands',
                                            )
    # Make the sub-commands required
    _subparsers.required = True

    # ... basic
    _ = _subparsers.add_parser('basic',
                                             help="Some basic PrimeKG stats.")

    # ... connectivity
    _ = _subparsers.add_parser('connectivity',
                               help="Node connectivity stats in PrimeKG.")

    # ... disease_examples
    _sub_cmd_parser = _subparsers.add_parser('disease_examples',
                                             help="Neighborhood of diseases in PrimeKG.")
    _sub_cmd_parser.add_argument('-t', '--neighbor_types', type=str, default=None,
                                 help="Neighborhood connectivity pattern.")
    _sub_cmd_parser.add_argument('-m', '--max_degree', type=int, default=10,
                                 help="Max degree of disease nodes. Default is 10. Set to 0 for no constraint.")

    # ... disease_cmp
    _ = _subparsers.add_parser('disease_cmp',
                               help="Compare diseases with and without treatments.")

    # ... drugs_for_disease
    _sub_cmd_parser = _subparsers.add_parser('drugs_for_disease',
                                             help="Drugs for the disease.")
    _sub_cmd_parser.add_argument('disease_node', type=str,
                                 help="Node name of a disease in PrimeKG.")

    # ... disease_drugs_report
    _sub_cmd_parser = _subparsers.add_parser('disease_drugs_report',
                                             help="Report on Diseases with related Drugs.")

    # ...

    _args = _argparser.parse_args()
    # .................................................................................................

    start_time = datetime.now()

    print("---------------------------------------------------------------------")
    print_cmd()

    if _args.subcmd == 'basic':

        basic_stats()

    elif _args.subcmd == 'connectivity':

        connectivity()

    elif _args.subcmd == 'disease_examples':

        disease_examples(max_degree=_args.max_degree, neighbor_types=_args.neighbor_types)

    elif _args.subcmd == 'disease_cmp':

        compare_diseases_on_treatable()

    elif _args.subcmd == 'drugs_for_disease':

        pp_drugs_for_disease(_args.disease_node)

    elif _args.subcmd == 'disease_drugs_report':

        disease_drugs_report()

    else:

        raise NotImplementedError(f"Command not implemented: {_args.subcmd}")

    # /

    print('\nTotal Run time =', datetime.now() - start_time)
    print()
