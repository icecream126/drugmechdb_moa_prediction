"""
Mechanisms of Action, from DrugMechDB
and mapping to entries in PrimeKG
"""

from collections import Counter
from itertools import chain, groupby
from typing import List, Tuple

import networkx as nx
import numpy as np
import pandas as pd

from drugmechcf.data.primekg import PrimeKG, load_primekg
from drugmechcf.data.drugmechdb import DrugMechDB, load_drugmechdb, PROTEIN_ENTITY_TYPE, GENE_ENTITY_TYPE

from drugmechcf.utils.misc import pp_underlined_hdg, split_camel_case, reset_df_index, ppmd_counts_df
from drugmechcf.utils.prettytable import pp_seq_key_count, pp_counts


# -----------------------------------------------------------------------------
#   Functions
# -----------------------------------------------------------------------------


def moas_primekg_join_stats():
    drugmechdb = load_drugmechdb()
    primekg: PrimeKG = load_primekg()
    print()

    n_graphs = 0
    n_graphs_dis_in_primekg = 0
    n_graphs_dis_is_multiple_primekg_nodes = 0
    n_graphs_drug_in_primekg = 0
    n_graphs_drug_and_dis_in_primekg = 0

    drugmechdb_diseases_mappedto_primekg = set()

    for moa in drugmechdb.indication_graphs:
        n_graphs += 1

        dis_id = moa.graph["disease_mesh"]
        drug_id = moa.graph["drugbank"]

        if (dis_nodes := primekg.disease_mesh_to_nodes.get(dis_id)) is not None:
            drugmechdb_diseases_mappedto_primekg.update(dis_nodes)
            n_graphs_dis_in_primekg += 1
            if len(dis_nodes) > 1:
                n_graphs_dis_is_multiple_primekg_nodes += 1

        if drug_id and primekg.has_node(drugbank_id_to_drug_node(drug_id)):
            n_graphs_drug_in_primekg += 1
            if dis_nodes is not None:
                n_graphs_drug_and_dis_in_primekg += 1

    print()
    print(f"nbr MoA graphs = {n_graphs:,d}")
    print(f"nbr MoAs with Disease in PrimeKG = {n_graphs_dis_in_primekg:,d}")
    print(f"nbr MoAs with Disease has multiple PrimeKG diseases = {n_graphs_dis_is_multiple_primekg_nodes:,d}")
    print(f"nbr MoAs with Drug in PrimeKG = {n_graphs_drug_in_primekg:,d}")
    print(f"nbr MoAs with both Drug and Disease in PrimeKG = {n_graphs_drug_and_dis_in_primekg:,d}")
    print()
    print("nbr Diseases in PrimeKG that map into a Disease in DrugMechDB =",
          format(len(drugmechdb_diseases_mappedto_primekg), ",d"))
    print()

    # Neighborhood Patterns

    # ---
    def nghbr_patt_counts(g: nx.DiGraph, nd, in_edges=True, counts=None,
                          skip=None):
        """
        Returns Dict: {(neighbor-type[str], nbr-neighbors[str]) => Count[int]}
        ... where Count is nbr times this fn is called,
                  nbr-neighbors = "1", "2", "3+"
        """
        if counts is None:
            counts = Counter()

        if skip is None:
            skip = ["Disease", "disease"]

        edges = g.in_edges(nd) if in_edges else g.out_edges(nd)
        g_cnts = Counter(g.nodes[u if in_edges else v]["EntityType"] for u, v in edges)
        for ntype, cnt in g_cnts.items():
            if ntype in skip:
                continue
            key = (ntype, str(cnt)) if cnt < 2 else (ntype, "2+")
            counts[key] += 1

        return counts

    def nghbr_patt_counts_coll(g: nx.DiGraph, nd, in_edges=True, counts=None, skip=None):
        """
        Returns Dict: {Key[str] => Count[int]}
        ... where Count is nbr times this fn is called,
        """
        if counts is None:
            counts = Counter()

        if skip is None:
            skip = ["Disease", "disease"]

        edges = g.in_edges(nd) if in_edges else g.out_edges(nd)
        g_cnts = Counter(g.nodes[u if in_edges else v]["EntityType"] for u, v in edges)

        key = ", ".join([f"{entype_abbr(ntype)}.{cnt if cnt < 3 else '3+'}"
                         for ntype, cnt in sorted(g_cnts.items())
                         if ntype not in skip])
        if key:
            counts[key] += 1

        return counts
    # ---

    # Disease neighborhoods in DrugMechDB

    pp_underlined_hdg("Disease Neighborhoods in DrugMechDB", overline=True, linechar="=")

    print("Note: Neighboring Disease nodes are ignored in these counts.\n")

    dis_nbr_counts = None
    dis_nbr_counts_coll = None
    for moa in drugmechdb.indication_graphs:
        dis_id = moa.graph["disease_mesh"]
        dis_nbr_counts = nghbr_patt_counts(moa, dis_id, in_edges=True, counts=dis_nbr_counts)
        dis_nbr_counts_coll = nghbr_patt_counts_coll(moa, dis_id, in_edges=True,
                                                     counts=dis_nbr_counts_coll)

    pp_underlined_hdg("Disease neighbor type counts in DrugMechDB MoAs")

    # noinspection PyUnboundLocalVariable
    pp_counts(["Ent-Type", "nbr Neighbors", "nbr MoAs"], ["s", "s", ",d"],
              rows=sorted([(nty, nn, ndis) for (nty, nn), ndis in dis_nbr_counts.items()],
                          key=lambda x: (x[0], -x[2])),
              count_col_idx=2,
              total_count=n_graphs)

    print()
    pp_underlined_hdg("Disease neighborhood patterns in DrugMechDB MoAs")

    pp_seq_key_count(dis_nbr_counts_coll.most_common(),
                     col_headings=["Neighborhood", "nbr MoAs"],
                     add_total=True, add_index=True, total_count=drugmechdb.nbr_indications())

    # Disease neighborhoods in PrimeKG

    skip_ents = ["disease", "drug"]

    print()
    pp_underlined_hdg("Disease Neighborhoods in PrimeKG", overline=True, linechar="=")

    print("Note: Neighboring Disease nodes are ignored in these counts.\n")

    dis_nbr_counts = None
    dis_nbr_counts_coll = None
    for dis_id in drugmechdb_diseases_mappedto_primekg:
        dis_nbr_counts = nghbr_patt_counts(primekg, dis_id, in_edges=True, counts=dis_nbr_counts,
                                           skip=skip_ents)
        dis_nbr_counts_coll = nghbr_patt_counts_coll(primekg, dis_id, in_edges=True,
                                                     counts=dis_nbr_counts_coll, skip=skip_ents)

    print()
    pp_underlined_hdg("Disease in-neighbor type counts in PrimeKG")

    # noinspection PyUnboundLocalVariable
    pp_counts(["Ent-Type", "nbr Neighbors", "nbr Diseases"], ["s", "s", ",d"],
              rows=sorted([(nty, nn, ndis) for (nty, nn), ndis in dis_nbr_counts.items()],
                          key=lambda x: (x[0], -x[2])),
              count_col_idx=2,
              total_count=len(drugmechdb_diseases_mappedto_primekg))

    print()
    pp_underlined_hdg("Disease in-neighborhood patterns in PrimeKG")

    pp_seq_key_count(dis_nbr_counts_coll.most_common(),
                     col_headings=["Neighborhood", "nbr Diseases"],
                     add_total=True, add_index=True,
                     total_count=len(drugmechdb_diseases_mappedto_primekg))

    print()
    return


def moa_basic_stats():
    drugmechdb = load_drugmechdb()
    print()

    # Graph Drug and Disease end points

    drug_dis_endpoint_data = [((g.graph["drug_id"], g.nodes[g.graph["drug_id"]]["EntityType"]),
                               (g.graph["disease_mesh"], g.nodes[g.graph["disease_mesh"]]["EntityType"]))
                              for g in drugmechdb.indication_graphs]

    unique_drug_disease_endpoints = set((ep[0][0], ep[1][0]) for ep in drug_dis_endpoint_data)

    drug_disease_n_multiple_moas: List[int] = []
    drug_disease_non_uniq_endpts: List[Tuple[str, str]] = []
    for drug_id, dis_id in unique_drug_disease_endpoints:
        if (n_moas := len(drugmechdb.get_indication_graphs(drug_id, dis_id))) > 1:
            drug_disease_n_multiple_moas.append(n_moas)
            drug_disease_non_uniq_endpts.append((drug_id, dis_id))

    print(f"nbr Indication graphs         = {drugmechdb.nbr_indications():5,d}")
    print(f"nbr unique Drug end points    = {len(set(ep[0] for ep in unique_drug_disease_endpoints)):5,d}")
    print(f"nbr unique Disease end points = {len(set(ep[1] for ep in unique_drug_disease_endpoints)):5,d}")
    print(f"nbr unique Drug-Disease pairs = {len(unique_drug_disease_endpoints):5,d}")
    print()

    # Nbr occurrences for each n_moas value
    print(f"nbr Drug-Disease with multiple MoAs = {len(set(drug_disease_non_uniq_endpts)):5,d}")
    n_moas_uniq, n_moas_counts = np.unique(drug_disease_n_multiple_moas, return_counts=True)
    # Is it more than just 2 MoAs for each repeated (drug, disease)?
    print(f"    max nbr repeats = {n_moas_uniq[-1]}")
    if len(n_moas_uniq) > 1:
        print("    (n_moas, n_drug_disease_pairs) =",
              "; ".join(f"({x}, {y})" for x, y in zip(n_moas_uniq, n_moas_counts)))
    print()
    print()

    ntype_src_counts = Counter([(ep[0][1], ep[0][0].split(":")[0]) for ep in drug_dis_endpoint_data] +
                               [(ep[1][1], ep[1][0].split(":")[0]) for ep in drug_dis_endpoint_data])
    pp_underlined_hdg("Drug and Disease end-points in MoA graphs")
    pp_counts(["Node-Type", "Ontology", "nbr MoAs"], ["s", "s", ",d"],
              # rows=sorted([(a, b, cnt) for (a, b), cnt in ntype_src_counts.items()],
              #             key=lambda x: (x[0], -x[2], x[1])),
              rows=[(a, b, cnt) for (a, b), cnt in ntype_src_counts.items()],
              count_col_idx=2)

    print()

    # Node types

    pp_underlined_hdg("Entities Report", linechar="=", overline=True)
    print()

    all_nodes = list(chain.from_iterable(moa.nodes(data='EntityType') for moa in drugmechdb.indication_graphs))
    uniq_nodes = set(all_nodes)

    ntype_counts = Counter(ntype for nd, ntype in all_nodes)

    pp_underlined_hdg("Node type: nbr Node Occurrences")
    pp_counts(["Node-Type", "nbr Nodes"], ["s", ",d"],
              rows=ntype_counts.most_common(),
              count_col_idx=1)
    print()

    ntype_src_counts = Counter((ntype, nd.split(':')[0]) for nd, ntype in all_nodes)

    pp_underlined_hdg("Node type, source in DrugMechDB: nbr Node Occurrences")
    pp_counts(["Node-Type", "Ontology", "nbr Nodes"], ["s", "s", ",d"],
              rows=sorted([(a, b, cnt) for (a, b), cnt in ntype_src_counts.items()],
                          key=lambda x: (x[0], -x[2], x[1])),
              count_col_idx=2)
    print()

    ntype_src_counts = Counter((ntype, nd.split(':')[0]) for nd, ntype in uniq_nodes)

    pp_underlined_hdg("Node type, source in DrugMechDB: Unique nodes")
    pp_counts(["Node-Type", "Ontology", "nbr Unique Nodes"], ["s", "s", ",d"],
              rows=sorted([(a, b, cnt) for (a, b), cnt in ntype_src_counts.items()],
                          key=lambda x: (x[0], -x[2], x[1])),
              count_col_idx=2)
    print()

    # Relations

    pp_underlined_hdg("Relations Report", linechar="=", overline=True)
    print()

    counts_df = drugmechdb.get_edge_counts()

    pp_underlined_hdg("Named Relations in DrugMechDB, sorted on decreasing nbr Edges")
    counts = counts_df.groupby('relation', as_index=False).agg(n_edges=pd.NamedAgg(column='count', aggfunc='sum'))
    ppmd_counts_df(counts, counts_col='n_edges', add_cum_pct_total=True, floatfmt=".2%")
    print()

    pp_underlined_hdg("Relations and head/tail Entity types in DrugMechDB, sorted on decreasing nbr Edges")
    ppmd_counts_df(counts_df, counts_col='count', add_cum_pct_total=True)
    print()

    pp_underlined_hdg("Relations and head/tail Entity types, by nbr Unique Edges")
    counts = drugmechdb.get_edge_counts(unique_edges=True)
    ppmd_counts_df(counts, counts_col="count", add_cum_pct_total=True)
    print()

    pp_underlined_hdg("Relations and head/tail Entity types in DrugMechDB, sorted on Head-type, decreasing nbr Edges")
    counts = reset_df_index(counts_df.sort_values(["htype", "count"], ascending=[True, False]))
    ppmd_counts_df(counts, counts_col="count", add_pct_total=True)
    print()

    pp_underlined_hdg("Relations and head/tail Entity types in DrugMechDB, sorted on Head and Tail types, " +
                      "decreasing nbr Edges")
    counts = reset_df_index(counts_df.sort_values(["htype", "ttype", "count"], ascending=[True, True, False]))
    ppmd_counts_df(counts, counts_col="count", add_pct_total=True)
    print()

    # Drug neighbors
    pp_underlined_hdg("Drug out-Neighbors Report", linechar="=", overline=True)
    print()

    nghbr_types = Counter(moa.nodes[tnd]["EntityType"] for moa in drugmechdb.indication_graphs
                          for hnd, tnd in moa.out_edges(moa.graph['drug_id']))
    pp_underlined_hdg("Node Freq of out-neighbor node types to Drug in MoAs")
    pp_seq_key_count(nghbr_types.most_common(), col_headings=["Node-Type", "nbr Nodes"],
                     add_total=True, add_index=True)
    print()

    nghbr_types = Counter(chain.from_iterable(set(moa.nodes[tnd]["EntityType"]
                                                  for hnd, tnd in moa.out_edges(moa.graph['drug_id']))
                                              for moa in drugmechdb.indication_graphs))
    pp_underlined_hdg("MoA Freq of out-neighbor node types to Drug in MoAs")
    pp_seq_key_count(nghbr_types.most_common(), col_headings=["Node-Type", "nbr MoAs"],
                     total_count=drugmechdb.nbr_indications(), add_total=True, add_index=True)
    print()

    # Disease neighbors
    pp_underlined_hdg("Disease in-Neighbors Report", linechar="=", overline=True)
    print()

    nghbr_types = Counter(moa.nodes[hnd]["EntityType"] for moa in drugmechdb.indication_graphs
                          for hnd, tnd in moa.in_edges(moa.graph['disease_mesh']))
    pp_underlined_hdg("Node Freq of in-neighbor node types to Disease in MoAs")
    pp_seq_key_count(nghbr_types.most_common(), col_headings=["Node-Type", "nbr Nodes"],
                     add_total=True, add_index=True)
    print()

    nghbr_types = Counter(chain.from_iterable(set(moa.nodes[hnd]["EntityType"]
                                                  for hnd, tnd in moa.in_edges(moa.graph['disease_mesh']))
                                              for moa in drugmechdb.indication_graphs))
    pp_underlined_hdg("MoA Freq of in-neighbor node types to Disease in MoAs")
    pp_seq_key_count(nghbr_types.most_common(), col_headings=["Node-Type", "nbr MoAs"],
                     total_count=drugmechdb.nbr_indications(), add_total=True, add_index=True)
    print()

    return


def disease_neighborhood_patterns():
    """
    Report on some specific patterns in disease neighborhood.
        nbr MoAs where dis has spl nghbr
        nbr MoA where spl nghbrs are the only nghbrs
        freq of nghbr types in those MoAs
        freq of nghbr-types of spl nghbrs
    """
    drugmechdb = load_drugmechdb()
    print()

    # Get MoAs where disease has a neighbor of a 'special' type

    dis_spl_nghbr_types = ["GrossAnatomicalStructure"]

    pp_underlined_hdg(f"Report on MoA's where Disease has neighbors of special types", linechar="=")
    print("Special types:", ", ".join(dis_spl_nghbr_types))
    print()

    nghbr_counts = {"Disease-special-neighbor-type": Counter(),
                    "Disease-other-neighbor-type": Counter(),
                    "Neighbor-of-Neighbor-type": Counter(),
                    }

    moas = []
    for moa in drugmechdb.indication_graphs:
        dis_neighbors = list(moa.predecessors(moa.graph['disease_mesh']))
        spl_nghbrs = [nd for nd in dis_neighbors if drugmechdb.get_entity_type(moa, nd) in dis_spl_nghbr_types]
        if not spl_nghbrs:
            continue

        moas.append(moa)

        spl_nghbr_types = set(drugmechdb.get_entity_type(moa, nd) for nd in spl_nghbrs)
        other_nghbr_types = set(drugmechdb.get_entity_type(moa, nd) for nd in dis_neighbors) - spl_nghbr_types

        nghbr_counts["Disease-special-neighbor-type"].update(spl_nghbr_types)
        nghbr_counts["Disease-other-neighbor-type"].update(other_nghbr_types)

        nghbr_nghbr_types = set(drugmechdb.get_entity_type(moa, nn)
                                for n in spl_nghbrs
                                for nn in moa.predecessors(n))
        nghbr_counts["Neighbor-of-Neighbor-type"].update(nghbr_nghbr_types)

    n_moas = len(moas)

    for k, counts in nghbr_counts.items():
        pp_underlined_hdg(k + "s")
        pp_counts(["Entity-type", "# MoAs"], ["s", ",d"],
                  rows=counts.most_common(), count_col_idx=1,
                  total_count=n_moas, total_count_name="Total nbr MoAs")
        print()

    return


def drug_disease_report():
    """
    Combined v/s individual neighborhoods of Drug, Disease entities.
    Drug-treats-Disease stats.
    """
    drugmechdb = load_drugmechdb()
    print()

    pp_underlined_hdg("Report on Drugs and Diseases in DrugMechDB", linechar="=", overline=True)
    print()

    drug_df = drugmechdb.get_all_drug_connections()
    n_drugs = drug_df.DrugId.value_counts().shape[0]
    n_diseases = drug_df.DiseaseId.value_counts().shape[0]

    print(f"Nbr MoA's in DrugMechDB       = {drugmechdb.nbr_indications():5,d}")
    print(f"Nbr unique Drug-Disease pairs = {drug_df[['DrugId', 'DiseaseId']].value_counts().shape[0]:5,d}")
    print(f"Nbr unique Drugs              = {n_drugs:5,d}")
    print(f"Nbr unique Diseases           = {n_diseases:5,d}")
    print()
    print()

    # --- Nbr MoAs per Drug-Disease pair ---

    # Counts per (drug, disease)
    drug_dis_counts = Counter(drugmechdb.get_moa_drug_disease_nodes(moa) for moa in drugmechdb.indication_graphs)
    # meta-counts
    n_moa_counts = Counter(drug_dis_counts.values())

    pp_underlined_hdg("Nbr MoAs per Drug-Disease pair")
    pp_seq_key_count(sorted(n_moa_counts.items()),
                     col_headings=["nbr-MoAs", "nbr-Drug-Disease"], col_types=["d", ",d"],
                     add_total=True)
    print()

    pp_underlined_hdg("Drug-Disease pairs with most MoAs")
    n_most_common = sum(n_dd for n_moa, n_dd in n_moa_counts.items() if n_moa >= 4)
    df = pd.DataFrame.from_records([(drug_id, drugmechdb.get_drug_name(drug_id),
                                     disease_id, drugmechdb.get_disease_name(disease_id),
                                     count)
                                    for (drug_id, disease_id), count in drug_dis_counts.most_common(n_most_common)],
                                   columns=["Drug-ID", "DrugName", "Disease-ID", "DiseaseName", "nbr-MoAs"])
    ppmd_counts_df(reset_df_index(df))
    print()
    print()

    # --- Neighbors ---

    pp_underlined_hdg("Drugs: nbr Diseases Treated")

    drug_dis_df = drug_df[['DrugId', 'DiseaseId']].drop_duplicates()
    pp_counts(["nbr treated Diseases", "nbr Drugs"], [",d", ",d"],
              rows=sorted(Counter(drug_dis_df.DrugId.value_counts().values).items()),
              count_col_idx=1,
              pct_col_hdg=None
              )
    print()

    pp_underlined_hdg("Drugs with most Diseases Treated")

    series = drug_df[['DrugId', 'DrugName', 'DiseaseId']].drop_duplicates()[['DrugId', 'DrugName']].value_counts()[:10]
    df = series.to_frame(name="n_Diseases").reset_index()
    ppmd_counts_df(reset_df_index(df))
    print()

    pp_underlined_hdg("Drugs: direct Neighbors, combined over all MoA's")

    drug_nghbrs_df = drug_df[['DrugId', 'NeighborType', 'NeighborId']].drop_duplicates()
    drug_nghbrs_freq = drug_nghbrs_df.NeighborType.value_counts() / n_drugs
    pp_seq_key_count(drug_nghbrs_freq.items(), ["Neighbor Type", "Freq."], ["s", ".3f"],
                     add_index=True, add_total=True)
    print()
    print()

    disease_df = drugmechdb.get_all_disease_connections()

    pp_underlined_hdg("Diseases: nbr Treating Drugs")

    dis_drug_df = disease_df[['DiseaseId', 'DrugId']].drop_duplicates()
    pp_counts(["nbr treating Drugs", "nbr Diseases"], [",d", ",d"],
              rows=sorted(Counter(dis_drug_df.DiseaseId.value_counts().values).items()),
              count_col_idx=1,
              pct_col_hdg=None
              )
    print()

    pp_underlined_hdg("Diseases with most Drugs Treating them")

    series = (drug_df[['DrugId', 'DiseaseId', 'DiseaseName']].drop_duplicates()[['DiseaseId', 'DiseaseName']]
              .value_counts()[:10])
    df = series.to_frame(name="n_Drugs").reset_index()
    ppmd_counts_df(reset_df_index(df))
    print()

    pp_underlined_hdg("Diseases: direct Neighbors, combined over all MoA's")

    disease_nghbrs_df = disease_df[['DiseaseId', 'NeighborType', 'NeighborId']].drop_duplicates()
    disease_nghbrs_freq = disease_nghbrs_df.NeighborType.value_counts() / n_diseases
    pp_seq_key_count(disease_nghbrs_freq.items(), ["Neighbor Type", "Freq."], ["s", ".3f"],
                     add_index=True, add_total=True)
    print()

    return


def gene_protein_neighborhoods():
    drugmechdb = load_drugmechdb()
    print()

    pp_underlined_hdg("Report on Gene / Protein interactions in DrugMechDB", linechar="~", overline=True)
    print()

    pp_neighborhoods(drugmechdb, PROTEIN_ENTITY_TYPE)
    pp_neighborhoods(drugmechdb, GENE_ENTITY_TYPE)

    return


def pp_neighborhoods(drugmechdb: DrugMechDB, anchor_type: str):

    pp_underlined_hdg(f"Neighbors of {anchor_type}")

    n_anchor_nodes = len([anchor_node
                          for moa in drugmechdb.indication_graphs
                          for anchor_node in moa.get_nodes_for_type(anchor_type)])

    # ("MoaIndex", "Node") is unique reference to each node occurrence across MoAs

    # noinspection PyArgumentList
    data = [(midx, v, k, moa.get_node_entity_type(u))
            for midx, moa in enumerate(drugmechdb.indication_graphs)
            for anchor_node in moa.get_nodes_for_type(anchor_type)
            for u, v, k in moa.in_edges(anchor_node, keys=True)]
    df_in = pd.DataFrame.from_records(data, columns=["MoaIndex", "Node", "Relation", "EntityType"])

    # noinspection PyArgumentList
    data = [(midx, u, k, moa.get_node_entity_type(v))
            for midx, moa in enumerate(drugmechdb.indication_graphs)
            for anchor_node in moa.get_nodes_for_type(anchor_type)
            for u, v, k in moa.in_edges(anchor_node, keys=True)]
    df_out = pd.DataFrame.from_records(data, columns=["MoaIndex", "Node", "Relation", "EntityType"])

    cols = ["Relation", "EntityType"]
    df_in_summary = summarize_edge_counts(df_in, n_anchor_nodes, cols)
    df_out_summary = summarize_edge_counts(df_out, n_anchor_nodes, cols)

    df_joined = df_in_summary.merge(df_out_summary, how='outer', on=cols, suffixes=('_in', '_out'))

    sort_cols = ["total_in", "total_out"]
    df_joined.sort_values(by=sort_cols, ascending=False, inplace=True)

    # Have to specify format for each col, as join can produce NaN values, converting integer data to float.
    floatfmt = [""] + ([""] * len(cols)) + ([",.0f", ".2f", ".0f", ",.0f"] * 2)

    # Replace 'nan' values (from np.NaN or pd.NA) with empty string
    print(reset_df_index(df_joined).to_markdown(intfmt=',d', floatfmt=floatfmt)
          .replace(" nan ", "     "))
    print("\n")

    return


def summarize_edge_counts(df: pd.DataFrame, n_anchor_nodes: int, cols=None):
    """
    Computes per-Node summary stats.
    """

    df_per_node_counts = df.value_counts(subset=None).reset_index()

    # Nbr occurrences of unique `cols` values
    df_nuniq = df_per_node_counts.value_counts(subset=cols).reset_index()

    count_cols = ([cols] if isinstance(cols, str) else cols) + ["count"]
    df_counts = df_per_node_counts[count_cols]

    # Add one dummy zero-counts row for each node missing key. This will make the min value be Zero where needed.
    data = []
    for row in df_nuniq.itertuples():
        key = list(row[1:-1])     # skip index (col 0) and `count` (last col)
        count = row[-1]
        if count < n_anchor_nodes:
            key.append(0)
            data.append(key)

    df_zeros = pd.DataFrame.from_records(data, columns=count_cols)
    df_counts = pd.concat([df_counts, df_zeros], ignore_index=True)

    df_summary = df_counts.groupby(cols, as_index=False) \
        .agg(total=pd.NamedAgg(column='count', aggfunc='sum'),
             # `mean` is per Node-occurrence average
             mean=pd.NamedAgg(column='count', aggfunc=lambda x: np.sum(x) / n_anchor_nodes),
             min=pd.NamedAgg(column='count', aggfunc='min'),
             max=pd.NamedAgg(column='count', aggfunc='max')
             )

    return df_summary


# -----------------------------------------------------------------------------
#   Functions - Misc
# -----------------------------------------------------------------------------


def drug_node_to_drugbank_id(primekg_node: str):
    """

    :param primekg_node: e.g. 'DrugBank:DB09130'
    :return: e.g. 'DB:DB09130'
    """
    assert primekg_node.startswith("DrugBank")
    return "DB" + primekg_node[primekg_node.index(":"):]


def drugbank_id_to_drug_node(drugbank_id: str):
    """

    :param drugbank_id: e.g. 'DB:DB09130'
    :return: e.g. 'DrugBank:DB09130'
    """
    assert drugbank_id.startswith("DB")
    return "DrugBank" + drugbank_id[drugbank_id.index(":"):]


def entype_abbr(entype: str, ww=3):
    words = split_camel_case(entype)
    abbr = ''.join(w[:ww] for w in words)
    return abbr


def moa_path_schemata(top_n: int = 25):
    """
    Schema: Entity-Type, Entity-Type, ...
    """
    dmdb = load_drugmechdb()

    moa_schema = Counter()
    for moa in dmdb.get_indication_graphs():
        source, target = dmdb.get_moa_drug_disease_nodes(moa)
        moa_schema.update([tuple(moa.get_node_entity_type(nd) for nd in path)
                           for path in nx.all_simple_paths(moa, source=source, target=target)])

    n_paths = moa_schema.total()

    pp_underlined_hdg("Report on MoA Path Schemata", linechar='=', overline=True)

    print(f"Total nbr indication graphs = {dmdb.nbr_indications():,d}")
    print(f"Total nbr of MoA Drug-Disease paths = {n_paths:,d}")
    print(f"Nbr unique path schemata = {len(moa_schema):,d}")
    print("\n")

    # --- EntityType Transition Frequencies ---

    pp_underlined_hdg("EntityType Transition Frequencies")

    type_transitions = Counter()
    for idx, relation, htype, ttype, count in dmdb.get_edge_counts(unique_edges=True).itertuples():
        type_transitions[(htype, ttype)] += count

    type_transition_totals = {k: sum(v[1] for v in g)
                              for k, g in groupby(sorted(type_transitions.items()), key=lambda x: x[0][0])}

    n_edges = sum(type_transition_totals.values())

    print("Notes: Head-Tail prob-Edge is Prob(Tail | Head).")
    print("       Head-ANY  prob-Edge is Prob(Head).")
    print("       Counts are for unique edges across all MoA's in DrugMechDB.")
    print(f"       Total nbr unique edges = {n_edges:,d}")
    print("\n")

    df_rows = []
    for k, g in groupby(sorted(type_transitions.items()), key=lambda x: x[0][0]):
        k_total = type_transition_totals[k]
        df_rows.extend([[pair[0], pair[1], count, count / k_total]
                        for pair, count in sorted(g, key=lambda x: - x[-1])])
        df_rows.append([k, 'ANY', k_total, k_total / n_edges])
        df_rows.append([])

    df_rows.append(['ANY', 'ANY', n_edges, 1.0])

    df = pd.DataFrame.from_records(df_rows, columns=["Head-type", "Tail-type", "n-Edges", "prob-Edge"])

    ppmd_counts_df(reset_df_index(df), floatfmt=['', '', '', ',.0f', '.1%'])
    print()

    # --- Most common schemata ---

    df = reset_df_index(pd.DataFrame.from_records([(', '.join(s), n) for s, n in moa_schema.most_common(top_n)],
                                                  columns=["Schema", "Count"]))
    df['pctPaths'] = df.Count / n_paths
    df['cum_pctPaths'] = df['pctPaths'].cumsum()

    pp_underlined_hdg("Most common schemata")

    ppmd_counts_df(df)
    print()

    ent_lists = ["Drug,Protein,Disease".split(","),
                 "Drug,Protein,BiologicalProcess,Disease".split(","),
                 "Drug,Protein,PhenotypicFeature,Disease".split(","),
                 "Drug,Protein,BiologicalProcess,PhenotypicFeature,Disease".split(",")
                 ]
    for entts in ent_lists:
        pp_schemata_constrained(moa_schema, entts, top_n=5)

    print()

    # --- Selected Interactions ---

    entts = "Drug, Protein, Protein, BiologicalProcess, BiologicalProcess, PhenotypicFeature, Disease".split(", ")
    ent_pairs = (list(zip(entts, entts[1:]))
                 + [("BiologicalProcess", "Disease"), ("Protein", "Disease"), ("Protein", "PhenotypicFeature")])
    pp_interaction_relns(dmdb, ent_pairs)

    print()
    return


def pp_schemata_constrained(moa_schema: Counter, entts: List[str], top_n: int = 5):
    # Restrict schema to entts, and len > 2
    constr_schema = Counter(dict((s, n) for s, n in moa_schema.items() if all(e in entts for e in s) and len(s) > 2))
    n_paths = constr_schema.total()
    tot_n_paths = moa_schema.total()

    pp_underlined_hdg("MoA Schemata Constrained to Entities")
    print("Entities:", ", ".join(entts))
    print("Number of schemata =", len(constr_schema))
    print(f"Total nbr paths in DrugMechDB = {n_paths:,d} ... {n_paths/tot_n_paths:.1%}")

    cs_schema_lens = [len(s) for s, n in constr_schema.items()]
    print(f"Schema lengths = [{min(cs_schema_lens), max(cs_schema_lens)}],  median={np.median(cs_schema_lens):.1f}")
    print()

    df = reset_df_index(pd.DataFrame.from_records([(', '.join(s), n) for s, n in constr_schema.most_common(top_n)],
                                                  columns=["Schema", "Count"]))
    df['pctPaths'] = df.Count / tot_n_paths
    df['cum_pctPaths'] = df['pctPaths'].cumsum()

    ppmd_counts_df(df)
    print()

    return


def pp_interaction_relns(dmdb: DrugMechDB, ent_pairs: List[Tuple[str, str]]):

    pp_underlined_hdg("Report on Interaction Relations", linechar='=', overline=True)

    for u, v in sorted(ent_pairs):
        pp_underlined_hdg(f"Interactions between: {u}, {v}")
        ppmd_counts_df(dmdb.get_edge_counts(head_type=u, tail_type=v),
                       counts_col="count", sort_on_counts=True, add_cum_pct_total=True)
        print()

    return


# ======================================================================================================
#   Main
# ======================================================================================================

# To run
# ------
# [Python]$ python -m drugmechcf.analyze.drugmechdb {basic | basic_join | drug_disease | ...}
#

if __name__ == "__main__":

    import argparse
    from datetime import datetime
    from drugmechcf.utils.misc import print_cmd

    _argparser = argparse.ArgumentParser(
        description='MoA and PrimeKG analysis',
    )

    _subparsers = _argparser.add_subparsers(dest='subcmd',
                                            title='Available commands',
                                            )
    # Make the sub-commands required
    _subparsers.required = True

    # ... basic_join
    _ = _subparsers.add_parser('basic_join',
                               help="Some basic MoA and PrimeKG stats.")

    # ... basic
    _ = _subparsers.add_parser('basic',
                               help="Some basic DrugMechDB stats.")

    # ... disease_neighborhood
    _ = _subparsers.add_parser('disease_neighborhood',
                               help="Stats on MoAs when Disease has special neighbors.")

    # ... drug_disease
    _ = _subparsers.add_parser('drug_disease',
                               help="Detailed Stats on Drugs and Diseases.")

    # ... gene_prot
    _ = _subparsers.add_parser('gene_prot',
                               help="Detailed Stats on neighborhood of Genes and Proteins.")

    # ... schemata
    _sub_cmd_parser = _subparsers.add_parser('schemata',
                                             help="Stats on Drug-Disease path schemata.")
    _sub_cmd_parser.add_argument('top_n', nargs="?", type=int, default=25,
                                 help="Nbr of most common schemata to display.")

    # ...

    _args = _argparser.parse_args()
    # .................................................................................................

    start_time = datetime.now()

    print("---------------------------------------------------------------------")
    print_cmd()

    if _args.subcmd == 'basic_join':

        moas_primekg_join_stats()

    elif _args.subcmd == 'basic':

        moa_basic_stats()

    elif _args.subcmd == 'disease_neighborhood':

        disease_neighborhood_patterns()

    elif _args.subcmd == 'drug_disease':

        drug_disease_report()

    elif _args.subcmd == 'gene_prot':

        gene_protein_neighborhoods()

    elif _args.subcmd == 'schemata':

        moa_path_schemata(_args.top_n)

    else:

        raise NotImplementedError(f"Command not implemented: {_args.subcmd}")

    # /

    print('\nTotal Run time =', datetime.now() - start_time)
    print()
