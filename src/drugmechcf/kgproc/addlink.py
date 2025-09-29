"""
Induce new MoA by adding a new link between 2 entities.
"""

from typing import Callable, List, Tuple
import random

import networkx as nx

from drugmechcf.data.drugmechdb import (PROTEIN_ENTITY_TYPE, CHEMICAL_ENTITY_TYPE, DRUG_ENTITY_TYPE,
                                        DrugMechDB, load_drugmechdb)
from drugmechcf.data.moagraph import MoaGraph
from drugmechcf.kgproc.simulate import QualitatveSimulator, DEFAULT_RELN_CAUSAL_MECHANISMS, get_valid_simulatable_moas
from drugmechcf.utils.misc import pp_funcargs, pp_underlined_hdg


# -----------------------------------------------------------------------------
#   Globals
# -----------------------------------------------------------------------------

# Dict: Relation => Inverse-relation. Target-in-reln must be one of these keys.
REQD_RELN_INVERSE = {"decreases activity of": "increases activity of",
                     "increases activity of": "decreases activity of",
                     "negatively regulates": "positively regulates",
                     "positively regulates": "negatively regulates",
                     "decreases abundance of": "increases abundance of",
                     "increases abundance of": "decreases abundance of",
                     }


# -----------------------------------------------------------------------------
#   Functions
# -----------------------------------------------------------------------------


def select_target_moas(drugmechdb: DrugMechDB) -> List[Tuple[MoaGraph, str, str]]:
    """
    Candidate target-MoA must have:
        - Protein-node
        - Edge (U, reln, Protein) where
            U = Drug, Protein or ChemicalSubstance
            reln = one of:
                    decreases activity of, -ive
                    increases activity of, +ive
                    negatively regulates, -ive
                    positively regulates, +ive
                    decreases abundance of, -ive
                    increases abundance of, +ive

    :param drugmechdb: DrugMechDB containing candidate target-MoA's, e.g. reduced-DrugMechDB
    :returns: target-MoA-data-seq: [(moa, protein-node, in-reln), ...]

        For reduced-dmdb retd by `get_valid_simulatable_moas()`:
            len( target-MoA-data-seq )  = 5,203
            nbr unique MoAs             = 2,474
    """

    target_moas = []
    for gidx in drugmechdb.nodetype_gidxs[PROTEIN_ENTITY_TYPE]:
        moa = drugmechdb.indication_graphs[gidx]

        target_moas.extend(select_valid_targets_in_moa(moa))

    return target_moas


def select_valid_targets_in_moa(target_moa: MoaGraph) -> List[Tuple[MoaGraph, str, str]]:
    """
    Select valid target Protein nodes in target_moa.
    :return:  [(target_moa, protein-node, in-reln), ...]
    """

    reqd_u_entity_types = [DRUG_ENTITY_TYPE, PROTEIN_ENTITY_TYPE, CHEMICAL_ENTITY_TYPE]

    target_data_seq = []

    for protein_node in target_moa.get_nodes_for_type(PROTEIN_ENTITY_TYPE):
        valid_in_relns = set()
        # noinspection PyArgumentList
        for u, _, reln in target_moa.in_edges(protein_node, keys=True):
            if target_moa.get_node_entity_type(u) in reqd_u_entity_types and reln in REQD_RELN_INVERSE:
                valid_in_relns.add(reln)

        for reln in valid_in_relns:
            target_data_seq.append((target_moa, protein_node, reln))

    return target_data_seq


def select_source_moas(reduced_drugmechdb: DrugMechDB,
                       target_moa_data: Tuple[MoaGraph, str, str],
                       has_indication: Callable[[str, str], bool] = None
                       ) -> List[MoaGraph]:
    """
    Constraints on Source-MoA:
        - Drug cannot be a node (interior or source) in target-MoA
        - Disease must be different from target-MoA
        - Must not contain `target-protein-node`
        - if `has_indication` provided, then Drug, Disease pair does not have a known indication relation.

    :param reduced_drugmechdb: reduced DrugMechDB containing only valid candidates
    :param target_moa_data: (target-moa, target-protein-node, target-in-reln), as returned by `select_target_moas()`.
    :param has_indication: A function which returns whether a (drug, disesase) pair has a known indication relation.

    :return: [source_moa, ...]

        For target-moa-data retd by `select_target_moas()`:
            len( source-moa-seq ) is in [2865,3456], with mean-length = 3216
    """

    target_moa, target_protein, _ = target_moa_data
    target_disease = reduced_drugmechdb.get_moa_target_disease_node(target_moa)

    source_moas = []
    for moa in reduced_drugmechdb.indication_graphs:
        if moa is target_moa:
            continue

        if (reduced_drugmechdb.get_moa_source_drug_node(moa) in target_moa or
                reduced_drugmechdb.get_moa_target_disease_node(moa) == target_disease):
            continue

        if target_protein in moa.get_nodes_for_type(PROTEIN_ENTITY_TYPE):
            continue

        if has_indication is not None:
            if has_indication(reduced_drugmechdb.get_moa_source_drug_node(moa), target_disease):
                continue

        source_moas.append(moa)

    return source_moas


def add_link(drugmechdb: DrugMechDB,
             source_moa: MoaGraph,
             source_node: str,
             target_moa_data: Tuple[MoaGraph, str, str]
             ) -> Tuple[MoaGraph, str]:
    """
    Create a new MoA by adding a link from `source_node` to `target_protein`.

    :param drugmechdb:
    :param source_moa:
    :param source_node:
    :param target_moa_data: target_moa, target_protein, target_in_reln

    :return:
        - new MoaGraph
        - the relation for the new link that was added
    """

    target_moa, target_protein, target_in_reln = target_moa_data

    source_moa_drug_node = drugmechdb.get_moa_source_drug_node(source_moa)
    target_moa_disease_node = drugmechdb.get_moa_target_disease_node(target_moa)

    # Simulate target-moa to get desired level at target-protein
    qsim = QualitatveSimulator(target_moa)
    qsim.simulate()
    target_protein_level = qsim.get_node_level(target_protein, clamp=True)

    assert target_protein_level is not None and target_protein_level != 0, \
        (f"MoA {drugmechdb.get_moa_id(target_moa)}, {target_protein}: " +
         f"Target protein level = {target_protein_level} is invalid!")

    # Determine level at source-node
    if source_node == source_moa_drug_node:
        source_node_level = 1
    else:
        qsim = QualitatveSimulator(source_moa)
        qsim.simulate()
        source_node_level = qsim.get_node_level(source_node, clamp=True)

    assert source_node_level is not None and source_node_level != 0, \
        (f"MoA {drugmechdb.get_moa_id(source_moa)}, {source_moa.get_qualified_node_name(source_node)}: " +
         f"Source node level = {source_node_level} is invalid!")

    # The value of this must be +1 or -1
    #   since source_node_level, target_protein_levellevels in {-1, +1}
    #   as levels = 0 or None have been removed by `assert`s above
    new_reln_change_direction = source_node_level * target_protein_level

    # Decide whether to use same reln as `target_in_reln`, or its inverse
    if new_reln_change_direction == DEFAULT_RELN_CAUSAL_MECHANISMS[target_in_reln]:
        new_reln = target_in_reln
    else:
        new_reln = REQD_RELN_INVERSE[target_in_reln]

    # -- Create new MoA by adding new edge (source_node, new_reln, target_protein) --

    new_moa = create_new_moa_add_link(drugmechdb,
                                      source_moa, source_node,
                                      new_reln,
                                      target_moa, target_protein,
                                      source_moa_drug_node=source_moa_drug_node,
                                      target_moa_disease_node=target_moa_disease_node,
                                      )

    return new_moa, new_reln


def create_new_moa_add_link(drugmechdb: DrugMechDB,
                            source_moa: MoaGraph,
                            source_node: str,
                            new_reln: str,
                            target_moa: MoaGraph,
                            target_protein: str,
                            source_moa_drug_node: str = None,
                            target_moa_disease_node: str = None,
                            ) -> MoaGraph:
    """
    Synthesizes a new MoaGraph for a MoA from source_moa_drug_node to target_moa_disease_node
    by adding an interaction edge (source_moa.source_node, new_reln, target_moa.target_protein).

    :return: new MoaGraph
    """

    if source_moa_drug_node is None:
        source_moa_drug_node = drugmechdb.get_moa_source_drug_node(source_moa)

    if target_moa_disease_node is None:
        target_moa_disease_node = drugmechdb.get_moa_target_disease_node(target_moa)

    # Copy Disease attrs from target, Drug attrs from source
    # noinspection PyUnresolvedReferences
    new_moa_attr = dict(disease=target_moa.graph["disease"],
                        disease_mesh=target_moa.graph["disease_mesh"],
                        drug=source_moa.graph["drug"],
                        drug_id=source_moa.graph["drug_id"],
                        # Give it an ID
                        _id=f'{source_moa.graph["drug_id"]}_{target_moa.graph["disease_mesh"]}_AddLink',
                        # As done when building DrugMechDB from raw data
                        sink_node=target_moa.graph["disease_mesh"],
                        root_node=source_moa.graph["drug_id"],
                        # New attrs
                        source_moa=source_moa.graph["_id"],
                        source_node=source_node,
                        source_node_type=drugmechdb.get_entity_type(source_moa, source_node),
                        target_moa=target_moa.graph["_id"],
                        target_protein=target_protein,
                        )
    new_moa = MoaGraph(**new_moa_attr)

    # Copy subgraphs from source, target
    if source_node == source_moa_drug_node:
        new_moa.add_node(source_node,
                         EntityType=drugmechdb.get_entity_type(source_moa, source_node),
                         name=drugmechdb.get_node_name(source_moa, source_node)
                         )
    else:
        copy_subgraph(new_moa, drugmechdb, source_moa, source_moa_drug_node, source_node)

    copy_subgraph(new_moa, drugmechdb, target_moa, target_protein, target_moa_disease_node)

    # Now add the new link
    if new_moa.has_edge(source_node, target_protein, key=new_reln):
        print(f"*** WARNING: New Link edge {(source_node, target_protein, new_reln)} already exists!")
    else:
        new_moa.add_edge(source_node, target_protein, key=new_reln, Relation=new_reln)

    return new_moa


def copy_subgraph(dest_moa: MoaGraph, drugmechdb: DrugMechDB, source_moa: MoaGraph, from_node: str, to_node: str):
    """
    Copy the sub-graph of `source_moa` between `from_node` and `to_node` into `dest_moa`.
    `source_moa` is a MoaGraph in `drugmechdb`.
    """
    # We know `source_moa` is a DAG.
    # We will rely on nx.all_simple_paths to cover all the nodes/edges connecting `from_node` to `to_node`.

    all_edges = set(edge for path in nx.all_simple_edge_paths(source_moa, from_node, to_node) for edge in path)

    # First add all the nodes
    all_nodes = set(u for u, v, k in all_edges) | set(v for u, v, k in all_edges)
    for node in all_nodes:
        # This will not add the (EntityType, node) if it already exists
        dest_moa.add_node(node,
                          EntityType=drugmechdb.get_entity_type(source_moa, node),
                          name=drugmechdb.get_node_name(source_moa, node))

    # Then add all the edges
    for u, v, k in all_edges:
        # Avoid duplicate edges (in case they were added before calling this fn)
        if not dest_moa.has_edge(u, v, key=k):
            dest_moa.add_edge(u, v, k, Relation=k)

    return


def test_add_link_samples(min_count=1, max_per_target=2, randomize=False, seed=42):

    pp_funcargs(test_add_link_samples)

    random.seed(seed)

    drugmechdb = load_drugmechdb()
    reduced_dmdb = get_valid_simulatable_moas(drugmechdb)
    print()

    target_moa_data_seq = select_target_moas(reduced_dmdb)

    print(f"Nbr MoA's in full DrugMechDB    = {drugmechdb.nbr_indications():,d}")
    print(f"Nbr MoA's in reduced DrugMechDB = {reduced_dmdb.nbr_indications():,d}")
    print(f"Nbr candidate Target MoA data   = {len(target_moa_data_seq):,d}")
    print(f"Nbr unique Target MoA's         = {len(set(x for x, _, _ in target_moa_data_seq)):,d}")
    print("\n")

    n = 0
    n_success = 0
    target_idx, count_for_target = 0, 0
    source_moas_seq = None

    while n < min_count:
        # select target_moa_data
        if randomize:
            target_moa_data = random.choice(target_moa_data_seq)
        else:
            if count_for_target >= max_per_target:
                target_idx += 1
                count_for_target = 0
                source_moas_seq = None

            target_moa_data = target_moa_data_seq[target_idx]
            count_for_target += 1

        # select source_moa
        if randomize:
            source_moas_seq = select_source_moas(reduced_dmdb, target_moa_data)
            source_moa = random.choice(source_moas_seq)
        else:
            if source_moas_seq is None:
                source_moas_seq = select_source_moas(reduced_dmdb, target_moa_data)

            source_moa = source_moas_seq[count_for_target - 1]

        # Test for source_node = Drug
        n += 1
        source_drug_node = reduced_dmdb.get_moa_source_drug_node(source_moa)
        _, _, success = verbose_test_add_link(n, reduced_dmdb, source_moa, source_drug_node, target_moa_data)
        n_success += success

        # Test protein nodes
        n_prot = 0
        for source_prot in source_moa.get_nodes_for_type(PROTEIN_ENTITY_TYPE):
            n_prot += 1
            n += 1
            _, _, success = verbose_test_add_link(n, reduced_dmdb, source_moa, source_prot, target_moa_data)
            n_success += success

        if n_prot > 0:
            print()
            print(f"--- Nbr source protein nodes tested = {n_prot}")
        else:
            print("--- No Protein Nodes in this source MoA:", reduced_dmdb.get_moa_id(source_moa))

        print("\n")

    print()
    pp_underlined_hdg("Test Summary", linechar="=", overline=True)
    print(f"Nbr samples tested     = {n:3,d}")
    print(f"Nbr successful tests   = {n_success:3,d}")
    print(f"Nbr failed Simulations = {n - n_success:3,d}")
    print()
    return


def verbose_test_add_link(test_nbr: int,
                          drugmechdb: DrugMechDB,
                          source_moa: MoaGraph,
                          source_node: str,
                          target_moa_data: Tuple[MoaGraph, str, str]):

    target_moa, target_protein, target_in_reln = target_moa_data

    source_drug_node = drugmechdb.get_moa_source_drug_node(source_moa)
    target_disease_node = drugmechdb.get_moa_target_disease_node(target_moa)

    if source_node != source_drug_node:
        hdg = (source_moa.get_qualified_node_name(source_drug_node) + " --> " +
               source_moa.get_qualified_node_name(source_node) + " ==> " +
               target_moa.get_qualified_node_name(target_protein) + " --> " +
               target_moa.get_qualified_node_name(target_disease_node)
               )
    else:
        hdg = (source_moa.get_qualified_node_name(source_node) + " ==> " +
               target_moa.get_qualified_node_name(target_protein) + " --> " +
               target_moa.get_qualified_node_name(target_disease_node)
               )

    pp_underlined_hdg(f"[{test_nbr}] {hdg}", linechar="=", overline=True)

    print()
    pp_underlined_hdg(f"Source MoA: {drugmechdb.get_moa_id(source_moa)}")
    source_moa.pprint()

    print()
    pp_underlined_hdg(f"Target MoA: {drugmechdb.get_moa_id(target_moa)}")
    target_moa.pprint()

    new_moa, new_reln = add_link(drugmechdb, source_moa, source_node, target_moa_data)

    extra_nodes_in_new_moa = set(new_moa.nodes) - {source_drug_node, source_node, target_protein, target_disease_node}

    print()
    pp_underlined_hdg("New MoA")
    print("    Source node =", source_moa.get_qualified_node_name(source_node))
    print("    Target node =", target_moa.get_qualified_node_name(target_protein))
    print("    New relation =", new_reln)
    print(f"    Total nbr nodes = {new_moa.number_of_nodes():2d}")
    print(f"    Extra nbr nodes = {len(extra_nodes_in_new_moa):2d}",
          " ... nodes other than Drug, Disease, source-Nd, target-Nd.")
    print(f"    Total nbr edges = {new_moa.number_of_edges():2d}")
    print( "    Entity types involved =", ", ".join(sorted(new_moa.entity_type_nodes.keys())))
    print()
    new_moa.pprint()

    print()
    pp_underlined_hdg("Simulation of New MoA")
    qsim = QualitatveSimulator(new_moa)
    dis_level = qsim.simulate(verbose=True)
    if dis_level is None or dis_level >= 0:
        print(f"\n*** WARNING: Invalid final Disease level = {dis_level}!\n")

    print("\n")

    return new_moa, new_reln, dis_level < 0


# ======================================================================================================
#   Main
# ======================================================================================================

# To run
# ------
# [Python]$ python -m drugmechcf.kgproc.addlink test

if __name__ == "__main__":

    import argparse
    from datetime import datetime
    from drugmechcf.utils.misc import print_cmd

    _argparser = argparse.ArgumentParser(
        description='Qualitative Simulation of DrugMechDB MoAs',
    )

    _subparsers = _argparser.add_subparsers(dest='subcmd',
                                            title='Available commands',
                                            )
    # Make the sub-commands required
    _subparsers.required = True

    # ... test [-r] ?count
    _sub_cmd_parser = _subparsers.add_parser('test',
                                             help="Test some samples of add-link.")
    _sub_cmd_parser.add_argument('-r', '--randomize', action='store_true',
                                 help="Randomize the MoA selection(s).")
    _sub_cmd_parser.add_argument('count', nargs="?", type=int, default=1,
                                 help="Min nbr of samples to test.")

    # ...

    _args = _argparser.parse_args()
    # .................................................................................................

    start_time = datetime.now()

    print("---------------------------------------------------------------------")
    print_cmd()

    if _args.subcmd == 'test':

        test_add_link_samples(min_count=_args.count, randomize=_args.randomize)

    else:

        raise NotImplementedError(f"Command not implemented: {_args.subcmd}")

    # /

    print('\nTotal Run time =', datetime.now() - start_time)
    print()
