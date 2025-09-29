"""
Simulation of MoA interactions
"""

from collections import defaultdict
from enum import IntEnum
import random
from typing import Dict, List, Optional

import networkx as nx

from drugmechcf.data.drugmechdb import DrugMechDB, load_drugmechdb
from drugmechcf.data.moagraph import MoaGraph
from drugmechcf.utils.misc import pp_underlined_hdg, pp_funcargs


# -----------------------------------------------------------------------------
#   Globals
# -----------------------------------------------------------------------------

class ChangeDirection(IntEnum):
    UNKNOWN = -999
    NEGATIVE = -1
    SAME = 0
    POSITIVE = 1

# /


# These are the 20 most frequent (nbr edges across DrugMechDB) relations.
DEFAULT_RELN_CAUSAL_MECHANISMS = {
    "positively regulates": ChangeDirection.POSITIVE,
    "positively correlated with": ChangeDirection.POSITIVE,
    "decreases activity of": ChangeDirection.NEGATIVE,
    "increases activity of": ChangeDirection.POSITIVE,
    "negatively regulates": ChangeDirection.NEGATIVE,
    "causes": ChangeDirection.POSITIVE,
    "participates in": ChangeDirection.SAME,
    "negatively correlated with": ChangeDirection.NEGATIVE,
    "increases abundance of": ChangeDirection.POSITIVE,
    "occurs in": ChangeDirection.SAME,
    "manifestation of": ChangeDirection.SAME,
    "in taxon": ChangeDirection.SAME,
    "contributes to": ChangeDirection.POSITIVE,
    "located in": ChangeDirection.SAME,
    "decreases abundance of": ChangeDirection.NEGATIVE,
    "correlated with": ChangeDirection.POSITIVE,
    "location of": ChangeDirection.SAME,
    "has metabolite": ChangeDirection.UNKNOWN,          # *** Un-determined direction
    "disrupts": ChangeDirection.NEGATIVE,
    "part of": ChangeDirection.SAME,
}


# -----------------------------------------------------------------------------
#   Classes
# -----------------------------------------------------------------------------


class QualitatveSimulator:

    def __init__(self, moa: MoaGraph, relation_causal_mechanisms: dict[str, ChangeDirection] = None):
        self.moa = moa

        if relation_causal_mechanisms is None:
            self.relation_causal_mechanisms = DEFAULT_RELN_CAUSAL_MECHANISMS
        else:
            self.relation_causal_mechanisms = relation_causal_mechanisms

        self.node_levels: Dict[str, Optional[int]] = dict()

        self.node_temp_levels: Dict[str, List[int | None]] = defaultdict(list)
        return

    def set_node_level(self, node: str, level: int):
        self.node_levels[node] = level

        self.node_temp_levels[node] = []
        if level is not None:
            self.node_temp_levels[node].append(level)

        return

    @staticmethod
    def clamp_level(level: int | None):
        if level is not None:
            if level < 0:
                level = -1
            elif level > 0:
                level = 1

        return level

    def accumulate_levels(self, node: str, clamp=True, default_level: Optional[int] = None, verbose=False):
        """
        Accumulate the temp-levels on the node so far, and set the node's level to the accumulated level.
        IF the node does not have any temp-levels THEN
            node set to default level, if needed.

        IF set_level(node, Level) has been called
        THEN accumulate is essentially a No-Op
        """
        levels = self.node_temp_levels[node]
        if levels:
            total_level = sum(levels)
            if clamp:
                new_level = self.clamp_level(total_level)
            else:
                new_level = total_level

            self.set_node_level(node, new_level)

            if verbose:
                clamping = f" = clamp({total_level:+d})" if clamp else ""
                print(f"  accumulate: {node} <- {new_level:+d}{clamping}")

        elif self.node_levels.get(node) != default_level:
            # This should not normally happen

            self.set_node_level(node, default_level)

            if verbose:
                print(f"  accumulate: {node} <- {default_level}")

        elif verbose:
            print(f"  accumulate: {node} = {self.node_levels.get(node)}")

        return

    def get_node_level(self, node: str, clamp=True, default_level: Optional[int] = None) -> int | None:
        level = self.node_levels.get(node, default_level)
        if clamp:
            level = self.clamp_level(level)

        return level

    def get_all_node_levels(self) -> Dict[str, Optional[int]]:
        return self.node_levels

    def add_to_node_level(self, node: str, level: int):
        self.node_temp_levels[node].append(level)
        return

    # noinspection PyUnusedLocal
    def transmit_pos(self, head: str, reln: str, tail: str):
        head_level = self.get_node_level(head, default_level=0)
        tail_level = head_level * 1
        self.add_to_node_level(tail, tail_level)
        return

    # noinspection PyUnusedLocal
    def transmit_neg(self, head: str, reln: str, tail: str):
        head_level = self.get_node_level(head, default_level=0)
        tail_level = head_level * (-1)
        self.add_to_node_level(tail, tail_level)
        return

    # noinspection PyUnusedLocal
    def transmit_same(self, head: str, reln: str, tail: str):
        head_level = self.get_node_level(head, default_level=0)
        tail_level = head_level
        self.add_to_node_level(tail, tail_level)
        return

    def get_relation_direction(self, reln: str):
        return self.relation_causal_mechanisms.get(reln, ChangeDirection.UNKNOWN)

    # noinspection PyUnusedLocal
    def transmit(self, head: str, reln: str, tail: str):
        transmit_fn_map = {
            ChangeDirection.POSITIVE: self.transmit_pos,
            ChangeDirection.NEGATIVE: self.transmit_neg,
            ChangeDirection.SAME: self.transmit_same
        }
        chg_dirn = self.get_relation_direction(reln)

        assert chg_dirn != ChangeDirection.UNKNOWN, \
            "No direction assigned to edge: " + self.moa.get_edge_repr(head, reln, tail)

        tx_method = transmit_fn_map[chg_dirn]
        # noinspection PyArgumentList
        return tx_method(head, reln, tail)

    def simulate(self, verbose=False):
        """
        Sets the MoA's `root_node` to +1, and propagates levels through the MoA.
        After simulation, you can check any node's level using `self.get_node_level()`.

        May raise an exception:
            networkx.NetworkXUnfeasible: Graph contains a cycle

        :return: level at the MoA's `sink_node`.
        """

        root = self.moa.get_root_node()
        self.set_node_level(root, 1)

        accumulated_nodes = set()

        # Top-sort ensures:
        #   - you only need to look at out-edges (assuming DAG)
        #   - `accumulate_levels()` gets called for every node, after all propagations to it.
        for head in nx.topological_sort(self.moa):

            self.accumulate_levels(head, verbose=False)
            accumulated_nodes.add(head)

            # noinspection PyArgumentList
            for _, tail, reln in self.moa.out_edges(head, keys=True):
                old_tail_level = self.get_node_level(tail)
                self.transmit(head, reln, tail)
                if verbose:
                    self.pp_transmission(head, reln, tail, old_tail_level)

        sink = self.moa.get_sink_node()
        sink_level = self.get_node_level(sink)

        if verbose:
            print()
            print("Final level of", self.moa.get_qualified_node_name(sink), "=", sink_level, flush=True)

        return sink_level

    def pp_transmission(self, head: str, reln: str, tail: str, old_tail_level: Optional[int]):
        head_repr = self.moa.get_qualified_node_name(head) + f" [{self.get_node_level(head, clamp=False)}]"
        reln_repr = f"{reln} [{self.get_relation_direction(reln).name}]"
        tail_rerpr = (self.moa.get_qualified_node_name(tail) +
                      f" [{old_tail_level} -> + {self.node_temp_levels[tail][-1]:+d}]")
        print(head_repr, reln_repr, tail_rerpr, sep=" | ", flush=True)
        return

# /


# -----------------------------------------------------------------------------
#   Functions
# -----------------------------------------------------------------------------


def find_simulatable_moas(dmdb: Optional[DrugMechDB] = None):
    """
    Get list of MoA's that can be simulated: every edge is of a relation that can be simulated,
    :return:

    Returns 3,592 / 4,846 = 74.1%
    """

    if dmdb is None:
        dmdb = load_drugmechdb()

    simulatable_relations = set(reln for reln, chgdir in DEFAULT_RELN_CAUSAL_MECHANISMS.items()
                                if chgdir is not ChangeDirection.UNKNOWN)

    moa_list = []
    for moa in dmdb.indication_graphs:
        is_simulatable = len(moa.relations - simulatable_relations) == 0

        if is_simulatable:
            moa_list.append(moa)

    return moa_list


def test(count=1, randomize=True, seed=42):
    pp_funcargs(test)

    dmdb: DrugMechDB = load_drugmechdb()

    moa_list = find_simulatable_moas(dmdb)

    print(f"Nbr simulatable MoA's found = {len(moa_list):,d} / {dmdb.nbr_indications():,d}")
    print()

    if randomize:
        random.seed(seed)
        random.shuffle(moa_list)

    n_warnings = 0

    n = None
    for n, moa in enumerate(moa_list[:count], start=1):
        simulator = QualitatveSimulator(moa)

        print()
        pp_underlined_hdg(f"[{n}] " + moa.get_graph_heading())

        dis_level = simulator.simulate(verbose=True)
        if dis_level >= 0:
            n_warnings += 1
            print(f"*** WARNING: Final disease level {dis_level} is Not Negative!")
        print()

    print()
    pp_underlined_hdg("Simulation results", overline=True)
    print(f"Nbr MoAs simulated      = {n:5,d}")
    print(f"Nbr simulation warnings = {n_warnings:5,d}")
    print()

    return


def get_valid_simulatable_moas(dmdb: Optional[DrugMechDB] = None) -> DrugMechDB:
    """
    Return list of MoAs that have correct simulation, resulting in Disease getting a -ive label.
    :return: New DrugMechDB instance containing only the valid MoA's

    ...
    returned count = 3,523 / 3,592 / 4,846
                ... 98.1% of 74.1% = 72.7%

    """

    if dmdb is None:
        dmdb = load_drugmechdb(verbose=False)

    reduced_dmdb = DrugMechDB()

    for moa in find_simulatable_moas(dmdb):
        simulator = QualitatveSimulator(moa)
        dis_level = simulator.simulate(verbose=False)
        if dis_level < 0:
            reduced_dmdb.add_indication_graph(dmdb.get_moa_source_drug_node(moa),
                                              dmdb.get_moa_target_disease_node(moa),
                                              moa)

    return reduced_dmdb


def basic_stats():
    """
    $> python -m kgproc.simulate stats

    Full DrugMechDB, nbr MoA's  = 4,846
    nbr Simulatable MoA's       = 3,592  ... 74.1%
       nbr Drug is not Root     =     0
       nbr Disease is not Sink  =     0
       nbr both                 =     0
    nbr Valid Simulatable MoA's = 3,523  ... 98.1% simulatable,  72.7% full DrugmechDB.


    Total Run time = 0:00:00.211061
    """

    dmdb = load_drugmechdb(verbose=False)

    simulatable_moas = find_simulatable_moas(dmdb)
    n_simulatable = len(simulatable_moas)

    n_drug_not_root = 0
    n_disease_not_sink = 0
    n_drug_disease_not_root_sink = 0

    for moa in simulatable_moas:
        drug_node, disease_node = dmdb.get_moa_drug_disease_nodes(moa)

        if drug_not_sink := (moa.get_root_node() != drug_node):
            n_drug_not_root += 1

        if disease_not_sink := (moa.get_sink_node() != disease_node):
            n_disease_not_sink += 1

        if drug_not_sink and disease_not_sink:
            n_drug_disease_not_root_sink += 1

    reduced_dmdb = get_valid_simulatable_moas(dmdb)

    print(f"Full DrugMechDB, nbr MoA's  = {dmdb.nbr_indications():5,d}")
    print(f"nbr Simulatable MoA's       = {n_simulatable:5,d}  ... {n_simulatable / dmdb.nbr_indications():.1%}")
    print(f"   nbr Drug is not Root     = {n_drug_not_root:5,d}")
    print(f"   nbr Disease is not Sink  = {n_disease_not_sink:5,d}")
    print(f"   nbr both                 = {n_drug_disease_not_root_sink:5,d}")
    print(f"nbr Valid Simulatable MoA's = {reduced_dmdb.nbr_indications():5,d}",
          f" ... {reduced_dmdb.nbr_indications() / n_simulatable:.1%} simulatable,",
          f" {reduced_dmdb.nbr_indications() / dmdb.nbr_indications():.1%} full DrugmechDB.")
    print()
    return


# ======================================================================================================
#   Main
# ======================================================================================================

# To run
# ------
# [Python]$ python -m drugmechcf.kgproc.simulate [ test | stats ]

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
                                             help="Simulate one or more MoAs.")
    _sub_cmd_parser.add_argument('-r', '--randomize', action='store_true',
                                 help="Randomize the MoA selection(s).")
    _sub_cmd_parser.add_argument('count', nargs="?", type=int, default=1,
                                 help="Nbr of MoA's to simulate.")

    # ... stats
    _sub_cmd_parser = _subparsers.add_parser('stats',
                                             help="Basic stats on simulatable DrugMechDB.")

    # ...

    _args = _argparser.parse_args()
    # .................................................................................................

    start_time = datetime.now()

    print("---------------------------------------------------------------------")
    print_cmd()

    if _args.subcmd == 'test':

        test(_args.count, randomize=_args.randomize)

    elif _args.subcmd == 'stats':

        basic_stats()

    else:

        raise NotImplementedError(f"Command not implemented: {_args.subcmd}")

    # /

    print('\nTotal Run time =', datetime.now() - start_time)
    print()
