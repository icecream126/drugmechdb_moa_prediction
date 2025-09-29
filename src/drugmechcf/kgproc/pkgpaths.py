"""
Paths in PrimeKG
"""

from collections import Counter, defaultdict
from typing import List, Union

import networkx as nx

from drugmechcf.data.primekg import PrimeKG, load_primekg
from drugmechcf.utils.misc import pp_underlined_hdg, pp_funcargs


# -----------------------------------------------------------------------------
#   Globals
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
#   Functions
# -----------------------------------------------------------------------------


def find_sss_paths(source: str,
                   maxlength: int | None,
                   pkg: PrimeKG = None,
                   target_types: Union[str, List[str]] = None,
                   minlength: int = 1,
                   ) -> List[List[str]]:
    """
    Find all single_source_shortest_path's from `source`.

    :param source: The start node for all paths requested
    :param maxlength: Max path length, in nbr edges. None means no limit.
    :param pkg: instance of `PrimeKG`
    :param target_types: Restrict destination nodes to one of these Entity Types
    :param minlength: Minimum path length, in nbr edges
    :return: List of Paths, where
        each Path is a Sequence of Node[str] starting at `source` and ending in the `destination` node,
        s.t. there for each consecutive pair of nodes u, v in the path, there is an edge (u, v) in `pkg`.
    """
    assert maxlength is None or maxlength >= minlength > 0

    if pkg is None:
        pkg = load_primekg()

    if isinstance(target_types, str):
        target_types = [target_types]

    if maxlength == 1:
        # faster to just get the edges from `source`
        all_paths = pkg.out_edges(source)
    else:
        all_paths = nx.single_source_shortest_path(pkg, source, cutoff=maxlength).items()

    all_paths = list(filter(lambda path: (len(path) > minlength)
                                         and (target_types is None or pkg.get_entity_type(path[-1]) in target_types),
                            all_paths))

    return all_paths


def find_sts_paths(target_node: str,
                   maxlength: int | None,
                   pkg: PrimeKG = None,
                   src_types: Union[str, List[str]] = None,
                   minlength: int = 1,
                   ) -> List[List[str]]:

    """
    Single target paths to `target_node`, og upto `maxlength`
    """

    assert minlength > 0
    assert maxlength is None or maxlength >= minlength > 0

    if pkg is None:
        pkg = load_primekg()

    if isinstance(src_types, str):
        src_types = [src_types]

    all_paths = nx.single_target_shortest_path(pkg, target_node, cutoff=maxlength)

    if not all_paths:
        return []

    all_paths = filter(lambda path: len(path) > minlength, all_paths.values())

    if src_types:
        all_paths = filter(lambda path: pkg.get_entity_type(path[0]) in src_types, all_paths)

    return list(all_paths)


# -----------------------------------------------------------------------------
#   Functions: Pretty-Printing Paths
# -----------------------------------------------------------------------------


def get_edge_name(pkg: PrimeKG, head, tail, sep=" || ", display_name_only=False) -> str:
    edges = pkg.get_edge_data(head, tail)

    if display_name_only:
        edge_name = sep.join(f"{rd['name']}" for rd in edges.values())
    else:
        edge_name = sep.join(f"{r} ({rd['name']})" for r, rd in edges.items())

    return edge_name


def pp_edge_to(pkg: PrimeKG, head, tail, indent: str = ""):
    reln = get_edge_name(pkg, head, tail)

    if indent:
        print(indent, end="")

    print("--", reln, "-->", pkg.get_qualified_node_name(tail))

    return


def pp_path(pkg: PrimeKG, path: List[str], prefix: str = "", edge_indent: str = "  "):

    print(prefix + pkg.get_qualified_node_name(path[0]), "-to-",
          pkg.get_qualified_node_name(path[-1]),
          f"[length = {len(path) - 1}]:"
          )

    u = path[0]
    for v in path[1:]:
        pp_edge_to(pkg, u, v, indent=edge_indent)
        u = v

    print()
    return


# -----------------------------------------------------------------------------
#   Functions: Reports
# -----------------------------------------------------------------------------


def disease_report(disease_node: str,
                   maxlength: int | None,
                   pkg: PrimeKG = None):
    pp_funcargs(disease_report)

    if pkg is None:
        pkg = load_primekg()

    print()
    pp_underlined_hdg(f"Disease report for: {pkg.get_qualified_node_name(disease_node)}",
                      linechar='=', overline=True)

    drug_disease_relns = ["indication", "off-label use", "contraindication"]

    pp_underlined_hdg("Drugs")

    # noinspection PyArgumentList
    drug_relns_seq = [(u, k) for u, v, k in pkg.in_edges(disease_node, keys=True) if k in drug_disease_relns]
    # convert to dict
    reln_drugs = defaultdict(set)
    for u, k in drug_relns_seq:
        reln_drugs[k].add(u)

    for reln in drug_disease_relns:
        drugs = sorted(reln_drugs[reln], key=lambda d: pkg.get_node_name(d))
        print(f"{reln}: {len(drugs)} drugs")
        for n, dd in enumerate(drugs, start=1):
            print(f"  [{n:2d}] {pkg.get_qualified_node_name(dd)}")
        print()

    print()
    pp_underlined_hdg("Paths from all Drugs")

    all_paths = find_sts_paths(disease_node, maxlength, pkg, src_types="drug", minlength=2)
    # exclude drug-drug interactions
    all_paths = list(filter(lambda p: pkg.get_entity_type(p[1]) != 'drug', all_paths))

    # Sort on Drug, increasing length
    drug = None
    n_drugs = 0
    np_in_d = 0
    n_drugs_by_reln = Counter()
    for n, path in enumerate(sorted(all_paths, key=lambda p: (p[0], len(p))), start=1):
        if drug != path[0]:
            n_drugs += 1
            np_in_d = 0
            drug = path[0]

            has_reln = None
            for k, v in reln_drugs.items():
                if drug in v:
                    has_reln = k
                    n_drugs_by_reln[k] += 1
                    break

            print()
            print(f"{n_drugs}: {pkg.get_entity_type(drug)}:{pkg.get_node_name(drug)}",
                  f" -- known {has_reln}" if has_reln else "")

        np_in_d += 1
        pp_path(pkg, path, prefix=f"[{n_drugs}.{np_in_d}] ({n}): ", edge_indent="      ")

    print()
    pp_underlined_hdg("Drug paths summary")
    print("Max length of paths =", maxlength)
    print(f"Total nbr paths = {len(all_paths):,d}")
    print(f"Nbr connected drugs = {n_drugs:,d}")
    for reln in drug_disease_relns:
        print(f"   ... {reln} = {n_drugs_by_reln[reln]:,d}")

    return


# ======================================================================================================
#   Main
# ======================================================================================================

# To run
# ------
# [Python]$ python -m drugmechcf.kgproc.pkgpaths ...

if __name__ == "__main__":

    import argparse
    from datetime import datetime
    from drugmechcf.utils.misc import print_cmd

    _argparser = argparse.ArgumentParser(
        description='PrimeKG paths',
    )

    _subparsers = _argparser.add_subparsers(dest='subcmd',
                                            title='Available commands',
                                            )
    # Make the sub-commands required
    _subparsers.required = True

    # ... disease
    _sub_cmd_parser = _subparsers.add_parser('disease',
                                             help="Disease report.")
    _sub_cmd_parser.add_argument('-m', '--maxlen', type=int, default=2,
                                 help="Maximum length of path, in nbr edges.")
    _sub_cmd_parser.add_argument('dis_node', type=str,
                                 help="Path to KGML options file.")

    # ...

    _args = _argparser.parse_args()
    # .................................................................................................

    start_time = datetime.now()

    print("---------------------------------------------------------------------")
    print_cmd()

    if _args.subcmd == 'disease':

        disease_report(_args.dis_node, _args.maxlen)

    else:

        raise NotImplementedError(f"Command not implemented: {_args.subcmd}")

    # /

    print('\nTotal Run time =', datetime.now() - start_time)
    print()
