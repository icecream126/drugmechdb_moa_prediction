"""
The MONDO ontology
"""


from collections import defaultdict
import dataclasses
import json
import os.path
import pickle
from typing import List, Optional, Sequence, Union

import networkx as nx
import pronto

from drugmechcf.utils.misc import ValidatedDataclass, NO_DEFAULT
from drugmechcf.utils.projconfig import get_project_config


# -----------------------------------------------------------------------------
#   Classes
# -----------------------------------------------------------------------------


@dataclasses.dataclass
class MondoOpts(ValidatedDataclass):
    descr: str = None

    data_base_dir: str = None
    """
    Base dir for the `srcdir`.
    Default is to use `utils.projconfig.ProjectConfig.get_input_data_dir()`. 
    """

    srcdir: str = NO_DEFAULT
    """
    Dir containing source files to build PrimeKG from.
    In an options file, path is relative to `data_base_dir`.
    """

    cachefile: str = None
    """
    Where built PrimeKG is cached.
    In an options file, path is relative to `srcdir`. 
    """

    include_obsolete_entries: bool = True
    """
    Whether nodes marked as 'obsolete' are to be included in the ontology.
    Useful when using this ontology with PrimeKG, which is based on an older version of Mondo.
    """

    def __post_init__(self):
        if self.data_base_dir is None:
            self.data_base_dir = get_project_config().get_input_data_dir()
        self.srcdir = os.path.join(self.data_base_dir, self.srcdir)
        if self.cachefile is not None:
            self.cachefile = os.path.join(self.srcdir, self.cachefile)
        return

    # noinspection PyMethodOverriding
    def assert_matches(self, other):
        return super().assert_matches(other, skip_fields = ["optsdir", "srcdir", "cachefile"])

    def to_json_file(self, json_file: str):
        jdict = dataclasses.asdict(self)

        # Undo `__post_init__`
        jdict['cachefile'] = os.path.relpath(self.cachefile, start=self.srcdir)
        jdict['srcdir'] = os.path.relpath(self.srcdir, start=self.data_base_dir)

        if self.data_base_dir == get_project_config().get_input_data_dir():
            del jdict['data_base_dir']

        with open(json_file, "w") as jf:
            json.dump(jdict, jf, indent=4)
        return

# /


class Mondo(nx.DiGraph):

    def __init__(self):
        super().__init__()

        self.opts: Optional[MondoOpts] = None
        self._source = None
        self._cache_path = None

        # Node.Index[int] => Node[str]. Index in [0, nNodes - 1]
        self.idx2node = []

        self.root_nodes = []
        self.leaf_nodes = []

        self.name_to_nodes = defaultdict(list)

        self.mesh2nodes = defaultdict(list)

        return

    # ...........................................................................................
    # Loading and Saving
    # ...........................................................................................

    @staticmethod
    def load(opts: Union[str, MondoOpts],
             rebuild: bool = False, verbose: bool = True) -> "Mondo":
        """
        Load from `opts.cachefile` or (re-)build and save to `opts.cachefile`.

        :param opts: Either MondoOpts instance, or path to JSON file containing MondoOpts
        :param rebuild: IF True THEN data is rebuilt and saved.
        :param verbose:
        :return: Mondo instance
        """
        if isinstance(opts, str):
            opts = MondoOpts.from_json_file(opts)

        if os.path.exists(opts.cachefile) and not rebuild:
            if verbose:
                print("Loading from cache:", opts.cachefile, "...", flush=True)

            with open(opts.cachefile, 'rb') as f:
                mondo = pickle.load(f, fix_imports=False)

            opts.assert_matches(mondo.opts)

            mondo._cache_path = opts.cachefile

        else:
            mondo = Mondo()
            # noinspection PyProtectedMember
            mondo._build_and_save(opts)

        return mondo

    def _build_and_save(self, opts: MondoOpts, verbose: bool = True):
        self.opts = opts
        self._load_from_srcdata()
        if self.opts.cachefile is not None:
            self.save_to(self.opts.cachefile, verbose)
        return

    def save_to(self, cachefile: str, verbose: bool = True):
        if verbose:
            print("Saving to cache ...", flush=True)

        self._cache_path = cachefile
        with open(cachefile, 'wb') as f:
            # Force funcs to None, o/w can't pickle
            self.opts.keep_node_func = None
            self.opts.keep_edge_func = None
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)

        if verbose:
            print(f"{self.__class__.__name__} instance cached to:", cachefile)
        return

    def _load_from_srcdata(self, verbose: bool = True):
        """
        Load data from source data files under `opts.srcdir`.
        """
        if verbose:
            print(f"Building {self.__class__.__name__} from data sources at:", self.opts.srcdir)

        mondo_ont = self.read_ontology(self.opts.srcdir)

        if verbose:
            print("Adding nodes ...", flush=True)

        # Nodes ...
        for term in mondo_ont.terms():
            if term.id.startswith("MONDO"):
                self.add_node_from_term(term)

        # Edges ...
        for term in mondo_ont.terms():
            if self._skip_if_obsolete(term) or not self.has_node(term.id):
                continue

            super_nodes = [t for t in term.superclasses(distance=1) if self.has_node(t.id)]

            if len(super_nodes) == 1:
                self.root_nodes.append(term.id)
            else:
                for parent in super_nodes[1:]:
                    self.add_edge(term.id, parent.id, Relation="child_of")
                    self.add_edge(parent.id, term.id, Relation="parent_of")

        if verbose:
            print(f"{self.__class__.__name__} built:",
                  f"    nbr Roots  = {len(self.root_nodes):,d}",
                  f"    nbr Leaves = {len(self.leaf_nodes):,d}",
                  f"    nbr Nodes  = {self.number_of_nodes():,d}",
                  f"    nbr Edges  = {self.number_of_edges():,d}",
                  sep="\n")
            if self.opts.include_obsolete_entries:
                print("Obsolete nodes included:")
                n_obsolete = sum([self.nodes[nd]["is_obsolete"] for nd in self.root_nodes])
                print(f"    nbr Root nodes that are obsolete = {n_obsolete:,d}")
                n_obsolete = sum([self.nodes[nd]["is_obsolete"] for nd in self.leaf_nodes])
                print(f"    nbr Leaf nodes that are obsolete = {n_obsolete:,d}")
                n_obsolete = sum([self.nodes[nd]["is_obsolete"] for nd in self.nodes])
                print(f"    nbr All nodes that are obsolete  = {n_obsolete:,d}")

            print()

        return

    def _skip_if_obsolete(self, term) -> bool:
        """
        Whether to skip this `term` when it is marked as obsolete.
        Controlled by option: `include_obsolete_entries`
        """
        return not self.opts.include_obsolete_entries and term.obsolete

    @staticmethod
    def read_ontology(srcdir: str) -> pronto.Ontology:
        fpath = os.path.join(srcdir, "mondo.obo.gz")
        if os.path.exists(fpath):
            onto = pronto.Ontology(fpath)
        else:
            fpath = os.path.join(srcdir, "mondo.obo")
            onto = pronto.Ontology(fpath)

        return onto

    def add_node_from_term(self, term: pronto.Term):
        if not term.id.startswith("MONDO") or self._skip_if_obsolete(term):
            return

        synonyms = sorted(set(s.description for s in term.synonyms if s.scope == "EXACT"))
        xrefs = sorted(set(x.id for x in term.xrefs))

        self.add_node(term.id, name=term.name, synonyms=synonyms, xrefs=xrefs,
                      is_leaf=term.is_leaf(), is_obsolete=term.obsolete)
        return

    # ...........................................................................................
    # Inherited methods for building KG
    # ...........................................................................................

    # noinspection PyMethodOverriding
    def add_node(self, node: str, name: str, synonyms: Sequence[str], xrefs: Sequence[str],
                 is_leaf: bool, is_obsolete: bool):

        self.idx2node.append(node)
        self.name_to_nodes[name].append(node)

        for xref_id in xrefs:
            if xref_id.startswith("MESH"):
                self.mesh2nodes[xref_id].append(node)

        if is_leaf:
            self.leaf_nodes.append(node)

        return super().add_node(node, name=name, synonyms=synonyms, xrefs=xrefs,
                                is_leaf=is_leaf, is_obsolete=is_obsolete)

    # ...........................................................................................
    # Other methods
    # ...........................................................................................

    def get_xrefs(self, mondo_id: str, for_ont: str = None) -> List[str]:
        """
        Get the MeSH ids, in format "MESH:1234", that are x-ref'd from node `mondo_id`.
        Raises `KeyError` if `mondo_id` not present in the ontology.

        :param mondo_id: Format = "MONDO:1234"
        :param for_ont: IF provided THEN it will retrieve only xrefs to that ontology
        """
        if for_ont is not None:
            for_ont = for_ont.upper()
            if for_ont.endswith(":"):
                for_ont = for_ont[:-1]
            xrefs = [x for x in self.nodes[mondo_id]["xrefs"] if x.startswith(for_ont)]
        else:
            xrefs = self.nodes[mondo_id]["xrefs"].copy()

        return xrefs

    def get_mesh_xrefs(self, mondo_id: str) -> List[str]:
        """
        Get the MeSH ids, in format "MESH:1234", that are x-ref'd from node `mondo_id`.
        Raises `KeyError` if `mondo_id` not present in the ontology.

        :param mondo_id: Format = "MONDO:1234"
        """
        return self.get_xrefs(mondo_id, for_ont="MESH")

    def get_parents(self, mondo_id):
        parents = [v for u, v, reln in self.out_edges(mondo_id, data="Relation") if reln == "child_of"]
        return parents

    def get_children(self, mondo_id):
        children = [v for u, v, reln in self.out_edges(mondo_id, data="Relation") if reln == "parent_of"]
        return children

# /


# -----------------------------------------------------------------------------
#   Functions
# -----------------------------------------------------------------------------


def load_mondo(opts: str = None, rebuild: bool = False, verbose=True):
    # Force local import, to avoid Pickle errors during loading like:
    # AttributeError: Can't get attribute 'PrimeKG' on <module 'exps.primekg...' from .../exps/.../trainbase.py'>
    # noinspection PyUnresolvedReferences
    from drugmechcf.data.mondo import MondoOpts, Mondo

    if opts is None:
        opts = MondoOpts(srcdir="MONDO", cachefile="mondo.pkl", include_obsolete_entries=True)

    return Mondo.load(opts, rebuild=rebuild, verbose=verbose)


# ======================================================================================================
#   Main
# ======================================================================================================

# To run
# ------
# [Python]$ python -m drugmechcf.data.mondo build [-r] [ MondoOpts-File ]

if __name__ == "__main__":

    import argparse
    from datetime import datetime
    from drugmechcf.utils.misc import print_cmd

    _argparser = argparse.ArgumentParser(
        description='MONDO ontology',
    )

    _subparsers = _argparser.add_subparsers(dest='subcmd',
                                            title='Available commands',
                                            )
    # Make the sub-commands required
    _subparsers.required = True

    # ... build
    _sub_cmd_parser = _subparsers.add_parser('build',
                                             help="Build the Mondo Ontology and save to cache.")
    _sub_cmd_parser.add_argument('-r', '--rebuild', action='store_true',
                                 help="Forced rebuild of the Mondo Ontology.")
    _sub_cmd_parser.add_argument('mondo_opts', type=str, nargs='?', default=None,
                                 help="Path to Mondo options file.")

    # ...

    _args = _argparser.parse_args()
    # .................................................................................................

    start_time = datetime.now()

    print("---------------------------------------------------------------------")
    print_cmd()

    if _args.subcmd == 'build':

        load_mondo(_args.mondo_opts, rebuild=_args.rebuild)

    else:

        raise NotImplementedError(f"Command not implemented: {_args.subcmd}")

    # /

    print('\nTotal Run time =', datetime.now() - start_time)
    print()
