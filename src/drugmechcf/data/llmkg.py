"""
For representing a BioKG extracted from a LLM
"""

import dataclasses
from collections import defaultdict, Counter
from functools import partial
import itertools
import os.path
import pickle
import regex
from typing import Dict, Iterable, List, Sequence, Union

import networkx as nx
import pandas as pd

from drugmechcf.text.textutils import standardize_chars_unidecode, translate_text
from drugmechcf.utils.misc import reset_df_index, ppmd_counts_df, pp_dict


# -----------------------------------------------------------------------------
#   Globals
# -----------------------------------------------------------------------------

# Ordered in most likely transition order, for pretty-printing needs
EXPECTED_ENTITY_TYPES = ["drug", "protein", "biological process", "chemical", "phenotype", "disease"]


# Token consists of a sequence of:
#   - Word-char (but not underscore '_')
#   - Unicode Accent char e.g. [é] in 'Montréal' => 'Montreal'
TOKEN_PATT = regex.compile(r"((?:(?!_)\w|\p{Mn})+)")

# Seq of any non-word chars, including underscore ('_')
NONWORD_CHARS_PATT = regex.compile(r"(?:\W|_)+")


# -----------------------------------------------------------------------------
#   Classes
# -----------------------------------------------------------------------------


class LLMKG(nx.MultiDiGraph):
    """
    Nodes are entities. Each Node has:
        node: str = f"{EntityType}:{Primary_Name}"
            EntityType included in `node` to ensure uniqueness.
        Data attributes:
            EntityType: str
            alt_names: List[str] = None

    Edge attributes:
        Relation: str. The same as the `key`.
        descriptions: list[str] [Optional] Any description(s) of this relationship.
    """

    def __init__(self, *, llm_opts, **attr):
        """
        Create an instance of `nx.MultiDiGraph` for extracting graphs from LLM using `llm.test_explore.TestExplore`.

        :param llm_opts: Instance of `llm.openai.OpenAICompletionOpts`, any other dataclass, or a dict[str, Any].
            The options used to create the LLM Chat client used to build this graph.

        Other optional attributes:
            forward_schema or backward_schema
            beam_width
        """

        super().__init__(**attr)

        self.llm_opts = llm_opts

        # The first entities used to start the graph
        self.seed_nodes: List[str] = []

        # --- Processed Entity-Type pairs when mining for KG ---
        # Source-nodes: (src_type, tgt_type) => Set[Node] of source-nodes
        self.mined_ixns_srcnodes = defaultdict(set)
        # Target-nodes: (src_type, tgt_type) => Set[Node] of target-nodes
        self.mined_ixns_tgtnodes = defaultdict(set)

        self._cache_path = None

        self.checkpoint_step = 0

        # Node-type => List[ Node[str], ... ]
        self.nodetype2nodes: Dict[str, List[str]] = defaultdict(list)

        # Node-type => List[ NormalizedName[str], ... ]
        self.nodetype2_normalized_names: Dict[str, List[str]] = defaultdict(list)

        # relation => head_type => tail_type => edge_count
        self.relations_summary = defaultdict(partial(defaultdict, Counter))

        return

    def set_seed_nodes(self, start_nodes: List[str]):
        self.seed_nodes = start_nodes
        return

    def set_checkpoint_step(self, step: int):
        self.checkpoint_step = step
        return

    # ...........................................................................................
    # Loading and Saving
    # ...........................................................................................

    @staticmethod
    def load(cachefile: str, verbose=False) -> "LLMKG":
        """
        Load from `cachefile`.
        """
        if verbose:
            print("Loading from cache:", cachefile, "...", flush=True)

        with open(cachefile, 'rb') as f:
            kg = pickle.load(f, fix_imports=False)

        kg._cache_path = cachefile

        if verbose:
            print(flush=True)

        return kg

    def save_to(self, cachefile: str, verbose: bool = True, create_dir: bool = False):
        if verbose:
            print("Saving to cache ...", flush=True)

        if create_dir:
            if not os.path.exists(cdir := os.path.dirname(cachefile)):
                print("Creating output dir:", cdir)
                os.mkdir(cdir)

        if verbose:
            if os.path.exists(cachefile):
                print("Overwriting existing file.", flush=True)

        self._cache_path = cachefile
        with open(cachefile, 'wb') as f:
            # noinspection PyTypeChecker
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)

        if verbose:
            print(f"{self.__class__.__name__} instance cached to:", cachefile)
        return

    # ...........................................................................................
    # Methods for building KG from data extracted from LLM
    # ...........................................................................................

    def find_entity(self, name: str, entity_type: str, alt_name: str = None) -> str | None:
        """
        Try to match the name.
        Occasionally, ChatGPT will switch `name` and `alt_name`, or use different characters.
        This is meant to take that into account, but not too aggressively (alt-name is not normalized).

        Some examples:
            protein:NF-kB -matches- protein:NF-κB                               ... the 'k'
            disease:Alzheimer's Disease -matches- disease:Alzheimer’s Disease   ... the apostrophe char
                but it won't match "Alzheimer Disease" (... missing 's)
        """
        new_node = f"{entity_type}:{name}"
        if self.has_node(new_node):
            return new_node

        if self.number_of_nodes() == 0:
            return None

        # Check with alt_name
        if alt_name is not None:
            new_node = f"{entity_type}:{alt_name}"
            if self.has_node(new_node):
                return new_node

        # Normalize name
        name = normalize(name)
        for nnm, nd in zip(self.nodetype2_normalized_names.get(entity_type, []),
                           self.nodetype2nodes.get(entity_type, [])):
            if nnm == name:
                return nd

        return None

    def has_entity(self, name: str, entity_type: str):
        new_node = f"{entity_type}:{name}"
        return self.has_node(new_node)

    def add_entity(self, name: str, entity_type: str,
                   alt_name: str = None,
                   ) -> str:
        """
        Add a node if it does not already exist. Return the node.
        :param name:
        :param entity_type:
        :param alt_name:
        """

        if node := self.find_entity(name, entity_type, alt_name):
            # Add the alt_name, if absent
            if alt_name is not None and self.get_node_alt_name(node) is None:
                self.nodes[node]["alt_name"] = alt_name
            return node

        node_for_adding = f"{entity_type}:{name}"

        if not self.has_node(node_for_adding):
            self.nodetype2nodes[entity_type].append(node_for_adding)
            self.nodetype2_normalized_names[entity_type].append(normalize(name))

            self.add_node(node_for_adding, EntityType=entity_type, name=name, alt_name=alt_name)

        return node_for_adding

    # noinspection PyMethodOverriding
    def add_interaction(self, head: str, tail: str, relation: str,
                        description: str = None,
                        bi_directional: bool = False):
        """
        Adds an edge of type `relation` from `head` to `tail`.
        The nodes MUST exist in the KG, else raises KeyError.
        """
        # These will raise KeyError if head or tail do not already exist.
        head_type = self.nodes[head]['EntityType']
        tail_type = self.nodes[tail]['EntityType']

        if not self.has_edge(head, tail, key=relation):
            self.add_edge(head, tail, key=relation, Relation=relation, descriptions=[description])
            self.relations_summary[relation][head_type][tail_type] += 1

        if bi_directional and not self.has_edge(tail, head, key=relation):
            self.relations_summary[relation][tail_type][head_type] += 1
            self.add_edge(tail, head, key=relation, Relation=relation, descriptions=[description])

        return

    # ...........................................................................................
    # Accessor Methods
    # ...........................................................................................

    def get_entity_types(self) -> List[str]:
        return list(self.nodetype2nodes.keys())

    def get_entity_type(self, node: str) -> str:
        return self.nodes[node]["EntityType"]

    def get_node_name(self, node: str) -> str:
        return self.nodes[node]["name"]

    def get_node_alt_name(self, node: str) -> str:
        return self.nodes[node]["alt_name"]

    def get_all_node_names(self, node: str) -> list[str]:
        names = [self.get_node_name(node)]
        if alt_name := self.get_node_alt_name(node):
            names.append(alt_name)
        return names

    def get_decorated_name(self, node: str) -> str:
        ndata = self.nodes[node]
        main_name = ndata["name"]
        if alt_name := ndata.get("alt_name"):
            return f"{main_name} ({alt_name})"
        else:
            return main_name

    def get_qualified_decorated_node_name(self, node) -> str:
        return f"{self.get_entity_type(node)}: {self.get_decorated_name(node)} [{node}]"

    def get_decorated_node(self, node) -> str:
        """
        Taking advantage of `node` = f'{EntityType}:{name}', shows longer name only if there is an alt-name.
        """
        if alt_name := self.get_node_alt_name(node):
            return f"{node} ({alt_name})"
        else:
            return node

    def get_nodes_for_types(self, entity_types: Union[str, Iterable[str]]) -> Sequence[str]:
        """
        Return List of Node for `Node.EntityType = entity_types`
        The nodes are in arbitrary (but replicable) order.
        """
        if isinstance(entity_types, str):
            return self.nodetype2nodes[entity_types]
        else:
            return list(itertools.chain.from_iterable(self.nodetype2nodes[et] for et in entity_types))

    def get_nbr_nodes_for_types(self, node_types: Union[str, Iterable[str]]) -> int:
        if node_types is None:
            return self.number_of_nodes()
        elif isinstance(node_types, str):
            node_types = [node_types]

        return len(self.get_nodes_for_types(node_types))

    def find_nodes(self, subname: str, entity_type: str = None) -> List[str]:
        nodes = self.get_nodes_for_types(entity_type) if entity_type is not None else self.nodes
        subname = subname.casefold()
        matching_nodes = []
        for nd in nodes:
            if subname in self.get_node_name(nd).casefold():
                matching_nodes.append(nd)
            # noinspection PyUnresolvedReferences
            elif (alt_name := self.get_node_alt_name(nd)) and subname in alt_name.casefold():
                matching_nodes.append(nd)

        return matching_nodes

    def get_edge_counts(self,
                        relation: str = None,
                        head_types: Union[str, List[str]] = None,
                        tail_types: Union[str, List[str]] = None) -> pd.DataFrame:
        """
        Returns DataFrame with cols: relation, htype, ttype, count
        """
        # Probably a hacky way to do this
        df = pd.DataFrame.from_records([(r, h, t, n)
                                        for r, hinfo in self.relations_summary.items()
                                        for h, tinfo in hinfo.items()
                                        for t, n in tinfo.items()],
                                       columns='relation htype ttype count'.split())

        mask = (df.relation == relation) if relation is not None else (df.relation != '')

        if head_types is not None:
            if isinstance(head_types, str):
                head_types = [head_types]
            mask &= df.htype.isin(head_types)
        if tail_types is not None:
            if isinstance(tail_types, str):
                tail_types = [tail_types]
            mask &= df.ttype.isin(tail_types)

        # Return sorted in descending order on 'count', re-number rows
        counts_df = reset_df_index(df[mask].sort_values(['count', 'relation'], ascending=[False, True]))

        return counts_df

    # ...........................................................................................
    # Reports
    # ...........................................................................................

    def report_params(self):
        print("Graph params:")
        print("-------------")
        print("    Graph Start nodes:  ", ", ".join(self.seed_nodes))
        # noinspection PyUnresolvedReferences
        for k, v in self.graph.items():
            print(f"    {k} =  {v}")
        print()

        if hasattr(self, "checkpoint_step"):
            print("    checkpoint_step =", self.checkpoint_step)
            print()

        llm_opts = self.llm_opts
        hdg = "LLM Client Opts:"
        if dataclasses.is_dataclass(llm_opts):
            # noinspection PyDataclass
            hdg = llm_opts.__class__.__name__ + ":"
            llm_opts = dataclasses.asdict(llm_opts)
        pp_dict(llm_opts, msg=hdg)

        print(flush=True)
        return

    def report_basic(self):
        self.report_params()
        print()
        print(f"Total nbr nodes = {self.number_of_nodes():,d}")
        print(f"Total nbr edges = {self.number_of_edges():,d}")
        print("\n")

        # Try to nicely order the Entity-Types
        # noinspection PyUnresolvedReferences
        if schema := self.graph.get("forward_schema"):
            entity_types = schema
        elif schema := self.graph.get("backward_schema"):
            entity_types = schema[-1::-1]
        else:
            entity_types = [e for e in EXPECTED_ENTITY_TYPES if e in self.get_entity_types()]

        # De-dup, remove asterisk if present
        ets, entity_types = entity_types, []
        for et in ets:
            if et.endswith("*"):
                et = et[:-1]
            if et not in entity_types:
                entity_types.append(et)

        entity_types += [e for e in self.get_entity_types() if e not in entity_types]

        print("Nbr nodes by EntityType:\n")
        df = pd.DataFrame.from_records([(et, self.get_nbr_nodes_for_types(et)) for et in entity_types],
                                       columns=["EntityType", "Count"])
        df.loc[len(df)] = ["TOTAL", df["Count"].sum()]
        ppmd_counts_df(reset_df_index(df), "Count", sort_on_counts=False)
        print()

        print("Nbr edges by Relation:\n")
        df = self.get_edge_counts()
        df.loc[len(df)] = ["TOTAL EDGES", "*", "*", df["count"].sum()]
        ppmd_counts_df(df, "count")

        print()
        return

    # ...........................................................................................
    # Paths
    # ...........................................................................................

    def pp_path(self, path: List[str], prefix: str = "", edge_indent: str = "  ", stream=None):

        print(prefix + self.get_decorated_node(path[0]), "-to-",
              self.get_decorated_node(path[-1]),
              f"[length = {len(path) - 1}]:",
              file=stream)

        u = path[0]
        for v in path[1:]:
            self.pp_edge_to(u, v, indent=edge_indent, stream=stream)
            u = v

        print(file=stream)
        return

    def pp_edge_to(self, head, tail, indent: str = "", reln_sep=" || ", stream=None):

        edges = self.get_edge_data(head, tail)

        edge_name = reln_sep.join(edges.keys())

        if indent:
            print(indent, end="", file=stream)

        print("--", edge_name, "-->", self.get_decorated_node(tail), file=stream)

        return

    def pp_shortest_paths_to_diseases(self, src_node: str):
        print("Shortest paths to diseases from:", self.get_decorated_node(src_node))
        print()

        disease_nodes = self.get_nodes_for_types("disease")

        n_connected_diseases = 0
        tot_paths = 0

        for tgt_node in disease_nodes:
            try:
                n_connected_diseases += 1
                np = 0
                for path in nx.all_shortest_paths(self, src_node, tgt_node):
                    np += 1
                    self.pp_path(path, prefix=f"[{n_connected_diseases}.{np}] ")

                tot_paths += np

            except nx.NetworkXNoPath:
                n_connected_diseases -= 1
                continue

        print(f"Total nbr diseases in graph = {len(disease_nodes):,d}")
        print("Nbr diseases connected to source node =", n_connected_diseases)
        print(f"Total nbr paths found = {tot_paths:,d}")
        print()
        return

# /


# -----------------------------------------------------------------------------
#   Functions
# -----------------------------------------------------------------------------

def normalize_old(txt: str, to_lower=True) -> str | None:
    """
    Less aggressive and slower than `normalize()`.
    Delimiters are retained.
    """
    if txt is None:
        return None

    # Since pattern has capturing parentheses, it can create empty string tokens at either end.
    tokens = [translate_text(standardize_chars_unidecode(t)) for t in TOKEN_PATT.split(txt) if t]

    if to_lower:
        tokens = [t.casefold() for t in tokens]

    return " ".join(tokens)


def normalize(txt: str, to_lower=True) -> str | None:
    """
    Normalizes a string:
        - Non-word chars, including '_', are considered equivalent to SPACE
        - Special non-ASCII chars are converted to ascii equivalent chars / words.
    """

    txt = txt.strip()

    if txt is None:
        return None

    txt = translate_text(standardize_chars_unidecode(txt))

    if to_lower:
        txt = txt.casefold()

    tokens = NONWORD_CHARS_PATT.split(txt)

    return " ".join(tokens)


def load_llmkg(llmkg_file: str, verbose=True) -> LLMKG:
    """
    Convenience function to load LLMKG.
    """
    # Force local import, to avoid Pickle errors during loading like:
    # AttributeError: Can't get attribute 'KGML' on <module 'data.kgml...' from .../data/...'>
    # noinspection PyUnresolvedReferences
    from data.llmkg import LLMKG

    return LLMKG.load(llmkg_file, verbose=verbose)


# ======================================================================================================
#   Main
# ======================================================================================================

# To run
# ------
# [Python]$ python -m drugmechcf.data.llmkg disease_paths LLMKG_cache_file SRC_NODE
#
# [Python]$ python -m drugmechcf.data.llmkg disease_paths ../Data/Sessions/LGraph/test.pkl 'drug:Propranolol'
#

if __name__ == "__main__":

    import argparse
    from datetime import datetime
    from drugmechcf.utils.misc import print_cmd

    _argparser = argparse.ArgumentParser(
        description='KGML-xDTD data',
    )

    _subparsers = _argparser.add_subparsers(dest='subcmd',
                                            title='Available commands',
                                            )
    # Make the sub-commands required
    _subparsers.required = True

    # ... report_basic
    _sub_cmd_parser = _subparsers.add_parser('report_basic',
                                             help="Basic stats.")
    _sub_cmd_parser.add_argument('llmkg_file', type=str, help="Path to LLMKG cache Pickle file.")

    # ... disease_paths
    _sub_cmd_parser = _subparsers.add_parser('disease_paths',
                                             help="Show shortest paths to all diseases.")
    _sub_cmd_parser.add_argument('llmkg_file', type=str, help="Path to LLMKG cache Pickle file.")
    _sub_cmd_parser.add_argument("src_node", type=str, help="Source node")

    # ...

    _args = _argparser.parse_args()
    # .................................................................................................

    start_time = datetime.now()

    print("---------------------------------------------------------------------")
    print_cmd()

    if _args.subcmd == 'report_basic':

        llmkg = load_llmkg(_args.llmkg_file, verbose=True)
        print()
        llmkg.report_basic()

    elif _args.subcmd == 'disease_paths':

        llmkg = load_llmkg(_args.llmkg_file, verbose=True)
        print()
        llmkg.pp_shortest_paths_to_diseases(_args.src_node)

    else:

        raise NotImplementedError(f"Command not implemented: {_args.subcmd}")

    # /

    print('\nTotal Run time =', datetime.now() - start_time)
    print()
