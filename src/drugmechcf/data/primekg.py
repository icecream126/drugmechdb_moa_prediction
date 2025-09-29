"""
PrimeKG as a networkx Graph
"""

from collections import defaultdict, Counter
import dataclasses
from functools import partial
import itertools
import json
import os.path
import pickle
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union

import networkx as nx
import numpy as np
import pandas as pd

from drugmechcf.data.mondo import load_mondo

from drugmechcf.utils.misc import ValidatedDataclass, NO_DEFAULT, reset_df_index
from drugmechcf.utils.projconfig import get_project_config
from drugmechcf.utils.prettytable import pp_seq_key_count


# -----------------------------------------------------------------------------
#   Globals
# -----------------------------------------------------------------------------

# KG_COLUMNS = [
#     'relation', 'display_relation', 'x_index', 'x_id', 'x_type', 'x_name', 'x_source',
#     'y_index', 'y_id', 'y_type', 'y_name', 'y_source'
# ]

# Mapping from {head/tail}-type mentioned in `relation`, to actual node type names
RELN_NODE_TYPES_DICT = {
    'anatomy': 'anatomy',
    'bioprocess': 'biological_process',
    'cellcomp': 'cellular_component',
    'disease': 'disease',
    'drug': 'drug',
    'effect': 'effect/phenotype',
    'exposure': 'exposure',
    'molfunc': 'molecular_function',
    'pathway': 'pathway',
    'phenotype': 'effect/phenotype',
    'protein': 'gene/protein'
}

DRUG_ENTITY_TYPE = "drug"
DISEASE_ENTITY_TYPE = "disease"
PHENOTYPE_ENTITY_TYPE = "effect/phenotype"

RELATION_INDICATION = "indication"
RELATION_OFF_LABEL = "off-label use"
RELATION_CONTRA_INDICATION = "contraindication"
DRUG_DISEASE_RELATIONS = [RELATION_INDICATION, RELATION_OFF_LABEL, RELATION_CONTRA_INDICATION]


# -----------------------------------------------------------------------------
#   Classes
# -----------------------------------------------------------------------------

@dataclasses.dataclass
class PrimeKGOpts(ValidatedDataclass):
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

    mondo_opts_file: str = None
    """
    Path relative to `data_base_dir` for mondo options file.
    Default is to get a default mondo ontology.
    """

    cachefile: str = None
    """
    Where built PrimeKG is cached.
    In an options file, path is relative to `srcdir`. 
    """

    use_mondo_for_disease_disease_direction: bool = False
    """
    The 'disease_disease' relation is a "parent-child" relation, but PrimeKG also includes reverse direction
    edges (i.e. child-parent) in the same relation. When this option is `True`, Mondo ontology is consulted for
    the correct direction, and the reverse direction edges are marked as "rev-disease_disease".
    Use with care, as PrimeKG includes many diseases that are 'obsolete' in the current version of Mondo. 
    """

    def __post_init__(self):
        if self.data_base_dir is None:
            self.data_base_dir = get_project_config().get_input_data_dir()
        self.srcdir = os.path.join(self.data_base_dir, self.srcdir)
        if self.cachefile is not None:
            self.cachefile = os.path.join(self.srcdir, self.cachefile)
        if self.mondo_opts_file is not None:
            self.mondo_opts_file = os.path.join(self.data_base_dir, self.mondo_opts_file)
        return

    # noinspection PyMethodOverriding
    def assert_matches(self, other):
        return super().assert_matches(other, skip_fields = ["data_base_dir", "optsdir", "srcdir", "cachefile"])

    def to_json_file(self, json_file: str):
        # noinspection PyTypeChecker
        jdict = dataclasses.asdict(self)

        # Undo `__post_init__`
        jdict['cachefile'] = os.path.relpath(self.cachefile, start=self.srcdir)
        jdict['srcdir'] = os.path.relpath(self.srcdir, start=self.data_base_dir)

        if self.data_base_dir == get_project_config().get_input_data_dir():
            del jdict['data_base_dir']

        if self.mondo_opts_file is not None:
            jdict['mondo_opts_file'] = os.path.relpath(self.data_base_dir, start=self.mondo_opts_file)

        with open(json_file, "w") as jf:
            # noinspection PyTypeChecker
            json.dump(jdict, jf, indent=4)
        return

# /


class PrimeKG(nx.MultiDiGraph):
    """
    Directed Multi-Graph.
    Node: str                           ... Format '{source}:{id}', e.g. 'NCBI:63932'
        EntityType: str                 ... e.g. 'gene/protein'
        Index: int ... in [0, nbr_Nodes - 1]   ... e.g. 1
            Use `KGML.get_node_at(index)` to map index to Node name.

        names: List[str]                 ... e.g. ['STEEP1', 'CXorf56']
        primekg_indices: List[int]       ... e.g. [1027, 34863]
        mesh_xrefs: List[sr]             ... e.g. ['MESH:0800482']

    Edge: ... (head-Node, tail-Node)
        key: str = Relation-Name (i.e. Edge-type)           ... e.g. 'protein_protein'
        Data:
            Relation: str = Relation-Name (i.e. Edge-type)  ... e.g. 'protein_protein'
            name: str = 'display_name' from `kg.csv`        ... e.g. 'ppi'

    Note on relation names:
        + "kg.csv" contains canonical and reverse relations, both with the same name.
          For heterogeneous relations (head-type != tail-type), the reversed relation will get a prefix "rev-".
        + Many homogeneous relations (e.g. "disease_disease") capture a 'parent-child' relationship,
          which is also the `display_name` used. However, these relations also include the reverse relationship
          without that being reflected in the relation name or the display_name.
        + Other homogenous relations (e.g. "protein_protein") will have the same name in both directions.

    Build stats [June 21, 2024]:
    8,100,498 rows read from /Users/Sunil/Home/Projects/LangKG/Data/PrimeKG/kg.csv.gz

    nbr Diseases that are obsolete in Mondo = 1,292
    nbr Diseases with MeSH xrefs = 6,762
    nbr Disease MeSH ids xref'd  = 8,113

    PrimeKG built:
        nbr Nodes = 129,312
        nbr Edges = 8,100,128


    Examples:
        >>> primekg = load_primekg()
        >>> primekg.nodes['DrugBank:DB15444']
            {'EntityType': 'drug',
             'Index': 18841,
             'names': ['Elexacaftor'],
             'primekg_indices': [14499],
             'mesh_xrefs': None}
        >>>  primekg.nodes['MONDO:9061']
            {'EntityType': 'disease',
             'Index': 25601,
             'names': ['cystic fibrosis'],
             'primekg_indices': [28714],
             'mesh_xrefs': ['MESH:D003550']}
        >>> primekg.get_edge_data('DrugBank:DB15444', 'MONDO:9061')
            {'indication': {'Relation': 'indication', 'name': 'indication'}}

    """

    def __init__(self):
        super().__init__()

        self.opts: Optional[PrimeKGOpts] = None
        self._source = None
        self._cache_path = None

        # Node-type => List[ Node, ... ]
        self.nodetype2nodes: Dict[str, List[str]] = defaultdict(list)

        # Node.Index[int] => Node[str]. Index in [0, nNodes - 1]
        self.idx2node = []

        # Node.primekg_indices [int] => Node[str]
        #   ... for retrieving node given a PrimeKG index, e.g. x_index, y_index
        #       Note that on 63 occasions, the same node has 2 PrimeKG indices.
        self.primekg_index_to_node = []

        # relation => head_type => tail_type => edge_count
        self.relations_summary = defaultdict(partial(defaultdict, Counter))
        # relation => display_name => head_type => tail_type => edge_count
        self.relations_det_summary = defaultdict(partial(defaultdict,  partial(defaultdict, Counter)))

        # Disease MESH-id => List[Disease node]
        self.disease_mesh_to_nodes = defaultdict(list)

        return

    # ...........................................................................................
    # Loading and Saving
    # ...........................................................................................

    @staticmethod
    def load(opts: Union[str, PrimeKGOpts],
             rebuild: bool = False, verbose: bool = True) -> "PrimeKG":
        """
        Load from `opts.cachefile` or (re-)build and save to `opts.cachefile`.

        :param opts: Either KGMLOpts instance, or path to JSON file containing KGMLOpts
        :param rebuild: IF True THEN data is rebuilt and saved.
        :param verbose:
        :return: KGML instance
        """
        if isinstance(opts, str):
            opts = PrimeKGOpts.from_json_file(opts)

        if os.path.exists(opts.cachefile) and not rebuild:
            if verbose:
                print("Loading from cache:", opts.cachefile, "...", flush=True)

            with open(opts.cachefile, 'rb') as f:
                kg = pickle.load(f, fix_imports=False)

            opts.assert_matches(kg.opts)

            kg._cache_path = opts.cachefile

        else:
            kg = PrimeKG()
            # noinspection PyProtectedMember
            kg._build_and_save(opts)

        if verbose:
            print(flush=True)

        return kg

    def _build_and_save(self, opts: PrimeKGOpts, verbose: bool = True):
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
            # noinspection PyTypeChecker
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)

        if verbose:
            print(f"{self.__class__.__name__} instance cached to:", cachefile)
        return

    def _load_from_srcdata(self, verbose: bool = True):
        """
        Load data from source data files under `opts.srcdir`.
        """
        mondo = load_mondo(self.opts.mondo_opts_file, verbose=verbose)

        if verbose:
            print(f"Building {self.__class__.__name__} from data sources at:", self.opts.srcdir)

        df_kg = self.read_kg(self.opts.srcdir, verbose)

        #
        # Add Nodes from `df_kg`
        #

        if verbose:
            print("Adding nodes ...", flush=True)

        # ---
        node_ids_not_in_mondo = Counter()

        def get_mesh_xrefs(src_: str, nodeid_: str):
            if not src_.startswith("MONDO"):
                return None

            node_ids_ = self.build_mondo_ids(src_, nodeid_)
            mesh_xrefs = set()
            for nid_ in node_ids_:
                try:
                    mesh_xrefs.update(mondo.get_mesh_xrefs(nid_))
                except KeyError:
                    node_ids_not_in_mondo[nid_] += 1
                    continue

            return sorted(mesh_xrefs)
        # ---

        for row in df_kg.itertuples():
            # noinspection PyUnresolvedReferences
            src, nodeid, nodetype, name, primkg_idx = row.x_source, row.x_id, row.x_type, row.x_name, row.x_index
            node = self._build_node(src, nodeid)
            try:
                self.add_node(node, nodetype, name, primkg_idx, get_mesh_xrefs(src, nodeid))
            except TypeError as e:
                print(f"--- Error in: {src=}, {nodeid=}")
                raise e
            # noinspection PyUnresolvedReferences
            src, nodeid, nodetype, name, primkg_idx = row.y_source, row.y_id, row.y_type, row.y_name, row.y_index
            node = self._build_node(src, nodeid)
            self.add_node(node, nodetype, name, primkg_idx, get_mesh_xrefs(src, nodeid))

        #
        # Add Edges from `df_kg`
        #

        # ---
        def is_parent_of(head_src, head_id, tail_src, tail_id):
            head_mondo_ids = self.build_mondo_ids(head_src, head_id)
            tail_mondo_ids = self.build_mondo_ids(tail_src, tail_id)
            parents_of_tail = set()
            for tid in tail_mondo_ids:
                try:
                    parents_of_tail.update(mondo.get_parents(tid))
                except KeyError:
                    continue

            return len(set(head_mondo_ids) & parents_of_tail) > 0
        # ---

        def get_relation_name(row_):
            displ_relation = row_.display_relation
            ntypes = row_.relation.split("_")[:2]
            if len(ntypes) == 2:
                head_type, tail_type = [RELN_NODE_TYPES_DICT[t] for t in ntypes]
            else:
                # For 'contraindication', 'indication', 'off-label use'
                head_type, tail_type = 'drug', 'disease'

            if head_type == tail_type == "disease":

                if not self.opts.use_mondo_for_disease_disease_direction:
                    return row_.relation, displ_relation

                # Check parent/child reln in mondo
                if is_parent_of(row_.x_source, row_.x_id, row_.y_source, row_.y_id):
                    return row_.relation, displ_relation
                elif is_parent_of(row_.y_source, row_.y_id, row_.x_source, row_.x_id):
                    return "rev-" + row_.relation, "rev-" + displ_relation
                else:
                    raise ValueError(f"head={row_.x_source}:{row_.x_id},",
                                     f"tail={row_.y_source}:{row_.y_id} -- not recognized in Mondo.")

            elif head_type == row_.x_type and tail_type == row_.y_type:
                return row_.relation, displ_relation
            elif head_type == row_.y_type and tail_type == row_.x_type:
                return "rev-" + row_.relation, "rev-" + displ_relation
            else:
                raise ValueError(f"Unrecognized types in row: {row_}")
        # ---

        if verbose:
            print("Adding edges ...", flush=True)

        for row in df_kg.itertuples():
            # noinspection PyTypeChecker
            relation, display_relation = get_relation_name(row)
            # noinspection PyUnresolvedReferences
            self.add_edge(self._build_node(row.x_source, row.x_id),
                          self._build_node(row.y_source, row.y_id),
                          relation, display_relation)

        if verbose:
            print()
            if mondo:
                if node_ids_not_in_mondo:
                    print()
                    print("Diseases not recognized in Mondo:")
                    pp_seq_key_count(node_ids_not_in_mondo.most_common(), add_index=True, add_total=True)
                    print()

                n_obsolete_diseases = sum([any(mondo.nodes[mondoid]["is_obsolete"]
                                               for mondoid in self.build_mondo_ids(nd))
                                           for nd in self.get_nodes_for_types(DISEASE_ENTITY_TYPE)
                                           ])
                print(f"nbr Diseases that are obsolete in Mondo = {n_obsolete_diseases:,d}")
                n_dis = len([nd for nd in self.get_nodes_for_types(DISEASE_ENTITY_TYPE)
                             if self.nodes[nd]["mesh_xrefs"]])
                print(f"nbr Diseases with MeSH xrefs = {n_dis:,d}")
                print(f"nbr Disease MeSH ids xref'd  = {len(self.disease_mesh_to_nodes):,d}")
                print()

            print(f"{self.__class__.__name__} built:",
                  f"    nbr Nodes = {self.number_of_nodes():,d}",
                  f"    nbr Edges = {self.number_of_edges():,d}",
                  sep="\n")
            print()

        return

    # noinspection PyUnresolvedReferences
    @staticmethod
    def read_kg(srcdir, verbose=True) -> pd.DataFrame:

        fpath = os.path.join(srcdir, "kg.csv.gz")

        # These cols have a mix of types: int, str. Convert them to str.
        df_kg = pd.read_csv(fpath, dtype={'x_id': str, 'y_id': str})

        if verbose:
            print(f"{df_kg.shape[0]:,d} rows read from", fpath)

        return df_kg

    @staticmethod
    def _build_node(source, nodeid):
        return f"{source}:{nodeid}"

    @staticmethod
    def build_mondo_ids(source_or_node: str, nodeid: str = None):
        assert source_or_node.startswith("MONDO")
        if nodeid is None:
            _, nodeid = source_or_node.split(":")

        return [f"MONDO:{int(nid):07d}" for nid in nodeid.split("_")]

    def _add_to_primekg_index(self, index, node):
        if index > len(self.primekg_index_to_node) - 1:
            self.primekg_index_to_node.extend([None] * (index - len(self.primekg_index_to_node) + 1))
        self.primekg_index_to_node[index] = node
        return

    # ...........................................................................................
    # Create new graph from this graph
    # ...........................................................................................

    def create_subgraph(self, nodes: Iterable[str], opts: PrimeKGOpts, save_to_cache: bool = True):
        """
        Create new KGML graph that is a sub-graph of self based on `nodes`.

        :param nodes: Nodes (node-names) from self to include in new graph.
        :param opts: Options for new graph.
        :param save_to_cache: Whether to save to opts.cachefile
        """
        newkg = PrimeKG()
        newkg.opts = opts
        for nodenm in nodes:
            newkg.add_node(nodenm, **self.nodes[nodenm])

        for h, t, edata in self.edges:
            try:
                newkg.add_edge(h, t, edata['Relation'], edata['name'])
            except KeyError:
                # ignore if head / tail nodes not in `newkg`
                continue

        if save_to_cache and newkg.opts.cachefile is not None:
            newkg.save_to(newkg.opts.cachefile)

        return newkg

    # ...........................................................................................
    # Inherited methods for building KG
    # ...........................................................................................

    # noinspection PyMethodOverriding,PyPep8Naming
    def add_node(self,
                 node_for_adding: str,
                 EntityType: str,
                 names: Union[str, List[str]],
                 primekg_indices: Union[int, List[int]],
                 mesh_xrefs: List[str] = None
                 ):

        if isinstance(names, str):
            names = [names]
        if isinstance(primekg_indices, int):
            primekg_indices = [primekg_indices]

        if (node := self.nodes.get(node_for_adding)) is not None:
            for name in names:
                if name not in node["names"]:
                    node["names"].append(name)
            for primekg_index in primekg_indices:
                if primekg_index not in node["primekg_indices"]:
                    node["primekg_indices"].append(primekg_index)
                    self._add_to_primekg_index(primekg_index, node_for_adding)
            return

        node_idx = len(self.idx2node)
        self.idx2node.append(node_for_adding)
        self.nodetype2nodes[EntityType].append(node_for_adding)

        for primekg_index in primekg_indices:
            self._add_to_primekg_index(primekg_index, node_for_adding)

        if EntityType == DISEASE_ENTITY_TYPE and mesh_xrefs is not None:
            for mesh_id in mesh_xrefs:
                self.disease_mesh_to_nodes[mesh_id].append(node_for_adding)

        super().add_node(node_for_adding, EntityType=EntityType, Index=node_idx,
                         names=names, primekg_indices=primekg_indices, mesh_xrefs=mesh_xrefs)
        return

    # noinspection PyMethodOverriding
    def add_edge(self, head: str, tail: str, relation: str, display_name: str):
        """
        Adds an edge of type `relation` from `head` to `tail`.
        The nodes MUST exist in the KG, else raises KeyError.

        :param head:
        :param tail:
        :param relation:
        :param display_name:
        """
        # These will raise KeyError if head or tail do not already exist.
        head_type = self.nodes[head]['EntityType']
        tail_type = self.nodes[tail]['EntityType']

        self.relations_summary[relation][head_type][tail_type] += 1
        self.relations_det_summary[relation][display_name][head_type][tail_type] += 1

        super().add_edge(head, tail, key=relation, Relation=relation, name=display_name)
        return

    # ...........................................................................................
    # Node, Edge, Relation and their counts
    # ...........................................................................................

    def get_entity_types(self) -> List[str]:
        return list(self.nodetype2nodes.keys())

    def get_entity_type(self, node: str) -> str:
        return self.nodes[node]["EntityType"]

    def get_all_node_names(self, node: str) -> list[str]:
        return self.nodes[node]["names"]

    def get_node_name(self, node: str) -> str:
        return self.get_all_node_names(node)[0]

    def get_qualified_node_name(self, node) -> str:
        return f"{self.get_entity_type(node)}: {self.get_node_name(node)} ({node})"

    def get_node_at(self, index: int) -> str:
        return self.idx2node[index]

    def get_node_at_primekg_index(self, index: int) -> str:
        return self.primekg_index_to_node[index]

    def get_node_index(self, node: str) -> int:
        return self.nodes[node]["Index"]

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

        return sum(len(self.get_nodes_for_types(ntype)) for ntype in node_types)

    def get_all_relation_entity_types(self, relation: str) -> Tuple[List[str], List[str]]:
        head_types = list(self.relations_summary[relation].keys())
        tail_types = list(set(itertools.chain.from_iterable(tinfo.keys()
                                                            for tinfo in self.relations_summary[relation].values())))
        return head_types, tail_types

    def get_all_relation_edges(self, relation: str) -> List[Tuple[str, str]]:
        """
        Returns List[(u, v), ...] of all edges (u, v) for `relation`, where u = head-node[str], v = tail-node[str].
        The edges are in arbitrary (but replicable) order.
        """
        # noinspection PyArgumentList
        edges = [(u, v) for u, v, k in self.edges(keys=True) if k == relation]
        return edges

    def get_nbr_relation_edges(self, relation: str) -> int:
        """
        Nbr. edges in KG with Relation = `relation`
        :param relation: str
        :return:
        """
        if (rinfo := self.relations_summary.get(relation)) is None:
            return 0
        return sum(cnt for tinfo in rinfo.values() for cnt in tinfo.values())

    def get_edge_counts(self, relation: str = None, head_type: str = None, tail_type: str = None,
                        with_display_name=False) -> pd.DataFrame:
        """
        Returns DataFrame with cols: relation, htype, ttype, count.
        IF `with_display_name` THEN cols: relation, display_name, htype, ttype, count.
        Data is sorted in descending order on 'count'.
        """
        # Probably a hacky way to do this
        if with_display_name:
            df = pd.DataFrame.from_records([(r, dr, h, t, n)
                                            for r, dinfo in self.relations_det_summary.items()
                                            for dr, hinfo in dinfo.items()
                                            for h, tinfo in hinfo.items()
                                            for t, n in tinfo.items()],
                                           columns='relation display_name htype ttype count'.split())
        else:
            df = pd.DataFrame.from_records([(r, h, t, n)
                                                for r, hinfo in self.relations_summary.items()
                                                    for h, tinfo in hinfo.items()
                                                        for t, n in tinfo.items()],
                                           columns='relation htype ttype count'.split())

        mask = (df.relation == relation) if relation is not None else (df.relation != '')
        if head_type is not None:
            mask &= df.htype == head_type
        if tail_type is not None:
            mask &= df.ttype == tail_type

        # Return sorted in descending order on 'count', re-number rows
        counts_df = reset_df_index(df[mask].sort_values(['count', 'relation'], ascending=[False, True]))

        return counts_df

    # ...........................................................................................
    # Methods: Special purpose
    # ...........................................................................................

    def get_disease_neighbors(self,
                              dis_node: str,
                              neighbor_types: List[str] = None
                              ) \
            -> Tuple[Dict[str, Set[str]], Dict[str, Set[str]]]:
        """
        Retrieves neighbors of disease node.
        :param dis_node:
        :param neighbor_types: If specified, Only these types allowed in non-Drug neighbors
        :return:
            - Dict of { Ent-type => List[ Neighbor-node[str] ] }            ... for non-Drug neighbors
            - Dict of { Drug_Disease_relation => List[ Drug_node[str] ] }   ... for Drug neighbors
        """
        neighbors = defaultdict(set)
        drug_neighbors = defaultdict(set)

        if self.get_entity_type(dis_node) != DISEASE_ENTITY_TYPE:
            return neighbors, drug_neighbors

        # noinspection PyArgumentList
        for head, tail, reln in self.in_edges(dis_node, keys=True):
            if reln.startswith("rev-"):
                continue

            ntype = self.get_entity_type(head)

            if reln in DRUG_DISEASE_RELATIONS:
                drug_neighbors[reln].add(head)
            elif not neighbor_types or ntype in neighbor_types:
                neighbors[ntype].add(head)

        # noinspection PyArgumentList
        for head, tail, reln in self.out_edges(dis_node, keys=True):
            if reln.startswith("rev-"):
                continue
            ntype = self.get_entity_type(tail)

            if not neighbor_types or ntype in neighbor_types:
                neighbors[ntype].add(tail)

        return neighbors, drug_neighbors

    def get_negative_drug_disease_samples(self, count: int,
                                          rgenr: np.random.Generator = None,
                                          seed: int = None,
                                          max_rejects: int = 20
                                          ) -> List[Tuple[str, str]]:
        """
        Returns List[ (Drug-node, Disease-node), ...] that can be used as -ive samples.
        They are guranteed not to participate in any of the 3 `DRUG_DISEASE_RELATIONS` relations.

        :param count: Max count of pairs to return
        :param rgenr:
        :param seed:
        :param max_rejects: Max nbr times to reject a random sample.
            In each iteration, A random sample is generated.
            If the random sample particiaptes in any of the 3 Drug-Disease relations, it is rejected.
                ELSE it is added to the collected negative samples.
            IF nbr rejections reaches `max_rejects`, iterations are stopped.

        """
        if rgenr is None:
            rgenr = np.random.default_rng(seed)

        drug_nodes = np.asarray(self.get_nodes_for_types(DRUG_ENTITY_TYPE))
        disease_nodes = np.asarray(self.get_nodes_for_types(DISEASE_ENTITY_TYPE))

        neg_samples = []

        n_rejects = 0

        while len(neg_samples) < count:
            drug_choice = rgenr.choice(drug_nodes, size=1, replace=False, shuffle=False)
            disease_choice = rgenr.choice(disease_nodes, size=1, replace=False, shuffle=False)
            # Extract the choice, and convert from numpy.str_ to str, just in case
            drug = str(drug_choice[0])
            disease = str(disease_choice[0])

            rejected = False
            for reln in DRUG_DISEASE_RELATIONS:
                if self.has_edge(drug, disease, key=reln):
                    rejected = True
                    break

            if rejected:
                n_rejects += 1
                if n_rejects >= max_rejects:
                    break
                else:
                    continue
            else:
                neg_samples.append((drug, disease))

        return neg_samples

    def find_nodes(self, name_substr: str):
        print(f"Searching for nodes containing: '{name_substr}'")
        name_lc = name_substr.casefold()
        n_found = 0
        for node, data in self.nodes(data=True):
            cn = 0
            for nd_name in data["names"]:
                cn += 1
                if name_lc in nd_name.casefold():
                    n_found += 1
                    sfx = ""
                    if cn > 1:
                        sfx = f" (matched on: '{nd_name}')"
                    print(f"{n_found:3d}: ", self.get_qualified_node_name(node), sfx, sep="")
                    break

        if n_found > 0:
            print()

        print(f"   ... {n_found} matching nodes found.")
        return

    # ...........................................................................................
    #   Methods for pretty-printing
    # ...........................................................................................

    def pp_edge(self, head, tail, reln, rdata=None, indent: str = "", stream=None):
        if indent:
            print(indent, end="", file=stream)

        if rdata:
            reln += f" ({rdata['name']})"

        print(self.get_qualified_node_name(head), reln, self.get_qualified_node_name(tail),
              sep=" | ", file=stream)

        return

    def pp_edges(self, node: str):
        print("In-edges to node:", self.get_qualified_node_name(node))
        n_in = 0
        # noinspection PyArgumentList
        for u, v, reln, rdata in self.in_edges(nbunch=node, keys=True, data=True):
            n_in += 1
            self.pp_edge(u, v, reln, rdata, indent=f"  [{n_in:2d}] ")
        if n_in == 0:
            print("    No in-coming edges.")
        print()

        print("Out-edges from node:", self.get_qualified_node_name(node))
        n_out = 0
        # noinspection PyArgumentList
        for u, v, reln, rdata in self.out_edges(nbunch=node, keys=True, data=True):
            n_out += 1
            self.pp_edge(u, v, reln, rdata, indent=f"  [{n_out:2d}] ")
        if n_out == 0:
            print("    No out-going edges.")
        print()
        return
# /


# -----------------------------------------------------------------------------
#   Functions
# -----------------------------------------------------------------------------


def show_nonuniq(df: pd.DataFrame, key_cols: List[str], val_cols: List[str], return_mapping: bool = False):
    """
    Checks whether df[key_cols] maps to unique value of df[val_cols].
    Will print all the non-unique mappings.

    Example:
    >>> dfkg = pd.read_csv('kg.csv.gz')
    >>> show_nonuniq(dfkg, ['x_source', 'x_id'], ['x_name'])

    Tests on PrimeKG/kg.csv show:
        x_index (129,375 values) -- maps uniquely to -- x_name

        ['x_source', 'x_id'] (129,312 values) -- non-uniq to -- x_index
                                              -- non-uniq to -- x_name

        ... 63 entries of ['x_source', 'x_id'] map to multiple names.
            61 are for NCBI gene, and they map to alrentative names.
                e.g. https://www.ncbi.nlm.nih.gov/gene/?term=63932%5Buid%5D
             2 are for UBERON (Uber-anatomy ontology), also mapping to alternative names.
                e.g. UBERON:0000468, UBERON:0000992

    :param df:
    :param key_cols:
    :param val_cols:
    :param return_mapping: Whether to return the Key-Value mapping. Default is no return.
    """
    vals_dict = defaultdict(set)
    n_kcols = len(key_cols)

    for row in df[key_cols + val_cols].itertuples():
        # row is a tuple, row[0] is the index value
        key = row[1 : 1 + n_kcols]
        val = row[1 + n_kcols:]
        vals_dict[key].add(val)

    n_nonu = 0
    for k, v in vals_dict.items():
        if len(v) > 1:
            n_nonu += 1
            print(f'{k} = {v}')
    print()
    print("Key cols =", key_cols)
    print("Val cols =", val_cols)
    print()
    print(f'Nbr uniq keys = {len(vals_dict):,d}')
    print(f'Nbr keys w multiple vals = {n_nonu}')

    if return_mapping:
        return vals_dict
    return


def load_primekg(opts: str = None, rebuild: bool = False, verbose=True):
    # Force local import, to avoid Pickle errors during loading like:
    # AttributeError: Can't get attribute 'PrimeKG' on <module 'exps.primekg...' from .../exps/.../trainbase.py'>
    # noinspection PyUnresolvedReferences
    from drugmechcf.data.primekg import PrimeKGOpts, PrimeKG

    if opts is None:
        opts = PrimeKGOpts(srcdir="PrimeKG", cachefile="primekg.pkl")

    return PrimeKG.load(opts, rebuild=rebuild, verbose=verbose)


# ======================================================================================================
#   Main
# ======================================================================================================

# To run
# ------
# [Python]$ python -m drugmechcf.data.primekg build [-r] [ PrimeKGOpts-File ]

if __name__ == "__main__":

    import argparse
    from datetime import datetime
    from drugmechcf.utils.misc import print_cmd

    _argparser = argparse.ArgumentParser(
        description='PrimeKG data',
    )

    _subparsers = _argparser.add_subparsers(dest='subcmd',
                                            title='Available commands',
                                            )
    # Make the sub-commands required
    _subparsers.required = True

    # ... build
    _sub_cmd_parser = _subparsers.add_parser('build',
                                             help="Build the KG and save to cache.")
    _sub_cmd_parser.add_argument('-r', '--rebuild', action='store_true',
                                 help="Forced rebuild of the PrimeKG graph.")
    _sub_cmd_parser.add_argument('primekgopts', type=str, nargs='?', default=None,
                                 help="Path to KGML options file.")

    # ...

    _args = _argparser.parse_args()
    # .................................................................................................

    start_time = datetime.now()

    print("---------------------------------------------------------------------")
    print_cmd()

    if _args.subcmd == 'build':

        load_primekg(_args.primekgopts, rebuild=_args.rebuild)

    else:

        raise NotImplementedError(f"Command not implemented: {_args.subcmd}")

    # /

    print('\nTotal Run time =', datetime.now() - start_time)
    print()
