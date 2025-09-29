"""
DrugMechDB
ref: https://github.com/SuLab/DrugMechDB
"""

from collections import defaultdict, Counter

import networkx as nx
from yaml import safe_load
import os
import pickle
from typing import Dict, List, Set, Tuple

import pandas as pd

from drugmechcf.utils.misc import reset_df_index
from drugmechcf.utils.projconfig import get_project_config

from drugmechcf.data.moagraph import MoaGraph


# -----------------------------------------------------------------------------
#   Globals
# -----------------------------------------------------------------------------

CHEMICAL_ENTITY_TYPE = "ChemicalSubstance"
DISEASE_ENTITY_TYPE = "Disease"
DRUG_ENTITY_TYPE = "Drug"
GENE_ENTITY_TYPE = "GeneFamily"
PHENOTYPE_ENTITY_TYPE = "PhenotypicFeature"
PROTEIN_ENTITY_TYPE = "Protein"


# -----------------------------------------------------------------------------
#   Classes
# -----------------------------------------------------------------------------


class DrugMechDB:
    """
    Content is a list of `MoaGraph` graphs.
    This class reflects the structure of DrugmechDB in that a Drug-Disease pair may have more than one MoAGraph.
    See `drugmechcf.data.drugmechdbuni.UnifiedDrugMechDB` for version that
     merges multiple MoAs for same Drug-Disease pair.

    Each graph `g` has attributes (access as `g.graph`):
        _id: str. Graph-id from DrugMechDB. 	... e.g. 'DB08949_MESH_D010383_1'
        disease: str. Disease name. 	        ... e.g. 'Pellagra'
        disease_mesh: str. MeSH id of disease. 	... e.g. 'MESH:D010383'
            ... This is the target Disease node
        drug: str. Drug name. 	                ... e.g. 'Inositol Niacinate'
        drugbank: str. Drug DrugBank id.	    ... e.g. 'DB:DB08949'
        drug_mesh: str. Drug MeSH id.	        ... e.g. 'MESH:C005193'
        drug_id: str.                           ... e.g. 'MESH:C005193'
            ... Only reliable way to get the Drug ID for the graph.

    The Drug and Disease end-point nodes of each graph are in the attributes:
        disease_node = moa.graph['disease_mesh']
        drug_node = moa.graph['drug_id']

    Each node in a graph is identified by its Ontology id, 	... e.g. 'MESH:C005193'
    and has attributes:
        EntityType: str. 	... e.g. 'Drug'
        name: str. 	        ... e.g. 'Inositol Niacinate'

    Each edge in a graph has `key` = Relation-name[str],    ... e.g. 'negatively correlated with'
    and has attribute:
        Relation: str = Relation-name.

    Example output of `drugmechdb = DrugMechDB()`:
    DrugMechDB built:
        nbr indications = 4,846
        nbr unique diseases = 744
        nbr unique drugs = 1,612
        nbr unique Drug-Disease pairs = 4,664
        nbr Drug-Disease pairs with multiple MoAs =   141
            max nbr repeats = 8
            (n_moas, n_drug_disease_pairs) = (2, 118); (3, 14); (4, 3); (5, 5); (8, 1)

    There are a total of 33k nodes of 14 types:
        BiologicalProcess: 7,987
        Protein: 7,080
        Disease: 4,965
        Drug: 4,924
        ChemicalSubstance: 2,249
        PhenotypicFeature: 1,505
        OrganismTaxon: 1,347
        GeneFamily: 1,028
        GrossAnatomicalStructure: 518
        MolecularActivity: 453
        CellularComponent: 444
        Pathway: 306
        Cell: 184
        MacromolecularComplex: 4

    There are a total of 32k edges of 66 types.
    Most common edges are:
        'positively regulates'
        'positively correlated with'
        'decreases activity of'
        'increases activity of'
        'negatively regulates'

    Looking at moa graph data:
        >>> drugmechdb = DrugMechDB()
        >>> moa = drugmechdb.indication_graphs[0]
        >>> moa.graph
            {'_id': 'DB00619_MESH_D015464_1',
             'disease': 'CML (ph+)',
             'disease_mesh': 'MESH:D015464',
             'drug': 'imatinib',
             'drug_mesh': 'MESH:D000068877',
             'drugbank': 'DB:DB00619',
             'drug_id': 'MESH:D000068877'}

    Raw files this object is built from:
        indication_paths.yaml
        deprecated_ids.txt

    """

    def __init__(self):
        self.indication_graphs: List[MoaGraph] = []

        # Indices into `self.indication_graphs`
        #
        # disease_node[str] => Set[int]
        self.disease_gidxs = defaultdict(set)
        # drug_node[str] => Set[int]
        self.drug_gidxs = defaultdict(set)
        # node.EntityType[str] => Set[int]
        self.nodetype_gidxs = defaultdict(set)
        #
        self.moa_id_to_gidx = dict()

        self._cache_path = None

        return

    def add_indication_graph(self, drug_node: str, disease_node: str, graph: MoaGraph) -> int:
        """
        Adds `graph` from `drug_node` to `disease_node` as an indication graph,
        updating local indices.

        :param drug_node:
        :param disease_node:
        :param graph:

        :return: Index of `graph` in the updated index.
        """

        gidx = len(self.indication_graphs)

        self.indication_graphs.append(graph)

        self.drug_gidxs[drug_node].add(gidx)
        self.disease_gidxs[disease_node].add(gidx)

        # Update node_type index
        for ntype in graph.get_entity_types():
            self.nodetype_gidxs[ntype].add(gidx)

        self.moa_id_to_gidx[graph.graph["_id"]] = gidx

        return gidx

    def nbr_indications(self) -> int:
        return len(self.indication_graphs)

    def get_indication_graph_with_id(self, moa_id: str):
        return self.indication_graphs[ self.moa_id_to_gidx[moa_id] ]

    def get_indication_graphs(self, drug_id: str = None, disease_id: str = None) -> List[MoaGraph]:
        if drug_id is None and disease_id is None:
            return self.indication_graphs

        gidxs = set()
        if drug_id is not None:
            gidxs = self.drug_gidxs.get(drug_id, set())

        if disease_id is not None:
            dis_gidxs = self.disease_gidxs.get(disease_id, set())
            if gidxs:
                gidxs = dis_gidxs & gidxs
            else:
                gidxs = dis_gidxs

        return [self.indication_graphs[gi] for gi in gidxs]

    def get_entity_types(self) -> List[str]:
        return list(self.nodetype_gidxs.keys())

    # ...........................................................................................
    # Moa-based properties
    # ...........................................................................................

    def get_drug_name(self, drug_id: str) -> str:
        """
        Get the node's name from one of the MoA's for that node.
        """
        moa = self.get_indication_graphs(drug_id=drug_id)[0]
        return self.get_node_name(moa, drug_id)

    def get_disease_name(self, disease_id: str) -> str:
        """
        Get the node's name from one of the MoA's for that node.
        """
        moa = self.get_indication_graphs(disease_id=disease_id)[0]
        return self.get_node_name(moa, disease_id)

    @staticmethod
    def get_entity_type(moa: MoaGraph, node: str) -> str:
        return moa.get_node_entity_type(node)

    @staticmethod
    def get_node_name(moa: MoaGraph, node: str) -> str:
        return moa.get_node_name(node)

    @staticmethod
    def get_moa_id(moa: MoaGraph) -> str:
        return moa.graph["_id"]

    @staticmethod
    def get_moa_source_drug_node(moa: MoaGraph) -> str:
        return moa.graph["drug_id"]

    @staticmethod
    def get_moa_target_disease_node(moa: MoaGraph) -> str:
        return moa.graph["disease_mesh"]

    @staticmethod
    def get_moa_drug_disease_nodes(moa: MoaGraph) -> Tuple[str, str]:
        return moa.graph["drug_id"], moa.graph["disease_mesh"]

    # ...........................................................................................
    # Load, save
    # ...........................................................................................

    @staticmethod
    def load(src_dirname: str = "DrugMechDB",
             cache_file_name: str = None,
             rebuild: bool = False,
             verbose: bool = True) -> "DrugMechDB":

        base_dir = os.path.join(get_project_config().get_input_data_dir(), src_dirname)

        cache_file = os.path.join(base_dir, cache_file_name)

        if os.path.exists(cache_file) and not rebuild:
            return DrugMechDB.load_from_cache(cache_file, verbose=verbose)

        return DrugMechDB.build_and_save(src_dirname, cache_file_name=cache_file_name, verbose=verbose)

    @staticmethod
    def load_from_cache(cache_file: str, verbose=True) -> "DrugMechDB":
        if verbose:
            print("Loading from cache:", cache_file, "...", flush=True)

        with open(cache_file, 'rb') as f:
            drugmechdb = pickle.load(f, fix_imports=False)

        drugmechdb._cache_path = cache_file

        if verbose:
            print(flush=True)

        return drugmechdb

    @staticmethod
    def build_and_save(src_dirname: str = "DrugMechDB",
                       cache_file_name: str = None,
                       verbose=True
                       ) -> "DrugMechDB":

        dmdb = DrugMechDB()

        base_dir = os.path.join(get_project_config().get_input_data_dir(), src_dirname)

        cache_file = None
        if cache_file_name is not None:
            cache_file = os.path.join(base_dir, cache_file_name)

        if verbose:
            print("Loading data from", base_dir, "...", flush=True)

        indications_path =  os.path.join(base_dir, "indication_paths.yaml")
        with open(indications_path) as f:
            indications = safe_load(f)

        deprecated_ids_path =  os.path.join(base_dir, "deprecated_ids.txt")
        deprecated_ids = pd.read_csv(deprecated_ids_path, header=None)[0].tolist()

        for indic in indications:
            if indic['graph']['_id'] in deprecated_ids:
                continue

            g = MoaGraph(**indic['graph'])

            for ndata in indic["nodes"]:
                g.add_node(ndata['id'], EntityType=ndata['label'], name=ndata['name'])
            for link in indic['links']:
                g.add_edge(link['source'], link['target'], key=link['key'], Relation=link['key'])

            # Most MoA drugs have a MeSH ID
            drug_node = indic['graph']['drug_mesh']
            if not drug_node or "," in drug_node:
                # ~50 of them have only a DrugBank ID. Some have multiple MeSH IDs.
                drug_node = indic['graph']['drugbank']

            # This ensures a consistent way to retrieve the Drug ID
            g.graph["drug_id"] = drug_node

            #
            # Now fix the data ... errors found on manual inspection
            #

            # ... the Drug Node is the one with in-degree == 0. Mark the correct node for the MoA graph.
            for nd, ntype in g.nodes(data="EntityType"):
                # noinspection PyCallingNonCallable
                if g.in_degree(nd) == 0:
                    g.graph["drug_id"] = drug_node = nd
                    if ntype not in ["Drug", "ChemicalSubstance"]:
                        g.change_node_entity_type(nd, "ChemicalSubstance")

                    break

            # ... The Disease node is the one with out-degree == 0. Make sure its type is "Disease".
            for nd, ndata in g.nodes(data=True):
                ntype = ndata["EntityType"]
                # noinspection PyCallingNonCallable
                if g.out_degree(nd) == 0:
                    if ntype != "Disease":
                        g.change_node_entity_type(nd, "Disease")
                    # Fix name ... occurs 1 time
                    if ndata["name"].startswith("MESH"):
                        g.change_node_name(nd, g.graph["disease"])
                    # Occurs 1 time
                    if g.graph['disease'] == 'dantron':
                        g.graph['disease'] = ndata["name"]

            # Set the root and sink nodes for the MoA graph
            g.set_root_node(g.graph["drug_id"])
            g.set_sink_node(g.graph["disease_mesh"])

            # Set the graph-name, now that all data has been fixed
            g.set_graph_name(f'DrugMechDB.MoA: {g.get_node_name(g.get_root_node())} treats '
                             + g.get_node_name(g.get_sink_node()))

            dmdb.add_indication_graph(drug_node, disease_node=indic['graph']['disease_mesh'], graph=g)

        if cache_file is not None:
            dmdb._save_to(cache_file, verbose=verbose)

        if verbose:
            print(f"{dmdb.__class__.__name__} built:")
            print(f"    nbr indications = {len(dmdb.indication_graphs):,d}")
            print(f"    nbr unique diseases = {len(dmdb.disease_gidxs):,d}")
            print(f"    nbr unique drugs = {len(dmdb.drug_gidxs):,d}")
            print()

        return dmdb

    def _save_to(self, cachefile: str, verbose: bool = True):
        if verbose:
            print("Saving to cache ...", flush=True)

        self._cache_path = cachefile
        with open(cachefile, 'wb') as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)

        if verbose:
            print(f"{self.__class__.__name__} instance cached to:", cachefile)
        return

    # ...........................................................................................
    # Special purpose
    # ...........................................................................................

    def get_all_drug_connections(self, drug_id: str = None) -> pd.DataFrame:
        """
        :return: DataFrame of unique rows, with cols (all `str`):
            DrugId, DrugName, Relation, NeighborType, NeighborId, NeighborName, DiseaseId, DiseaseName
            ... representing a path going Out from Drug to Disease:
                DrugId -> NeighborId --- -> DiseaseId
        """

        records = []
        drug_nodes = [drug_id] if drug_id is not None else self.drug_gidxs
        for drug_nd in drug_nodes:
            for moa in self.get_indication_graphs(drug_id=drug_nd):
                drug_name = self.get_node_name(moa, drug_nd)
                disease_id = self.get_moa_target_disease_node(moa)

                for _, nghbr, reln in moa.out_edges(nbunch=drug_nd, keys=True):
                    records.append((drug_nd, drug_name, reln,
                                    self.get_entity_type(moa, nghbr), nghbr, self.get_node_name(moa, nghbr),
                                    disease_id, self.get_node_name(moa, disease_id)))

                # In case there are any in-edges to the Drug (not expected).
                for nghbr, _, reln in moa.in_edges(nbunch=drug_nd, keys=True):
                    records.append((drug_nd, drug_name, reln,
                                    self.get_entity_type(moa, nghbr), nghbr, self.get_node_name(moa, nghbr),
                                    disease_id, self.get_node_name(moa, disease_id)))

        df = pd.DataFrame.from_records(records, columns=["DrugId", "DrugName", "Relation",
                                                         "NeighborType", "NeighborId", "NeighborName",
                                                         "DiseaseId", "DiseaseName"])
        df = df.drop_duplicates()
        return df

    def get_all_disease_connections(self, disease_id: str = None) -> pd.DataFrame:
        """
        :return: DataFrame of unique rows, with cols (all `str`):
            DiseaseId, DiseaseName, Relation, NeighborType, NeighborId, NeighborName, DrugId, DrugName
            ... representing a path coming In to Disease from Drug:
                DiseaseId <- NeighborId <- --- DrugId
        """

        records = []
        disease_nodes = [disease_id] if disease_id is not None else self.disease_gidxs
        for disease_nd in disease_nodes:
            for moa in self.get_indication_graphs(disease_id=disease_nd):
                disease_name = self.get_node_name(moa, disease_nd)
                drug_id = self.get_moa_source_drug_node(moa)

                for nghbr, _, reln in moa.in_edges(nbunch=disease_nd, keys=True):
                    records.append((disease_nd, disease_name, reln,
                                    self.get_entity_type(moa, nghbr), nghbr, self.get_node_name(moa, nghbr),
                                    drug_id, self.get_node_name(moa, drug_id)))

                # In case there are any out-edges from Disease (not expected).
                for _, nghbr, reln in moa.out_edges(nbunch=disease_nd, keys=True):
                    records.append((disease_nd, disease_name, reln,
                                    self.get_entity_type(moa, nghbr), nghbr, self.get_node_name(moa, nghbr),
                                    drug_id, self.get_node_name(moa, drug_id)))

        df = pd.DataFrame.from_records(records, columns=["DiseaseId", "DiseaseName", "Relation",
                                                         "NeighborType", "NeighborId", "NeighborName",
                                                         "DrugId", "DrugName"])
        df = df.drop_duplicates()
        return df

    def get_disease_neighbors(self, moa: MoaGraph, dis_node, neighbor_types: List[str] = None) \
            -> Tuple[Dict[str, Set[str]], Dict[str, Set[str]]]:
        """
        Retrieves neighbors of disease node in the `moa`.
        :param moa:
        :param dis_node:
        :param neighbor_types:
        :return:
            - Dict of { Ent-type => Set[ Neighbor-node[str] ] }            ... for non-Drug neighbors
            - Dict of { Drug_Disease_relation => Set[ Drug_node[str] ] }   ... for Drug neighbors
              includes special relation:
                "indication" => { DrugID }
        """
        neighbors = defaultdict(set)

        drug_neighbors = defaultdict(set)
        drug_neighbors["indication"].add(self.get_moa_source_drug_node(moa))

        if self.get_entity_type(moa, dis_node) != DISEASE_ENTITY_TYPE:
            return neighbors, drug_neighbors

        for head, tail, reln in moa.in_edges(dis_node, keys=True):
            ntype = self.get_entity_type(moa, head)

            if ntype == DRUG_ENTITY_TYPE:
                drug_neighbors[reln].add(head)
            elif not neighbor_types or ntype in neighbor_types:
                neighbors[ntype].add(head)

        for head, tail, reln in moa.out_edges(dis_node, keys=True):
            ntype = self.get_entity_type(moa, tail)

            if not neighbor_types or ntype in neighbor_types:
                neighbors[ntype].add(tail)

        return neighbors, drug_neighbors

    def get_edge_counts(self,
                        relation: str = None, head_type: str = None, tail_type: str = None,
                        unique_edges: bool = False
                        ) -> pd.DataFrame:
        """
        Returns DataFrame with cols: relation, htype, ttype, count.
        Data is sorted in descending order on 'count'.

        IF `unique_edges` THEN count is nbr of Unique Edges, unique by (reln, htype, head, ttype, tail)
        ELSE count is total nbr edges for (relation, htype, ttype)
        """

        if unique_edges:
            edge_counts = Counter((reln, self.get_entity_type(moa, h), h, self.get_entity_type(moa, t), t)
                                  for moa in self.indication_graphs
                                  for (h, t, reln) in moa.edges(keys=True)
                                  )

            df_u = pd.DataFrame.from_records([(r, ht, tt, n) for (r, ht, h, tt, t), n in edge_counts.items()],
                                             columns='relation htype ttype n_edges'.split())

            df = df_u.value_counts(['relation', 'htype', 'ttype']).reset_index()
            df.index += 1

        else:
            edge_counts = Counter((reln, self.get_entity_type(moa, h), self.get_entity_type(moa, t))
                                  for moa in self.indication_graphs
                                  for (h, t, reln) in moa.edges(keys=True)
                                  )

            df = pd.DataFrame.from_records([(r, ht, tt, n) for (r, ht, tt), n in edge_counts.items()],
                                           columns='relation htype ttype count'.split())

        mask = (df.relation == relation) if relation is not None else (df.relation != '')
        if head_type is not None:
            mask &= df.htype == head_type
        if tail_type is not None:
            mask &= df.ttype == tail_type

        # Return sorted in descending order on 'count', re-number rows
        counts_df = reset_df_index(df[mask].sort_values(['count', 'relation'], ascending=[False, True]))

        return counts_df

# /


def load_drugmechdb(src_dirname: str = "DrugMechDB", cache_file_name: str = "DrugMechDB.pkl",
                    rebuild: bool = False, verbose=True):
    # Force local import, to avoid Pickle errors during loading like:
    # AttributeError: Can't get attribute 'DrugMechDB' on <module 'exps.primekg...' from .../exps/.../trainbase.py'>
    # noinspection PyUnresolvedReferences
    from drugmechcf.data.drugmechdb import DrugMechDB

    return DrugMechDB.load(src_dirname=src_dirname, cache_file_name=cache_file_name, rebuild=rebuild, verbose=verbose)


def moa_examples(count: int = 5, min_length: int = 1, entity_types: List[str] = None):
    drugmechdb = load_drugmechdb()

    if entity_types:
        gidxs = drugmechdb.nodetype_gidxs[entity_types[0]]
        for et in entity_types[1:]:
            gidxs = gidxs & drugmechdb.nodetype_gidxs[et]
    else:
        gidxs = range(drugmechdb.nbr_indications())

    print()
    n_found = 0
    for gidx in gidxs:
        moa = drugmechdb.indication_graphs[gidx]
        if (splen := nx.shortest_path_length(moa, moa.get_root_node(), moa.get_sink_node())) >= min_length:
            n_found += 1
            print()
            print(f"--- [{n_found}] ---")
            moa.pprint()

            print("shortest-path-length =", splen)
            print("Entity types in graph =", ", ".join(sorted(moa.get_entity_types())))

            print()
            print("========================")

        if n_found >= count:
            break

    print()
    if n_found == 0:
        print("No paths found")
    else:
        print(n_found, "paths shown.")

    print()
    return


# ======================================================================================================
#   Main
# ======================================================================================================

# To run
# ------
# [Python]$ python -m drugmechcf.data.drugmechdb build [-r]

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

    # ... examples
    _sub_cmd_parser = _subparsers.add_parser('examples',
                                             help="Show example MoA graphs.")
    _sub_cmd_parser.add_argument('-c', '--count', type=int, default=5,
                                 help="Max nbr of examples.")
    _sub_cmd_parser.add_argument('-l', '--min_length', type=int, default=1,
                                 help="Min length of shortest Drug-to-Disease path.")
    _sub_cmd_parser.add_argument('entity_types', nargs="*",
                                 help="Restrict to MoA's with these entity types.")

    # ...

    _args = _argparser.parse_args()
    # .................................................................................................

    start_time = datetime.now()

    print("---------------------------------------------------------------------")
    print_cmd()

    if _args.subcmd == 'build':

        load_drugmechdb(rebuild=_args.rebuild)

    elif _args.subcmd == 'examples':

        moa_examples(count=_args.count, min_length=_args.min_length, entity_types=_args.entity_types)

    else:

        raise NotImplementedError(f"Command not implemented: {_args.subcmd}")

    # /

    print('\nTotal Run time =', datetime.now() - start_time)
    print()
