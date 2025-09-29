"""
Analysis of AddLink sessions
"""

import json
import os.path
from typing import Any, Dict, List, Tuple, Union

import networkx as nx
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import pandas as pd

from drugmechcf.data.drugmechdb import DrugMechDB, load_drugmechdb
from drugmechcf.llm.prompt_types import DrugDiseasePromptInfo

from drugmechcf.graphmatch.graphmatcher import GraphMatchScore
from drugmechcf.llm.test_common import is_minimal_match_cumi, is_minimal_target_moa

from drugmechcf.utils.misc import pp_funcargs, pp_underlined_hdg, pp_dict


# -----------------------------------------------------------------------------
#   Functions: Analyze
# -----------------------------------------------------------------------------


def analyze_session(session_json_file: str):
    pp_funcargs(analyze_session)

    drugmechdb = load_drugmechdb()
    print()

    with open(session_json_file) as jf:
        sesn_dict = json.load(jf)

    assert sesn_dict["args"]["query_type"] == "ADD_LINK"

    # bool: is_true_pos => [Tuple[int, float], ...]
    dist_diln_scores = dict(tp=[], fn=[])

    n_gt, n_gf = 0, 0
    n_tp, n_fn = 0, 0

    cum_metrics = None

    for session in sesn_dict["session"]:
        sesn_metrics = GraphMatchScore(**session["metrics"])
        cum_metrics = sesn_metrics.accumulate_metrics(cum_metrics)

        if not sesn_metrics.has_ref_graph:
            n_gf += 1
            continue
        else:
            n_gt += 1

        if not sesn_metrics.has_target_graph:
            n_fn += 1
        else:
            n_tp += 1

        prompt_info = DrugDiseasePromptInfo.from_serialized(session["prompt_info"])

        dist_diln = drug_prot_distance(prompt_info, drugmechdb)

        dist_diln_scores["tp" if sesn_metrics.has_target_graph else "fn"].append(dist_diln)
    # /

    print()
    pp_underlined_hdg("Metrics Summary")
    pprint_accumulated_metrics(cum_metrics, n_tested=n_gt + n_gf, n_valid_llm_response=n_tp + n_fn,
                               n_samples_skipped=-999, n_skipped_llm_response_inc=-999)

    print()
    pp_underlined_hdg("Summary of Drug -to- Source-node scores")

    tp_scores = np.asarray(dist_diln_scores["tp"])
    fn_scores = np.asarray(dist_diln_scores["fn"])

    print(f"Total nbr queries   = {n_gt + n_gf:5,d}")
    print(f"Nbr Ground-Trues    = {n_gt:5,d}")
    print(f"Nbr True-Positives  = {n_tp:5,d} ... {n_tp / n_gt:.1%}")
    print(f"Nbr False-Negatives = {n_fn:5,d} ... {n_fn / n_gt:.1%}")
    print()

    if n_tp > 0:
        mask_soruce_node_is_protein_tp = tp_scores[:, 0] != 0
        # noinspection PyUnresolvedReferences
        print(f"Nbr True-Pos source-node is Protein  = {mask_soruce_node_is_protein_tp.sum():3d}",
              f" ... {mask_soruce_node_is_protein_tp.sum() / n_tp:.1%}")
    else:
        mask_soruce_node_is_protein_tp = np.asarray([], dtype=np.bool_)

    if n_fn > 0:
        mask_soruce_node_is_protein_fn = fn_scores[:, 0] != 0
        # noinspection PyUnresolvedReferences
        print(f"Nbr False-Neg source-node is Protein = {mask_soruce_node_is_protein_fn.sum():3d}",
              f" ... {mask_soruce_node_is_protein_fn.sum() / n_fn:.1%}")
    else:
        mask_soruce_node_is_protein_fn = np.asarray([], dtype=np.bool_)

    if n_tp + n_fn > 0:
        print()

    # noinspection PyUnresolvedReferences
    if mask_soruce_node_is_protein_tp.sum() == 0 and mask_soruce_node_is_protein_fn.sum() == 0:
        return

    tp_scores = tp_scores[mask_soruce_node_is_protein_tp]
    fn_scores = fn_scores[mask_soruce_node_is_protein_fn]

    # ---
    def pp_dist_diln_summary(hdg, scores):
        print(hdg)
        print("  Shortest distance, Drug-Protein:")
        print("   ", f"range = [{np.amin(scores[:, 0])}, {np.amax(scores[:, 0])}], ",
              f"mean = {np.mean(scores[:, 0]):.3f}, median = {np.median(scores[:, 0]):.3f}")
        print("  Dilution of source-Protein node:")
        print("   ", f"range = [{np.amin(scores[:, 1]):.3f}, {np.amax(scores[:, 1]):.3f}], ",
              f"mean = {np.mean(scores[:, 1]):.3f}, median = {np.median(scores[:, 1]):.3f}")
        print()
        return
    # ---

    if tp_scores.shape[0] > 0:
        pp_dist_diln_summary("True-Positive scores:", tp_scores)

    if fn_scores.shape[0] > 0:
        pp_dist_diln_summary("False-Negative scores:", fn_scores)

    return


def drug_prot_distance(prompt_info: DrugDiseasePromptInfo, drugmechdb: DrugMechDB) -> Tuple[int, float]:
    """
    Calc the distance, dilution from source-Drug to source-Node in an AddLink query.

    :return:
        - int: Shortest distance from source-Drug to source-Node
        - float: Dilution = 1 / nbr of all nodes within that distance from source-Drug (not incl. source-Drug)
    """

    source_drug = prompt_info.drug_id
    source_node = prompt_info.edit_link_info.source_node

    if source_drug == source_node:
        return 0, 1.0

    source_moa = drugmechdb.get_indication_graph_with_id(prompt_info.edit_link_info.source_moa_id)
    shortest_distance = nx.shortest_path_length(source_moa, source=source_drug, target=source_node)

    all_nodes_at_that_distance = set(nx.dfs_preorder_nodes(source_moa, source=source_drug,
                                                           depth_limit=shortest_distance))

    # Subtract the source_node from the set
    dilution = 1 / (len(all_nodes_at_that_distance) - 1)

    return shortest_distance, dilution


# -----------------------------------------------------------------------------
#   Functions: Compare
# -----------------------------------------------------------------------------


def compare_sessions(session_json_1: str, session_json_2: str):

    pp_funcargs(compare_sessions)

    with open(session_json_1) as jf:
        sesn_dict_1 = json.load(jf)

    with open(session_json_2) as jf:
        sesn_dict_2 = json.load(jf)

    print()
    pp_underlined_hdg("Sessions Query Summary")

    name_1 = os.path.splitext(os.path.basename(session_json_1))[0]
    name_2 = os.path.splitext(os.path.basename(session_json_2))[0]

    qsummary_1 = get_session_query_summary(sesn_dict_1)
    qsummary_2 = get_session_query_summary(sesn_dict_2)

    df = pd.DataFrame.from_dict({k: [qsummary_1[k], qsummary_2[k]] for k in qsummary_1 if k != 'tested_samples'},
                                orient='index', columns=[name_1, name_2])

    print(df.to_markdown(intfmt=","))
    print("\n")

    pp_underlined_hdg("Comparison of Query Samples")

    s1_samples = qsummary_1["tested_samples"]
    s2_samples = qsummary_2["tested_samples"]

    s1_set = set(s1_samples)
    s2_set = set(s2_samples)
    common_set = s1_set & s2_set

    print("Ordered lists equal   =", s1_samples == s2_samples)
    print("Sets are equal        =", s1_set == s2_set)
    print("Nbr of common samples =", len(common_set))
    print("Extra samples in", name_1, "=", len(s1_set - common_set))
    print("Extra samples in", name_2, "=", len(s2_set - common_set))
    print()

    return


def get_session_query_summary(sesn_dict: Dict[str, Any] | str) -> Dict[str, Any]:
    if isinstance(sesn_dict, str):
        with open(sesn_dict) as jf:
            sesn_dict = json.load(jf)

    sesn_args = sesn_dict["args"]
    sesn_qdata = sesn_dict["session"]

    assert sesn_args["query_type"] == "ADD_LINK", "Session was not of type ADD_LINK!"

    tested_samples = extract_query_params(sesn_dict)

    qsummary = dict(AddLink_type="Drug-Protein" if sesn_args["source_node_is_drug"] else "Protein-Protein",
                    Randomized=sesn_args["random"],
                    nbrQueries=len(sesn_qdata),
                    tested_samples=tested_samples,
                    nbrUniqSamples=len(set(tested_samples)),
                    nbrUniqDrugs=len(set(drug for drug, src_nd, tgt_nd, disease in tested_samples)),
                    nbrUniqDiseases=len(set(disease for drug, src_nd, tgt_nd, disease in tested_samples)),
                    nbrUniqDrugDiseasePairs=len(set((drug, disease)
                                                    for drug, src_nd, tgt_nd, disease in tested_samples)),
                    )
    return qsummary


def extract_query_params(sesn_dict: Dict[str, Any]) -> List[Tuple[str, str, str, str]]:
    """
    Returns List: [(source-Drug, source-Node, target-Node, targt-Disease), ...]
    """

    tested_samples = []
    for qdata in sesn_dict["session"]:
        promp_info = qdata["prompt_info"]
        add_link_info = promp_info["add_link_info"]
        tested_samples.append((promp_info["drug_id"], add_link_info["source_node"],
                               add_link_info["target_node"], promp_info["disease_id"]))

    return tested_samples


def pprint_accumulated_metrics(cum_metrics: Dict[str, List[float]],
                               n_tested: int,
                               n_valid_llm_response: int,
                               n_samples_skipped: int,
                               n_skipped_llm_response_inc: int
                               ) \
        -> Dict[str, Union[int, float, List[float]]] | None:
    """
    Prints stats.
    Returns computed metrics.

    :param cum_metrics:
    :param n_tested:
    :param n_valid_llm_response:
    :param n_samples_skipped:
    :param n_skipped_llm_response_inc:
    """

    if cum_metrics is not None:
        n_samples = len(cum_metrics["n_nodes_ref"])

        # Metrics on Yes/No response
        y_true = np.asarray(cum_metrics["has_ref_graph"], dtype=bool)
        y_pred = np.asarray(cum_metrics["has_target_graph"], dtype=bool)
        n_gt = y_true.sum()
        n_gf = y_true.shape[0] - n_gt
    else:
        n_samples = n_gt = n_gf = 0

    print("Graph match stats:")
    print(f"    Nbr MoA's skipped       = {n_samples_skipped:5,d}   ... (in examples or empty disease prompt)")
    print(f"    Nbr MoA's tested on LLM = {n_tested:5,d}")
    print(f"    Nbr valid LLM response  = {n_samples:5,d} ... Counts match? {n_samples == n_valid_llm_response}")
    print(f"    Nbr incomplete response = {n_skipped_llm_response_inc:5,d}")
    print()
    print(f"    Nbr positive samples    = {n_gt:5,d}")
    print(f"    Nbr negative samples    = {n_gf:5,d}")
    print()

    if cum_metrics is None:
        print()
        print("No metrics to accumulate!")
        print()
        return None

    # Test for minimal match -- only end-points and nodes in prompt, when there are additional nodes

    if n_gt > 0:
        minimal_match_idxs = [i for i in range(n_samples) if is_minimal_match_cumi(cum_metrics, i)]
        minimal_target_idxs = [i for i in range(n_samples) if is_minimal_target_moa(cum_metrics, i)]

        print(f"    Nbr MoAs retd by LLM with minimal match    = {len(minimal_match_idxs):,d}")
        print( "    Proportion of true MoAs with minimal match =",
               f"{len(minimal_match_idxs) / n_gt:.1%}" if n_gt else 0)
        print(f"    Nbr minimal MoAs retd by LLM with full match             = {len(minimal_target_idxs):,d}")
        print( "    Proportion of true MoAs that are minimal with full match =",
               f"{len(minimal_target_idxs) / n_gt:.1%}" if n_gt else 0)
        print(f"    Potentially very different MoA from LLM = {len(minimal_match_idxs) - len(minimal_target_idxs)}",
              f"... {(len(minimal_match_idxs) - len(minimal_target_idxs)) / n_gt:.1%}" if n_gt else "")
        print()

    # Metrics on Yes/No response
    # noinspection PyUnboundLocalVariable
    prec, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0.0)

    n_true_pos = np.logical_and(y_true, y_pred).sum()
    n_true_neg = np.logical_and(~y_true, ~y_pred).sum()
    n_false_neg = np.logical_and(y_true, ~y_pred).sum()
    n_false_pos = np.logical_and(~y_true, y_pred).sum()

    metrics = dict(n_gt=n_gt, n_gf=n_gf,
                   n_true_pos=n_true_pos, n_true_neg=n_true_neg,
                   n_false_neg=n_false_neg, n_false_pos=n_false_pos,
                   accuracy = accuracy_score(y_true, y_pred, normalize=True),
                   accuracy_gt = accuracy_score(y_true[y_true], y_pred[y_true], normalize=True) if n_gt else 0,
                   accuracy_gf = accuracy_score(y_true[~y_true], y_pred[~y_true], normalize=True) if n_gf else 0,
                   precision=prec, recall=recall, f1=f1,
                   )

    pp_dict(metrics, "Binary metrics")

    if n_gt > 0 and y_pred.sum() > 0:

        # Limit summary metrics to True-Positives
        tp_mask = np.logical_and(y_true, y_pred)

        print()
        print("Graph match scores:")

        for k, vals in cum_metrics.items():
            if k in ["has_ref_graph", "has_target_graph"]:
                vals = np.asarray(vals, dtype=np.int32)
            elif vals is not None:
                vals = np.asarray(vals)

            print(f"    {k}:", end="  ")
            if vals is None or len(vals) == 0:
                print("... (empty)")
                continue

            if np.isnan(vals).sum() == len(vals):
                print("... (all NaN)")
                continue

            # Limit to True-positives
            vals = vals[tp_mask]

            v_mean = np.nanmean(vals)
            q1, q2, q3 = np.nanpercentile(vals, [25, 50, 75])

            print(f"mean = {v_mean:.3f}")
            print(f"\trange = [{np.nanmin(vals):.3f}, {np.nanmax(vals):.3f}], ",
                  f"quartiles = [{q1:.3f}, {q2:.3f}, {q3:.3f}]")

            metrics[f"{k}:mean"] = v_mean
            metrics[f"{k}:quartiles"] = [q1, q2, q3]

    print()
    return metrics


# ======================================================================================================
#   Main
# ======================================================================================================

# To run
# ------
#
# [Python]$ python -m drugmechcf.analyze.addlink_sessions {analyze | cmp | ...}
#

if __name__ == "__main__":

    import argparse
    from datetime import datetime
    from drugmechcf.utils.misc import print_cmd

    _argparser = argparse.ArgumentParser(
        description='Analyze Add-Link sessions.',
    )

    _subparsers = _argparser.add_subparsers(dest='subcmd',
                                            title='Available commands',
                                            )
    # Make the sub-commands required
    _subparsers.required = True

    # ... analyze
    _sub_cmd_parser = _subparsers.add_parser('analyze',
                                             help="Analyze one session.")
    _sub_cmd_parser.add_argument('json_file', type=str,
                                 help="JSON session file.")

    # ... cmp
    _sub_cmd_parser = _subparsers.add_parser('cmp',
                                             help="Analyze one session.")
    _sub_cmd_parser.add_argument('json_1', type=str,
                                 help="JSON session file.")
    _sub_cmd_parser.add_argument('json_2', type=str,
                                 help="JSON session file.")

    # ...

    _args = _argparser.parse_args()
    # .................................................................................................

    start_time = datetime.now()

    print("---------------------------------------------------------------------")
    print_cmd()

    if _args.subcmd == 'analyze':

        analyze_session(_args.json_file)

    elif _args.subcmd == 'cmp':

        compare_sessions(_args.json_1, _args.json_2)

    else:

        raise NotImplementedError(f"Command not implemented: {_args.subcmd}")

    # /

    print('\nTotal Run time =', datetime.now() - start_time)
    print()
