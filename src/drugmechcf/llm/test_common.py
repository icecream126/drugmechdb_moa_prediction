"""
Common functions
"""

import json
import re
from typing import Tuple, Dict, List, Optional, Union

import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

from drugmechcf.data.drugmechdb import DrugMechDB
from drugmechcf.data.moagraph import MoaGraph
from drugmechcf.data.text2graph import MoaFromText

from drugmechcf.graphmatch.graphmatcher import BasicGraphMatcher, GraphMatchScore

from drugmechcf.llm.drugmechdb_prompt_builder import DrugMechPromptBuilder
from drugmechcf.llm.openai import OpenAICompletionClient, CompletionOutput, get_openai_client
from drugmechcf.llm.prompt_types import EditLinkInfo, PromptStyle, QueryType, DrugDiseasePromptInfo

from drugmechcf.utils.misc import pp_underlined_hdg, capitalize_words, pp_dict, suppressed_stdout


# -----------------------------------------------------------------------------
#   Globals
# -----------------------------------------------------------------------------

# Pattern to parse LLM response.
YES_NO_PATT = re.compile(r"(YES|NO)", re.IGNORECASE)


# For BasicGraphMatcher, equivalent entity types in DrugMechDB: Sequence[Sequence[str]].
# Each entry is seq of Entity-Type that are considered equivalent (Entity names across these can map to each other).
# See: BasicGraphMatcher.set_entity_type_equivalences()
ENTITY_TYPE_EQUIVALENCES = [("Protein", "GeneFamily")]


# -----------------------------------------------------------------------------
#   Functions
# -----------------------------------------------------------------------------


def test_edit_link_response(edit_link_info: EditLinkInfo,
                            drugmechdb: DrugMechDB,
                            reference_moa: MoaGraph,
                            prompt_builder: DrugMechPromptBuilder,
                            prompt_style: PromptStyle,
                            prompt_version: int,
                            query_type: QueryType,
                            grmatcher: BasicGraphMatcher = None,
                            llm: OpenAICompletionClient = None,
                            model_key: str = None,
                            temperature: float = 0.2,
                            verbose=True,
                            debug_no_llm=False,
                            debug_drug_disease_prompt_only=False,
                            show_full_prompt=False,
                            ) \
        -> Tuple[DrugDiseasePromptInfo | None, CompletionOutput | None, GraphMatchScore | None]:
    """
    For a +ive sample (there is a reference MoA),
    Create a prompt, invoke LLM, and match LLM response to reference.

    :return:
        - DrugDiseasePromptInfo | None if no LLM prompt possible or No valid New MoA
        - CompletionOutput (LLM response) | None if no LLM prompt possible or No valid New MoA
        - GraphMatchScore | None if: no LLM prompt or incomplete LLM response
    """

    if debug_drug_disease_prompt_only:
        prompt_info = prompt_builder.build_prompt_info(drugmechdb.get_moa_source_drug_node(reference_moa),
                                                       drugmechdb.get_moa_target_disease_node(reference_moa),
                                                       prompt_style,
                                                       prompt_version=prompt_version,
                                                       query_type=query_type,
                                                       moa=reference_moa,
                                                       edit_link_info=edit_link_info
                                                       )
        dd_prompt = prompt_builder.get_drug_disease_prompt(prompt_info) if prompt_info else None
        print("LLM Drug-Disease prompt:", dd_prompt, sep="\n  ")
        print(flush=True)
        return prompt_info, None, None

    prompt_info = prompt_builder.get_full_drug_disease_prompt(drugmechdb.get_moa_source_drug_node(reference_moa),
                                                              drugmechdb.get_moa_target_disease_node(reference_moa),
                                                              prompt_style,
                                                              prompt_version=prompt_version,
                                                              query_type=query_type,
                                                              moa=reference_moa,
                                                              edit_link_info=edit_link_info,
                                                              is_negative_sample=False,
                                                              verbose=False)
    if not prompt_info:
        return None, None, None

    if verbose:
        print("LLM sub-Prompt:", prompt_info.drug_disease_subprompt, sep="\n  ")
        print()
        if show_full_prompt:
            pp_underlined_hdg("Full Prompt:")
            print(prompt_info.full_prompt)
            print("---")

    if debug_no_llm:
        return prompt_info, None, None

    if llm is None:
        llm = get_openai_client(model_key=model_key, temperature=temperature)

    llm_response = llm(user_prompt=prompt_info.full_prompt)

    if verbose:
        print()

        pp_underlined_hdg("Response from LLM - ChatGPT-4o")
        if llm_response.is_complete_response():
            print(llm_response.message, flush=True)
        else:
            print("LLM stop reason =", llm_response.finish_reason)
        print("---\n", flush=True)

    if not llm_response.is_complete_response():
        return prompt_info, llm_response, None

    dis_name_in_llm_response = prompt_builder.get_dis_name_in_llm_response(prompt_style, prompt_info.disease_name)
    llm_graph_name = f"ChatGPT-4o.MoA: {prompt_info.drug_name} treats {prompt_info.disease_name}"

    if grmatcher is None:
        grmatcher = BasicGraphMatcher(verbosity=1)
        grmatcher.set_entity_type_equivalences(ENTITY_TYPE_EQUIVALENCES)

    match_metrics = test_moa_match(grmatcher, reference_moa, llm_response.message,
                                   prompt_info.drug_name, dis_name_in_llm_response,
                                   prompt_builder.formal_to_kg_entity_type_names(),
                                   llm_graph_name,
                                   match_entity_types=True,
                                   ref_nodes_in_prompt=prompt_info.disease_prompt_nodes,
                                   verbose=verbose
                                   )
    return prompt_info, llm_response, match_metrics


def test_moa_match(grmatcher: BasicGraphMatcher,
                   dmdb_moa: MoaGraph | None,
                   llm_response: str,
                   drug_name: str,
                   dis_name_in_llm_response: str,
                   llm_etype_to_dmdb: Dict[str, str],
                   llm_graph_name: str,
                   match_entity_types = True,
                   ref_nodes_in_prompt: List[str] = None,
                   verbose: bool = True,
                   ) -> GraphMatchScore:

    llm_moa = get_moa_from_llm_response(llm_response, drug_name, dis_name_in_llm_response,
                                        llm_etype_to_dmdb, llm_graph_name)

    metrics = grmatcher.match_graphs(dmdb_moa, llm_moa,
                                     force_match_end_points=True,
                                     match_entity_types=match_entity_types,
                                     ref_nodes_in_prompt=ref_nodes_in_prompt)

    if llm_moa:
        if verbose:
            pp_underlined_hdg("LLM MoA")
            llm_moa.pprint()
            print("---\n")

        if metrics is not None:
            print("Metrics for LLM response match to reference DrugMechDB graph:")
            metrics.pprint(indent=4)

    else:
        if verbose:
            print("No MoA from LLM")
            print()

    return metrics


def get_moa_from_llm_response(llm_response: str,
                              drug_name: str,
                              dis_name_in_llm_response: str,
                              llm_etype_to_dmdb: Dict[str, str],
                              llm_graph_name: str) -> Optional[MoaFromText]:

    lines = llm_response.splitlines()
    s = 0
    yes_no: Optional[str] = None
    for s, line_ in enumerate(lines):
        line_ = line_.strip()
        # Check if line begins with YES or NO
        if match := re.match(YES_NO_PATT, line_):
            yes_no = match.group(1)
            break

    if yes_no is None:
        raise ValueError(f"Unrecognized response from LLM:\n{llm_response}")

    if yes_no.casefold() == "no":
        return None

    moa = MoaFromText(lines[s + 1:],
                      graph_name=llm_graph_name,
                      entity_type_map=llm_etype_to_dmdb,
                      preferred_root=("Drug", capitalize_words(drug_name)),
                      preferred_sink=("Disease", dis_name_in_llm_response)
                      )

    return moa


# -----------------------------------------------------------------------------
#   Functions: Metrics
# -----------------------------------------------------------------------------


def is_minimal_match(n_nodes_ref, n_nodes_target,
                     n_prompt_ref_nodes, n_nodes_matched_ref, n_prompt_ref_nodes_matched):
    """
    A minimal match is when the target could have matched to more nodes than those 'mentioned' in the prompt,
    but did not.

    A match is also minimal when when n_nodes_matched_ref == 0!

    So a match is NOT considered minimal when:
        - target MoA is empty.
        - the ref-MoA does not have interior nodes other than Prompt nodes.
        - Any non-prompt interior nodes are matched, even if all prompt nodes are not matched.
    """

    # The calculations assume that the 2 End-Points are not considered prompt nodes

    # ref-MoA does not have any other non-prompt interior nodes
    if n_nodes_ref <= n_prompt_ref_nodes + 2:
        return False

    if n_nodes_target == 0:
        return False

    return n_nodes_matched_ref <= n_prompt_ref_nodes_matched + 2


def is_minimal_match_cumi(cum_metrics, i):
    if not (cum_metrics["has_ref_graph"][i] and cum_metrics["has_target_graph"][i]):
        return False
    return is_minimal_match(cum_metrics["n_nodes_ref"][i],
                            cum_metrics["n_nodes_target"][i],
                            cum_metrics["n_prompt_ref_nodes"][i],
                            cum_metrics["n_nodes_matched_ref"][i],
                            cum_metrics["n_prompt_ref_nodes_matched"][i])


def is_minimal_target_moa(cum_metrics, i):
    """
    A target-MoA is minimal, if it only contains nodes corresponding to the End-Points and the Prompt Nodes.
    More specifically, it is minimal when:
        - is_minimal_match() is True, AND
        - All the target nodes got matched
    """

    return (is_minimal_match_cumi(cum_metrics, i) and
            cum_metrics["n_nodes_matched_target"][i] == cum_metrics["n_nodes_target"][i])


def pprint_accumulated_metrics(cum_metrics: Dict[str, List[float]],
                               n_tested: int,
                               n_valid_llm_response: int,
                               n_samples_skipped: int,
                               n_skipped_llm_response_inc: int
                               )\
        -> Dict[str, Union[int, float, List[float]]] | None:
    """
    Prints stats.
    Prints summary stats on GraphMatchScore, limited to True-Positives.
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

    if n_gt > 0:
        # noinspection PyUnboundLocalVariable
        metrics["prop_very_diff_moas"] = (len(minimal_match_idxs) - len(minimal_target_idxs)) / n_gt
    else:
        # This metric not meaningful for Negative Samples.
        metrics["prop_very_diff_moas"] = np.NaN

    if n_gt > 0 and y_pred.sum() > 0:

        # Limit summary metrics to True-Positives
        tp_mask = np.logical_and(y_true, y_pred)

        print()
        pp_underlined_hdg("GraphMatch scores for True-Positives:")

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

            if vals is None or len(vals) == 0:
                print("... (empty)")
                continue

            v_mean = np.nanmean(vals)
            q1, q2, q3 = np.nanpercentile(vals, [25, 50, 75])

            print(f"mean = {v_mean:.3f}")
            print(f"\trange = [{np.nanmin(vals):.3f}, {np.nanmax(vals):.3f}], ",
                  f"quartiles = [{q1:.3f}, {q2:.3f}, {q3:.3f}]")

            metrics[f"{k}:mean"] = v_mean
            metrics[f"{k}:quartiles"] = [q1, q2, q3]

    print()
    return metrics


def extract_main_moa_metrics(session_json_file: str) -> dict[str, float]:
    with open(session_json_file) as f:
        jdict = json.load(f)

    cum_metrics = None
    for sample in jdict["session"]:
        if sample.get("metrics"):
            sample_metrics = GraphMatchScore(**sample["metrics"])
            cum_metrics = sample_metrics.accumulate_metrics(cum_metrics)

    n_tested = len(jdict["session"])
    n_valid_samples = len(cum_metrics["has_ref_graph"])

    with suppressed_stdout():
        metrics = pprint_accumulated_metrics(cum_metrics, n_tested, n_valid_samples,
                                             0, n_tested - n_valid_samples)

    # Default value for absent metric is np.NaN
    main_metrics = dict((k, metrics.get(k, np.NaN))
                        for k in ["n_gt", "n_gf",
                                  "prop_very_diff_moas",
                                  "recall",
                                  "n_true_pos", "n_true_neg", "n_false_pos", "n_false_neg",
                                  "accuracy_gt", "accuracy_gf",
                                  "interior_node_match_score:mean",
                                  "edge_match_score_reduced:mean"
                                  ])

    return main_metrics
