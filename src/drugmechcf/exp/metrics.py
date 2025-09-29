"""
Analyze and summarize metrics
"""

from collections import Counter
import glob
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from drugmechcf.utils.misc import reset_df_index


# -----------------------------------------------------------------------------
#   Functions
# -----------------------------------------------------------------------------


def read_counterfactuals_metrics(excel_file: str = "../Data/Sessions/Latest/LatestMetrics.xlsx",
                                 sheet: str = "Counterfactuals",
                                 ) -> pd.DataFrame | None:

    df = pd.read_excel(excel_file, sheet_name=sheet)

    hdg_i = None
    for i, row in enumerate(df.itertuples()):
        if row[1] == "Query Type":
            hdg_i = i
            break

    if hdg_i is None:
        print("Headings row not found!")
        return None

    # Make DF from the strict-metrics data, skipping empty lines
    data = pd.DataFrame.from_records([t[1:10] for t in df.loc[9:34].itertuples() if not pd.isna(t[1])],
                                     columns=df.loc[8].values[:9])

    return data


def summarize_counterfactual_metrics(excel_file: str = "../Data/Sessions/Latest/LatestMetrics.xlsx",
                                     sheet: str = "Counterfactuals",
                                     ):

    df = read_counterfactuals_metrics(excel_file, sheet)

    if df is None:
        return

    print("Headings =", ", ".join(df.columns.values))
    print()

    pp_summary_metrics("All Counterfactuals", df)

    # Open, Closed

    pp_summary_metrics("All samples, Open world",
                       df[df["Known MoA?"] == "no"])

    pp_summary_metrics("All samples, Closed world",
                       df[df["Known MoA?"] == "Yes"])

    # Pos, Neg

    pp_summary_metrics("All Positive samples",
                       df[df["Pos / Neg"] == "pos"])

    pp_summary_metrics("All Negative samples",
                       df[df["Pos / Neg"] == "neg"])

    # Pos - Neg, Open - Closed

    pp_summary_metrics("Positive samples, Open world",
                       df[(df["Pos / Neg"] == "pos") & (df["Known MoA?"] == "no")])

    pp_summary_metrics("Negative samples, Open world",
                       df[(df["Pos / Neg"] == "neg") & (df["Known MoA?"] == "no")])

    pp_summary_metrics("Positive samples, Closed world",
                       df[(df["Pos / Neg"] == "pos") & (df["Known MoA?"] == "Yes")])

    pp_summary_metrics("Negative samples, Closed world",
                       df[(df["Pos / Neg"] == "neg") & (df["Known MoA?"] == "Yes")])

    # DPI - PPI, Open - Closed

    pp_summary_metrics("Source is Drug, Open world",
                       df[(df["Src is Drug?"] == "Yes") & (df["Known MoA?"] == "no")])

    pp_summary_metrics("Source is Not Drug, Open world",
                       df[(df["Src is Drug?"] == "no") & (df["Known MoA?"] == "no")])

    pp_summary_metrics("Source is Drug, Closed world",
                       df[(df["Src is Drug?"] == "Yes") & (df["Known MoA?"] == "Yes")])

    pp_summary_metrics("Source is Not Drug, Closed world",
                       df[(df["Src is Drug?"] == "no") & (df["Known MoA?"] == "Yes")])

    # Change + Delete, not Drug, Open World

    pp_summary_metrics("Change + Delete, not Drug, Open World - Positives",
                       df[ ((df["Query Type"] == "Change Link") | (df["Query Type"] == "Delete Link")) &
                           (df["Src is Drug?"] == "no") & (df["Known MoA?"] == "no") & (df["Pos / Neg"] == "pos")])

    pp_summary_metrics("Change + Delete, not Drug, Open World - Negatives",
                       df[ ((df["Query Type"] == "Change Link") | (df["Query Type"] == "Delete Link")) &
                           (df["Src is Drug?"] == "no") & (df["Known MoA?"] == "no") & (df["Pos / Neg"] == "neg")])

    return


def pp_summary_metrics(hdg: str, df: pd.DataFrame):

    print("## " + hdg)
    print()

    # For the 4 models
    mdf = df[df.columns.values[-4:]]

    smdf = pd.DataFrame.from_records([["min"] + mdf.min().values.tolist(),
                                      ["max"] + mdf.max().values.tolist(),
                                      ["Median"] + mdf.median().values.tolist(),
                                      ["Mean"] + mdf.mean().values.tolist(),
                                      ["s.d."] + mdf.std(ddof=0).values.tolist(),
                                      ],
                                     columns=["Metric"] + df.columns.values.tolist()[-4:])

    print(reset_df_index(smdf).to_markdown(floatfmt=".3f", index=False))
    print("\n")
    return


# noinspection PyPep8Naming
def regression_on_mod_depth_params(data_file="./Data/Sessions/Latest/attrib_params_pos_ppi.txt",
                                   n_model_runs=100,
                                   test_size=0.25):
    df = pd.read_csv(data_file)
    X = df.drop('response_is_correct', axis=1)
    y = df['response_is_correct']

    n_correct = df['response_is_correct'].sum()
    n_samples = df.shape[0]
    expected_accuracy_min = n_correct / n_samples
    if expected_accuracy_min < 0.5:
        expected_accuracy_min = 1.0 - expected_accuracy_min

    print("Columns:", ", ".join(df.columns.values.tolist()))
    print(f"nbr Samples read = {n_samples:,d}")
    print(f"nbr Correct response = {n_correct}, {n_correct/n_samples:.1%}")
    print()
    print(f"Desired min Accuracy of LR = {expected_accuracy_min:.3f}")
    print()

    model_runs_data = []
    for _ in range(n_model_runs):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
        model = LogisticRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        run_data = [accuracy_score(y_test, y_pred), *model.coef_[0].tolist(), model.intercept_[0]]
        model_runs_data.append(run_data)

    model_runs_data = np.asarray(model_runs_data)

    var_names = ['accuracy'] + df.columns.values.tolist()[1:] + ['intercept']
    for i, vnm in enumerate(var_names):
        data = model_runs_data[:, i]
        print(f'{vnm}: mean = {np.mean(data):.5f}, s.d. = {np.std(data, ddof=0):.5f}')
        print()

    return


def get_llm_opt_freq(files_patt: str):
    files = glob.glob(files_patt)
    print()
    print("nbr Files =", len(files))

    opt_cnt = Counter()
    for file in files:
        with open(file) as jf:
            jdict = json.load(jf)
            opt_cnt.update([s['opt_match_metrics']["option_key"] for s in jdict['session']])

    n_samples = sum(opt_cnt.values())
    print("nbr samples read =", n_samples)
    print()

    maxw = max(len(k) for k in opt_cnt)

    for k, v in opt_cnt.most_common():
        print(f"{k:{maxw}s} = {v:5,d} ... {v/n_samples:6.1%}")

    print()
    return
