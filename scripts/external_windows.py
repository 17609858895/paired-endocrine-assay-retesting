"""Compare identical measured concentration windows across laboratories."""
import numpy as np
import pandas as pd
from paths import RESULTS

RUN = ["run_item", "lab", "run"]


def case_curves(data):
    table = data.replace([np.inf, -np.inf], np.nan).dropna(subset=["yc_relative_pct", "sc_relative_pct"])
    table = table.loc[table.yc_lowest_dose_mg_ml.eq(table.sc_lowest_dose_mg_ml)]
    sizes = table.groupby(RUN).size().rename("doses").reset_index()
    qualified = sizes[sizes.doses.ge(5)]
    support = qualified.groupby("run_item").agg(labs=("lab", "nunique"), runs=("run", "size"))
    support = support[support.labs.ge(2)].reset_index()
    if support.empty:
        raise ValueError("Two laboratories with at least five matched doses per run are required for a shared case.")
    chosen = support.sort_values(["labs", "runs", "run_item"], ascending=[False, False, True]).iloc[0]
    runs = qualified.loc[qualified.run_item.eq(chosen.run_item)].copy()
    lab_support = runs.groupby("lab").size().sort_index().sort_values(ascending=False, kind="stable")
    curves = []
    for lab in lab_support.index[:2]:
        best_run = runs[runs.lab.eq(lab)].sort_values(["doses", "run"], ascending=[False, True]).iloc[0]
        curves.append(table[(table.run_item == chosen.run_item) & (table.lab == lab) &
                            (table.run == best_run.run)].sort_values("dose_mg_ml"))
    return curves


def common_dose_changes(data):
    """Compare two common measured doses, first within run and then across laboratories."""
    records = []
    for (item, lab, run), frame in data.groupby(RUN):
        curve = frame.sort_values("dose_mg_ml")
        first = curve.iloc[0]
        if (len(curve) < 2 or min(first.yc_mean, first.sc_mean) <= 0 or
                first.dose_mg_ml != first.yc_lowest_dose_mg_ml or
                first.dose_mg_ml != first.sc_lowest_dose_mg_ml):
            continue
        for point in curve.iloc[1:].itertuples():
            gap = 100 * (point.sc_mean / first.sc_mean - point.yc_mean / first.yc_mean)
            records.append((item, lab, run, first.dose_mg_ml, point.dose_mg_ml, gap))
    frame = pd.DataFrame(records, columns=RUN + ["low_dose", "high_dose", "gap_pp"])
    keys = ["run_item", "low_dose", "high_dose"]
    support = frame.groupby(keys).agg(labs=("lab", "nunique"), runs=("run", "size")).reset_index()
    support = support.loc[support.labs.ge(2)].copy()
    if support.empty:
        raise ValueError("No pair of dose levels shared by two laboratories.")
    support["log_width"] = np.log10(support.high_dose / support.low_dose)
    order = ["run_item", "labs", "runs", "log_width", "low_dose", "high_dose"]
    representative = support.sort_values(order, ascending=[True, False, False, False, True, True])
    selected = representative.drop_duplicates("run_item")
    windows = frame.merge(support[keys], on=keys, validate="many_to_one")
    medians = windows.groupby(keys + ["lab"]).agg(gap_pp=("gap_pp", "median"), runs=("run", "size")).reset_index()
    extreme = medians.groupby(keys).gap_pp.agg(["min", "max"]).reset_index()
    extreme["direction"] = np.select([extreme["min"] > 0, extreme["max"] < 0],
                                     ["all_positive", "all_negative"], default="mixed_or_zero")
    return medians.merge(selected[keys], on=keys, validate="many_to_one"), extreme


if __name__=="__main__":
    data = pd.read_csv(RESULTS/"external_calux_paired_doses.csv")
    data = data.loc[data.run_item.ne(47)].copy()  # Additional blinded test of the same substance as item 3.
    changes, windows = common_dose_changes(data)
    changes.to_csv(RESULTS/"common_dose_comparisons.csv", index=False)
    windows.to_csv(RESULTS/"all_common_windows.csv", index=False)
    pd.concat(case_curves(data), ignore_index=True).to_csv(RESULTS/"selected_curves.csv", index=False)
