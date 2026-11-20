"""Compare conditions on matched sample IDs and repeats of the same endpoint."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

from paths import DATA, RESULTS as OUT
TASKS = {
    "AR_challenge": (1816, 765), "ER_challenge": (2053, 789),
    "AR_blocker": (764, 1822), "ER_blocker": (788, 2211),
}
KEY = ["structure_id", "dsstox_substance_id"]
MEASURES = ["across", "within_a", "within_b", "net_b_minus_a"]


def summarize(units, scope, task):
    # Each structure contributes once, averaging its registered substances first.
    per_structure = units.groupby("structure_id")[MEASURES].mean()
    vals = per_structure.mean().to_dict()
    return dict(contrast=task, subset=scope, structures=len(per_structure),
                substances=len(units), matched_spids=int(units["count"].sum()), **vals)


def repeat_metrics(units, structures, task, rng):
    annotated = units.merge(structures, on="structure_id", validate="many_to_one")
    per_structure = annotated.groupby(["scaffold", "structure_id"])[MEASURES].mean()
    per_structure["across_minus_a"] = per_structure.across - per_structure.within_a
    per_structure["across_minus_b"] = per_structure.across - per_structure.within_b
    grouped = per_structure.groupby(level="scaffold")
    totals = grouped.sum().to_numpy()
    sizes = grouped.size().to_numpy()
    weights = rng.multinomial(len(sizes), np.ones(len(sizes))/len(sizes), size=5000)
    samples = (weights @ totals)/(weights @ sizes)[:, None]
    limits = np.quantile(samples, [.025, .975], axis=0)
    return [dict(contrast=task, measure=col, structures=len(per_structure), scaffolds=len(sizes),
                 estimate=float(per_structure[col].mean()), lo=limits[0,i], hi=limits[1,i])
            for i,col in enumerate(per_structure.columns)]


def main():
    OUT.mkdir(exist_ok=True)
    endpoints = sorted({v for pair in TASKS.values() for v in pair})
    raw = pd.read_parquet(DATA / "release_activity.parquet", filters=[("aeid", "in", endpoints)])
    valid = raw.loc[raw.hitcall.ge(0) & raw.conc_unit.eq("uM")].dropna(subset=["spid", "dsstox_substance_id"])
    identity = pd.read_csv(DATA / "substance_to_structure.csv", usecols=["structure_id", "dtxsid"])
    valid = valid.merge(identity.rename(columns={"dtxsid": KEY[1]}), on=KEY[1], validate="many_to_one")
    valid["y"] = valid.hitcall.ge(.9).astype(int)
    sample = valid.groupby(KEY + ["aeid", "spid"], as_index=False).agg(
        lo=("y", "min"), hi=("y", "max"), raw_rows=("y", "size"))
    eligible = sample.loc[sample.lo.eq(sample.hi)].rename(columns={"lo": "y"})
    registry = pd.read_csv(DATA/"structures.csv", usecols=["structure_id", "scaffold"], keep_default_na=False)
    summary, details, intervals, patterns = [], [], [], []
    rng = np.random.default_rng(20261121)
    for task, (ea, eb) in TASKS.items():
        a = eligible.loc[eligible.aeid.eq(ea), KEY + ["spid", "y"]]
        b = eligible.loc[eligible.aeid.eq(eb), KEY + ["spid", "y"]]
        matched = a.merge(b, on=KEY + ["spid"], suffixes=("_a", "_b"), validate="one_to_one")
        matched["across"] = matched.y_a.ne(matched.y_b)
        matched["net_b_minus_a"] = matched.y_b - matched.y_a
        units = matched.groupby(KEY, as_index=False).agg(
            count=("spid", "size"), a_hits=("y_a", "sum"), b_hits=("y_b", "sum"),
            across=("across", "mean"), net_b_minus_a=("net_b_minus_a", "mean"))
        multi = units["count"].gt(1)
        for side in ["a", "b"]:
            n, k = units.loc[multi, "count"], units.loc[multi, f"{side}_hits"]
            units.loc[multi, f"within_{side}"] = 2*k*(n-k)/(n*(n-1))
        # On the repeated sample subset, A, B and across measurements use exactly
        # the same registered substances and the same shared set of sample IDs.
        for label, sub in [("same_spid", units), ("repeated_shared_spids", units.loc[multi])]:
            item = summarize(sub, label, task)
            if label == "same_spid":
                item["within_a"] = item["within_b"] = np.nan
            summary.append(item)
        repeated = units.loc[multi].copy()
        intervals.extend(repeat_metrics(repeated, registry, task, rng))
        any_cross = repeated.loc[repeated.across.gt(0)]
        always_10 = any_cross.a_hits.eq(any_cross["count"]) & any_cross.b_hits.eq(0)
        always_01 = any_cross.b_hits.eq(any_cross["count"]) & any_cross.a_hits.eq(0)
        patterns.append(dict(contrast=task, repeated_substances=len(repeated),
                             any_cross=len(any_cross), all_10=int(always_10.sum()),
                             all_01=int(always_01.sum()), varying=int((~(always_10 | always_01)).sum()),
                             no_cross=int(repeated.across.eq(0).sum())))
        units.insert(0, "contrast", task)
        details.append(units)
    result = pd.DataFrame(summary)
    assert result.structures.gt(0).all()
    assert (result.loc[result.subset.eq("repeated_shared_spids"), "matched_spids"] >=
            2*result.loc[result.subset.eq("repeated_shared_spids"), "substances"]).all()
    result.to_csv(OUT/"sample_control_summary.csv", index=False)
    pd.DataFrame(intervals).to_csv(OUT/"sample_control_intervals.csv", index=False)
    patterns = pd.DataFrame(patterns)
    assert (patterns.any_cross == patterns[["all_10", "all_01", "varying"]].sum(axis=1)).all()
    patterns.to_csv(OUT/"sample_control_patterns.csv", index=False)
    pd.concat(details, ignore_index=True).to_parquet(OUT/"sample_control_substances.parquet", index=False)
    audit = dict(raw_records=len(raw), mapped_valid_records_with_ids=len(valid),
                 sample_endpoint_keys=len(sample), discordant_sample_keys_excluded=int(sample.lo.ne(sample.hi).sum()),
                 sample_id_mapping_max_substances=int(valid.groupby("spid")[KEY[1]].nunique().max()),
                 pairing_key=KEY + ["spid"], per_structure_weighting="Equal substances in structure, equal structures in mean")
    (OUT/"sample_controls_check.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
