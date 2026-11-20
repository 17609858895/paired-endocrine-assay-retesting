"""Validate persisted paired predictions and summarize discordance calibration."""
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn

from paths import DATA, RESULTS as OUT
Q = ["q00", "q01", "q10", "q11"]


def run():
    x = pd.read_parquet(OUT / "dependence_predictions.parquet")
    extra = pd.read_parquet(OUT / "dependence_direct_calibrated_predictions.parquet")
    x = pd.concat([x, extra], ignore_index=True)
    k = ["contrast", "structure_id"]
    by = k + ["model"]
    manifest = json.loads((OUT / "dependence_manifest.json").read_text())
    assert len(x) == manifest["structure_pairs"] * 6 and not x.duplicated(by).any()
    assert x.groupby(k).model.nunique().eq(6).all()
    assert x.groupby("scaffold").fold.nunique().eq(1).all()
    assert x.groupby("structure_id").fold.nunique().eq(1).all()
    q = x[Q].to_numpy()
    assert np.isfinite(q).all() and (q > 0).all()
    sum_error = float(np.max(np.abs(q.sum(axis=1)-1)))
    assert sum_error < 1e-12
    fixed = x[x.model.isin(["independent", "constant_or", "score_or"])]
    spread = fixed.groupby(k)[["pA", "pB"]].max() - fixed.groupby(k)[["pA", "pB"]].min()
    max_spread = float(spread.to_numpy().max())
    assert max_spread < 1e-10
    expected = pd.read_parquet(DATA / "context_pairs.parquet", columns=k + ["y_a", "y_b"])
    expected["raw_state"] = 2 * expected.y_a + expected.y_b
    aligned = x.merge(expected[k + ["raw_state"]], on=k, validate="many_to_one", how="left")
    assert aligned.state.eq(aligned.raw_state).all()
    x["observed"] = x.state.isin([1, 2]).astype(int)
    x["brier"] = (x.observed - x.p_discord)**2
    x["nll"] = -np.log(np.maximum(q[np.arange(len(x)), x.state.to_numpy()], 1e-14))
    x["hits"] = x.observed * x.selected_top10
    keys = ["contrast", "model"]
    metrics = x.groupby(keys).agg(n=("observed", "size"), nll=("nll", "mean"),
                                  top10_hits=("hits", "sum"), top10_budget=("selected_top10", "sum"),
                                  observed_discordance=("observed", "mean"),
                                  predicted_discordance=("p_discord", "mean"), discord_brier=("brier", "mean"))
    stored = pd.concat([pd.read_csv(OUT / n) for n in
                        ["dependence_summary.csv","dependence_direct_calibrated_summary.csv"]]).set_index(keys)
    cols = ["n", "nll", "top10_hits", "top10_budget"]
    assert np.allclose(metrics[cols], stored.loc[metrics.index, cols], rtol=0, atol=1e-12)
    metrics.to_csv(OUT / "dependence_calibration.csv")
    x["scaffold_set"] = np.where(x.scaffold.eq("<none>"), "no_ring_scaffold", "ring_scaffold")
    x.groupby(keys + ["scaffold_set"]).agg(
        n=("state", "size"), actual_discordance=("observed", "mean"),
        predicted_discordance=("p_discord", "mean"), nll=("nll", "mean"),
        hits_with_fullfold_budget=("hits", "sum"), selections_with_fullfold_budget=("selected_top10", "sum")
    ).to_csv(OUT / "dependence_ring_sensitivity.csv")
    comparisons = [("independent", "constant_or"), ("constant_or", "score_or"),
                   ("constant_or", "multinomial_calibrated"), ("score_or", "multinomial_calibrated")]
    intervals = []
    for contrast, rows in x.groupby("contrast"):
        for base, new in comparisons:
            index = ["structure_id", "scaffold"]
            a = rows[rows.model.eq(base)].set_index(index)
            b = rows[rows.model.eq(new)].set_index(index).reindex(a.index)
            diff = (b.nll - a.nll).groupby(level="scaffold").agg(["sum", "count"]).to_numpy()
            rng = np.random.default_rng(838)
            estimates = []
            for chunk in range(15):
                choices = rng.integers(0, len(diff), (100, len(diff)))
                aggregate = diff[choices].sum(axis=1)
                estimates.extend(aggregate[:, 0] / aggregate[:, 1])
            intervals.append(dict(contrast=contrast, reference=base, comparison=new,
                                  delta_nll=(b.nll - a.nll).mean(), lo=np.quantile(estimates, .025),
                                  hi=np.quantile(estimates, .975), scaffolds=len(diff)))
    pd.DataFrame(intervals).to_csv(OUT / "dependence_calibrated_interval.csv", index=False)
    structs = pd.read_csv(DATA / "structures.csv", usecols=["structure_id", "parent_key"])
    measured = x[k + ["scaffold", "fold"]].drop_duplicates()
    measured = measured.merge(structs, on="structure_id", validate="many_to_one")
    parents = measured.groupby("parent_key").fold.nunique()
    cross = parents[parents.gt(1)].index
    chemistry = []
    for contrast, rows in measured.groupby("contrast"):
        counts = rows.scaffold.value_counts()
        by_parent = rows.groupby("parent_key").fold.nunique()
        within_task = by_parent[by_parent.gt(1)].index
        chemistry.append({"contrast":contrast,"structures":len(rows),"scaffolds":len(counts),
                          "largest_scaffold":counts.index[0],"largest_scaffold_n":int(counts.iloc[0]),
                          "scaffoldless_n":int(rows.scaffold.eq("<none>").sum()),
                          "nonmissing_parent_keys":int(rows.parent_key.notna().sum()),
                          "parent_keys_in_multiple_task_folds":len(within_task),
                          "structures_with_crossfold_parent":int(rows.parent_key.isin(within_task).sum())})
    pd.DataFrame(chemistry).to_csv(OUT / "dependence_chemical_domain.csv", index=False)
    check = {"records_verified":len(x),"tasks":4,"models":6,"global_crossfold_scaffolds":0,
             "global_crossfold_structures":0,"multi_fold_parent_keys_across_all_tasks":len(cross),
             "largest_probability_sum_error":sum_error,"largest_fixed_margin_difference":max_spread,
             "versions":{"python":platform.python_version(), "numpy":np.__version__,
                         "pandas":pd.__version__, "scipy":scipy.__version__,"sklearn":sklearn.__version__}}
    (OUT / "dependence_qa.json").write_text(json.dumps(check, indent=2), encoding="utf-8")


if __name__ == "__main__":
    run()
