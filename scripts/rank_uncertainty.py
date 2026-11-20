"""Fixed-prediction bootstrap for the per-fold top-decile screen."""
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd


from paths import RESULTS as OUT
TASKS = ["AR_challenge", "ER_challenge", "AR_blocker", "ER_blocker"]
MODELS = ["empirical", "independent", "constant_or", "score_or", "multinomial",
          "multinomial_calibrated", "lightgbm_joint", "lightgbm_discordance", "absolute_marginal_gap"]
COMPARISONS = [("constant_or", "independent"), ("score_or", "constant_or"),
               ("lightgbm_joint", "independent"), ("lightgbm_joint", "multinomial_calibrated")]
B = 5000
BATCH = 100
BOOTSTRAP_SEED = 20261122
ORIGINAL_TIE_SEED = 838
ID = ["contrast", "fold", "structure_id", "state"]
SOURCES = ["dependence_predictions.parquet", "dependence_direct_calibrated_predictions.parquet",
           "nonlinear_predictions.parquet", "nonlinear_rank_predictions.parquet",
           "dependence_global_folds.csv", "all_ranking_comparisons.csv"]


def scores():
    frames = [pd.read_parquet(OUT/name) for name in SOURCES[:4]]
    fold_table = pd.read_csv(OUT/SOURCES[4], keep_default_na=False)
    assert np.array_equal(fold_table.structure_id, np.arange(len(fold_table)))
    assert fold_table.groupby("scaffold").fold.nunique().eq(1).all()
    tie = np.random.default_rng(ORIGINAL_TIE_SEED).permutation(len(fold_table))
    all_scores = pd.concat([f[ID+["model","p_discord","selected_top10"]] for f in frames], ignore_index=True)
    gap = frames[0].loc[frames[0].model.eq("independent"), ID+["pA","pB"]].copy()
    gap["model"] = "absolute_marginal_gap"
    gap["p_discord"] = gap.pA.sub(gap.pB).abs()
    matrix = pd.concat([all_scores[ID+["model","p_discord"]], gap[ID+["model","p_discord"]]]).pivot(
        index=ID, columns="model", values="p_discord"
    )
    assert not matrix[MODELS].isna().any().any()
    matrix = matrix.reset_index().merge(fold_table, on=["structure_id","fold"], how="left", validate="many_to_one")
    assert matrix.scaffold.notna().all()
    assert not matrix.duplicated(["contrast","structure_id"]).any()
    decisions = all_scores.pivot(index=ID, columns="model", values="selected_top10")
    return matrix, decisions, tie


def weighted_hits(weights, budgets, sorted_scaffolds, sorted_truth):
    # In rank order, a structure occurs once per copy of its sampled scaffold.
    # The last selected structure can contribute only the unfilled places.
    rank_weights = weights[:, sorted_scaffolds]
    offsets = np.cumsum(rank_weights, axis=1, dtype=np.int32) - rank_weights
    counts = np.clip(budgets[:,None] - offsets, 0, rank_weights)
    assert np.array_equal(counts.sum(axis=1), budgets)
    return counts[:, sorted_truth].sum(axis=1)


def one_fold(task, fold_id, fold, ref, tie):
    ids = fold.structure_id.to_numpy()
    truth = fold.state.isin([1,2]).to_numpy()
    reserved = (len(ids) + 9)//10
    groups, scaffold_names = pd.factorize(fold.scaffold, sort=True)
    k = len(scaffold_names)
    n_by_group = np.bincount(groups)
    pos_by_group = np.bincount(groups, weights=truth.astype(int)).astype(int)
    ordered, observed, details = {}, [], []
    for model in MODELS:
        order = np.lexsort((tie[ids], -fold[model].to_numpy()))
        ordered[model] = order
        observed.append(int(truth[order[:reserved]].sum()))
        if model in ref:
            np.testing.assert_array_equal(np.sort(np.flatnonzero(ref[model].to_numpy())),
                                          np.sort(order[:reserved]))
        details.append(dict(contrast=task, fold=fold_id, model=model, n=len(ids), scaffolds=k,
                            largest_scaffold=scaffold_names[np.argmax(n_by_group)],
                            largest_scaffold_n=int(n_by_group.max()), discordant=int(truth.sum()),
                            budget=reserved,hits=observed[-1],random_expected=reserved*truth.mean()))
    # columns after the models are N, budget, expected random hits
    result = np.zeros((B, len(MODELS)+3))
    if k == 1:
        result[:] = np.array(observed+[len(ids),reserved,reserved*truth.mean()])
        return result, details
    rng = np.random.default_rng([BOOTSTRAP_SEED,TASKS.index(task),int(fold_id)])
    for start in range(0,B,BATCH):
        count = min(BATCH,B-start)
        weights = rng.multinomial(k, np.full(k,1/k), size=count).astype(np.int32)
        n = weights@n_by_group
        budget = (n+9)//10
        expected = budget * (weights@pos_by_group) / n
        part = result[start:start+count]
        part[:,-3:] = np.column_stack([n,budget,expected])
        for j,model in enumerate(MODELS):
            order = ordered[model]
            hits = weighted_hits(weights, budget, groups[order], truth[order])
            part[:,j] = hits
            if start == 0:
                literal = np.repeat(truth[order],weights[0,groups[order]])
                assert int(literal[:budget[0]].sum()) == hits[0]
    return result, details


def qrange(arr):
    lo, hi = np.quantile(arr,[.025,.975])
    return {"lo":lo,"hi":hi}


def main():
    predictions, decisions, tie = scores()
    old = pd.read_csv(OUT/"all_ranking_comparisons.csv").set_index(["contrast","model"])
    report, contrasts, all_folds, resampled_sizes = [], [], [], []
    for task in TASKS:
        group = predictions.loc[predictions.contrast.eq(task)]
        total = np.zeros((B, len(MODELS)+3))
        details = []
        for fold_id, fold in group.groupby("fold"):
            ref = decisions.reindex(pd.MultiIndex.from_frame(fold[ID]))
            arr, part = one_fold(task,fold_id,fold,ref,tie)
            total += arr
            details.extend(part)
        all_folds.extend(details)
        observed = pd.DataFrame(details).groupby("model")[["n","discordant","budget","hits","random_expected"]].sum().loc[MODELS]
        expected, budget = total[:,-1], total[:,-2]
        assert np.min(expected)>0 and np.all(budget>0)
        rates = total[:,:len(MODELS)]/budget[:,None]
        enrichment = total[:,:len(MODELS)]/expected[:,None]
        point_rate = observed.hits/observed.budget
        point_enrichment = observed.hits/observed.random_expected
        np.testing.assert_allclose(observed,old.loc[task].loc[MODELS,observed.columns], rtol=1e-13,atol=1e-12)
        np.testing.assert_allclose(point_enrichment,old.loc[task].loc[MODELS,"enrichment"],rtol=1e-13)
        meta = observed.loc[MODELS[0]].to_dict()
        resampled_sizes.append(dict(contrast=task, observed_n=int(meta["n"]), observed_budget=int(meta["budget"]),
                               n_lo=qrange(total[:,-3])["lo"], n_hi=qrange(total[:,-3])["hi"],
                               budget_lo=qrange(budget)["lo"], budget_hi=qrange(budget)["hi"],
                               expected_lo=qrange(expected)["lo"],expected_hi=qrange(expected)["hi"]))
        for j, model in enumerate(MODELS):
            row = observed.loc[model]
            header = dict(contrast=task, model=model, n=int(row.n), discordant=int(row.discordant),
                          hits=int(row.hits), budget=int(row.budget), random_expected=row.random_expected)
            for metric, point, distribution in [("hit_rate",point_rate,rates),("enrichment",point_enrichment,enrichment)]:
                report.append({**header,"metric":metric,"estimate":point[model],**qrange(distribution[:,j])})
        for target,reference in COMPARISONS:
            j, k = MODELS.index(target), MODELS.index(reference)
            header = dict(contrast=task, target=target, reference=reference, budget=int(meta["budget"]),
                          observed_target_hits=int(observed.loc[target,"hits"]),
                          observed_reference_hits=int(observed.loc[reference,"hits"]))
            for metric,point,distribution in [("hit_rate_difference",point_rate,rates),
                                              ("enrichment_difference",point_enrichment,enrichment)]:
                draws = distribution[:,j]-distribution[:,k]
                contrasts.append({**header,"metric":metric,"estimate":point[target]-point[reference],**qrange(draws)})
    pd.DataFrame(report).to_csv(OUT/"rank_uncertainty_models.csv", index=False)
    pd.DataFrame(contrasts).to_csv(OUT/"rank_uncertainty_paired.csv", index=False)
    pd.DataFrame(all_folds).to_csv(OUT/"rank_uncertainty_folds.csv", index=False)
    pd.DataFrame(resampled_sizes).to_csv(OUT/"rank_uncertainty_pools.csv", index=False)
    audit = {"fixed_predictions":True, "replicates":B, "bootstrap_seed":BOOTSTRAP_SEED,
             "original_shared_tiebreak_seed":ORIGINAL_TIE_SEED,
             "task_fold_strata":len(TASKS)*5,"models":len(MODELS),
             "original_saved_row_decisions_reproduced":"all eight models with stored decisions",
             "original_taskwise_hits_budgets_and_exact_random_expectations_reproduced":len(TASKS)*len(MODELS),
             "source_hashes": {name:hashlib.sha256((OUT/name).read_bytes()).hexdigest() for name in SOURCES}}
    (OUT/"rank_uncertainty_audit.json").write_text(json.dumps(audit,indent=2),encoding="utf8")


if __name__ == "__main__":
    main()
