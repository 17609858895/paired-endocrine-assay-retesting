"""Conditional repeat-state transitions and alternative screening budgets."""
import numpy as np
import pandas as pd
from paths import RESULTS
from rank_uncertainty import ID, MODELS, scores

SUB = ["structure_id", "dsstox_substance_id"]
BUDGETS = [5, 10, 15, 20]


def state_recurrence():
    pairs = pd.read_csv(RESULTS / "sample_range_pair_inventory.csv")
    assert not pairs.duplicated(["contrast", *SUB, "spid_i", "spid_j"]).any()
    pairs["state_i"] = 2 * pairs.y_a_i + pairs.y_b_i
    pairs["state_j"] = 2 * pairs.y_a_j + pairs.y_b_j
    sides = []
    for first, other in [("i","j"), ("j","i")]:
        sides.append(pairs[["contrast", *SUB, "spid_"+first, "spid_"+other, "state_"+first, "state_"+other]]
                     .set_axis(["contrast", *SUB, "index_spid", "other_spid", "index_state", "other_state"], axis=1))
    joint = pd.concat(sides,ignore_index=True)
    assert not joint.duplicated(["contrast", *SUB, "index_spid", "other_spid"]).any()
    # Choose a structure, substance and distinct ordered pair of SPIDs uniformly
    # at each level. Both orientations are bookkeeping, not independent samples.
    by_substance = joint.groupby(["contrast", *SUB]).size().rename("pair_count").reset_index()
    by_substance["substances"] = by_substance.groupby(["contrast","structure_id"]).pair_count.transform("size")
    by_substance["weight"] = 1.0/(by_substance.pair_count*by_substance.substances)
    joint = joint.merge(by_substance[["contrast", *SUB, "weight"]], on=["contrast",*SUB],validate="many_to_one")
    output = []
    published = pd.read_csv(RESULTS/"sample_control_summary.csv").query("subset == 'repeated_shared_spids'").set_index("contrast")
    for task,g in joint.groupby("contrast"):
        total = g.weight.sum()
        np.testing.assert_allclose(total, published.loc[task,"structures"])
        np.testing.assert_allclose(g.loc[g.index_state.isin([1,2]),"weight"].sum()/total, published.loc[task,"across"])
        for anchor in [0,1,2,3]:
            sub = g.loc[g.index_state.eq(anchor)]
            denominator = sub.weight.sum()
            for replicate in [0,1,2,3]:
                obs = sub.loc[sub.other_state.eq(replicate)]
                output.append(dict(contrast=task,index_state=anchor,other_state=replicate,
                    index_structures=sub.structure_id.nunique(),index_spids=sub.index_spid.nunique(),
                    index_ordered_pairs=len(sub),cell_ordered_pairs=len(obs),
                    index_weight=denominator,cell_weight=obs.weight.sum(),
                    conditional_pct=100*obs.weight.sum()/denominator if denominator else np.nan))
    out = pd.DataFrame(output)
    np.testing.assert_allclose(out.groupby(["contrast","index_state"]).conditional_pct.sum(min_count=1).dropna(),100.)
    return out


def budgets_from_predictions():
    matrix,stored,tie=scores()
    totals,intersections=[],[]
    for (task,fold),group in matrix.groupby(["contrast","fold"]):
        ids=group.structure_id.to_numpy()
        truth=group.state.isin([1,2]).to_numpy()
        orders = {model: np.lexsort((tie[ids], -group[model].to_numpy())) for model in MODELS}
        for pct in BUDGETS:
            budget=(len(group)*pct+99)//100
            selections={}
            for model,order in orders.items():
                selected=np.zeros(len(group),bool)
                selected[order[:budget]]=True
                if pct==10 and model in stored.columns:
                    ref=stored.loc[pd.MultiIndex.from_frame(group[ID]),model].to_numpy()
                    assert np.array_equal(selected,ref)
                selections[model]=selected
                totals.append(dict(contrast=task,fold=fold,pct=pct,model=model,n=len(group),
                    discordant=int(truth.sum()),budget=budget,hits=int(np.sum(selected&truth)),
                    random_expected=budget*truth.mean()))
            if pct==10:
                left,right=selections["lightgbm_joint"],selections["independent"]
                for subset,mask in [("both",left&right),("lightgbm_only",left&~right),("independent_only",right&~left)]:
                    intersections.append(dict(contrast=task,fold=fold,subset=subset,structures=int(mask.sum()),
                                              hits=int((mask&truth).sum())))
    folds=pd.DataFrame(totals)
    summary=folds.groupby(["contrast","model","pct"],as_index=False)[["n","discordant","budget","hits","random_expected"]].sum()
    summary["enrichment"]=summary.hits/summary.random_expected
    summary["hit_rate"]=summary.hits/summary.budget
    ref=pd.read_csv(RESULTS/"all_ranking_comparisons.csv").set_index(["contrast","model"]).sort_index()
    orig=summary.loc[summary.pct.eq(10)].set_index(["contrast","model"]).sort_index()
    for col in ["budget","hits","n","discordant","random_expected","enrichment"]:
        np.testing.assert_allclose(orig[col],ref[col],rtol=1e-13,atol=1e-13)
    sets=pd.DataFrame(intersections)
    for (task,fold),r in sets.groupby(["contrast","fold"]):
        r=r.set_index("subset")
        info=folds.loc[folds.contrast.eq(task)&folds.fold.eq(fold)&folds.pct.eq(10)].set_index("model")
        for name,model in [("lightgbm_only","lightgbm_joint"),("independent_only","independent")]:
            np.testing.assert_equal(r.loc[[name,"both"],"hits"].sum(),info.loc[model,"hits"])
            np.testing.assert_equal(r.loc[[name,"both"],"structures"].sum(),info.loc[model,"budget"])
    return folds,summary,sets


if __name__ == "__main__":
    state_recurrence().to_csv(RESULTS/"state_recurrence.csv", index=False)
    fold_results, summary, overlap = budgets_from_predictions()
    fold_results.to_csv(RESULTS/"budget_results_by_fold.csv", index=False)
    summary.to_csv(RESULTS/"budget_results.csv", index=False)
    overlap.to_csv(RESULTS/"top_decile_list_overlap.csv", index=False)
