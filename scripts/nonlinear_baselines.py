"""Fixed-budget tree benchmarks on the existing global scaffold assignment."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import lightgbm
from lightgbm import LGBMClassifier
from scipy.special import softmax
from sklearn.metrics import average_precision_score
from paired_dependence import CONTRASTS, DATA, OUT, QCOLS, SEED, evaluate
from paired_direct_calibration import fit_calibrator


def tree(x, y, test, classes=4):
    clf = LGBMClassifier(n_estimators=100, learning_rate=.035, num_leaves=15,
                         min_child_samples=25, reg_lambda=3., colsample_bytree=.85,
                         random_state=SEED, n_jobs=3, verbosity=-1, deterministic=True,
                         force_col_wise=True)
    clf.fit(x, y)
    scores = clf.predict_proba(test)
    missing = classes - len(clf.classes_)
    prior_mass = .5/(len(y) + classes*.5)
    result = np.full((len(test), classes), prior_mass)
    result[:,clf.classes_] = scores*(1-missing*prior_mass)
    return result


def calibrate(raw, y, test):
    # Same inner-OOF calibrator and reference category as multinomial regression.
    theta = fit_calibrator(np.log(np.maximum(raw,1e-14)), y)
    return softmax(theta[0]*np.log(np.maximum(test,1e-14)) + np.r_[0.,theta[1:]],axis=1)


def top_mask(scores, tie, ids):
    order = np.lexsort((tie[ids], -np.asarray(scores)))
    mask = np.zeros(len(ids), bool)
    mask[order[:int(np.ceil(.1*len(ids)))]] = True
    return mask


def run():
    folds = pd.read_csv(OUT/"dependence_global_folds.csv", keep_default_na=False)
    features = np.load(DATA/"fingerprints.npz")["fp"]
    assert np.array_equal(folds.structure_id, np.arange(len(features)))
    assert folds.groupby("scaffold").fold.nunique().max() == 1
    tie = np.random.default_rng(SEED).permutation(len(folds))
    pairs = pd.read_parquet(DATA/"context_pairs.parquet", columns=["structure_id","contrast","y_a","y_b"])
    pairs = pairs.loc[pairs.contrast.isin(CONTRASTS)].merge(folds, on="structure_id", validate="many_to_one")
    pairs["state"] = 2*pairs.y_a + pairs.y_b
    preds, ranks = [], []
    for task, group in pairs.groupby("contrast", sort=False):
        for fold in sorted(group.fold.unique()):
            tr, te = group.loc[group.fold.ne(fold)], group.loc[group.fold.eq(fold)]
            xtr, xte = features[tr.structure_id], features[te.structure_id]
            y = tr.state.to_numpy(int)
            inner = np.zeros((len(y),4))
            for split in sorted(tr.fold.unique()):
                val = tr.fold.eq(split).to_numpy()
                inner[val] = tree(xtr[~val], y[~val], xtr[val])
            q = calibrate(inner, y, tree(xtr, y, xte))
            assert (q>0).all() and np.allclose(q.sum(axis=1),1.)
            d = te[["contrast","structure_id","scaffold","fold","state"]].copy()
            d["model"] = "lightgbm_joint"
            d[QCOLS] = q
            d["pA"], d["pB"] = q[:,2]+q[:,3], q[:,1]+q[:,3]
            d["p_discord"] = q[:,1]+q[:,2]
            d["selected_top10"] = top_mask(d.p_discord, tie, te.structure_id)
            preds.append(d)
            r = te[["contrast","structure_id","fold","state"]].copy()
            r["p_discord"] = tree(xtr,np.isin(y,[1,2]).astype(int),xte,classes=2)[:,1]
            r["selected_top10"] = top_mask(r.p_discord, tie, te.structure_id)
            r["model"] = "lightgbm_discordance"
            ranks.append(r)
            print(task, fold, flush=True)
    d = pd.concat(preds,ignore_index=True)
    d.to_parquet(OUT/"nonlinear_predictions.parquet", index=False)
    for by, name in [(["contrast","model"],"summary"),(["contrast","model","fold"],"folds")]:
        rows = [dict(zip(by,k)) | evaluate(s) for k,s in d.groupby(by)]
        pd.DataFrame(rows).to_csv(OUT/f"nonlinear_{name}.csv", index=False)
    binary = pd.concat(ranks,ignore_index=True)
    binary.to_parquet(OUT/"nonlinear_rank_predictions.parquet", index=False)
    base = pd.read_parquet(OUT/"dependence_predictions.parquet")
    # Additional pure-ranking comparators and the exact expectation of within-fold random selection.
    gap = base.loc[base.model.eq("independent")].copy()
    gap["p_discord"] = (gap.pA-gap.pB).abs()
    gap["model"] = "absolute_marginal_gap"
    for _, sub in gap.groupby(["contrast","fold"]):
        gap.loc[sub.index,"selected_top10"] = top_mask(sub.p_discord,tie,sub.structure_id)
    direct = pd.read_parquet(OUT/"dependence_direct_calibrated_predictions.parquet")
    all_ranks = pd.concat([base, direct, gap, d, binary],ignore_index=True)
    rank_rows = []
    for (task,method), sub in all_ranks.groupby(["contrast","model"]):
        truth = sub.state.isin([1,2])
        expected = sum(t.selected_top10.sum()*t.state.isin([1,2]).mean() for _,t in sub.groupby("fold"))
        hits = int((sub.selected_top10 & truth).sum())
        rank_rows.append(dict(contrast=task,model=method,n=len(sub),discordant=int(truth.sum()),
                              average_precision=average_precision_score(truth,sub.p_discord),
                              budget=int(sub.selected_top10.sum()),hits=hits,random_expected=expected,
                              enrichment=hits/expected))
    pd.DataFrame(rank_rows).to_csv(OUT/"all_ranking_comparisons.csv",index=False)
    (OUT/"nonlinear_settings.json").write_text(json.dumps(dict(lightgbm=lightgbm.__version__,
        trees=100,leaves=15,seed=SEED,learning_rate=.035,min_child_samples=25,reg_lambda=3.,
        feature_fraction=.85, joint_calibration="inner group-held-out temperature and class intercepts",
        decision_threshold="per-fold top ceil(10%); shared seeded random ties"),indent=2),encoding="utf-8")


if __name__ == "__main__":
    run()
