"""Scaffold-cross-fitted experiment on paired ToxCast readouts. No model selection."""
import hashlib
import json
import os
from pathlib import Path

for name in ["OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
    os.environ[name] = "1"

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.sparse import csr_matrix
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

from paths import DATA, RESULTS as OUT
CONTRASTS = ["AR_challenge", "ER_challenge", "AR_blocker", "ER_blocker"]
SEED = 838
QCOLS = ["q00", "q01", "q10", "q11"]


def grouped_folds(structures):
    counts = structures.scaffold.value_counts().rename_axis("scaffold").reset_index(name="size")
    counts = counts.sort_values(["size", "scaffold"], ascending=[False, True], kind="stable")
    sizes = np.zeros(5, dtype=int)
    assignment = {}
    for scaffold, size in counts.itertuples(index=False, name=None):
        fold = int(sizes.argmin())
        assignment[scaffold] = fold
        sizes[fold] += size
    return structures.scaffold.map(assignment)


def binary_scores(x_train, y_train, x_pred):
    model = LogisticRegression(C=1.0, solver="liblinear", max_iter=1000, random_state=SEED)
    model.fit(x_train, y_train)
    assert model.n_iter_.max() < 1000
    return model.decision_function(x_pred)


def calibration(raw_train, y_train, raw_test):
    # Ridge penalized slope; intercept unpenalized, fixed before testing.
    def objective(b):
        z = b[0] + b[1] * raw_train
        loss = np.logaddexp(0.0, z) - y_train * z
        return loss.sum() + .5 * (b[1] - 1.0) ** 2
    fit = minimize(objective, np.array([0., 1.]), method="L-BFGS-B", bounds=[(-15, 15), (0.01, 8)])
    if not fit.success:
        raise RuntimeError(fit.message)
    convert = lambda z: np.clip(expit(fit.x[0] + fit.x[1] * z), 1e-5, 1-1e-5)
    return convert(raw_train), convert(raw_test)


def joint_prob(a, b, log_or):
    ratio = np.exp(np.clip(log_or, -12, 12))
    term = 1 + (ratio - 1) * (a + b)
    disc = np.maximum(0, term * term - 4 * (ratio - 1) * ratio * a * b)
    root = np.sqrt(disc)
    denominator = term + root
    # Avoid cancellation when the stable numerator and denominator are both tiny.
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(term >= 0, 2 * ratio * a * b / denominator, (term - root) / (2 * (ratio - 1)))
    return np.column_stack([1-a-b+t, b-t, a-t, t])


def design(a, b):
    left, right = logit(a), logit(b)
    return np.column_stack([.5 * (left+right), np.abs(left-right)])


def learn_or(a, b, states, test_a, test_b, adaptive):
    train = design(a,b)
    test = design(test_a,test_b)
    scale = np.maximum(train.std(axis=0), .01)
    center = train.mean(axis=0)
    train, test = (train-center)/scale, (test-center)/scale
    train = np.column_stack([np.ones(len(a)), train])[:, :3 if adaptive else 1]
    test = np.column_stack([np.ones(len(test_a)), test])[:, :train.shape[1]]
    def objective(coef):
        eta = train @ coef
        joint = joint_prob(a, b, eta)
        cells = joint[np.arange(len(a)), states]
        prob = np.maximum(cells, 1e-14)
        loss = -np.log(prob).sum() + .5 * np.square(coef[1:]).sum()
        # For fixed margins d(log OR)/d(q11) is the sum of all four reciprocals.
        dq11 = 1 / np.reciprocal(np.maximum(joint, 1e-300)).sum(axis=1)
        same = np.where((states == 0) | (states == 3), 1.0, -1.0)
        dloss = -same * dq11 / prob
        dloss[(cells <= 1e-14) | (np.abs(eta) >= 12)] = 0
        grad = train.T @ dloss
        grad[1:] += coef[1:]
        return loss, grad
    fit = minimize(objective, np.zeros(train.shape[1]), jac=True, method="L-BFGS-B", bounds=[(-10,10)] * train.shape[1])
    if not fit.success:
        raise RuntimeError(fit.message)
    return joint_prob(test_a,test_b,test @ fit.x), fit.x.tolist()


def evaluate(d):
    states = d.state.to_numpy()
    q = d[QCOLS].to_numpy()
    discordant = np.isin(states,[1,2])
    losses = -np.log(np.maximum(q[np.arange(len(d)),states], 1e-14))
    return {"n":len(d), "positives":int(discordant.sum()), "nll":float(losses.mean()),
            "discord_ap":float(average_precision_score(discordant, d.p_discord)),
            "top10_hits":int((d.selected_top10 & discordant).sum()),
            "top10_budget":int(d.selected_top10.sum())}


def run():
    structs = pd.read_csv(DATA / "structures.csv").sort_values("structure_id").reset_index(drop=True)
    fp = csr_matrix(np.load(DATA / "fingerprints.npz")["fp"],dtype=float)
    assert np.array_equal(structs.structure_id,np.arange(fp.shape[0]))
    structs["fold"] = grouped_folds(structs)
    assert structs.groupby("scaffold").fold.nunique().max() == 1
    structs[["structure_id","scaffold","fold"]].to_csv(OUT/"dependence_global_folds.csv",index=False)
    pairs = pd.read_parquet(DATA/"context_pairs.parquet", columns=["structure_id","y_a","y_b","contrast"])
    pairs = pairs[pairs.contrast.isin(CONTRASTS)].merge(structs[["structure_id","scaffold","fold"]],on="structure_id",validate="many_to_one")
    assert not pairs.duplicated(["structure_id","contrast"]).any()
    pairs["state"] = (2*pairs.y_a + pairs.y_b).astype(int)
    tie = np.random.default_rng(SEED).permutation(len(structs))
    results, coefficient_records = [], []
    for contrast in CONTRASTS:
        task = pairs[pairs.contrast.eq(contrast)].copy()
        for fold in range(5):
            train = task[task.fold.ne(fold)].reset_index(drop=True)
            test = task[task.fold.eq(fold)].reset_index(drop=True)
            train_ids = train.structure_id.to_numpy()
            test_ids = test.structure_id.to_numpy()
            assert set(train.scaffold).isdisjoint(test.scaffold)
            marginal = []
            for side in ["a","b"]:
                labels = train["y_"+side].to_numpy()
                inner_scores = np.empty(len(train))
                for inner in sorted(train.fold.unique()):
                    validation = train.fold.eq(inner).to_numpy()
                    inner_scores[validation] = binary_scores(fp[train_ids[~validation]],labels[~validation],fp[train_ids[validation]])
                final_scores = binary_scores(fp[train_ids],labels,fp[test_ids])
                marginal.append(calibration(inner_scores,labels,final_scores))
            (a,test_a),(b,test_b) = marginal
            counts = np.bincount(train.state,minlength=4) + .5
            probabilities = {"empirical":np.tile(counts / counts.sum(),(len(test),1)),
                             "independent":joint_prob(test_a,test_b,0.)}
            for method in ["constant_or","score_or"]:
                prob, coef = learn_or(a,b,train.state.to_numpy(),test_a,test_b,method=="score_or")
                probabilities[method] = prob
                coefficient_records.append({"contrast":contrast,"fold":fold,"model":method,"coefficients":json.dumps(coef)})
            classifier = LogisticRegression(C=1., solver="lbfgs", max_iter=1000, random_state=SEED)
            classifier.fit(fp[train_ids],train.state)
            assert classifier.n_iter_.max()<1000 and np.array_equal(classifier.classes_,np.arange(4))
            probabilities["multinomial"] = classifier.predict_proba(fp[test_ids])
            for method, q in probabilities.items():
                assert (q>0).all() and np.allclose(q.sum(axis=1),1.0)
                out = test[["contrast","structure_id","scaffold","fold","state"]].copy()
                out["model"] = method
                out[QCOLS] = q
                out["pA"] = q[:,2]+q[:,3]
                out["pB"] = q[:,1]+q[:,3]
                if method in ("independent","constant_or","score_or"):
                    assert np.allclose(out.pA,test_a,atol=1e-10,rtol=0.) and np.allclose(out.pB,test_b,atol=1e-10,rtol=0.)
                out["p_discord"] = q[:,1]+q[:,2]
                order = np.lexsort((tie[test_ids],-out.p_discord.to_numpy()))
                selected = np.zeros(len(out),dtype=bool)
                selected[order[:int(np.ceil(.1*len(test)))]]=True
                out["selected_top10"] = selected
                results.append(out)
            print(contrast,fold,"complete",flush=True)
    predictions=pd.concat(results,ignore_index=True)
    predictions.to_parquet(OUT/"dependence_predictions.parquet",index=False)
    pd.DataFrame(coefficient_records).to_csv(OUT/"dependence_coefficients.csv",index=False)
    for keys, filename in [(["contrast","model","fold"],"folds"),(["contrast","model"],"summary")]:
        metrics=[dict(zip(keys,key)) | evaluate(frame) for key,frame in predictions.groupby(keys,sort=False)]
        pd.DataFrame(metrics).to_csv(OUT/("dependence_"+filename+".csv"),index=False)
    intervals=[]
    for contrast, sub in predictions.groupby("contrast"):
        losses = sub[["structure_id","scaffold","model","state"]].copy()
        losses["nll"] = -np.log(np.maximum(sub[QCOLS].to_numpy()[np.arange(len(sub)),sub.state.to_numpy()],1e-14))
        table = losses.pivot(index=["structure_id","scaffold"],columns="model",values="nll").reset_index()
        for reference, comparison in [("independent","constant_or"),("constant_or","score_or"),("multinomial","score_or")]:
            delta=table[comparison]-table[reference]
            groups=delta.groupby(table.scaffold).agg(["sum","count"]).to_numpy()
            rng=np.random.default_rng(SEED)
            draws=[]
            for i in range(1500):
                sample=groups[rng.integers(0,len(groups),len(groups))].sum(axis=0)
                draws.append(sample[0]/sample[1])
            intervals.append(dict(contrast=contrast,reference=reference,comparison=comparison,
                                  delta_nll=delta.mean(),lo=np.quantile(draws,.025),hi=np.quantile(draws,.975),
                                  scaffolds=len(groups)))
    pd.DataFrame(intervals).to_csv(OUT/"dependence_interval.csv",index=False)
    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()
    manifest={"seed":SEED,"structure_pairs":len(pairs),"unique_structures":pairs.structure_id.nunique(),
              "predictions":len(predictions),"models":predictions.model.unique().tolist(),
              "script_sha256":digest(Path(__file__)),"data_sha256":{name:digest(DATA/name) for name in
              ["structures.csv","fingerprints.npz","context_pairs.parquet"]}}
    (OUT/"dependence_manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")


if __name__ == "__main__":
    run()
