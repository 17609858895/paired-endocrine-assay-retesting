"""Training-only calibration of the direct four-category model as a fairness check."""
from paired_dependence import (
    CONTRASTS, DATA, OUT, QCOLS, SEED, csr_matrix, evaluate, minimize, np, pd,
    LogisticRegression,
)
from scipy.special import logsumexp, softmax


def raw_logits(x, data, held):
    model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000, random_state=SEED)
    model.fit(x[data.structure_id], data.state)
    assert model.n_iter_.max() < 1000
    scores = model.predict_proba(x[held.structure_id])
    # Only possible in inner training splits. Assign each absent class the fixed
    # Jeffreys mass, rescaling observed classes to preserve their relative odds.
    missing = 4 - len(model.classes_)
    single = 0.5 / (len(data) + 2)
    prob = np.full((len(held), 4), single)
    prob[:, model.classes_] = (1 - missing * single) * scores
    return np.log(np.maximum(prob, 1e-14)), missing


def fit_calibrator(log_p, states):
    n = len(states)
    indicator = np.eye(4)[states]
    # Reference class is 00. One shared positive scale and three offsets.
    def objective(theta):
        offsets = np.r_[0., theta[1:]]
        values = theta[0] * log_p + offsets
        reg = theta - np.r_[1., 0., 0., 0.]
        loss = np.sum(logsumexp(values, axis=1) - values[np.arange(n), states])
        residual = softmax(values, axis=1) - indicator
        grad = np.r_[(residual * log_p).sum(), residual[:, 1:].sum(axis=0)] + reg
        return loss + 0.5 * (reg @ reg), grad
    fit = minimize(objective, [1., 0., 0., 0.], jac=True, method="L-BFGS-B",
                   bounds=[(.01, 8)] + [(-15, 15)] * 3)
    if not fit.success:
        raise RuntimeError(fit.message)
    return fit.x


def main():
    pred = pd.read_parquet(OUT / "dependence_predictions.parquet")
    roster = pred.loc[pred.model.eq("independent"),
                      ["structure_id", "scaffold", "contrast", "fold", "state"]].copy()
    fp = csr_matrix(np.load(DATA / "fingerprints.npz")["fp"], dtype=float)
    tie = np.random.default_rng(SEED).permutation(fp.shape[0])
    records, parameters = [], []
    for contrast in CONTRASTS:
        task = roster[roster.contrast.eq(contrast)].reset_index(drop=True)
        for outer in range(5):
            train = task[task.fold.ne(outer)].reset_index(drop=True)
            test = task[task.fold.eq(outer)].copy()
            inner_logits = np.empty((len(train), 4))
            missing_inner = 0
            for inner in sorted(train.fold.unique()):
                valid = train.fold.eq(inner)
                inner_logits[valid], missing = raw_logits(fp, train[~valid], train[valid])
                missing_inner += int(missing > 0)
            raw_test, missing = raw_logits(fp, train, test)
            assert not missing
            theta = fit_calibrator(inner_logits, train.state.to_numpy())
            q = softmax(theta[0] * raw_test + np.r_[0., theta[1:]], axis=1)
            test[QCOLS] = q
            test["pA"] = q[:, 2] + q[:, 3]
            test["pB"] = q[:, 1] + q[:, 3]
            test["p_discord"] = q[:, 1] + q[:, 2]
            test["model"] = "multinomial_calibrated"
            order = np.lexsort((tie[test.structure_id], -test.p_discord.to_numpy()))
            keep = np.zeros(len(test), dtype=bool)
            keep[order[:int(np.ceil(.1 * len(test)))]] = True
            test["selected_top10"] = keep
            records.append(test)
            parameters.append(dict(contrast=contrast, fold=outer, inner_with_missing_class=missing_inner,
                                   slope=theta[0], offset01=theta[1], offset10=theta[2], offset11=theta[3]))
        print(contrast, "calibrated", flush=True)
    result = pd.concat(records, ignore_index=True)
    result.to_parquet(OUT / "dependence_direct_calibrated_predictions.parquet", index=False)
    pd.DataFrame(parameters).to_csv(OUT / "dependence_direct_calibrated_parameters.csv", index=False)
    for keys, suffix in [(["contrast", "model"], "summary"), (["contrast", "model", "fold"], "folds")]:
        rows = [dict(zip(keys, k)) | evaluate(g) for k, g in result.groupby(keys, sort=False)]
        pd.DataFrame(rows).to_csv(OUT / f"dependence_direct_calibrated_{suffix}.csv", index=False)


if __name__ == "__main__":
    main()
