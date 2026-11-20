"""Recalculate matched-state and potency contrasts from the public-release pair table."""
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd


from paths import WORK as SOURCE, RESULTS as OUTPUT
CONTRASTS = {
    "AR_challenge": (1816, 765),
    "ER_challenge": (2053, 789),
    "AR_blocker": (764, 1822),
    "ER_blocker": (788, 2211),
}
SUBSETS = ("all", "equal_range", "both_viability_inactive", "equal_range_and_viability")
N_BOOT = 5000
SEED = 20261121


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def subset_rows(pairs, name, has_viability):
    if "viability" in name and not has_viability:
        return None
    selected = np.ones(len(pairs), dtype=bool)
    if "equal_range" in name:
        selected &= pairs.equal_concentration_range.to_numpy()
    if "viability" in name:
        selected &= pairs.viability_both_observed_inactive.to_numpy()
    return pairs.loc[selected]


def scaffold_draws(n_groups, rng):
    for start in range(0, N_BOOT, 100):
        size = min(100, N_BOOT - start)
        indices = rng.integers(n_groups, size=(size, n_groups))
        indexed = indices + np.arange(size)[:, None] * n_groups
        yield np.bincount(indexed.ravel(), minlength=size * n_groups).reshape(size, n_groups)


def largest_cluster(pairs):
    sizes = pairs.groupby("scaffold", sort=True).size()
    return sizes.idxmax(), int(sizes.max()), len(sizes)


def state_summary(pairs, rng):
    counts = pairs.groupby(["y_a", "y_b"]).size().to_dict()
    states = {f"n{a}{b}": int(counts.get((a, b), 0)) for a in (0, 1) for b in (0, 1)}
    delta = pairs.y_b.astype(int) - pairs.y_a.astype(int)
    grouped = delta.groupby(pairs.scaffold, sort=True)
    sums, sizes = grouped.sum().to_numpy(), grouped.size().to_numpy()
    estimates = np.concatenate([
        (draws @ sums) / (draws @ sizes) for draws in scaffold_draws(len(sizes), rng)
    ])
    lower, upper = np.quantile(estimates, [.025, .975])
    largest, n_largest, n_scaffolds = largest_cluster(pairs)
    outside = pairs.scaffold.ne(largest)
    return {
        "status": "available", "n": len(pairs), **states,
        "discordance": (states["n10"] + states["n01"]) / len(pairs),
        "net_change_b_minus_a": delta.mean(), "net_ci_low": lower, "net_ci_high": upper,
        "n_scaffolds": n_scaffolds, "largest_scaffold": largest,
        "largest_scaffold_n": n_largest, "largest_scaffold_fraction": n_largest / len(pairs),
        "n_outside_largest": int(outside.sum()), "net_without_largest": delta[outside].mean(),
    }


def potency_summary(pairs, rng):
    double_active = pairs[pairs.y_a.eq(1) & pairs.y_b.eq(1)]
    measured = double_active.dropna(subset=["ac50_a", "ac50_b", "delta_log10_ac50"])
    result = {"status": "available", "both_active_n": len(double_active), "potency_n": len(measured)}
    if measured.empty:
        return result
    measured = measured.sort_values("delta_log10_ac50")
    values = measured.delta_log10_ac50.to_numpy()
    group_index, groups = pd.factorize(measured.scaffold, sort=True)
    medians = []
    for draws in scaffold_draws(len(groups), rng):
        cumulative = np.cumsum(draws[:, group_index], axis=1)
        totals = cumulative[:, -1]
        # Median of cluster-resampled records with integer replication weights.
        index_lo = (cumulative <= ((totals - 1) // 2)[:, None]).sum(axis=1)
        index_hi = (cumulative <= (totals // 2)[:, None]).sum(axis=1)
        medians.extend((values[index_lo] + values[index_hi]) / 2)
    largest, n_largest, n_scaffolds = largest_cluster(measured)
    result.update({
        "potency_scaffolds": n_scaffolds, "potency_largest_scaffold": largest,
        "potency_largest_n": n_largest,
        "delta_q10": np.quantile(values, .10), "delta_q25": np.quantile(values, .25),
        "delta_median": np.median(values), "delta_q75": np.quantile(values, .75),
        "delta_q90": np.quantile(values, .90), "fraction_positive": (values > 0).mean(),
        "median_ci_low": np.quantile(medians, .025), "median_ci_high": np.quantile(medians, .975),
        "median_without_potency_largest": measured.loc[measured.scaffold.ne(largest), "delta_log10_ac50"].median(),
    })
    return result


def main():
    OUTPUT.mkdir(exist_ok=True, parents=True)
    pair_path = SOURCE / "data/context_pairs.parquet"
    structure_path = SOURCE / "data/structures.csv"
    benchmark_path = SOURCE / "results/context_contrasts.csv"
    pairs = pd.read_parquet(pair_path)
    pairs = pairs[pairs.contrast.isin(CONTRASTS)]
    structures = pd.read_csv(structure_path, usecols=["structure_id", "scaffold"])
    pairs = pairs.merge(structures, on="structure_id", how="left", validate="many_to_one")
    assert pairs.scaffold.notna().all()
    assert not pairs.duplicated(["structure_id", "contrast"]).any()
    assert set(pairs.y_a) == set(pairs.y_b) == {0, 1}
    assert np.array_equal(
        np.isclose(pairs.conc_min_a, pairs.conc_min_b) & np.isclose(pairs.conc_max_a, pairs.conc_max_b),
        pairs.equal_concentration_range,
    )
    assert np.array_equal(
        pairs.viability_a.eq(0) & pairs.viability_b.eq(0), pairs.viability_both_observed_inactive
    )
    assert np.allclose(np.log10(pairs.ac50_b) - np.log10(pairs.ac50_a), pairs.delta_log10_ac50, equal_nan=True)
    benchmark = pd.read_csv(benchmark_path).set_index("contrast")
    state_rows, potency_rows = [], []
    for i, (contrast, (a, b)) in enumerate(CONTRASTS.items()):
        family = pairs[pairs.contrast.eq(contrast)]
        assert family.aeid_a.eq(a).all() and family.aeid_b.eq(b).all()
        assert family.viability_a.notna().any() == ("challenge" in contrast)
        assert family.viability_b.notna().any() == ("challenge" in contrast)
        for j, subset in enumerate(SUBSETS):
            keys = {"contrast": contrast, "subset": subset, "aeid_a": a, "aeid_b": b}
            sample = subset_rows(family, subset, "challenge" in contrast)
            if sample is None:
                missing = {**keys, "status": "paired viability controls unavailable"}
                state_rows.append(missing)
                potency_rows.append(missing.copy())
                continue
            # Separate deterministic streams for each contrast, subset and estimator.
            result = state_summary(sample, np.random.default_rng([SEED, i, j, 0]))
            potency = potency_summary(sample, np.random.default_rng([SEED, i, j, 1]))
            assert sum(result[key] for key in ("n00", "n01", "n10", "n11")) == result["n"]
            assert np.isclose(result["net_change_b_minus_a"], (result["n01"] - result["n10"]) / len(sample))
            if subset == "all":
                old = benchmark.loc[contrast]
                for key in ("n", "n00", "n01", "n10", "n11"):
                    assert result[key] == old[key]
                assert potency["potency_n"] == old.potency_pairs
                assert np.isclose(potency["delta_median"], old.median_delta_log10_ac50)
            state_rows.append({**keys, **result})
            potency_rows.append({**keys, **potency})
    states, potency = pd.DataFrame(state_rows), pd.DataFrame(potency_rows)
    states.to_csv(OUTPUT / "paired_sensitivity.csv", index=False, na_rep="NA")
    potency.to_csv(OUTPUT / "paired_potency.csv", index=False, na_rep="NA")
    info = {
        "input_sha256": {p.name: sha256(p) for p in [pair_path, structure_path, benchmark_path]},
        "status": "passed", "matched_structure_contrast_rows": len(pairs),
        "unique_structures_in_four_contrasts": int(pairs.structure_id.nunique()),
        "represented_conditions": int(states.status.eq("available").sum()),
        "structurally_unavailable_conditions": int(states.status.ne("available").sum()),
        "bootstrap_replicates": N_BOOT, "base_seed": SEED,
        "estimand": "structure-weighted difference in activity prevalence, B minus A",
        "uncertainty": "percentile resampling of observed Murcko scaffold groups, including the pooled ring-free group",
        "input_range_inversions": int((pairs.conc_min_a.gt(pairs.conc_max_a) | pairs.conc_min_b.gt(pairs.conc_max_b)).sum()),
        "checks": ["unique structure within each pair", "source endpoint IDs", "binary labels", "range flags",
                   "viability flags", "potency deltas", "all four full-set cell counts and potency medians reproduce the input summaries"],
    }
    (OUTPUT / "paired_sensitivity_checks.json").write_text(json.dumps(info, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
