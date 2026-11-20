"""Audit repeat controls using unordered pairs of independently registered sample IDs.

The original control uses marginal within-condition discordance. Enumerating SPID
pairs reproduces this result and permits an exact range check on all four records.
"""
from pathlib import Path
from itertools import combinations
import json

import numpy as np
import pandas as pd


from paths import DATA, RESULTS as OUT
CODES = {
    "AR_challenge": (1816, 765), "ER_challenge": (2053, 789),
    "AR_blocker": (764, 1822), "ER_blocker": (788, 2211),
}
SUB = ["structure_id", "dsstox_substance_id"]
PAIR = SUB + ["spid_i", "spid_j"]
MEASURES = ["across", "within_a", "within_b", "net_b_minus_a", "across_minus_a", "across_minus_b"]
BOUNDS = ["conc_min", "conc_max"]
SEED = 20261121
N_BOOT = 5000


def load():
    ep = [val for pair in CODES.values() for val in pair]
    raw = pd.read_parquet(DATA / "release_activity.parquet", filters=[("aeid", "in", ep)])
    keep = raw.hitcall.ge(0) & raw.conc_unit.eq("uM")
    mapped = pd.read_csv(DATA / "substance_to_structure.csv", usecols=["dtxsid", "structure_id"])
    valid = raw.loc[keep].dropna(subset=["spid", SUB[1]]).merge(
        mapped.rename(columns={"dtxsid": SUB[1]}), on=SUB[1], validate="many_to_one"
    )
    key = SUB + ["aeid", "spid"]
    assert not valid.duplicated(key).any(), "This frozen release has one record per sample–endpoint key."
    assert valid.groupby("spid")[SUB[1]].nunique().max() == 1
    assert valid[BOUNDS].notna().all().all() and valid.conc_min.le(valid.conc_max).all()
    assert valid.conc_min.gt(0).all()
    valid["y"] = valid.hitcall.ge(.9).astype(int)
    return valid, dict(n_raw=len(raw), n_mapped_valid=len(valid), n_distinct_sample_endpoint_keys=len(valid[key].drop_duplicates()))


def make_pairs(valid, registry):
    all_pairs = []
    summaries = []
    patterns = []
    matched_list = []
    for contrast, (a, b) in CODES.items():
        cols = SUB + ["spid", "y"] + BOUNDS
        matched = valid.loc[valid.aeid.eq(a), cols].merge(
            valid.loc[valid.aeid.eq(b), cols], on=SUB + ["spid"], suffixes=("_a", "_b"), validate="one_to_one"
        )
        matched["across"] = matched.y_a.ne(matched.y_b).astype(float)
        matched["net_b_minus_a"] = matched.y_b - matched.y_a
        per_structure = matched.groupby(SUB)[["across", "net_b_minus_a"]].mean().groupby("structure_id").mean()
        summaries.append(dict(contrast=contrast, structures=len(per_structure),
                              substances=len(matched[SUB].drop_duplicates()), matched_spids=len(matched),
                              **per_structure.mean().to_dict()))
        matches = matched[SUB + ["spid", "across"]].copy()
        matches["contrast"] = contrast
        matched_list.append(matches)
        paired = matched.merge(matched, on=SUB, suffixes=("_i", "_j"))
        paired = paired.loc[paired.spid_i.lt(paired.spid_j)].copy()
        paired["contrast"] = contrast
        paired["across"] = (paired.across_i + paired.across_j) / 2
        paired["within_a"] = paired.y_a_i.ne(paired.y_a_j).astype(float)
        paired["within_b"] = paired.y_b_i.ne(paired.y_b_j).astype(float)
        paired["net_b_minus_a"] = (paired.net_b_minus_a_i + paired.net_b_minus_a_j) / 2
        paired["across_minus_a"] = paired.across - paired.within_a
        paired["across_minus_b"] = paired.across - paired.within_b
        paired["range_per_spid"] = True
        paired["range_all_four"] = True
        paired["range_all_four_close"] = True
        for name in BOUNDS:
            four = [f"{name}_{side}_{spid}" for side in ("a", "b") for spid in ("i", "j")]
            for sample in ["i", "j"]:
                paired["range_per_spid"] &= paired[f"{name}_a_{sample}"].eq(paired[f"{name}_b_{sample}"])
            paired["range_all_four"] &= paired[four].min(axis=1).eq(paired[four].max(axis=1))
            for c1, c2 in combinations(four, 2):
                paired["range_all_four_close"] &= np.isclose(paired[c1], paired[c2]) & np.isclose(paired[c2], paired[c1])
        assert (paired.range_all_four <= paired.range_per_spid).all()
        assert (paired.range_all_four <= paired.range_all_four_close).all()
        # When each SPID's A/B bounds match, compare the two lower and upper bounds.
        fold_diffs = []
        for name in BOUNDS:
            bounds = paired[[f"{name}_a_i", f"{name}_a_j"]]
            fold_diffs.append(bounds.max(axis=1) / bounds.min(axis=1))
        paired["max_fold_between_spids"] = np.maximum(*fold_diffs).where(paired.range_per_spid)
        all_pairs.append(paired)
        by_sub = paired.groupby(SUB).agg(
            all10=("net_b_minus_a", lambda x: x.eq(-1).all()),
            all01=("net_b_minus_a", lambda x: x.eq(1).all()),
            any_cross=("across", lambda x: x.gt(0).any()))
        patterns.append(dict(contrast=contrast, repeated_substances=len(by_sub), any_cross=int(by_sub.any_cross.sum()),
                             all_10=int(by_sub.all10.sum()), all_01=int(by_sub.all01.sum()),
                             varying=int((by_sub.any_cross & ~by_sub.all10 & ~by_sub.all01).sum()),
                             no_cross=int((~by_sub.any_cross).sum())))
    joined = pd.concat(all_pairs, ignore_index=True).merge(registry, on="structure_id", validate="many_to_one")
    matched = pd.concat(matched_list, ignore_index=True)
    assert not joined.duplicated(["contrast"] + PAIR).any()
    return joined, matched, pd.DataFrame(summaries), pd.DataFrame(patterns)


def common(data, keys):
    return data.loc[data.groupby(keys).contrast.transform("nunique").eq(len(CODES))]


def summarize(data, pool, rule, seed):
    rows = []
    for i, contrast in enumerate(CODES):
        subset = data.loc[data.contrast.eq(contrast)]
        joined_spids = pd.concat([subset[SUB + ["spid_i"]].rename(columns={"spid_i": "spid"}),
                                  subset[SUB + ["spid_j"]].rename(columns={"spid_j": "spid"})]).drop_duplicates()
        substance = subset.groupby(["scaffold"] + SUB, sort=True)[MEASURES].mean()
        structure = substance.groupby(["scaffold", "structure_id"], sort=True)[MEASURES].mean()
        summed = structure.groupby("scaffold").sum()
        sizes = structure.groupby("scaffold").size()
        total = len(structure)
        header = dict(pool=pool, range_rule=rule, contrast=contrast, structures=total, substances=len(substance),
                      shared_spids=len(joined_spids), spid_pairs=len(subset), scaffolds=len(sizes))
        if total == 0 or rule in {"all_four_exact", "all_four_isclose", "all_sixteen_exact"}:
            # Only 1–6 structures and 1–4 scaffolds support the frozen exact-range subsets.
            assert total <= 6
            reason = "empty" if total == 0 else "descriptive_sparse_exact_range"
            rows.extend({**header, "measure": col, "estimate": structure[col].mean(), "lo": np.nan,
                         "hi": np.nan, "interval_status": reason}
                        for col in MEASURES)
            continue
        rng = np.random.default_rng([SEED, seed, i])
        boots = []
        for start in range(0, N_BOOT, 200):
            weights = rng.multinomial(len(sizes), np.ones(len(sizes)) / len(sizes), size=min(200, N_BOOT - start))
            boots.append((weights @ summed.to_numpy()) / (weights @ sizes.to_numpy())[:, None])
        lo, hi = np.quantile(np.concatenate(boots), [.025, .975], axis=0)
        for j, col in enumerate(MEASURES):
            informative = structure[col].nunique() > 1
            rows.append({**header, "measure": col, "estimate": structure[col].mean(),
                         "lo": lo[j] if informative else np.nan, "hi": hi[j] if informative else np.nan,
                         "interval_status": "scaffold_percentile" if informative else "constant_observation"})
    return rows


def verify_old(single, patterns, replicated, pairs):
    original = pd.read_csv(OUT / "sample_control_summary.csv")
    old_patterns = pd.read_csv(OUT / "sample_control_patterns.csv").set_index("contrast")
    pd.testing.assert_frame_equal(patterns.set_index("contrast").sort_index(axis=1).sort_index(),
                                  old_patterns.sort_index(axis=1).sort_index())
    detailed = pd.read_parquet(OUT / "sample_control_substances.parquet")
    detailed = detailed.loc[detailed["count"].gt(1)].set_index(["contrast"]+SUB)
    paired = pairs.groupby(["contrast"] + SUB)[MEASURES[:4]].mean()
    pd.testing.assert_frame_equal(paired.sort_index(), detailed[MEASURES[:4]].sort_index(),
                                  check_exact=False, rtol=1e-13, atol=1e-15)
    # Independently reconstruct the existing interval table on its original draws.
    interval_ref = pd.read_csv(OUT/"sample_control_intervals.csv").set_index(["contrast", "measure"])
    interval_rng = np.random.default_rng(SEED)
    for contrast in CODES:
        groups = pairs.loc[pairs.contrast.eq(contrast)].groupby(["scaffold"]+SUB)[MEASURES].mean()
        by_structure = groups.groupby(["scaffold", "structure_id"]).mean()
        by_scaffold = by_structure.groupby("scaffold")
        sums, sizes = by_scaffold.sum().to_numpy(), by_scaffold.size().to_numpy()
        weights = interval_rng.multinomial(len(sizes), np.ones(len(sizes))/len(sizes), size=N_BOOT)
        lower, upper = np.quantile((weights@sums)/(weights@sizes)[:,None], [.025,.975], axis=0)
        reference = interval_ref.loc[contrast].loc[MEASURES]
        np.testing.assert_allclose(reference["lo"].to_numpy(), lower, rtol=1e-13, atol=1e-15)
        np.testing.assert_allclose(reference["hi"].to_numpy(), upper, rtol=1e-13, atol=1e-15)
    for _, row in original.iterrows():
        if row["subset"] == "same_spid":
            redo = single.set_index("contrast").loc[row.contrast]
            for col in ["structures", "substances", "matched_spids", "across", "net_b_minus_a"]:
                np.testing.assert_allclose(redo[col], row[col], rtol=1e-13, atol=1e-15)
        else:
            sel = replicated.loc[(replicated.pool == "contrast_specific") &
                                 (replicated.range_rule == "unrestricted") &
                                 (replicated.contrast == row.contrast)].set_index("measure")
            for key in ["structures", "substances", "shared_spids"]:
                assert sel.iloc[0][key] == row["matched_spids" if key == "shared_spids" else key]
            for col in MEASURES[:4]:
                np.testing.assert_allclose(sel.loc[col, "estimate"], row[col], rtol=1e-13, atol=1e-15)


def main():
    valid, audit = load()
    registry = pd.read_csv(DATA / "structures.csv", usecols=["scaffold", "structure_id"], keep_default_na=False)
    pairs, matched, single, patterns = make_pairs(valid, registry)
    results = []
    rules = {"unrestricted": pairs, "within_spid_exact": pairs.loc[pairs.range_per_spid],
             "all_four_exact": pairs.loc[pairs.range_all_four],
             "all_four_isclose": pairs.loc[pairs.range_all_four_close]}
    for tolerance in [1, 2, 5]:
        rules[f"within_spid_exact_between_{tolerance}pct"] = pairs.loc[
            pairs.max_fold_between_spids.le(1 + tolerance/100)]
    for j, (rule, eligible) in enumerate(rules.items()):
        sets = {"contrast_specific": eligible,
                "common_structures": common(eligible, ["structure_id"]),
                "common_spid_pairs": common(eligible, PAIR)}
        for k, (pool, panel) in enumerate(sets.items()):
            results.extend(summarize(panel, pool, rule, 10*j+k))
    # Further require identical minima and maxima across all eight endpoints and both SPIDs.
    strict = common(rules["all_four_exact"], PAIR).copy()
    groups = strict.groupby(PAIR)
    strict = strict.loc[groups.conc_min_a_i.transform("nunique").eq(1) &
                        groups.conc_max_a_i.transform("nunique").eq(1)]
    results.extend(summarize(strict, "common_spid_pairs", "all_sixteen_exact", 90))
    results = pd.DataFrame(results)
    verify_old(single, patterns, results, pairs)
    inventory_cols = ["contrast"] + PAIR + ["range_per_spid", "range_all_four", "range_all_four_close",
                                             "max_fold_between_spids"]
    inventory_cols += [f"{name}_{side}_{tag}" for name in ["y", *BOUNDS] for side in ["a", "b"] for tag in ["i", "j"]]
    pairs[inventory_cols].to_csv(OUT/"sample_range_pair_inventory.csv", index=False)
    results.to_csv(OUT/"sample_range_summary.csv", index=False, na_rep="NA")
    diagnostic = pairs.loc[pairs.range_per_spid].groupby("contrast").max_fold_between_spids
    diagnostic.agg(spid_pairs="size", q10=lambda x: x.quantile(.1), q25=lambda x: x.quantile(.25),
                   median="median", q75=lambda x: x.quantile(.75), q90=lambda x: x.quantile(.9),
                   maximum="max").to_csv(OUT/"sample_range_boundary_shifts.csv")
    # Report the number of distinct structures when an intersection shared only by ID is requested.
    coverage = matched.groupby("structure_id").contrast.nunique().value_counts().sort_index()
    audit.update({"reproduced_all_previous_sample_summary_counts_and_point_estimates": True,
                  "reproduced_all_preexisting_scaffold_percentile_limits": True,
                  "reproduced_every_repeated_substance_measure": True,
                  "reproduced_previous_substance_pattern_counts": True,
                  "sample_endpoint_pairs_have_unique_raw_measurement": True,
                  "n_structures_by_number_of_contrasts_with_any_shared_spid": {str(k): int(v) for k,v in coverage.items()},
                  "raw_record_bounds_missing": 0, "four_contrasts_share_same_scaffold_source": True,
                  "four_exact_vs_symmetric_default_isclose_pair_membership_differences": int((pairs.range_all_four != pairs.range_all_four_close).sum()),
                  "pair_weighting": "equal unordered SPID pairs within registered substance, equal substances within structure, equal structures in estimate",
                  "n_bootstrap": N_BOOT, "seed": SEED, "rtol": 1e-5, "atol": 1e-8})
    (OUT/"sample_range_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(results.loc[results.measure.eq("across"), ["pool","range_rule","contrast","structures","shared_spids","estimate"]].to_string(index=False))


if __name__ == "__main__":
    main()
