"""Build the paired assay table and global fingerprints from invitrodb v4.3."""
import argparse
import io
import json
import zipfile
from pathlib import Path
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.Scaffolds import MurckoScaffold
from paths import DATA, RESULTS

TASKS = {"AR_challenge": (1816, 765), "ER_challenge": (2053, 789),
         "AR_blocker": (764, 1822), "ER_blocker": (788, 2211)}
VIABILITY = {1816: 1817, 765: 766, 2053: 2054, 789: 790}


def canonical(s):
    if not isinstance(s, str):
        return None
    mol = Chem.MolFromSmiles(s)
    return Chem.MolToSmiles(mol, isomericSmiles=True) if mol is not None else None


def parent_identifier(mol):
    """Construct a conservative overlap key, separate from training identity."""
    try:
        parent = rdMolStandardize.FragmentParent(rdMolStandardize.Cleanup(mol))
        parent = rdMolStandardize.Uncharger().uncharge(parent)
        Chem.RemoveStereochemistry(parent)
        for atom in parent.GetAtoms():
            atom.SetIsotope(0)
        parent = rdMolStandardize.TautomerEnumerator().Canonicalize(parent)
        return Chem.MolToSmiles(parent, isomericSmiles=False)
    except (RuntimeError, ValueError):
        return None


def activity(archive):
    target = sorted({x for ab in TASKS.values() for x in ab} | set(VIABILITY.values()))
    member = [x for x in archive.namelist() if x.startswith("mc5-6_") and x.endswith(".csv")]
    if len(member) != 1:
        raise ValueError("Expected one mc5-6 winning-model CSV in the summary archive.")
    cols = ["dsstox_substance_id","spid","aeid","hitcall","ac50","conc_min","conc_max","conc_unit"]
    universe, kept, rows = set(), [], 0
    with archive.open(member[0]) as f:
        for chunk in pd.read_csv(f,usecols=cols,chunksize=60000,low_memory=False,
                                dtype={"dsstox_substance_id":"string","spid":"string","conc_unit":"string"}):
            universe.update(chunk.dsstox_substance_id.dropna().unique())
            rows += len(chunk)
            kept.append(chunk.loc[chunk.aeid.isin(target),cols])
    raw = pd.concat(kept,ignore_index=True)
    raw.to_parquet(DATA/"release_activity.parquet",index=False)
    return raw, universe, {"activity_member":member[0], "release_rows":rows,
                           "release_substances":len(universe),"core_endpoint_rows":len(raw)}


def structures(archive, universe):
    candidates = []
    for table in ["ar.er.qsar", "ar.er.lit"]:
        with archive.open(f"endocrine_models/{table}.xlsx") as f:
            source = pd.read_excel(io.BytesIO(f.read()),usecols=["dtxsid","casrn","canonical_smiles"])
        candidates.append(source.loc[source.dtxsid.isin(universe)].drop_duplicates())
    d = pd.concat(candidates,ignore_index=True).drop_duplicates().copy()
    d["smiles_canon"] = d.canonical_smiles.map(canonical)
    ok = d.dropna(subset=["smiles_canon"])
    ambiguous = ok.groupby("dtxsid").smiles_canon.nunique().loc[lambda s:s.gt(1)].index
    mapping = ok.loc[~ok.dtxsid.isin(ambiguous)].drop_duplicates("dtxsid").sort_values("dtxsid").copy()
    graphs = sorted(mapping.smiles_canon.unique())
    fp_generator = rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=1024)
    units, fps = [], []
    for i,s in enumerate(graphs):
        mol=Chem.MolFromSmiles(s)
        units.append(dict(structure_id=i,smiles_canon=s,parent_key=parent_identifier(mol),
                          scaffold=MurckoScaffold.MurckoScaffoldSmiles(mol=mol) or "<none>"))
        fps.append(fp_generator.GetFingerprintAsNumPy(mol))
    registry=pd.DataFrame(units)
    mapping=mapping.merge(registry[["smiles_canon","structure_id"]],on="smiles_canon",validate="many_to_one")
    mapping.to_csv(DATA/"substance_to_structure.csv",index=False)
    registry.to_csv(DATA/"structures.csv",index=False)
    np.savez_compressed(DATA/"fingerprints.npz",fp=np.array(fps,dtype="uint8"))
    return mapping, {"unambiguous_substances":len(mapping),"canonical_graphs":len(registry),
                     "conflicting_substance_structures":len(ambiguous)}


def pairs(raw,mapping):
    valid=raw.loc[raw.hitcall.ge(0)&raw.conc_unit.eq("uM")].merge(
        mapping[["dtxsid","structure_id"]],left_on="dsstox_substance_id",right_on="dtxsid",validate="many_to_one")
    valid["y"]=valid.hitcall.ge(.9).astype(int)
    valid["quantified"]=valid.ac50.where(valid.y.eq(1)&valid.ac50.gt(0)&valid.ac50.between(valid.conc_min,valid.conc_max))
    units=valid.groupby(["structure_id","aeid"],as_index=False).agg(
        y=("y","first"),label_count=("y","nunique"),n_records=("y","size"),ac50=("quantified","median"),
        conc_min=("conc_min","max"),conc_max=("conc_max","min"))
    units=units.loc[units.label_count.eq(1)].drop(columns="label_count")
    tables, checks=[], []
    for task,(a,b) in TASKS.items():
        p=units.loc[units.aeid.eq(a)].merge(units.loc[units.aeid.eq(b)],on="structure_id",
                                           suffixes=("_a","_b"),validate="one_to_one")
        p["contrast"]=task
        p["delta_log10_ac50"]=np.log10(p.ac50_b)-np.log10(p.ac50_a)
        p["equal_concentration_range"]=np.isclose(p.conc_min_a,p.conc_min_b)&np.isclose(p.conc_max_a,p.conc_max_b)
        p["common_concentration_range"]=np.minimum(p.conc_max_a,p.conc_max_b).ge(np.maximum(p.conc_min_a,p.conc_min_b))
        for side,eid in [("a",a),("b",b)]:
            result=units.loc[units.aeid.eq(VIABILITY.get(eid,-1))].set_index("structure_id").y
            p["viability_"+side]=p.structure_id.map(result)
        p["viability_both_observed_inactive"]=p.viability_a.eq(0)&p.viability_b.eq(0)
        row={"contrast":task,"n":len(p),"potency_pairs":int(p.delta_log10_ac50.notna().sum()),
             "median_delta_log10_ac50":p.delta_log10_ac50.median()}
        for u,v in [(0,0),(0,1),(1,0),(1,1)]:
            row[f"n{u}{v}"]=int((p.y_a.eq(u)&p.y_b.eq(v)).sum())
        tables.append(p)
        checks.append(row)
    panel=pd.concat(tables,ignore_index=True)
    panel.to_parquet(DATA/"context_pairs.parquet",index=False)
    pd.DataFrame(checks).to_csv(RESULTS/"context_contrasts.csv",index=False)
    return {"paired_structures":panel.structure_id.nunique(),"structure_contrast_pairs":len(panel)}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive",type=Path,help="EPA INVITRODB_SUMMARY.zip, version 4.3")
    args=parser.parse_args()
    RDLogger.DisableLog("rdApp.*")
    with zipfile.ZipFile(args.archive) as archive:
        raw, universe, audit=activity(archive)
        print("Read source activity",audit,flush=True)
        mapping, stats=structures(archive,universe)
    audit.update(stats)
    audit.update(pairs(raw,mapping))
    print(audit,flush=True)
    (RESULTS/"input_audit.json").write_text(json.dumps(audit,indent=2),encoding="utf8")


if __name__=="__main__":
    main()
