"""Recover numeric concentration tables from Appendix D of the public JRC report."""
import re
import json
import numpy as np
import pandas as pd
import fitz
from external_calux import E, O, HEAD

NUMBER = r"[+\-]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[+\-]?\d+)?"
CELL = re.compile(rf"({NUMBER})\s*\(\s*({NUMBER})\s*\)", re.I)
LABS = {"Citoxlab","Envigo","RISE","BDS"}


def lines(page):
    out = []
    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            s = "".join(span["text"] for span in line["spans"]).strip().replace("−","-")
            x0,y0,x1,y1=line["bbox"]
            cx,cy=(x0+x1)/2,(y0+y1)/2
            direction=line["dir"]
            u,v=(-cy,cx) if direction[1]<-.9 else (cx,cy)
            out.append((s,u,v,direction))
    return out


def parse(segment, item, table, side, pdf_page):
    heading=next(l for l in reversed(segment) if l[0]=="[mg/ml]")
    _, ux, vy, direction=heading
    body=[l for l in segment if l[3]==direction]
    runs=sorted([l for l in body if abs(l[2]-vy)<17 and
                 re.fullmatch(r"run\s+\d+",l[0])],key=lambda l:l[1])
    labnames=sorted([l for l in body if l[0] in LABS and vy-40<l[2]<vy],key=lambda l:l[1])
    # All lab blocks have three accepted biological runs. Their ordering is explicit in the header.
    assert len(labnames)*3==len(runs) and len(runs) in [3,12], (table,runs,labnames)
    phases=[l for l in body if abs(l[2]-vy)<2 and l[0] in {"COM","PRE"}]
    centers=np.array([v[1] for v in runs])
    cuts=np.r_[(ux+centers[0])/2,(centers[:-1]+centers[1:])/2,np.inf]
    # Locate each concentration by the printed mg/ml column (shared across both arms).
    numeric_row=[l for l in body if l[2]>vy+2 and abs(l[1]-ux)<3 and re.fullmatch(NUMBER,l[0],re.I)]
    data=[]
    for conc, _, v, _ in numeric_row:
        row=[l for l in body if abs(l[2]-v)<2.0 and l[1]>=cuts[0]]
        for i, (run,_,_,_) in enumerate(runs):
            chunks=[l for l in row if cuts[i] <= l[1]<cuts[i+1]]
            text="".join(l[0] for l in sorted(chunks,key=lambda l:l[1])).replace(" ","")
            if not text:
                continue
            m=CELL.fullmatch(text)
            phase=""
            if phases:
                phase=min(phases,key=lambda l:abs(l[1]-centers[i]))[0]
            data.append(dict(run_item=item,item=3 if item==47 else item,lab=labnames[i//3][0],
                             run=int(run.split()[-1]),dose_mg_ml=float(conc),measurement=side,
                             mean=float(m[1]) if m else np.nan,sd=float(m[2]) if m else np.nan,
                             cell_text=text,phase=phase,pdf_page=pdf_page,table=table))
    assert any(pd.notna(x["mean"]) for x in data), (table,heading)
    return data


def main():
    inv=pd.read_csv(O/"external_calux_table_inventory.csv")
    doc=fitz.open(E/"AR_CALUX_JRC_statistical.pdf")
    records=[]
    for page in inv.pdf_page.unique():
        all_lines=lines(doc[int(page)-1])
        start=0
        for i,line in enumerate(all_lines):
            m=HEAD.search(line[0])
            if m:
                tab,item,measure=m.groups()
                side="Yc" if measure.startswith("Relative") else "Sc"
                records.extend(parse(all_lines[start:i],int(item),int(tab),side,int(page)))
                start=i+1
    cells=pd.DataFrame(records)
    assert cells.table.nunique()==77
    ids=["run_item","item","lab","run","dose_mg_ml"]
    assert not cells.duplicated(ids+["measurement"]).any()
    cells.to_csv(O/"external_calux_dose_cells.csv",index=False)
    chems=pd.read_csv(O/"external_calux_named_chemicals.csv")
    valid=cells.dropna(subset=["mean"]).copy()
    valid["lowest_dose_mg_ml"]=valid.groupby(ids[:-1]+["measurement"]).dose_mg_ml.transform("min")
    first=valid[valid.lowest_dose_mg_ml.eq(valid.dose_mg_ml)][ids[:-1]+["measurement","mean"]]
    valid=valid.merge(first.rename(columns={"mean":"lowest_dose_mean"}),on=ids[:-1]+["measurement"],validate="many_to_one")
    valid["relative_pct"]=100*valid["mean"]/valid.lowest_dose_mean
    cols=["mean","sd","lowest_dose_mean","lowest_dose_mg_ml","relative_pct","table","pdf_page","phase"]
    def subset(code):
        return valid.loc[valid.measurement.eq(code),ids+cols].rename(columns={k:f"{code.lower()}_{k}" for k in cols})
    paired=subset("Yc").merge(subset("Sc"),on=ids,validate="one_to_one")
    paired=paired.merge(chems[["item","chemical","casrn","article_final_antagonist","has_toxcast_pair","y_a","y_b"]],
                        on="item",validate="many_to_one")
    paired.to_csv(O/"external_calux_paired_doses.csv",index=False)
    other=cells.loc[cells["mean"].isna(),"cell_text"].value_counts().to_dict()
    stats={"printed_nonempty_cells":len(cells),"parsed_mean_sd_cells":len(valid),
           "other_cells":other,"paired_doses":len(paired),"paired_unique_items":paired.item.nunique(),
           "paired_unique_runs":len(paired[ids[:-1]].drop_duplicates()),
           "paired_unique_items_with_toxcast_pair":paired.loc[paired.has_toxcast_pair,"item"].nunique()}
    (O/"external_calux_dose_qc.json").write_text(json.dumps(stats,indent=2),encoding="utf-8")
    print(stats)


if __name__=="__main__":
    main()
