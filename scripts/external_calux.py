"""Audit JRC AR-CALUX identity, paired-measurement availability, and CAS overlap."""
from pathlib import Path
from io import BytesIO
from zipfile import ZipFile
import re
import json
import hashlib
import subprocess
import unicodedata

import fitz
import pandas as pd
from lxml import etree

from paths import EVIDENCE as E, RESULTS as O, DATA as D
HEAD = re.compile(r"Table\s+(\d+):\s*Test item (\d+)\.\s*(Relative induction Yc|Specificity control Sc)", re.I)


def norm(x):
    s = unicodedata.normalize("NFKC", x).lower()
    s = re.sub("[^a-z0-9αβ]", "", s)
    aliases = {
        "benzylbutylphthalate":"butylbenzylphthalate", "17αethynylestradiol":"17αethinylestradiol",
        "fluoxymestrone":"fluoxymesterone", "245t":"245trichlorophenoxyaceticacid",
        "nandrolone":"19nortestosterone","nandrolone19nortestosterone":"19nortestosterone",
        "norethinodrone":"norethindrone","norethinodroneacetate":"norethindroneacetate",
        "ici182780fulvestrant":"fulvestrant"
    }
    return aliases.get(s,s)


def identities():
    direct = fitz.open(E/"AR_CALUX_JRC_main.pdf")
    p = direct[147].get_text(sort=True)
    codes = []
    for line in p.splitlines():
        m = re.match(r"\s*(\d+)\s+(.+?)\s+(CIT|SPT)-[23]-",line)
        if m:
            codes.append((int(m[1]),m[2].strip()))
    assert [r[0] for r in codes] == list(range(1,47))
    with ZipFile(E/"AR_CALUX_2021_supplement.zip") as archive:
        with ZipFile(BytesIO(archive.read("kfab078_supplementary_data.docx"))) as w:
            doc = etree.fromstring(w.read("word/document.xml"))
    ns = {"w":doc.nsmap["w"]}
    tbl = doc.xpath("w:body/w:tbl", namespaces=ns)[0]
    lookup = {}
    for row in tbl.xpath("w:tr",namespaces=ns)[1:]:
        cells = ["".join(c.xpath(".//w:t/text()",namespaces=ns)).strip()
                 for c in row.xpath("w:tc",namespaces=ns)]
        lookup[norm(cells[0])] = cells[1]
    report_text = subprocess.check_output(["pdftotext", "-layout", "-enc", "UTF-8", str(E / "AR_CALUX_JRC_main.pdf"), "-"]).decode("utf-8")
    for name, cas in [("2-tert-Butylanthraquinone","84-47-9"),("Arochlor1254","11097-69-1")]:
        assert any(name.lower() in line.lower() and cas in line for line in
                   report_text.splitlines())
        lookup[norm(name)] = cas
    xml = etree.fromstring((E/"AR_CALUX_2021.xml").read_bytes())
    rows = xml.xpath('//table-wrap[@id="kfab078-T7"]//tbody/tr')
    final = {}
    for row in rows:
        cells = ["".join(c.itertext()).strip() for c in row]
        final[norm(cells[0])] = cells[8]
    t4_rows = xml.xpath('//table-wrap[@id="kfab078-T4"]//tbody/tr')
    out = []
    for row in t4_rows:
        cells = ["".join(c.itertext()).strip() for c in row]
        if len(cells)==9 and norm(cells[0]) in lookup:
            out.append(dict(chemical=cells[0], **{f"ant_final_lab{i+1}":cells[i+5] for i in range(4)}))
    assert len(out)==20
    pd.DataFrame(out).to_csv(O/"external_calux_article_table4.csv",index=False)
    table = pd.DataFrame(codes,columns=["item","chemical"])
    table["casrn"] = table.chemical.map(lambda v: lookup[norm(v)])
    table["article_final_antagonist"] = table.chemical.map(lambda v:final.get(norm(v)))
    assert table.casrn.is_unique and table.article_final_antagonist.notna().sum()==44
    return table


def inventory():
    report = fitz.open(E/"AR_CALUX_JRC_statistical.pdf")
    found=[]
    for i,p in enumerate(report):
        for m in HEAD.finditer(" ".join(p.get_text().split())):
            number, item, heading = m.groups()
            if int(number)<136:
                continue
            found.append(dict(table=int(number),run_item=int(item),item=3 if int(item)==47 else int(item),
                              pdf_page=i+1,measurement="Sc" if heading.startswith("Specificity") else "Yc"))
    found=pd.DataFrame(found)
    assert set(found.table)==set(range(136,213))
    return found


def main():
    chems = identities()
    tab = inventory()
    tab.to_csv(O/"external_calux_table_inventory.csv",index=False)
    n = tab.groupby(["item","measurement"]).size().unstack(fill_value=0)
    chems = chems.join(n,on="item").rename(columns={"Yc":"yc_tables","Sc":"sc_tables"})
    local = pd.read_csv(D/"substance_to_structure.csv",dtype={"casrn":str})
    pairs = pd.read_parquet(D/"context_pairs.parquet",columns=["contrast","structure_id","y_a","y_b"])
    pairs = pairs[pairs.contrast.eq("AR_challenge")].drop(columns="contrast")
    candidates=chems[["item","casrn"]].merge(local[["casrn","dtxsid","structure_id"]],on="casrn",how="inner")
    matches=candidates.merge(pairs,on="structure_id",how="left",validate="many_to_one")
    matches.to_csv(O/"external_calux_cas_identity_rows.csv",index=False)
    structure = matches.drop_duplicates(["item","structure_id"])
    assert not structure.duplicated("item").any(), "Ambiguous structural identity"
    chems = chems.merge(structure[["item","structure_id","y_a","y_b"]],on="item",how="left",validate="one_to_one")
    chems["mapped_structure"]=chems.structure_id.notna()
    chems["has_toxcast_pair"]=chems.y_a.notna() & chems.y_b.notna()
    chems.to_csv(O/"external_calux_named_chemicals.csv",index=False)
    stats = {"chemical_identities":len(chems),"yc_chemical_identities":int(chems.yc_tables.gt(0).sum()),
             "sc_chemical_identities":int(chems.sc_tables.gt(0).sum()),"named_article_final":int(chems.article_final_antagonist.notna().sum()),
             "mapped_local":int(chems.mapped_structure.sum()),"local_ar_pair":int(chems.has_toxcast_pair.sum()),
             "local_ar_pair_with_sc":int((chems.has_toxcast_pair & chems.sc_tables.gt(0)).sum()),
             "table4_substances":len(pd.read_csv(O/"external_calux_article_table4.csv")),
             "source_sha256":{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in E.glob("AR_CALUX*")
                              if p.suffix in {".pdf",".zip",".xml"}}}
    (O/"external_calux_inventory.json").write_text(json.dumps(stats,indent=2),encoding="utf-8")
    print(stats)


if __name__=="__main__":
    main()
