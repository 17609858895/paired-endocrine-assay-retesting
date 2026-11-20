"""Independent PDF text extraction check on the numerical source cells."""
from collections import Counter
import json
import re
import subprocess
import shutil

import numpy as np
import pandas as pd
from external_calux import E, O


def run():
    cmd = shutil.which("pdftotext")
    if not cmd:
        raise RuntimeError("Poppler pdftotext is required for independent verification.")
    original = subprocess.check_output(
        [cmd,"-layout","-enc","UTF-8",str(E/"AR_CALUX_JRC_statistical.pdf"),"-"]
    ).decode("utf-8").replace("−","-").split("\f")
    cells = pd.read_csv(O/"external_calux_dose_cells.csv")
    paired = pd.read_csv(O/"external_calux_paired_doses.csv")
    checks = []
    for n, rows in cells.groupby("pdf_page"):
        raw = original[int(n)-1]
        pairs = re.findall(r"(?<![\w.])(-?\d+(?:\.\d+)?)\s*\(\s*(\d+(?:\.\d+)?)\s*\)",raw)
        actual = Counter((float(a),float(b)) for a,b in pairs)
        expected = Counter(rows.dropna(subset=["mean"])[["mean","sd"]].itertuples(index=False,name=None))
        missing = expected - actual
        surplus = actual - expected
        checks.append(dict(pdf_page=n, coordinate_cells=expected.total(), poppler_cells=actual.total(),
                           numeric_cells_missing_from_poppler=missing.total(),numeric_cells_only_in_poppler=surplus.total(),
                           only_coordinate=str(dict(missing)),only_poppler=str(dict(surplus))))
    status=pd.DataFrame(checks)
    status.to_csv(O/"external_calux_pdf_check.csv",index=False)
    # Directly read from JRC Tables 170–171 (single lab) and 136–137 (multi-lab).
    manual = [(22,"RISE",16,4.5e-8,105.2,277.2),(22,"RISE",16,1.5e-4,8.9,200.1)]
    for item, lab, run_id, dose, a, b in manual:
        target=paired[(paired.run_item==item) & paired.lab.eq(lab) & paired.run.eq(run_id)
                      & np.isclose(paired.dose_mg_ml,dose,rtol=1e-12,atol=0)]
        assert len(target)==1 and target.yc_mean.iloc[0]==a and target.sc_mean.iloc[0]==b
    for dose, sc in [(1.5e-5,223.),(5e-5,198.1),(1.5e-4,232.3)]:
        target=cells[(cells.run_item.eq(1)) & cells.lab.eq("Envigo") & cells.run.eq(5)
                     & cells.measurement.eq("Sc") & np.isclose(cells.dose_mg_ml,dose,rtol=1e-12,atol=0)]
        assert len(target)==1 and target["mean"].iloc[0]==sc
    print(status[status.numeric_cells_missing_from_poppler.ne(0)|status.numeric_cells_only_in_poppler.ne(0)].to_string(index=False)[:3200])
    assert status.numeric_cells_missing_from_poppler.sum() == status.numeric_cells_only_in_poppler.sum() == 0
    print("Poppler comparison:",status[["coordinate_cells","poppler_cells","numeric_cells_missing_from_poppler","numeric_cells_only_in_poppler"]].sum().to_dict())


if __name__=="__main__":
    run()
