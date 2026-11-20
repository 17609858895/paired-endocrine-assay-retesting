# Paired endocrine assay responses

Code to reconstruct four androgen and estrogen receptor assay comparisons from ToxCast invitrodb v4.3, evaluate repeat samples, predict paired calls on held-out chemical scaffolds, and recover paired dose responses from the independent AR-CALUX ring trial.

## Inputs

Download the original files and use the local names listed below. The AR-CALUX files should share one directory. The EPA ZIP can remain compressed.

| Local filename | Public source | Content used |
|---|---|---|
| `INVITRODB_SUMMARY.zip` | [EPA v4.3 archive](https://clowder.edap-cluster.com/api/files/68af6b70e4b02565fc7c3a98/blob) | Winning-model activity CSV, ar.er.qsar.xlsx and ar.er.lit.xlsx |
| `AR_CALUX_JRC_main.pdf` | [JRC validation study report](https://op.europa.eu/en/publication-detail/-/publication/c86d54bb-c70a-11ea-adf7-01aa75ed71a1/language-en) | English PDF with the blind-code key in Annex 13.4 |
| `AR_CALUX_JRC_statistical.pdf` | [JRC statistical report](https://op.europa.eu/en/publication-detail/-/publication/9536b72d-c70b-11ea-adf7-01aa75ed71a1/language-en) | English 340-page PDF with the concentration tables in Appendix D |
| `AR_CALUX_2021.xml` | [Europe PMC full-text XML](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC8557474/fullTextXML) | Published outcome tables |
| `AR_CALUX_2021_supplement.zip` | [Europe PMC supplementary files](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC8557474/supplementaryFiles) | Supplementary chemical identifiers |

Input fingerprints used in development are in `source_hashes.tsv`. The two JRC source documents must be the regular downloadable PDFs in English. Page layout is used to assign dose cells to runs and laboratories. A publisher update to an XML wrapper or ZIP compression may change the checksum even if its scientific content is unchanged.

## Run

The analysis environment used Python 3.12.3. Install the Python dependencies and make Poppler's `pdftotext` executable available on the system PATH for the ring-trial extraction and independent PDF check.

```sh
python -m pip install -r requirements.txt
python run_analysis.py --archive /path/to/INVITRODB_SUMMARY.zip --calux-sources /path/to/calux --workdir /path/to/derived
```

Stages can also be run separately with `--stages prepare`, `--stages assay`, `--stages prediction`, or `--stages ring_trial`. Run preparation before the other stages. Prediction uses the assay-stage sample-pair inventory for the final recurrence calculation. The work directory contains all derived files. The input ZIP is approximately 7.5 GB. Extraction reads the full winning-model CSV to establish the chemical universe, then retains the 12 endpoints used for the paired and viability analyses.

## Outputs and denominators

| Output under `results/` | What it measures |
|---|---|
| `input_audit.json`, `context_contrasts.csv`, `paired_sensitivity.csv`, `paired_potency.csv` | Identity resolution, response counts, net activity differences, scaffold uncertainty and in-range paired potency |
| `sample_control_summary.csv`, `sample_range_summary.csv`, `state_recurrence.csv` | Same-sample versus within-condition differences, shared sample cohorts, and conditional repeat states |
| `dependence_summary.csv`, `dependence_direct_calibrated_summary.csv`, `nonlinear_summary.csv` | Joint probabilities and four-state held-out log loss across the same global scaffold folds |
| `rank_uncertainty_models.csv`, `rank_uncertainty_paired.csv`, `budget_results.csv`, `top_decile_list_overlap.csv` | Fold-stratified retest yield, conditional bootstrap intervals, alternative budgets and the overlap between retest lists |
| `external_calux_named_chemicals.csv`, `external_calux_paired_doses.csv`, `external_calux_pdf_check.csv`, `all_common_windows.csv` | Identity overlap, actual paired dose readings, numerical comparison of two PDF parsers and cross-laboratory concentration windows |

Four tasks use 16,873 distinct structure–contrast records across 6,203 unique chemical structures. Chemical fingerprints contain 1,024 Morgan bits of radius 2. Conflicting structure–endpoint labels are excluded from prediction. Sample variability is estimated from the original valid sample records, which preserve disagreements. Each sample pair is considered in both orientations for state recurrence, with equal weight for each structure, for its substances and for ordered pairs within each substance before conditioning on the index state.

The primary retest budget is the top 10% within each test fold, rounded upward. Saved scores also support 5%, 15% and 20% sensitivity checks. Random expectation applies the actual fold-specific budget to its own observed positive fraction. Interval estimates resample scaffolds within each outer fold while keeping fitted predictions fixed. The external data comprise 1,404 matched concentrations from 177 runs across 29 substances. Nineteen of these have both ToxCast AR challenge readouts. Shared windows can reuse substances and experiments.
