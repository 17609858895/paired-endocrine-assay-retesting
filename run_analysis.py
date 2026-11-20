"""Run the analysis using local copies of the original public releases."""
import argparse
import os
from pathlib import Path
import subprocess
import sys


STAGES = {
    "prepare": ["prepare_inputs"],
    "assay": ["paired_sensitivity", "sample_controls", "sample_range_sensitivity"],
    "prediction": ["paired_dependence", "paired_direct_calibration", "check_dependence",
                   "nonlinear_baselines", "rank_uncertainty", "repeat_states_and_budgets"],
    "ring_trial": ["external_calux", "external_calux_doses", "external_calux_check", "external_windows"],
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, help="EPA INVITRODB_SUMMARY.zip (v4.3)")
    parser.add_argument("--calux-sources", type=Path, help="Directory containing the four original JRC/Europe PMC files")
    parser.add_argument("--workdir", type=Path, required=True, help="Writable directory for all derived data and results")
    parser.add_argument("--stages", choices=STAGES, nargs="+", default=list(STAGES))
    args = parser.parse_args()
    if "prepare" in args.stages and (args.archive is None or not args.archive.is_file()):
        parser.error("The prepare stage requires --archive and the original release ZIP.")
    source = (args.calux_sources or (args.workdir / "sources")).resolve()
    if "ring_trial" in args.stages:
        expected = ["AR_CALUX_JRC_main.pdf", "AR_CALUX_JRC_statistical.pdf",
                    "AR_CALUX_2021.xml", "AR_CALUX_2021_supplement.zip"]
        missing = [name for name in expected if not (source/name).is_file()]
        if missing:
            parser.error("Missing public sources in --calux-sources: "+", ".join(missing))
    scripts = Path(__file__).resolve().parent/"scripts"
    env = os.environ.copy()
    env.update(ENDOCRINE_WORKDIR=str(args.workdir.resolve()), CALUX_INPUTS=str(source),
               PYTHONUTF8="1", PYTHONDONTWRITEBYTECODE="1", OPENBLAS_NUM_THREADS="1",
               MKL_NUM_THREADS="1", OMP_NUM_THREADS="1")
    for stage in STAGES:
        if stage not in args.stages:
            continue
        for script in STAGES[stage]:
            command = [sys.executable, "-B", str(scripts/(script+".py"))]
            if stage == "prepare":
                command.append(str(args.archive.resolve()))
            print(f"Running {script}",flush=True)
            subprocess.run(command,env=env,check=True)


if __name__ == "__main__":
    main()
