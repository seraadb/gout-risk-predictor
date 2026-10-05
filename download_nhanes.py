"""
NHANES Auto-Downloader
=============================================================================
Automatically downloads and organizes all NHANES .XPT files needed for the
gout risk prediction pipeline, across all 7 survey cycles (2007-2023).

Saves each file directly into the correct folder structure that
gout_prediction_pipeline.py expects:
    nhanes_raw/<cycle>/<COMPONENT>.XPT

USAGE:
    python download_nhanes.py

If a file already exists locally, it is skipped (safe to re-run / resume).
If a particular file genuinely doesn't exist for a given cycle (some
components were added/renamed over time), a warning is printed and the
script continues with everything else.
=============================================================================
"""

import time
from pathlib import Path
import urllib.request
import urllib.error

# -----------------------------------------------------------------------
# CONFIG: cycle folder name -> (start_year_used_in_url, suffix, prefix)
# CDC uses a SUFFIX for most cycles (e.g. DEMO_E.XPT) but a PREFIX for the
# disrupted 2017-March 2020 cycle (e.g. P_DEMO.XPT).
# -----------------------------------------------------------------------
CYCLES = {
    "2007-2008": {"start_year": 2007, "suffix": "E", "prefix": None},
    "2009-2010": {"start_year": 2009, "suffix": "F", "prefix": None},
    "2011-2012": {"start_year": 2011, "suffix": "G", "prefix": None},
    "2013-2014": {"start_year": 2013, "suffix": "H", "prefix": None},
    "2015-2016": {"start_year": 2015, "suffix": "I", "prefix": None},
    # NOTE: using the STANDALONE 2017-2018 cycle (suffix J), NOT the combined
    # "2017-March 2020" pre-pandemic file. NHANES DROPPED the gout question
    # (MCQ160n) when it merged 2017-2018 with the incomplete 2019-2020 data,
    # so only the standalone 2017-2018 release still has self-reported gout.
    # 2019-2020 alone was never released as a standalone public file and is
    # excluded here (NHANES itself advises against using it in isolation).
    "2017-2018": {"start_year": 2017, "suffix": "J", "prefix": None},
    "2021-2023": {"start_year": 2021, "suffix": "L", "prefix": None},
}

# NHANES stopped asking about gout after the 2017-2018 cycle. For cycles with
# no self-reported gout item, gout_prediction_pipeline.py instead derives a
# medication-based proxy (allopurinol, febuxostat, colchicine, etc.) from the
# RXQ_RX prescription medications file, which requires this component too.
CYCLES_NEEDING_MEDICATION_PROXY = {"2021-2023"}

# 2017-2018's Cholesterol/Glucose/Triglyceride files also carry a suffix J
# (HDL_J, TRIGLY_J, GLU_J) just like every other standalone cycle, so no
# special-casing is needed there -- only the outcome variable differs.

# Component files needed, matching gout_prediction_pipeline.py's
# load_nhanes_cycle(). Keys are the base component code used by CDC.
COMPONENTS = [
    "DEMO",     # Demographics
    "MCQ",      # Medical Conditions (gout outcome, where available)
    "BMX",      # Body Measures
    "BIOPRO",   # Standard Biochemistry Profile (uric acid, creatinine)
    "HDL",      # HDL Cholesterol
    "GLU",      # Fasting Glucose
    "TRIGLY",   # Triglycerides & LDL Cholesterol
    "DIQ",      # Diabetes questionnaire
    "BPQ",      # Blood Pressure & Cholesterol questionnaire
    "ALQ",      # Alcohol Use
    "PAQ",      # Physical Activity
    "DR1TOT",   # Dietary Total Nutrient Intake, Day 1
    "RXQ_RX",   # Prescription Medications (for gout-medication proxy)
]

OUTPUT_ROOT = Path("./nhanes_raw")
BASE_URL = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public"
REQUEST_DELAY_SECONDS = 1.0   # be polite to CDC's servers between requests


def build_url(component: str, cycle_info: dict) -> str:
    year = cycle_info["start_year"]
    if cycle_info["prefix"]:
        filename = f"{cycle_info['prefix']}{component}.XPT"
    else:
        filename = f"{component}_{cycle_info['suffix']}.XPT"
    return f"{BASE_URL}/{year}/DataFiles/{filename}"


def download_file(url: str, dest: Path) -> bool:
    """Download a single file. Returns True on success, False otherwise."""
    if dest.exists() and dest.stat().st_size > 0:
        print(f"    [skip] {dest.name} already exists")
        return True

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=60) as response:
            data = response.read()
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        print(f"    [ok]   {dest.name}  ({len(data)/1024:.1f} KB)")
        return True
    except urllib.error.HTTPError as e:
        print(f"    [MISSING] {dest.name}  (HTTP {e.code}) -- {url}")
        return False
    except Exception as e:
        print(f"    [ERROR] {dest.name}  -- {e}")
        return False


def main():
    print("=" * 70)
    print("NHANES AUTO-DOWNLOADER")
    print("=" * 70)

    total_files = 0
    total_ok = 0
    missing_summary = []

    for cycle_name, cycle_info in CYCLES.items():
        print(f"\nCycle {cycle_name}:")
        cycle_dir = OUTPUT_ROOT / cycle_name

        for component in COMPONENTS:
            url = build_url(component, cycle_info)
            dest = cycle_dir / f"{component}.XPT"
            total_files += 1
            ok = download_file(url, dest)
            if ok:
                total_ok += 1
            else:
                missing_summary.append((cycle_name, component, url))
            time.sleep(REQUEST_DELAY_SECONDS)

    print("\n" + "=" * 70)
    print(f"DONE: {total_ok}/{total_files} files downloaded successfully.")
    print("=" * 70)

    if missing_summary:
        print("\nThe following files could NOT be found automatically.")
        print("This can happen when a component was renamed or didn't exist")
        print("in that particular cycle -- check the NHANES site manually")
        print("for these specific ones:\n")
        for cycle_name, component, url in missing_summary:
            print(f"  - {cycle_name} / {component}\n      tried: {url}")
    else:
        print("\nAll files downloaded! You're ready to run gout_prediction_pipeline.py")


if __name__ == "__main__":
    main()
