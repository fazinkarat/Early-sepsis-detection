#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data Cleaning Pipeline — Core Cohort Builder (robust to column name variants)

Builds a clean ICU-stay cohort from:
- PATIENTS.csv
- ADMISSIONS.csv
- ICUSTAYS.csv

Computes:
- AGE_ADMIT (years)
- ICU_LOS_HOURS
- HOSPITAL_LOS_DAYS
- IN_HOSPITAL_MORTALITY (0/1)
- MULTI_ADMIT_FLAG (readmission within dataset)

Usage examples (Windows PowerShell):
  python scripts\data_cleaning.py --data-dir "C:\AML\Early-sepsis-detection\data"
"""

import os
import json
import argparse
from datetime import datetime
from typing import Dict, Optional, List

import pandas as pd
import numpy as np

REQUIRED_KEYS = ["patients_csv","admissions_csv","icustays_csv"]
ALL_KEYS = REQUIRED_KEYS + ["diagnoses_icd_csv","d_icd_diagnoses_csv","chartevents_csv","labevents_csv"]

# -------- helpers --------
def log(cfg: Dict[str,str], msg: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"{ts} | {msg}"
    print(line)
    try:
        with open(cfg["log_file"], "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass

def pick_data_dir(user_arg: Optional[str]) -> str:
    cands: List[str] = []
    if user_arg: cands.append(user_arg)
    env = os.environ.get("MIMIC_DATA_DIR")
    if env: cands.append(env)
    here = os.path.dirname(os.path.abspath(__file__))
    cands.append(os.path.join(here, "data"))
    cands.append(os.getcwd())
    cands.append(r"C:\AML\Early-sepsis-detection\data")
    cands.append("/mnt/data")
    seen, uniq = set(), []
    for c in cands:
        c = os.path.normpath(c)
        if c not in seen:
            uniq.append(c); seen.add(c)
    for d in uniq:
        if os.path.isdir(d): return d
    return uniq[-1]

def compute_age_years_safe(df: pd.DataFrame) -> pd.Series:
    """Age in whole years at admission without using Timedelta (avoids int64 overflow)."""
    admit = df["ADMITTIME"]
    dob   = df["DOB"]
    age = (admit.dt.year - dob.dt.year).astype("float64")  # start with year diff
    # subtract 1 if birthday hasn't occurred yet this year
    mday_admit = admit.dt.month * 100 + admit.dt.day
    mday_dob   = dob.dt.month   * 100 + dob.dt.day
    not_had_bday = (mday_admit < mday_dob).astype("int8")
    # handle NaT safely
    mask = admit.notna() & dob.notna()
    out = pd.Series(np.nan, index=df.index, dtype="float64")
    out[mask] = (age[mask] - not_had_bday[mask]).astype("float64")
    return out


def build_config(data_dir: str) -> Dict[str,str]:
    out_dir = os.path.join(data_dir, "clean")
    log_dir = os.path.join(out_dir, "logs")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    return {
        "patients_csv": os.path.join(data_dir, "PATIENTS.csv"),
        "admissions_csv": os.path.join(data_dir, "ADMISSIONS.csv"),
        "icustays_csv": os.path.join(data_dir, "ICUSTAYS.csv"),
        "diagnoses_icd_csv": os.path.join(data_dir, "DIAGNOSES_ICD.csv"),
        "d_icd_diagnoses_csv": os.path.join(data_dir, "D_ICD_DIAGNOSES.csv"),
        "chartevents_csv": os.path.join(data_dir, "CHARTEVENTS.csv"),
        "labevents_csv": os.path.join(data_dir, "LABEVENTS.csv"),
        "out_parquet": os.path.join(out_dir, "clean_cohort.parquet"),
        "out_preview_csv": os.path.join(out_dir, "clean_cohort_preview.csv"),
        "log_file": os.path.join(log_dir, "cleaning_log.txt"),
    }

def check_files(cfg: Dict[str,str]) -> bool:
    print("\n=== File existence check ===")
    ok = True
    for k in ALL_KEYS:
        p = cfg[k]
        exists = os.path.exists(p)
        tag = "OK " if exists else "MISS"
        req = "required" if k in REQUIRED_KEYS else "optional"
        print(f"[{tag}] {req:8s} -> {p}")
        if not exists and req == "required":
            ok = False
    print("============================\n")
    return ok

def read_csv_loose(path: str) -> pd.DataFrame:
    """Read without parse_dates; standardize headers to UPPERCASE, then coerce dates later."""
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path, low_memory=False)
    df.columns = [str(c).strip().upper() for c in df.columns]  # <= normalize headers
    return df


def coerce_datetime(df: pd.DataFrame, cols: List[str]) -> None:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")

def require_columns(df: pd.DataFrame, needed: List[str], table_name: str):
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(
            f"{table_name}: missing required columns {missing}. "
            f"Found columns: {list(df.columns)}"
        )

# -------- flexible loaders (handle column variants) --------
def load_patients(path: str) -> pd.DataFrame:
    df = read_csv_loose(path)
    if df.empty:
        return df

    # Rename common variants to canonical names
    rename_map = {
        "DATE_OF_BIRTH": "DOB",
        "DATEOFBIRTH": "DOB",
        "BIRTHDATE": "DOB",
        "DATE_OF_DEATH": "DOD",
        "DATEOFDEATH": "DOD",
        "DEATHDATE": "DOD",
        "SEX": "GENDER",
    }
    present_map = {k:v for k,v in rename_map.items() if k in df.columns}
    if present_map:
        df = df.rename(columns=present_map)

    require_columns(df, ["SUBJECT_ID"], "PATIENTS")
    # DOB is required for age; if missing, fail with a clear message
    if "DOB" not in df.columns:
        raise ValueError(
            "PATIENTS: cannot find a DOB column. "
            "Tried variants: DOB / DATE_OF_BIRTH / DATEOFBIRTH / BIRTHDATE."
        )

    coerce_datetime(df, ["DOB", "DOD"])
    if "GENDER" in df.columns:
        df["GENDER"] = df["GENDER"].astype("category")
    return df

def load_admissions(path: str) -> pd.DataFrame:
    df = read_csv_loose(path)
    if df.empty:
        return df
    rename_map = {
        "ADMIT_TIME": "ADMITTIME",
        "ADMISSION_TIME": "ADMITTIME",
        "DISCH_TIME": "DISCHTIME",
        "DEATH_TIME": "DEATHTIME",
    }
    present_map = {k:v for k,v in rename_map.items() if k in df.columns}
    if present_map:
        df = df.rename(columns=present_map)

    require_columns(df, ["SUBJECT_ID","HADM_ID","ADMITTIME","DISCHTIME"], "ADMISSIONS")
    coerce_datetime(df, ["ADMITTIME","DISCHTIME","DEATHTIME"])
    return df

def load_icustays(path: str) -> pd.DataFrame:
    df = read_csv_loose(path)
    if df.empty:
        return df
    rename_map = {
        "IN_TIME": "INTIME",
        "OUT_TIME": "OUTTIME",
    }
    present_map = {k:v for k,v in rename_map.items() if k in df.columns}
    if present_map:
        df = df.rename(columns=present_map)

    require_columns(df, ["SUBJECT_ID","HADM_ID","ICUSTAY_ID","INTIME","OUTTIME"], "ICUSTAYS")
    coerce_datetime(df, ["INTIME","OUTTIME"])
    return df

# -------- core pipeline --------
def build_cohort(cfg: Dict[str,str]) -> Optional[pd.DataFrame]:
    patients  = load_patients(cfg["patients_csv"])
    admissions = load_admissions(cfg["admissions_csv"])
    icu       = load_icustays(cfg["icustays_csv"])

    if patients.empty or admissions.empty or icu.empty:
        return None

    # First ICU stay per admission
    icu_sorted = icu.sort_values(["HADM_ID","INTIME"]).drop_duplicates("HADM_ID", keep="first")
    cohort = admissions.merge(icu_sorted, on=["SUBJECT_ID","HADM_ID"], how="inner", validate="one_to_one")

    # Merge demographics (DOD optional)
    demo_cols = ["SUBJECT_ID","DOB","GENDER"] + (["DOD"] if "DOD" in patients.columns else [])
    cohort = cohort.merge(patients[demo_cols], on="SUBJECT_ID", how="left", validate="many_to_one")

    # --- Robust age (no Timedelta math) ---
    cohort["AGE_ADMIT"] = compute_age_years_safe(cohort)

    # --- LOS with regular Timedelta math (compatible with older pandas) ---
    icu_td  = (cohort["OUTTIME"]   - cohort["INTIME"])
    hosp_td = (cohort["DISCHTIME"] - cohort["ADMITTIME"])

    cohort["ICU_LOS_HOURS"]     = icu_td  / pd.Timedelta(hours=1)
    cohort["HOSPITAL_LOS_DAYS"] = hosp_td / pd.Timedelta(days=1)

    # sanity: drop impossible LOS values (negative or absurdly large)
    cohort.loc[(cohort["ICU_LOS_HOURS"] < 0) | (cohort["ICU_LOS_HOURS"] > 24*60), "ICU_LOS_HOURS"] = np.nan   # >60 days
    cohort.loc[(cohort["HOSPITAL_LOS_DAYS"] < 0) | (cohort["HOSPITAL_LOS_DAYS"] > 365*2), "HOSPITAL_LOS_DAYS"] = np.nan  # >2 years


    # In-hospital mortality (guard missing DEATHTIME)
    if "DEATHTIME" in cohort.columns:
        died_in_hosp = (
            cohort["DEATHTIME"].notna()
            & (cohort["DEATHTIME"] >= cohort["ADMITTIME"])
            & (cohort["DEATHTIME"] <= cohort["DISCHTIME"])
        )
        cohort["IN_HOSPITAL_MORTALITY"] = died_in_hosp.astype(int)
    else:
        cohort["IN_HOSPITAL_MORTALITY"] = 0

    # Readmission flag (coarse)
    cohort["MULTI_ADMIT_FLAG"] = cohort.groupby("SUBJECT_ID")["HADM_ID"].transform("count").gt(1).astype(int)

    # Drop rows with missing essentials (after robust computations)
    na_critical = ["ADMITTIME","DISCHTIME","INTIME","OUTTIME","AGE_ADMIT"]
    before = len(cohort)
    cohort = cohort.dropna(subset=na_critical)
    after = len(cohort)
    log(cfg, f"Dropped {before - after} rows with missing critical times/age.")

    # Order columns
    preferred = [
        "SUBJECT_ID","HADM_ID","ICUSTAY_ID","GENDER","AGE_ADMIT",
        "ADMITTIME","DISCHTIME","DEATHTIME","INTIME","OUTTIME",
        "ICU_LOS_HOURS","HOSPITAL_LOS_DAYS","IN_HOSPITAL_MORTALITY","MULTI_ADMIT_FLAG"
    ]
    rest = [c for c in cohort.columns if c not in preferred]
    cohort = cohort[preferred + rest].drop_duplicates(["SUBJECT_ID","HADM_ID"]).reset_index(drop=True)
    return cohort


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=str, default=None, help="Folder with PATIENTS.csv, ADMISSIONS.csv, ICUSTAYS.csv")
    args = ap.parse_args()

    data_dir = pick_data_dir(args.data_dir)
    cfg = build_config(data_dir)

    log(cfg, "=== Data Cleaning Run Started ===")
    log(cfg, json.dumps(cfg, indent=2))

    if not check_files(cfg):
        log(cfg, "[FATAL] Required CSVs missing. Use --data-dir to point to your folder.")
        return

    try:
        cohort = build_cohort(cfg)
        if cohort is None or cohort.empty:
            log(cfg, "[FATAL] Cohort is empty (failed to load/join core tables).")
            return
    except Exception as e:
        log(cfg, f"[FATAL] {e}")
        return

    # Order + save (Parquet with CSV fallback)
    preferred = [
        "SUBJECT_ID","HADM_ID","ICUSTAY_ID","GENDER","AGE_ADMIT",
        "ADMITTIME","DISCHTIME","DEATHTIME","INTIME","OUTTIME",
        "ICU_LOS_HOURS","HOSPITAL_LOS_DAYS","IN_HOSPITAL_MORTALITY","MULTI_ADMIT_FLAG"
    ]
    rest = [c for c in cohort.columns if c not in preferred]
    cohort = cohort[preferred + rest].drop_duplicates(["SUBJECT_ID","HADM_ID"]).reset_index(drop=True)

    try:
        cohort.to_parquet(cfg["out_parquet"], index=False)
        log(cfg, f"Saved cohort to {cfg['out_parquet']}")
    except Exception as e:
        csv_fallback = os.path.join(os.path.dirname(cfg["out_parquet"]), "clean_cohort.csv")
        cohort.to_csv(csv_fallback, index=False)
        log(cfg, f"[WARN] Parquet failed ({e}); saved CSV to {csv_fallback}")

    cohort.sample(min(1000, len(cohort))).to_csv(cfg["out_preview_csv"], index=False)
    log(cfg, f"Saved preview to {cfg['out_preview_csv']}")
    log(cfg, "=== Data Cleaning Completed ===")

if __name__ == "__main__":
    main()
