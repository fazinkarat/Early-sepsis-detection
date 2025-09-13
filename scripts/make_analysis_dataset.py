#!/usr/bin/env python3
"""
make_analysis_dataset.py (robust, fallback-friendly)
- Normalizes headers to UPPERCASE for all CSVs.
- Parses date columns only if present.
- Computes age safely (no nanosecond subtraction overflow).
- Sepsis label from ICD-9 with guaranteed 'SEPSIS' column.
- Vitals mapping: dictionary first, then unit/range heuristics fallback.
- Chunks only CHARTEVENTS / LABEVENTS to handle large files.
"""
import os, re, gc, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Optional: silence mixed dtype warnings from huge CSVs
warnings.filterwarnings("ignore", category=pd.errors.DtypeWarning)

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
PLOT_DIR = os.path.join(os.path.dirname(__file__), "..", "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

# ---------- IO helpers ----------
def read_csv_upper(path, want_cols=None, want_dates=None, chunksize=None):
    """
    For large tables (when chunksize is provided), yields normalized chunks.
    For small tables (no chunksize), returns a DataFrame with normalized headers.
    """
    if chunksize is not None:
        it = pd.read_csv(path, low_memory=True, chunksize=chunksize)
        for chunk in it:
            chunk.columns = chunk.columns.str.upper()
            if want_cols is not None:
                use = [c.upper() for c in want_cols if c.upper() in chunk.columns]
                chunk = chunk[use]
            if want_dates is not None:
                for c in want_dates:
                    cu = c.upper()
                    if cu in chunk.columns:
                        chunk[cu] = pd.to_datetime(chunk[cu], errors='coerce')
            yield chunk
        return

    df = pd.read_csv(path, low_memory=False)
    df.columns = df.columns.str.upper()
    if want_cols is not None:
        use = [c.upper() for c in want_cols if c.upper() in df.columns]
        df = df[use]
    if want_dates is not None:
        for c in want_dates:
            cu = c.upper()
            if cu in df.columns:
                df[cu] = pd.to_datetime(df[cu], errors='coerce')
    return df

def read_core_df(path, want_cols=None, want_dates=None):
    """
    Force DataFrame (no chunking) and normalize headers.
    Use for small tables: PATIENTS, ADMISSIONS, ICUSTAYS, DIAGNOSES_ICD, dictionaries.
    """
    df = pd.read_csv(path, low_memory=False)
    df.columns = df.columns.str.upper()
    if want_dates:
        for c in want_dates:
            cu = c.upper()
            if cu in df.columns:
                df[cu] = pd.to_datetime(df[cu], errors="coerce")
    if want_cols:
        use = [c.upper() for c in want_cols if c.upper() in df.columns]
        df = df[use]
    return df

# ---------- small utils ----------
def normalize_icd(code):
    if pd.isna(code): return None
    return re.sub(r"\D", "", str(code))

def map_feature(label, mapping):
    if pd.isna(label): return None
    for feat, pat in mapping.items():
        if re.search(pat, str(label), flags=re.IGNORECASE):
            return feat
    return None

def main():
    # --- Load core tables (force DF, normalized headers) ---
    patients  = read_core_df(os.path.join(DATA_DIR, "PATIENTS.csv"),
                             want_cols=["SUBJECT_ID","GENDER","DOB","DOD"],
                             want_dates=["DOB","DOD"])
    admissions = read_core_df(os.path.join(DATA_DIR, "ADMISSIONS.csv"),
                              want_cols=["SUBJECT_ID","HADM_ID","ADMITTIME","DISCHTIME","DEATHTIME"],
                              want_dates=["ADMITTIME","DISCHTIME","DEATHTIME"])
    icustays  = read_core_df(os.path.join(DATA_DIR, "ICUSTAYS.csv"),
                             want_cols=["ROW_ID","SUBJECT_ID","HADM_ID","ICUSTAY_ID","INTIME","OUTTIME","LOS"],
                             want_dates=["INTIME","OUTTIME"])
    diag      = read_core_df(os.path.join(DATA_DIR, "DIAGNOSES_ICD.csv"),
                             want_cols=["SUBJECT_ID","HADM_ID","ICD9_CODE"])

    print("PATIENTS columns:", list(patients.columns))
    print("ADMISSIONS columns:", list(admissions.columns))
    print("ICUSTAYS columns:", list(icustays.columns))

    # --- Build merged ICU dataframe ---
    icu = icustays.merge(
        admissions[["SUBJECT_ID","HADM_ID","ADMITTIME"]],
        on=["SUBJECT_ID","HADM_ID"], how="left"
    ).merge(
        patients[["SUBJECT_ID","GENDER","DOB"]],
        on="SUBJECT_ID", how="left"
    )

    # --- SAFE age computation (avoid ns overflow) ---
    icu["AGE_YEARS"] = np.nan
    mask = icu["ADMITTIME"].notna() & icu["DOB"].notna()

    # calendar-year difference
    age_years = icu.loc[mask, "ADMITTIME"].dt.year - icu.loc[mask, "DOB"].dt.year
    # adjust if birthday hasn’t occurred yet this year
    adm_md = icu.loc[mask, "ADMITTIME"].dt.strftime("%m%d").astype(int)
    dob_md = icu.loc[mask, "DOB"].dt.strftime("%m%d").astype(int)
    age_years = age_years - (adm_md < dob_md).astype(int)

    icu.loc[mask, "AGE_YEARS"] = age_years
    icu["AGE_YEARS"] = icu["AGE_YEARS"].clip(lower=0, upper=120)

    # --- Base cohort for later joins ---
    icustays_basic = icu[[
        "SUBJECT_ID","HADM_ID","ICUSTAY_ID","INTIME","OUTTIME","LOS","GENDER","AGE_YEARS"
    ]].copy()

    # --- Sepsis label from ICD-9 (guaranteed column) ---
    sepsis_codes = {"99591", "99592", "78552"}
    diag["ICD9_NORM"] = diag["ICD9_CODE"].apply(normalize_icd)
    diag["SEPSIS"] = diag["ICD9_NORM"].isin(sepsis_codes).astype(int)
    hadm_sepsis = diag.groupby(["SUBJECT_ID", "HADM_ID"], as_index=False)["SEPSIS"].max()

    icustays_lbl = icustays_basic.merge(hadm_sepsis, on=["SUBJECT_ID","HADM_ID"], how="left")
    icustays_lbl["SEPSIS"] = icustays_lbl["SEPSIS"].fillna(0).astype(int)

    # --- Dictionaries (optional) ---
    d_labitems_path = os.path.join(DATA_DIR, "D_LABITEMS.csv")
    have_d_lab = os.path.exists(d_labitems_path)
    if have_d_lab:
        d_labitems = read_core_df(d_labitems_path, want_cols=["ITEMID","LABEL","FLUID","CATEGORY"])
    else:
        d_labitems = pd.DataFrame(columns=["ITEMID","LABEL","FLUID","CATEGORY"])

    d_items_path = os.path.join(DATA_DIR, "D_ITEMS.csv")
    have_d_items = os.path.exists(d_items_path)
    if have_d_items:
        d_items = read_core_df(d_items_path, want_cols=["ITEMID","LABEL","CATEGORY"])
    else:
        d_items = pd.DataFrame(columns=["ITEMID","LABEL","CATEGORY"])

    vital_name_patterns = {
        "HEART_RATE": r"(?i)\bheart\s*rate\b|HR\b",
        "RESP_RATE": r"(?i)\bresp(iratory)?\s*rate\b|\bRR\b",
        "TEMP_C": r"(?i)\btemp(erature)?\b",
        "SYSTOLIC_BP": r"(?i)\b(systolic|sbp)\b",
        "MEAN_BP": r"(?i)\bmean\s*(arterial)?\s*pressure\b|\bMAP\b",
        "SPO2": r"(?i)\b(SpO2|O2\s*saturation|oxygen\s*saturation)\b"
    }
    lab_name_patterns = {
        "LACTATE": r"(?i)\blactate\b",
        "WBC": r"(?i)\bwhite blood cell|WBC\b",
        "PLATELETS": r"(?i)\bplatelet",
        "CREATININE": r"(?i)\bcreatinine\b"
    }

    if have_d_items and len(d_items):
        vital_itemids = {name: set(d_items.loc[d_items["LABEL"].fillna("").str.contains(pat, regex=True), "ITEMID"].tolist())
                         for name, pat in vital_name_patterns.items()}
    else:
        vital_itemids = {k: set() for k in vital_name_patterns}

    if have_d_lab and len(d_labitems):
        lab_itemids = {name: set(d_labitems.loc[d_labitems["LABEL"].fillna("").str.contains(pat, regex=True), "ITEMID"].tolist())
                       for name, pat in lab_name_patterns.items()}
    else:
        lab_itemids = {k: set() for k in lab_name_patterns}

    # --- First-24h ICU window ---
    icustay_windows = icustays_lbl[[
        "SUBJECT_ID","HADM_ID","ICUSTAY_ID","INTIME","OUTTIME","SEPSIS","GENDER","AGE_YEARS"
    ]].copy()
    icustay_windows = icustay_windows[icustay_windows["INTIME"].notna()]
    icustay_windows["END_24H"] = icustay_windows["INTIME"] + pd.to_timedelta(24, unit="h")
    valid_icustays = set(icustay_windows["ICUSTAY_ID"].astype(int).tolist()) if "ICUSTAY_ID" in icustay_windows.columns else set()

    # --- CHARTEVENTS (vitals) ---
    vitals_path = os.path.join(DATA_DIR, "CHARTEVENTS.csv")
    vital_events = []
    if os.path.exists(vitals_path) and len(valid_icustays):
        cols = ["SUBJECT_ID","HADM_ID","ICUSTAY_ID","ITEMID","CHARTTIME","VALUENUM","VALUEUOM"]
        for chunk in read_csv_upper(vitals_path, want_cols=cols, want_dates=["CHARTTIME"], chunksize=1_000_000):
            if "ICUSTAY_ID" in chunk.columns:
                chunk = chunk[chunk["ICUSTAY_ID"].isin(valid_icustays)]
            # join time window
            chunk = chunk.merge(icustay_windows[["ICUSTAY_ID","INTIME","END_24H"]], on="ICUSTAY_ID", how="left")
            if "CHARTTIME" in chunk.columns:
                chunk = chunk[(chunk["CHARTTIME"] >= chunk["INTIME"]) & (chunk["CHARTTIME"] < chunk["END_24H"])]
            if have_d_items and any(len(s)>0 for s in vital_itemids.values()):
                allowed = set().union(*vital_itemids.values())
                chunk = chunk[chunk["ITEMID"].isin(allowed)]
            else:
                # heuristic filter by units/value presence
                chunk = chunk[(chunk["VALUENUM"].notna()) &
                              (chunk["VALUEUOM"].fillna("").str.contains("bpm|C|F|mmHg|%", case=False))]
            # keep VALUEUOM for heuristics
            vital_events.append(chunk[["ICUSTAY_ID","ITEMID","VALUENUM","VALUEUOM"]])

    vitals_24h = pd.concat(vital_events, ignore_index=True) if vital_events else pd.DataFrame(columns=["ICUSTAY_ID","ITEMID","VALUENUM","VALUEUOM"])
    del vital_events; gc.collect()

    # --- Map vitals to features (dictionary first, then heuristics) ---
    if len(vitals_24h):
        # dictionary-based mapping (if we have D_ITEMS)
        if have_d_items and "ITEMID" in vitals_24h.columns and len(d_items):
            vitals_24h = vitals_24h.merge(d_items[["ITEMID","LABEL"]], on="ITEMID", how="left")
            vitals_24h["FEAT"] = vitals_24h["LABEL"].apply(lambda x: map_feature(x, vital_name_patterns))

        # ensure FEAT is object-typed so we can store strings safely
        if "FEAT" not in vitals_24h.columns:
            vitals_24h["FEAT"] = pd.Series(index=vitals_24h.index, dtype=object)
        else:
            vitals_24h["FEAT"] = vitals_24h["FEAT"].astype(object)

        # heuristic fallback if FEAT missing or all NaN
        if ("FEAT" not in vitals_24h.columns) or (vitals_24h["FEAT"].isna().all()):
            if "VALUEUOM" not in vitals_24h.columns:
                vitals_24h["VALUEUOM"] = ""
            uom = vitals_24h["VALUEUOM"].fillna("").str.lower()
            vitals_24h["FEAT"] = np.nan

            # temperature (C/F) -> convert F to C if value looks Fahrenheit
            temp_mask = (uom.str.contains("c") | uom.str.contains("f")) & vitals_24h["VALUENUM"].notna()
            if temp_mask.any():
                vals = pd.to_numeric(vitals_24h.loc[temp_mask, "VALUENUM"], errors="coerce")
                is_f = uom[temp_mask].str.contains("f")
                vals_c = vals.copy()
                vals_c.loc[is_f & vals.notna()] = (vals.loc[is_f & vals.notna()] - 32) * 5.0/9.0
                vitals_24h.loc[temp_mask, "VALUENUM"] = vals_c
                vitals_24h.loc[temp_mask, "FEAT"] = "TEMP_C"

            # SpO2: %
            spo2_mask = uom.str.contains("%")
            vitals_24h.loc[spo2_mask, "FEAT"] = vitals_24h.loc[spo2_mask, "FEAT"].fillna("SPO2")

            # mmHg → treat as MEAN_BP (can't reliably split SBP/MAP without labels)
            bp_mask = uom.str.contains("mmhg")
            vitals_24h.loc[bp_mask, "FEAT"] = vitals_24h.loc[bp_mask, "FEAT"].fillna("MEAN_BP")

            # bpm → RR if ≤40 else HR (simple heuristic)
            bpm_mask = uom.str.contains("bpm") & vitals_24h["VALUENUM"].notna()
            if bpm_mask.any():
                vals_bpm = pd.to_numeric(vitals_24h.loc[bpm_mask, "VALUENUM"], errors="coerce")
                rr_rows = bpm_mask.copy()
                hr_rows = bpm_mask.copy()
                rr_rows.loc[rr_rows] = vals_bpm.le(40).fillna(False).values
                hr_rows.loc[hr_rows] = vals_bpm.gt(40).fillna(False).values
                vitals_24h.loc[rr_rows, "FEAT"] = vitals_24h.loc[rr_rows, "FEAT"].fillna("RESP_RATE")
                vitals_24h.loc[hr_rows, "FEAT"] = vitals_24h.loc[hr_rows, "FEAT"].fillna("HEART_RATE")

            # drop rows we still cannot map
            vitals_24h = vitals_24h[vitals_24h["FEAT"].notna()]

    # --- Aggregate vitals ---
    if len(vitals_24h) and "FEAT" in vitals_24h.columns:
        vitals_24h["VALUENUM"] = pd.to_numeric(vitals_24h["VALUENUM"], errors="coerce")
        agg_vitals = vitals_24h.groupby(["ICUSTAY_ID","FEAT"])["VALUENUM"].mean().unstack("FEAT")
    else:
        agg_vitals = pd.DataFrame()

    # --- LABEVENTS (labs) ---
    labs_path = os.path.join(DATA_DIR, "LABEVENTS.csv")
    lab_events = []
    if os.path.exists(labs_path):
        cols = ["SUBJECT_ID","HADM_ID","ITEMID","CHARTTIME","VALUENUM"]
        for chunk in read_csv_upper(labs_path, want_cols=cols, want_dates=["CHARTTIME"], chunksize=1_000_000):
            lab_events.append(chunk)

    labevents_all = pd.concat(lab_events, ignore_index=True) if lab_events else pd.DataFrame(columns=cols if os.path.exists(labs_path) else [])
    del lab_events; gc.collect()

    if len(labevents_all):
        labevents_all = labevents_all.merge(icustay_windows[["SUBJECT_ID","HADM_ID","ICUSTAY_ID","INTIME","END_24H"]],
                                            on=["SUBJECT_ID","HADM_ID"], how="inner")
        labevents_24h = labevents_all[(labevents_all["CHARTTIME"] >= labevents_all["INTIME"]) &
                                      (labevents_all["CHARTTIME"] < labevents_all["END_24H"])].copy()
        if have_d_lab and len(d_labitems):
            labevents_24h = labevents_24h.merge(d_labitems[["ITEMID","LABEL"]], on="ITEMID", how="left")
            labevents_24h["FEAT"] = labevents_24h["LABEL"].apply(lambda x: map_feature(x, {
                "LACTATE": r"(?i)lactate",
                "WBC": r"(?i)white blood cell|WBC",
                "PLATELETS": r"(?i)platelet",
                "CREATININE": r"(?i)creatinine"
            }))
            labevents_24h = labevents_24h[labevents_24h["FEAT"].notna()]
        if len(labevents_24h):
            labevents_24h["VALUENUM"] = pd.to_numeric(labevents_24h["VALUENUM"], errors="coerce")
            agg_labs = labevents_24h.groupby(["ICUSTAY_ID","FEAT"])["VALUENUM"].median().unstack("FEAT")
        else:
            agg_labs = pd.DataFrame()
    else:
        agg_labs = pd.DataFrame()

    # --- Join features ---
    features = icustays_lbl.set_index("ICUSTAY_ID")[["SEPSIS","GENDER","AGE_YEARS"]].copy()
    if len(agg_vitals):
        features = features.join(agg_vitals, how="left")
    if len(agg_labs):
        features = features.join(agg_labs, how="left")

    # reset first, THEN lowercase so index name also gets lowercased
    features.reset_index(inplace=True)              # brings ICUSTAY_ID out as a column
    features.columns = [c.lower() for c in features.columns]  # now we have 'icustay_id'


    # --- Clean & transform ---
    keep_cols = ["icustay_id","sepsis","gender","age_years"]
    numeric_cols = features.columns.difference(keep_cols)
    non_missing = features[numeric_cols].notna().mean() if len(numeric_cols) else pd.Series([], dtype=float)
    good_numeric = non_missing[non_missing >= 0.6].index.tolist()

    clean = features[keep_cols + good_numeric].copy() if good_numeric else features[keep_cols].copy()
    if "gender" in clean.columns:
        clean["gender"] = (clean["gender"].astype(str).str.upper() == "M").astype(int)

    for col in good_numeric:
        clean[col] = pd.to_numeric(clean[col], errors="coerce")
        clean[col] = clean[col].fillna(clean[col].median())

    for col in ["lactate","wbc","platelets","creatinine"]:
        if col in clean.columns:
            clean[f"log1p_{col}"] = np.log1p(clean[col])

    for col in good_numeric:
        mu, sigma = clean[col].mean(), clean[col].std(ddof=0)
        if sigma and sigma > 0:
            clean[f"z_{col}"] = (clean[col] - mu) / sigma

    out_preview = os.path.join(DATA_DIR, "analysis_table_preview.csv")
    clean.head(20).to_csv(out_preview, index=False)
    print("Saved preview to", out_preview)

    # --- Plots ---
    # Class balance
    if "sepsis" in clean.columns:
        plt.figure()
        clean["sepsis"].value_counts().sort_index().plot(kind="bar")
        plt.xticks([0,1], ["Non-sepsis","Sepsis"], rotation=0)
        plt.title("Class balance (ICU stays)")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(os.path.join(PLOT_DIR, "class_balance.png"))

    # Age histogram
    if "age_years" in clean.columns:
        plt.figure()
        if "sepsis" in clean.columns:
            clean[clean["sepsis"]==0]["age_years"].plot(kind="hist", alpha=0.6, bins=30)
            clean[clean["sepsis"]==1]["age_years"].plot(kind="hist", alpha=0.6, bins=30)
            plt.legend(["Non-sepsis","Sepsis"])
        else:
            clean["age_years"].plot(kind="hist", bins=30)
        plt.xlabel("Age (years)")
        plt.title("Age distribution by class")
        plt.tight_layout()
        plt.savefig(os.path.join(PLOT_DIR, "age_hist_by_class.png"))

    # Boxplots for some features
    def boxplot_feature(col, title):
        if col in clean.columns and "sepsis" in clean.columns:
            plt.figure()
            data = [clean.loc[clean["sepsis"]==0, col], clean.loc[clean["sepsis"]==1, col]]
            plt.boxplot(data, labels=["Non-sepsis","Sepsis"])
            plt.title(title)
            plt.tight_layout()
            plt.savefig(os.path.join(PLOT_DIR, f"box_{col}.png"))

    for col in ["heart_rate","resp_rate","temp_c","mean_bp","spo2","lactate","wbc","platelets","creatinine"]:
        boxplot_feature(col, f"{col.replace('_',' ').title()} by class")

    # Correlation heatmap (z-scored)
    z_cols = [c for c in clean.columns if c.startswith("z_")]
    if len(z_cols) >= 2:
        corr = clean[z_cols].corr()
        plt.figure(figsize=(7,6))
        plt.imshow(corr, interpolation='nearest')
        plt.colorbar()
        plt.xticks(range(len(z_cols)), z_cols, rotation=90)
        plt.yticks(range(len(z_cols)), z_cols)
        plt.title("Correlation heatmap (z-scored features)")
        plt.tight_layout()
        plt.savefig(os.path.join(PLOT_DIR, "corr_heatmap.png"))

    print("Saved plots to", PLOT_DIR)

if __name__ == "__main__":
    main()
