"""
data_prep.py — Stage 2: data cleaning, feature engineering, leakage-safe
preprocessing and splitting for the Hospital Readmission pipeline.

Implement the functions below; they are imported by Data_Preparation.ipynb and by
src/train.py. Expected end state: a model frame of ~69,973 rows at ~9% positive, and a
stratified train/val/test split with the preprocessor fit on the TRAIN split only.
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.preprocessing import FunctionTransformer
from sklearn.model_selection import train_test_split

import config as cfg

ID_CODE_COLS = ["admission_type_id", "discharge_disposition_id", "admission_source_id"]

# ---------------------------------------------------------------------------------------
# Module constants (defaults used only when the matching key is absent from config.py)
# ---------------------------------------------------------------------------------------
#: identifiers / raw label columns that must never reach the feature matrix
LEAKY_COLS = ["encounter_id", "patient_nbr", "readmitted"]

#: ICD-9 chapter ranges -> clinical category (Strack et al., 2014 grouping).
#: Justification: diag_1/2/3 hold ~700-900 distinct codes each. The ICD-9 taxonomy already
#: groups codes into clinically coherent body-system chapters, so this collapses the
#: cardinality ~70x without inventing an arbitrary mapping and generalises across hospitals.
ICD9_GROUPS = [
    ((390, 459), "Circulatory"), ((785, 785), "Circulatory"),
    ((460, 519), "Respiratory"), ((786, 786), "Respiratory"),
    ((520, 579), "Digestive"), ((787, 787), "Digestive"),
    ((580, 629), "Genitourinary"), ((788, 788), "Genitourinary"),
    ((800, 999), "Injury"),
    ((710, 739), "Musculoskeletal"),
    ((140, 239), "Neoplasms"),
]

#: age is an ordered 10-year band -> numeric midpoint keeps the ordering in one column
AGE_MIDPOINT = {
    "[0-10)": 5, "[10-20)": 15, "[20-30)": 25, "[30-40)": 35, "[40-50)": 45,
    "[50-60)": 55, "[60-70)": 65, "[70-80)": 75, "[80-90)": 85, "[90-100)": 95,
}

#: fixed, domain-defined specialty shortlist. Hard-coded (NOT learned from the frame) so the
#: grouping is deterministic and cannot leak information from the validation/test rows.
TOP_SPECIALTIES = [
    "InternalMedicine", "Emergency/Trauma", "Family/GeneralPractice", "Cardiology",
    "Surgery-General", "Nephrology", "Orthopedics", "Orthopedics-Reconstructive",
    "Radiologist", "Pulmonology", "Psychiatry", "Urology", "ObstetricsandGynecology",
    "Surgery-Cardiovascular/Thoracic", "Gastroenterology",
]

#: a medication column whose modal value covers more than this share is treated as
#: zero-variance (e.g. examide / citoglipton are constant "No") and dropped in clean().
NEAR_CONSTANT_THRESHOLD = 0.999


def _cfg(name, default):
    """Read a key from config.py, falling back to a module default if it is not defined."""
    value = getattr(cfg, name, None)
    return default if value is None else value


# =======================================================================================
# Loading
# =======================================================================================
def load_raw():
    """TODO: read cfg.RAW_CSV with na_values=['?'] so '?' becomes NaN; return the DataFrame."""
    path = _cfg("RAW_CSV", "data/diabetic_data.csv")
    df = pd.read_csv(path, na_values=["?"], low_memory=False)
    return df


# =======================================================================================
# Stage 2.1 — Data cleaning
# =======================================================================================
def clean(df):
    """TODO (Stage 2.1): keep the FIRST encounter per patient (drop_duplicates('patient_nbr'))
    to prevent patient-level leakage; drop expired/hospice discharges
    (discharge_disposition_id in cfg.EXPIRED_HOSPICE_DISPOSITIONS); build the binary target
    cfg.TARGET = 1 if readmitted == '<30' else 0; drop the cfg.DROP_COLS that are present.
    Return the cleaned frame (~69,973 rows, ~9% positive)."""
    # Stage 2.1 — cleaning.
    #
    # 2.1.1  '?' -> NaN (re-applied defensively), whitespace stripped, 'Unknown/Invalid' gender
    #        treated as missing, and the unusable columns removed: cfg.DROP_COLS (e.g. weight
    #        ~97% missing, payer_code ~40% and purely administrative) plus every constant /
    #        near-constant medication column, which carries no signal.
    # 2.1.2  keep the FIRST encounter per patient (drop_duplicates('patient_nbr')) so rows are
    #        independent and no patient can span two splits; drop expired/hospice discharges
    #        (discharge_disposition_id in cfg.EXPIRED_HOSPICE_DISPOSITIONS), whose label is
    #        structurally invalid because those patients cannot be readmitted.
    # 2.1.3  build the binary target cfg.TARGET = 1 if readmitted == '<30' else 0.
    #
    # The original index is preserved so downstream splits can be traced back to raw rows.
    # Returns the cleaned frame (~69,973 rows, ~9% positive).
    target = _cfg("TARGET", "readmitted_30d")
    expired = list(_cfg("EXPIRED_HOSPICE_DISPOSITIONS", [11, 13, 14, 19, 20, 21]))
    drop_cols = list(_cfg("DROP_COLS", ["weight", "payer_code"]))
    med_cols = list(_cfg("MED_COLS", []))

    d = df.copy()

    # ---- 2.1.1 invalid / missing value treatment --------------------------------------
    obj_cols = d.select_dtypes(include="object").columns
    d[obj_cols] = (d[obj_cols]
                   .apply(lambda s: s.str.strip())
                   .replace({"?": np.nan, "": np.nan, "None": np.nan, "nan": np.nan}))
    if "gender" in d.columns:
        # 3 records carry 'Unknown/Invalid' -> treat as missing, keep the rows for imputation
        d["gender"] = d["gender"].replace("Unknown/Invalid", np.nan)

    # ---- 2.1.2 patient deduplication + invalid records --------------------------------
    # encounter_id increases with admission time, so sorting makes keep='first' the earliest stay
    if "encounter_id" in d.columns:
        d = d.sort_values("encounter_id")
    if "patient_nbr" in d.columns:
        d = d.drop_duplicates(subset="patient_nbr", keep="first")
    if "discharge_disposition_id" in d.columns:
        d = d[~d["discharge_disposition_id"].isin(expired)]

    # ---- 2.1.3 target variable creation ----------------------------------------------
    d[target] = (d["readmitted"] == "<30").astype(int)

    # ---- 2.1.1 (cont.) unusable / zero-variance columns ------------------------------
    protected = {target}
    to_drop = [c for c in drop_cols if c in d.columns and c not in protected]
    constant = [c for c in d.columns
                if c not in protected and c not in to_drop and d[c].nunique(dropna=False) <= 1]
    near_constant = [
        c for c in med_cols
        if c in d.columns and c not in to_drop and c not in constant
        and d[c].value_counts(normalize=True, dropna=False).iloc[0] > NEAR_CONSTANT_THRESHOLD
    ]
    # the raw label is always removed: the binary target replaces it
    label_cols = ["readmitted"] if "readmitted" in d.columns else []

    d = d.drop(columns=to_drop + constant + near_constant + label_cols)

    clean.dropped_ = {"unusable": to_drop, "constant": constant,
                      "near_constant": near_constant, "label": label_cols}
    return d


# =======================================================================================
# Stage 2.2 — Feature engineering
# =======================================================================================
def map_icd9(code):
    """Map a single raw ICD-9 code to a clinical category (2.2.1)."""
    if pd.isna(code):
        return "Missing"
    s = str(code).strip()
    if s.upper().startswith(("V", "E")):          # V = supplementary, E = external cause
        return "Other"
    try:
        v = float(s)
    except ValueError:
        return "Other"
    if 250 <= v < 251:                            # 250.xx = diabetes mellitus
        return "Diabetes"
    for (low, high), label in ICD9_GROUPS:
        if low <= v <= high:
            return label
    return "Other"


def engineer_features(df):
    """TODO (Stage 2.2): group ICD9 diag_1/2/3 into clinical categories; map the age band to a
    numeric midpoint; service_utilization = number_outpatient + number_emergency + number_inpatient;
    num_med_changes = count of 'Up'/'Down' across cfg.MED_COLS; collapse high-cardinality
    medical_specialty to top-k + 'Other'. Return the engineered frame."""
    # Stage 2.2 — dataset-specific feature engineering.
    #
    # 2.2.1  diag_1/2/3 -> clinical categories via map_icd9 (circulatory, respiratory, diabetes,
    #        injury, digestive, genitourinary, musculoskeletal, neoplasms, other, missing),
    #        replaced IN PLACE so the column names stay diag_1/2/3.
    # 2.2.2  age band -> numeric midpoint (in place, '[70-80)' -> 75); service_utilization =
    #        number_outpatient + number_emergency + number_inpatient (prior utilisation is the
    #        strongest EDA signal); num_med_changes = count of 'Up'/'Down' across cfg.MED_COLS
    #        (regimen instability proxy) and num_meds_prescribed (polypharmacy); medical_specialty
    #        collapsed in place to top-k + 'Other' with an explicit 'Missing' level (~49% missing,
    #        and the missingness is informative).
    #
    # Transforms are applied in place (rather than adding *_group / age_midpoint copies) so the
    # frame keeps the raw schema that src/generate_current_batch.py shifts (age + 10, diag_1 ->
    # 'Circulatory'). Every derived column it does not recompute is deliberately avoided, so a
    # simulated batch can never carry a stale engineered feature.
    #
    # Nominal ID codes (ID_CODE_COLS) are cast to string so they are one-hot encoded rather than
    # treated as magnitudes. Returns the engineered frame; the index is preserved.
    med_cols = list(_cfg("MED_COLS", []))
    top_specialties = list(_cfg("TOP_K_SPECIALTIES", TOP_SPECIALTIES))

    d = df.copy()
    created = []

    # ---- 2.2.1 ICD-9 diagnosis grouping (in place) -----------------------------------
    for c in ["diag_1", "diag_2", "diag_3"]:
        if c in d.columns:
            d[c] = d[c].apply(map_icd9).astype(object)
            created.append(c)

    # ---- 2.2.2 (a) age midpoint (in place) -------------------------------------------
    if "age" in d.columns:
        d["age"] = d["age"].map(AGE_MIDPOINT).astype(float)
        created.append("age")

    # ---- 2.2.2 (b) service utilisation ------------------------------------------------
    util_cols = [c for c in ["number_outpatient", "number_emergency", "number_inpatient"]
                 if c in d.columns]
    if util_cols:
        d["service_utilization"] = d[util_cols].sum(axis=1)
        created.append("service_utilization")

    # ---- 2.2.2 (c) medication-change count -------------------------------------------
    meds = [c for c in med_cols if c in d.columns]
    if meds:
        d["num_med_changes"] = d[meds].isin(["Up", "Down"]).sum(axis=1)
        d["num_meds_prescribed"] = d[meds].ne("No").sum(axis=1)
        created += ["num_med_changes", "num_meds_prescribed"]

    # ---- 2.2.2 (d) medical_specialty top-k grouping (in place) ------------------------
    if "medical_specialty" in d.columns:
        spec = d["medical_specialty"]
        d["medical_specialty"] = np.where(
            spec.isna(), "Missing",
            np.where(spec.isin(top_specialties), spec, "Other")).astype(object)
        created.append("medical_specialty")

    # ---- nominal ID codes are categorical, not numeric --------------------------------
    for c in ID_CODE_COLS:
        if c in d.columns:
            d[c] = d[c].astype("Int64").astype(str)

    engineer_features.created_ = created
    return d


# =======================================================================================
# Stage 2.3 — Feature transformation
# =======================================================================================
def get_feature_lists(df):
    """TODO: return (numeric_cols, categorical_cols), excluding cfg.TARGET."""
    # Return (numeric_cols, categorical_cols), excluding cfg.TARGET and identifier columns.
    target = _cfg("TARGET", "readmitted_30d")
    exclude = set([target]) | set(LEAKY_COLS)
    features = df.drop(columns=[c for c in exclude if c in df.columns])
    numeric = features.select_dtypes(include=np.number).columns.tolist()
    categorical = features.select_dtypes(exclude=np.number).columns.tolist()
    return numeric, categorical


def _one_hot_encoder():
    """OneHotEncoder(handle_unknown='ignore') across sklearn versions (sparse_output >= 1.2)."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def _to_str(v):
    """Canonical string for a categorical value; NaN stays NaN for the imputer."""
    if pd.isna(v):
        return np.nan
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    return str(v)


def as_category_strings(X):
    """Cast every categorical cell to a string inside the pipeline.

    ID codes are strings at training time but come back as ints after a CSV round-trip
    (reference sample, monitoring batch) or a JSON request. Without this step the encoder
    would see 1 != '1', treat the code as unknown and silently zero it out.
    """
    return pd.DataFrame(X).apply(lambda s: s.map(_to_str)).astype(object)


def build_preprocessor(numeric, categorical):
    """TODO (Stage 2.3): return ONE ColumnTransformer — numeric: median SimpleImputer + StandardScaler;
    categorical: constant SimpleImputer + OneHotEncoder(handle_unknown='ignore'). Do NOT fit it here
    (it is fit on the TRAIN split only)."""
    # Stage 2.3 — ONE ColumnTransformer, returned UNFITTED.
    #
    # numeric     : median SimpleImputer (robust to the right skew seen in EDA) + StandardScaler
    # categorical : string normalisation (ID codes stay '1', not 1, after CSV/JSON round-trips) +
    #               constant SimpleImputer ('Missing' kept as its own informative level) +
    #               OneHotEncoder(handle_unknown='ignore') so unseen categories are safe at
    #               inference time instead of raising.
    #
    # Fitting happens on the TRAIN split only (in the notebook / src/train.py), never here.
    numeric_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("to_str", FunctionTransformer(as_category_strings, feature_names_out="one-to-one")),
        ("impute", SimpleImputer(strategy="constant", fill_value="Missing")),
        ("onehot", _one_hot_encoder()),
    ])
    preprocessor = ColumnTransformer(
        transformers=[("num", numeric_pipe, list(numeric)),
                      ("cat", categorical_pipe, list(categorical))],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return preprocessor


# =======================================================================================
# Model frame + Stage 2.4 splitting
# =======================================================================================
def get_model_frame():
    """TODO: return engineer_features(clean(load_raw()))."""
    return engineer_features(clean(load_raw()))


def get_splits(df=None):
    """TODO (Stage 2.4): create a STRATIFIED train/val/test split using cfg.RANDOM_STATE,
    cfg.TEST_SIZE and cfg.VAL_SIZE (val taken from the train split). Return
    X_train, X_val, X_test, y_train, y_val, y_test. The class ratio must be preserved across splits;
    the preprocessor is fit on TRAIN only (in the notebook / train.py), never on the full data."""
    # Stage 2.4.1 — STRATIFIED train/val/test split created BEFORE any fitting.
    #
    # Uses cfg.RANDOM_STATE, cfg.TEST_SIZE and cfg.VAL_SIZE, with validation carved out of the
    # train split. Identifier columns are removed from X so nothing non-clinical leaks in. Rows
    # are already one-per-patient (2.1.2), so a stratified row split is also a patient split and
    # the class ratio is preserved across all three splits.
    #
    # Returns X_train, X_val, X_test, y_train, y_val, y_test.
    target = _cfg("TARGET", "readmitted_30d")
    random_state = _cfg("RANDOM_STATE", 42)
    test_size = _cfg("TEST_SIZE", 0.20)
    val_size = _cfg("VAL_SIZE", 0.20)

    if df is None:
        df = get_model_frame()

    y = df[target].astype(int)
    X = df.drop(columns=[c for c in [target] + LEAKY_COLS if c in df.columns])

    # 1) hold out the test set
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state)
    # 2) carve validation out of the TRAIN split
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=val_size, stratify=y_train, random_state=random_state)

    return X_train, X_val, X_test, y_train, y_val, y_test
