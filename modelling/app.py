import io
import os
import re
import sqlite3
import pandas as pd
import numpy as np
import streamlit as st
import altair as alt
import plotly.express as px
import streamlit.components.v1 as components  # (unused, kept for future)
import difflib, re
# =============================
# Simple, clean UI
# Preview → Hierarchy (and lots of helpers under the hood)
# =============================

st.set_page_config(page_title="QuantMatrix AI - Data Prep & Insights", layout="wide")

# Company branding header with brand colors
col1, col2, col3 = st.columns([1, 2, 1])
with col1:
    st.image("1.jpg", width=100)
with col2:
    st.markdown(
        """
        <div style="
            background: linear-gradient(135deg, #41C185 0%, #458EE2 100%);
            padding: 2rem;
            border-radius: 16px;
            margin-bottom: 2rem;
            color: white;
            box-shadow: 0 8px 32px rgba(65, 193, 133, 0.3);
        ">
            <h1 style="font-size: 48px; font-weight: 700; margin: 0; color: white; font-family: 'Inter', sans-serif;">🔬 QuantMatrix AI Solutions</h1>
            <h2 style="font-size: 32px; font-weight: 600; margin: 10px 0; color: white; font-family: 'Inter', sans-serif;">Data Preparation & Analytics Platform</h2>
            <p style="font-size: 18px; font-weight: 400; margin: 0; color: rgba(255,255,255,0.9); font-family: 'Inter', sans-serif;">Professional-grade data processing, merging, and insights generation</p>
        </div>
        """,
        unsafe_allow_html=True
    )
with col3:
    st.markdown("")
    st.markdown("")
    st.markdown("")
    st.markdown('<p style="font-size: 14px; color: #666666; text-align: right;">Powered by Streamlit & Python</p>', unsafe_allow_html=True)

# QuantMatrix AI Brand Styling
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    /* Global Font Family */
    .stApp, .stMarkdown, .stButton, .stSelectbox, .stTextInput, .stNumberInput, .stCheckbox, .stExpander, .stTabs {
        font-family: 'Inter', sans-serif !important;
    }
    
    /* Main Header with Brand Colors */
    .main-header {
        background: linear-gradient(135deg, #41C185 0%, #458EE2 100%);
        padding: 2rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        color: white;
        box-shadow: 0 8px 32px rgba(65, 193, 133, 0.3);
    }
    
    /* Metric Boxes with Brand Colors */
    .metric-box { 
        background: linear-gradient(135deg, #FFFFFF 0%, #F5F5F5 100%); 
        padding: 20px 24px; 
        border-radius: 16px; 
        border: 2px solid #41C185;
        box-shadow: 0 4px 16px rgba(65, 193, 133, 0.15);
        margin: 8px;
        transition: all 0.3s ease;
    }
    .metric-box:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 24px rgba(65, 193, 133, 0.25);
        border-color: #458EE2;
    }
    
    /* Section Headers with Brand Colors */
    .section-header {
        background: linear-gradient(135deg, #41C185 0%, #458EE2 100%);
        color: white;
        padding: 16px 24px;
        border-radius: 12px;
        margin: 24px 0 20px 0;
        font-weight: 600;
        font-size: 20px;
        box-shadow: 0 4px 16px rgba(65, 193, 133, 0.2);
    }
    
    /* Success Box with Brand Green */
    .success-box {
        background: linear-gradient(135deg, #41C185 0%, #4CD494 100%);
        border: 2px solid #41C185;
        border-radius: 12px;
        padding: 20px;
        margin: 16px 0;
        color: white;
        box-shadow: 0 4px 16px rgba(65, 193, 133, 0.2);
    }
    
    /* Warning Box with Brand Yellow */
    .warning-box {
        background: linear-gradient(135deg, #FFBD59 0%, #FFCF87 100%);
        border: 2px solid #FFBD59;
        border-radius: 12px;
        padding: 20px;
        margin: 16px 0;
        color: #333333;
        box-shadow: 0 4px 16px rgba(255, 189, 89, 0.2);
    }
    
    /* Info Box with Brand Blue */
    .info-box {
        background: linear-gradient(135deg, #458EE2 0%, #5A9EFF 100%);
        border: 2px solid #458EE2;
        border-radius: 12px;
        padding: 20px;
        margin: 16px 0;
        color: white;
        box-shadow: 0 4px 16px rgba(69, 142, 226, 0.2);
    }
    
    /* Primary Buttons with Brand Colors */
    .stButton > button[data-baseweb="button"] {
        background: linear-gradient(135deg, #41C185 0%, #4CD494 100%) !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 12px 24px !important;
        font-weight: 600 !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 16px !important;
        color: white !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 16px rgba(65, 193, 133, 0.3) !important;
    }
    .stButton > button[data-baseweb="button"]:hover {
        background: linear-gradient(135deg, #3AB075 0%, #41C185 100%) !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 24px rgba(65, 193, 133, 0.4) !important;
    }
    
    /* Secondary Buttons */
    .stButton > button:not([data-baseweb="button"]) {
        background: linear-gradient(135deg, #458EE2 0%, #5A9EFF 100%) !important;
        border: 2px solid #458EE2 !important;
        border-radius: 12px !important;
        padding: 10px 20px !important;
        font-weight: 500 !important;
        font-family: 'Inter', sans-serif !important;
        color: white !important;
        transition: all 0.3s ease !important;
    }
    .stButton > button:not([data-baseweb="button"]):hover {
        background: linear-gradient(135deg, #3A7BC8 0%, #458EE2 100%) !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(69, 142, 226, 0.3) !important;
    }
    
    /* Form Controls with Brand Styling */
    .stSelectbox > div > div {
        border-radius: 12px !important;
        border: 2px solid #E0E0E0 !important;
        transition: all 0.3s ease !important;
    }
    .stSelectbox > div > div:hover {
        border-color: #41C185 !important;
        box-shadow: 0 0 0 3px rgba(65, 193, 133, 0.1) !important;
    }
    
    .stTextInput > div > div > input {
        border-radius: 12px !important;
        border: 2px solid #E0E0E0 !important;
        padding: 12px 16px !important;
        font-family: 'Inter', sans-serif !important;
        transition: all 0.3s ease !important;
    }
    .stTextInput > div > div > input:focus {
        border-color: #41C185 !important;
        box-shadow: 0 0 0 3px rgba(65, 193, 133, 0.1) !important;
    }
    
    .stNumberInput > div > div > input {
        border-radius: 12px !important;
        border: 2px solid #E0E0E0 !important;
        padding: 12px 16px !important;
        font-family: 'Inter', sans-serif !important;
        transition: all 0.3s ease !important;
    }
    .stNumberInput > div > div > input:focus {
        border-color: #41C185 !important;
        box-shadow: 0 0 0 3px rgba(65, 193, 133, 0.1) !important;
    }
    
    /* Checkboxes with Brand Colors */
    .stCheckbox > div > div {
        border-radius: 8px !important;
        border: 2px solid #E0E0E0 !important;
        transition: all 0.3s ease !important;
    }
    .stCheckbox > div > div:hover {
        border-color: #41C185 !important;
    }
    
    /* Expanders with Brand Styling */
    .stExpander > div > div {
        border-radius: 16px !important;
        border: 2px solid #E0E0E0 !important;
        transition: all 0.3s ease !important;
    }
    .stExpander > div > div:hover {
        border-color: #458EE2 !important;
        box-shadow: 0 4px 16px rgba(69, 142, 226, 0.1) !important;
    }
    
    /* Tabs with Brand Colors */
    .stTabs > div > div > div > div {
        border-radius: 16px 16px 0 0 !important;
        background: #F5F5F5 !important;
        border: 2px solid #E0E0E0 !important;
        transition: all 0.3s ease !important;
    }
    .stTabs > div > div > div > div[aria-selected="true"] {
        background: linear-gradient(135deg, #41C185 0%, #4CD494 100%) !important;
        color: white !important;
        border-color: #41C185 !important;
    }
    
    /* Radio Buttons with Brand Colors */
    .stRadio > div > div > label {
        font-family: 'Inter', sans-serif !important;
        font-weight: 500 !important;
        color: #333333 !important;
    }
    
    /* Multiselect with Brand Styling */
    .stMultiSelect > div > div {
        border-radius: 12px !important;
        border: 2px solid #E0E0E0 !important;
        transition: all 0.3s ease !important;
    }
    .stMultiSelect > div > div:hover {
        border-color: #41C185 !important;
        box-shadow: 0 0 0 3px rgba(65, 193, 133, 0.1) !important;
    }
    
    /* Dataframe Styling */
    .stDataFrame {
        border-radius: 12px !important;
        border: 2px solid #E0E0E0 !important;
        overflow: hidden !important;
    }
    
    /* Custom Brand Elements */
    .brand-gradient {
        background: linear-gradient(135deg, #41C185 0%, #458EE2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .brand-accent {
        color: #FFBD59;
        font-weight: 600;
    }
    
    /* Responsive Design */
    @media (max-width: 768px) {
        .metric-box {
            padding: 16px 20px;
            margin: 6px;
        }
        .section-header {
            padding: 12px 20px;
            font-size: 18px;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------- Helpers --------

@st.cache_data(show_spinner=False, ttl=3600)  # Cache for 1 hour
def _read_from_bytes(data: bytes, filetype: str, sheet_name: str | int | None) -> pd.DataFrame:
    if filetype == "csv":
        return pd.read_csv(io.BytesIO(data), low_memory=False)
    else:
        xls = pd.ExcelFile(io.BytesIO(data))
        use_sheet = sheet_name if sheet_name is not None else 0
        return pd.read_excel(xls, sheet_name=use_sheet)

@st.cache_data(show_spinner=False, ttl=1800)  # Cache for 30 minutes
def quick_stats(df: pd.DataFrame) -> pd.DataFrame:
    # Only compute stats for first 10000 rows for performance
    sample_df = df.head(10000) if len(df) > 10000 else df
    stats = pd.DataFrame({
        "dtype": sample_df.dtypes.astype(str),
        "non_null": sample_df.notna().sum(),
        "nulls": sample_df.isna().sum(),
        "% null": (sample_df.isna().mean() * 100).round(2),
        "nunique": sample_df.nunique(dropna=True),
    }).reset_index(names=["column"]).sort_values(["% null", "column"], ascending=[False, True])
    return stats

def duplicate_name_table(df: pd.DataFrame) -> pd.DataFrame:
    cols = list(df.columns)
    s = pd.Series(cols)
    dupes = s.duplicated(keep=False)
    return pd.DataFrame({
        "#": range(1, len(cols) + 1),
        "column": cols,
        "duplicate_name": dupes.map({True: "Yes", False: "No"})
    })

def _clean_colname(name: object) -> str:
    if pd.isna(name):
        base = ""
    else:
        base = str(name)
    base = base.strip()
    # Treat Unnamed columns as empty
    if re.fullmatch(r"Unnamed: ?\d+", base, flags=re.I):
        base = ""
    # Replace non-alphanumeric with underscores and lowercase
    base = re.sub(r"[^0-9A-Za-z]+", "_", base)
    base = re.sub(r"_+", "_", base).strip("_")
    return base.lower()

def dedupe_columns(cols: list[str]) -> list[str]:
    out: list[str] = []
    seen: dict[str, int] = {}
    for i, c in enumerate(cols):
        c2 = _clean_colname(c)
        if not c2:
            c2 = f"col_{i+1}"
        if c2 in seen:
            seen[c2] += 1
            c2 = f"{c2}_{seen[c2]}"
        else:
            seen[c2] = 0
        out.append(c2)
    return out



# ---------- Preview filter UI helpers ----------
INT64_MAX = 2**63 - 1
INT64_MIN = -2**63

def _column_exceeds_int64_bounds(ser: pd.Series) -> bool:
    try:
        if pd.api.types.is_integer_dtype(ser) or pd.api.types.is_float_dtype(ser):
            nums = pd.to_numeric(ser, errors='coerce')
        else:
            nums = pd.to_numeric(ser, errors='coerce')
        if nums.empty:
            return False
        mx, mn = nums.max(skipna=True), nums.min(skipna=True)
        if pd.isna(mx) and pd.isna(mn):
            return False
        return (pd.notna(mx) and mx > INT64_MAX) or (pd.notna(mn) and mn < INT64_MIN)
    except Exception:
        return False

def make_arrow_safe_preview(df: pd.DataFrame, n: int = 100) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    preview = df.head(n).copy()
    for col in preview.columns:
        ser = preview[col]
        if _column_exceeds_int64_bounds(ser):
            preview[col] = ser.astype(str)
    if _column_exceeds_int64_bounds(preview.index.to_series()):
        preview = preview.reset_index(drop=True)
    return preview

def st_dataframe_safe(df: pd.DataFrame, n: int = 100, **kwargs) -> None:
    safe = make_arrow_safe_preview(df, n=n)
    st.dataframe(safe, **kwargs)

def _is_datetime_series(ser: pd.Series) -> bool:
    if pd.api.types.is_datetime64_any_dtype(ser):
        return True
    if ser.dtype == object:
        # try a small sample to avoid heavy parse on huge columns
        sample = ser.dropna().astype(str).head(200)
        if sample.empty:
            return False
        parsed = pd.to_datetime(sample, errors='coerce')
        return parsed.notna().mean() >= 0.8
    return False

@st.cache_data(show_spinner=False, ttl=1800)  # Cache for 30 minutes
def get_unique_values(df_hash, col_name):
    """Cache unique values for faster filtering"""
    # This is a workaround - we pass a hash of the dataframe
    # In real usage, the df would be passed directly
    return None

@st.cache_data(show_spinner=False, ttl=1800)  # Cache for 30 minutes
def _get_filter_options(df: pd.DataFrame, col_name: str):
    """Cache filter options to avoid recalculation"""
    if col_name not in df.columns:
        return None, None, None
    
    ser = df[col_name]
    
    # For categorical/text columns
    if not pd.api.types.is_numeric_dtype(ser):
        # Limit unique values for performance
        unique_vals = ser.dropna().unique()
        if len(unique_vals) > 1000:
            unique_vals = unique_vals[:1000]  # Limit to first 1000 for performance
        return "categorical", sorted(unique_vals.tolist()), None
    
    # For numeric columns
    nums = pd.to_numeric(ser, errors="coerce")
    nmin = float(nums.min(skipna=True)) if nums.notna().any() else 0.0
    nmax = float(nums.max(skipna=True)) if nums.notna().any() else 0.0
    return "numeric", nmin, nmax

def _build_filters_ui(df: pd.DataFrame, key_prefix: str) -> pd.DataFrame:
    """
    Multi-filter UI that applies only when the user clicks 'Apply filters'.
    Controls are arranged in columns for better UX.
    """
    if df is None or df.empty:
        return df

    draft_key = f"{key_prefix}_filters_draft"
    applied_key = f"{key_prefix}_filters_applied"
    st.session_state.setdefault(draft_key, {})
    st.session_state.setdefault(applied_key, {})

    with st.expander("🔍 Multi-Filter", expanded=False):
        # Step 1 (outside the form so selecting columns can update the UI layout):
        filter_cols = st.multiselect(
            "Select columns to filter:",
            options=list(df.columns),
            key=f"{key_prefix}_filter_cols",
            help="Select multiple columns, then set values and click 'Apply filters'"
        )

        # Step 2 (inside a form; changes are applied only on submit):
        draft: dict = st.session_state[draft_key]
        apply_filters = False
        clear_filters = False
        with st.form(f"{key_prefix}_filter_form", clear_on_submit=False):
            if filter_cols:
                grid = st.columns(3)
                for i, col in enumerate(filter_cols):
                    with grid[i % 3]:
                        st.markdown(f"**{col}**")
                        ftype, fvals, frange = _get_filter_options(df, col)

                        if ftype == "categorical" and fvals and len(fvals) <= 1000:
                            prev = draft.get(col, {}).get("values", [])
                            sel = st.multiselect(
                                "Values",
                                options=fvals,
                                default=prev,
                                key=f"{key_prefix}_vals_{col}"
                            )
                            draft[col] = {"type": "categorical", "values": sel}
                        elif ftype == "numeric" and frange is not None:
                            nmin, nmax = frange
                            prev = draft.get(col, {})
                            vmin = st.number_input(
                                "Min",
                                min_value=float(nmin),
                                max_value=float(nmax),
                                value=float(prev.get("min", nmin)),
                                key=f"{key_prefix}_min_{col}"
                            )
                            vmax = st.number_input(
                                "Max",
                                min_value=float(vmin),
                                max_value=float(nmax),
                                value=float(prev.get("max", nmax)),
                                key=f"{key_prefix}_max_{col}"
                            )
                            draft[col] = {"type": "numeric", "min": vmin, "max": vmax}
                        else:
                            prev = draft.get(col, {}).get("search", "")
                            term = st.text_input(
                                "Search",
                                value=str(prev),
                                key=f"{key_prefix}_search_{col}"
                            )
                            draft[col] = {"type": "search", "search": term}

            c1, c2, _ = st.columns([1, 1, 3])
            with c1:
                apply_filters = st.form_submit_button("Apply filters", type="primary", disabled=not bool(filter_cols))
            with c2:
                clear_filters = st.form_submit_button("Clear", disabled=not bool(st.session_state[applied_key]))

        # Update applied filters only after submit
        if apply_filters:
            st.session_state[applied_key] = st.session_state[draft_key].copy()
        if clear_filters:
            st.session_state[draft_key] = {}
            st.session_state[applied_key] = {}

        # Apply currently applied filters
        filtered = df.copy()
        info = []
        for col, spec in st.session_state[applied_key].items():
            if col not in filtered.columns:
                continue
            if spec.get("type") == "categorical" and spec.get("values"):
                filtered = filtered[filtered[col].isin(spec["values"])]
                info.append(f"{col}: {len(spec['values'])} values")
            elif spec.get("type") == "numeric":
                vmin = spec.get("min")
                vmax = spec.get("max")
                if vmin is not None and vmax is not None:
                    filtered = filtered[(filtered[col] >= vmin) & (filtered[col] <= vmax)]
                    info.append(f"{col}: {vmin}–{vmax}")
            elif spec.get("type") == "search" and spec.get("search"):
                term = str(spec.get("search", ""))
                filtered = filtered[filtered[col].astype(str).str.contains(term, case=False, na=False)]
                info.append(f"{col}: contains '{term}'")

        if info:
            st.success(f"✅ **Filters Applied:** {' | '.join(info)}")
            st.caption(f"📊 **Result:** **{len(filtered):,}** / {len(df):,} rows ({(len(filtered)/len(df)*100):.1f}%)")

        return filtered


def _col_kind(s: pd.Series) -> str:
    try:
        if pd.api.types.is_datetime64_any_dtype(s):
            return "datetime"
        if pd.api.types.is_numeric_dtype(s):
            return "numeric"
    except Exception:
        pass
    return "string"

def _norm_name(name: str) -> str:
    if name is None:
        return ""
    # normalize like your merge normalizer would
    n = re.sub(r"[^0-9a-zA-Z]+", " ", str(name).strip().lower())
    n = re.sub(r"\s+", " ", n).strip()
    # common suffix/prefix noise
    n = re.sub(r"\b(id|code|key|no|num|number)\b", "", n).strip()
    return n

def _normalize_series_for_join(s: pd.Series, kind: str) -> pd.Series:
    if kind == "numeric":
        # integers as-is; floats rounded for stable equality
        if pd.api.types.is_float_dtype(s):
            return pd.to_numeric(s, errors="coerce").round(6)
        return pd.to_numeric(s, errors="coerce")
    if kind == "datetime":
        # coerce and drop time part for robust matching
        return pd.to_datetime(s, errors="coerce").dt.date.astype("string")
    # string-ish
    s = s.astype("string")
    s = s.str.strip().str.lower()
    s = s.replace({"": pd.NA})
    return s

def _value_sets(s: pd.Series, unique_cap: int = 50000) -> set:
    """Return a set of unique non-null values (capped for performance)."""
    vals = pd.unique(s.dropna())
    # cap uniques for big columns
    if len(vals) > unique_cap:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(vals), size=unique_cap, replace=False)
        vals = vals[idx]
    return set(vals.tolist())

def _jaccard_and_cover(a: set, b: set) -> tuple[float, float, float]:
    if not a and not b:
        return 0.0, 0.0, 0.0
    inter = len(a & b)
    union = len(a | b) or 1
    jacc = inter / union
    left_cov = inter / (len(a) or 1)
    right_cov = inter / (len(b) or 1)
    return jacc, left_cov, right_cov

def _relation_guess(left_uni_ratio: float, right_uni_ratio: float) -> str:
    l = left_uni_ratio >= 0.9
    r = right_uni_ratio >= 0.9
    if l and r: return "1:1"
    if l and not r: return "1:M"
    if not l and r: return "M:1"
    return "M:M"

def suggest_join_keys(
    left_df: pd.DataFrame,
    right_df: pd.DataFrame,
    top_k: int = 20,
    only_viable: bool = True,
) -> pd.DataFrame:
    """
    Return a ranked DataFrame of candidate join keys between left_df and right_df.
    Columns: left_col, right_col, score, name_sim, jaccard, left_cov, right_cov,
             left_uni_ratio, right_uni_ratio, type_pair, relation_guess
    """
    results = []
    # lightweight precompute
    left_len  = max(len(left_df), 1)
    right_len = max(len(right_df), 1)

    # choose a manageable set of columns (exclude all-null)
    left_cols  = [c for c in left_df.columns if left_df[c].notna().any()]
    right_cols = [c for c in right_df.columns if right_df[c].notna().any()]

    # basic pruning: pair columns with compatible kinds OR similar names
    left_kinds  = {c: _col_kind(left_df[c])  for c in left_cols}
    right_kinds = {c: _col_kind(right_df[c]) for c in right_cols}
    name_norm_l = {c: _norm_name(c) for c in left_cols}
    name_norm_r = {c: _norm_name(c) for c in right_cols}

    # cache normalized series & unique sets
    norm_left  = {}
    norm_right = {}
    uniq_left  = {}
    uniq_right = {}

    for lc in left_cols:
        kind = left_kinds[lc]
        s = _normalize_series_for_join(left_df[lc], kind)
        norm_left[lc] = s
        uniq_left[lc] = _value_sets(s)

    for rc in right_cols:
        kind = right_kinds[rc]
        s = _normalize_series_for_join(right_df[rc], kind)
        norm_right[rc] = s
        uniq_right[rc] = _value_sets(s)

    for lc in left_cols:
        for rc in right_cols:
            lk, rk = left_kinds[lc], right_kinds[rc]
            type_compat = 1.0 if lk == rk else 0.7 if {"numeric","string"} == {lk, rk} else 0.6 if {"datetime","string"} == {lk, rk} else 0.5

            name_sim = difflib.SequenceMatcher(None, name_norm_l[lc], name_norm_r[rc]).ratio()

            # prune clearly unlikely pairs unless names look similar
            if lk != rk and name_sim < 0.55 and type_compat < 0.7:
                continue

            a, b = uniq_left[lc], uniq_right[rc]
            jacc, lcov, rcov = _jaccard_and_cover(a, b)

            # coverage too tiny? skip unless names are very similar
            if only_viable and (lcov < 0.03 and rcov < 0.03) and name_sim < 0.85:
                continue

            # uniqueness heuristics
            l_uni = left_df[lc].nunique(dropna=True) / left_len
            r_uni = right_df[rc].nunique(dropna=True) / right_len
            rel = _relation_guess(l_uni, r_uni)

            # relation weighting (prefer 1:1 or 1:M / M:1)
            rel_w = 1.00 if rel == "1:1" else 0.95 if rel in ("1:M","M:1") else 0.85

            # final score (0..1)
            score = (
                0.55 * jacc +
                0.25 * ((lcov + rcov) / 2.0) +
                0.15 * name_sim +
                0.05 * type_compat
            ) * rel_w

            results.append({
                "left_col": lc, "right_col": rc,
                "score": round(float(score), 6),
                "name_sim": round(float(name_sim), 4),
                "jaccard": round(float(jacc), 4),
                "left_cov": round(float(lcov), 4),
                "right_cov": round(float(rcov), 4),
                "left_uni_ratio": round(float(l_uni), 4),
                "right_uni_ratio": round(float(r_uni), 4),
                "type_pair": f"{lk}-{rk}",
                "relation_guess": rel,
            })

    if not results:
        return pd.DataFrame(columns=[
            "left_col","right_col","score","name_sim","jaccard","left_cov","right_cov",
            "left_uni_ratio","right_uni_ratio","type_pair","relation_guess"
        ])

    out = pd.DataFrame(results).sort_values(["score","jaccard","name_sim"], ascending=[False, False, False])
    # keep top_k distinct left/right pairs by best score
    out = out.head(top_k).reset_index(drop=True)
    return out
# ----------------------------------------------------------------------------- 



# -------- Reading --------
@st.cache_data(show_spinner=False)
def read_tabular_file(
    file, *,
    filetype: str,
    sheet_name: str | int | None,
) -> pd.DataFrame:
    """Read CSV/XLSX with default assumptions and clean/dedupe columns."""
    data = file.getvalue() if hasattr(file, "getvalue") else file.read()
    df = _read_from_bytes(data, filetype, sheet_name)
    df = df.loc[:, df.notna().any(axis=0)]
    df.columns = dedupe_columns(list(df.columns))
    return df

# ---- Arrow-safe display helpers (fix OverflowError during st.dataframe) ----
INT64_MAX = 2**63 - 1
INT64_MIN = -2**63

def _column_exceeds_int64_bounds(ser: pd.Series) -> bool:
    """Return True if any numeric values in the Series exceed Arrow int64 bounds."""
    try:
        if pd.api.types.is_integer_dtype(ser) or pd.api.types.is_float_dtype(ser):
            # Fast path for numeric dtypes
            with np.errstate(over='ignore'):
                max_v = pd.to_numeric(ser, errors='coerce').max(skipna=True)
                min_v = pd.to_numeric(ser, errors='coerce').min(skipna=True)
        else:
            # Object/mixed: inspect numeric view
            nums = pd.to_numeric(ser, errors='coerce')
            max_v = nums.max(skipna=True)
            min_v = nums.min(skipna=True)
        if pd.isna(max_v) and pd.isna(min_v):
            return False
        return (pd.notna(max_v) and max_v > INT64_MAX) or (pd.notna(min_v) and min_v < INT64_MIN)
    except Exception:
        return False

def make_arrow_safe_preview(df: pd.DataFrame, n: int = 100) -> pd.DataFrame:
    """Return a head(n) that is safe for Arrow by casting out-of-range integer columns to str."""
    if df is None or df.empty:
        return df
    preview = df.head(n).copy()
    for col in preview.columns:
        ser = preview[col]
        if _column_exceeds_int64_bounds(ser):
            preview[col] = ser.astype(str)
    # Also make sure the index won't trip Arrow if it's giant integers
    if _column_exceeds_int64_bounds(preview.index.to_series()):
        preview = preview.reset_index(drop=True)
    return preview

def st_dataframe_safe(df: pd.DataFrame, n: int = 100, **kwargs) -> None:
    safe = make_arrow_safe_preview(df, n=n)
    st.dataframe(safe, **kwargs)

# ---- Misc analytics helpers (kept; some are used in Hierarchy) ----
@st.cache_data(show_spinner=False)
def infer_column_types(df: pd.DataFrame) -> dict[str, list[str]]:
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    datetime_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
    object_like = [c for c in df.columns if c not in numeric_cols + datetime_cols]
    categorical_cols: list[str] = []
    for c in object_like:
        nunique = df[c].nunique(dropna=True)
        if nunique <= max(50, int(len(df) * 0.02)):
            categorical_cols.append(c)
    return {"numeric": numeric_cols, "categorical": categorical_cols, "datetime": datetime_cols}

def get_sampled_df(df: pd.DataFrame, max_rows: int | None) -> pd.DataFrame:
    if max_rows is None or max_rows <= 0 or len(df) <= max_rows:
        return df
    return df.sample(n=max_rows, random_state=42)

@st.cache_data(show_spinner=False)
def compute_numeric_corr(df_num: pd.DataFrame, method: str = "pearson") -> pd.DataFrame:
    if df_num.empty or df_num.shape[1] < 2:
        return pd.DataFrame()
    return df_num.corr(method=method)


def _cramers_v_from_table(table: np.ndarray) -> float:
    if table.size == 0:
        return np.nan
    n = table.sum()
    if n == 0:
        return np.nan
    row_sums = table.sum(axis=1)[:, None]
    col_sums = table.sum(axis=0)[None, :]
    expected = row_sums * col_sums / max(n, 1)
    with np.errstate(divide='ignore', invalid='ignore'):
        chi2 = np.nansum((table - expected) ** 2 / np.where(expected == 0, np.nan, expected))
    k = table.shape[1]
    r = table.shape[0]
    denom = n * (min(k - 1, r - 1))
    if denom <= 0:
        return np.nan
    return float(np.sqrt(max(chi2, 0.0) / denom))

@st.cache_data(show_spinner=False)
def compute_cramers_v_matrix(df_cat: pd.DataFrame, max_unique: int = 50) -> pd.DataFrame:
    cols = [c for c in df_cat.columns if df_cat[c].nunique(dropna=True) <= max_unique]
    if len(cols) < 2:
        return pd.DataFrame()
    result = pd.DataFrame(index=cols, columns=cols, dtype=float)
    for i, a in enumerate(cols):
        for j, b in enumerate(cols):
            if j < i:
                continue
            try:
                tab = pd.crosstab(df_cat[a], df_cat[b]).to_numpy()
                v = _cramers_v_from_table(tab)
            except Exception:
                v = np.nan
            result.loc[a, b] = v
            result.loc[b, a] = v
    np.fill_diagonal(result.values, 1.0)
    return result

@st.cache_data(show_spinner=False)
def missing_by_column(df: pd.DataFrame) -> pd.DataFrame:
    miss = df.isna().mean().sort_values(ascending=False)
    return miss.rename("pct_missing").rename_axis("column").reset_index()

# -------- Persistence (SQLite) --------
DB_PATH = os.path.join(os.path.dirname(__file__), "data_cache.db")

def _connect_db():
    return sqlite3.connect(DB_PATH)

def _sanitize_table_name(name: str) -> str:
    base = re.sub(r"[^0-9A-Za-z_]+", "_", name.strip().lower()).strip("_")
    if not base:
        base = "dataset"
    if not base.startswith("ds_"):
        base = f"ds_{base}"
    return base

@st.cache_data(show_spinner=False)
def list_saved_tables() -> list[str]:
    try:
        with _connect_db() as conn:
            cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;")
            return [r[0] for r in cur.fetchall()]
    except Exception:
        return []

def save_df_to_db(df: pd.DataFrame, name: str, replace: bool = True) -> str:
    tbl = _sanitize_table_name(name)
    with _connect_db() as conn:
        df.to_sql(tbl, conn, if_exists=("replace" if replace else "fail"), index=False)
    return tbl

def write_df_sqlite(df: pd.DataFrame, conn: sqlite3.Connection, table: str, overwrite: bool, chunk_size: int | None) -> None:
    """Robust writer: avoids 'too many SQL variables' and handles huge ints by casting to TEXT."""
    SQLITE_LIMIT = 999
    ncols = max(1, len(df.columns))
    max_rows = max(1, (SQLITE_LIMIT // ncols) - 1)
    effective_chunk = max_rows if (chunk_size is None or chunk_size <= 0) else min(chunk_size, max_rows)

    def _prepare_df_for_sqlite(frame: pd.DataFrame) -> pd.DataFrame:
        prepared = frame.copy()
        for col in prepared.columns:
            ser = prepared[col]
            nums = pd.to_numeric(ser, errors='coerce')
            max_val = nums.max(skipna=True)
            min_val = nums.min(skipna=True)
            if (pd.notna(max_val) and max_val > INT64_MAX) or (pd.notna(min_val) and min_val < INT64_MIN):
                prepared[col] = ser.astype(str)
                continue
            if pd.api.types.is_integer_dtype(ser):
                try:
                    if ser.max() > INT64_MAX or ser.min() < INT64_MIN:
                        prepared[col] = ser.astype(str)
                except Exception:
                    prepared[col] = ser.astype(str)
        return prepared

    df_sql = _prepare_df_for_sqlite(df)
    df_sql.to_sql(
        table,
        conn,
        if_exists=("replace" if overwrite else "fail"),
        index=False,
        chunksize=effective_chunk,
        method="multi",
    )

# -------- Sidebar: Upload --------
with st.sidebar:
    st.markdown('<div class="section-header" style="background: linear-gradient(135deg, #FFBD59 0%, #FFCF87 100%); color: #333333;">📁 **Data Upload & Management**</div>', unsafe_allow_html=True)
    
    # Primary upload
    st.markdown("**Primary Dataset**")
    file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx", "xls"], help="Upload your main dataset for analysis")

    filetype = None
    sheet_name = None

    if file is not None:
        name = file.name.lower()
        if name.endswith(".csv"):
            filetype = "csv"
        elif name.endswith(".xlsx") or name.endswith(".xls"):
            filetype = "xlsx"

        if filetype == "xlsx":
            try:
                xls = pd.ExcelFile(file)
                sheets = xls.sheet_names
                sheet_name = st.selectbox("Sheet", options=list(range(len(sheets))), format_func=lambda i: sheets[i])
                file.seek(0)
            except Exception as e:
                st.warning(f"Couldn't read sheets: {e}")
                sheet_name = None

    st.markdown("---")
    st.markdown("**💾 Database Management**")
    # Default OFF so uploads don't auto-save (prevents SQLite overflow during preview)
    st.checkbox("Auto-save uploads to DB", value=False, key="auto_save_uploads",
                help="Automatically save uploaded files to local database for quick reloads")
    
    st.markdown("**📂 Load Saved Dataset**")
    saved_tables = list_saved_tables()
    selected_table = st.selectbox("Select saved dataset", options=[""] + saved_tables, index=0, help="Choose from previously saved datasets")
    
    cols_load_del = st.columns(2)
    with cols_load_del[0]:
        load_from_db = st.button("🔄 Load", disabled=(selected_table == ""), use_container_width=True)
    with cols_load_del[1]:
        if st.button("🗑️ Delete", disabled=(selected_table == ""), use_container_width=True):
            try:
                delete_tbl = selected_table
                with _connect_db() as conn:
                    conn.execute(f"DROP TABLE IF EXISTS '{delete_tbl}'")
                    conn.commit()
                if st.session_state.get("active_table") == delete_tbl:
                    st.session_state["active_table"] = ""
                st.success(f"✅ Deleted '{delete_tbl}'")
                st.cache_data.clear()
                st.rerun()
            except Exception as e:
                st.error(f"❌ Delete failed: {e}")

    if load_from_db and selected_table:
        st.session_state["active_table"] = selected_table
        st.success(f"✅ Loaded '{selected_table}'")
    active_table = st.session_state.get("active_table", "")
    if active_table:
        st.markdown(f"**Active:** `{active_table}`")

    # ---- Secondary dataset ----
    st.markdown("---")
    st.markdown('<div class="section-header" style="background: linear-gradient(135deg, #458EE2 0%, #5A9EFF 100%);">🔄 **Secondary Dataset**</div>', unsafe_allow_html=True)
    sec_file = st.file_uploader("Upload secondary file", type=["csv", "xlsx", "xls"], key="sec_uploader", help="Upload a second dataset for merging")
    sec_filetype = None
    sec_sheet_name = None
    if sec_file is not None:
        sname = sec_file.name.lower()
        if sname.endswith(".csv"):
            sec_filetype = "csv"
        elif sname.endswith(".xlsx") or sname.endswith(".xls"):
            sec_filetype = "xlsx"
        if sec_filetype == "xlsx":
            try:
                sec_xls = pd.ExcelFile(sec_file)
                sec_sheets = sec_xls.sheet_names
                sec_sheet_name = st.selectbox("Secondary sheet", options=list(range(len(sec_sheets))), format_func=lambda i: sec_sheets[i], key="sec_sheet")
                sec_file.seek(0)
            except Exception as e:
                st.warning(f"Couldn't read secondary sheets: {e}")
                sec_sheet_name = None

    saved_tables2 = list_saved_tables()
    selected_table2 = st.selectbox("Saved datasets (secondary)", options=[""] + saved_tables2, index=0, key="sec_saved_sel")
    cols_sec = st.columns(2)
    with cols_sec[0]:
        load_sec = st.button("Load secondary", disabled=(selected_table2 == ""), key="btn_load_sec")
    with cols_sec[1]:
        if st.button("Clear secondary", key="btn_clear_sec"):
            st.session_state["active_table_secondary"] = ""
            st.session_state["secondary_df_buffer"] = None
            st.success("Cleared secondary dataset")

    if load_sec and selected_table2:
        st.session_state["active_table_secondary"] = selected_table2
        st.success(f"Loaded secondary '{selected_table2}'")

    # Secondary upload → preview-only unless auto-save is ON
    if sec_file is not None and sec_filetype is not None:
        try:
            sec_df = read_tabular_file(sec_file, filetype=sec_filetype, sheet_name=sec_sheet_name)
            st.session_state["secondary_df_buffer"] = sec_df

            if st.session_state.get("auto_save_uploads", False):
                sec_base = os.path.splitext(sec_file.name)[0]
                sec_save_default = f"{sec_base}_sheet{sec_sheet_name}_sec" if isinstance(sec_sheet_name, int) else f"{sec_base}_sec"
                sec_tbl = _sanitize_table_name(sec_save_default)
                with _connect_db() as conn:
                    write_df_sqlite(sec_df, conn, sec_tbl, overwrite=True, chunk_size=100_000)
                st.session_state["active_table_secondary"] = sec_tbl
                st.success(f"Saved secondary to table '{sec_tbl}' in {DB_PATH}")
                st.cache_data.clear()
            else:
                st.session_state["active_table_secondary"] = ""
                st.info("Secondary loaded for preview only (not saved to DB).")
        except Exception as e:
            st.warning(f"Secondary load failed: {e}")

    active_table_secondary = st.session_state.get("active_table_secondary", "")
    if active_table_secondary:
        st.caption(f"Active secondary dataset: {active_table_secondary}")

    # Global options
    st.markdown("---")
    st.markdown('<div class="section-header" style="background: linear-gradient(135deg, #41C185 0%, #4CD494 100%);">⚙️ **Global Settings**</div>', unsafe_allow_html=True)
    
    st.markdown("**📊 Chart Performance**")
    chart_sample_limit = st.number_input(
        "Max rows for charts", min_value=1_000, max_value=200_000, value=50_000, step=5_000,
        help="Limit chart data for better performance"
    )
    
    st.markdown("**🔍 Data Filtering**")
    filter_query = st.text_input(
        "Pandas query filter", value="", placeholder="Example: region == 'East' and sales > 1000",
        help="Advanced filtering using pandas query syntax. Leave blank for no filter."
    )

# If nothing chosen yet, prompt
has_primary_any = bool(st.session_state.get("active_table") or file)
has_secondary_any = bool(st.session_state.get("active_table_secondary") or st.session_state.get("secondary_df_buffer") is not None)

if not (has_primary_any or has_secondary_any):
    st.markdown('<div class="info-box">', unsafe_allow_html=True)
    st.markdown("**🚀 Get Started**")
    st.markdown("Upload a CSV/XLSX file or load a saved dataset from the sidebar to begin your data analysis journey.")
    st.markdown("</div>", unsafe_allow_html=True)
    st.stop()

# -------- Read & Basic Info --------
active_table = st.session_state.get("active_table")
source_label = ""
df = None

if active_table:
    try:
        df = pd.read_sql_query(f"SELECT * FROM '{active_table}'", _connect_db())
    except Exception:
        df = None
    source_label = f"database table: {active_table}"
elif file is not None:
    df = read_tabular_file(file, filetype=filetype, sheet_name=sheet_name)
    base_name = os.path.splitext(file.name)[0]
    save_default = f"{base_name}_sheet{sheet_name}" if isinstance(sheet_name, int) else base_name
    source_label = f"uploaded file: {file.name}"
    key_uploaded = f"{file.name}|{sheet_name}"
    if st.session_state.get("auto_save_uploads", False):
        if st.session_state.get("_saved_key") != key_uploaded:
            try:
                tbl = _sanitize_table_name(save_default)
                with _connect_db() as conn:
                    write_df_sqlite(df, conn, tbl, overwrite=True, chunk_size=100_000)
                st.session_state["_saved_key"] = key_uploaded
                st.success(f"Auto-saved upload to table '{tbl}' in {DB_PATH}")
                st.cache_data.clear()
            except Exception as e:
                st.warning(f"Auto-save failed: {e}")
else:
    sec_df_buffer = st.session_state.get("secondary_df_buffer")
    if active_table_secondary:
        df = pd.read_sql_query(f"SELECT * FROM '{active_table_secondary}'", _connect_db())
        source_label = f"database table: {active_table_secondary} (secondary)"
    elif sec_df_buffer is not None:
        df = sec_df_buffer
        source_label = "uploaded secondary (preview-only)"

# Apply saved column order if present
if "col_order" in st.session_state and isinstance(st.session_state["col_order"], list) and df is not None:
    saved = [c for c in st.session_state["col_order"] if c in df.columns]
    remaining = [c for c in df.columns if c not in saved]
    if saved:
        df = df[saved + remaining]

# Apply optional filter
if df is not None and filter_query:
    try:
        df = df.query(filter_query, engine="python")
    except Exception as e:
        st.warning(f"Filter ignored due to error: {e}")

rows, cols = (0, 0) if df is None else df.shape
mem = 0.0 if df is None else df.memory_usage(index=True).sum() / (1024 ** 2)

# Top metrics with brand styling
st.markdown('<div class="section-header" style="background: linear-gradient(135deg, #41C185 0%, #458EE2 100%);">📊 **Dataset Overview**</div>', unsafe_allow_html=True)

m1, m2, m3, m4 = st.columns([1, 1, 1, 1])
with m1:
    st.markdown('<div class="metric-box">', unsafe_allow_html=True)
    st.metric("📈 Rows", f"{rows:,}")
    st.markdown("</div>", unsafe_allow_html=True)
with m2:
    st.markdown('<div class="metric-box">', unsafe_allow_html=True)
    st.metric("🏷️ Columns", f"{cols:,}")
    st.markdown("</div>", unsafe_allow_html=True)
with m3:
    st.markdown('<div class="metric-box">', unsafe_allow_html=True)
    st.metric("❌ Null Cells", f"{0 if df is None else int(df.isna().sum().sum()):,}")
    st.markdown("</div>", unsafe_allow_html=True)
with m4:
    st.markdown('<div class="metric-box">', unsafe_allow_html=True)
    st.metric("💾 Memory", f"{mem:,.2f} MB")
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown(f"**Source:** `{source_label or '(none)'}`")

# Tabs with brand styling
st.markdown("---")
preview_tab, hierarchy_tab, sales_tab = st.tabs([
    "👀 **Data Preview**", 
    "🧬 **Hierarchy Analysis**", 
    "📊 **Sales & Analytics**"
])

# Dataset switcher - only show when needed
active_table_secondary = st.session_state.get("active_table_secondary", "")
if st.session_state.get("active_table") and active_table_secondary:
    # Only show dataset switcher in preview tab, not everywhere
    if 'show_dataset_switcher' not in st.session_state:
        st.session_state.show_dataset_switcher = False
    
    if st.button("🔄 Switch Dataset", key="toggle_dataset_switcher"):
        st.session_state.show_dataset_switcher = not st.session_state.show_dataset_switcher
    
    if st.session_state.show_dataset_switcher:
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.markdown("**🔄 Dataset Selection**")
        choice = st.radio(
            "Active Dataset:",
            options=["Primary", "Secondary"],
            index=0,
            horizontal=True,
            key="dataset_choice",
        )
        if choice == "Secondary":
            df = pd.read_sql_query(f"SELECT * FROM '{active_table_secondary}'", _connect_db())
            source_label = f"database table: {active_table_secondary} (secondary)"
        st.markdown("</div>", unsafe_allow_html=True)


# -------- Preview --------
with preview_tab:
    st.markdown('<div class="section-header" style="background: linear-gradient(135deg, #41C185 0%, #4CD494 100%);">👀 **Data Preview & Exploration**</div>', unsafe_allow_html=True)

    # Figure out secondary for preview: from DB if present, else memory buffer
    sec_df = None
    if active_table_secondary:
        try:
            sec_df = pd.read_sql_query(f"SELECT * FROM '{active_table_secondary}'", _connect_db())
        except Exception as e:
            st.markdown('<div class="warning-box">', unsafe_allow_html=True)
            st.warning(f"⚠️ Failed to load secondary dataset from DB: {e}")
            st.markdown("</div>", unsafe_allow_html=True)
            sec_df = st.session_state.get("secondary_df_buffer")
    else:
        sec_df = st.session_state.get("secondary_df_buffer")

    fdf_primary = None
    fdf_secondary = None

    # Primary + Secondary side-by-side (each with its own multi-filter)
    if df is not None and sec_df is not None and not sec_df.empty:
        colA, colB = st.columns(2)

        with colA:
            st.markdown("### Primary")
            fdf_primary = _build_filters_ui(df, key_prefix="pri")
            st_dataframe_safe(fdf_primary, n=100, use_container_width=True, height=380)

        with colB:
            st.markdown("### Secondary")
            fdf_secondary = _build_filters_ui(sec_df, key_prefix="sec")
            st_dataframe_safe(fdf_secondary, n=100, use_container_width=True, height=380)

    # Only Primary
    elif df is not None:
        st.markdown("### Primary")
        fdf_primary = _build_filters_ui(df, key_prefix="pri")
        st_dataframe_safe(fdf_primary, n=100, use_container_width=True, height=380)
        st.info("Load a secondary file to get a second independent filter panel.")

    # Only Secondary
    elif sec_df is not None and not sec_df.empty:
        st.markdown("### Secondary (no primary loaded)")
        fdf_secondary = _build_filters_ui(sec_df, key_prefix="sec")
        st_dataframe_safe(fdf_secondary, n=100, use_container_width=True, height=380)
    else:
        st.info("No data to preview yet.")

    # ---------- Key suggestions (between Preview and Merge) ----------
    have_both = (
        isinstance(fdf_primary, pd.DataFrame) and not fdf_primary.empty and
        isinstance(fdf_secondary, pd.DataFrame) and not fdf_secondary.empty
    )
    if have_both:
        st.markdown("---")
        with st.expander("🔎 Key suggestions (before Merge)", expanded=True):
            # persistent store for suggestions & selection
            if "join_key_suggestions" not in st.session_state:
                st.session_state["join_key_suggestions"] = None
            if "pick_sugg_idx" not in st.session_state:
                st.session_state["pick_sugg_idx"] = 0

            cA, cB, _ = st.columns([1,1,3])
            with cA:
                if st.button("Suggest join keys", type="primary", key="btn_suggest_keys"):
                    try:
                        sugg = suggest_join_keys(fdf_primary, fdf_secondary, top_k=30)
                        st.session_state["join_key_suggestions"] = sugg
                        if isinstance(sugg, pd.DataFrame) and not sugg.empty:
                            st.session_state["pick_sugg_idx"] = 0
                    except Exception as e:
                        st.error(f"Suggestion failed: {e}")

            with cB:
                # convenience: clear suggestions
                if st.button("Clear", key="btn_clear_sugg"):
                    st.session_state["join_key_suggestions"] = None

            sugg = st.session_state.get("join_key_suggestions")

            def _apply_mapping_row(row):
                """Prefill Merge UI using a chosen suggestion row."""
                left_col = str(row["left_col"])
                right_col = str(row["right_col"])
                same_name = (left_col == right_col)

                # We'll assume Primary is the left and Secondary is the right for suggestions
                st.session_state["merge_left_ds"]  = "Primary"
                st.session_state["merge_right_ds"] = "Secondary"

                if same_name:
                    st.session_state["merge_key_mode"]    = "Use common column names"
                    st.session_state["merge_keys_common"] = [left_col]
                    st.session_state["merge_left_keys"]   = []
                    st.session_state["merge_right_keys"]  = []
                else:
                    st.session_state["merge_key_mode"]    = "Map different column names"
                    st.session_state["merge_left_keys"]   = [left_col]
                    st.session_state["merge_right_keys"]  = [right_col]
                    st.session_state["merge_keys_common"] = []

                # sensible defaults for key normalization
                st.session_state["merge_norm_strip"]       = True
                st.session_state["merge_norm_as_str"]      = True
                st.session_state.setdefault("merge_norm_lower", False)
                st.session_state["merge_exclude_missing"]  = True  # default on

            if isinstance(sugg, pd.DataFrame) and not sugg.empty:
                # Show the ranked suggestions with the most useful fields up front
                st.dataframe(
                    sugg.assign(
                        pair=lambda d: d["left_col"].astype(str) + " ⇄ " + d["right_col"].astype(str)
                    )[
                        ["pair","score","relation_guess","jaccard","left_cov","right_cov",
                         "name_sim","left_uni_ratio","right_uni_ratio","type_pair"]
                    ],
                    use_container_width=True,
                    hide_index=True,
                    height=260
                )

                st.caption(
                    "Score blends value overlap (Jaccard & coverage), name similarity, type compatibility, "
                    "and a preference for 1:1 / 1:M relationships."
                )

                idx = st.selectbox(
                    "Pick a mapping to pre-fill Merge",
                    options=list(range(len(sugg))),
                    index=st.session_state.get("pick_sugg_idx", 0),
                    format_func=lambda i: f"{sugg.loc[i,'left_col']} ⇄ {sugg.loc[i,'right_col']} "
                                          f"(score {sugg.loc[i,'score']:.2f}, {sugg.loc[i,'relation_guess']})",
                    key="pick_sugg_idx",
                )

                b1, b2 = st.columns([1,1])
                with b1:
                    if st.button("Use this mapping", key="btn_apply_mapping"):
                        row = sugg.loc[idx]
                        _apply_mapping_row(row)
                        st.success(
                            f"Pre-filled Merge with {row['left_col']} ⇄ {row['right_col']} "
                            f"({row['relation_guess']}, score {row['score']:.2f})"
                        )
                        st.rerun()
                with b2:
                    if st.button("Use top suggestion", key="btn_apply_top"):
                        row = sugg.iloc[0]
                        _apply_mapping_row(row)
                        st.success(
                            f"Pre-filled Merge with {row['left_col']} ⇄ {row['right_col']} "
                            f"({row['relation_guess']}, score {row['score']:.2f})"
                        )
                        st.rerun()
            else:
                st.info("Click **Suggest join keys** to analyze both datasets.")
    # ---------- /Key suggestions ----------

    # ---------- Merge filtered datasets (simplified and guided) ----------
    have_both = isinstance(fdf_primary, pd.DataFrame) and not fdf_primary.empty and \
                isinstance(sec_df, pd.DataFrame) and not sec_df.empty

    if have_both:
        st.markdown("---")
        st.markdown('<div class="section-header" style="background: linear-gradient(135deg, #458EE2 0%, #5A9EFF 100%);">🔗 **Dataset Merging & Integration**</div>', unsafe_allow_html=True)
        with st.expander("🚀 **Advanced Merge Operations**", expanded=True):
            st.markdown("**Step 1: Choose your datasets**")
            st.caption("Primary dataset will be on the left, Secondary dataset will be on the right")
            
            # Show dataset info
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Primary Dataset**")
                st.caption(f"Shape: {fdf_primary.shape[0]:,} rows × {fdf_primary.shape[1]:,} cols")
                st.dataframe(fdf_primary.head(3), use_container_width=True, height=120)
            
            with col2:
                st.markdown("**Secondary Dataset**")
                st.caption(f"Shape: {sec_df.shape[0]:,} rows × {sec_df.shape[1]:,} cols")
                st.dataframe(sec_df.head(3), use_container_width=True, height=120)

            st.markdown("**Step 2: Choose merge options**")
            with st.form("merge_options_form"):
                merge_type = st.selectbox(
                    "How do you want to merge?",
                    options=[
                        "Keep all rows from Primary, add matching data from Secondary (Left Join)",
                        "Keep all rows from Secondary, add matching data from Primary (Right Join)", 
                        "Keep only rows that match in both datasets (Inner Join)",
                        "Keep all rows from both datasets (Outer Join)"
                    ],
                    index=0,
                    help="Left Join is usually what you want - keeps all your primary data and adds matching secondary data"
                )
                merge_options_submit = st.form_submit_button("Apply merge options")
            
            # Map selection to pandas merge type
            merge_type_map = {
                "Keep all rows from Primary, add matching data from Secondary (Left Join)": "left",
                "Keep all rows from Secondary, add matching data from Primary (Right Join)": "right",
                "Keep only rows that match in both datasets (Inner Join)": "inner", 
                "Keep all rows from both datasets (Outer Join)": "outer"
            }
            how = merge_type_map[merge_type]

            st.markdown("**Step 3: Choose merge keys**")
            
            # Find common columns
            primary_cols = list(fdf_primary.columns)
            secondary_cols = list(sec_df.columns)
            common_cols = sorted(set(primary_cols).intersection(set(secondary_cols)))
            
            if common_cols:
                st.success(f"✅ Found {len(common_cols)} common column(s): {', '.join(common_cols[:5])}{'...' if len(common_cols) > 5 else ''}")
                
                # Auto-suggest the best key
                best_key = None
                if common_cols:
                    # Look for common key-like names
                    key_like_names = ['id', 'key', 'code', 'name', 'product', 'customer', 'order', 'invoice']
                    for col in common_cols:
                        col_lower = col.lower()
                        if any(key_word in col_lower for key_word in key_like_names):
                            best_key = col
                            break
                    if not best_key and common_cols:
                        best_key = common_cols[0]  # fallback to first common column
                
                if best_key:
                    st.info(f"💡 Suggested key: **{best_key}** (auto-detected)")
                
                # Let user choose keys (defer effect until submit)
                with st.form("merge_keys_form"):
                    selected_keys = st.multiselect(
                        "Select merge key(s) - columns that should match between datasets",
                        options=common_cols,
                        default=[best_key] if best_key else [],
                        help="Choose columns that contain the same values in both datasets (e.g., product ID, customer ID)"
                    )
                    keys_submit = st.form_submit_button("Apply keys")
                
                if selected_keys:
                                st.markdown("**Step 4: Data Preparation & Merge**")
            
            # Enhanced key normalization options in organized columns
            st.markdown("**🔧 Key Normalization Settings** (to ensure proper matching)")
            
            # First row of options
            norm_row1 = st.columns(4)
            with norm_row1[0]:
                normalize_keys = st.checkbox("Normalize keys", value=True, 
                                            help="Clean and standardize key values for better matching")
            with norm_row1[1]:
                strip_spaces = st.checkbox("Strip spaces", value=True,
                                          help="Remove leading/trailing whitespace")
            with norm_row1[2]:
                convert_to_str = st.checkbox("Convert to string", value=True,
                                            help="Convert all keys to string format")
            with norm_row1[3]:
                case_insensitive = st.checkbox("Case insensitive", value=False,
                                              help="Treat uppercase/lowercase as same")
            
            # Second row of options
            norm_row2 = st.columns(3)
            with norm_row2[0]:
                memory_limit = st.selectbox(
                    "Memory limit for merge:",
                    options=["1 GB", "2 GB", "5 GB", "10 GB", "Unlimited"],
                    index=1,
                    help="Prevent memory overflow during large merges"
                )
            with norm_row2[1]:
                max_result_rows = st.number_input(
                    "Max result rows:",
                    min_value=1000000,
                    max_value=100000000,
                    value=10000000,
                    step=1000000,
                    help="Stop merge if result would exceed this limit"
                )
            with norm_row2[2]:
                show_memory_usage = st.checkbox("Show memory usage", value=True,
                                               help="Display memory consumption during merge")
            
            # Prepare dataframes with normalized keys
            left_df = fdf_primary.copy()
            right_df = sec_df.copy()
            
            # Normalize keys if requested
            if normalize_keys:
                for key in selected_keys:
                    # Left dataset
                    if key in left_df.columns:
                        # Smart numeric normalization
                        if pd.api.types.is_numeric_dtype(left_df[key]):
                            # Convert to numeric first, then to string to remove .0
                            left_df[key] = pd.to_numeric(left_df[key], errors='coerce').astype(str)
                        elif convert_to_str:
                            left_df[key] = left_df[key].astype(str)
                        
                        if strip_spaces:
                            left_df[key] = left_df[key].str.strip()
                        if case_insensitive:
                            left_df[key] = left_df[key].str.lower()
                        # Remove 'nan' strings and .0 from numeric strings
                        left_df[key] = left_df[key].replace(['nan', 'None', ''], pd.NA)
                        # Remove .0 from numeric strings (e.g., '5897.0' -> '5897')
                        left_df[key] = left_df[key].str.replace(r'\.0$', '', regex=True)
                    
                    # Right dataset  
                    if key in right_df.columns:
                        # Smart numeric normalization
                        if pd.api.types.is_numeric_dtype(right_df[key]):
                            # Convert to numeric first, then to string to remove .0
                            right_df[key] = pd.to_numeric(right_df[key], errors='coerce').astype(str)
                        elif convert_to_str:
                            right_df[key] = right_df[key].astype(str)
                        
                        if strip_spaces:
                            right_df[key] = right_df[key].str.strip()
                        if case_insensitive:
                            right_df[key] = right_df[key].str.lower()
                        # Remove 'nan' strings and .0 from numeric strings
                        right_df[key] = right_df[key].replace(['nan', 'None', ''], pd.NA)
                        # Remove .0 from numeric strings (e.g., '5897.0' -> '5897')
                        right_df[key] = right_df[key].str.replace(r'\.0$', '', regex=True)
            
            # Drop rows with missing keys
            left_df = left_df.dropna(subset=selected_keys)
            right_df = right_df.dropna(subset=selected_keys)
            
            # Show sample of normalized key values
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown(f"**Sample {selected_keys[0]} values from Primary (normalized):**")
                if not left_df.empty:
                    sample_primary = left_df[selected_keys[0]].head(10).tolist()
                    st.code(sample_primary)
                    # Show data type info
                    st.caption(f"Data type: {left_df[selected_keys[0]].dtype}")
                else:
                    st.warning("No valid keys after normalization")
            
            with col_b:
                st.markdown(f"**Sample {selected_keys[0]} values from Secondary (normalized):**")
                if not right_df.empty:
                    sample_secondary = right_df[selected_keys[0]].head(10).tolist()
                    st.code(sample_secondary)
                    # Show data type info
                    st.caption(f"Data type: {right_df[selected_keys[0]].dtype}")
                else:
                    st.warning("No valid keys after normalization")
            
            # Simple merge button - always show when keys are selected
            if selected_keys:
                st.markdown("---")
                st.markdown("### 🚀 Ready to Merge!")
                st.info(f"📊 **Selected {len(selected_keys)} merge key(s):** {', '.join(selected_keys)}")
                
                # Simple merge button
                if st.button("🚀 Merge Now", type="primary", use_container_width=True):
                    try:
                        # Basic validation
                        if not left_df.empty and not right_df.empty:
                            # Simple merge without complex analysis for now
                            with st.spinner(f"Merging {len(left_df):,} × {len(right_df):,} rows..."):
                                merged_df = pd.merge(
                                    left_df,
                                    right_df,
                                    how=how,
                                    left_on=selected_keys,
                                    right_on=selected_keys,
                                    suffixes=('_primary', '_secondary'),
                                    copy=False,
                                    indicator=True
                                )
                            
                            st.session_state["merged_df"] = merged_df
                            
                            # Show merge statistics
                            merge_stats = merged_df['_merge'].value_counts()
                            st.success(f"✅ Merge successful! Result: **{merged_df.shape[0]:,} rows × {merged_df.shape[1]:,} cols**")
                            
                            # Display merge indicator statistics
                            if '_merge' in merged_df.columns:
                                st.caption("**Merge Statistics:**")
                                for category, count in merge_stats.items():
                                    if category == 'both':
                                        st.caption(f"✅ Matched in both: {count:,} rows")
                                    elif category == 'left_only':
                                        st.caption(f"⬅️ Only in Primary: {count:,} rows")
                                    elif category == 'right_only':
                                        st.caption(f"➡️ Only in Secondary: {count:,} rows")
                                
                                # Remove merge indicator column for cleaner output
                                merged_df = merged_df.drop('_merge', axis=1)
                            
                            # Show result
                            st.markdown("**Merged dataset preview:**")
                            st.dataframe(merged_df.head(100), use_container_width=True, height=400)
                            
                            # Download option
                            csv_bytes = merged_df.to_csv(index=False).encode("utf-8")
                            st.download_button(
                                "📥 Download merged CSV",
                                data=csv_bytes,
                                file_name="merged_dataset.csv",
                                mime="text/csv",
                                use_container_width=True
                            )
                            
                            # Save to DB option
                            with st.expander("💾 Save to database (optional)"):
                                save_name = st.text_input("Table name:", value="merged_dataset")
                                if st.button("Save to SQLite"):
                                    try:
                                        tbl = _sanitize_table_name(save_name)
                                        with _connect_db() as conn:
                                            write_df_sqlite(merged_df, conn, tbl, overwrite=True, chunk_size=100_000)
                                        st.success(f"Saved to table '{tbl}' in database")
                                        st.cache_data.clear()
                                    except Exception as e:
                                        st.error(f"Save failed: {e}")
                        else:
                            st.error("❌ Cannot merge: One or both datasets are empty after normalization")
                    except Exception as e:
                        st.error(f"❌ Merge failed: {str(e)}")
                        st.info("💡 Try selecting different key columns or check if your data types match")
            else:
                st.info("Please select at least one merge key to continue")
            
            # Fallback: manual column mapping if no common columns
            if not common_cols:
                st.warning("❌ No common column names found between datasets")
                st.info("💡 Try renaming columns in one dataset to match the other, or use the 'Map different column names' option below")
                
                # Fallback: manual column mapping
                st.markdown("**Alternative: Map different column names**")
                col_a, col_b = st.columns(2)
                with col_a:
                    left_key = st.selectbox("Primary dataset key column:", options=primary_cols)
                with col_b:
                    right_key = st.selectbox("Secondary dataset key column:", options=secondary_cols)
                
                if left_key and right_key:
                    if st.button("🔗 Merge with different column names", type="primary"):
                        try:
                            merged_df = pd.merge(
                                fdf_primary,
                                sec_df,
                                how=how,
                                left_on=[left_key],
                                right_on=[right_key],
                                suffixes=('_primary', '_secondary'),
                                copy=False
                            )
                            
                            st.session_state["merged_df"] = merged_df
                            st.success(f"✅ Merge successful! Result: **{merged_df.shape[0]:,} rows × {merged_df.shape[1]:,} cols**")
                            
                            st.markdown("**Merged dataset preview:**")
                            st.dataframe(merged_df.head(100), use_container_width=True, height=400)
                            
                        except Exception as e:
                            st.error(f"❌ Merge failed: {str(e)}")
    else:
        st.caption("To enable merging, load **both** Primary and Secondary datasets first.")



# -------- Hierarchy --------
with hierarchy_tab:
    if df is None or df.empty:
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.markdown("**📊 Hierarchy Analysis**")
        st.markdown("Load a dataset in the sidebar to explore hierarchical relationships and build multi-level categorizations.")
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.markdown('<div class="section-header" style="background: linear-gradient(135deg, #FFBD59 0%, #FFCF87 100%); color: #333333;">🧬 **Hierarchical Data Analysis**</div>', unsafe_allow_html=True)
        st.markdown("**Build multi-level hierarchies** (e.g., Category → Subcategory → Brand) to view parent→child mappings and counts.")

        # Scope filter for entire hierarchy tab
        filt_l, filt_r = st.columns(2)
        with filt_l:
            hier_filter_col = st.selectbox("Filter column (hierarchy scope)", options=[""] + list(df.columns), index=0, key="hier_filter_col")
        with filt_r:
            if hier_filter_col:
                try:
                    vals = pd.Index(df[hier_filter_col].astype(str).dropna().unique())
                    vals = pd.Index(sorted(vals))
                    hier_filter_val = st.selectbox("Filter value", options=[""] + list(vals)[:2000], index=0, key="hier_filter_val")
                except Exception:
                    hier_filter_val = ""
            else:
                hier_filter_val = ""
        df_h = df
        if hier_filter_col and hier_filter_val:
            try:
                df_h = df[df[hier_filter_col].astype(str) == hier_filter_val]
            except Exception:
                df_h = df

        level_cols = st.multiselect("Hierarchy levels (in order)", options=list(df.columns), key="hier_levels")
        if level_cols:
            # 1) Level-wise funnel-style bar (unique values per level)
            meta = pd.DataFrame({
                "level": level_cols,
                "unique_values": [df_h[c].nunique(dropna=False) for c in level_cols],
            })
            meta["level_order"] = range(len(meta))
            st.markdown("**Level-wise coverage (unique values per selected level)**")
            funnel = (
                alt.Chart(meta)
                .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
                .encode(
                    y=alt.Y("level:N", sort=level_cols, title="Level"),
                    x=alt.X("unique_values:Q", title="Unique values"),
                    tooltip=["level", "unique_values"],
                    color=alt.value("#4c78a8"),
                )
                .properties(height=max(120, 40 * len(level_cols)))
            )
            st.altair_chart(funnel, use_container_width=True)

            # 2) Path counts table + bar of top paths
            @st.cache_data(show_spinner=False)
            def _group_counts_cached(frame: pd.DataFrame, cols: tuple[str, ...]) -> pd.DataFrame:
                return frame.groupby(list(cols), dropna=False).size().reset_index(name="rows")

            group_counts = _group_counts_cached(df_h, tuple(level_cols))
            # Render NaN as <NA> for clearer labels
            lab = group_counts[level_cols].copy()
            for c in level_cols:
                lab[c] = lab[c].where(lab[c].notna(), other="<NA>")
            group_counts["path"] = lab.astype(str).agg(" / ".join, axis=1)

            st.markdown("**Path counts** (groupby across selected levels)")
            show_full = st.checkbox("Show full table (may be large)", value=False, key="show_full_paths")
            gc_sorted = group_counts.sort_values("rows", ascending=False)
            if show_full:
                st_dataframe_safe(gc_sorted, n=5000, use_container_width=True, height=420)
            else:
                st_dataframe_safe(gc_sorted.head(500), n=500, use_container_width=True, height=420)

            st.markdown("**Top paths by rows**")
            top_paths = group_counts.nlargest(20, "rows")
            bar = (
                alt.Chart(top_paths)
                .mark_bar()
                .encode(
                    x=alt.X("rows:Q", title="Rows"),
                    y=alt.Y("path:N", sort="-x", title=""),
                    tooltip=["path", "rows"],
                    color=alt.value("#72b7b2"),
                )
                .properties(height=max(200, 18 * len(top_paths)))
            )
            st.altair_chart(bar, use_container_width=True)

            with st.expander("Hierarchy explorer (nested)", expanded=False):
                st.caption("Step-by-step selector; choose a value at each level to drill down.")
                sub = df_h.copy()
                for i, lvl in enumerate(level_cols):
                    try:
                        vals = sub[lvl].astype(str).fillna("<NA>").value_counts().reset_index()
                        vals.columns = [lvl, "rows"]
                        options = [""] + list(vals[lvl].head(2000))
                        sel = st.selectbox(f"{i+1}) {lvl}", options=options, index=0, key=f"hexp_sel_{i}_{lvl}")
                        if not sel:
                            break
                        sub = sub[sub[lvl].astype(str).fillna("<NA>") == sel]
                    except Exception as e:
                        st.warning(f"Failed to load level '{lvl}': {e}")
                        break
                st.metric("Rows at current selection", f"{len(sub):,}")
                st_dataframe_safe(sub, n=100, use_container_width=True)

            st.markdown("---")
            st.subheader("🔀 Cross hierarchy drill")

            if len(level_cols) < 2:
                st.info("Select at least two hierarchy levels above.")
            else:
                ctrl1, ctrl2 = st.columns(2)
                with ctrl1:
                    min_rows = st.number_input("Min rows to display", min_value=0, max_value=1_000_000, value=0, step=10, key="xh_min_rows")
                with ctrl2:
                    pair_options = [f"{level_cols[i]} vs {level_cols[i+1]}" for i in range(len(level_cols) - 1)]
                    pair_idx = st.radio(
                        "Level pair",
                        options=list(range(len(pair_options))),
                        index=0,
                        format_func=lambda i: pair_options[i],
                        horizontal=True,
                        key="xh_pair_idx",
                    )

                with st.expander("Counts matrix (adjacent levels)", expanded=True):
                    try:
                        row_level = level_cols[pair_idx]
                        col_level = level_cols[pair_idx + 1]
                        sub2 = df_h.copy()
                        sub2[row_level] = sub2[row_level].astype(str).fillna("<NA>")
                        sub2[col_level] = sub2[col_level].astype(str).fillna("<NA>")
                        mat = pd.crosstab(sub2[row_level], sub2[col_level])
                        if min_rows > 0:
                            mat = mat.where(mat >= min_rows, other=0)
                        row_order = mat.sum(axis=1).sort_values(ascending=False).index
                        col_order = mat.sum(axis=0).sort_values(ascending=False).index
                        MAX_DIM = 80
                        row_order = row_order[:MAX_DIM]
                        col_order = col_order[:MAX_DIM]
                        mat = mat.loc[row_order, col_order]
                        cm_l, cm_r = st.columns([3, 1])
                        with cm_l:
                            # Index could be giant ints; make safe by resetting
                            st.dataframe(mat.reset_index(), use_container_width=True, height=min(800, 32 * (len(mat.index) + 2)))
                        with cm_r:
                            st.markdown("**Unique counts**")
                            uc = pd.DataFrame({
                                row_level: [sub2[row_level].nunique()],
                                col_level: [sub2[col_level].nunique()],
                            }).T.reset_index()
                            uc.columns = ["level", "nunique"]
                            st_dataframe_safe(uc, n=200, use_container_width=True, height=120)

                            st.markdown("**Row-level mapping**")
                            row_map = sub2.groupby(row_level)[col_level].nunique().rename("unique_children").to_frame()
                            row_tot = sub2.groupby(row_level).size().rename("rows_total")
                            row_map = row_map.join(row_tot).sort_values(["unique_children", "rows_total"], ascending=[False, False]).reset_index()
                            row_filter = st.selectbox(
                                "Filter",
                                options=["All", "Only unique (1 child)", "Only non-unique (>1)"],
                                index=0,
                                key=f"row_map_filter_{row_level}_{col_level}",
                            )
                            if row_filter == "Only unique (1 child)":
                                row_map = row_map[row_map["unique_children"] == 1]
                            elif row_filter == "Only non-unique (>1)":
                                row_map = row_map[row_map["unique_children"] > 1]
                            st_dataframe_safe(row_map.head(300), n=300, use_container_width=True, height=300)

                            st.markdown("**Column-level mapping**")
                            col_map = sub2.groupby(col_level)[row_level].nunique().rename("unique_parents").to_frame()
                            col_tot = sub2.groupby(col_level).size().rename("rows_total")
                            col_map = col_map.join(col_tot).sort_values(["unique_parents", "rows_total"], ascending=[False, False]).reset_index()
                            col_filter = st.selectbox(
                                "Filter",
                                options=["All", "Only unique (1 parent)", "Only non-unique (>1)"],
                                index=0,
                                key=f"col_map_filter_{row_level}_{col_level}",
                            )
                            if col_filter == "Only unique (1 parent)":
                                col_map = col_map[col_map["unique_parents"] == 1]
                            elif col_filter == "Only non-unique (>1)":
                                col_map = col_map[col_map["unique_parents"] > 1]
                            st_dataframe_safe(col_map.head(300), n=300, use_container_width=True, height=300)
                    except Exception as e:
                        st.warning(f"Counts matrix failed: {e}")

                # Uniqueness across remaining levels
                with st.expander("Uniqueness across remaining levels", expanded=False):
                    try:
                        anchor_level = st.selectbox("Anchor level", options=level_cols, index=0, key="anchor_level_uniq")
                        other_levels = [lvl for lvl in level_cols if lvl != anchor_level]
                        if not other_levels:
                            st.info("Select at least two levels to compute uniqueness.")
                        else:
                            dfu = df_h.copy()
                            dfu = dfu[[anchor_level] + other_levels].copy()
                            for c in [anchor_level] + other_levels:
                                dfu[c] = dfu[c].astype(str).fillna("<NA>")
                            combo = dfu[other_levels].agg(" / ".join, axis=1)
                            tmp = pd.DataFrame({anchor_level: dfu[anchor_level], "combo": combo})
                            stats = tmp.groupby(anchor_level).agg(unique_combos=("combo", "nunique"), rows_total=("combo", "size")).reset_index()
                            flt = st.selectbox("Filter", options=["All", "Only unique (1 combo)", "Only non-unique (>1)"], index=0)
                            if flt == "Only unique (1 combo)":
                                stats = stats[stats["unique_combos"] == 1]
                            elif flt == "Only non-unique (>1)":
                                stats = stats[stats["unique_combos"] > 1]
                            st_dataframe_safe(stats.head(1000), n=1000, use_container_width=True, height=360)
                    except Exception as e:
                        st.warning(f"Uniqueness computation failed: {e}")

                # Treemap across the chosen levels
                with st.expander("Tree view (Treemap)", expanded=True):
                    cmin, cmax = st.columns(2)
                    with cmin:
                        min_rows_t = st.number_input("Min rows per node (treemap)", min_value=0, max_value=1_000_000, value=int(min_rows), step=10)
                    with cmax:
                        max_rows_t = st.number_input("Max rows per node (treemap, 0 = no max)", min_value=0, max_value=1_000_000, value=0, step=10)
                    try:
                        counts = (
                            df_h[level_cols]
                            .copy()
                            .astype({lvl: str for lvl in level_cols})
                            .fillna("<NA>")
                            .value_counts(level_cols)
                            .reset_index(name="rows")
                        )
                        if min_rows_t > 0:
                            counts = counts[counts["rows"] >= min_rows_t]
                        if max_rows_t > 0:
                            counts = counts[counts["rows"] <= max_rows_t]
                        first = level_cols[0]
                        figt = px.treemap(counts, path=level_cols, values="rows", color=first)
                        figt.update_traces(textinfo="label+value")
                        figt.update_layout(height=700, uniformtext_minsize=10, uniformtext_mode="hide", margin=dict(t=10, l=0, r=0, b=0))
                        st.plotly_chart(figt, use_container_width=True)
                    except Exception as e:
                        st.warning(f"Tree view failed: {e}")


# -------- Sales / Group By (Tab 3) --------
with sales_tab:
    st.markdown('<div class="section-header" style="background: linear-gradient(135deg, #41C185 0%, #458EE2 100%);">📊 **Sales Analytics & Group By Operations**</div>', unsafe_allow_html=True)

    # Load Secondary like Preview tab
    sec_df_sales = None
    if active_table_secondary:
        try:
            sec_df_sales = pd.read_sql_query(f"SELECT * FROM '{active_table_secondary}'", _connect_db())
        except Exception as e:
            st.markdown('<div class="warning-box">', unsafe_allow_html=True)
            st.warning(f"⚠️ Failed to load secondary dataset from DB for Sales tab: {e}")
            st.markdown("</div>", unsafe_allow_html=True)
            sec_df_sales = st.session_state.get("secondary_df_buffer")
    else:
        sec_df_sales = st.session_state.get("secondary_df_buffer")

    # Available sources
    sources = []
    if isinstance(df, pd.DataFrame) and not df.empty:
        sources.append("Primary")
    if isinstance(sec_df_sales, pd.DataFrame) and not sec_df_sales.empty:
        sources.append("Secondary")

    if not sources:
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.markdown("**📊 Sales Analytics Ready**")
        st.markdown("Load at least one dataset (Primary or Secondary) to begin your sales analysis and group-by operations.")
        st.markdown("</div>", unsafe_allow_html=True)
        st.stop()

    # 1) Pick dataset (Primary / Secondary)
    sales_src = st.radio(
        "Sales dataset",
        options=sources,
        index=0,
        horizontal=True,
        key="sales_tab_src"
    )
    base_df = df if sales_src == "Primary" else sec_df_sales
    if base_df is None or base_df.empty:
        st.error("Selected dataset is empty or not loaded properly.")
        st.info("Please load a valid dataset first.")
        cols = []
    else:
        cols = list(map(str, base_df.columns))

    # Optional quick filter panel
    with st.expander("🔎 Optional filters before aggregation", expanded=False):
        fdf_sales = _build_filters_ui(base_df, key_prefix="sales")
        st_dataframe_safe(fdf_sales, n=100, use_container_width=True, height=260)

    work_df = fdf_sales if isinstance(fdf_sales, pd.DataFrame) else base_df

    # ---------------- Computed columns (before group by) ----------------
    st.markdown("---")
    with st.expander("🧩 Computed columns (create before grouping)", expanded=False):
        st.caption("Add one or more computed columns. They will be visible in the preview and available for grouping/aggregation.")

        # Keep specs in session
        if "sales_computed_specs" not in st.session_state:
            st.session_state["sales_computed_specs"] = []

        bmode = st.radio(
            "Builder mode",
            ["Column ±×÷ Column", "Column ±×÷ Constant", "IF condition… THEN … ELSE …", "Custom formula"],
            horizontal=False,
            key="sales_comp_mode"
        )

        new_name = st.text_input("New column name", value="", key="sales_comp_name")

        spec = None
        if bmode == "Column ±×÷ Column":
            c1, c2, c3 = st.columns([2,1,2])
            with c1:
                a_col = st.selectbox("Left column", options=cols, key="sales_comp_a")
            with c2:
                op = st.selectbox("Operator", options=["+", "-", "*", "/", "//", "%"], key="sales_comp_op")
            with c3:
                b_col = st.selectbox("Right column", options=cols, key="sales_comp_b")
            spec = {"mode":"binop_col", "name":new_name, "a":a_col, "op":op, "b":b_col}

        elif bmode == "Column ±×÷ Constant":
            c1, c2, c3 = st.columns([2,1,2])
            with c1:
                a_col = st.selectbox("Column", options=cols, key="sales_comp_col")
            with c2:
                op = st.selectbox("Operator", options=["+", "-", "*", "/", "//", "%"], key="sales_comp_op2")
            with c3:
                k = st.number_input("Constant", value=0.0, key="sales_comp_k")
            spec = {"mode":"binop_const", "name":new_name, "a":a_col, "op":op, "k":k}

        elif bmode == "IF condition… THEN … ELSE …":
            c1, c2, c3, c4 = st.columns([2,1,2,2])
            with c1:
                a_col = st.selectbox("Column to test", options=cols, key="sales_comp_if_col")
            with c2:
                cmp_op = st.selectbox("Compare", options=[">", ">=", "==", "!=", "<=", "<"], key="sales_comp_cmp")
            with c3:
                rhs_mode = st.selectbox("Compare against", options=["Constant", "Column"], key="sales_comp_rhs_mode")
            with c4:
                rhs_val = (
                    st.number_input("Constant", value=0.0, key="sales_comp_rhs_const")
                    if rhs_mode == "Constant" else
                    st.selectbox("Column", options=cols, key="sales_comp_rhs_col")
                )
            d1, d2 = st.columns(2)
            with d1:
                then_val = st.text_input("THEN value (use number; or column name in {curly})", value="1", key="sales_comp_then")
            with d2:
                else_val = st.text_input("ELSE value (number; or {column})", value="0", key="sales_comp_else")
            spec = {"mode":"ifelse", "name":new_name, "a":a_col, "cmp":cmp_op, "rhs_mode":rhs_mode, "rhs":rhs_val, "then":then_val, "else":else_val}

        else:  # Custom formula
            st.caption("Use arithmetic with column names. Examples: `qty * price`, `(amount - discount) / amount`, `round(net_amount, 2)`")
            formula = st.text_input("Formula", value="", key="sales_comp_formula")
            st.caption("Allowed functions: round, abs; Operators: + - * / // % ** (columns must exist).")
            spec = {"mode":"formula", "name":new_name, "expr":formula}

        add_btn = st.button("➕ Add computed column", type="primary", key="sales_comp_add")
        if add_btn:
            if not new_name.strip():
                st.error("Please give the new column a name.")
            else:
                st.session_state["sales_computed_specs"].append(spec)
                st.success(f"Added computed column: {new_name}")

        # Manage existing specs
        specs = st.session_state["sales_computed_specs"]
        if specs:
            st.markdown("**Current computed columns**")
            for i, s in enumerate(specs):
                cols_rm = st.columns([6,1])
                with cols_rm[0]:
                    st.write(f"- `{s['name']}` · **{s['mode']}** → {s}")
                with cols_rm[1]:
                    if st.button("🗑️", key=f"sales_comp_del_{i}"):
                        specs.pop(i)
                        st.experimental_rerun()
            if st.button("Clear all computed columns", key="sales_comp_clear"):
                st.session_state["sales_computed_specs"] = []
                st.experimental_rerun()

        # Apply computed columns to a working copy for preview/aggregation
        if work_df is None or work_df.empty:
            st.warning("No working dataset available for computed columns.")
            gdf_comp = None
        else:
            gdf_comp = work_df.copy()

        def _safe_to_number_or_series(val, df_):
            # number literal, {col} reference, or plain string -> try number
            if isinstance(val, (int, float, np.number)):
                return val
            txt = str(val).strip()
            if txt.startswith("{") and txt.endswith("}") and txt[1:-1] in df_.columns:
                return df_[txt[1:-1]]
            try:
                return float(txt)
            except Exception:
                return txt  # leave as-is (may be a string assignment)

        def _apply_spec(df_, s):
            m = s["mode"]
            name = s["name"]
            out = None
            try:
                if m == "binop_col":
                    A, B, op = df_[s["a"]], df_[s["b"]], s["op"]
                    if op == "+": out = A + B
                    elif op == "-": out = A - B
                    elif op == "*": out = A * B
                    elif op == "/": out = A / B.replace({0: np.nan}) if hasattr(B, "replace") else A / B
                    elif op == "//": out = A // B.replace({0: np.nan}) if hasattr(B, "replace") else A // B
                    elif op == "%": out = A % B
                elif m == "binop_const":
                    A, k, op = df_[s["a"]], s["k"], s["op"]
                    if op == "+": out = A + k
                    elif op == "-": out = A - k
                    elif op == "*": out = A * k
                    elif op == "/": out = A / (k if k != 0 else np.nan)
                    elif op == "//": out = A // (k if k != 0 else np.nan)
                    elif op == "%": out = A % k
                elif m == "ifelse":
                    A = df_[s["a"]]
                    rhs = df_[s["rhs"]] if s["rhs_mode"] == "Column" else s["rhs"]
                    cmp = s["cmp"]
                    if s["rhs_mode"] == "Constant":
                        rhs_series = rhs
                    else:
                        rhs_series = rhs
                    if cmp == ">": cond = A > rhs_series
                    elif cmp == ">=": cond = A >= rhs_series
                    elif cmp == "==": cond = A == rhs_series
                    elif cmp == "!=": cond = A != rhs_series
                    elif cmp == "<=": cond = A <= rhs_series
                    elif cmp == "<": cond = A < rhs_series
                    then_v = _safe_to_number_or_series(s["then"], df_)
                    else_v = _safe_to_number_or_series(s["else"], df_)
                    out = np.where(cond, then_v, else_v)
                else:  # formula
                    # Very limited safe eval: only round, abs, and existing columns
                    allowed_funcs = {"round": round, "abs": abs, "np": np}
                    # Build locals with column names
                    local_env = {c: df_[c] for c in df_.columns}
                    local_env.update(allowed_funcs)
                    expr = s["expr"]
                    out = pd.eval(expr, engine="python", local_dict=local_env)
                if out is None:
                    raise ValueError("Empty result")
                df_[name] = out
            except Exception as e:
                st.error(f"Failed to compute '{name}': {e}")
            return df_

        # Apply all specs in order
        for _s in st.session_state["sales_computed_specs"]:
            gdf_comp = _apply_spec(gdf_comp, _s)

        # Show a quick preview
        if st.checkbox("Show preview with computed columns", value=False, key="sales_comp_preview"):
            st_dataframe_safe(gdf_comp, n=150, use_container_width=True, height=320)

    # Use the computed-working df going forward
    work_df2 = gdf_comp if 'gdf_comp' in locals() and gdf_comp is not None else work_df
    if work_df2 is None or work_df2.empty:
        st.error("No working dataset available for aggregation.")
        cols2 = []
    else:
        cols2 = list(map(str, work_df2.columns))

    # ---------------- Multi-metric aggregation ----------------
    st.markdown("---")
    st.markdown('<div class="section-header" style="background: linear-gradient(135deg, #FFBD59 0%, #FFCF87 100%); color: #333333;">🧮 **Multi-Metric Aggregation & Analysis**</div>', unsafe_allow_html=True)

    # Define aggregation options first
    agg_options = ["Sum", "Average (mean)", "Count (values)", "Count distinct", "Min", "Max", "Median"]
    func_map = {
        "Sum": "sum",
        "Average (mean)": "mean",
        "Count (values)": "count",
        "Count distinct": "nunique",
        "Min": "min",
        "Max": "max",
        "Median": "median",
    }
    
    # Better organized UI with columns
    agg_col1, agg_col2 = st.columns([1, 1])
    
    with agg_col1:
        # 2) Choose group-by columns
        st.markdown("**📊 Group By Columns**")
        group_by_cols = st.multiselect(
            "Select columns to group by:",
            options=cols2,
            key="sales_tab_group_cols2",
            help="Select multiple columns to create hierarchical groupings"
        )
        
        # 3) Choose one or more sales fields + per-field aggregation
        if not cols2:
            st.warning("No columns available for selection.")
            sales_fields = []
        else:
            # Default guess: common names or first 1–3 numeric columns
            if work_df2 is not None and not work_df2.empty:
                num_cols2 = work_df2.select_dtypes(include="number").columns.tolist()
            else:
                num_cols2 = []
                
            common_sales_names = {
                "sales","sale","amount","revenue","qty","quantity","value",
                "price","mrp","net_amount","netamount","gross_amount","netvalue","invoice_amount","bill_amount"
            }
            default_measures = [c for c in cols2 if c.lower() in common_sales_names]
            if not default_measures:
                default_measures = num_cols2[:3] if num_cols2 else (cols2[:1] if cols2 else [])

            sales_fields = st.multiselect(
                "📈 Metrics (select one or many):",
                options=cols2,
                default=default_measures,
                key="sales_tab_measures"
            )
    
        with agg_col2:
            st.markdown("**⚙️ Aggregation Options**")
        
        # Per-field aggregation selectors in organized layout
        if sales_fields:
            st.markdown("**Per-field aggregation:**")
            per_field_funcs = {}
            for f in sales_fields:
                per_field_funcs[f] = st.selectbox(
                    f"Aggregation for **{f}**:",
                    options=agg_options,
                    index=0,
                    key=f"sales_tab_func_{f}"
                )
        else:
            st.info("Select metrics above to configure aggregation")
            per_field_funcs = {}

    if not sales_fields:
        st.info("Pick at least one metric to aggregate.")
        st.stop()

    cA, cB, cC = st.columns(3)
    with cA:
        add_rows = st.checkbox("Also add row count", value=False, key="sales_tab_rows")
    with cB:
        sort_desc = st.checkbox("Sort by metric (desc)", value=True, key="sales_tab_sort_desc2")
    with cC:
        topn = st.number_input("Top N (0 = all)", min_value=0, value=0, step=1, key="sales_tab_topn2")

    cD, cE = st.columns(2)
    with cD:
        add_pct = st.checkbox("Add % of total per metric", value=True, key="sales_tab_pct2")
    with cE:
        add_total = st.checkbox("Add grand total row", value=True, key="sales_tab_total2")

    run_multi = st.button("Compute aggregation", type="primary", key="sales_tab_run_multi")

    if run_multi:
        gdf = work_df2.copy()

        # Handle grouping key NA policy (consistent with earlier UI)
        keep_missing_keys = st.session_state.get("sales_tab_keep_missing_keys", True)
        if group_by_cols:
            if keep_missing_keys:
                for gc in group_by_cols:
                    if pd.api.types.is_string_dtype(gdf[gc]):
                        gdf[gc] = gdf[gc].astype("string").fillna("<NA>").str.strip()
                        gdf[gc] = gdf[gc].replace("", "<NA>")
                    else:
                        gdf[gc] = gdf[gc].fillna("<NA>")
            else:
                gdf = gdf.dropna(subset=group_by_cols, how="any")

        # Build agg map: one function per selected metric
        agg_map_per_field = {f: func_map[per_field_funcs[f]] for f in sales_fields}

        if not group_by_cols:
            # Scalar output: compute each metric independently
            data = {}
            for f, fn in agg_map_per_field.items():
                if fn == "nunique":
                    data[f"{fn}_{f}"] = [int(gdf[f].nunique(dropna=True))]
                else:
                    data[f"{fn}_{f}"] = [getattr(gdf[f], fn)()]
            if add_rows:
                data["rows"] = [int(len(gdf))]
            base_df = pd.DataFrame(data)
        else:
            grp = gdf.groupby(group_by_cols, dropna=not keep_missing_keys)
            base_df = grp.agg(agg_map_per_field).reset_index()

            # Flatten column names if needed (in case pandas returns MultiIndex)
            if isinstance(base_df.columns, pd.MultiIndex):
                base_df.columns = ["_".join([str(x) for x in tup if x != ""]) for tup in base_df.columns.to_flat_index()]

            # Optional row count
            if add_rows:
                sz = grp.size().reset_index(name="rows")
                base_df = base_df.merge(sz, on=group_by_cols, how="left")

        # Persist heavy result only once
        st.session_state["sales_group_base_df"] = base_df
        st.session_state["sales_group_by_cols"] = group_by_cols

        st.markdown('<div class="success-box">', unsafe_allow_html=True)
        st.success(f"✅ **Aggregation computed. Use options below to format, sort and save.")
        st.markdown("</div>", unsafe_allow_html=True)
        
    # ---- Lightweight formatting/display on previously computed result ----
    base_df = st.session_state.get("sales_group_base_df")
    by_cols = st.session_state.get("sales_group_by_cols", [])
    if isinstance(base_df, pd.DataFrame) and not base_df.empty:
        metric_cols_all = [c for c in base_df.columns if c not in by_cols]

        # Display controls that don't trigger heavy recompute
        st.markdown("---")
        st.markdown("**Format and view results**")
        sort_by = st.selectbox(
            "Sort by",
            options=metric_cols_all if metric_cols_all else base_df.columns,
            index=0,
            key="sales_tab_sort_by"
        )
        sort_desc = st.checkbox("Sort descending", value=True, key="sales_tab_sort_desc_display")
        topn = st.number_input("Top N (0 = all)", min_value=0, value=int(topn or 0), step=1, key="sales_tab_topn_display")
        add_pct = st.checkbox("Add % of total per metric", value=True, key="sales_tab_pct_display")
        add_total = st.checkbox("Add grand total row", value=True, key="sales_tab_total_display")

        # Build display df fast from base
        out_df = base_df.copy()
        # Ensure group-by columns are strings for safe display
        for gc in by_cols:
            if gc in out_df.columns:
                out_df[gc] = out_df[gc].astype("string").fillna("<NA>")

        metric_cols = [c for c in out_df.columns if c not in by_cols]
        for mc in metric_cols:
            out_df[mc] = pd.to_numeric(out_df[mc], errors="coerce")

        if add_pct and metric_cols:
            for mc in metric_cols:
                if pd.api.types.is_numeric_dtype(out_df[mc]):
                    total = float(out_df[mc].sum())
                    out_df[f"{mc}_pct"] = (out_df[mc] / total).fillna(0.0) if total != 0.0 else 0.0

        out_df = out_df.sort_values(sort_by, ascending=not sort_desc, kind="mergesort")
        if topn and int(topn) > 0:
            out_df = out_df.head(int(topn))

        if add_total and by_cols:
            total_row = {gc: "<TOTAL>" for gc in by_cols}
            for mc in metric_cols:
                if pd.api.types.is_numeric_dtype(out_df[mc]):
                    total_row[mc] = out_df[mc].sum()
            for mc in [c for c in out_df.columns if c.endswith("_pct")]:
                total_row[mc] = 1.0
            out_df = pd.concat([out_df, pd.DataFrame([total_row])], ignore_index=True)

        # Display results in organized tabs
        result_tab1, result_tab2, result_tab3 = st.tabs([
            "📊 **Data Table**", 
            "📈 **Interactive Charts**", 
            "💾 **Download & Save**"
        ])

        with result_tab1:
            st_dataframe_safe(out_df, n=300, use_container_width=True, height=480)

        with result_tab2:
            st.markdown("### 📈 Interactive Charts")
            if not out_df.empty and len(out_df) > 1:
                # Chart options
                chart_col1, chart_col2 = st.columns(2)
                with chart_col1:
                    chart_type = st.selectbox(
                        "Chart type:",
                        options=["Bar Chart", "Line Chart", "Scatter Plot", "Pie Chart"],
                        index=0
                    )
                with chart_col2:
                    if group_by_cols:
                        x_axis = st.selectbox("X-axis:", options=group_by_cols, index=0)
                    else:
                        x_axis = "index"
                
                # Y-axis selection (metrics)
                if metric_cols:
                    y_axis = st.selectbox("Y-axis (metric):", options=metric_cols, index=0)
                    
                    # Create charts based on selection
                    if chart_type == "Bar Chart":
                        if group_by_cols:
                            # Grouped bar chart
                            fig = px.bar(
                                out_df.head(50),  # Limit to top 50 for readability
                                x=x_axis,
                                y=y_axis,
                                title=f"{y_axis} by {x_axis}",
                                color_discrete_sequence=['#1f77b4']
                            )
                        else:
                            # Simple bar chart
                            fig = px.bar(
                                out_df,
                                x=out_df.index,
                                y=y_axis,
                                title=f"{y_axis} Overview"
                            )
                        fig.update_layout(height=500)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    elif chart_type == "Line Chart":
                        if group_by_cols:
                            fig = px.line(
                                out_df.head(50),
                                x=x_axis,
                                y=y_axis,
                                title=f"{y_axis} Trend by {x_axis}"
                            )
                        else:
                            fig = px.line(
                                out_df,
                                y=y_axis,
                                title=f"{y_axis} Trend"
                            )
                        fig.update_layout(height=500)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    elif chart_type == "Scatter Plot":
                        if len(group_by_cols) >= 2:
                            fig = px.scatter(
                                out_df.head(100),
                                x=group_by_cols[0],
                                y=y_axis,
                                size=y_axis,
                                color=group_by_cols[1] if len(group_by_cols) > 1 else None,
                                title=f"{y_axis} vs {group_by_cols[0]}"
                            )
                        else:
                            fig = px.scatter(
                                out_df.head(100),
                                x=out_df.index,
                                y=y_axis,
                                title=f"{y_axis} Distribution"
                            )
                        fig.update_layout(height=500)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    elif chart_type == "Pie Chart":
                        if group_by_cols:
                            # Top 10 values for pie chart
                            top_data = out_df.nlargest(10, y_axis)
                            fig = px.pie(
                                top_data,
                                values=y_axis,
                                names=x_axis,
                                title=f"Top 10 {y_axis} by {x_axis}"
                            )
                            fig.update_layout(height=500)
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.info("Pie charts work best with grouped data")
                
                # Summary statistics
                if metric_cols:
                    st.markdown("### 📊 Summary Statistics")
                    summary_cols = st.columns(len(metric_cols))
                    for i, metric in enumerate(metric_cols):
                        with summary_cols[i]:
                            if pd.api.types.is_numeric_dtype(out_df[metric]):
                                st.metric(
                                    f"Total {metric}",
                                    f"{out_df[metric].sum():,.2f}"
                                )
                                st.caption(f"Avg: {out_df[metric].mean():,.2f}")
                            else:
                                st.metric(
                                    f"Total {metric}",
                                    f"{out_df[metric].count():,}"
                                )
            else:
                st.info("Need at least 2 rows of data to create meaningful charts")
        
        with result_tab3:
            st.markdown("### 💾 Download & Save Options")
            csv_bytes = out_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download aggregation CSV",
                data=csv_bytes,
                file_name=f"sales_aggregation_{sales_src.lower()}.csv",
                mime="text/csv",
                key="dl_sales_tab_csv2"
            )

            with st.expander("Save aggregation to DB (optional)"):
                default_name = f"sales_agg_{sales_src.lower()}"
                agg_tbl_name = st.text_input("Table name", value=default_name, key="save_sales_tab_tbl2")
                save_btn = st.button("Save aggregation to SQLite", key="save_sales_tab_btn2")
                if save_btn:
                    try:
                        tbl = _sanitize_table_name(agg_tbl_name)
                        with _connect_db() as conn:
                            write_df_sqlite(out_df, conn, tbl, overwrite=True, chunk_size=100_000)
                        st.success(f"Saved aggregated table to '{tbl}' in {DB_PATH}")
                        st.cache_data.clear()
                    except Exception as e:
                        st.error(f"Save failed: {e}")
