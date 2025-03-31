import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import datetime

# --------------------------------------------------------
# Theme Configuration (From previous themed version)
# --------------------------------------------------------
PUBLIX_GREEN_DARK = "#00543D"
PUBLIX_GREEN_BRIGHT = "#5F8F38"
BACKGROUND_COLOR = "#F9F9F9"
CARD_BACKGROUND_COLOR = "#FFFFFF"
TEXT_COLOR_DARK = "#333333"
TEXT_COLOR_MEDIUM = "#666666"
TEXT_COLOR_LIGHT = "#FFFFFF"
BORDER_COLOR = "#EAEAEA"
CAT_COLORS = { # Using original categories
    "Star Performer": PUBLIX_GREEN_BRIGHT,
    "Needs Stabilization": "#FFA726", # Orange warning
    "Improving": "#42A5F5", # Blue for improving
    "Requires Intervention": "#EF5350", # Red for low
    "default": TEXT_COLOR_MEDIUM
}

# --------------------------------------------------------
# Helper Functions (Using ORIGINAL script's robust versions)
# --------------------------------------------------------

@st.cache_data
def load_data(uploaded_file):
    # --- (Load Data function from ORIGINAL script - clubpubdash_v22.py) ---
    """
    Reads CSV/XLSX file into a pandas DataFrame, standardizes key columns,
    and sorts by week/store. Expects columns:
      - Store # (or store ID)
      - Week or Date
      - Engaged Transaction %
      - Optional: Weekly Rank, Quarter to Date %, etc.
    """
    if uploaded_file is None: return pd.DataFrame() # Return empty if no file
    filename = uploaded_file.name.lower()
    try:
        if filename.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif filename.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(uploaded_file)
        else:
            st.error("Unsupported file type. Please upload CSV or Excel.")
            return pd.DataFrame()
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return pd.DataFrame()

    df.columns = standardize_columns(df.columns) # Use original standardize function

    # --- Data Cleaning (from original) ---
    required_cols = ['Store #', 'Engaged Transaction %']
    if not ('Date' in df.columns or 'Week' in df.columns):
        st.error("Data must contain either a 'Week' or 'Date' column.")
        return pd.DataFrame()
    if not all(col in df.columns for col in required_cols):
        st.error(f"Data must contain columns: {', '.join(required_cols)}")
        return pd.DataFrame()

    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        # Drop rows where Date could not be parsed (original script didn't explicitly do this, but good practice)
        df = df.dropna(subset=['Date'])
        # Derive Week from Date if Week column doesn't exist (original logic)
        if 'Week' not in df.columns and pd.api.types.is_datetime64_any_dtype(df['Date']):
             df['Week'] = df['Date'].dt.isocalendar().week.astype(int)

    percent_cols = ['Engaged Transaction %', 'Quarter to Date %']
    for col in percent_cols:
        if col in df.columns:
            # Convert to string first to handle mixed types, remove %, then convert to numeric
            df[col] = pd.to_numeric(df[col].astype(str).str.replace('%', '', regex=False), errors='coerce')

    # Drop rows with missing essential data (added check for Week/Date presence)
    essential_cols = ['Store #', 'Engaged Transaction %']
    if 'Week' in df.columns: essential_cols.append('Week')
    elif 'Date' in df.columns: essential_cols.append('Date')
    df = df.dropna(subset=essential_cols)


    if 'Weekly Rank' in df.columns:
        # Handle potential errors during conversion and ensure integer type
        df['Weekly Rank'] = pd.to_numeric(df['Weekly Rank'], errors='coerce')
        # Only convert to Int64 if the column exists and isn't all NaN after coercion
        if 'Weekly Rank' in df.columns and not df['Weekly Rank'].isna().all():
             df['Weekly Rank'] = df['Weekly Rank'].astype('Int64')

    if 'Store #' in df.columns:
        df['Store #'] = df['Store #'].astype(str)

    # Ensure Week is integer if present and sort (original logic)
    if 'Week' in df.columns:
        # Ensure 'Week' is numeric before converting to int
        df['Week'] = pd.to_numeric(df['Week'], errors='coerce')
        df = df.dropna(subset=['Week']) # Remove rows where Week is not numeric
        df['Week'] = df['Week'].astype(int)
        df = df.sort_values(['Week', 'Store #'])
        # Derive Quarter from Week if needed (original logic)
        if 'Quarter' not in df.columns:
             df['Quarter'] = ((df['Week'] - 1) // 13 + 1).astype(int)
    elif 'Date' in df.columns:
         df = df.sort_values(['Date', 'Store #'])
         # Derive Quarter from Date if needed (original logic)
         if 'Quarter' not in df.columns and pd.api.types.is_datetime64_any_dtype(df['Date']):
              df['Quarter'] = df['Date'].dt.quarter

    return df


def standardize_columns(columns):
    # --- (Standardize Columns function from ORIGINAL script - clubpubdash_v22.py) ---
    """
    Renames columns to standard internal names for consistency. More robust checking.
    """
    new_cols = []
    processed_indices = set()
    cols_lower = [str(col).strip().lower() for col in columns]

    for i, col_lower in enumerate(cols_lower):
        if i in processed_indices: continue
        original_col = columns[i] # Keep original case for fallback

        # Prioritize specific keywords for accuracy
        if col_lower == 'store #': new_cols.append('Store #')
        elif 'store' in col_lower and ('#' in col_lower or 'id' in col_lower or 'number' in col_lower): new_cols.append('Store #')
        elif 'engaged transaction %' in col_lower: new_cols.append('Engaged Transaction %')
        elif 'engage' in col_lower and ('%' in col_lower or 'transaction' in col_lower): new_cols.append('Engaged Transaction %')
        elif ('week' in col_lower and 'ending' in col_lower) or col_lower == 'date' or cl == 'week ending': new_cols.append('Date') # Match original logic closer
        elif col_lower == 'week' or (col_lower.startswith('week') and not any(s in col_lower for s in ['rank', 'end', 'date'])): new_cols.append('Week')
        elif 'rank' in col_lower and 'week' in col_lower: new_cols.append('Weekly Rank')
        elif 'quarter' in col_lower or 'qtd' in col_lower: new_cols.append('Quarter to Date %')
        else: new_cols.append(original_col) # Keep original if no specific match
        processed_indices.add(i)

    # Simple duplicate check (can be enhanced)
    if len(new_cols) != len(set(new_cols)):
         st.warning("Potential duplicate columns detected after standardization. Check input file headers.")
    return new_cols


def calculate_trend(group, window=4, engagement_col='Engaged Transaction %', week_col='Week'):
    # --- (Calculate Trend function from ORIGINAL script - clubpubdash_v22.py) ---
    """
    Calculates a trend label (Upward, Downward, etc.) based on a simple
    linear slope of the last `window` data points in 'Engaged Transaction %'.
    Uses centered X for potentially better numerical stability.
    """
    if engagement_col not in group.columns or week_col not in group.columns: return "Stable" # Default to Stable if columns missing
    # Drop NaNs in the engagement column for the specific group before checking length
    group = group.dropna(subset=[engagement_col])
    if len(group) < 2: return "Stable" # Not enough data points

    sorted_data = group.sort_values(week_col, ascending=True).tail(window)
    if len(sorted_data) < 2: return "Insufficient Data" # Not enough data in window

    x = sorted_data[week_col].values
    y = sorted_data[engagement_col].values

    # Check for NaN again after slicing (should be redundant due to earlier dropna, but safe)
    valid_indices = ~np.isnan(y)
    x = x[valid_indices]
    y = y[valid_indices]
    if len(x) < 2: return "Insufficient Data" # Not enough valid points in window

    # Center X to avoid numeric issues (from original script)
    x_mean = np.mean(x)
    x_centered = x - x_mean

    # Check for variance
    if np.sum(x_centered**2) == 0 or np.std(y) == 0 :
        return "Stable"

    # Calculate slope using simplified formula for centered X
    try:
        slope = np.sum(x_centered * (y - np.mean(y))) / np.sum(x_centered**2)
    except (np.linalg.LinAlgError, ValueError, FloatingPointError): # Catch potential errors
        return "Calculation Error" # Or Stable? Defaulting to Calculation Error

    # Thresholds from original script
    strong_threshold = 0.5
    mild_threshold = 0.1

    if slope > strong_threshold: return "Strong Upward"
    elif slope > mild_threshold: return "Upward"
    elif slope < -strong_threshold: return "Strong Downward"
    elif slope < -mild_threshold: return "Downward"
    else: return "Stable"


def find_anomalies(df, z_threshold=2.0, engagement_col='Engaged Transaction %', store_col='Store #', week_col='Week'):
    # --- (Find Anomalies function from ORIGINAL script - clubpubdash_v22.py) ---
    """
    Calculates week-over-week changes in Engaged Transaction % for each store
    and flags any changes whose Z-score exceeds the given threshold.
    Returns a DataFrame of anomalies with potential explanations including rank change.
    """
    if df.empty or not all(col in df.columns for col in [engagement_col, store_col, week_col]):
         return pd.DataFrame() # Return empty if essential columns missing

    anomalies_list = []
    # Ensure data is sorted correctly for diff calculation
    df_sorted = df.sort_values([store_col, week_col])

    for store_id, grp in df_sorted.groupby(store_col):
        # Ensure group has numeric engagement and week data, dropna before diff
        grp = grp.dropna(subset=[engagement_col, week_col])
        grp = grp.copy() # Avoid SettingWithCopyWarning
        # Use assign to prevent potential warnings with chained indexing
        grp['diffs'] = grp[engagement_col].diff()
        diffs = grp['diffs'].dropna() # Calculate differences and remove the first NaN

        if len(diffs) < 2: # Need at least 2 differences to calculate std dev reliably
            continue

        mean_diff = diffs.mean()
        std_diff = diffs.std(ddof=0) # Use population std dev

        # Skip if standard deviation is zero or NaN
        if std_diff == 0 or pd.isna(std_diff):
            continue

        # Iterate through the *group* where diff is not NaN
        anomaly_candidates = grp[grp['diffs'].notna()]
        for idx, row in anomaly_candidates.iterrows():
             diff_val = row['diffs']
             z = (diff_val - mean_diff) / std_diff

             if abs(z) >= z_threshold:
                 # Get previous row safely using index
                 try:
                     # Find the positional index of the current row within the group
                     current_pos_index = grp.index.get_loc(idx)
                     # Check if there is a previous row
                     if current_pos_index > 0:
                          prev_idx = grp.index[current_pos_index - 1]
                          prev_row = grp.loc[prev_idx]
                     else: # First row in group, no previous row exists
                          prev_row = pd.Series(dtype='object') # Empty series
                 except KeyError: # Handle cases where index might not be found (shouldn't happen often)
                      prev_row = pd.Series(dtype='object')

                 # Safely get values from current and previous rows
                 week_cur = row.get(week_col)
                 week_prev = prev_row.get(week_col)
                 val_cur = row.get(engagement_col)
                 rank_cur = row.get('Weekly Rank') # Optional
                 rank_prev = prev_row.get('Weekly Rank') # Optional

                 # Build explanation (from original script)
                 explanation = ""
                 if diff_val >= 0:
                     explanation = "Engagement spiked significantly."
                     if pd.notna(rank_prev) and pd.notna(rank_cur) and rank_cur < rank_prev:
                          explanation += f" Rank improved from {int(rank_prev)} to {int(rank_cur)}."
                     elif pd.notna(prev_row.get(engagement_col)):
                          explanation += f" Jumped from {prev_row.get(engagement_col):.2f}%."
                 else:
                     explanation = "Sharp drop in engagement."
                     if pd.notna(rank_prev) and pd.notna(rank_cur) and rank_cur > rank_prev:
                          explanation += f" Rank dropped from {int(rank_prev)} to {int(rank_cur)}."
                     elif pd.notna(prev_row.get(engagement_col)):
                          explanation += f" Dropped from {prev_row.get(engagement_col):.2f}%."


                 anomalies_list.append({
                     'Store #': store_id,
                     'Week': int(week_cur) if pd.notna(week_cur) else None,
                     'Engaged Transaction %': round(val_cur, 2) if pd.notna(val_cur) else None,
                     'Change %pts': round(diff_val, 2) if pd.notna(diff_val) else None,
                     'Z-score': round(z, 2),
                     'Prev Week': int(week_prev) if pd.notna(week_prev) else None,
                     'Rank': int(rank_cur) if pd.notna(rank_cur) else None, # Convert to int if not NaN
                     'Prev Rank': int(rank_prev) if pd.notna(rank_prev) else None, # Convert to int if not NaN
                     'Possible Explanation': explanation
                 })

    if not anomalies_list: return pd.DataFrame()
    anomalies_df = pd.DataFrame(anomalies_list)
    anomalies_df['Abs Z'] = anomalies_df['Z-score'].abs()
    anomalies_df = anomalies_df.sort_values('Abs Z', ascending=False).drop(columns=['Abs Z'])
    return anomalies_df


# --------------------------------------------------------
# Altair Chart Theming (Using publix_clean_theme)
# --------------------------------------------------------
def publix_clean_theme():
    # --- (Theme definition - same as previous version) ---
    return {
        "config": {
            "background": CARD_BACKGROUND_COLOR,
            "title": { "anchor": "start", "color": TEXT_COLOR_DARK, "fontSize": 14, "fontWeight": "bold"},
            "axis": {
                "domainColor": BORDER_COLOR,"gridColor": BORDER_COLOR,"gridDash": [1, 3],"labelColor": TEXT_COLOR_MEDIUM,
                "labelFontSize": 10,"titleColor": TEXT_COLOR_MEDIUM,"titleFontSize": 11,"titlePadding": 10,
                "tickColor": BORDER_COLOR,"tickSize": 5,
            },
            "axisX": {"grid": False, "labelAngle": 0},
            "axisY": {"grid": True, "labelPadding": 5, "ticks": False},
            "header": {"labelFontSize": 11, "titleFontSize": 12, "labelColor": TEXT_COLOR_MEDIUM, "titleColor": TEXT_COLOR_DARK},
            "legend": None, # Keep legends off by default for cleanliness
            "range": { "category": [PUBLIX_GREEN_BRIGHT, PUBLIX_GREEN_DARK, TEXT_COLOR_MEDIUM],"heatmap": "greens","ramp": "greens", },
            "view": {"stroke": None},
            "line": {"stroke": PUBLIX_GREEN_BRIGHT, "strokeWidth": 2.5},
            "point": {"fill": PUBLIX_GREEN_BRIGHT, "size": 10, "stroke": CARD_BACKGROUND_COLOR, "strokeWidth": 0.5},
            "bar": {"fill": PUBLIX_GREEN_BRIGHT},
            "rule": {"stroke": TEXT_COLOR_MEDIUM, "strokeDash": [4, 4]},
            "text": {"color": TEXT_COLOR_DARK, "fontSize": 11},
            "area": {"fill": PUBLIX_GREEN_BRIGHT, "opacity": 0.15, "line": {"stroke": PUBLIX_GREEN_BRIGHT, "strokeWidth": 2}}
        }
    }
alt.themes.register("publix_clean", publix_clean_theme)
alt.themes.enable("publix_clean")

# --------------------------------------------------------
# Chart Helper Functions (From themed version, with type hints)
# --------------------------------------------------------
def create_sparkline(data, y_col, y_title):
    # --- (Sparkline function - same as previous version with type hints) ---
    if data is None or data.empty or y_col not in data.columns or 'Week' not in data.columns: return None
    if not isinstance(data, pd.DataFrame):
         try: data = pd.DataFrame(data)
         except Exception: return None
    line = alt.Chart(data).mark_line().encode(
        x=alt.X('Week:O', axis=None), y=alt.Y(f'{y_col}:Q', axis=None),
        tooltip=[alt.Tooltip('Week:O'), alt.Tooltip(f'{y_col}:Q', format='.2f', title=y_title)]
    ).properties(width=80, height=30)
    area = line.mark_area()
    return area + line

def create_donut_chart(value, title="Engagement"):
    # --- (Donut chart function - same as previous version with type hints) ---
     value = max(0, min(100, value))
     source = pd.DataFrame({"category": [title, "Remaining"], "value": [value, 100 - value]})
     base = alt.Chart(source).encode(theta=alt.Theta("value:Q", stack=True))
     pie = base.mark_arc(outerRadius=60, innerRadius=45).encode(
         color=alt.Color("category:N", scale=alt.Scale(domain=[title, "Remaining"], range=[PUBLIX_GREEN_BRIGHT, BORDER_COLOR]), legend=None),
         order=alt.Order("value:Q", sort="descending")
     )
     text = base.mark_text(radius=0, align='center', baseline='middle', fontSize=18, fontWeight='bold').encode(
         text=alt.condition(alt.datum.category == title, alt.Text("value:Q", format=".0f"), alt.value("")),
         order=alt.Order("value:Q", sort="descending"), color=alt.value(TEXT_COLOR_DARK)
     )
     donut_chart = (pie + text).properties(height=120, width=120)
     return donut_chart

# --------------------------------------------------------
# Streamlit Page Config & Main Layout
# --------------------------------------------------------
st.set_page_config(
    page_title="Publix Engagement Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS Injection (Same as themed version) ---
st.markdown(f"""<style>
    body {{ font-family: sans-serif; }} .stApp {{ background-color: {BACKGROUND_COLOR}; }}
    [data-testid="stSidebar"] > div:first-child {{ background-color: {PUBLIX_GREEN_DARK}; padding-top: 1rem; }}
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] label, [data-testid="stSidebar"] .st-caption {{ color: {TEXT_COLOR_LIGHT} !important; font-weight: normal; }}
    [data-testid="stSidebar"] .st-emotion-cache-1bzkvze {{ color: {TEXT_COLOR_LIGHT} !important; }} /* File uploader label */
    [data-testid="stSidebar"] small {{ color: rgba(255, 255, 255, 0.7) !important; }} /* File uploader secondary text */
    [data-testid="stSidebar"] .stSelectbox > div, [data-testid="stSidebar"] .stMultiselect > div {{ background-color: rgba(255, 255, 255, 0.1); border: none; border-radius: 4px; }}
    [data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] > div, [data-testid="stSidebar"] .stMultiselect div[data-baseweb="select"] > div {{ color: {TEXT_COLOR_LIGHT}; }}
    [data-testid="stSidebar"] .stSelectbox svg, [data-testid="stSidebar"] .stMultiselect svg {{ fill: {TEXT_COLOR_LIGHT}; }}
    [data-testid="stSidebar"] .stMultiselect span[data-baseweb="tag"] svg {{ fill: {TEXT_COLOR_LIGHT}; }}
    [data-testid="stSidebar"] .stButton>button {{ background-color: transparent; color: {TEXT_COLOR_LIGHT}; border: 1px solid rgba(255, 255, 255, 0.5); font-weight: normal; padding: 5px 10px; margin-top: 0.5rem; }}
    [data-testid="stSidebar"] .stButton>button:hover {{ background-color: rgba(255, 255, 255, 0.1); border-color: {TEXT_COLOR_LIGHT}; }}
    [data-testid="stSidebar"] .stFileUploader button {{ border: 1px dashed rgba(255, 255, 255, 0.5); background-color: rgba(255, 255, 255, 0.05); color: {TEXT_COLOR_LIGHT}; }}
    [data-testid="stSidebar"] .stFileUploader button:hover {{ border-color: {TEXT_COLOR_LIGHT}; background-color: rgba(255, 255, 255, 0.1); }}
    .main .block-container {{ padding: 1.5rem 2rem 2rem 2rem; max-width: 95%; }}
    .metric-card {{ background-color: {CARD_BACKGROUND_COLOR}; border: 1px solid {BORDER_COLOR}; border-radius: 8px; padding: 1rem 1.2rem; box-shadow: 0 1px 3px rgba(0,0,0,0.03); height: 100%; display: flex; flex-direction: column; justify-content: space-between; }}
    .chart-card {{ background-color: {CARD_BACKGROUND_COLOR}; border: 1px solid {BORDER_COLOR}; border-radius: 8px; padding: 1.2rem; box-shadow: 0 1px 3px rgba(0,0,0,0.03); }}
    .stMetric {{ background-color: transparent !important; border: none !important; padding: 0 !important; text-align: left; color: {TEXT_COLOR_DARK}; font-size: 1rem; height: 100%; display: flex; flex-direction: column; }}
    .stMetric > label {{ color: {TEXT_COLOR_MEDIUM} !important; font-weight: normal !important; font-size: 0.85em !important; margin-bottom: 0.25rem; order: 1; }}
    .stMetric > div:nth-of-type(1) {{ color: {TEXT_COLOR_DARK} !important; font-size: 2.0em !important; font-weight: bold !important; line-height: 1.1 !important; margin-bottom: 0.25rem; order: 0; }}
    .stMetric > div:nth-of-type(2) {{ font-size: 0.85em !important; font-weight: bold; color: {TEXT_COLOR_MEDIUM} !important; order: 2; margin-top: auto; }}
    .stMetric .stMetricDelta {{ padding-top: 5px; }}
    .stMetric .stMetricDelta span[style*="color: rgb(46, 125, 50)"] {{ color: {PUBLIX_GREEN_BRIGHT} !important; }}
    .stMetric .stMetricDelta span[style*="color: rgb(198, 40, 40)"] {{ color: #D32F2F !important; }}
    .stMetric .metric-sparkline {{ margin-top: auto; order: 3; padding-top: 10px; line-height: 0; opacity: 0.7; }}
    div[data-baseweb="tab-list"] {{ border-bottom: 2px solid {BORDER_COLOR}; padding-left: 0; margin-bottom: 1.5rem; }}
    button[data-baseweb="tab"] {{ background-color: transparent; color: {TEXT_COLOR_MEDIUM}; padding: 0.6rem 0.1rem; margin-right: 1.5rem; border-bottom: 2px solid transparent; font-weight: normal; }}
    button[data-baseweb="tab"]:hover {{ background-color: transparent; color: {TEXT_COLOR_DARK}; border-bottom-color: {BORDER_COLOR}; }}
    button[data-baseweb="tab"][aria-selected="true"] {{ color: {PUBLIX_GREEN_BRIGHT}; font-weight: bold; border-bottom-color: {PUBLIX_GREEN_BRIGHT}; }}
    h1, h2, h3 {{ color: {TEXT_COLOR_DARK}; font-weight: bold; }}
    h3 {{ margin-top: 1.5rem; margin-bottom: 0.8rem; font-size: 1.3rem; }}
    h4 {{ margin-top: 1rem; margin-bottom: 0.5rem; font-size: 1.1rem; color: {TEXT_COLOR_MEDIUM}; font-weight: bold; }}
    [data-testid="stExpander"] {{ border: 1px solid {BORDER_COLOR}; border-radius: 6px; background-color: {CARD_BACKGROUND_COLOR}; }}
    [data-testid="stExpander"] summary {{ font-weight: normal; color: {TEXT_COLOR_MEDIUM}; }}
    [data-testid="stExpander"] summary:hover {{ color: {PUBLIX_GREEN_BRIGHT}; }}
    hr {{ border-top: 1px solid {BORDER_COLOR}; margin: 1.5rem 0; }}
</style>""", unsafe_allow_html=True)

# --------------------------------------------------------
# Sidebar Setup (Using themed version's setup)
# --------------------------------------------------------
st.sidebar.markdown(f"""<div style="text-align: center; padding-bottom: 1rem;">
<svg width="40" height="40" viewBox="0 0 100 100" fill="none" xmlns="http://www.w3.org/2000/svg">
<circle cx="50" cy="50" r="50" fill="{TEXT_COLOR_LIGHT}"/>
<path d="M63.7148 25H42.4297C35.1445 25 29.2852 30.8594 29.2852 38.1445V43.6055C29.2852 46.6523 31.7734 49.1406 34.8203 49.1406H46.0117V61.8555C46.0117 64.9023 48.4999 67.3906 51.5468 67.3906H57.1406C60.1875 67.3906 62.6757 64.9023 62.6757 61.8555V49.1406H66.4804C68.0234 49.1406 69.2851 47.8789 69.2851 46.3359V31.0195C69.2851 27.7148 66.7304 25 63.7148 25ZM57.1406 55.6641H51.5468V43.6055C51.5468 40.5586 49.0585 38.0703 46.0117 38.0703H40.4179C39.1992 38.0703 38.2226 39.0469 38.2226 40.1914V43.6055C38.2226 44.75 39.1992 45.7266 40.4179 45.7266H51.5468C54.5937 45.7266 57.1406 48.2148 57.1406 51.2617V55.6641Z" fill="{PUBLIX_GREEN_DARK}"/>
</svg></div>""", unsafe_allow_html=True)

st.sidebar.header("Data Upload")
data_file = st.sidebar.file_uploader("Engagement Data", type=['csv', 'xlsx', 'xls'], key="primary_upload", label_visibility="collapsed")

# Load data using the robust function from the original script
df = load_data(data_file)

if df.empty:
    st.info("⬆️ Please upload an engagement data file using the sidebar to begin.")
    st.stop()

# --- Sidebar Filters (Using original script's robust logic for deriving options) ---
st.sidebar.header("Filters")
quarter_choice = "All"
if 'Quarter' in df.columns and df['Quarter'].notna().any():
    # Safely get unique quarters, handle potential NaN/None after coercion
    quarters = pd.to_numeric(df['Quarter'], errors='coerce').dropna().unique()
    quarters = sorted([int(q) for q in quarters]) # Convert to int after dropping NaN
    quarter_options = ["All"] + [f"Q{q}" for q in quarters]
    quarter_choice = st.sidebar.selectbox("Quarter", quarter_options, index=0, key="quarter_select")
else:
    st.sidebar.caption("Quarter data not found.")

week_choice = "All"
if 'Week' in df.columns and df['Week'].notna().any():
    # Filter available weeks based on selected quarter *before* creating options
    if quarter_choice != "All":
        try:
            q_num_filter = int(quarter_choice[1:])
            available_weeks = df.loc[df['Quarter'] == q_num_filter, 'Week'].dropna().unique()
        except (ValueError, IndexError, TypeError): # Catch potential errors if Quarter column is weird
             available_weeks = df['Week'].dropna().unique() # Fallback
    else:
        available_weeks = df['Week'].dropna().unique()

    available_weeks = sorted([int(w) for w in available_weeks]) # Ensure integer weeks
    if available_weeks:
        week_options = ["All"] + [str(w) for w in available_weeks]
        week_choice = st.sidebar.selectbox("Week", week_options, index=0, key="week_select")
    else:
         week_choice = "All" # Default if no weeks available for selection
else:
     st.sidebar.caption("Week data not found.")


store_list = []
if 'Store #' in df.columns:
     store_list = sorted(df['Store #'].dropna().unique().tolist())
store_choice = st.sidebar.multiselect("Store(s)", store_list, default=[], key="store_select") if store_list else []

# --- Advanced Settings ---
# Using an expander for less clutter, add back anomaly threshold
st.sidebar.markdown("---")
with st.sidebar.expander("Analysis Settings", expanded=False):
    trend_analysis_weeks = st.slider("Trend Window (Weeks)", 3, 8, 4, key="trend_window", help="Weeks for recent trend calculation.")
    z_threshold = st.slider("Anomaly Sensitivity (Z)", 1.0, 3.0, 2.0, 0.1, key="anomaly_z", help="Lower = more sensitive.")
    # show_ma option can be added back if moving averages are reintegrated fully


# --- Filter DataFrame (Consistent application) ---
df_filtered = df.copy()
q_num = None
if quarter_choice != "All":
    try:
        q_num = int(quarter_choice[1:])
        df_filtered = df_filtered[df_filtered['Quarter'] == q_num]
    except (ValueError, IndexError, TypeError): pass # Ignore filter if Quarter is invalid

week_num = None
if week_choice != "All":
    try:
        week_num = int(week_choice)
        df_filtered = df_filtered[df_filtered['Week'] == week_num]
    except ValueError: pass # Ignore filter if Week is invalid

if store_choice:
    df_filtered = df_filtered[df_filtered['Store #'].isin(store_choice)]

if df_filtered.empty:
    st.warning("No data matches the selected filters.")
    st.stop()

# --- Pre-calculate stats for filtered data (using original script's approach) ---
# Determine current and previous week based on *original data* before week filtering
# to ensure delta calculation is meaningful even when a single week is selected.
current_week_actual = df['Week'].max() if 'Week' in df.columns else None
prev_week_actual = None
if current_week_actual:
    prev_weeks_df = df.loc[df['Week'] < current_week_actual, 'Week']
    if not prev_weeks_df.empty:
         prev_week_actual = prev_weeks_df.max()

# Calculate metrics based on the *filtered data* for the DISPLAY PERIOD
# Use week_num if a specific week is selected, otherwise use the max week in the filtered set
display_week = week_num if week_num is not None else df_filtered['Week'].max() if 'Week' in df_filtered.columns and not df_filtered.empty else None
display_prev_week = None
if display_week is not None:
     possible_prev_display_weeks = df_filtered.loc[df_filtered['Week'] < display_week, 'Week'].dropna()
     if not possible_prev_display_weeks.empty:
          display_prev_week = possible_prev_display_weeks.max()

# Averages for display week and previous display week
latest_avg = df_filtered.loc[df_filtered['Week'] == display_week, 'Engaged Transaction %'].mean() if display_week is not None else None
prev_avg = df_filtered.loc[df_filtered['Week'] == display_prev_week, 'Engaged Transaction %'].mean() if display_prev_week is not None else None
delta_val = latest_avg - prev_avg if pd.notna(latest_avg) and pd.notna(prev_avg) else None

# Top/Bottom performer for the display week/period
if display_week is not None:
    perf_data = df_filtered[df_filtered['Week'] == display_week]
else: # Use overall filtered data if no specific week
     perf_data = df_filtered
store_perf_display = perf_data.groupby('Store #')['Engaged Transaction %'].mean()
top_store, bottom_store, top_val, bottom_val = None, None, None, None
if not store_perf_display.empty:
    top_store = store_perf_display.idxmax()
    bottom_store = store_perf_display.idxmin()
    top_val = store_perf_display.max()
    bottom_val = store_perf_display.min()

# Calculate overall average for the filtered period (used if no specific week)
overall_avg_engagement_filtered = df_filtered['Engaged Transaction %'].mean()

# --------------------------------------------------------
# Main Content Area - Tabs (Using themed structure)
# --------------------------------------------------------
tab_overview, tab_comparison, tab_categories, tab_anomalies = st.tabs([
    "Overview & Trends", "Store Comparison", "Performance Categories", "Anomalies"
])

# --- TAB 1: Overview & Trends ---
with tab_overview:
    st.markdown("### Key Metrics")
    col1, col2, col3 = st.columns(3)

    # Metric 1: Average Engagement
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        metric_label = "Avg Engagement"
        if store_choice: metric_label = f"Avg ({len(store_choice)} Stores)" if len(store_choice)>1 else f"Store {store_choice[0]}"
        metric_suffix = f"(Wk {int(display_week)})" if display_week is not None else "(Period Avg)"
        display_avg_val = latest_avg if display_week is not None else overall_avg_engagement_filtered # Use appropriate average

        st.metric(
            f"{metric_label} {metric_suffix}",
            f"{display_avg_val:.1f}%" if pd.notna(display_avg_val) else "N/A",
            delta=f"{delta_val:.1f} pts vs Wk {int(display_prev_week)}" if pd.notna(delta_val) and display_prev_week is not None else None,
            delta_color="normal"
        )
        # Sparkline based on overall filtered trend
        avg_trend_data_filtered = df_filtered.groupby('Week')['Engaged Transaction %'].mean().reset_index() if 'Week' in df_filtered.columns else pd.DataFrame()
        sparkline_avg = create_sparkline(avg_trend_data_filtered.tail(12), 'Engaged Transaction %', 'Avg Trend')
        if sparkline_avg:
            st.markdown('<div class="metric-sparkline">', unsafe_allow_html=True)
            st.altair_chart(sparkline_avg, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Metric 2: Donut Chart
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown("<h4 style='text-align: center; margin-bottom: 0;'>Engagement Rate</h4>", unsafe_allow_html=True)
        donut_value = display_avg_val # Use the same average as the first metric
        if pd.notna(donut_value):
            donut_chart = create_donut_chart(donut_value, title="Engaged")
            st.altair_chart(donut_chart, use_container_width=True)
        else:
             st.text("N/A")
        st.markdown('</div>', unsafe_allow_html=True)

    # Metric 3: Top Performer
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric(
            f"Top Performer {metric_suffix}", # Use same suffix as Avg card
            f"Store {top_store}" if top_store else "N/A",
            help=f"{top_val:.1f}%" if pd.notna(top_val) else None # Display value in tooltip
        )
        # Sparkline for top store's trend within filtered period
        if top_store:
            top_store_trend_data = df_filtered[df_filtered['Store #'] == top_store].sort_values('Week')
            sparkline_top = create_sparkline(top_store_trend_data.tail(12), 'Engaged Transaction %', 'Top Store Trend')
            if sparkline_top:
                st.markdown('<div class="metric-sparkline">', unsafe_allow_html=True)
                st.altair_chart(sparkline_top, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # --- Main Trend Chart ---
    st.markdown("### Engagement Trend Over Time")
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    if 'Week' not in df_filtered.columns or 'Engaged Transaction %' not in df_filtered.columns:
         st.warning("Missing 'Week' or 'Engaged Transaction %' for trend chart.")
    else:
        # Base chart using the filtered data
        base_chart = alt.Chart(df_filtered).encode(
             x=alt.X('Week:O', title='Week')
         )
        # Store lines
        lines = base_chart.mark_line(point=False).encode(
             y=alt.Y('Engaged Transaction %:Q', title='Engaged Transaction %', scale=alt.Scale(zero=False)),
             color=alt.Color('Store #:N', legend=None), # Legend off for cleanliness by default
             tooltip=[alt.Tooltip('Store #:N'), alt.Tooltip('Week:O'), alt.Tooltip('Engaged Transaction %:Q', format='.1f')]
         )
        chart_layers = [lines]

        # Interactive selection (optional, can add back if needed)
        # store_selection = alt.selection_multi(fields=['Store #'], bind='legend')
        # lines = lines.add_selection(store_selection).encode(opacity=alt.condition(store_selection, alt.value(1), alt.value(0.2)))

        # District average line (calculated from filtered data)
        if (not store_choice or len(store_choice) > 1) and not avg_trend_data_filtered.empty:
            district_avg_line = alt.Chart(avg_trend_data_filtered).mark_line(
                 strokeDash=[3,3], color='black', opacity=0.7, size=2
             ).encode(
                 x=alt.X('Week:O'),
                 y=alt.Y('Engaged Transaction %:Q'),
                 tooltip=[alt.Tooltip('Week:O'), alt.Tooltip('Engaged Transaction %:Q', format='.1f', title='District Avg')]
             )
            chart_layers.append(district_avg_line)

        # Combine layers
        chart = alt.layer(*chart_layers).interactive() # Enable zoom/pan
        st.altair_chart(chart.properties(height=350), use_container_width=True)
        if (not store_choice or len(store_choice) > 1) and not avg_trend_data_filtered.empty: st.caption("Dashed line indicates average across selected stores/period.")
    st.markdown('</div>', unsafe_allow_html=True)


# --- TAB 2: Store Comparison ---
with tab_comparison:
    st.markdown("### Store Performance Comparison")
    comp_period_label = f"Week {int(week_choice)}" if week_num is not None else "Selected Period Avg"
    st.markdown(f"Comparing **Engagement Percentage** for: **{comp_period_label}**")

    if 'Store #' not in df_filtered.columns or 'Engaged Transaction %' not in df_filtered.columns:
        st.warning("Missing required data for comparison.")
    elif len(df_filtered['Store #'].unique()) < 2:
        st.info("At least two stores required for comparison.")
    else:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        # Use the perf_data calculated earlier (based on display_week or overall filtered)
        comp_data = store_perf_display.reset_index()
        comp_data.columns = ['Store #', 'Engaged Transaction %'] # Rename for consistency

        if not comp_data.empty:
            comp_data = comp_data.sort_values('Engaged Transaction %', ascending=False)
            bar_chart = alt.Chart(comp_data).mark_bar().encode(
                y=alt.Y('Store #:N', title=None, sort='-x'), # No Y axis title
                x=alt.X('Engaged Transaction %:Q', title='Engaged Transaction %'),
                tooltip=[alt.Tooltip('Store #:N'), alt.Tooltip('Engaged Transaction %:Q', format='.1f')]
                # color=alt.value(PUBLIX_GREEN_BRIGHT) # Optionally set bar color
            )
            # Average line based on the comparison data itself
            comparison_avg = comp_data['Engaged Transaction %'].mean()
            rule = alt.Chart(pd.DataFrame({'avg': [comparison_avg]})).mark_rule(
                 color='black', strokeDash=[3,3], size=1.5
             ).encode(x=alt.X('avg:Q'), tooltip=[alt.Tooltip('avg:Q', title='Average', format='.1f')])

            st.altair_chart((bar_chart + rule).properties(height=alt.Step(18)), use_container_width=True)
            st.caption(f"Average engagement across selection: {comparison_avg:.1f}%.")
        else:
            st.warning("Could not compute comparison data.")
        st.markdown('</div>', unsafe_allow_html=True)

        # --- Weekly Rank Tracking (from original script) ---
        if 'Weekly Rank' in df_filtered.columns and 'Week' in df_filtered.columns:
             st.markdown("### Weekly Rank Tracking")
             st.markdown('<div class="chart-card">', unsafe_allow_html=True)
             # Use df_filtered directly as rank is weekly
             rank_data = df_filtered[['Week', 'Store #', 'Weekly Rank']].dropna(subset=['Weekly Rank'])
             # Ensure Rank is integer
             if not rank_data.empty and pd.api.types.is_numeric_dtype(rank_data['Weekly Rank']):
                  rank_data['Weekly Rank'] = rank_data['Weekly Rank'].astype(int)

                  # Determine rank domain dynamically for inversion
                  min_rank_val = rank_data['Weekly Rank'].max() + 1 # Y-axis bottom (worst rank + buffer)
                  max_rank_val = rank_data['Weekly Rank'].min() -1 # Y-axis top (best rank - buffer)
                  if max_rank_val < 0: max_rank_val = 0 # Ensure top is not negative

                  rank_chart = alt.Chart(rank_data).mark_line(point=True).encode(
                       x=alt.X('Week:O', title='Week'),
                       y=alt.Y('Weekly Rank:Q', title='Rank (Lower is Better)',
                               scale=alt.Scale(domain=[min_rank_val, max_rank_val])), # Inverted scale
                       color=alt.Color('Store #:N', legend=None), # Legend off for cleanliness
                       tooltip=[alt.Tooltip('Store #:N'), alt.Tooltip('Week:O'), alt.Tooltip('Weekly Rank:Q', title='Rank')]
                   ).interactive().properties(height=300)

                  st.altair_chart(rank_chart, use_container_width=True)
                  st.caption("Shows weekly performance rank (lower number is better).")
             else:
                  st.info("Weekly rank data not available or not numeric for the selected filters.")
             st.markdown('</div>', unsafe_allow_html=True)


# --- TAB 3: Performance Categories (Using ORIGINAL script's logic) ---
with tab_categories:
    st.markdown("### Store Performance Categories")
    st.caption("Categorized based on average engagement vs. median and trend correlation.")

    if not all(col in df_filtered.columns for col in ['Store #', 'Week', 'Engaged Transaction %']):
         st.warning("Missing data required for categorization.")
    else:
        # --- Category Calculation (Using ORIGINAL script's logic) ---
        # 1) Calculate average & std dev (consistency)
        store_stats_cat = df_filtered.groupby('Store #')['Engaged Transaction %'].agg(['mean', 'std']).reset_index()
        store_stats_cat.columns = ['Store #', 'Average Engagement', 'Std Dev']
        store_stats_cat['Std Dev'] = store_stats_cat['Std Dev'].fillna(0.0)

        # 2) Calculate trend correlation (Week vs. Engagement) - more robust trend indicator
        trend_corr_data = []
        for store_id, grp in df_filtered.groupby('Store #'):
            # Need at least 3 points for meaningful correlation
            if len(grp) >= 3:
                 grp_cleaned = grp.sort_values('Week').dropna(subset=['Engaged Transaction %', 'Week'])
                 # Check for variance after cleaning NaNs
                 if len(grp_cleaned) >= 2 and grp_cleaned['Week'].nunique() > 1 and grp_cleaned['Engaged Transaction %'].nunique() > 1:
                     try:
                         # Use pandas correlation method directly
                         corr_val = grp_cleaned[['Week', 'Engaged Transaction %']].corr(method='pearson').iloc[0, 1]
                         # Check if corr_val is NaN (can happen with perfect horizontal/vertical lines)
                         if pd.isna(corr_val): corr_val = 0.0
                     except Exception: # Catch any other errors during correlation calculation
                          corr_val = 0.0
                 else: # No variance or not enough points after cleaning
                     corr_val = 0.0
            else: # Not enough points initially
                corr_val = 0.0
            trend_corr_data.append({'Store #': store_id, 'Trend Correlation': corr_val})

        if trend_corr_data:
             trend_df_cat = pd.DataFrame(trend_corr_data)
             store_stats_cat = store_stats_cat.merge(trend_df_cat, on='Store #', how='left')
             # Fill NaN correlations with 0 (stable)
             store_stats_cat['Trend Correlation'] = store_stats_cat['Trend Correlation'].fillna(0.0)
        else: # Handle case where no trend data could be generated
             store_stats_cat['Trend Correlation'] = 0.0

        # 3) Assign Category based on Median Engagement and Trend Correlation
        if not store_stats_cat.empty:
             median_engagement = store_stats_cat['Average Engagement'].median()
             trend_threshold = 0.1 # Correlation threshold for positive/negative trend

             def assign_category_original(row):
                 avg_eng = row['Average Engagement']
                 trend_corr = row['Trend Correlation']
                 is_above_median = avg_eng >= median_engagement
                 has_positive_trend = trend_corr > trend_threshold
                 has_negative_trend = trend_corr < -trend_threshold # Use negative threshold

                 if is_above_median:
                     if has_negative_trend: return "Needs Stabilization"
                     else: return "Star Performer" # High and stable/improving
                 else: # Below Median
                     if has_positive_trend: return "Improving"
                     else: return "Requires Intervention" # Low and stable/declining

             store_stats_cat['Category'] = store_stats_cat.apply(assign_category_original, axis=1)
        else: # Handle case where store_stats is empty
             store_stats_cat = pd.DataFrame(columns=['Store #', 'Average Engagement', 'Std Dev', 'Trend Correlation', 'Category'])


        # --- Display Category Definitions & Results ---
        if not store_stats_cat.empty:
             category_order = ["Star Performer", "Needs Stabilization", "Improving", "Requires Intervention"]
             category_definitions = { # Using original categories
                 "Star Performer": {"icon": "⭐", "desc": "High engagement with stable/positive trend.", "color": CAT_COLORS["Star Performer"]},
                 "Needs Stabilization": {"icon": "⚠️", "desc": "High engagement but negative trend.", "color": CAT_COLORS["Needs Stabilization"]},
                 "Improving": {"icon": "📈", "desc": "Below median but positive trend.", "color": CAT_COLORS["Improving"]},
                 "Requires Intervention": {"icon": "🚨", "desc": "Below median with flat/negative trend.", "color": CAT_COLORS["Requires Intervention"]}
             }

             # Display Definitions using columns and styled cards
             st.markdown("#### Category Definitions")
             col_defs1, col_defs2 = st.columns(2)
             defs_list = list(category_definitions.items())
             for i, (cat, info) in enumerate(defs_list):
                 target_col = col_defs1 if i < len(defs_list) / 2 else col_defs2
                 with target_col:
                     st.markdown(f"""<div class="metric-card" style="border-left: 5px solid {info['color']};">
                                     <h4 style="color: {info['color']}; margin:0;">{info['icon']} {cat}</h4>
                                     <small>{info['desc']}</small>
                                     </div>""", unsafe_allow_html=True)

             # Display Stores by Category
             st.markdown("#### Stores by Category")
             st.markdown('<div class="chart-card">', unsafe_allow_html=True) # Use card for layout
             cols_cat_disp = st.columns(len(category_order))
             for i, cat in enumerate(category_order):
                  cat_stores = store_stats_cat[store_stats_cat['Category'] == cat]
                  info = category_definitions[cat]
                  with cols_cat_disp[i]:
                      st.markdown(f"<h4 style='color:{info['color']}; border-bottom: 2px solid {info['color']}; padding-bottom: 5px; margin-bottom: 10px;'>{info['icon']} {cat} ({len(cat_stores)})</h4>", unsafe_allow_html=True)
                      if not cat_stores.empty:
                           cat_stores_sorted = cat_stores.sort_values('Average Engagement', ascending=(cat not in ["Star Performer", "Needs Stabilization"]))
                           for idx, store_row in cat_stores_sorted.iterrows():
                                trend_corr_val = store_row['Trend Correlation']
                                trend_info = f"<small>(Trend: {trend_corr_val:.2f})</small>" if abs(trend_corr_val) > trend_threshold else "<small>(Trend: Stable)</small>"
                                st.markdown(f"**Store {store_row['Store #']}** <small>({store_row['Average Engagement']:.1f}%)</small><br>{trend_info}", unsafe_allow_html=True)
                      else:
                           st.caption("None")
             st.markdown('</div>', unsafe_allow_html=True)

             # --- Detailed Store View (Add back if needed, simplified here) ---
             # st.markdown("---")
             # st.markdown("#### Detailed Store View")
             # # Add selectbox and display logic here, similar to original script Tab 3

        else:
             st.warning("Could not calculate store statistics for categorization.")


# --- TAB 4: Anomalies ---
with tab_anomalies:
    st.markdown("### Anomalies")
    st.caption(f"Highlights significant week-over-week changes (Z-score > {z_threshold:.1f}). Includes rank changes if available.")

    # Calculate anomalies using the robust function
    anomalies_found = find_anomalies(df_filtered, z_threshold=z_threshold) # Use sidebar setting

    st.markdown('<div class="chart-card">', unsafe_allow_html=True) # Use card for layout
    if anomalies_found.empty:
        st.success(f"✅ No significant weekly changes detected (Z > {z_threshold:.1f}).")
    else:
        st.warning(f"🚨 Found {len(anomalies_found)} anomalies.")
        # Display relevant columns including explanation and rank if available
        display_cols = ['Store #', 'Week', 'Engaged Transaction %', 'Change %pts', 'Z-score']
        if 'Rank' in anomalies_found.columns and 'Prev Rank' in anomalies_found.columns and not anomalies_found['Rank'].isna().all():
             display_cols.extend(['Rank', 'Prev Rank'])
        display_cols.append('Possible Explanation')

        st.dataframe(
            anomalies_found[display_cols],
            hide_index=True, use_container_width=True
        )
    st.markdown('</div>', unsafe_allow_html=True)


# --- Footer ---
st.sidebar.markdown("---")
st.sidebar.caption(f"© Publix Supermarkets, Inc. {datetime.date.today().year}")
