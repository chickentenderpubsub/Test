import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import datetime

# --------------------------------------------------------
# Theme Configuration (Inspired by Reference Image)
# --------------------------------------------------------
PUBLIX_GREEN_DARK = "#00543D" # Darker green for sidebar, potentially primary elements
PUBLIX_GREEN_BRIGHT = "#5F8F38" # Brighter green for accents, charts
BACKGROUND_COLOR = "#F9F9F9" # Very light gray/off-white background
CARD_BACKGROUND_COLOR = "#FFFFFF" # White background for cards
TEXT_COLOR_DARK = "#333333" # Primary text
TEXT_COLOR_MEDIUM = "#666666" # Secondary text (labels)
TEXT_COLOR_LIGHT = "#FFFFFF" # Text on dark background (sidebar)
BORDER_COLOR = "#EAEAEA" # Subtle border for cards
# Add colors from categories for consistency if needed elsewhere
CAT_COLORS = {
    "High Performer": PUBLIX_GREEN_BRIGHT,
    "High & Declining": "#FFA726", # Orange warning
    "Low & Improving": "#42A5F5", # Blue for improving
    "Low Performer": "#EF5350", # Red for low
    "default": TEXT_COLOR_MEDIUM # Fallback color
    }

# --------------------------------------------------------
# Helper Functions (Adding explicit types in Altair)
# --------------------------------------------------------

@st.cache_data
def load_data(uploaded_file):
    # --- (Load Data - same as previous version) ---
    if uploaded_file is None: return pd.DataFrame()
    filename = uploaded_file.name.lower()
    try:
        if filename.endswith('.csv'): df = pd.read_csv(uploaded_file)
        elif filename.endswith(('.xlsx', '.xls')): df = pd.read_excel(uploaded_file)
        else: st.error("Unsupported file type."); return pd.DataFrame()
    except Exception as e: st.error(f"Error reading file: {e}"); return pd.DataFrame()

    df.columns = standardize_columns(df.columns)

    # --- Data Cleaning ---
    required_cols = ['Store #', 'Engaged Transaction %']
    if not ('Date' in df.columns or 'Week' in df.columns):
        st.error("Data needs 'Week' or 'Date' column."); return pd.DataFrame()
    if not all(col in df.columns for col in required_cols):
        st.error(f"Data needs: {', '.join(required_cols)}."); return pd.DataFrame()

    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.dropna(subset=['Date'])
        if 'Week' not in df.columns and pd.api.types.is_datetime64_any_dtype(df['Date']):
            df['Week'] = df['Date'].dt.isocalendar().week.astype(int)

    percent_cols = ['Engaged Transaction %', 'Quarter to Date %']
    for col in percent_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace('%', '', regex=False), errors='coerce')

    essential_cols = ['Store #', 'Engaged Transaction %']
    if 'Week' in df.columns: essential_cols.append('Week')
    elif 'Date' in df.columns: essential_cols.append('Date')
    df = df.dropna(subset=essential_cols)

    if 'Weekly Rank' in df.columns:
        df['Weekly Rank'] = pd.to_numeric(df['Weekly Rank'], errors='coerce').astype('Int64')
    df['Store #'] = df['Store #'].astype(str)

    if 'Week' in df.columns:
        df['Week'] = pd.to_numeric(df['Week'], errors='coerce').dropna().astype(int)
        df = df.sort_values(['Week', 'Store #'])
        if 'Quarter' not in df.columns:
             df['Quarter'] = df['Week'].apply(lambda w: (int(w) - 1) // 13 + 1 if pd.notna(w) else None).astype('Int64')
    elif 'Date' in df.columns:
        df = df.sort_values(['Date', 'Store #'])
        if 'Quarter' not in df.columns and pd.api.types.is_datetime64_any_dtype(df['Date']):
             df['Quarter'] = df['Date'].dt.quarter.astype('Int64')

    return df

def standardize_columns(columns):
    # --- (Standardize Columns - same as previous version) ---
    new_cols = []
    processed_indices = set()
    cols_lower = [str(col).strip().lower() for col in columns]

    for i, col_lower in enumerate(cols_lower):
        if i in processed_indices: continue
        original_col = columns[i]
        if 'store' in col_lower and ('#' in col_lower or 'id' in col_lower or 'number' in col_lower): new_cols.append('Store #')
        elif 'engage' in col_lower and ('%' in col_lower or 'transaction' in col_lower): new_cols.append('Engaged Transaction %')
        elif ('week' in col_lower and 'end' in col_lower) or col_lower == 'date': new_cols.append('Date')
        elif col_lower == 'week' or (col_lower.startswith('week') and not any(s in col_lower for s in ['rank', 'end', 'date'])): new_cols.append('Week')
        elif 'rank' in col_lower and 'week' in col_lower: new_cols.append('Weekly Rank')
        elif 'quarter' in col_lower or 'qtd' in col_lower: new_cols.append('Quarter to Date %')
        else: new_cols.append(original_col)
        processed_indices.add(i)
    return new_cols


def calculate_trend(group, window=4, engagement_col='Engaged Transaction %', week_col='Week'):
    # --- (Calculate Trend - same as previous version) ---
    if engagement_col not in group.columns or week_col not in group.columns: return "Missing Data"
    if len(group) < 2: return "Insufficient Data"
    sorted_data = group.sort_values(week_col, ascending=True).tail(window)
    if len(sorted_data) < 2: return "Insufficient Data"
    sorted_data = sorted_data.dropna(subset=[engagement_col])
    if len(sorted_data) < 2: return "Insufficient Data"
    x = sorted_data[week_col].values; y = sorted_data[engagement_col].values
    if np.std(x) == 0 or np.std(y) == 0: return "Stable"
    try: coeffs = np.polyfit(x, y, 1); slope = coeffs[0]
    except (np.linalg.LinAlgError, ValueError): return "Calculation Error"
    strong_threshold = 0.5; mild_threshold = 0.1
    if slope > strong_threshold: return "Strong Upward"
    elif slope > mild_threshold: return "Upward"
    elif slope < -strong_threshold: return "Strong Downward"
    elif slope < -mild_threshold: return "Downward"
    else: return "Stable"

def find_anomalies(df, z_threshold=2.0, engagement_col='Engaged Transaction %', store_col='Store #', week_col='Week'):
     # --- (Find Anomalies - same as previous version) ---
    if df.empty or not all(col in df.columns for col in [engagement_col, store_col, week_col]): return pd.DataFrame()
    anomalies_list = []; df_sorted = df.sort_values([store_col, week_col])
    for store_id, grp in df_sorted.groupby(store_col):
        grp = grp.dropna(subset=[engagement_col]); diffs = grp[engagement_col].diff().dropna()
        if len(diffs) < 2: continue
        mean_diff = diffs.mean(); std_diff = diffs.std(ddof=0)
        if std_diff == 0 or pd.isna(std_diff): continue
        for idx, diff_val in diffs.items():
            if pd.isna(diff_val) or std_diff == 0: continue
            z = (diff_val - mean_diff) / std_diff
            if abs(z) >= z_threshold:
                current_row = grp.loc[idx]
                explanation = f"Significant {'increase' if diff_val >= 0 else 'decrease'} ({diff_val:+.2f} pts)"
                anomalies_list.append({
                    'Store #': store_id,
                    'Week': int(current_row.get(week_col)) if pd.notna(current_row.get(week_col)) else None,
                    'Engaged Transaction %': round(current_row.get(engagement_col), 2) if pd.notna(current_row.get(engagement_col)) else None,
                    'Change %pts': round(diff_val, 2) if pd.notna(diff_val) else None,
                    'Z-score': round(z, 2),
                    'Explanation': explanation
                })
    if not anomalies_list: return pd.DataFrame()
    anomalies_df = pd.DataFrame(anomalies_list)
    anomalies_df['Abs Z'] = anomalies_df['Z-score'].abs()
    return anomalies_df.sort_values('Abs Z', ascending=False).drop(columns=['Abs Z'])


# --------------------------------------------------------
# Altair Chart Theming (Clean, Publix Green Focus)
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
            "legend": None,
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

# Helper to create sparkline chart (ADDED EXPLICIT TYPES)
def create_sparkline(data, y_col, y_title):
    if data is None or data.empty or y_col not in data.columns or 'Week' not in data.columns:
        return None
     # Ensure data is DataFrame for safety
    if not isinstance(data, pd.DataFrame):
         try: data = pd.DataFrame(data)
         except Exception: return None # Cannot convert

    line = alt.Chart(data).mark_line().encode(
        x=alt.X('Week:O', axis=None), # Ordinal type for week
        y=alt.Y(f'{y_col}:Q', axis=None), # Quantitative type for value
        tooltip=[
            alt.Tooltip('Week:O'), # Ordinal type
            alt.Tooltip(f'{y_col}:Q', format='.2f', title=y_title) # Quantitative type
        ]
    ).properties(width=80, height=30)
    area = line.mark_area()
    return area + line

# Helper to create donut chart (ADDED EXPLICIT TYPES)
def create_donut_chart(value, title="Engagement"):
     value = max(0, min(100, value))
     source = pd.DataFrame({"category": [title, "Remaining"], "value": [value, 100 - value]})
     base = alt.Chart(source).encode(
         # Quantitative type for theta
         theta=alt.Theta("value:Q", stack=True)
     )
     pie = base.mark_arc(outerRadius=60, innerRadius=45).encode(
         # Nominal type for color category
         color=alt.Color("category:N",
                          scale=alt.Scale(domain=[title, "Remaining"], range=[PUBLIX_GREEN_BRIGHT, BORDER_COLOR]),
                          legend=None),
         order=alt.Order("value:Q", sort="descending") # Quantitative type for order
     )
     text = base.mark_text(radius=0, align='center', baseline='middle', fontSize=18, fontWeight='bold').encode(
         # Text derived from quantitative value
         text=alt.condition(
             alt.datum.category == title,
             alt.Text("value:Q", format=".0f"), # Quantitative type
             alt.value("")
         ),
         # Quantitative type for order
         order=alt.Order("value:Q", sort="descending"),
         color=alt.value(TEXT_COLOR_DARK)
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

# --- Custom CSS Injection (Same as previous version) ---
st.markdown(f"""
<style>
    /* --- Body and Background --- */
    body {{ font-family: sans-serif; }}
    .stApp {{ background-color: {BACKGROUND_COLOR}; }}
    /* --- Sidebar --- */
    [data-testid="stSidebar"] > div:first-child {{ background-color: {PUBLIX_GREEN_DARK}; padding-top: 1rem; }}
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] label, [data-testid="stSidebar"] .st-caption {{ color: {TEXT_COLOR_LIGHT} !important; font-weight: normal; }}
    [data-testid="stSidebar"] .st-emotion-cache-1bzkvze {{ color: {TEXT_COLOR_LIGHT} !important; }}
    [data-testid="stSidebar"] small {{ color: rgba(255, 255, 255, 0.7) !important; }}
    [data-testid="stSidebar"] .stSelectbox > div, [data-testid="stSidebar"] .stMultiselect > div {{ background-color: rgba(255, 255, 255, 0.1); border: none; border-radius: 4px; }}
    [data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] > div, [data-testid="stSidebar"] .stMultiselect div[data-baseweb="select"] > div {{ color: {TEXT_COLOR_LIGHT}; }}
    [data-testid="stSidebar"] .stSelectbox svg, [data-testid="stSidebar"] .stMultiselect svg {{ fill: {TEXT_COLOR_LIGHT}; }}
    [data-testid="stSidebar"] .stMultiselect span[data-baseweb="tag"] svg {{ fill: {TEXT_COLOR_LIGHT}; }}
    [data-testid="stSidebar"] .stButton>button {{ background-color: transparent; color: {TEXT_COLOR_LIGHT}; border: 1px solid rgba(255, 255, 255, 0.5); font-weight: normal; padding: 5px 10px; margin-top: 0.5rem; }}
    [data-testid="stSidebar"] .stButton>button:hover {{ background-color: rgba(255, 255, 255, 0.1); border-color: {TEXT_COLOR_LIGHT}; }}
    [data-testid="stSidebar"] .stFileUploader button {{ border: 1px dashed rgba(255, 255, 255, 0.5); background-color: rgba(255, 255, 255, 0.05); color: {TEXT_COLOR_LIGHT}; }}
    [data-testid="stSidebar"] .stFileUploader button:hover {{ border-color: {TEXT_COLOR_LIGHT}; background-color: rgba(255, 255, 255, 0.1); }}
    /* --- Main Content Area --- */
    .main .block-container {{ padding: 1.5rem 2rem 2rem 2rem; max-width: 95%; }}
    /* --- Card Styling --- */
    .metric-card {{ background-color: {CARD_BACKGROUND_COLOR}; border: 1px solid {BORDER_COLOR}; border-radius: 8px; padding: 1rem 1.2rem; box-shadow: 0 1px 3px rgba(0,0,0,0.03); height: 100%; display: flex; flex-direction: column; justify-content: space-between; }}
    .chart-card {{ background-color: {CARD_BACKGROUND_COLOR}; border: 1px solid {BORDER_COLOR}; border-radius: 8px; padding: 1.2rem; box-shadow: 0 1px 3px rgba(0,0,0,0.03); }}
    /* --- Metric Styling inside Cards --- */
    .stMetric {{ background-color: transparent !important; border: none !important; padding: 0 !important; text-align: left; color: {TEXT_COLOR_DARK}; font-size: 1rem; height: 100%; display: flex; flex-direction: column; }}
    .stMetric > label {{ color: {TEXT_COLOR_MEDIUM} !important; font-weight: normal !important; font-size: 0.85em !important; margin-bottom: 0.25rem; order: 1; }}
    .stMetric > div:nth-of-type(1) {{ color: {TEXT_COLOR_DARK} !important; font-size: 2.0em !important; font-weight: bold !important; line-height: 1.1 !important; margin-bottom: 0.25rem; order: 0; }}
    .stMetric > div:nth-of-type(2) {{ font-size: 0.85em !important; font-weight: bold; color: {TEXT_COLOR_MEDIUM} !important; order: 2; margin-top: auto; }}
    .stMetric .stMetricDelta {{ padding-top: 5px; }}
    .stMetric .stMetricDelta span[style*="color: rgb(46, 125, 50)"] {{ color: {PUBLIX_GREEN_BRIGHT} !important; }}
    .stMetric .stMetricDelta span[style*="color: rgb(198, 40, 40)"] {{ color: #D32F2F !important; }}
    .stMetric .metric-sparkline {{ margin-top: auto; order: 3; padding-top: 10px; line-height: 0; opacity: 0.7; }}
    /* --- Tabs --- */
    div[data-baseweb="tab-list"] {{ border-bottom: 2px solid {BORDER_COLOR}; padding-left: 0; margin-bottom: 1.5rem; }}
    button[data-baseweb="tab"] {{ background-color: transparent; color: {TEXT_COLOR_MEDIUM}; padding: 0.6rem 0.1rem; margin-right: 1.5rem; border-bottom: 2px solid transparent; font-weight: normal; }}
    button[data-baseweb="tab"]:hover {{ background-color: transparent; color: {TEXT_COLOR_DARK}; border-bottom-color: {BORDER_COLOR}; }}
    button[data-baseweb="tab"][aria-selected="true"] {{ color: {PUBLIX_GREEN_BRIGHT}; font-weight: bold; border-bottom-color: {PUBLIX_GREEN_BRIGHT}; }}
    /* --- General Elements --- */
    h1, h2, h3 {{ color: {TEXT_COLOR_DARK}; font-weight: bold; }}
    h3 {{ margin-top: 1.5rem; margin-bottom: 0.8rem; font-size: 1.3rem; }}
    h4 {{ margin-top: 1rem; margin-bottom: 0.5rem; font-size: 1.1rem; color: {TEXT_COLOR_MEDIUM}; font-weight: bold; }}
    [data-testid="stExpander"] {{ border: 1px solid {BORDER_COLOR}; border-radius: 6px; background-color: {CARD_BACKGROUND_COLOR}; }}
    [data-testid="stExpander"] summary {{ font-weight: normal; color: {TEXT_COLOR_MEDIUM}; }}
    [data-testid="stExpander"] summary:hover {{ color: {PUBLIX_GREEN_BRIGHT}; }}
    hr {{ border-top: 1px solid {BORDER_COLOR}; margin: 1.5rem 0; }}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------------
# Sidebar Setup
# --------------------------------------------------------
st.sidebar.markdown(f"""
<div style="text-align: center; padding-bottom: 1rem;">
<svg width="40" height="40" viewBox="0 0 100 100" fill="none" xmlns="http://www.w3.org/2000/svg">
<circle cx="50" cy="50" r="50" fill="{TEXT_COLOR_LIGHT}"/>
<path d="M63.7148 25H42.4297C35.1445 25 29.2852 30.8594 29.2852 38.1445V43.6055C29.2852 46.6523 31.7734 49.1406 34.8203 49.1406H46.0117V61.8555C46.0117 64.9023 48.4999 67.3906 51.5468 67.3906H57.1406C60.1875 67.3906 62.6757 64.9023 62.6757 61.8555V49.1406H66.4804C68.0234 49.1406 69.2851 47.8789 69.2851 46.3359V31.0195C69.2851 27.7148 66.7304 25 63.7148 25ZM57.1406 55.6641H51.5468V43.6055C51.5468 40.5586 49.0585 38.0703 46.0117 38.0703H40.4179C39.1992 38.0703 38.2226 39.0469 38.2226 40.1914V43.6055C38.2226 44.75 39.1992 45.7266 40.4179 45.7266H51.5468C54.5937 45.7266 57.1406 48.2148 57.1406 51.2617V55.6641Z" fill="{PUBLIX_GREEN_DARK}"/>
</svg></div>""", unsafe_allow_html=True)

st.sidebar.header("Data Upload")
data_file = st.sidebar.file_uploader("Engagement Data", type=['csv', 'xlsx', 'xls'], key="primary_upload", label_visibility="collapsed")

df = load_data(data_file)

if df.empty:
    st.info("⬆️ Please upload an engagement data file using the sidebar to begin.")
    st.stop()

st.sidebar.header("Filters")
# --- Sidebar Filters ---
quarter_choice = "All"
if 'Quarter' in df.columns and df['Quarter'].notna().any():
    quarters = sorted(df['Quarter'].dropna().unique().astype(int).tolist())
    quarter_options = ["All"] + [f"Q{q}" for q in quarters]
    quarter_choice = st.sidebar.selectbox("Quarter", quarter_options, index=0, key="quarter_select")

week_choice = "All"
if 'Week' in df.columns and df['Week'].notna().any():
    if quarter_choice != "All":
        try: q_num = int(quarter_choice[1:]); available_weeks = sorted(df.loc[df['Quarter'] == q_num, 'Week'].dropna().unique().astype(int).tolist())
        except (ValueError, IndexError): available_weeks = sorted(df['Week'].dropna().unique().astype(int).tolist())
    else: available_weeks = sorted(df['Week'].dropna().unique().astype(int).tolist())
    if available_weeks: week_options = ["All"] + [str(w) for w in available_weeks]; week_choice = st.sidebar.selectbox("Week", week_options, index=0, key="week_select")
    else: week_choice = "All"

store_list = sorted(df['Store #'].dropna().unique().tolist()) if 'Store #' in df.columns else []
store_choice = st.sidebar.multiselect("Store(s)", store_list, default=[], key="store_select") if store_list else []

# --- Filter DataFrames ---
df_filtered = df.copy()
if quarter_choice != "All" and 'Quarter' in df_filtered.columns:
    try: q_num_filter = int(quarter_choice[1:]); df_filtered = df_filtered[df_filtered['Quarter'] == q_num_filter]
    except (ValueError, IndexError): pass
if week_choice != "All" and 'Week' in df_filtered.columns:
    try: week_num_filter = int(week_choice); df_filtered = df_filtered[df_filtered['Week'] == week_num_filter]
    except ValueError: pass
if store_choice and 'Store #' in df_filtered.columns:
    df_filtered = df_filtered[df_filtered['Store #'].isin(store_choice)]

if df_filtered.empty:
    st.warning("No data matches the selected filters.")
    st.stop()

# --- Pre-calculate overall stats ---
overall_avg_engagement = df_filtered['Engaged Transaction %'].mean()
latest_week = df_filtered['Week'].max() if 'Week' in df_filtered.columns and not df_filtered['Week'].isna().all() else None
earliest_week = df_filtered['Week'].min() if 'Week' in df_filtered.columns and not df_filtered['Week'].isna().all() else None

prev_week = None
if latest_week is not None:
    possible_prev_weeks = df_filtered.loc[df_filtered['Week'] < latest_week, 'Week'].dropna()
    if not possible_prev_weeks.empty: prev_week = possible_prev_weeks.max()

latest_week_data = df_filtered[df_filtered['Week'] == latest_week] if latest_week is not None else pd.DataFrame()
prev_week_data = df_filtered[df_filtered['Week'] == prev_week] if prev_week is not None else pd.DataFrame()

latest_avg = latest_week_data['Engaged Transaction %'].mean() if not latest_week_data.empty else None
prev_avg = prev_week_data['Engaged Transaction %'].mean() if not prev_week_data.empty else None
delta_val = latest_avg - prev_avg if pd.notna(latest_avg) and pd.notna(prev_avg) else None

avg_trend_data = df_filtered.groupby('Week')['Engaged Transaction %'].mean().reset_index() if 'Week' in df_filtered.columns else pd.DataFrame()

# --------------------------------------------------------
# Main Content Area - Tabs
# --------------------------------------------------------
tab_overview, tab_comparison, tab_categories, tab_anomalies = st.tabs([
    "Overview & Trends", "Store Comparison", "Performance Categories", "Data Issues"
])

# --- TAB 1: Overview & Trends ---
with tab_overview:
    st.markdown("### Key Metrics")
    col1, col2, col3 = st.columns(3)

    with col1: # Avg Engagement Card
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        metric_label = "Avg Engagement"
        if store_choice: metric_label = f"Avg ({len(store_choice)} Stores)" if len(store_choice)>1 else f"Store {store_choice[0]}"
        metric_suffix = f"(Wk {int(latest_week)})" if latest_week else ""
        st.metric(
            f"{metric_label} {metric_suffix}",
            f"{latest_avg:.1f}%" if pd.notna(latest_avg) else ("N/A" if latest_week else f"{overall_avg_engagement:.1f}%"), # Show overall if no latest week
            delta=f"{delta_val:.1f} pts vs Wk {int(prev_week)}" if pd.notna(delta_val) and prev_week is not None else None,
            delta_color="normal"
        )
        sparkline_avg = create_sparkline(avg_trend_data.tail(12), 'Engaged Transaction %', 'Avg Trend')
        if sparkline_avg:
            st.markdown('<div class="metric-sparkline">', unsafe_allow_html=True)
            st.altair_chart(sparkline_avg, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2: # Donut Chart Card
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown("<h4 style='text-align: center; margin-bottom: 0;'>Current Rate</h4>", unsafe_allow_html=True)
        donut_value = latest_avg if pd.notna(latest_avg) else overall_avg_engagement
        if pd.notna(donut_value):
            donut_chart = create_donut_chart(donut_value, title="Engaged")
            st.altair_chart(donut_chart, use_container_width=True)
        else:
             st.text("N/A")
        st.markdown('</div>', unsafe_allow_html=True)

    with col3: # Top Performer Card
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        top_store, top_val = None, None
        if not latest_week_data.empty and 'Store #' in latest_week_data.columns:
            perf = latest_week_data.groupby('Store #')['Engaged Transaction %'].mean()
            if not perf.empty: top_store = perf.idxmax(); top_val = perf.max()
        st.metric(
            f"Top Performer (Wk {int(latest_week)})" if latest_week else "Top Performer",
            f"Store {top_store}" if top_store else "N/A",
            help=f"{top_val:.1f}%" if pd.notna(top_val) else None
        )
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
        line = alt.Chart(df_filtered).mark_line(point=False).encode(
             x=alt.X('Week:O', title='Week'),
             y=alt.Y('Engaged Transaction %:Q', title='Engaged Transaction %', scale=alt.Scale(zero=False)),
             # Only show legend if few stores are selected, otherwise it gets cluttered
             color=alt.Color('Store #:N', legend=alt.Legend(title="Store", orient="top-left") if store_choice and 0 < len(store_choice) <= 10 else None),
             tooltip=[alt.Tooltip('Store #:N'), alt.Tooltip('Week:O'), alt.Tooltip('Engaged Transaction %:Q', format='.1f')]
         )
        chart_layer = [line]
        # Add district average if viewing multiple stores
        if (not store_choice or len(store_choice) > 1) and not avg_trend_data.empty:
            district_avg_line = alt.Chart(avg_trend_data).mark_line(
                 strokeDash=[3,3], color='black', opacity=0.7
             ).encode(
                 x=alt.X('Week:O'),
                 y=alt.Y('Engaged Transaction %:Q'),
                 tooltip=[alt.Tooltip('Week:O'), alt.Tooltip('Engaged Transaction %:Q', format='.1f', title='District Avg')]
             )
            chart_layer.append(district_avg_line)

        chart = alt.layer(*chart_layer).interactive()
        st.altair_chart(chart.properties(height=300), use_container_width=True)
        if (not store_choice or len(store_choice) > 1) and not avg_trend_data.empty: st.caption("Dashed line indicates district average.")
    st.markdown('</div>', unsafe_allow_html=True)

# --- TAB 2: Store Comparison ---
with tab_comparison:
    st.markdown("### Store Performance Comparison")
    comp_period_label = f"Week {int(week_choice)}" if week_choice != "All" else "Selected Period Avg"
    st.markdown(f"Comparing **Engagement Percentage** for: **{comp_period_label}**")

    if 'Store #' not in df_filtered.columns or 'Engaged Transaction %' not in df_filtered.columns:
        st.warning("Missing required data for comparison.")
    elif len(df_filtered['Store #'].unique()) < 2:
        st.info("At least two stores required for comparison.")
    else:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        comp_data = pd.DataFrame() # Initialize
        if week_choice != "All":
            try:
                week_num_comp = int(week_choice)
                comp_data = df_filtered[df_filtered['Week'] == week_num_comp]
                if comp_data.empty:
                    comp_data = df_filtered.groupby('Store #', as_index=False)['Engaged Transaction %'].mean()
                    st.caption(f"(No data for Wk {week_num_comp}, showing period average)")
                else:
                    comp_data = comp_data.groupby('Store #', as_index=False)['Engaged Transaction %'].mean()
            except ValueError:
                comp_data = df_filtered.groupby('Store #', as_index=False)['Engaged Transaction %'].mean()
                st.caption("(Invalid week selection, showing period average)")
        else:
            comp_data = df_filtered.groupby('Store #', as_index=False)['Engaged Transaction %'].mean()

        if not comp_data.empty:
            comp_data = comp_data.sort_values('Engaged Transaction %', ascending=False)
            bar_chart = alt.Chart(comp_data).mark_bar().encode(
                y=alt.Y('Store #:N', title=None, sort='-x'),
                x=alt.X('Engaged Transaction %:Q', title='Engaged Transaction %'),
                tooltip=[alt.Tooltip('Store #:N'), alt.Tooltip('Engaged Transaction %:Q', format='.1f')]
            )
            district_avg_comp = comp_data['Engaged Transaction %'].mean()
            rule = alt.Chart(pd.DataFrame({'avg': [district_avg_comp]})).mark_rule(
                 color='black', strokeDash=[3,3], size=1.5
             ).encode(x=alt.X('avg:Q'), tooltip=[alt.Tooltip('avg:Q', title='Average', format='.1f')])

            st.altair_chart((bar_chart + rule).properties(height=alt.Step(18)), use_container_width=True) # Step height
            st.caption(f"Average engagement across selection: {district_avg_comp:.1f}%.")
        else:
            st.warning("Could not compute comparison data.")
        st.markdown('</div>', unsafe_allow_html=True)


# --- TAB 3: Performance Categories ---
with tab_categories:
    st.markdown("### Store Performance Categories")
    st.caption("Categorized based on average engagement and recent trend within the filtered data.")

    if not all(col in df_filtered.columns for col in ['Store #', 'Week', 'Engaged Transaction %']):
         st.warning("Missing data required for categorization.")
    else:
        store_stats_cat = df_filtered.groupby('Store #')['Engaged Transaction %'].agg(['mean', 'std']).reset_index()
        if not store_stats_cat.empty:
            median_engagement = store_stats_cat['mean'].median()
            trends = []
            for store_id, grp in df_filtered.groupby('Store #'):
                 trends.append({'Store #': store_id, 'Trend': calculate_trend(grp, window=6)})
            trend_df_cat = pd.DataFrame(trends)
            store_stats_cat = store_stats_cat.merge(trend_df_cat, on='Store #', how='left').fillna({'Trend': 'Stable'})

            def assign_category_simple(row):
                 is_above_median = row['mean'] >= median_engagement
                 is_improving = "Upward" in row['Trend']
                 is_declining = "Downward" in row['Trend']
                 if is_above_median: return "High & Declining" if is_declining else "High Performer"
                 else: return "Low & Improving" if is_improving else "Low Performer"
            store_stats_cat['Category'] = store_stats_cat.apply(assign_category_simple, axis=1)

            category_order = ["High Performer", "High & Declining", "Low & Improving", "Low Performer"]

            st.markdown('<div class="chart-card">', unsafe_allow_html=True)
            cols_cat = st.columns(len(category_order))
            for i, cat in enumerate(category_order):
                 cat_stores = store_stats_cat[store_stats_cat['Category'] == cat]
                 cat_color = CAT_COLORS.get(cat, CAT_COLORS['default']) # Get color from dict
                 with cols_cat[i]:
                     st.markdown(f"<h4 style='color:{cat_color}; border-bottom: 2px solid {cat_color}; padding-bottom: 5px; margin-bottom: 10px;'>{cat} ({len(cat_stores)})</h4>", unsafe_allow_html=True)
                     if not cat_stores.empty:
                          # Sort stores by performance within category
                          cat_stores_sorted = cat_stores.sort_values('mean', ascending=(cat not in ["High Performer", "High & Declining"]))
                          for idx, store_row in cat_stores_sorted.iterrows():
                               trend_info = f"- <small>{store_row['Trend']}</small>" if store_row['Trend'] != 'Stable' else ""
                               st.markdown(f"**Store {store_row['Store #']}** ({store_row['mean']:.1f}%) {trend_info}", unsafe_allow_html=True)
                     else:
                          st.caption("None in this category.")
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.warning("Could not calculate store statistics for categorization.")


# --- TAB 4: Data Issues / Anomalies ---
with tab_anomalies:
    st.markdown("### Data Issues & Anomalies")
    st.caption("Highlights potential data errors or significant week-to-week changes.")

    col_anom1, col_anom2 = st.columns(2)
    with col_anom1:
        st.markdown("#### Significant Weekly Changes")
        st.markdown('<div class="chart-card" style="height: 95%;">', unsafe_allow_html=True) # Use card and try to match height
        anomalies_found = find_anomalies(df_filtered, z_threshold=2.0)
        if anomalies_found.empty:
            st.success("✅ No significant weekly changes detected (Z > 2.0).")
        else:
            st.warning(f"🚨 Found {len(anomalies_found)} instances.")
            st.dataframe(
                anomalies_found[['Store #', 'Week', 'Change %pts', 'Z-score', 'Explanation']], # Simplified view
                hide_index=True, use_container_width=True, height=250 # Limit height
            )
        st.markdown('</div>', unsafe_allow_html=True)

    with col_anom2:
        st.markdown("#### Potential Missing Weekly Data")
        st.markdown('<div class="chart-card" style="height: 95%;">', unsafe_allow_html=True) # Use card and try to match height
        if latest_week is not None and earliest_week is not None and 'Week' in df_filtered.columns and 'Store #' in df_filtered.columns:
            expected_weeks = set(range(int(earliest_week), int(latest_week) + 1))
            stores = df_filtered['Store #'].unique()
            missing_entries = []
            for store in stores:
                store_weeks = set(df_filtered[df_filtered['Store #'] == store]['Week'].unique())
                missing_for_store = expected_weeks - store_weeks
                if missing_for_store:
                    missing_entries.append({"Store #": store, "Missing Weeks Count": len(missing_for_store), "Example Missing": min(missing_for_store)})
            if missing_entries:
                 missing_df = pd.DataFrame(missing_entries)
                 st.warning(f" Found {len(missing_df)} store(s) with potential missing weekly entries.")
                 st.dataframe(missing_df, hide_index=True, use_container_width=True, height=250) # Limit height
            else:
                 st.success("✅ No obvious weekly data gaps detected.")
        else:
             st.info("Requires sufficient 'Week' and 'Store #' data for gap analysis.")
        st.markdown('</div>', unsafe_allow_html=True)


# --- Footer ---
st.sidebar.markdown("---")
st.sidebar.caption(f"© Publix Supermarkets, Inc. {datetime.date.today().year}")
