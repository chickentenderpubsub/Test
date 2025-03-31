import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import datetime

# --------------------------------------------------------
# Theme Configuration
# --------------------------------------------------------
PUBLIX_GREEN_DARK = "#00543D"
PUBLIX_GREEN_BRIGHT = "#5F8F38"
BACKGROUND_COLOR = "#F9F9F9"
CARD_BACKGROUND_COLOR = "#FFFFFF"
TEXT_COLOR_DARK = "#333333"
TEXT_COLOR_MEDIUM = "#666666"
TEXT_COLOR_LIGHT = "#FFFFFF"
BORDER_COLOR = "#EAEAEA"
CAT_COLORS = {
    "Star Performer": PUBLIX_GREEN_BRIGHT,
    "Needs Stabilization": "#FFA726",  # Orange warning
    "Improving": "#42A5F5",  # Blue for improving
    "Requires Intervention": "#EF5350",  # Red for low
    "default": TEXT_COLOR_MEDIUM
}

# --------------------------------------------------------
# Helper Functions
# --------------------------------------------------------

@st.cache_data
def load_data(uploaded_file):
    """
    Reads CSV/XLSX file into a pandas DataFrame, standardizes key columns,
    and sorts by week/store.
    """
    if uploaded_file is None:
        return pd.DataFrame()
    
    filename = uploaded_file.name.lower()
    try:
        if filename.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif filename.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(uploaded_file)
        else:
            st.error("Unsupported file type.")
            return pd.DataFrame()
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return pd.DataFrame()

    # Standardize column names
    df.columns = standardize_columns(df.columns)
    
    # Check for required columns
    required_cols = ['Store #', 'Engaged Transaction %']
    if not ('Date' in df.columns or 'Week' in df.columns):
        st.error("Data must contain 'Week' or 'Date' column.")
        return pd.DataFrame()
    
    if not all(col in df.columns for col in required_cols):
        st.error(f"Data must contain: {', '.join(required_cols)}.")
        return pd.DataFrame()
    
    # Parse dates
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.dropna(subset=['Date'])
        # Create Week from Date if it doesn't exist
        if 'Week' not in df.columns and pd.api.types.is_datetime64_any_dtype(df['Date']):
            df['Week'] = df['Date'].dt.isocalendar().week.astype(int)
    
    # Convert percentage columns to numeric
    percent_cols = ['Engaged Transaction %', 'Quarter to Date %']
    for col in percent_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace('%', '', regex=False), errors='coerce')
    
    # Drop rows with missing essential data
    essential_cols = ['Store #', 'Engaged Transaction %']
    if 'Week' in df.columns:
        essential_cols.append('Week')
    elif 'Date' in df.columns:
        essential_cols.append('Date')
    df = df.dropna(subset=essential_cols)
    
    # Convert data types
    if 'Weekly Rank' in df.columns:
        df['Weekly Rank'] = pd.to_numeric(df['Weekly Rank'], errors='coerce')
        if 'Weekly Rank' in df.columns and not df['Weekly Rank'].isna().all():
            df['Weekly Rank'] = df['Weekly Rank'].astype('Int64')
    
    if 'Store #' in df.columns:
        df['Store #'] = df['Store #'].astype(str)
    
    # Ensure Week is integer if present
    if 'Week' in df.columns:
        df['Week'] = pd.to_numeric(df['Week'], errors='coerce')
        df = df.dropna(subset=['Week'])
        df['Week'] = df['Week'].astype(int)
        df = df.sort_values(['Week', 'Store #'])
        # Create Quarter from Week if it doesn't exist
        if 'Quarter' not in df.columns:
            df['Quarter'] = ((df['Week'] - 1) // 13 + 1).astype(int)
    elif 'Date' in df.columns:
        df = df.sort_values(['Date', 'Store #'])
        # Create Quarter from Date if it doesn't exist
        if 'Quarter' not in df.columns and pd.api.types.is_datetime64_any_dtype(df['Date']):
            df['Quarter'] = df['Date'].dt.quarter
    
    return df


def standardize_columns(columns):
    """
    Renames columns to standard internal names for consistency.
    """
    new_cols = []
    processed_indices = set()
    cols_lower = [str(col).strip().lower() for col in columns]

    for i, col_lower in enumerate(cols_lower):
        if i in processed_indices:
            continue
        original_col = columns[i]

        if col_lower == 'store #':
            new_cols.append('Store #')
        elif 'store' in col_lower and ('#' in col_lower or 'id' in col_lower or 'number' in col_lower):
            new_cols.append('Store #')
        elif 'engaged transaction %' in col_lower:
            new_cols.append('Engaged Transaction %')
        elif 'engage' in col_lower and ('%' in col_lower or 'transaction' in col_lower):
            new_cols.append('Engaged Transaction %')
        elif ('week' in col_lower and 'ending' in col_lower) or col_lower == 'date' or col_lower == 'week ending':
            new_cols.append('Date')
        elif col_lower == 'week' or (col_lower.startswith('week') and not any(s in col_lower for s in ['rank', 'end', 'date'])):
            new_cols.append('Week')
        elif 'rank' in col_lower and 'week' in col_lower:
            new_cols.append('Weekly Rank')
        elif 'quarter' in col_lower or 'qtd' in col_lower:
            new_cols.append('Quarter to Date %')
        else:
            new_cols.append(original_col)
        
        processed_indices.add(i)

    if len(new_cols) != len(set(new_cols)):
        st.warning("Potential duplicate columns detected after standardization. Check headers.")
    
    return new_cols


def calculate_trend(group, window=4, engagement_col='Engaged Transaction %', week_col='Week'):
    """
    Calculates a trend label (Upward, Downward, etc.) based on the
    linear slope of the last 'window' data points.
    """
    if engagement_col not in group.columns or week_col not in group.columns:
        return "Stable"
    
    group = group.dropna(subset=[engagement_col])
    if len(group) < 2:
        return "Stable"
    
    sorted_data = group.sort_values(week_col, ascending=True).tail(window)
    if len(sorted_data) < 2:
        return "Insufficient Data"
    
    x = sorted_data[week_col].values
    y = sorted_data[engagement_col].values
    
    # Filter out NaN values
    valid_indices = ~np.isnan(y)
    x = x[valid_indices]
    y = y[valid_indices]
    
    if len(x) < 2:
        return "Insufficient Data"
    
    # Center X to avoid numeric issues
    x_mean = np.mean(x)
    x_centered = x - x_mean
    
    if np.sum(x_centered**2) == 0 or np.std(y) == 0:
        return "Stable"
    
    try:
        slope = np.sum(x_centered * (y - np.mean(y))) / np.sum(x_centered**2)
    except (np.linalg.LinAlgError, ValueError, FloatingPointError):
        return "Calculation Error"
    
    strong_threshold = 0.5
    mild_threshold = 0.1
    
    if slope > strong_threshold:
        return "Strong Upward"
    elif slope > mild_threshold:
        return "Upward"
    elif slope < -strong_threshold:
        return "Strong Downward"
    elif slope < -mild_threshold:
        return "Downward"
    else:
        return "Stable"


def find_anomalies(df, z_threshold=2.0, engagement_col='Engaged Transaction %', store_col='Store #', week_col='Week'):
    """
    Calculates week-over-week changes in Engaged Transaction % for each store
    and flags any changes whose Z-score exceeds the given threshold.
    Returns a DataFrame of anomalies with potential explanations.
    """
    if df.empty or not all(col in df.columns for col in [engagement_col, store_col, week_col]):
        return pd.DataFrame()
    
    anomalies_list = []
    df_sorted = df.sort_values([store_col, week_col])
    
    for store_id, grp in df_sorted.groupby(store_col):
        # Prepare data
        grp = grp.dropna(subset=[engagement_col, week_col])
        grp = grp.copy()
        grp['diffs'] = grp[engagement_col].diff()
        diffs = grp['diffs'].dropna()
        
        if len(diffs) < 2:
            continue
        
        mean_diff = diffs.mean()
        std_diff = diffs.std(ddof=0)
        
        if std_diff == 0 or pd.isna(std_diff):
            continue
        
        # Identify anomalies
        anomaly_candidates = grp[grp['diffs'].notna()]
        for idx, row in anomaly_candidates.iterrows():
            diff_val = row['diffs']
            z = (diff_val - mean_diff) / std_diff
            
            if abs(z) >= z_threshold:
                try:
                    current_pos_index = grp.index.get_loc(idx)
                    prev_row = grp.loc[grp.index[current_pos_index - 1]] if current_pos_index > 0 else pd.Series(dtype='object')
                except KeyError:
                    prev_row = pd.Series(dtype='object')
                
                week_cur = row.get(week_col)
                week_prev = prev_row.get(week_col)
                val_cur = row.get(engagement_col)
                val_prev = prev_row.get(engagement_col)  # Get prev value for explanation
                rank_cur = row.get('Weekly Rank')
                rank_prev = prev_row.get('Weekly Rank')
                
                # Generate explanation
                explanation = ""
                if diff_val >= 0:
                    explanation = "Engagement spiked significantly."
                    if pd.notna(rank_prev) and pd.notna(rank_cur) and rank_cur < rank_prev:
                        explanation += f" Rank improved from {int(rank_prev)} to {int(rank_cur)}."
                    elif pd.notna(val_prev):
                        explanation += f" Jumped from {val_prev:.2f}%."
                else:
                    explanation = "Sharp drop in engagement."
                    if pd.notna(rank_prev) and pd.notna(rank_cur) and rank_cur > rank_prev:
                        explanation += f" Rank dropped from {int(rank_prev)} to {int(rank_cur)}."
                    elif pd.notna(val_prev):
                        explanation += f" Dropped from {val_prev:.2f}%."
                
                anomalies_list.append({
                    'Store #': store_id,
                    'Week': int(week_cur) if pd.notna(week_cur) else None,
                    'Engaged Transaction %': round(val_cur, 2) if pd.notna(val_cur) else None,
                    'Change %pts': round(diff_val, 2) if pd.notna(diff_val) else None,
                    'Z-score': round(z, 2),
                    'Prev Week': int(week_prev) if pd.notna(week_prev) else None,
                    'Rank': int(rank_cur) if pd.notna(rank_cur) else None,
                    'Prev Rank': int(rank_prev) if pd.notna(rank_prev) else None,
                    'Possible Explanation': explanation
                })
    
    if not anomalies_list:
        return pd.DataFrame()
    
    anomalies_df = pd.DataFrame(anomalies_list)
    anomalies_df['Abs Z'] = anomalies_df['Z-score'].abs()
    return anomalies_df.sort_values('Abs Z', ascending=False).drop(columns=['Abs Z'])


# --------------------------------------------------------
# Altair Chart Theming
# --------------------------------------------------------
def publix_clean_theme():
    """
    Define a clean Publix-branded theme for Altair charts.
    """
    return {
        "config": {
            "background": CARD_BACKGROUND_COLOR,
            "title": {
                "anchor": "start",
                "color": TEXT_COLOR_DARK,
                "fontSize": 14,
                "fontWeight": "bold"
            },
            "axis": {
                "domainColor": BORDER_COLOR,
                "gridColor": BORDER_COLOR,
                "gridDash": [1, 3],
                "labelColor": TEXT_COLOR_MEDIUM,
                "labelFontSize": 10,
                "titleColor": TEXT_COLOR_MEDIUM,
                "titleFontSize": 11,
                "titlePadding": 10,
                "tickColor": BORDER_COLOR,
                "tickSize": 5,
            },
            "axisX": {
                "grid": False,
                "labelAngle": 0
            },
            "axisY": {
                "grid": True,
                "labelPadding": 5,
                "ticks": False
            },
            "header": {
                "labelFontSize": 11,
                "titleFontSize": 12,
                "labelColor": TEXT_COLOR_MEDIUM,
                "titleColor": TEXT_COLOR_DARK
            },
            "legend": None,
            "range": {
                "category": [PUBLIX_GREEN_BRIGHT, PUBLIX_GREEN_DARK, TEXT_COLOR_MEDIUM],
                "heatmap": "greens",
                "ramp": "greens",
            },
            "view": {"stroke": None},
            "line": {
                "stroke": PUBLIX_GREEN_BRIGHT,
                "strokeWidth": 2.5
            },
            "point": {
                "fill": PUBLIX_GREEN_BRIGHT,
                "size": 10,
                "stroke": CARD_BACKGROUND_COLOR,
                "strokeWidth": 0.5
            },
            "bar": {"fill": PUBLIX_GREEN_BRIGHT},
            "rule": {
                "stroke": TEXT_COLOR_MEDIUM,
                "strokeDash": [4, 4]
            },
            "text": {
                "color": TEXT_COLOR_DARK,
                "fontSize": 11
            },
            "area": {
                "fill": PUBLIX_GREEN_BRIGHT,
                "opacity": 0.15,
                "line": {
                    "stroke": PUBLIX_GREEN_BRIGHT,
                    "strokeWidth": 2
                }
            }
        }
    }

alt.themes.register("publix_clean", publix_clean_theme)
alt.themes.enable("publix_clean")

# --------------------------------------------------------
# Chart Helper Functions
# --------------------------------------------------------
def create_sparkline(data, y_col, y_title):
    """
    Creates a small sparkline chart for metrics.
    """
    if data is None or data.empty or y_col not in data.columns or 'Week' not in data.columns:
        return None
    
    if not isinstance(data, pd.DataFrame):
        try:
            data = pd.DataFrame(data)
        except Exception:
            return None
    
    line = alt.Chart(data).mark_line().encode(
        x=alt.X('Week:O', axis=None),
        y=alt.Y(f'{y_col}:Q', axis=None),
        tooltip=[
            alt.Tooltip('Week:O'),
            alt.Tooltip(f'{y_col}:Q', format='.2f', title=y_title)
        ]
    ).properties(width=80, height=30)
    
    area = line.mark_area()
    return area + line

def create_donut_chart(value, title="Engagement"):
    """
    Creates a donut chart for visualizing a percentage.
    """
    value = max(0, min(100, value)) if pd.notna(value) else 0  # Handle potential NaN
    source = pd.DataFrame({"category": [title, "Remaining"], "value": [value, 100 - value]})
    
    base = alt.Chart(source).encode(theta=alt.Theta("value:Q", stack=True))
    
    pie = base.mark_arc(outerRadius=60, innerRadius=45).encode(
        color=alt.Color(
            "category:N",
            scale=alt.Scale(
                domain=[title, "Remaining"],
                range=[PUBLIX_GREEN_BRIGHT, BORDER_COLOR]
            ),
            legend=None
        ),
        order=alt.Order("value:Q", sort="descending")
    )
    
    text = base.mark_text(
        radius=0,
        align='center',
        baseline='middle',
        fontSize=18,
        fontWeight='bold'
    ).encode(
        text=alt.condition(
            alt.datum.category == title,
            alt.Text("value:Q", format=".0f"),
            alt.value(" ")  # Ensure space if no value
        ),
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

# Custom CSS for Publix theming
st.markdown("""<style>
    body { font-family: sans-serif; }
    .stApp { background-color: """ + BACKGROUND_COLOR + """; }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] > div:first-child { 
        background-color: """ + PUBLIX_GREEN_DARK + """; 
        padding-top: 1rem; 
    }
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] label, 
    [data-testid="stSidebar"] .st-caption { 
        color: """ + TEXT_COLOR_LIGHT + """ !important; 
        font-weight: normal; 
    }
    [data-testid="stSidebar"] .st-emotion-cache-1bzkvze { 
        color: """ + TEXT_COLOR_LIGHT + """ !important; 
    }
    [data-testid="stSidebar"] small { 
        color: rgba(255, 255, 255, 0.7) !important; 
    }
    
    /* Sidebar inputs styling */
    [data-testid="stSidebar"] .stSelectbox > div, 
    [data-testid="stSidebar"] .stMultiselect > div { 
        background-color: rgba(255, 255, 255, 0.1); 
        border: none; 
        border-radius: 4px; 
    }
    [data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] > div, 
    [data-testid="stSidebar"] .stMultiselect div[data-baseweb="select"] > div { 
        color: """ + TEXT_COLOR_LIGHT + """; 
    }
    [data-testid="stSidebar"] .stSelectbox svg, 
    [data-testid="stSidebar"] .stMultiselect svg { 
        fill: """ + TEXT_COLOR_LIGHT + """; 
    }
    [data-testid="stSidebar"] .stMultiselect span[data-baseweb="tag"] svg { 
        fill: """ + TEXT_COLOR_LIGHT + """; 
    }
    
    /* Sidebar button styling */
    [data-testid="stSidebar"] .stButton>button { 
        background-color: transparent; 
        color: """ + TEXT_COLOR_LIGHT + """; 
        border: 1px solid rgba(255, 255, 255, 0.5); 
        font-weight: normal; 
        padding: 5px 10px; 
        margin-top: 0.5rem; 
    }
    [data-testid="stSidebar"] .stButton>button:hover { 
        background-color: rgba(255, 255, 255, 0.1); 
        border-color: """ + TEXT_COLOR_LIGHT + """; 
    }
    
    /* File uploader styling */
    [data-testid="stSidebar"] .stFileUploader button { 
        border: 1px dashed rgba(255, 255, 255, 0.5); 
        background-color: rgba(255, 255, 255, 0.05); 
        color: """ + TEXT_COLOR_LIGHT + """; 
    }
    [data-testid="stSidebar"] .stFileUploader button:hover { 
        border-color: """ + TEXT_COLOR_LIGHT + """; 
        background-color: rgba(255, 255, 255, 0.1); 
    }
    
    /* Main content styling */
    .main .block-container { 
        padding: 1.5rem 2rem 2rem 2rem; 
        max-width: 95%; 
    }
    
    /* Card styling */
    .metric-card { 
        background-color: """ + CARD_BACKGROUND_COLOR + """; 
        border: 1px solid """ + BORDER_COLOR + """; 
        border-radius: 8px; 
        padding: 1rem 1.2rem; 
        box-shadow: 0 1px 3px rgba(0,0,0,0.03); 
        height: 100%; 
        display: flex; 
        flex-direction: column; 
        justify-content: space-between; 
    }
    .chart-card { 
        background-color: """ + CARD_BACKGROUND_COLOR + """; 
        border: 1px solid """ + BORDER_COLOR + """; 
        border-radius: 8px; 
        padding: 1.2rem; 
        box-shadow: 0 1px 3px rgba(0,0,0,0.03); 
    }
    
    /* Metric styling */
    .stMetric { 
        background-color: transparent !important; 
        border: none !important; 
        padding: 0 !important; 
        text-align: left; 
        color: """ + TEXT_COLOR_DARK + """; 
        font-size: 1rem; 
        height: 100%; 
        display: flex; 
        flex-direction: column; 
    }
    .stMetric > label { 
        color: """ + TEXT_COLOR_MEDIUM + """ !important; 
        font-weight: normal !important; 
        font-size: 0.85em !important; 
        margin-bottom: 0.25rem; 
        order: 1; 
    }
    .stMetric > div:nth-of-type(1) { 
        color: """ + TEXT_COLOR_DARK + """ !important; 
        font-size: 2.0em !important; 
        font-weight: bold !important; 
        line-height: 1.1 !important; 
        margin-bottom: 0.25rem; 
        order: 0; 
    }
    .stMetric > div:nth-of-type(2) { 
        font-size: 0.85em !important; 
        font-weight: bold; 
        color: """ + TEXT_COLOR_MEDIUM + """ !important; 
        order: 2; 
        margin-top: auto; 
    }
    .stMetric .stMetricDelta { 
        padding-top: 5px; 
    }
    .stMetric .stMetricDelta span[style*="color: rgb(46, 125, 50)"] { 
        color: """ + PUBLIX_GREEN_BRIGHT + """ !important; 
    }
    .stMetric .stMetricDelta span[style*="color: rgb(198, 40, 40)"] { 
        color: #D32F2F !important; 
    }
    .stMetric .metric-sparkline { 
        margin-top: auto; 
        order: 3; 
        padding-top: 10px; 
        line-height: 0; 
        opacity: 0.7; 
    }
    
    /* Tab styling */
    div[data-baseweb="tab-list"] { 
        border-bottom: 2px solid """ + BORDER_COLOR + """; 
        padding-left: 0; 
        margin-bottom: 1.5rem; 
    }
    button[data-baseweb="tab"] { 
        background-color: transparent; 
        color: """ + TEXT_COLOR_MEDIUM + """; 
        padding: 0.6rem 0.1rem; 
        margin-right: 1.5rem; 
        border-bottom: 2px solid transparent; 
        font-weight: normal; 
    }
    button[data-baseweb="tab"]:hover { 
        background-color: transparent; 
        color: """ + TEXT_COLOR_DARK + """; 
        border-bottom-color: """ + BORDER_COLOR + """; 
    }
    button[data-baseweb="tab"][aria-selected="true"] { 
        color: """ + PUBLIX_GREEN_BRIGHT + """; 
        font-weight: bold; 
        border-bottom-color: """ + PUBLIX_GREEN_BRIGHT + """; 
    }
    
    /* Text formatting */
    h1, h2, h3 { 
        color: """ + TEXT_COLOR_DARK + """; 
        font-weight: bold; 
    }
    h3 { 
        margin-top: 1.5rem; 
        margin-bottom: 0.8rem; 
        font-size: 1.3rem; 
    }
    h4 { 
        margin-top: 1rem; 
        margin-bottom: 0.5rem; 
        font-size: 1.1rem; 
        color: """ + TEXT_COLOR_MEDIUM + """; 
        font-weight: bold; 
    }
    
    /* Expander styling */
    [data-testid="stExpander"] { 
        border: 1px solid """ + BORDER_COLOR + """; 
        border-radius: 6px; 
        background-color: """ + CARD_BACKGROUND_COLOR + """; 
    }
    [data-testid="stExpander"] summary { 
        font-weight: normal; 
        color: """ + TEXT_COLOR_MEDIUM + """; 
    }
    [data-testid="stExpander"] summary:hover { 
        color: """ + PUBLIX_GREEN_BRIGHT + """; 
    }
    
    /* Horizontal rule */
    hr { 
        border-top: 1px solid """ + BORDER_COLOR + """; 
        margin: 1.5rem 0; 
    }
</style>""", unsafe_allow_html=True)

# --------------------------------------------------------
# Sidebar Setup
# --------------------------------------------------------
st.sidebar.markdown(f"""<div style="text-align: center; padding-bottom: 1rem;">
<svg width="40" height="40" viewBox="0 0 100 100" fill="none" xmlns="http://www.w3.org/2000/svg">
<circle cx="50" cy="50" r="50" fill="{TEXT_COLOR_LIGHT}"/>
<path d="M63.7148 25H42.4297C35.1445 25 29.2852 30.8594 29.2852 38.1445V43.6055C29.2852 46.6523 31.7734 49.1406 34.8203 49.1406H46.0117V61.8555C46.0117 64.9023 48.4999 67.3906 51.5468 67.3906H57.1406C60.1875 67.3906 62.6757 64.9023 62.6757 61.8555V49.1406H66.4804C68.0234 49.1406 69.2851 47.8789 69.2851 46.3359V31.0195C69.2851 27.7148 66.7304 25 63.7148 25ZM57.1406 55.6641H51.5468V43.6055C51.5468
