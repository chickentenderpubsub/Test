import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import datetime
from typing import Optional, Dict, List, Any, Tuple

# --- Configuration Constants ---

APP_TITLE = "Club Publix Engagement Dashboard"
PAGE_LAYOUT = "wide"
INITIAL_SIDEBAR_STATE = "expanded"

# Column Names (Use these constants throughout the code)
COL_STORE_ID = 'Store #'
COL_WEEK = 'Week'
COL_DATE = 'Date'
COL_ENGAGED_PCT = 'Engaged Transaction %'
COL_QTD_PCT = 'Quarter to Date %'
COL_RANK = 'Weekly Rank'
COL_QUARTER = 'Quarter'
COL_MA_4W = 'MA_4W' # 4-week Moving Average
COL_PERIOD = 'Period' # For comparison data
COL_CATEGORY = 'Category' # Performance category
COL_TREND_CORR = 'Trend Correlation'
COL_AVG_ENGAGEMENT = 'Average Engagement'
COL_CONSISTENCY = 'Consistency'
COL_ACTION_PLAN = 'Action Plan'
COL_EXPLANATION = 'Explanation'
COL_CHANGE_PCT_PTS = 'Change %pts'
COL_Z_SCORE = 'Z-score'
COL_POSSIBLE_EXPLANATION = 'Possible Explanation'
COL_PREV_WEEK = 'Prev Week'
COL_PREV_RANK = 'Prev Rank'

# Potential input column names and their standardized mapping
# Case-insensitive matching will be used
INPUT_COLUMN_MAP_PATTERNS = {
    'quarter': COL_QTD_PCT,
    'qtd': COL_QTD_PCT,
    'rank': COL_RANK,
    'week ending': COL_DATE,
    'date': COL_DATE,
    'week': COL_WEEK, # If 'date'/'week ending' not found, look for 'week'
    'store': COL_STORE_ID,
    'engaged': COL_ENGAGED_PCT,
    'engagement': COL_ENGAGED_PCT,
}

# CSS Styles
APP_CSS = """
<style>
    .metric-card { background-color: #f5f5f5; border-radius: 10px; padding: 15px; box-shadow: 2px 2px 5px rgba(0,0,0,0.1); }
    .highlight-good { color: #2E7D32; font-weight: bold; }
    .highlight-bad { color: #C62828; font-weight: bold; }
    .highlight-neutral { color: #F57C00; font-weight: bold; }
    .dashboard-title { color: #1565C0; text-align: center; padding-bottom: 20px; }
    .caption-text { font-size: 0.85em; color: #555; }
    /* Style for category/trend cards */
    .info-card {
        background-color:#2C2C2C;
        padding:15px;
        border-radius:5px;
        margin-bottom:10px;
        border-left: 5px solid #ccc; /* Default border */
        color: #FFFFFF;
        height: 100%; /* Make cards in a row equal height */
        display: flex; /* Enable flexbox */
        flex-direction: column; /* Stack content vertically */
        justify-content: space-between; /* Distribute space */
    }
    .info-card h4 { margin-top: 0; margin-bottom: 5px; }
    .info-card p { margin: 5px 0; }
    .info-card .value { font-weight: bold; }
    .info-card .label { font-size: 0.9em; color: #BBBBBB; }
    .info-card .change { font-size: 0.9em; }
    .info-card .trend-icon { font-size: 1.2em; margin-right: 5px; }
    /* Ensure consistent height for metric cards in columns */
     div[data-testid="stMetric"] {
        /* Add styles if needed, e.g., min-height */
    }
</style>
"""

# Analysis Defaults & Settings
DEFAULT_Z_THRESHOLD = 2.0
DEFAULT_TREND_WINDOW = 4
MIN_TREND_WINDOW = 3
MAX_TREND_WINDOW = 8
DEFAULT_SHOW_MA = True
RECENT_TRENDS_WINDOW = 4 # Default weeks for recent trend analysis
RECENT_TRENDS_SENSITIVITY_MAP = {"Low": 0.5, "Medium": 0.3, "High": 0.2}

# Performance Categories
CAT_STAR = "Star Performer"
CAT_STABILIZE = "Needs Stabilization"
CAT_IMPROVING = "Improving"
CAT_INTERVENTION = "Requires Intervention"
CAT_UNCATEGORIZED = "Uncategorized"

PERFORMANCE_CATEGORIES = {
    CAT_STAR: {"icon": "⭐", "color": "#2E7D32", "explanation": "High engagement with stable or improving trend", "action": "Maintain current strategies. Document and share best practices with other stores.", "short_action": "Share best practices"},
    CAT_STABILIZE: {"icon": "⚠️", "color": "#F57C00", "explanation": "High engagement but recent downward trend", "action": "Investigate recent changes or inconsistencies. Reinforce processes to prevent decline.", "short_action": "Reinforce successful processes"},
    CAT_IMPROVING: {"icon": "📈", "color": "#1976D2", "explanation": "Below average engagement but trending upward", "action": "Continue positive momentum. Intensify efforts driving improvement.", "short_action": "Continue positive momentum"},
    CAT_INTERVENTION: {"icon": "🚨", "color": "#C62828", "explanation": "Below average engagement with flat or declining trend", "action": "Urgent attention needed. Develop a comprehensive improvement plan.", "short_action": "Needs comprehensive support"},
    CAT_UNCATEGORIZED: {"icon": "❓", "color": "#757575", "explanation": "Not enough data or unusual pattern", "action": "Monitor closely. Ensure data quality.", "short_action": "Monitor closely"},
}

# Trend Classification Thresholds
TREND_STRONG_UP = 0.5
TREND_UP = 0.1
TREND_DOWN = -0.1
TREND_STRONG_DOWN = -0.5

# Altair Chart Settings
CHART_HEIGHT_DEFAULT = 400
CHART_HEIGHT_TALL = 500
CHART_HEIGHT_SHORT = 300
HEATMAP_ROW_HEIGHT = 20
COMPARISON_BAR_HEIGHT = 25
COLOR_SCHEME_OPTIONS = ["Blues", "Greens", "Purples", "Oranges", "Reds", "Viridis"]
DEFAULT_COLOR_SCHEME = "blues"

# --- Helper Functions ---

def safe_mean(series: pd.Series) -> Optional[float]:
    """Calculate mean of a series, handling potential empty or all-NaN series."""
    if series is None or series.empty or series.isna().all():
        return None
    return series.mean()

def safe_max(series: pd.Series) -> Optional[Any]:
    """Get max value of a series, handling potential empty or all-NaN series."""
    if series is None or series.empty or series.isna().all():
        return None
    return series.max()

def safe_min(series: pd.Series) -> Optional[Any]:
    """Get min value of a series, handling potential empty or all-NaN series."""
    if series is None or series.empty or series.isna().all():
        return None
    return series.min()

def safe_idxmax(series: pd.Series) -> Optional[Any]:
    """Get index of max value, handling potential empty or all-NaN series."""
    if series is None or series.empty or series.isna().all():
        return None
    try:
        return series.idxmax()
    except ValueError: # Handle case where all values are NaN
        return None

def safe_idxmin(series: pd.Series) -> Optional[Any]:
    """Get index of min value, handling potential empty or all-NaN series."""
    if series is None or series.empty or series.isna().all():
        return None
    try:
        return series.idxmin()
    except ValueError: # Handle case where all values are NaN
        return None

def format_delta(value: Optional[float], unit: str = "%") -> str:
    """Format a delta value with sign and unit."""
    if value is None or pd.isna(value):
        return "N/A"
    return f"{value:+.2f}{unit}"

def format_percentage(value: Optional[float]) -> str:
    """Format a value as a percentage string."""
    if value is None or pd.isna(value):
        return "N/A"
    return f"{value:.2f}%"

# --- Data Loading and Preprocessing ---

@st.cache_data
def read_data_file(uploaded_file) -> Optional[pd.DataFrame]:
    """Reads an uploaded CSV or Excel file into a DataFrame."""
    if uploaded_file is None:
        return None
    try:
        name = uploaded_file.name.lower()
        if name.endswith('.csv'):
            # Try to sniff encoding, default to utf-8, fallback to latin1
            try:
                df = pd.read_csv(uploaded_file, encoding='utf-8')
            except UnicodeDecodeError:
                uploaded_file.seek(0) # Reset file pointer
                df = pd.read_csv(uploaded_file, encoding='latin1')
        elif name.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(uploaded_file)
        else:
            st.error(f"Unsupported file type: {uploaded_file.name}. Please upload CSV or Excel.")
            return None
        return df
    except Exception as e:
        st.error(f"Error reading file '{uploaded_file.name}': {e}")
        return None

def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Standardizes column names based on predefined patterns."""
    df.columns = [str(col).strip() for col in df.columns] # Ensure col names are strings
    col_map = {}
    current_cols = df.columns.tolist()
    mapped_cols = set() # Track which target columns have been mapped

    # Define mapping priorities (e.g., 'week ending' before generic 'week')
    patterns_priority = [
        ('week ending', COL_DATE),
        ('date', COL_DATE),
        ('store', COL_STORE_ID),
        ('engaged', COL_ENGAGED_PCT),
        ('engagement', COL_ENGAGED_PCT),
        ('rank', COL_RANK),
        ('quarter', COL_QTD_PCT),
        ('qtd', COL_QTD_PCT),
        ('week', COL_WEEK) # Lower priority for generic 'week'
    ]

    # Apply mappings based on priority
    for pattern, target_col in patterns_priority:
        if target_col in mapped_cols: continue # Skip if target already mapped

        for col in current_cols:
            if col in col_map: continue # Skip if source column already mapped

            cl = col.lower()
            # Check if pattern is in the lowercased column name
            if pattern in cl:
                 # Specific exclusion for 'week' to avoid mapping 'weekly rank'
                 if pattern == 'week' and ('rank' in cl or 'ending' in cl):
                     continue
                 # Map if target not already mapped
                 if target_col not in mapped_cols:
                     col_map[col] = target_col
                     mapped_cols.add(target_col)
                     # Once a target is mapped, break inner loop if pattern allows only one source
                     # (e.g., don't map multiple columns to 'Store #')
                     # For now, allow multiple sources to potentially map if names overlap,
                     # but the first one found based on priority wins for the target.
                     # This logic might need refinement based on actual data variations.


    df_renamed = df.rename(columns=col_map)

    # Check for essential columns after renaming
    missing_essentials = []
    if COL_STORE_ID not in df_renamed.columns: missing_essentials.append('Store ID')
    if COL_ENGAGED_PCT not in df_renamed.columns: missing_essentials.append('Engagement %')
    if COL_DATE not in df_renamed.columns and COL_WEEK not in df_renamed.columns: missing_essentials.append('Date or Week')

    if missing_essentials:
         st.warning(f"Could not find columns for: {', '.join(missing_essentials)}. Please check input file headers.")
         # Return the renamed df anyway, subsequent steps will handle missing columns
         return df_renamed


    return df_renamed

def preprocess_data(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Performs type conversions, cleaning, and adds derived columns."""

    # --- Type Conversion ---
    # Convert percentages first
    for percent_col in [COL_ENGAGED_PCT, COL_QTD_PCT]:
        if percent_col in df.columns:
            # Convert to string, remove '%', handle potential errors, then convert to numeric
            # Replace non-breaking spaces and other potential whitespace issues
            s = df[percent_col].astype(str).str.replace('%', '', regex=False).str.replace(r'\s+', '', regex=True)
            df[percent_col] = pd.to_numeric(s, errors='coerce')
            # Optional: Check for unusually large/small percentages (e.g., > 100)
            # if (df[percent_col] > 100).any() or (df[percent_col] < 0).any():
            #     st.warning(f"Found potential invalid values (outside 0-100) in '{percent_col}'.")

    # Ensure essential column exists and drop rows where it's NaN AFTER conversion
    if COL_ENGAGED_PCT not in df.columns:
         # Standardization should have warned, but double-check
         st.error(f"Essential column '{COL_ENGAGED_PCT}' not found.")
         return None
    initial_rows = len(df)
    df = df.dropna(subset=[COL_ENGAGED_PCT])
    if df.empty:
        st.warning(f"No valid data remaining after removing rows with missing '{COL_ENGAGED_PCT}'. (Removed {initial_rows} rows).")
        return None
    if len(df) < initial_rows:
         st.caption(f"Removed {initial_rows - len(df)} rows with missing '{COL_ENGAGED_PCT}'.")


    # --- Handle Date/Week and Quarter derivation ---
    if COL_DATE in df.columns:
        df[COL_DATE] = pd.to_datetime(df[COL_DATE], errors='coerce')
        df = df.dropna(subset=[COL_DATE])
        if df.empty:
             st.warning("No valid data remaining after handling dates.")
             return None
        # Use isocalendar for potentially more standard week definition
        df[COL_WEEK] = df[COL_DATE].dt.isocalendar().week.astype(int)
        df[COL_QUARTER] = df[COL_DATE].dt.quarter.astype(int)
    elif COL_WEEK in df.columns:
        # Ensure Week is numeric integer
        df[COL_WEEK] = pd.to_numeric(df[COL_WEEK], errors='coerce')
        df = df.dropna(subset=[COL_WEEK])
        if df.empty:
             st.warning("No valid data remaining after handling week numbers.")
             return None
        df[COL_WEEK] = df[COL_WEEK].astype(int)
        # Derive Quarter from Week (approximate, assumes standard fiscal year)
        df[COL_QUARTER] = ((df[COL_WEEK] - 1) // 13 + 1).astype(int)
        # Validate derived quarter is within 1-4
        df[COL_QUARTER] = df[COL_QUARTER].apply(lambda x: x if 1 <= x <= 4 else pd.NA)
        df = df.dropna(subset=[COL_QUARTER]) # Remove rows with invalid derived quarter
        df[COL_QUARTER] = df[COL_QUARTER].astype(int)
        if df.empty:
             st.warning("No valid data remaining after deriving quarters from week numbers.")
             return None
    else:
        # Standardization should have warned
        st.error(f"Missing required time column: Neither '{COL_DATE}' nor '{COL_WEEK}' found.")
        return None

    # --- Convert other columns ---
    if COL_RANK in df.columns:
        df[COL_RANK] = pd.to_numeric(df[COL_RANK], errors='coerce').astype('Int64') # Use nullable Int
    if COL_STORE_ID in df.columns:
        # Trim whitespace from store IDs
        df[COL_STORE_ID] = df[COL_STORE_ID].astype(str).str.strip()
    else:
         st.error(f"Missing required '{COL_STORE_ID}' column.")
         return None

    # --- Final Checks & Sort ---
    # Check for duplicate entries for the same store in the same week
    duplicates = df.duplicated(subset=[COL_STORE_ID, COL_WEEK], keep=False)
    if duplicates.any():
        st.warning(f"Found {duplicates.sum()} duplicate entries for the same store and week. Consider cleaning the input data. Using the first occurrence.")
        df = df.drop_duplicates(subset=[COL_STORE_ID, COL_WEEK], keep='first')


    df = df.sort_values([COL_STORE_ID, COL_WEEK]) # Sort by store then week
    return df

def load_and_process_data(uploaded_file) -> Optional[pd.DataFrame]:
    """Orchestrates the data loading and preprocessing steps."""
    df = read_data_file(uploaded_file)
    if df is None:
        return None
    df_standardized = standardize_columns(df.copy())
    df_processed = preprocess_data(df_standardized)
    return df_processed

# --- Filtering Logic ---

def filter_dataframe(df: Optional[pd.DataFrame], quarter_choice: str, week_choice: str, store_choice: List[str]) -> pd.DataFrame:
    """Filters the DataFrame based on sidebar selections."""
    if df is None or df.empty:
        return pd.DataFrame() # Return empty DataFrame if input is invalid

    df_filtered = df.copy()

    # Filter by Quarter
    if quarter_choice != "All":
        if COL_QUARTER in df_filtered.columns:
            try:
                q_num = int(quarter_choice.replace('Q', ''))
                df_filtered = df_filtered[df_filtered[COL_QUARTER] == q_num]
            except ValueError:
                st.warning(f"Invalid Quarter selection: {quarter_choice}")
        else:
            st.warning("Quarter column not available for filtering.")

    # Filter by Week
    if week_choice != "All":
         if COL_WEEK in df_filtered.columns:
            try:
                week_num = int(week_choice)
                df_filtered = df_filtered[df_filtered[COL_WEEK] == week_num]
            except ValueError:
                st.warning(f"Invalid Week selection: {week_choice}")
         else:
             st.warning("Week column not available for filtering.")

    # Filter by Store
    if store_choice: # If list is not empty
        if COL_STORE_ID in df_filtered.columns:
            df_filtered = df_filtered[df_filtered[COL_STORE_ID].isin(store_choice)]
        else:
            st.warning("Store ID column not available for filtering.")

    return df_filtered

# --- Analysis Functions ---

def calculate_trend_slope(group: pd.DataFrame, window: int) -> float:
    """
    Calculates the slope of the engagement trend over a specified window
    using linear regression. Handles potential issues with insufficient data,
    NaNs, or constant values.
    """
    required_cols = [COL_WEEK, COL_ENGAGED_PCT]
    if not all(col in group.columns for col in required_cols): return 0.0

    # Ensure data is sorted by week within the group
    data = group.sort_values(COL_WEEK).tail(window)

    # Check for sufficient data points
    if len(data) < 2: return 0.0

    # Prepare x (week) and y (engagement) values
    x = data[COL_WEEK].values
    y = data[COL_ENGAGED_PCT].values

    # Check for NaNs in y (should be handled by preprocessing, but double-check)
    if pd.isna(y).any(): return 0.0

    # Check for constant x or y values (polyfit can fail or give misleading results)
    if len(set(x)) < 2 or len(set(y)) < 2: return 0.0

    # Center x values for potentially better numerical stability
    x_centered = x - np.mean(x)

    try:
        # Calculate slope using polyfit (degree 1 for linear)
        slope = np.polyfit(x_centered, y, 1)[0]
        # Return slope, ensuring it's not NaN
        return slope if pd.notna(slope) else 0.0
    except (np.linalg.LinAlgError, ValueError):
        # Handle cases where polyfit might fail (e.g., singular matrix)
        return 0.0


def classify_trend(slope: float) -> str:
    """Classifies trend based on slope value using predefined thresholds."""
    if slope > TREND_STRONG_UP: return "Strong Upward"
    elif slope > TREND_UP: return "Upward"
    elif slope < TREND_STRONG_DOWN: return "Strong Downward"
    elif slope < TREND_DOWN: return "Downward"
    else: return "Stable"

def calculate_store_trends(df: pd.DataFrame, window: int) -> pd.Series:
    """Calculates the trend classification for each store over the given period."""
    required_cols = [COL_STORE_ID, COL_WEEK, COL_ENGAGED_PCT]
    if df.empty or not all(col in df.columns for col in required_cols):
        return pd.Series(dtype=str)

    # Group by store and apply the slope calculation
    store_slopes = df.groupby(COL_STORE_ID).apply(lambda g: calculate_trend_slope(g, window))

    # Classify the trend based on the calculated slopes
    store_trends = store_slopes.apply(classify_trend)
    return store_trends

def get_executive_summary_data(df_filtered: pd.DataFrame, df_all: pd.DataFrame, store_choice: List[str], all_stores_list: List[str], trend_window: int) -> Dict[str, Any]:
    """Calculates all metrics needed for the executive summary section."""
    summary = {
        'current_week': None, 'prev_week': None, 'current_avg': None, 'prev_avg': None,
        'top_store': None, 'top_val': None, 'bottom_store': None, 'bottom_val': None,
        'avg_label': "District Avg Engagement", 'store_trends': pd.Series(dtype=str),
        'delta_val': None, 'trend_dir': 'flat', 'trend_class': 'highlight-neutral'
    }
    required_cols = [COL_WEEK, COL_ENGAGED_PCT, COL_STORE_ID]
    if df_filtered.empty or not all(col in df_filtered.columns for col in required_cols):
        return summary

    # Determine current and previous week within the filtered data
    available_weeks = sorted(df_filtered[COL_WEEK].unique())
    if not available_weeks: return summary

    summary['current_week'] = int(available_weeks[-1])
    if len(available_weeks) > 1:
        summary['prev_week'] = int(available_weeks[-2])
    else:
        # If only one week filtered, look for previous week in the *unfiltered* data
        # Consider quarter context if available
        current_quarter = df_filtered[COL_QUARTER].iloc[0] if COL_QUARTER in df_filtered.columns and not df_filtered.empty else None
        potential_prev_weeks = df_all[df_all[COL_WEEK] < summary['current_week']]
        if current_quarter and COL_QUARTER in potential_prev_weeks.columns:
            potential_prev_weeks = potential_prev_weeks[potential_prev_weeks[COL_QUARTER] == current_quarter]

        if not potential_prev_weeks.empty:
             summary['prev_week'] = int(potential_prev_weeks[COL_WEEK].max())


    # Calculate averages for the determined weeks
    current_data = df_filtered[df_filtered[COL_WEEK] == summary['current_week']]
    summary['current_avg'] = safe_mean(current_data[COL_ENGAGED_PCT])

    if summary['prev_week'] is not None:
        # Use df_all to get previous week data, then apply store filter if needed
        prev_data_all = df_all[df_all[COL_WEEK] == summary['prev_week']]
        prev_data_filtered = filter_dataframe(prev_data_all, "All", str(summary['prev_week']), store_choice) # Apply store filter
        summary['prev_avg'] = safe_mean(prev_data_filtered[COL_ENGAGED_PCT])


    # Calculate Top/Bottom Performers for the current week from filtered data
    if not current_data.empty:
        store_perf_current = current_data.groupby(COL_STORE_ID)[COL_ENGAGED_PCT].mean()
        summary['top_store'] = safe_idxmax(store_perf_current)
        summary['bottom_store'] = safe_idxmin(store_perf_current)
        summary['top_val'] = safe_max(store_perf_current)
        summary['bottom_val'] = safe_min(store_perf_current)

    # Calculate Trends using all data in the filtered period
    summary['store_trends'] = calculate_store_trends(df_filtered, trend_window)

    # Determine Average Label based on store selection
    if store_choice and len(store_choice) == 1:
        summary['avg_label'] = f"Store {store_choice[0]} Engagement"
    elif store_choice and len(store_choice) < len(all_stores_list):
        summary['avg_label'] = "Selected Stores Avg Engagement"
    # Default is "District Avg Engagement"

    # Calculate Delta and Trend Direction/Class for the average
    if summary['current_avg'] is not None and summary['prev_avg'] is not None:
        summary['delta_val'] = summary['current_avg'] - summary['prev_avg']
        # Add a small tolerance to avoid classifying tiny changes
        if summary['delta_val'] > 0.01:
            summary['trend_dir'] = "up"
            summary['trend_class'] = "highlight-good"
        elif summary['delta_val'] < -0.01:
            summary['trend_dir'] = "down"
            summary['trend_class'] = "highlight-bad"
        # Default is 'flat' / 'highlight-neutral'

    return summary

def generate_key_insights(df_filtered: pd.DataFrame, store_trends: pd.Series, store_perf: pd.Series) -> List[str]:
    """Generates a list of key insight strings based on filtered data."""
    insights = []
    required_cols = [COL_STORE_ID, COL_ENGAGED_PCT]
    if df_filtered.empty or not all(col in df_filtered.columns for col in required_cols):
        return ["No data available for insights."]

    # 1. Consistency Insights (Standard Deviation)
    store_std = df_filtered.groupby(COL_STORE_ID)[COL_ENGAGED_PCT].std().fillna(0)
    if not store_std.empty and len(store_std) > 1: # Need >1 store for comparison
        most_consistent_store = safe_idxmin(store_std)
        least_consistent_store = safe_idxmax(store_std)
        if most_consistent_store is not None:
            insights.append(f"**Store {most_consistent_store}** shows the most consistent engagement (std: {store_std[most_consistent_store]:.2f}%).")
        if least_consistent_store is not None and least_consistent_store != most_consistent_store:
            insights.append(f"**Store {least_consistent_store}** has the most variable engagement (std: {store_std[least_consistent_store]:.2f}%).")

    # 2. Trending Stores Insights (Based on calculated trends for the period)
    if not store_trends.empty:
        trending_up = store_trends[store_trends.isin(["Upward", "Strong Upward"])].index.tolist()
        trending_down = store_trends[store_trends.isin(["Downward", "Strong Downward"])].index.tolist()
        if trending_up:
            insights.append("Stores showing positive trends over the period: " + ", ".join(f"**{s}**" for s in trending_up))
        if trending_down:
            insights.append("Stores needing attention (downward trend over period): " + ", ".join(f"**{s}**" for s in trending_down))

    # 3. Performance Gap Insights (Based on average performance over the period)
    if not store_perf.empty and len(store_perf) > 1:
        top_val = safe_max(store_perf)
        bottom_val = safe_min(store_perf)
        if top_val is not None and bottom_val is not None:
            gap = top_val - bottom_val
            insights.append(f"Performance gap between highest ({top_val:.2f}%) and lowest ({bottom_val:.2f}%) stores (avg over period): **{gap:.2f}%**")
            if gap > 10: # Arbitrary threshold for large gap warning
                insights.append("🚨 Large performance gap suggests opportunities for knowledge sharing from top performers.")

    if not insights:
         return ["No specific insights generated from the current data selection."]

    return insights[:5] # Limit to top 5 insights

def calculate_moving_averages(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates the 4-week moving average for engagement for each store."""
    if df is None or df.empty:
        return pd.DataFrame(columns=df.columns.tolist() + [COL_MA_4W] if df is not None else [COL_MA_4W])

    required_cols = [COL_STORE_ID, COL_WEEK, COL_ENGAGED_PCT]
    # Ensure the column exists even if calculation fails or no data
    if COL_MA_4W not in df.columns:
        df[COL_MA_4W] = np.nan

    if not all(col in df.columns for col in required_cols):
        # Return df with NaN MA column if required cols missing
        return df

    # Sort before calculating rolling average
    df = df.sort_values([COL_STORE_ID, COL_WEEK])

    # Calculate MA within each store group
    df[COL_MA_4W] = df.groupby(COL_STORE_ID, group_keys=False)[COL_ENGAGED_PCT].apply(
        lambda s: s.rolling(window=DEFAULT_TREND_WINDOW, min_periods=1).mean()
    )
    return df

def calculate_district_trends(df: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Calculates the district average and its moving average."""
    default_cols = [COL_WEEK, 'Average Engagement %', COL_MA_4W]
    if df is None or df.empty:
        return pd.DataFrame(columns=default_cols)

    required_cols = [COL_WEEK, COL_ENGAGED_PCT]
    if not all(col in df.columns for col in required_cols):
        return pd.DataFrame(columns=default_cols)

    # Calculate weekly average across selected stores
    dist_trend = df.groupby(COL_WEEK, as_index=False)[COL_ENGAGED_PCT].mean()
    dist_trend = dist_trend.rename(columns={COL_ENGAGED_PCT: 'Average Engagement %'})
    dist_trend = dist_trend.sort_values(COL_WEEK)

    # Calculate moving average of the district average
    dist_trend[COL_MA_4W] = dist_trend['Average Engagement %'].rolling(window=DEFAULT_TREND_WINDOW, min_periods=1).mean()
    return dist_trend


def calculate_performance_categories(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates performance stats and assigns categories to stores based on filtered data."""
    required_cols = [COL_STORE_ID, COL_WEEK, COL_ENGAGED_PCT]
    output_cols = [COL_STORE_ID, COL_AVG_ENGAGEMENT, COL_CONSISTENCY, COL_TREND_CORR, COL_CATEGORY, COL_ACTION_PLAN, COL_EXPLANATION]
    if df.empty or not all(col in df.columns for col in required_cols):
        return pd.DataFrame(columns=output_cols)

    # Calculate Mean and Std Dev for each store over the period
    store_stats = df.groupby(COL_STORE_ID)[COL_ENGAGED_PCT].agg(['mean', 'std']).reset_index()
    store_stats.columns = [COL_STORE_ID, COL_AVG_ENGAGEMENT, COL_CONSISTENCY]
    # Fill NaN std dev (for stores with 1 data point) with 0
    store_stats[COL_CONSISTENCY] = store_stats[COL_CONSISTENCY].fillna(0.0)

    # Calculate trend correlation (simple linear correlation between week and engagement)
    def safe_corr(group):
        # Ensure sufficient data points and variance for correlation
        if len(group) < 3 or group[COL_WEEK].nunique() < 2 or group[COL_ENGAGED_PCT].nunique() < 2 or group[COL_ENGAGED_PCT].isna().any() or group[COL_WEEK].isna().any():
            return 0.0
        try:
             # Use pandas correlation method directly
             corr_val = group[COL_WEEK].corr(group[COL_ENGAGED_PCT])
             return corr_val if pd.notna(corr_val) else 0.0
        except Exception:
             # Catch any unexpected errors during correlation calculation
             return 0.0

    trend_corr = df.groupby(COL_STORE_ID).apply(safe_corr)
    store_stats[COL_TREND_CORR] = store_stats[COL_STORE_ID].map(trend_corr).fillna(0.0)

    # Assign categories based on median engagement and trend correlation
    median_engagement = store_stats[COL_AVG_ENGAGEMENT].median()
    # Handle case where median might be NaN if all averages are NaN
    if pd.isna(median_engagement):
         # If median can't be calculated, categorization is difficult. Assign Uncategorized.
         st.warning("Could not calculate median engagement. Store categorization may be inaccurate.")
         store_stats[COL_CATEGORY] = CAT_UNCATEGORIZED
    else:
        conditions = [
            # Note: Order matters here. More specific conditions first.
            (store_stats[COL_AVG_ENGAGEMENT] >= median_engagement) & (store_stats[COL_TREND_CORR] < TREND_DOWN), # High Perf, Declining Trend -> Stabilize
            (store_stats[COL_AVG_ENGAGEMENT] >= median_engagement),                                         # High Perf, Stable/Improving Trend -> Star
            (store_stats[COL_AVG_ENGAGEMENT] < median_engagement) & (store_stats[COL_TREND_CORR] > TREND_UP),   # Low Perf, Improving Trend -> Improving
            (store_stats[COL_AVG_ENGAGEMENT] < median_engagement)                                          # Low Perf, Stable/Declining Trend -> Intervention
        ]
        choices = [CAT_STABILIZE, CAT_STAR, CAT_IMPROVING, CAT_INTERVENTION]
        store_stats[COL_CATEGORY] = np.select(conditions, choices, default=CAT_UNCATEGORIZED).astype(str)

    # Map explanations and action plans using .get for safety
    store_stats[COL_ACTION_PLAN] = store_stats[COL_CATEGORY].map(lambda x: PERFORMANCE_CATEGORIES.get(x, {}).get('action', 'N/A'))
    store_stats[COL_EXPLANATION] = store_stats[COL_CATEGORY].map(lambda x: PERFORMANCE_CATEGORIES.get(x, {}).get('explanation', 'N/A'))

    # Ensure all output columns exist
    for col in output_cols:
        if col not in store_stats.columns:
            store_stats[col] = pd.NA # Or appropriate default

    return store_stats[output_cols] # Return with defined columns


def find_anomalies(df: pd.DataFrame, z_threshold: float) -> pd.DataFrame:
    """Detects anomalies based on week-over-week changes using Z-score within the filtered data."""
    required_cols = [COL_STORE_ID, COL_WEEK, COL_ENGAGED_PCT]
    output_cols = [COL_STORE_ID, COL_WEEK, COL_ENGAGED_PCT, COL_CHANGE_PCT_PTS, COL_Z_SCORE, COL_PREV_WEEK, COL_RANK, COL_PREV_RANK, COL_POSSIBLE_EXPLANATION]
    if df.empty or not all(col in df.columns for col in required_cols):
        return pd.DataFrame(columns=output_cols)

    anomalies = []
    df = df.sort_values([COL_STORE_ID, COL_WEEK])

    for store_id, grp in df.groupby(COL_STORE_ID):
        # Calculate differences within the group
        # Ensure Engagement Pct is numeric before diff
        grp[COL_ENGAGED_PCT] = pd.to_numeric(grp[COL_ENGAGED_PCT], errors='coerce')
        diffs = grp[COL_ENGAGED_PCT].diff() # Keep first NaN

        # Need at least 3 data points (2 differences) to calculate std dev reliably
        valid_diffs = diffs.dropna()
        if len(valid_diffs) < 2: continue

        mean_diff = valid_diffs.mean()
        std_diff = valid_diffs.std()

        # Avoid division by zero or near-zero std dev
        if std_diff == 0 or pd.isna(std_diff) or std_diff < 1e-6: continue

        # Iterate through differences, starting from the second row (index 1 in the group's context)
        grp_reset = grp.reset_index() # Use 0-based index for iteration
        diffs_reset = grp_reset[COL_ENGAGED_PCT].diff()

        for i in range(1, len(grp_reset)):
            diff_val = diffs_reset.iloc[i]
            if pd.isna(diff_val): continue

            z = (diff_val - mean_diff) / std_diff

            if abs(z) >= z_threshold:
                current_row = grp_reset.iloc[i]
                prev_row = grp_reset.iloc[i-1]

                # Safely get rank values, default to None if column missing or value NaN
                rank_cur = int(current_row[COL_RANK]) if COL_RANK in current_row and pd.notna(current_row[COL_RANK]) else None
                rank_prev = int(prev_row[COL_RANK]) if COL_RANK in prev_row and pd.notna(prev_row[COL_RANK]) else None

                anomaly_record = {
                    COL_STORE_ID: store_id,
                    COL_WEEK: int(current_row[COL_WEEK]),
                    COL_ENGAGED_PCT: current_row[COL_ENGAGED_PCT],
                    COL_CHANGE_PCT_PTS: diff_val,
                    COL_Z_SCORE: z,
                    COL_PREV_WEEK: int(prev_row[COL_WEEK]),
                    COL_RANK: rank_cur,
                    COL_PREV_RANK: rank_prev,
                }
                anomalies.append(anomaly_record)

    if not anomalies:
        return pd.DataFrame(columns=output_cols)

    anomalies_df = pd.DataFrame(anomalies)

    # Add explanations based on change direction
    anomalies_df[COL_POSSIBLE_EXPLANATION] = np.where(
        anomalies_df[COL_CHANGE_PCT_PTS] >= 0,
        "Engagement spiked significantly vs store's typical weekly change.",
        "Sharp drop in engagement vs store's typical weekly change."
    )

    # Add rank change details if rank columns exist and values are not NaN
    if COL_RANK in anomalies_df.columns and COL_PREV_RANK in anomalies_df.columns:
        improve_mask = (anomalies_df[COL_CHANGE_PCT_PTS] >= 0) & anomalies_df[COL_PREV_RANK].notna() & anomalies_df[COL_RANK].notna() & (anomalies_df[COL_PREV_RANK] > anomalies_df[COL_RANK])
        decline_mask = (anomalies_df[COL_CHANGE_PCT_PTS] < 0) & anomalies_df[COL_PREV_RANK].notna() & anomalies_df[COL_RANK].notna() & (anomalies_df[COL_PREV_RANK] < anomalies_df[COL_RANK])

        # Use .loc for safe assignment
        if improve_mask.any():
            anomalies_df.loc[improve_mask, COL_POSSIBLE_EXPLANATION] += " Rank improved from " + anomalies_df.loc[improve_mask, COL_PREV_RANK].astype(int).astype(str) + " to " + anomalies_df.loc[improve_mask, COL_RANK].astype(int).astype(str) + "."
        if decline_mask.any():
            anomalies_df.loc[decline_mask, COL_POSSIBLE_EXPLANATION] += " Rank dropped from " + anomalies_df.loc[decline_mask, COL_PREV_RANK].astype(int).astype(str) + " to " + anomalies_df.loc[decline_mask, COL_RANK].astype(int).astype(str) + "."


    # Sort by absolute Z-score and format numeric columns
    if not anomalies_df.empty:
        anomalies_df['Abs Z'] = anomalies_df[COL_Z_SCORE].abs()
        anomalies_df = anomalies_df.sort_values('Abs Z', ascending=False).drop(columns=['Abs Z'])
        anomalies_df[[COL_ENGAGED_PCT, COL_Z_SCORE, COL_CHANGE_PCT_PTS]] = anomalies_df[[COL_ENGAGED_PCT, COL_Z_SCORE, COL_CHANGE_PCT_PTS]].round(2)

    # Ensure all expected output columns exist
    for col in output_cols:
        if col not in anomalies_df.columns:
             anomalies_df[col] = pd.NA # Or appropriate default

    return anomalies_df[output_cols]


def generate_recommendations(df_filtered: pd.DataFrame, store_stats: pd.DataFrame, anomalies_df: pd.DataFrame, trend_window: int) -> pd.DataFrame:
    """Generates store-specific recommendations based on category, trend, and anomalies."""
    recommendations = []
    output_cols = [COL_STORE_ID, COL_CATEGORY, 'Current Trend', COL_AVG_ENGAGEMENT, 'Recommendation']
    required_cols = [COL_STORE_ID, COL_WEEK, COL_ENGAGED_PCT]

    if df_filtered.empty or not all(col in df_filtered.columns for col in required_cols):
        return pd.DataFrame(columns=output_cols)

    all_store_ids = sorted(df_filtered[COL_STORE_ID].unique())
    if not all_store_ids:
        return pd.DataFrame(columns=output_cols)

    # Recalculate trends based on the *filtered* data for current context recommendations
    store_trends_filtered = calculate_store_trends(df_filtered, trend_window)

    for store_id in all_store_ids:
        store_data_filtered = df_filtered[df_filtered[COL_STORE_ID] == store_id]
        if store_data_filtered.empty: continue

        avg_eng = safe_mean(store_data_filtered[COL_ENGAGED_PCT])
        trend = store_trends_filtered.get(store_id, "Stable") # Get trend for this store in filtered period

        # Get category from pre-calculated stats (based on same filtered period)
        category = CAT_UNCATEGORIZED
        action_plan = PERFORMANCE_CATEGORIES[CAT_UNCATEGORIZED]['action']
        if not store_stats.empty and COL_STORE_ID in store_stats.columns:
            cat_row = store_stats[store_stats[COL_STORE_ID] == store_id]
            if not cat_row.empty:
                # Use .get with default values for safety
                category = cat_row.iloc[0].get(COL_CATEGORY, CAT_UNCATEGORIZED)
                action_plan = cat_row.iloc[0].get(COL_ACTION_PLAN, PERFORMANCE_CATEGORIES.get(category, {}).get('action', 'N/A'))


        # Base recommendation on category (action plan)
        rec = action_plan if action_plan != 'N/A' else "Review performance data." # Default recommendation

        # --- Refine recommendation based on *current* trend vs category ---
        # If a star performer is currently declining
        if category == CAT_STAR and trend in ["Downward", "Strong Downward"]:
             rec = f"{action_plan}. However, recent trend is concerning ({trend}). Investigate potential causes."
        # If an intervention store is currently improving
        elif category == CAT_INTERVENTION and trend in ["Upward", "Strong Upward"]:
             rec = f"Showing recent improvement ({trend})! Reinforce positive actions. {action_plan}"
        # If a stabilize store is no longer declining
        elif category == CAT_STABILIZE and trend not in ["Downward", "Strong Downward"]:
             rec = f"Stabilization efforts may be working (recent trend: {trend}). Maintain focus on consistency. {action_plan}"
        # If an improving store is no longer improving
        elif category == CAT_IMPROVING and trend not in ["Upward", "Strong Upward"]:
             rec = f"Improvement seems stalled (recent trend: {trend}). Re-evaluate strategies or seek support. {action_plan}"


        # --- Append anomaly note if significant anomaly exists for this store ---
        store_anoms = anomalies_df[anomalies_df[COL_STORE_ID] == store_id] if not anomalies_df.empty else pd.DataFrame()
        if not store_anoms.empty:
            # Get the most significant anomaly for this store (already sorted)
            biggest_anomaly = store_anoms.iloc[0]
            change_type = 'positive spike' if biggest_anomaly[COL_CHANGE_PCT_PTS] > 0 else 'negative drop'
            rec += f" **Anomaly Alert:** Investigate significant {change_type} in Week {int(biggest_anomaly[COL_WEEK])} (Z={biggest_anomaly[COL_Z_SCORE]:.1f})."


        recommendations.append({
            COL_STORE_ID: store_id,
            COL_CATEGORY: category,
            'Current Trend': trend, # Trend based on filtered data
            COL_AVG_ENGAGEMENT: round(avg_eng, 2) if avg_eng is not None else None,
            'Recommendation': rec
        })

    rec_df = pd.DataFrame(recommendations)

    # Ensure all expected output columns exist
    for col in output_cols:
        if col not in rec_df.columns:
            rec_df[col] = pd.NA

    return rec_df[output_cols]


def calculate_recent_performance_trends(df: pd.DataFrame, trend_window: int, momentum_threshold: float) -> pd.DataFrame:
    """Analyzes short-term trends (improving, stable, declining) over a recent window within the provided DataFrame."""
    directions = []
    output_cols = ['store', 'direction', 'strength', 'indicator', 'start_value', 'current_value', 'total_change', 'color', 'weeks', 'slope']
    required_cols = [COL_STORE_ID, COL_WEEK, COL_ENGAGED_PCT]

    if df.empty or not all(col in df.columns for col in required_cols):
        return pd.DataFrame(columns=output_cols)

    for store_id, data in df.groupby(COL_STORE_ID):
        # Ensure enough data points for the specified window
        if len(data) < trend_window: continue

        # Get the most recent 'trend_window' weeks for this store
        recent = data.sort_values(COL_WEEK).tail(trend_window)
        vals = recent[COL_ENGAGED_PCT].values

        # Ensure no NaNs in the values used for calculation
        if pd.isna(vals).any(): continue
        # Ensure we have at least 2 values for comparison
        if len(vals) < 2: continue

        # --- Calculate Change Metric ---
        # Compare mean of first half vs second half for trend direction sensitivity
        if trend_window <= 3:
            # For short windows, compare start vs end
            first_half_mean = vals[0]
            second_half_mean = vals[-1]
        else:
            split_point = trend_window // 2
            first_half_mean = vals[:split_point].mean()
            second_half_mean = vals[-split_point:].mean()

        change = second_half_mean - first_half_mean

        # --- Calculate Other Metrics ---
        start_val = vals[0]
        current_val = vals[-1]
        total_change = current_val - start_val
        # Use existing slope calculation for consistency
        slope = calculate_trend_slope(recent, trend_window)

        # --- Classify Direction & Strength ---
        if abs(change) < momentum_threshold:
            direction, strength, color = "Stable", "Holding Steady", PERFORMANCE_CATEGORIES.get(CAT_STABILIZE, {}).get('color', '#757575') # Use neutral color
        elif change > 0:
            direction = "Improving"
            strength = "Strong Improvement" if change > 2 * momentum_threshold else "Gradual Improvement"
            color = PERFORMANCE_CATEGORIES.get(CAT_IMPROVING, {}).get('color', '#1976D2') # Use improving color
        else: # change < 0
            direction = "Declining"
            strength = "Significant Decline" if change < -2 * momentum_threshold else "Gradual Decline"
            color = PERFORMANCE_CATEGORIES.get(CAT_INTERVENTION, {}).get('color', '#C62828') # Use declining color

        # --- Choose Indicator ---
        if direction == "Improving":
            indicator = "🔼" if "Strong" in strength else "↗️"
        elif direction == "Declining":
            indicator = "🔽" if "Significant" in strength else "↘️"
        else: # Stable
            indicator = "➡️"

        directions.append({
            'store': store_id,
            'direction': direction,
            'strength': strength,
            'indicator': indicator,
            'start_value': start_val,
            'current_value': current_val,
            'total_change': total_change,
            'color': color,
            'weeks': trend_window,
            'slope': slope
        })

    dir_df = pd.DataFrame(directions)
    # Ensure all output columns exist
    for col in output_cols:
        if col not in dir_df.columns:
             dir_df[col] = pd.NA

    return dir_df[output_cols]


# --- Charting Functions ---

def create_engagement_trend_chart(df_plot: pd.DataFrame, dist_trend: Optional[pd.DataFrame], df_comp_plot: Optional[pd.DataFrame], dist_trend_comp: Optional[pd.DataFrame], show_ma: bool, view_option: str, stores_to_show: Optional[List[str]] = None) -> Optional[alt.LayerChart]:
    """Creates the layered Altair chart for engagement trends."""
    required_cols = [COL_STORE_ID, COL_WEEK, COL_ENGAGED_PCT]
    if df_plot.empty or not all(col in df_plot.columns for col in required_cols):
        return None

    layers = []
    color_scale = alt.Scale(scheme='category10') # Consistent color scheme

    # --- Tooltip Definitions ---
    tooltip_base = [
        alt.Tooltip(COL_STORE_ID, title='Store'),
        alt.Tooltip(COL_WEEK, title='Week', type='ordinal'), # Treat week as ordinal for tooltip
        alt.Tooltip(COL_ENGAGED_PCT, format='.2f', title='Engaged %')
    ]
    tooltip_ma = [
        alt.Tooltip(COL_STORE_ID, title='Store'),
        alt.Tooltip(COL_WEEK, title='Week', type='ordinal'),
        alt.Tooltip(COL_MA_4W, format='.2f', title=f'{DEFAULT_TREND_WINDOW}W MA')
    ] if COL_MA_4W in df_plot.columns else tooltip_base # Fallback if MA not calculated

    tooltip_dist = [
        alt.Tooltip(COL_WEEK, title='Week', type='ordinal'),
        alt.Tooltip('Average Engagement %', format='.2f', title='District Avg')
    ] if dist_trend is not None and 'Average Engagement %' in dist_trend.columns else [alt.Tooltip(COL_WEEK, title='Week', type='ordinal')]

    tooltip_dist_ma = [
        alt.Tooltip(COL_WEEK, title='Week', type='ordinal'),
        alt.Tooltip(COL_MA_4W, format='.2f', title=f'District {DEFAULT_TREND_WINDOW}W MA')
    ] if dist_trend is not None and COL_MA_4W in dist_trend.columns else tooltip_dist

    tooltip_dist_comp = [
         alt.Tooltip(COL_WEEK, title='Week', type='ordinal'),
         alt.Tooltip('Average Engagement %', format='.2f', title='Comp. Period Avg')
    ] if dist_trend_comp is not None and 'Average Engagement %' in dist_trend_comp.columns else [alt.Tooltip(COL_WEEK, title='Week', type='ordinal')]

    tooltip_dist_comp_ma = [
         alt.Tooltip(COL_WEEK, title='Week', type='ordinal'),
         alt.Tooltip(COL_MA_4W, format='.2f', title=f'Comp. Period {DEFAULT_TREND_WINDOW}W MA')
    ] if dist_trend_comp is not None and COL_MA_4W in dist_trend_comp.columns else tooltip_dist_comp


    # --- Store Lines ---
    plot_stores = True
    data_for_stores = df_plot.copy()
    # Handle Custom Selection mode
    if view_option == "Custom Selection":
        if stores_to_show:
            data_for_stores = data_for_stores[data_for_stores[COL_STORE_ID].isin(stores_to_show)]
            if data_for_stores.empty:
                plot_stores = False # Don't plot if filter results in empty data
        else:
            plot_stores = False # Don't plot if no stores selected in this mode

    # Add store lines if applicable
    if plot_stores and not data_for_stores.empty:
        # Define base chart for stores
        base_chart_stores = alt.Chart(data_for_stores).encode(
             x=alt.X(f'{COL_WEEK}:O', title='Week'), # Treat week as ordinal on axis
             color=alt.Color(f'{COL_STORE_ID}:N', scale=color_scale, title="Store")
        )

        # Add selection for non-custom modes
        store_sel = alt.selection_point(fields=[COL_STORE_ID], bind='legend') if view_option != "Custom Selection" else None
        opacity = alt.condition(store_sel, alt.value(1), alt.value(0.2)) if store_sel else alt.value(1)
        stroke_width = alt.condition(store_sel, alt.value(3), alt.value(1.5)) if store_sel else alt.value(3 if view_option == "Custom Selection" else 1.5)

        # Main line
        line_chart = base_chart_stores.mark_line().encode(
            y=alt.Y(f'{COL_ENGAGED_PCT}:Q', title='Engaged Transaction %'),
            opacity=opacity,
            strokeWidth=stroke_width,
            tooltip=tooltip_base
        )
        if store_sel: line_chart = line_chart.add_params(store_sel)
        layers.append(line_chart)

        # Points (only for custom selection for clarity)
        if view_option == "Custom Selection":
            layers.append(base_chart_stores.mark_point(filled=True, size=80).encode(
                y=f'{COL_ENGAGED_PCT}:Q',
                tooltip=tooltip_base
            ))

        # Moving Average line
        if show_ma and COL_MA_4W in data_for_stores.columns and not data_for_stores[COL_MA_4W].isna().all():
            ma_opacity = alt.condition(store_sel, alt.value(0.8), alt.value(0.1)) if store_sel else alt.value(1)
            ma_chart = base_chart_stores.mark_line(strokeDash=[2,2]).encode(
                y=alt.Y(f'{COL_MA_4W}:Q'), # Title comes from axis if shared
                opacity=ma_opacity,
                strokeWidth=alt.value(2 if view_option == "Custom Selection" else 1.5),
                tooltip=tooltip_ma
            )
            if store_sel: ma_chart = ma_chart.add_params(store_sel) # Link opacity
            layers.append(ma_chart)


    # --- District Average Lines ---
    if dist_trend is not None and not dist_trend.empty:
        base_chart_dist = alt.Chart(dist_trend).encode(x=alt.X(f'{COL_WEEK}:O', title='Week'))
        # District Average
        layers.append(base_chart_dist.mark_line(color='black', strokeDash=[4,2], size=3).encode(
            y=alt.Y('Average Engagement %:Q', title='Engaged Transaction %'), # Use shared Y axis title
            tooltip=tooltip_dist
        ).properties(title="District Average"))

        # District Moving Average
        if show_ma and COL_MA_4W in dist_trend.columns and not dist_trend[COL_MA_4W].isna().all():
            layers.append(base_chart_dist.mark_line(color='black', strokeDash=[1,1], size=2, opacity=0.7).encode(
                y=f'{COL_MA_4W}:Q',
                tooltip=tooltip_dist_ma
            ).properties(title=f"District {DEFAULT_TREND_WINDOW}W MA"))


    # --- Comparison Period District Lines ---
    if dist_trend_comp is not None and not dist_trend_comp.empty:
        base_chart_dist_comp = alt.Chart(dist_trend_comp).encode(x=alt.X(f'{COL_WEEK}:O', title='Week'))
        # Comparison District Average
        layers.append(base_chart_dist_comp.mark_line(color='#555555', strokeDash=[4,2], size=2).encode(
            y=alt.Y('Average Engagement %:Q'), # Shared Y axis
            tooltip=tooltip_dist_comp
        ).properties(title="Comparison Period Avg"))

        # Comparison District Moving Average
        if show_ma and COL_MA_4W in dist_trend_comp.columns and not dist_trend_comp[COL_MA_4W].isna().all():
            layers.append(base_chart_dist_comp.mark_line(color='#555555', strokeDash=[1,1], size=1.5, opacity=0.7).encode(
                y=f'{COL_MA_4W}:Q',
                tooltip=tooltip_dist_comp_ma
            ).properties(title=f"Comparison Period {DEFAULT_TREND_WINDOW}W MA"))


    if not layers:
        return None

    # Combine layers and set properties
    final_chart = alt.layer(*layers).resolve_scale(
        y='shared' # Ensure all lines use the same Y axis scale
    ).properties(
        height=CHART_HEIGHT_DEFAULT
    ).interactive() # Enable zooming and panning

    return final_chart


def create_heatmap(df_heatmap: pd.DataFrame, sort_method: str, color_scheme: str) -> Optional[alt.Chart]:
    """Creates the engagement heatmap."""
    required_cols = [COL_STORE_ID, COL_WEEK, COL_ENGAGED_PCT]
    if df_heatmap.empty or not all(col in df_heatmap.columns for col in required_cols) or df_heatmap[COL_ENGAGED_PCT].dropna().empty:
        return None

    # Rename for Altair field name validity if needed (Store # is fine, but consistency)
    df_heatmap = df_heatmap.rename(columns={COL_STORE_ID: 'StoreID', COL_ENGAGED_PCT: 'EngagedPct'})

    # Determine store sort order
    if sort_method == "Average Engagement":
        # Sort by average engagement over the period shown in the heatmap
        store_order = df_heatmap.groupby('StoreID')['EngagedPct'].mean().sort_values(ascending=False).index.tolist()
    else: # Recent Performance (Sort by performance in the last week shown in the heatmap)
        most_recent_week = safe_max(df_heatmap[COL_WEEK])
        if most_recent_week is None:
             store_order = sorted(df_heatmap['StoreID'].unique()) # Fallback sort
        else:
             # Sort by performance in the most recent week available in the heatmap data
             last_week_data = df_heatmap[df_heatmap[COL_WEEK] == most_recent_week]
             store_order = last_week_data.sort_values('EngagedPct', ascending=False)['StoreID'].tolist()
             # Add stores present in heatmap but not in the last week (e.g., if they stopped reporting), sorted alphabetically
             stores_in_last_week = set(store_order)
             all_stores_in_heatmap = set(df_heatmap['StoreID'].unique())
             missing_stores = sorted(list(all_stores_in_heatmap - stores_in_last_week))
             store_order.extend(missing_stores)


    num_stores = len(store_order)
    chart_height = max(CHART_HEIGHT_SHORT, HEATMAP_ROW_HEIGHT * num_stores) # Dynamic height based on number of stores

    try:
        heatmap_chart = alt.Chart(df_heatmap).mark_rect().encode(
            x=alt.X(f'{COL_WEEK}:O', title='Week', axis=alt.Axis(labelAngle=0)), # Keep week labels horizontal
            y=alt.Y('StoreID:O', title='Store', sort=store_order),
            color=alt.Color('EngagedPct:Q', title='Engaged %', scale=alt.Scale(scheme=color_scheme), legend=alt.Legend(orient='right')),
            tooltip=['StoreID', alt.Tooltip(f'{COL_WEEK}:O', title='Week'), alt.Tooltip('EngagedPct:Q', format='.2f', title='Engaged %')]
        ).properties(
            height=chart_height
        )
        return heatmap_chart
    except Exception as e:
        st.error(f"Error creating heatmap: {e}")
        return None


def create_comparison_bar_chart(comp_data: pd.DataFrame, district_avg: Optional[float], title: str) -> Optional[alt.LayerChart]:
    """Creates the store comparison bar chart with average line."""
    required_cols = [COL_STORE_ID, COL_ENGAGED_PCT]
    if comp_data.empty or not all(col in comp_data.columns for col in required_cols):
        return None

    num_stores = len(comp_data[COL_STORE_ID].unique())
    chart_height = max(CHART_HEIGHT_SHORT, COMPARISON_BAR_HEIGHT * num_stores) # Dynamic height

    # Base bar chart
    bar_chart = alt.Chart(comp_data).mark_bar().encode(
        y=alt.Y(f'{COL_STORE_ID}:N', title='Store', sort='-x'), # Sort descending by engagement value on X-axis
        x=alt.X(f'{COL_ENGAGED_PCT}:Q', title='Engaged Transaction %'),
        color=alt.Color(f'{COL_ENGAGED_PCT}:Q', scale=alt.Scale(scheme=DEFAULT_COLOR_SCHEME), legend=None), # Simple color scale, hide legend
        tooltip=[alt.Tooltip(COL_STORE_ID, title='Store'), alt.Tooltip(f'{COL_ENGAGED_PCT}:Q', format='.2f', title='Engaged %')]
    ).properties(
        title=title,
        height=chart_height
    )

    # Add district average line only if average is valid
    if district_avg is not None and pd.notna(district_avg):
        avg_rule = alt.Chart(pd.DataFrame({'avg': [district_avg]})).mark_rule(
            color='red', strokeDash=[4, 4], size=2
        ).encode(
            x='avg:Q',
            tooltip=[alt.Tooltip('avg:Q', title='Average (Selected Stores)', format='.2f')] # Clarify average scope
        )
        # Layer the rule over the bars
        return alt.layer(bar_chart, avg_rule)
    else:
        # Return only the bar chart if average is not available
        return bar_chart


def create_relative_comparison_chart(comp_data: pd.DataFrame, district_avg: Optional[float]) -> Optional[alt.LayerChart]:
    """Creates the bar chart showing performance relative to the average for selected stores."""
    required_cols = [COL_STORE_ID, COL_ENGAGED_PCT]
    if comp_data.empty or not all(col in comp_data.columns for col in required_cols):
        return None
    if district_avg is None or pd.isna(district_avg):
         st.caption("_Relative comparison requires a valid average for the selected stores/period._")
         return None # Cannot calculate relative difference without a valid average


    # Calculate difference and percentage difference
    comp_data['Difference'] = comp_data[COL_ENGAGED_PCT] - district_avg
    # Avoid division by zero if average is zero, assign 0% difference
    comp_data['Percentage'] = (comp_data['Difference'] / district_avg * 100) if district_avg != 0 else 0.0

    num_stores = len(comp_data[COL_STORE_ID].unique())
    chart_height = max(CHART_HEIGHT_SHORT, COMPARISON_BAR_HEIGHT * num_stores) # Dynamic height

    # Determine color domain based on min/max percentage difference
    min_perc = safe_min(comp_data['Percentage'])
    max_perc = safe_max(comp_data['Percentage'])

    # Handle cases where min/max might be None or NaN
    min_perc = min_perc if pd.notna(min_perc) else 0
    max_perc = max_perc if pd.notna(max_perc) else 0
    # Ensure range has some width if min == max
    if min_perc == max_perc:
        min_perc -= 1
        max_perc += 1

    # Ensure 0 is included in the domain for the midpoint color, handle cases where all values are same sign
    domain = sorted(list(set([min_perc, 0, max_perc])))
    # If only two unique values in domain (e.g., [0, 50] or [-50, 0]), adjust to ensure 3 points for color scale
    if len(domain) == 2:
        if 0 in domain: # e.g. [0, 50] -> [0, 0, 50] or [-50, 0] -> [-50, 0, 0]
             domain.insert(domain.index(0), 0)
        else: # e.g. [10, 50] -> [10, (10+50)/2, 50] - add midpoint
             domain.insert(1, (domain[0]+domain[1])/2)
    elif len(domain) == 1: # e.g., all are 0
         domain = [domain[0] - 1, domain[0], domain[0] + 1]


    # Define color range (Red -> Gray/Neutral -> Green)
    color_range = [PERFORMANCE_CATEGORIES[CAT_INTERVENTION]['color'], '#BBBBBB', PERFORMANCE_CATEGORIES[CAT_STAR]['color']]

    # Base chart for relative difference
    diff_chart = alt.Chart(comp_data).mark_bar().encode(
        # Keep sort order consistent with the absolute chart (by Engaged %)
        y=alt.Y(f'{COL_STORE_ID}:N', title='Store', sort=alt.EncodingSortField(field=COL_ENGAGED_PCT, order='descending')),
        x=alt.X('Percentage:Q', title='% Difference from Average'),
        color=alt.Color('Percentage:Q', scale=alt.Scale(domain=domain, range=color_range, type='linear'), legend=None), # Hide legend
        tooltip=[
            alt.Tooltip(COL_STORE_ID, title='Store'),
            alt.Tooltip(f'{COL_ENGAGED_PCT}:Q', format='.2f', title='Engaged %'),
            alt.Tooltip('Percentage:Q', format='+.2f', title='% Diff from Avg')
        ]
    ).properties(
        height=chart_height,
        title="Performance Relative to Average (Selected Stores)"
    )

    # Add zero line for reference
    center_rule = alt.Chart(pd.DataFrame({'center': [0]})).mark_rule(color='black').encode(x='center:Q')

    return alt.layer(diff_chart, center_rule)


def create_rank_trend_chart(rank_data: pd.DataFrame) -> Optional[alt.Chart]:
    """Creates the line chart for tracking weekly ranks."""
    required_cols = [COL_WEEK, COL_STORE_ID, COL_RANK]
    if rank_data.empty or not all(col in rank_data.columns for col in required_cols):
        return None

    # Ensure rank is numeric for plotting scale, drop rows where rank is missing
    rank_data = rank_data.dropna(subset=[COL_RANK]).copy() # Use copy to avoid SettingWithCopyWarning
    if rank_data.empty: return None
    rank_data[COL_RANK] = rank_data[COL_RANK].astype(int)

    # Determine rank domain (min/max rank) - reverse scale so 1 is at the top
    min_rank = safe_min(rank_data[COL_RANK])
    max_rank = safe_max(rank_data[COL_RANK])

    # Handle case where min/max are None or equal
    if min_rank is None or max_rank is None:
        rank_domain = [10, 0] # Default range if no ranks found
    elif min_rank == max_rank:
         # Add padding if only one rank value exists
         rank_domain = [max_rank + 1, min_rank - 1]
    else:
         # Add padding to min/max for better axis display
         rank_domain = [max_rank + 1, min_rank - 1]
    # Ensure domain minimum is not negative
    if rank_domain[1] < 0: rank_domain[1] = 0


    # Base chart for rank trends
    rank_chart_base = alt.Chart(rank_data).mark_line(point=True).encode(
        x=alt.X(f'{COL_WEEK}:O', title='Week'),
        # Use reversed scale domain, ensure zero is not included unless rank is actually 0
        y=alt.Y(f'{COL_RANK}:Q', title='Rank', scale=alt.Scale(domain=rank_domain, zero=False)),
        color=alt.Color(f'{COL_STORE_ID}:N', scale=alt.Scale(scheme='category10'), title="Store"),
        tooltip=[
            alt.Tooltip(COL_STORE_ID, title='Store'),
            alt.Tooltip(f'{COL_WEEK}:O', title='Week'),
            alt.Tooltip(f'{COL_RANK}:Q', title='Rank')
        ]
    ).properties(
        height=CHART_HEIGHT_SHORT,
        title="Weekly Rank Tracking"
    )

    # Add interactive legend selection
    rank_sel = alt.selection_point(fields=[COL_STORE_ID], bind='legend')
    rank_chart_interactive = rank_chart_base.add_params(rank_sel).encode(
        opacity=alt.condition(rank_sel, alt.value(1), alt.value(0.2)),
        strokeWidth=alt.condition(rank_sel, alt.value(3), alt.value(1.5))
    ).interactive() # Add interactive zooming/panning

    return rank_chart_interactive

def create_recent_trend_bar_chart(dir_df: pd.DataFrame) -> Optional[alt.LayerChart]:
    """Creates the bar chart showing total change for recent trends."""
    required_cols = ['store', 'total_change', 'direction', 'weeks']
    if dir_df.empty or not all(col in dir_df.columns for col in required_cols):
        return None

    num_stores = len(dir_df['store'].unique())
    chart_height = max(CHART_HEIGHT_SHORT, COMPARISON_BAR_HEIGHT * num_stores)
    # Get the actual window size used from the data (should be consistent)
    trend_window_used = dir_df["weeks"].iloc[0] if not dir_df.empty else RECENT_TRENDS_WINDOW


    # Base chart for total change bars
    change_chart = alt.Chart(dir_df).mark_bar().encode(
        x=alt.X('total_change:Q', title=f'Change in Engagement % (Last {trend_window_used} Weeks)'),
        # Sort bars by the total change value
        y=alt.Y('store:N', sort=alt.EncodingSortField(field='total_change', order='descending'), title='Store'),
        color=alt.Color('direction:N',
                        scale=alt.Scale(domain=['Improving', 'Stable', 'Declining'],
                                        # Use consistent category colors where applicable
                                        range=[PERFORMANCE_CATEGORIES.get(CAT_IMPROVING, {}).get('color', '#1976D2'),
                                               '#757575', # Use a distinct gray for stable
                                               PERFORMANCE_CATEGORIES.get(CAT_INTERVENTION, {}).get('color', '#C62828')]),
                        title='Recent Trend'),
        tooltip=[alt.Tooltip('store:N', title='Store'),
                 alt.Tooltip('direction:N', title='Direction'),
                 alt.Tooltip('strength:N', title='Performance'),
                 alt.Tooltip('start_value:Q', format='.2f', title='Starting Value'),
                 alt.Tooltip('current_value:Q', format='.2f', title='Current Value'),
                 alt.Tooltip('total_change:Q', format='+.2f', title='Total Change')]
    ).properties(
        height=chart_height,
        title="Recent Performance Change (within Heatmap Range)"
    )

    # Add zero line for reference, make it visible against potentially dark bars
    zero_line = alt.Chart(pd.DataFrame({'x': [0]})).mark_rule(color='white', strokeDash=[3, 3], opacity=0.7).encode(x='x:Q')

    return alt.layer(change_chart, zero_line)


# --- UI Display Functions ---

def display_sidebar(
    df: Optional[pd.DataFrame],
    data_file_exists: bool # Flag to indicate if primary file is already uploaded
) -> Tuple[str, str, List[str], float, bool, int]:
    """Creates and manages the Streamlit sidebar elements for filters and settings.
       File uploaders are handled outside this function.
    """
    # Initialize filter defaults
    quarter_choice = "All"
    week_choice = "All"
    store_choice = []
    z_threshold = DEFAULT_Z_THRESHOLD
    show_ma = DEFAULT_SHOW_MA
    trend_analysis_weeks = DEFAULT_TREND_WINDOW
    store_list = [] # Initialize store list

    # --- Display Filters and Settings only if data is loaded ---
    if data_file_exists and df is not None and not df.empty:
        st.sidebar.header("Filters")

        # Get available stores from the loaded data
        if COL_STORE_ID in df.columns:
             store_list = sorted(df[COL_STORE_ID].unique().tolist())

        # --- Quarter Filter ---
        if COL_QUARTER in df.columns:
            quarters = sorted(df[COL_QUARTER].dropna().unique().tolist())
            quarter_options = ["All"] + [f"Q{int(q)}" for q in quarters]
            # Use session state to remember selection if possible, otherwise default to 0
            idx = st.session_state.get("quarter_filter_idx", 0)
            if idx >= len(quarter_options): idx = 0 # Reset if index out of bounds
            quarter_choice = st.sidebar.selectbox(
                "Select Quarter", quarter_options, index=idx, key="quarter_filter",
                on_change=lambda: st.session_state.update(quarter_filter_idx=quarter_options.index(st.session_state.quarter_filter))
                )
        else:
            st.sidebar.markdown("_Quarter information not available._")


        # --- Week Filter (dependent on Quarter selection) ---
        if COL_WEEK in df.columns:
            # Determine available weeks based on quarter selection
            weeks_in_period = df.copy()
            if quarter_choice != "All" and COL_QUARTER in weeks_in_period.columns:
                try:
                    q_num = int(quarter_choice.replace('Q', ''))
                    weeks_in_period = weeks_in_period[weeks_in_period[COL_QUARTER] == q_num]
                except (ValueError, KeyError):
                    pass # Keep all weeks if quarter filter fails

            available_weeks = sorted(weeks_in_period[COL_WEEK].unique())

            # Create week options only if weeks are available
            if available_weeks:
                 week_options = ["All"] + [str(int(w)) for w in available_weeks]
                 # Try to preserve week selection if it's still valid, else default to "All"
                 current_week_selection = st.session_state.get("week_filter_val", "All")
                 if current_week_selection not in week_options:
                      current_week_selection = "All"
                 week_choice = st.sidebar.selectbox(
                      "Select Week", week_options, index=week_options.index(current_week_selection), key="week_filter",
                      on_change=lambda: st.session_state.update(week_filter_val=st.session_state.week_filter)
                      )
            else:
                 st.sidebar.markdown("_No weeks available for selected quarter._")
                 week_options = ["All"] # Ensure options list is not empty
                 week_choice = "All" # Force 'All' if no specific weeks
        else:
             st.sidebar.markdown("_Week information not available._")


        # --- Store Filter ---
        if store_list:
            # Try to preserve store selection
            current_store_selection = st.session_state.get("store_filter_val", [])
            # Filter out any previously selected stores that are no longer in the list
            valid_store_selection = [s for s in current_store_selection if s in store_list]
            store_choice = st.sidebar.multiselect(
                 "Select Store(s)", store_list, default=valid_store_selection, key="store_filter",
                 on_change=lambda: st.session_state.update(store_filter_val=st.session_state.store_filter)
                 )
        else:
            st.sidebar.markdown("_Store information not available._")


        # --- Advanced Settings ---
        with st.sidebar.expander("Advanced Settings", expanded=False):
            z_threshold = st.slider("Anomaly Z-score Threshold", 1.0, 3.0, float(st.session_state.get("z_slider_val", DEFAULT_Z_THRESHOLD)), 0.1, key="z_slider", on_change=lambda: st.session_state.update(z_slider_val=st.session_state.z_slider))
            show_ma = st.checkbox(f"Show {DEFAULT_TREND_WINDOW}-week moving average", value=st.session_state.get("ma_checkbox_val", DEFAULT_SHOW_MA), key="ma_checkbox", on_change=lambda: st.session_state.update(ma_checkbox_val=st.session_state.ma_checkbox))
            trend_analysis_weeks = st.slider("Trend analysis window (weeks)", MIN_TREND_WINDOW, MAX_TREND_WINDOW, int(st.session_state.get("trend_window_slider_val", DEFAULT_TREND_WINDOW)), key="trend_window_slider", on_change=lambda: st.session_state.update(trend_window_slider_val=st.session_state.trend_window_slider))
            st.caption("Adjust sensitivity for anomaly detection and overall trend analysis.")

    # --- Show prompt if no data file is uploaded yet ---
    elif not data_file_exists:
         st.sidebar.info("Upload data file to see filters.")


    # --- Footer and Help (Always display) ---
    now = datetime.datetime.now()
    st.sidebar.markdown("---")
    # Consider removing copyright if not applicable/official
    # st.sidebar.caption(f"© Publix Super Markets, Inc. {now.year}")
    st.sidebar.caption(f"Dashboard Refactored - Run: {now.strftime('%Y-%m-%d %H:%M')}")
    with st.sidebar.expander("Help & Information"):
        st.markdown("#### Using This Dashboard")
        st.markdown("- **Upload Data**: Start by uploading your engagement data file.")
        st.markdown("- **Apply Filters**: Use the filters to focus on specific time periods or stores.")
        st.markdown("- **Explore Tabs**: Each tab provides different insights:")
        st.markdown("  - `Engagement Trends`: Performance over time")
        st.markdown("  - `Store Comparison`: Compare stores directly")
        st.markdown("  - `Store Performance Categories`: Categories and action plans")
        st.markdown("  - `Anomalies & Insights`: Unusual patterns and opportunities")
        st.markdown("#### Disclaimer")
        st.markdown("_This is not an official Publix tool. Use data responsibly._")


    # Return only the filter values, file objects are handled outside
    return quarter_choice, week_choice, store_choice, z_threshold, show_ma, trend_analysis_weeks


def display_executive_summary(summary_data: Dict[str, Any]):
    """Displays the executive summary metrics and text."""
    st.subheader("Executive Summary")
    # Check if essential data for summary exists
    if summary_data.get('current_week') is None or summary_data.get('current_avg') is None:
        st.info("Not enough data to generate executive summary based on current filters.")
        return

    col1, col2, col3 = st.columns(3)

    # --- Metric 1: Average Engagement ---
    avg_display = format_percentage(summary_data['current_avg'])
    # Determine delta only if previous week and average are available
    delta_str = "N/A"
    if summary_data.get('prev_week') is not None and summary_data.get('prev_avg') is not None:
        delta_str = format_delta(summary_data.get('delta_val')) # Use .get for safety
    col1.metric(
        label=f"{summary_data.get('avg_label', 'Avg Engagement')} (Week {summary_data['current_week']})",
        value=avg_display,
        delta=delta_str,
        help=f"Average engagement for Week {summary_data['current_week']}. Delta compares to Week {summary_data.get('prev_week', 'N/A')}."
    )

    # --- Metric 2 & 3: Top/Bottom Performers ---
    # Top Performer
    if summary_data.get('top_store') is not None:
        top_perf_str = f"Store {summary_data['top_store']} — {format_percentage(summary_data.get('top_val'))}"
        col2.metric(
            label=f"Top Performer (Week {summary_data['current_week']})",
            value=top_perf_str,
            help="Highest average engagement for the week among selected stores."
        )
        # Add trend for top performer (trend calculated over the whole filtered period)
        top_trend = summary_data.get('store_trends', pd.Series(dtype=str)).get(summary_data['top_store'], "N/A")
        t_color_class = "highlight-good" if top_trend in ["Upward","Strong Upward"] else "highlight-bad" if top_trend in ["Downward","Strong Downward"] else "highlight-neutral"
        col2.markdown(f"<small>Trend (Period): <span class='{t_color_class}'>{top_trend}</span></small>", unsafe_allow_html=True)
    else:
        col2.metric(f"Top Performer (Week {summary_data['current_week']})", "N/A")

    # Bottom Performer
    if summary_data.get('bottom_store') is not None:
        bottom_perf_str = f"Store {summary_data['bottom_store']} — {format_percentage(summary_data.get('bottom_val'))}"
        col3.metric(
            label=f"Bottom Performer (Week {summary_data['current_week']})",
            value=bottom_perf_str,
            help="Lowest average engagement for the week among selected stores."
        )
         # Add trend for bottom performer
        bottom_trend = summary_data.get('store_trends', pd.Series(dtype=str)).get(summary_data['bottom_store'], "N/A")
        b_color_class = "highlight-good" if bottom_trend in ["Upward","Strong Upward"] else "highlight-bad" if bottom_trend in ["Downward","Strong Downward"] else "highlight-neutral"
        col3.markdown(f"<small>Trend (Period): <span class='{b_color_class}'>{bottom_trend}</span></small>", unsafe_allow_html=True)
    else:
        col3.metric(f"Bottom Performer (Week {summary_data['current_week']})", "N/A")


    # --- Summary Text ---
    if summary_data.get('delta_val') is not None and summary_data.get('prev_week') is not None:
        st.markdown(f"Week {summary_data['current_week']} average engagement is <span class='{summary_data.get('trend_class', 'highlight-neutral')}'>{abs(summary_data['delta_val']):.2f} points {summary_data.get('trend_dir', 'flat')}</span> from Week {summary_data['prev_week']}.", unsafe_allow_html=True)
    elif summary_data.get('current_avg') is not None:
        st.markdown(f"Week {summary_data['current_week']} engagement average: <span class='highlight-neutral'>{format_percentage(summary_data['current_avg'])}</span>", unsafe_allow_html=True)
    # No else needed, covered by the initial check


def display_key_insights(insights: List[str]):
    """Displays the generated key insights list."""
    st.subheader("Key Insights")
    if not insights or insights == ["No data available for insights."]:
        st.info("Not enough data to generate key insights for the current selection.")
        return
    # Display insights as a list
    insight_text = ""
    for i, point in enumerate(insights, start=1):
        insight_text += f"{i}. {point}\n"
    st.markdown(insight_text)


def display_engagement_trends_tab(df_filtered: pd.DataFrame, df_comp_filtered: Optional[pd.DataFrame], show_ma: bool, district_trend: Optional[pd.DataFrame], district_trend_comp: Optional[pd.DataFrame]):
    """Displays the content for the Engagement Trends tab."""
    st.subheader("Engagement Trends Over Time")

    # --- View Options ---
    view_option = st.radio("View mode:", ["All Stores", "Custom Selection", "Recent Trends"], horizontal=True,
                           help="All Stores: View all stores | Custom Selection: Pick specific stores | Recent Trends: Focus on recent weeks", key="trends_view_mode")

    # Data preparation for plotting
    df_plot = df_filtered.copy()
    df_comp_plot = df_comp_filtered.copy() if df_comp_filtered is not None else None

    # Add period column if comparison data exists and is not empty
    if df_comp_plot is not None and not df_comp_plot.empty:
        df_plot[COL_PERIOD] = 'Current'
        df_comp_plot[COL_PERIOD] = 'Comparison'
        # Combine only if comparison data is not empty
        df_plot = pd.concat([df_plot, df_comp_plot], ignore_index=True)


    # Calculate Moving Averages on the potentially combined data
    df_plot = calculate_moving_averages(df_plot) # Will add MA_4W column


    # --- Recent Trends Specific Filters & Metrics ---
    stores_to_show_custom = []
    if view_option == "Recent Trends":
        all_weeks = sorted(df_plot[COL_WEEK].unique()) if COL_WEEK in df_plot.columns else []
        if len(all_weeks) > 1:
            min_week, max_week = min(all_weeks), max(all_weeks)
            # Suggest a default range of last 8 weeks or all if fewer than 8
            default_start = all_weeks[0] if len(all_weeks) <= 8 else all_weeks[-8]
            default_end = all_weeks[-1]
            # Ensure default_start is not after default_end if few weeks exist
            if default_start > default_end: default_start = default_end

            try:
                recent_weeks_range = st.select_slider(
                    "Select weeks to display:",
                    options=all_weeks,
                    value=(default_start, default_end), # Pass tuple here
                    help="Adjust to show a shorter or longer recent period",
                    key="recent_weeks_slider"
                )
            except st.errors.StreamlitAPIException:
                 # Fallback if default value might be invalid (e.g., only one week)
                 st.warning("Could not set default week range. Please select manually.")
                 recent_weeks_range = st.select_slider(
                     "Select weeks to display:",
                     options=all_weeks,
                     value=(min_week, max_week), # Fallback to full range
                     help="Adjust to show a shorter or longer recent period",
                     key="recent_weeks_slider_fallback"
                 )


            # Filter data based on selected range
            if recent_weeks_range and len(recent_weeks_range) == 2:
                start_week, end_week = recent_weeks_range
                # Ensure start <= end
                if start_week > end_week: start_week, end_week = end_week, start_week

                # Filter the main plot data and the district trends
                df_plot = df_plot[(df_plot[COL_WEEK] >= start_week) & (df_plot[COL_WEEK] <= end_week)]
                if district_trend is not None:
                     district_trend = district_trend[(district_trend[COL_WEEK] >= start_week) & (district_trend[COL_WEEK] <= end_week)]
                if district_trend_comp is not None:
                     district_trend_comp = district_trend_comp[(district_trend_comp[COL_WEEK] >= start_week) & (district_trend_comp[COL_WEEK] <= end_week)]

            # --- Display Recent Trend Metrics ---
            st.markdown("---") # Separator
            st.markdown("##### Metrics for Selected Recent Weeks")
            col1, col2 = st.columns(2)
            with col1:
                metric_val, metric_delta = "N/A", None
                help_text = "Requires at least two weeks in the selected range."
                if district_trend is not None and len(district_trend) >= 2:
                    last_two = district_trend.sort_values(COL_WEEK).tail(2)
                    cur_val = last_two['Average Engagement %'].iloc[1]
                    prev_val = last_two['Average Engagement %'].iloc[0]
                    metric_val = format_percentage(cur_val)
                    if pd.notna(cur_val) and pd.notna(prev_val) and prev_val != 0:
                        change_pct = ((cur_val - prev_val) / prev_val * 100)
                        metric_delta = f"{change_pct:+.1f}%"
                        help_text = f"Change between Week {last_two[COL_WEEK].iloc[0]} and Week {last_two[COL_WEEK].iloc[1]}."
                    elif pd.notna(cur_val):
                         help_text = f"Latest week ({last_two[COL_WEEK].iloc[1]}) average shown. Previous week data missing or zero."

                st.metric("District Trend (Week-over-Week)", metric_val, metric_delta, help=help_text)

            with col2:
                 metric_val, metric_delta = "N/A", None
                 help_text = "Data unavailable for the last week in range."
                 if not df_plot.empty:
                     last_week = safe_max(df_plot[COL_WEEK])
                     if last_week is not None:
                         last_week_data = df_plot[df_plot[COL_WEEK] == last_week]
                         if not last_week_data.empty:
                            top_store_idx = safe_idxmax(last_week_data[COL_ENGAGED_PCT])
                            if top_store_idx is not None:
                                 best_store_row = last_week_data.loc[top_store_idx]
                                 metric_val = f"Store {best_store_row[COL_STORE_ID]}"
                                 metric_delta = format_percentage(best_store_row[COL_ENGAGED_PCT])
                                 help_text = f"Best performing store in Week {last_week}."
                 st.metric(f"Top Performer (Last Week)", metric_val, metric_delta, delta_color="off", help=help_text)

            st.markdown("---") # Separator


        elif len(all_weeks) == 1:
             st.info("Only one week of data available in the current selection. Cannot show recent trends over time.")
             # Keep df_plot as is for the single week chart
        else:
             st.info("No data available for recent trends view.")
             df_plot = pd.DataFrame() # Make empty to prevent chart error


    elif view_option == "Custom Selection":
        available_stores = sorted(df_filtered[COL_STORE_ID].unique()) if COL_STORE_ID in df_filtered.columns else []
        if available_stores:
             # Use session state for multiselect default
             default_selection = st.session_state.get("custom_store_select_val", [])
             # Ensure default selection is valid within available stores
             valid_default = [s for s in default_selection if s in available_stores]
             stores_to_show_custom = st.multiselect(
                  "Select stores to compare:", options=available_stores, default=valid_default, key="custom_store_select",
                  on_change=lambda: st.session_state.update(custom_store_select_val=st.session_state.custom_store_select)
                  )
             if not stores_to_show_custom:
                  st.info("Please select at least one store to display in Custom Selection mode.")
                  # Don't plot if no stores selected in this mode
                  df_plot = pd.DataFrame() # Make df_plot empty
        else:
             st.info("No stores available for selection.")
             df_plot = pd.DataFrame()


    # --- Create and Display Trend Chart ---
    trend_chart = create_engagement_trend_chart(df_plot, district_trend, df_comp_plot, district_trend_comp, show_ma, view_option, stores_to_show_custom)

    if trend_chart:
        st.altair_chart(trend_chart, use_container_width=True)
        # Caption for view mode
        caption = ""
        if view_option == "All Stores":
            caption = "**All Stores View:** Shows all store trends with interactive legend. Black dashed line = average for selected stores/period."
        elif view_option == "Custom Selection":
             caption = "**Custom Selection View:** Shows only selected stores. Black dashed line = average for selected stores/period."
        elif view_option == "Recent Trends":
             caption = "**Recent Trends View:** Focuses on selected weeks. Black dashed line = average for selected stores/period."

        if df_comp_plot is not None and not df_comp_plot.empty:
            caption += " Gray dashed line = comparison period's average."
        if show_ma:
             caption += f" Lighter dashed lines = {DEFAULT_TREND_WINDOW}-week moving averages."
        st.caption(caption)

    elif view_option != "Custom Selection" or stores_to_show_custom: # Only show info if not waiting for custom selection
        st.info("No data available to display engagement trend chart for the current selection.")


    # --- Weekly Engagement Heatmap ---
    st.subheader("Weekly Engagement Heatmap")
    with st.expander("Heatmap Settings", expanded=False):
        colA, colB = st.columns(2)
        with colA:
            sort_method = st.selectbox("Sort stores by:", ["Average Engagement", "Recent Performance"], index=st.session_state.get("heatmap_sort_idx", 0), key="heatmap_sort", on_change=lambda: st.session_state.update(heatmap_sort_idx=["Average Engagement", "Recent Performance"].index(st.session_state.heatmap_sort)))
        with colB:
            color_scheme = st.selectbox("Color scheme:", COLOR_SCHEME_OPTIONS, index=st.session_state.get("heatmap_color_idx", 0), key="heatmap_color", on_change=lambda: st.session_state.update(heatmap_color_idx=COLOR_SCHEME_OPTIONS.index(st.session_state.heatmap_color))).lower()


    # Filter heatmap data by week range slider
    heatmap_df_base = df_filtered.copy() # Use original filtered data for heatmap base
    weeks_list = sorted(heatmap_df_base[COL_WEEK].unique()) if COL_WEEK in heatmap_df_base.columns else []


    if len(weeks_list) > 1:
        min_w, max_w = int(min(weeks_list)), int(max(weeks_list))
        # Ensure slider values are within the available range
        slider_min = min_w
        slider_max = max_w
        # Get default from session state or calculate
        default_slider_start = st.session_state.get("heatmap_slider_val_start", slider_min)
        default_slider_end = st.session_state.get("heatmap_slider_val_end", slider_max)
        # Ensure defaults are within the current options
        if default_slider_start not in weeks_list: default_slider_start = slider_min
        if default_slider_end not in weeks_list: default_slider_end = slider_max
        # Ensure start <= end
        if default_slider_start > default_slider_end: default_slider_start, default_slider_end = default_slider_end, default_slider_start

        default_slider_val = (default_slider_start, default_slider_end)

        # Ensure options are unique if list has duplicates (shouldn't with unique())
        unique_weeks_list = sorted(list(set(weeks_list)))

        try:
            selected_range = st.select_slider(
                "Select week range for heatmap:",
                options=unique_weeks_list,
                value=default_slider_val,
                key="heatmap_week_slider",
                 on_change=lambda: st.session_state.update(heatmap_slider_val_start=st.session_state.heatmap_week_slider[0], heatmap_slider_val_end=st.session_state.heatmap_week_slider[1])
            )
        except (st.errors.StreamlitAPIException, ValueError, IndexError):
             # Fallback if default value is outside options (e.g., single week data)
             st.warning("Could not set default heatmap week range. Please select manually.")
             selected_range = st.select_slider(
                 "Select week range for heatmap:",
                 options=unique_weeks_list,
                 value=(unique_weeks_list[0], unique_weeks_list[-1]), # Use min/max from options
                 key="heatmap_week_slider_fallback"
             )


        if selected_range and len(selected_range) == 2:
            start_w, end_w = selected_range
            # Ensure start <= end
            if start_w > end_w: start_w, end_w = end_w, start_w
            heatmap_df = heatmap_df_base[(heatmap_df_base[COL_WEEK] >= start_w) & (heatmap_df_base[COL_WEEK] <= end_w)].copy()
        else:
             heatmap_df = heatmap_df_base # Fallback if slider fails
    elif len(weeks_list) == 1:
         heatmap_df = heatmap_df_base # Only one week, show it
         st.caption(f"Heatmap showing data for the only available week: {weeks_list[0]}")
    else:
         heatmap_df = pd.DataFrame() # No weeks available


    heatmap_chart = create_heatmap(heatmap_df, sort_method, color_scheme)
    if heatmap_chart:
        st.altair_chart(heatmap_chart, use_container_width=True)
        min_week_hm = safe_min(heatmap_df[COL_WEEK]) if not heatmap_df.empty else "N/A"
        max_week_hm = safe_max(heatmap_df[COL_WEEK]) if not heatmap_df.empty else "N/A"
        st.caption(f"Showing data from Week {int(min_week_hm) if min_week_hm != 'N/A' else 'N/A'} to Week {int(max_week_hm) if max_week_hm != 'N/A' else 'N/A'}. Stores sorted by {sort_method.lower()}. Darker colors = higher engagement.")

    else:
        st.info("No data available for the heatmap based on current filters.")


    # --- Recent Performance Trends Section (Based on Heatmap Range) ---
    st.subheader("Recent Performance Trends (within Heatmap Range)")
    with st.expander("About This Section", expanded=False): # Default collapsed
        st.write("This section shows which stores are **improving**, **stable**, or **declining** over the last several weeks *within the date range selected for the heatmap above*, focusing on short-term momentum.")


    col1, col2 = st.columns(2)
    with col1:
        # Use session state for slider default
        default_recent_window = st.session_state.get("recent_trend_window_val", RECENT_TRENDS_WINDOW)
        trend_window_recent = st.slider("Number of recent weeks to analyze", MIN_TREND_WINDOW, MAX_TREND_WINDOW, default_recent_window, key="recent_trend_window", on_change=lambda: st.session_state.update(recent_trend_window_val=st.session_state.recent_trend_window))
    with col2:
        # Use session state for slider default
        default_sensitivity = st.session_state.get("recent_trend_sensitivity_val", "Medium")
        sensitivity = st.select_slider("Sensitivity to small changes", options=["Low", "Medium", "High"], value=default_sensitivity, key="recent_trend_sensitivity", on_change=lambda: st.session_state.update(recent_trend_sensitivity_val=st.session_state.recent_trend_sensitivity))
        momentum_threshold = RECENT_TRENDS_SENSITIVITY_MAP[sensitivity]


    # Use the heatmap_df as it's already filtered by the week slider
    recent_trends_df = calculate_recent_performance_trends(heatmap_df, trend_window_recent, momentum_threshold)


    if recent_trends_df.empty:
        st.info(f"Not enough data (requires at least {trend_window_recent} weeks within the heatmap range) to analyze recent trends.")

    else:
        # Display summary metrics
        col_imp, col_stab, col_dec = st.columns(3)
        imp_count = (recent_trends_df['direction'] == 'Improving').sum()
        stab_count = (recent_trends_df['direction'] == 'Stable').sum()
        dec_count = (recent_trends_df['direction'] == 'Declining').sum()
        col_imp.metric("Improving", f"{imp_count} stores", delta="↗️", delta_color="off", help=f"Stores showing improvement over the last {trend_window_recent} weeks in the selected range.") # Use off color for icons
        col_stab.metric("Stable", f"{stab_count} stores", delta="➡️", delta_color="off", help=f"Stores holding steady over the last {trend_window_recent} weeks in the selected range.")
        col_dec.metric("Declining", f"{dec_count} stores", delta="↘️", delta_color="inverse", help=f"Stores showing decline over the last {trend_window_recent} weeks in the selected range.") # Use inverse for down arrow



        # Display cards for each store grouped by direction
        for direction in ["Improving", "Stable", "Declining"]: # Ensure consistent order
            group = recent_trends_df[recent_trends_df['direction'] == direction]
            if group.empty: continue

            color = group.iloc[0]['color'] # Get color from the first store in the group
            st.markdown(f"<div style='border-left:5px solid {color}; padding-left:10px; margin:20px 0 10px;'><h4 style='color:{color};'>{direction} ({len(group)} stores)</h4></div>", unsafe_allow_html=True)


            cols = st.columns(min(3, len(group))) # Max 3 columns
            for i, (_, store_data) in enumerate(group.iterrows()):
                with cols[i % 3]: # Cycle through columns
                    change_disp = format_delta(store_data['total_change'])
                    border_color = store_data['color']
                    card_html = f"""
                    <div class='info-card' style='border-left-color: {border_color};'>
                        <h4 style='color:{border_color}; text-align:center;'>{store_data['indicator']} Store {store_data['store']}</h4>
                        <p style='text-align:center;'><strong>{store_data['strength']}</strong></p>
                        <p class='change' style='text-align:center;'>{change_disp} over {store_data['weeks']} weeks</p>
                        <p class='label' style='text-align:center;'>({format_percentage(store_data['start_value'])} → {format_percentage(store_data['current_value'])})</p>
                    </div>
                    """
                    st.markdown(card_html, unsafe_allow_html=True)


        # Display bar chart of changes
        recent_trend_bar_chart = create_recent_trend_bar_chart(recent_trends_df)
        if recent_trend_bar_chart:
             st.altair_chart(recent_trend_bar_chart, use_container_width=True)
        else:
             st.info("Could not generate recent trend bar chart.")


def display_store_comparison_tab(df_filtered: pd.DataFrame, week_choice: str):
    """Displays the content for the Store Comparison tab."""
    st.subheader("Store Performance Comparison")

    required_cols = [COL_STORE_ID, COL_WEEK, COL_ENGAGED_PCT]
    if df_filtered.empty or not all(col in df_filtered.columns for col in required_cols):
         st.warning("Data is missing required columns for comparison.")
         return

    all_stores = sorted(df_filtered[COL_STORE_ID].unique())
    if len(all_stores) < 2:
        st.info("Please select at least two stores in the sidebar filters (or clear store selection) to enable comparison.")
        return

    # Prepare comparison data (either single week or period average)
    comp_data = pd.DataFrame()
    comp_title = "Store Comparison"

    if week_choice != "All":
        try:
             week_num = int(week_choice)
             comp_data_base = df_filtered[df_filtered[COL_WEEK] == week_num].copy()
             comp_title = f"Store Comparison - Week {week_choice}"
             # Ensure we still group by store in case of duplicate entries (shouldn't happen)
             comp_data = comp_data_base.groupby(COL_STORE_ID, as_index=False)[COL_ENGAGED_PCT].mean()
        except ValueError:
             st.error(f"Invalid week selected for comparison: {week_choice}. Showing period average instead.")
             # Fallback to period average
             week_choice = "All"

    if week_choice == "All": # Handles both the 'All' selection and the fallback
        comp_data = df_filtered.groupby(COL_STORE_ID, as_index=False)[COL_ENGAGED_PCT].mean()
        comp_title = "Store Comparison - Period Average"

    # Check if comp_data is valid after potential grouping/filtering
    if comp_data.empty or COL_ENGAGED_PCT not in comp_data.columns:
        st.warning("No comparison data available for the selected week/period.")
        return

    comp_data = comp_data.sort_values(COL_ENGAGED_PCT, ascending=False)
    # Calculate average based on the specific data being compared
    comparison_avg = safe_mean(comp_data[COL_ENGAGED_PCT])

    # --- Absolute Performance Chart ---
    st.markdown("#### Absolute Engagement Percentage")
    comparison_chart = create_comparison_bar_chart(comp_data, comparison_avg, comp_title)
    if comparison_chart:
        st.altair_chart(comparison_chart, use_container_width=True)
        avg_text = f"average engagement ({format_percentage(comparison_avg)})" if comparison_avg is not None else "average engagement (N/A)"
        st.caption(f"Red dashed line indicates the {avg_text} for the selected stores and period.")
    else:
        st.info("Could not generate absolute comparison chart.")

    # --- Relative Performance Chart ---
    st.markdown("#### Performance Relative to Average")
    relative_chart = create_relative_comparison_chart(comp_data, comparison_avg)
    if relative_chart:
        st.altair_chart(relative_chart, use_container_width=True)
        st.caption("Green bars = above selected average, red bars = below selected average.")
    else:
        # Message already shown in create function if avg is invalid
        pass


    # --- Weekly Rank Tracking ---
    if COL_RANK in df_filtered.columns:
        st.markdown("#### Weekly Rank Tracking (Lower is Better)")
        # Use only data that has a rank for this chart
        rank_data = df_filtered[[COL_WEEK, COL_STORE_ID, COL_RANK]].dropna(subset=[COL_RANK])
        if not rank_data.empty:
            rank_chart = create_rank_trend_chart(rank_data)
            if rank_chart:
                st.altair_chart(rank_chart, use_container_width=True)
                st.caption("Lower rank number = better performance. Click legend items to highlight specific stores.")
            else:
                st.info("Could not generate rank tracking chart.")
        else:
            st.info("Weekly rank data not available or not numeric for the selected period.")
    else:
        # Only show message if rank column was expected but not found
        # st.info("Weekly Rank column not found in the data.")
        pass # Don't clutter if rank is just not provided


def display_performance_categories_tab(store_stats: pd.DataFrame):
    """Displays the content for the Store Performance Categories tab."""
    st.subheader("Store Performance Categories")
    st.write("Stores are categorized based on their average engagement level relative to the median and their performance trend correlation over the selected period.")

    if store_stats.empty or COL_CATEGORY not in store_stats.columns:
        st.warning("Performance categories could not be calculated. Ensure data is loaded and filters are applied.")
        return

    # --- Category Overview Cards ---
    st.markdown("#### Category Definitions & Actions")
    colA, colB = st.columns(2)
    cols = [colA, colB, colA, colB] # Assign columns cyclically
    # Define display order for categories
    cat_order = [CAT_STAR, CAT_IMPROVING, CAT_STABILIZE, CAT_INTERVENTION]

    for i, cat in enumerate(cat_order):
        if cat in PERFORMANCE_CATEGORIES:
            info = PERFORMANCE_CATEGORIES[cat]
            with cols[i]:
                 # Use flexbox properties in the card style for equal height
                 card_html = f"""
                 <div class='info-card' style='border-left-color: {info['color']};'>
                     <div> <h4 style='color:{info['color']};'>{info['icon']} {cat}</h4>
                         <p>{info['explanation']}</p>
                     </div>
                     <div> <p><strong>Action:</strong> {info['short_action']}</p>
                     </div>
                 </div>
                 """
                 st.markdown(card_html, unsafe_allow_html=True)


    # --- Stores per Category ---
    st.markdown("#### Stores by Category")
    for cat in cat_order: # Iterate in display order
        subset = store_stats[store_stats[COL_CATEGORY] == cat]
        if subset.empty: continue

        color = PERFORMANCE_CATEGORIES.get(cat, {}).get('color', '#757575')
        icon = PERFORMANCE_CATEGORIES.get(cat, {}).get('icon', '')

        # Category Header
        st.markdown(f"<div style='border-left:5px solid {color}; padding-left:15px; margin: 20px 0 10px 0;'><h4 style='color:{color}; margin-bottom: 0;'>{icon} {cat} ({len(subset)} stores)</h4></div>", unsafe_allow_html=True)


        # Display store cards within the category
        cols = st.columns(min(4, len(subset))) # Max 4 columns for store cards
        # Sort within category: High-to-low avg for top cats, Low-to-high avg for bottom cats
        sort_ascending = (cat in [CAT_IMPROVING, CAT_INTERVENTION])
        subset = subset.sort_values(COL_AVG_ENGAGEMENT, ascending=sort_ascending)

        for i, (_, store) in enumerate(subset.iterrows()):
            with cols[i % 4]: # Cycle through columns
                # Safely get values using .get()
                store_id_disp = store.get(COL_STORE_ID, 'N/A')
                avg_eng_disp = format_percentage(store.get(COL_AVG_ENGAGEMENT))
                trend_corr = store.get(COL_TREND_CORR, 0.0)

                # Determine trend icon based on correlation thresholds
                if trend_corr > TREND_UP: trend_icon = "📈" # Improving
                elif trend_corr < TREND_DOWN: trend_icon = "📉" # Declining
                else: trend_icon = "➡️" # Stable

                card_html = f"""
                <div class='info-card' style='border-left-color: {color};'>
                    <h4 style='color:{color}; text-align:center;'>Store {store_id_disp}</h4>
                    <p style='text-align:center;'><span class='value'>{avg_eng_disp}</span> <span class='label'>Avg</span></p>
                    <p style='text-align:center;'>{trend_icon} <span class='value'>{trend_corr:.2f}</span> <span class='label'>Trend Corr.</span></p>
                </div>
                """
                st.markdown(card_html, unsafe_allow_html=True)


    # --- Store-Specific Action Plan ---
    st.markdown("#### Detailed Action Plan per Store")
    store_list_options = sorted(store_stats[COL_STORE_ID].unique()) if COL_STORE_ID in store_stats.columns else []


    if not store_list_options:
        st.info("No stores available to select for detailed plan.")
        return

    # Use session state for selectbox default
    default_store = st.session_state.get("category_store_select_val", store_list_options[0])
    # Ensure default is valid
    if default_store not in store_list_options: default_store = store_list_options[0]

    selected_store = st.selectbox(
        "Select a store:", store_list_options, index=store_list_options.index(default_store), key="category_store_select",
        on_change=lambda: st.session_state.update(category_store_select_val=st.session_state.category_store_select)
        )


    if selected_store:
        # Retrieve the row safely
        row_df = store_stats[store_stats[COL_STORE_ID] == selected_store]
        if row_df.empty:
             st.warning(f"Could not find data for selected store: {selected_store}")
             return

        row = row_df.iloc[0] # Get the first (and only) row as a Series
        # Safely get values from the row Series
        cat = row.get(COL_CATEGORY, CAT_UNCATEGORIZED)
        color = PERFORMANCE_CATEGORIES.get(cat, {}).get('color', '#757575')
        icon = PERFORMANCE_CATEGORIES.get(cat, {}).get('icon', '')
        avg_val = row.get(COL_AVG_ENGAGEMENT)
        corr = row.get(COL_TREND_CORR, 0.0)
        explanation = row.get(COL_EXPLANATION, "N/A")
        action_plan = row.get(COL_ACTION_PLAN, "N/A")

        # Determine trend description based on correlation thresholds
        if corr > TREND_STRONG_UP: trend_desc, trend_icon = "Strong positive trend", "🔼"
        elif corr > TREND_UP: trend_desc, trend_icon = "Mild positive trend", "↗️"
        elif corr < TREND_STRONG_DOWN: trend_desc, trend_icon = "Strong negative trend", "🔽"
        elif corr < TREND_DOWN: trend_desc, trend_icon = "Mild negative trend", "↘️"
        else: trend_desc, trend_icon = "Stable trend", "➡️"

        # Display the detailed card
        detail_html = f"""
        <div class='info-card' style='border-left-color: {color}; padding: 20px;'>
            <h3 style='color:{color}; margin-top:0;'>{icon} Store {selected_store} - {cat}</h3>
            <p><strong>Average Engagement (Period):</strong> {format_percentage(avg_val)}</p>
            <p><strong>Trend Correlation (Period):</strong> {trend_icon} {trend_desc} ({corr:.2f})</p>
            <p><strong>Explanation:</strong> {explanation}</p>
            <h4 style='color:{color}; margin-top:1em;'>Recommended Action Plan:</h4>
            <p>{action_plan}</p>
        </div>
        """
        st.markdown(detail_html, unsafe_allow_html=True)

        # --- Additional context for lower-performing categories ---
        if cat in [CAT_IMPROVING, CAT_INTERVENTION]:
            st.markdown("---")
            st.markdown("##### Improvement Opportunities & Context")
            # Suggest learning partners (Star Performers)
            top_stores = store_stats[store_stats[COL_CATEGORY] == CAT_STAR][COL_STORE_ID].tolist()
            if top_stores:
                partners = ", ".join(f"**Store {s}**" for s in top_stores)
                partner_color = PERFORMANCE_CATEGORIES.get(CAT_STAR, {}).get('color', '#2E7D32')
                st.markdown(f"<div class='info-card' style='border-left-color: {partner_color};'><h4 style='color:{partner_color}; margin-top:0;'>Potential Learning Partners</h4><p>Consider reviewing strategies from top performers: {partners}</p></div>", unsafe_allow_html=True)
            else:
                 st.markdown("<div class='info-card' style='border-left-color: #ccc;'><h4 style='margin-top:0;'>Potential Learning Partners</h4><p>No stores currently categorized as 'Star Performers' in this period.</p></div>", unsafe_allow_html=True)


            # Show gap to median engagement
            median_eng = store_stats[COL_AVG_ENGAGEMENT].median()
            current_eng = avg_val if avg_val is not None else 0
            # Ensure median is valid before calculating gain
            if pd.notna(median_eng):
                 gain = median_eng - current_eng
                 if gain > 0: # Only show if below median
                      gain_color = PERFORMANCE_CATEGORIES.get(CAT_IMPROVING, {}).get('color', '#1976D2')
                      st.markdown(f"<div class='info-card' style='border-left-color: {gain_color}; margin-top:15px;'><h4 style='color:{gain_color}; margin-top:0;'>Gap to Median</h4><p>Current average: <strong>{format_percentage(current_eng)}</strong> | District median: <strong>{format_percentage(median_eng)}</strong> | Potential gain to median: <strong>{gain:.2f}%</strong></p></div>", unsafe_allow_html=True)

                 else: # Already at or above median
                      st.markdown(f"<div class='info-card' style='border-left-color: #ccc; margin-top:15px;'><h4 style='margin-top:0;'>Gap to Median</h4><p>Store is already performing at or above the median engagement ({format_percentage(median_eng)}).</p></div>", unsafe_allow_html=True)

            else: # Median could not be calculated
                 st.markdown(f"<div class='info-card' style='border-left-color: #ccc; margin-top:15px;'><h4 style='margin-top:0;'>Gap to Median</h4><p>Median engagement could not be calculated.</p></div>", unsafe_allow_html=True)



def display_anomalies_insights_tab(anomalies_df: pd.DataFrame, recommendations_df: pd.DataFrame, z_threshold: float):
    """Displays the content for the Anomalies & Insights tab."""
    st.subheader("Anomaly Detection")
    st.write(f"This section highlights significant week-over-week changes in engagement, defined as changes exceeding a Z-score of **{z_threshold:.1f}** relative to the store's own historical weekly changes *within the selected filter period*.")


    if anomalies_df.empty:
        st.info(f"No significant anomalies detected (Z-score > {z_threshold:.1f}) for the selected stores and period.")
    else:
        st.markdown("#### Detected Anomalies")
        # Select and rename columns for display
        display_cols = [
            COL_STORE_ID, COL_WEEK, COL_ENGAGED_PCT, COL_CHANGE_PCT_PTS,
            COL_Z_SCORE, COL_RANK, COL_PREV_RANK, COL_POSSIBLE_EXPLANATION
        ]
        # Ensure columns exist before selecting/renaming
        anomalies_display = anomalies_df[[col for col in display_cols if col in anomalies_df.columns]].copy()
        rename_map = {
            COL_ENGAGED_PCT: 'Engagement %',
            COL_CHANGE_PCT_PTS: 'Change Pts',
            COL_Z_SCORE: 'Z-Score',
            COL_RANK: 'Current Rank',
            COL_PREV_RANK: 'Previous Rank',
            COL_POSSIBLE_EXPLANATION: 'Possible Explanation'
        }
        anomalies_display.rename(columns=rename_map, inplace=True)

        # Define column configuration for formatting
        column_config={
            "Engagement %": st.column_config.NumberColumn(format="%.2f%%"),
            "Change Pts": st.column_config.NumberColumn(format="%+.2f pts"), # Add units
            "Z-Score": st.column_config.NumberColumn(format="%.2f"),
             # Use format="d" for integers, handle potential NAs gracefully
            "Current Rank": st.column_config.NumberColumn(format="%d"),
            "Previous Rank": st.column_config.NumberColumn(format="%d"),
            "Possible Explanation": st.column_config.TextColumn(width="large") # Allow more width
        }
        # Remove config for columns that might be missing (like rank)
        final_config = {k: v for k, v in column_config.items() if k in anomalies_display.columns}


        st.dataframe(
             anomalies_display,
             column_config=final_config,
             hide_index=True,
             use_container_width=True
        )
        st.caption("Anomalies are sorted by the magnitude of the Z-score (most significant first).")


    st.subheader("Store-Specific Recommendations")
    st.write("Based on the overall performance category (calculated over the filtered period), the *current* trend within the filtered period, and any detected anomalies.")


    if recommendations_df.empty:
        st.info("No recommendations available. Ensure data is loaded and filters are applied.")
    else:
        # Rename columns for better display
         rec_display = recommendations_df.rename(columns={
             COL_AVG_ENGAGEMENT: 'Avg Engagement % (Period)',
             COL_CATEGORY: 'Category (Period)'
         })
         # Select column order
         rec_display_cols = [COL_STORE_ID, 'Category (Period)', 'Current Trend', 'Avg Engagement % (Period)', 'Recommendation']
         # Ensure columns exist before selecting
         rec_display = rec_display[[col for col in rec_display_cols if col in rec_display.columns]]

         st.dataframe(
              rec_display,
              column_config={
                   "Avg Engagement % (Period)": st.column_config.NumberColumn(format="%.2f%%"),
                   "Recommendation": st.column_config.TextColumn(width="large")
              },
              hide_index=True,
              use_container_width=True
         )


# --- Main Application Flow ---

def main():
    """Main function to run the Streamlit application."""
    # --- Page Config (Must be the first Streamlit command) ---
    st.set_page_config(
        page_title=APP_TITLE,
        layout=PAGE_LAYOUT,
        initial_sidebar_state=INITIAL_SIDEBAR_STATE
    )

    # --- Initialize Session State ---
    # Used to preserve widget states across reruns
    if "quarter_filter_idx" not in st.session_state: st.session_state.quarter_filter_idx = 0
    if "week_filter_val" not in st.session_state: st.session_state.week_filter_val = "All"
    if "store_filter_val" not in st.session_state: st.session_state.store_filter_val = []
    if "z_slider_val" not in st.session_state: st.session_state.z_slider_val = DEFAULT_Z_THRESHOLD
    if "ma_checkbox_val" not in st.session_state: st.session_state.ma_checkbox_val = DEFAULT_SHOW_MA
    if "trend_window_slider_val" not in st.session_state: st.session_state.trend_window_slider_val = DEFAULT_TREND_WINDOW
    if "heatmap_sort_idx" not in st.session_state: st.session_state.heatmap_sort_idx = 0
    if "heatmap_color_idx" not in st.session_state: st.session_state.heatmap_color_idx = 0
    if "heatmap_slider_val_start" not in st.session_state: st.session_state.heatmap_slider_val_start = None # Initialize later
    if "heatmap_slider_val_end" not in st.session_state: st.session_state.heatmap_slider_val_end = None
    if "category_store_select_val" not in st.session_state: st.session_state.category_store_select_val = None # Initialize later
    if "recent_trend_window_val" not in st.session_state: st.session_state.recent_trend_window_val = RECENT_TRENDS_WINDOW
    if "recent_trend_sensitivity_val" not in st.session_state: st.session_state.recent_trend_sensitivity_val = "Medium"


    # --- Apply CSS and Title ---
    st.markdown(APP_CSS, unsafe_allow_html=True)
    st.markdown(f"<h1 class='dashboard-title'>{APP_TITLE}</h1>", unsafe_allow_html=True)
    st.markdown("Analyze **Club Publix** engagement data. Upload weekly data to explore KPIs, trends, and opportunities across stores. Use sidebar filters to refine the view.")


    # --- Sidebar: File Uploaders (Create these only ONCE) ---
    st.sidebar.header("Data Input")
    # Use unique keys for file uploaders
    data_file = st.sidebar.file_uploader(
        "Upload engagement data (Excel or CSV)",
        type=['csv', 'xlsx'],
        key="data_uploader_widget", # Unique key
        help="Upload the primary data file containing weekly engagement percentages."
    )
    comp_file = st.sidebar.file_uploader(
        "Optional: Upload comparison data (prior period)",
        type=['csv', 'xlsx'],
        key="comp_uploader_widget", # Unique key
        help="Upload a similar file for a previous period (e.g., last year) for comparison."
    )


    # --- Data Loading ---
    # Initialize DataFrames to None
    df_all = None
    df_comp_all = None
    store_list = [] # Initialize empty store list

    # Process primary data file if uploaded
    if data_file:
        df_all = load_and_process_data(data_file)
        if df_all is not None and not df_all.empty:
             if COL_STORE_ID in df_all.columns:
                  store_list = sorted(df_all[COL_STORE_ID].unique())
             # Initialize session state defaults based on loaded data if not already set
             if st.session_state.heatmap_slider_val_start is None and COL_WEEK in df_all.columns:
                 weeks = sorted(df_all[COL_WEEK].unique())
                 if weeks:
                      st.session_state.heatmap_slider_val_start = weeks[0]
                      st.session_state.heatmap_slider_val_end = weeks[-1]
             if st.session_state.category_store_select_val is None and store_list:
                  st.session_state.category_store_select_val = store_list[0]

        else:
             st.error("Failed to load or process primary data file. Please check the file format, required columns (Store #, Week/Date, Engaged Transaction %), and ensure data exists.")
             # Keep df_all as None or empty to prevent further processing


    # Process comparison data file if uploaded
    if comp_file:
        df_comp_all = load_and_process_data(comp_file)
        if df_comp_all is None or df_comp_all.empty:
             st.warning("Failed to load or process comparison data file. It will be ignored.")
             df_comp_all = None # Ensure it's None if loading failed


    # --- Sidebar: Filters & Settings (Run this section based on whether data loaded) ---
    # Pass df_all and a boolean indicating if data_file exists and was loaded successfully
    quarter_choice, week_choice, store_choice, z_threshold, show_ma, trend_analysis_weeks = display_sidebar(
        df=df_all,
        data_file_exists=(df_all is not None and not df_all.empty) # Pass status of primary data
    )


    # --- Main Panel Logic ---
    # Show initial message if no data file is uploaded
    if not data_file:
        st.info("Please upload a primary engagement data file using the sidebar to begin analysis.")
        st.markdown("#### Required Columns")
        st.markdown(f"- `{COL_STORE_ID}`\n- `{COL_WEEK}` or `{COL_DATE}` (e.g., 'Week Ending')\n- `{COL_ENGAGED_PCT}`")
        st.markdown("#### Optional Columns")
        st.markdown(f"- `{COL_RANK}` (e.g., 'Weekly Rank')\n- `{COL_QTD_PCT}` (e.g., 'Quarter to Date %')")
        st.stop() # Stop execution until file is uploaded


    # Proceed only if primary data loaded successfully
    if df_all is None or df_all.empty:
         # Error message was already shown during loading if it failed
         st.warning("Cannot proceed with analysis as primary data is missing or invalid.")
         st.stop()


    # --- Data Filtering ---
    # Apply filters based on sidebar selections
    df_filtered = filter_dataframe(df_all, quarter_choice, week_choice, store_choice)
    df_comp_filtered = filter_dataframe(df_comp_all, quarter_choice, week_choice, store_choice) if df_comp_all is not None else None


    # Check if filtering resulted in empty data
    if df_filtered.empty:
        st.error("No data available for the selected filters (Quarter, Week, Store). Please adjust filters or check the uploaded data.")
        st.stop()


    # --- Perform Calculations on Filtered Data ---
    # Execute analysis functions only after data is loaded and filtered
    summary_data = get_executive_summary_data(df_filtered, df_all, store_choice, store_list, trend_analysis_weeks)
    store_perf_filtered = df_filtered.groupby(COL_STORE_ID)[COL_ENGAGED_PCT].mean() if COL_STORE_ID in df_filtered.columns and COL_ENGAGED_PCT in df_filtered.columns else pd.Series(dtype=float)
    key_insights = generate_key_insights(df_filtered, summary_data['store_trends'], store_perf_filtered)
    district_trend = calculate_district_trends(df_filtered)
    district_trend_comp = calculate_district_trends(df_comp_filtered) if df_comp_filtered is not None else None
    store_stats = calculate_performance_categories(df_filtered) # Categories based on filtered period
    anomalies_df = find_anomalies(df_filtered, z_threshold) # Anomalies within filtered period
    recommendations_df = generate_recommendations(df_filtered, store_stats, anomalies_df, trend_analysis_weeks)



    # --- Display Main Content ---
    display_executive_summary(summary_data)
    st.markdown("---") # Add a separator
    display_key_insights(key_insights)
    st.markdown("---") # Add a separator


    # --- Tabs ---
    # Define tab names with icons for better UI
    tab_titles = [
        "📊 Engagement Trends",
        "📈 Store Comparison",
        "📋 Store Performance", # Renamed for clarity
        "💡 Anomalies & Insights"
    ]
    tab1, tab2, tab3, tab4 = st.tabs(tab_titles)


    with tab1:
        display_engagement_trends_tab(df_filtered, df_comp_filtered, show_ma, district_trend, district_trend_comp)

    with tab2:
        display_store_comparison_tab(df_filtered, week_choice)

    with tab3:
        # Pass store_stats which contains category info
        display_performance_categories_tab(store_stats)

    with tab4:
        display_anomalies_insights_tab(anomalies_df, recommendations_df, z_threshold)


# --- Entry Point ---
if __name__ == "__main__":
    main()