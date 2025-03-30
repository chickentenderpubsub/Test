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
    }
    .info-card h4 { margin-top: 0; margin-bottom: 5px; }
    .info-card p { margin: 5px 0; }
    .info-card .value { font-weight: bold; }
    .info-card .label { font-size: 0.9em; color: #BBBBBB; }
    .info-card .change { font-size: 0.9em; }
    .info-card .trend-icon { font-size: 1.2em; margin-right: 5px; }
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
    CAT_STAR: {"icon": "⭐", "color": "#2E7D32", "explanation": "High engagement with stable or improving trend", "action": "Share best practices", "short_action": "Share best practices"},
    CAT_STABILIZE: {"icon": "⚠️", "color": "#F57C00", "explanation": "High engagement but recent downward trend", "action": "Investigate recent changes. Reinforce processes.", "short_action": "Reinforce successful processes"},
    CAT_IMPROVING: {"icon": "📈", "color": "#1976D2", "explanation": "Below average engagement but trending upward", "action": "Continue positive momentum. Intensify efforts.", "short_action": "Continue positive momentum"},
    CAT_INTERVENTION: {"icon": "🚨", "color": "#C62828", "explanation": "Below average engagement with flat or declining trend", "action": "Urgent attention needed. Develop improvement plan.", "short_action": "Needs comprehensive support"},
    CAT_UNCATEGORIZED: {"icon": "❓", "color": "#757575", "explanation": "Not enough data or unusual pattern", "action": "Monitor closely.", "short_action": "Monitor closely"},
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
    """Calculate mean of a series, handling potential empty series."""
    return series.mean() if not series.empty else None

def safe_max(series: pd.Series) -> Optional[Any]:
    """Get max value of a series, handling potential empty series."""
    return series.max() if not series.empty else None

def safe_min(series: pd.Series) -> Optional[Any]:
    """Get min value of a series, handling potential empty series."""
    return series.min() if not series.empty else None

def safe_idxmax(series: pd.Series) -> Optional[Any]:
    """Get index of max value, handling potential empty series."""
    return series.idxmax() if not series.empty else None

def safe_idxmin(series: pd.Series) -> Optional[Any]:
    """Get index of min value, handling potential empty series."""
    return series.idxmin() if not series.empty else None

def format_delta(value: Optional[float], unit: str = "%") -> str:
    """Format a delta value with sign and unit."""
    if value is None:
        return "N/A"
    return f"{value:+.2f}{unit}"

def format_percentage(value: Optional[float]) -> str:
    """Format a value as a percentage string."""
    if value is None:
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
            df = pd.read_csv(uploaded_file)
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
    df.columns = [col.strip() for col in df.columns]
    col_map = {}
    current_cols = df.columns.tolist()
    mapped_cols = set()

    # Prioritize specific date/week columns first
    date_col_found = False
    for col in current_cols:
        cl = col.lower()
        if 'week ending' in cl or cl == 'date':
            if COL_DATE not in mapped_cols:
                col_map[col] = COL_DATE
                mapped_cols.add(COL_DATE)
                date_col_found = True
                break # Found the preferred date column

    # Map other columns
    for col in current_cols:
        if col in col_map: continue # Already mapped
        cl = col.lower()
        # Map week only if date wasn't found and week hasn't been mapped
        if 'week' in cl and not date_col_found and COL_WEEK not in mapped_cols:
             # Check if it looks like a week number (e.g., "Week 1", "Week")
             # Avoid mapping things like "Weekly Rank" here
            if cl == 'week' or cl.startswith('week '):
                 col_map[col] = COL_WEEK
                 mapped_cols.add(COL_WEEK)
        else:
            for pattern, target_col in INPUT_COLUMN_MAP_PATTERNS.items():
                if pattern in cl and target_col not in mapped_cols:
                    # Avoid mapping 'week' if it's part of 'weekly rank' etc.
                    if pattern == 'week' and ('rank' in cl or 'ending' in cl):
                        continue
                    col_map[col] = target_col
                    mapped_cols.add(target_col)
                    break # Move to next column once mapped

    df = df.rename(columns=col_map)
    # Keep only potentially mapped columns + original unmapped ones if needed
    # For simplicity here, we assume the mapped columns are the primary ones needed.
    # A more robust approach might involve explicitly listing required output columns.
    return df

def preprocess_data(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Performs type conversions, cleaning, and adds derived columns."""
    # Convert percentages
    for percent_col in [COL_ENGAGED_PCT, COL_QTD_PCT]:
        if percent_col in df.columns:
            # Convert to string, remove '%', then convert to numeric
            df[percent_col] = pd.to_numeric(df[percent_col].astype(str).str.replace('%', '', regex=False), errors='coerce')

    # Ensure essential column exists and drop rows where it's NaN
    if COL_ENGAGED_PCT not in df.columns:
         st.error(f"Essential column '{COL_ENGAGED_PCT}' not found after standardization. Please check input file.")
         return None
    df = df.dropna(subset=[COL_ENGAGED_PCT])
    if df.empty:
        st.warning("No valid data remaining after removing rows with missing engagement percentage.")
        return None

    # Handle Date/Week and Quarter derivation
    if COL_DATE in df.columns:
        df[COL_DATE] = pd.to_datetime(df[COL_DATE], errors='coerce')
        df = df.dropna(subset=[COL_DATE])
        if df.empty:
             st.warning("No valid data remaining after handling dates.")
             return None
        df[COL_WEEK] = df[COL_DATE].dt.isocalendar().week.astype(int) # Use ISO week
        df[COL_QUARTER] = df[COL_DATE].dt.quarter.astype(int)
    elif COL_WEEK in df.columns:
        # Ensure Week is numeric integer
        df[COL_WEEK] = pd.to_numeric(df[COL_WEEK], errors='coerce')
        df = df.dropna(subset=[COL_WEEK])
        if df.empty:
             st.warning("No valid data remaining after handling week numbers.")
             return None
        df[COL_WEEK] = df[COL_WEEK].astype(int)
        # Derive Quarter from Week (approximate)
        df[COL_QUARTER] = ((df[COL_WEEK] - 1) // 13 + 1).astype(int)
    else:
        st.error(f"Missing required time column: Neither '{COL_DATE}' nor '{COL_WEEK}' found.")
        return None

    # Convert other columns
    if COL_RANK in df.columns:
        df[COL_RANK] = pd.to_numeric(df[COL_RANK], errors='coerce').astype('Int64') # Use nullable Int
    if COL_STORE_ID in df.columns:
        df[COL_STORE_ID] = df[COL_STORE_ID].astype(str)
    else:
         st.error(f"Missing required '{COL_STORE_ID}' column.")
         return None

    # Sort and return
    df = df.sort_values([COL_WEEK, COL_STORE_ID])
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

def filter_dataframe(df: pd.DataFrame, quarter_choice: str, week_choice: str, store_choice: List[str]) -> pd.DataFrame:
    """Filters the DataFrame based on sidebar selections."""
    if df is None or df.empty:
        return pd.DataFrame() # Return empty DataFrame if input is invalid

    df_filtered = df.copy()

    # Filter by Quarter
    if quarter_choice != "All":
        try:
            q_num = int(quarter_choice.replace('Q', ''))
            if COL_QUARTER in df_filtered.columns:
                df_filtered = df_filtered[df_filtered[COL_QUARTER] == q_num]
            else:
                st.warning("Quarter column not available for filtering.")
        except ValueError:
            st.warning(f"Invalid Quarter selection: {quarter_choice}")

    # Filter by Week
    if week_choice != "All":
        try:
            week_num = int(week_choice)
            if COL_WEEK in df_filtered.columns:
                df_filtered = df_filtered[df_filtered[COL_WEEK] == week_num]
            else:
                st.warning("Week column not available for filtering.")
        except ValueError:
            st.warning(f"Invalid Week selection: {week_choice}")

    # Filter by Store
    if store_choice: # If list is not empty
        if COL_STORE_ID in df_filtered.columns:
            df_filtered = df_filtered[df_filtered[COL_STORE_ID].isin(store_choice)]
        else:
            st.warning("Store ID column not available for filtering.")

    return df_filtered

# --- Analysis Functions ---

def calculate_trend_slope(group: pd.DataFrame, window: int) -> float:
    """Calculates the slope of the engagement trend over a window."""
    if len(group) < 2: return 0.0
    data = group.sort_values(COL_WEEK).tail(window)
    if len(data) < 2: return 0.0
    # Use simple linear regression slope
    # Center x values for potentially better numerical stability
    x = data[COL_WEEK].values - np.mean(data[COL_WEEK].values)
    y = data[COL_ENGAGED_PCT].values
    # Check for NaNs in y after filtering
    if np.isnan(y).any(): return 0.0
    # Check for constant x or y values which polyfit handles poorly
    if np.all(x == x[0]) or np.all(y == y[0]): return 0.0

    try:
        slope = np.polyfit(x, y, 1)[0]
        return slope if not np.isnan(slope) else 0.0
    except (np.linalg.LinAlgError, ValueError):
        # Handle cases where polyfit might fail
        return 0.0


def classify_trend(slope: float) -> str:
    """Classifies trend based on slope value."""
    if slope > TREND_STRONG_UP: return "Strong Upward"
    elif slope > TREND_UP: return "Upward"
    elif slope < TREND_STRONG_DOWN: return "Strong Downward"
    elif slope < TREND_DOWN: return "Downward"
    else: return "Stable"

def calculate_store_trends(df: pd.DataFrame, window: int) -> pd.Series:
    """Calculates the trend classification for each store."""
    if df.empty or COL_STORE_ID not in df.columns or COL_WEEK not in df.columns or COL_ENGAGED_PCT not in df.columns:
        return pd.Series(dtype=str)

    store_slopes = df.groupby(COL_STORE_ID).apply(lambda g: calculate_trend_slope(g, window))
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
    if df_filtered.empty or COL_WEEK not in df_filtered.columns or COL_ENGAGED_PCT not in df_filtered.columns:
        return summary

    # Determine current and previous week
    available_weeks = sorted(df_filtered[COL_WEEK].unique())
    if not available_weeks: return summary

    summary['current_week'] = int(available_weeks[-1])
    if len(available_weeks) > 1:
        summary['prev_week'] = int(available_weeks[-2])
    else:
        # Look for previous week in the *unfiltered* data within the same quarter if applicable
        current_quarter = df_filtered[COL_QUARTER].iloc[0] if COL_QUARTER in df_filtered.columns and not df_filtered.empty else None
        prev_weeks_all = sorted(df_all[df_all[COL_WEEK] < summary['current_week']][COL_WEEK].unique())
        if current_quarter:
            prev_weeks_all = sorted(df_all[(df_all[COL_WEEK] < summary['current_week']) & (df_all[COL_QUARTER] == current_quarter)][COL_WEEK].unique())

        if prev_weeks_all:
            summary['prev_week'] = int(prev_weeks_all[-1])


    # Calculate averages
    current_data = df_filtered[df_filtered[COL_WEEK] == summary['current_week']]
    summary['current_avg'] = safe_mean(current_data[COL_ENGAGED_PCT])

    if summary['prev_week'] is not None:
        # Use df_all to get previous week data if it wasn't in the filtered set (e.g., single week selected)
        prev_data = df_all[df_all[COL_WEEK] == summary['prev_week']]
        # Apply store filter if necessary
        if store_choice:
             prev_data = prev_data[prev_data[COL_STORE_ID].isin(store_choice)]
        summary['prev_avg'] = safe_mean(prev_data[COL_ENGAGED_PCT])


    # Calculate Top/Bottom Performers for the current week
    if not current_data.empty:
        store_perf_current = current_data.groupby(COL_STORE_ID)[COL_ENGAGED_PCT].mean()
        summary['top_store'] = safe_idxmax(store_perf_current)
        summary['bottom_store'] = safe_idxmin(store_perf_current)
        summary['top_val'] = safe_max(store_perf_current)
        summary['bottom_val'] = safe_min(store_perf_current)

    # Calculate Trends (using all data in the filtered period for trend calculation)
    summary['store_trends'] = calculate_store_trends(df_filtered, trend_window)

    # Determine Average Label
    if store_choice and len(store_choice) == 1:
        summary['avg_label'] = f"Store {store_choice[0]} Engagement"
    elif store_choice and len(store_choice) < len(all_stores_list):
        summary['avg_label'] = "Selected Stores Avg Engagement"

    # Calculate Delta and Trend Direction/Class
    if summary['current_avg'] is not None and summary['prev_avg'] is not None:
        summary['delta_val'] = summary['current_avg'] - summary['prev_avg']
        if summary['delta_val'] > 0.01: # Add small tolerance
            summary['trend_dir'] = "up"
            summary['trend_class'] = "highlight-good"
        elif summary['delta_val'] < -0.01:
            summary['trend_dir'] = "down"
            summary['trend_class'] = "highlight-bad"

    return summary

def generate_key_insights(df_filtered: pd.DataFrame, store_trends: pd.Series, store_perf: pd.Series) -> List[str]:
    """Generates a list of key insight strings."""
    insights = []
    if df_filtered.empty or store_trends.empty or store_perf.empty or len(store_perf) < 1:
        return ["No data available for insights."]

    # 1. Consistency Insights
    if COL_STORE_ID in df_filtered.columns and COL_ENGAGED_PCT in df_filtered.columns:
        store_std = df_filtered.groupby(COL_STORE_ID)[COL_ENGAGED_PCT].std().fillna(0)
        if not store_std.empty:
            most_consistent = safe_idxmin(store_std)
            least_consistent = safe_idxmax(store_std)
            if most_consistent:
                insights.append(f"**Store {most_consistent}** shows the most consistent engagement (std: {store_std[most_consistent]:.2f}%).")
            if least_consistent:
                insights.append(f"**Store {least_consistent}** has the most variable engagement (std: {store_std[least_consistent]:.2f}%).")

    # 2. Trending Stores Insights
    trending_up = store_trends[store_trends.isin(["Upward", "Strong Upward"])].index.tolist()
    trending_down = store_trends[store_trends.isin(["Downward", "Strong Downward"])].index.tolist()
    if trending_up:
        insights.append("Stores showing positive trends: " + ", ".join(f"**{s}**" for s in trending_up))
    if trending_down:
        insights.append("Stores needing attention (downward trend): " + ", ".join(f"**{s}**" for s in trending_down))

    # 3. Performance Gap Insights
    if len(store_perf) > 1:
        top_val = safe_max(store_perf)
        bottom_val = safe_min(store_perf)
        if top_val is not None and bottom_val is not None:
            gap = top_val - bottom_val
            insights.append(f"Gap between highest ({top_val:.2f}%) and lowest ({bottom_val:.2f}%) performing stores: **{gap:.2f}%**")
            if gap > 10: # Arbitrary threshold for large gap
                insights.append("🚨 Large performance gap suggests opportunities for knowledge sharing from top performers.")

    return insights[:5] # Limit to top 5 insights

def calculate_moving_averages(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates the 4-week moving average for engagement."""
    if df is None or df.empty or COL_ENGAGED_PCT not in df.columns:
        # Ensure the column exists even if empty or calculation fails
        if df is not None:
             df[COL_MA_4W] = np.nan
        return df if df is not None else pd.DataFrame()


    df = df.sort_values([COL_STORE_ID, COL_WEEK])
    # Calculate MA within each store group
    # Handle potential grouping issues if COL_STORE_ID is missing (though checked earlier)
    if COL_STORE_ID in df.columns:
        df[COL_MA_4W] = df.groupby(COL_STORE_ID)[COL_ENGAGED_PCT].transform(
            lambda s: s.rolling(window=4, min_periods=1).mean()
        )
    else:
         df[COL_MA_4W] = np.nan # Assign NaN if grouping column is missing
    return df

def calculate_district_trends(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates the district average and its moving average."""
    if df is None or df.empty or COL_WEEK not in df.columns or COL_ENGAGED_PCT not in df.columns:
        return pd.DataFrame(columns=[COL_WEEK, 'Average Engagement %', COL_MA_4W])

    dist_trend = df.groupby(COL_WEEK, as_index=False)[COL_ENGAGED_PCT].mean()
    dist_trend = dist_trend.rename(columns={COL_ENGAGED_PCT: 'Average Engagement %'})
    dist_trend = dist_trend.sort_values(COL_WEEK)
    dist_trend[COL_MA_4W] = dist_trend['Average Engagement %'].rolling(window=4, min_periods=1).mean()
    return dist_trend


def calculate_performance_categories(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates performance stats and assigns categories to stores."""
    required_cols = [COL_STORE_ID, COL_WEEK, COL_ENGAGED_PCT]
    if df.empty or not all(col in df.columns for col in required_cols):
        return pd.DataFrame(columns=[COL_STORE_ID, COL_AVG_ENGAGEMENT, COL_CONSISTENCY, COL_TREND_CORR, COL_CATEGORY, COL_ACTION_PLAN, COL_EXPLANATION])

    store_stats = df.groupby(COL_STORE_ID)[COL_ENGAGED_PCT].agg(['mean', 'std']).reset_index()
    store_stats.columns = [COL_STORE_ID, COL_AVG_ENGAGEMENT, COL_CONSISTENCY]
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
    # Handle case where median might be NaN if all averages are NaN (unlikely but possible)
    if pd.isna(median_engagement):
         median_engagement = 0 # Assign a default, though categorization might be less meaningful

    conditions = [
        (store_stats[COL_AVG_ENGAGEMENT] >= median_engagement) & (store_stats[COL_TREND_CORR] < TREND_DOWN), # High Perf, Declining Trend
        (store_stats[COL_AVG_ENGAGEMENT] >= median_engagement), # High Perf, Stable/Improving Trend
        (store_stats[COL_AVG_ENGAGEMENT] < median_engagement) & (store_stats[COL_TREND_CORR] > TREND_UP),   # Low Perf, Improving Trend
        (store_stats[COL_AVG_ENGAGEMENT] < median_engagement)  # Low Perf, Stable/Declining Trend
    ]
    choices = [CAT_STABILIZE, CAT_STAR, CAT_IMPROVING, CAT_INTERVENTION]
    store_stats[COL_CATEGORY] = np.select(conditions, choices, default=CAT_UNCATEGORIZED).astype(str)

    # Map explanations and action plans using .get for safety
    store_stats[COL_ACTION_PLAN] = store_stats[COL_CATEGORY].map(lambda x: PERFORMANCE_CATEGORIES.get(x, {}).get('action', 'N/A'))
    store_stats[COL_EXPLANATION] = store_stats[COL_CATEGORY].map(lambda x: PERFORMANCE_CATEGORIES.get(x, {}).get('explanation', 'N/A'))


    return store_stats


def find_anomalies(df: pd.DataFrame, z_threshold: float) -> pd.DataFrame:
    """Detects anomalies based on week-over-week changes using Z-score."""
    required_cols = [COL_STORE_ID, COL_WEEK, COL_ENGAGED_PCT]
    if df.empty or not all(col in df.columns for col in required_cols):
        return pd.DataFrame()

    anomalies = []
    df = df.sort_values([COL_STORE_ID, COL_WEEK])

    for store_id, grp in df.groupby(COL_STORE_ID):
        grp = grp.reset_index() # Keep original index if needed, but easier to work with 0-based index here
        diffs = grp[COL_ENGAGED_PCT].diff() # Keep first NaN
        if len(diffs.dropna()) < 2: continue # Need at least 2 differences to calculate std dev

        mean_diff = diffs.mean()
        std_diff = diffs.std()

        # Avoid division by zero or near-zero std dev
        if std_diff == 0 or pd.isna(std_diff) or std_diff < 1e-6: continue

        # Iterate through differences, starting from the second row (index 1)
        for i in range(1, len(grp)):
            diff_val = diffs.iloc[i]
            if pd.isna(diff_val): continue

            z = (diff_val - mean_diff) / std_diff

            if abs(z) >= z_threshold:
                current_row = grp.iloc[i]
                prev_row = grp.iloc[i-1]

                rank_cur = int(current_row[COL_RANK]) if COL_RANK in grp.columns and pd.notna(current_row[COL_RANK]) else None
                rank_prev = int(prev_row[COL_RANK]) if COL_RANK in grp.columns and pd.notna(prev_row[COL_RANK]) else None

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
        return pd.DataFrame()

    anomalies_df = pd.DataFrame(anomalies)

    # Add explanations based on change and rank change
    anomalies_df[COL_POSSIBLE_EXPLANATION] = np.where(
        anomalies_df[COL_CHANGE_PCT_PTS] >= 0,
        "Engagement spiked significantly. Possible promotion or event impact.",
        "Sharp drop in engagement. Potential system issue or loss of engagement."
    )

    # Add rank change details if available
    improve_mask = (anomalies_df[COL_CHANGE_PCT_PTS] >= 0) & anomalies_df[COL_PREV_RANK].notna() & anomalies_df[COL_RANK].notna() & (anomalies_df[COL_PREV_RANK] > anomalies_df[COL_RANK])
    decline_mask = (anomalies_df[COL_CHANGE_PCT_PTS] < 0) & anomalies_df[COL_PREV_RANK].notna() & anomalies_df[COL_RANK].notna() & (anomalies_df[COL_PREV_RANK] < anomalies_df[COL_RANK])

    # Use .loc for safe assignment
    if improve_mask.any():
        anomalies_df.loc[improve_mask, COL_POSSIBLE_EXPLANATION] += " (Improved from rank " + anomalies_df.loc[improve_mask, COL_PREV_RANK].astype(int).astype(str) + " to " + anomalies_df.loc[improve_mask, COL_RANK].astype(int).astype(str) + ".)"
    if decline_mask.any():
        anomalies_df.loc[decline_mask, COL_POSSIBLE_EXPLANATION] += " (Dropped from rank " + anomalies_df.loc[decline_mask, COL_PREV_RANK].astype(int).astype(str) + " to " + anomalies_df.loc[decline_mask, COL_RANK].astype(int).astype(str) + ".)"


    # Sort by absolute Z-score and format
    anomalies_df['Abs Z'] = anomalies_df[COL_Z_SCORE].abs()
    anomalies_df = anomalies_df.sort_values('Abs Z', ascending=False).drop(columns=['Abs Z'])
    anomalies_df[[COL_ENGAGED_PCT, COL_Z_SCORE, COL_CHANGE_PCT_PTS]] = anomalies_df[[COL_ENGAGED_PCT, COL_Z_SCORE, COL_CHANGE_PCT_PTS]].round(2)

    return anomalies_df


def generate_recommendations(df_filtered: pd.DataFrame, store_stats: pd.DataFrame, anomalies_df: pd.DataFrame, trend_window: int) -> pd.DataFrame:
    """Generates store-specific recommendations based on category, trend, and anomalies."""
    recommendations = []
    all_store_ids = sorted(df_filtered[COL_STORE_ID].unique()) if COL_STORE_ID in df_filtered.columns else []

    if not all_store_ids or df_filtered.empty:
        return pd.DataFrame()

    # Recalculate trends based on the *filtered* data for current context
    store_trends_filtered = calculate_store_trends(df_filtered, trend_window)

    for store_id in all_store_ids:
        store_data_filtered = df_filtered[df_filtered[COL_STORE_ID] == store_id]
        if store_data_filtered.empty: continue

        avg_eng = safe_mean(store_data_filtered[COL_ENGAGED_PCT])
        trend = store_trends_filtered.get(store_id, "Stable") # Get trend for this store

        # Get category from pre-calculated stats
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

        # Refine recommendation based on *current* trend in filtered data
        if category == CAT_STAR and trend in ["Downward", "Strong Downward"]:
             rec = "High performer, but recent trend is down. Investigate potential causes."
        elif category == CAT_INTERVENTION and trend in ["Upward", "Strong Upward"]:
             rec = "Showing recent improvement! Continue efforts and monitor closely."
        elif category == CAT_STABILIZE and trend not in ["Downward", "Strong Downward"]:
             rec = "Stabilization efforts may be working. Maintain focus on consistency."
        elif category == CAT_IMPROVING and trend not in ["Upward", "Strong Upward"]:
             rec = "Improvement seems stalled. Re-evaluate strategies or seek support."


        # Append anomaly note if significant anomaly exists for this store in the filtered period
        store_anoms = anomalies_df[anomalies_df[COL_STORE_ID] == store_id] if not anomalies_df.empty else pd.DataFrame()
        if not store_anoms.empty:
            # Check if the anomaly week is within the filtered weeks
            filtered_weeks = df_filtered[COL_WEEK].unique()
            relevant_anoms = store_anoms[store_anoms[COL_WEEK].isin(filtered_weeks)]
            if not relevant_anoms.empty:
                biggest_relevant = relevant_anoms.iloc[0] # Already sorted by Z-score
                change_type = 'positive spike' if biggest_relevant[COL_CHANGE_PCT_PTS] > 0 else 'negative drop'
                rec += f" Note: Investigate significant {change_type} in Week {int(biggest_relevant[COL_WEEK])} (Z={biggest_relevant[COL_Z_SCORE]:.1f})."

        recommendations.append({
            COL_STORE_ID: store_id,
            COL_CATEGORY: category,
            'Current Trend': trend, # Trend based on filtered data
            COL_AVG_ENGAGEMENT: round(avg_eng, 2) if avg_eng is not None else None,
            'Recommendation': rec
        })

    return pd.DataFrame(recommendations)


def calculate_recent_performance_trends(df: pd.DataFrame, trend_window: int, momentum_threshold: float) -> pd.DataFrame:
    """Analyzes short-term trends (improving, stable, declining) over a recent window."""
    directions = []
    required_cols = [COL_STORE_ID, COL_WEEK, COL_ENGAGED_PCT]
    if df.empty or not all(col in df.columns for col in required_cols):
        return pd.DataFrame()

    for store_id, data in df.groupby(COL_STORE_ID):
        if len(data) < trend_window: continue

        recent = data.sort_values(COL_WEEK).tail(trend_window)
        vals = recent[COL_ENGAGED_PCT].values
        # Ensure no NaNs in the values used for calculation
        if pd.isna(vals).any(): continue

        # Simple comparison of first vs second half mean (or start vs end for small windows)
        if trend_window <= 3:
            # Check if enough values exist
            if len(vals) < 2: continue
            first_half_mean = vals[0]
            second_half_mean = vals[-1]
        else:
            split_point = trend_window // 2
            # Ensure enough values for split means
            if len(vals) < trend_window: continue # Should be handled by outer check, but safety first
            first_half_mean = vals[:split_point].mean()
            second_half_mean = vals[-split_point:].mean()

        change = second_half_mean - first_half_mean
        start_val = recent.iloc[0][COL_ENGAGED_PCT]
        current_val = recent.iloc[-1][COL_ENGAGED_PCT]
        total_change = current_val - start_val
        slope = calculate_trend_slope(recent, trend_window) # Use existing slope calculation

        # Classify direction based on the change between halves
        if abs(change) < momentum_threshold:
            direction, strength, color = "Stable", "Holding Steady", PERFORMANCE_CATEGORIES.get(CAT_STABILIZE, {}).get('color', '#757575') # Use neutral color
        elif change > 0:
            direction = "Improving"
            strength = "Strong Improvement" if change > 2 * momentum_threshold else "Gradual Improvement"
            color = PERFORMANCE_CATEGORIES.get(CAT_IMPROVING, {}).get('color', '#1976D2') # Use improving color
        else:
            direction = "Declining"
            strength = "Significant Decline" if change < -2 * momentum_threshold else "Gradual Decline"
            color = PERFORMANCE_CATEGORIES.get(CAT_INTERVENTION, {}).get('color', '#C62828') # Use declining color

        # Choose indicator based on direction and strength
        if direction == "Improving":
            indicator = "🔼" if "Strong" in strength else "↗️"
        elif direction == "Declining":
            indicator = "🔽" if "Significant" in strength else "↘️"
        else:
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

    return pd.DataFrame(directions)


# --- Charting Functions ---

def create_engagement_trend_chart(df_plot: pd.DataFrame, dist_trend: pd.DataFrame, df_comp_plot: Optional[pd.DataFrame], dist_trend_comp: Optional[pd.DataFrame], show_ma: bool, view_option: str, stores_to_show: Optional[List[str]] = None) -> Optional[alt.LayerChart]:
    """Creates the layered Altair chart for engagement trends."""
    if df_plot.empty or COL_WEEK not in df_plot.columns or COL_ENGAGED_PCT not in df_plot.columns:
        return None

    layers = []
    color_scale = alt.Scale(scheme='category10') # Consistent color scheme

    # Tooltip definition
    tooltip_base = [
        alt.Tooltip(COL_STORE_ID, title='Store'),
        alt.Tooltip(COL_WEEK, title='Week', type='ordinal'), # Treat week as ordinal for tooltip
        alt.Tooltip(COL_ENGAGED_PCT, format='.2f', title='Engaged %')
    ]
    tooltip_ma = [
        alt.Tooltip(COL_STORE_ID, title='Store'),
        alt.Tooltip(COL_WEEK, title='Week', type='ordinal'),
        alt.Tooltip(COL_MA_4W, format='.2f', title='4W MA')
    ]
    tooltip_dist = [
        alt.Tooltip(COL_WEEK, title='Week', type='ordinal'),
        alt.Tooltip('Average Engagement %', format='.2f', title='District Avg')
    ]
    tooltip_dist_ma = [
        alt.Tooltip(COL_WEEK, title='Week', type='ordinal'),
        alt.Tooltip(COL_MA_4W, format='.2f', title='District 4W MA')
    ]
    tooltip_dist_comp = [
         alt.Tooltip(COL_WEEK, title='Week', type='ordinal'),
         alt.Tooltip('Average Engagement %', format='.2f', title='Comp. Period Avg')
    ]
    tooltip_dist_comp_ma = [
         alt.Tooltip(COL_WEEK, title='Week', type='ordinal'),
         alt.Tooltip(COL_MA_4W, format='.2f', title='Comp. Period 4W MA')
    ]


    # --- Store Lines ---
    if view_option == "Custom Selection":
        if stores_to_show:
            data_sel = df_plot[df_plot[COL_STORE_ID].isin(stores_to_show)]
            if not data_sel.empty:
                # Main line
                layers.append(alt.Chart(data_sel).mark_line(strokeWidth=3).encode(
                    x=alt.X(f'{COL_WEEK}:O', title='Week'), # Treat week as ordinal on axis
                    y=alt.Y(f'{COL_ENGAGED_PCT}:Q', title='Engaged Transaction %'),
                    color=alt.Color(f'{COL_STORE_ID}:N', scale=color_scale, title="Store"),
                    tooltip=tooltip_base
                ))
                # Points on line
                layers.append(alt.Chart(data_sel).mark_point(filled=True, size=80).encode(
                    x=f'{COL_WEEK}:O', y=f'{COL_ENGAGED_PCT}:Q',
                    color=alt.Color(f'{COL_STORE_ID}:N', scale=color_scale),
                    tooltip=tooltip_base
                ))
                # Moving Average line
                if show_ma and COL_MA_4W in data_sel.columns and not data_sel[COL_MA_4W].isna().all():
                    layers.append(alt.Chart(data_sel).mark_line(strokeDash=[2,2], strokeWidth=2).encode(
                        x=f'{COL_WEEK}:O', y=alt.Y(f'{COL_MA_4W}:Q', title='4W MA'),
                        color=alt.Color(f'{COL_STORE_ID}:N', scale=color_scale),
                        tooltip=tooltip_ma
                    ))
        else:
            # If custom selection is chosen but no stores are selected, don't add store lines
            pass
    else: # All Stores or Recent Trends view
        # Interactive legend selection
        store_sel = alt.selection_point(fields=[COL_STORE_ID], bind='legend')

        # Main store lines (opacity changes on selection)
        store_line = alt.Chart(df_plot).mark_line(strokeWidth=1.5).encode(
            x=alt.X(f'{COL_WEEK}:O', title='Week'),
            y=alt.Y(f'{COL_ENGAGED_PCT}:Q', title='Engaged Transaction %'),
            color=alt.Color(f'{COL_STORE_ID}:N', scale=color_scale, title="Store"),
            opacity=alt.condition(store_sel, alt.value(1), alt.value(0.2)),
            strokeWidth=alt.condition(store_sel, alt.value(3), alt.value(1.5)),
            tooltip=tooltip_base
        ).add_params(store_sel)
        layers.append(store_line)

        # Moving average lines (opacity changes on selection)
        if show_ma and COL_MA_4W in df_plot.columns and not df_plot[COL_MA_4W].isna().all():
            ma_line = alt.Chart(df_plot).mark_line(strokeDash=[2,2], strokeWidth=1.5).encode(
                x=f'{COL_WEEK}:O', y=alt.Y(f'{COL_MA_4W}:Q', title='4W MA'),
                color=alt.Color(f'{COL_STORE_ID}:N', scale=color_scale),
                opacity=alt.condition(store_sel, alt.value(0.8), alt.value(0.1)),
                tooltip=tooltip_ma
            ).add_params(store_sel) # Link opacity to the same legend selection
            layers.append(ma_line)

    # --- District Average Lines ---
    if dist_trend is not None and not dist_trend.empty:
        # District Average
        layers.append(alt.Chart(dist_trend).mark_line(color='black', strokeDash=[4,2], size=3).encode(
            x=alt.X(f'{COL_WEEK}:O', title='Week'),
            y=alt.Y('Average Engagement %:Q', title='Engaged Transaction %'), # Use shared Y axis title
            tooltip=tooltip_dist
        ).properties(title="District Average")) # Add title for clarity in potential combined legend

        # District Moving Average
        if show_ma and COL_MA_4W in dist_trend.columns and not dist_trend[COL_MA_4W].isna().all():
            layers.append(alt.Chart(dist_trend).mark_line(color='black', strokeDash=[1,1], size=2, opacity=0.7).encode(
                x=f'{COL_WEEK}:O', y=f'{COL_MA_4W}:Q',
                tooltip=tooltip_dist_ma
            ).properties(title="District 4W MA"))

    # --- Comparison Period District Lines ---
    # Check df_comp_plot as well, as dist_trend_comp might be calculated on empty df
    if dist_trend_comp is not None and not dist_trend_comp.empty and df_comp_plot is not None and not df_comp_plot.empty:
        # Comparison District Average
        layers.append(alt.Chart(dist_trend_comp).mark_line(color='#555555', strokeDash=[4,2], size=2).encode(
            x=alt.X(f'{COL_WEEK}:O', title='Week'),
            y=alt.Y('Average Engagement %:Q'), # Shared Y axis
            tooltip=tooltip_dist_comp
        ).properties(title="Comparison Period Avg"))

        # Comparison District Moving Average
        if show_ma and COL_MA_4W in dist_trend_comp.columns and not dist_trend_comp[COL_MA_4W].isna().all():
            layers.append(alt.Chart(dist_trend_comp).mark_line(color='#555555', strokeDash=[1,1], size=1.5, opacity=0.7).encode(
                x=f'{COL_WEEK}:O', y=f'{COL_MA_4W}:Q',
                tooltip=tooltip_dist_comp_ma
            ).properties(title="Comparison Period 4W MA"))


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


    # Rename for Altair field name validity if needed (Store # is fine)
    df_heatmap = df_heatmap.rename(columns={COL_STORE_ID: 'StoreID', COL_ENGAGED_PCT: 'EngagedPct'})

    # Determine store sort order
    if sort_method == "Average Engagement":
        store_order = df_heatmap.groupby('StoreID')['EngagedPct'].mean().sort_values(ascending=False).index.tolist()
    else: # Recent Performance
        most_recent_week = safe_max(df_heatmap[COL_WEEK])
        if most_recent_week is None:
             store_order = sorted(df_heatmap['StoreID'].unique()) # Fallback sort
        else:
             # Sort by performance in the most recent week available in the heatmap data
             store_order = df_heatmap[df_heatmap[COL_WEEK] == most_recent_week].sort_values('EngagedPct', ascending=False)['StoreID'].tolist()
             # Add stores present in heatmap but not in the last week (e.g., if they stopped reporting), sorted alphabetically
             stores_in_last_week = set(store_order)
             all_stores_in_heatmap = set(df_heatmap['StoreID'].unique())
             missing_stores = sorted(list(all_stores_in_heatmap - stores_in_last_week))
             store_order.extend(missing_stores)


    num_stores = len(store_order)
    chart_height = max(CHART_HEIGHT_SHORT, HEATMAP_ROW_HEIGHT * num_stores) # Dynamic height

    heatmap_chart = alt.Chart(df_heatmap).mark_rect().encode(
        x=alt.X(f'{COL_WEEK}:O', title='Week'),
        y=alt.Y('StoreID:O', title='Store', sort=store_order),
        color=alt.Color('EngagedPct:Q', title='Engaged %', scale=alt.Scale(scheme=color_scheme), legend=alt.Legend(orient='right')),
        tooltip=['StoreID', alt.Tooltip(f'{COL_WEEK}:O', title='Week'), alt.Tooltip('EngagedPct:Q', format='.2f', title='Engaged %')]
    ).properties(
        height=chart_height
    )

    return heatmap_chart


def create_comparison_bar_chart(comp_data: pd.DataFrame, district_avg: Optional[float], title: str) -> Optional[alt.LayerChart]:
    """Creates the store comparison bar chart with average line."""
    required_cols = [COL_STORE_ID, COL_ENGAGED_PCT]
    if comp_data.empty or not all(col in comp_data.columns for col in required_cols):
        return None

    num_stores = len(comp_data[COL_STORE_ID].unique())
    chart_height = max(CHART_HEIGHT_SHORT, COMPARISON_BAR_HEIGHT * num_stores) # Dynamic height

    bar_chart = alt.Chart(comp_data).mark_bar().encode(
        y=alt.Y(f'{COL_STORE_ID}:N', title='Store', sort='-x'), # Sort descending by engagement
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
            tooltip=[alt.Tooltip('avg:Q', title='District Average', format='.2f')]
        )
        return alt.layer(bar_chart, avg_rule)
    else:
        # Return only the bar chart if average is not available
        return bar_chart


def create_relative_comparison_chart(comp_data: pd.DataFrame, district_avg: Optional[float]) -> Optional[alt.LayerChart]:
    """Creates the bar chart showing performance relative to district average."""
    required_cols = [COL_STORE_ID, COL_ENGAGED_PCT]
    if comp_data.empty or not all(col in comp_data.columns for col in required_cols) or district_avg is None or pd.isna(district_avg):
         st.caption("_Relative comparison requires a valid average._")
         return None # Cannot calculate relative difference without a valid average


    comp_data['Difference'] = comp_data[COL_ENGAGED_PCT] - district_avg
    # Avoid division by zero if average is zero
    comp_data['Percentage'] = (comp_data['Difference'] / district_avg * 100) if district_avg != 0 else 0.0

    num_stores = len(comp_data[COL_STORE_ID].unique())
    chart_height = max(CHART_HEIGHT_SHORT, COMPARISON_BAR_HEIGHT * num_stores) # Dynamic height

    # Determine color domain based on min/max percentage difference
    min_perc = safe_min(comp_data['Percentage'])
    max_perc = safe_max(comp_data['Percentage'])

    # Handle cases where min/max might be None or NaN
    min_perc = min_perc if pd.notna(min_perc) else 0
    max_perc = max_perc if pd.notna(max_perc) else 0

    # Ensure 0 is included in the domain for the midpoint color, handle cases where all values are same sign
    domain = sorted(list(set([min_perc, 0, max_perc])))
    # If only two unique values in domain (e.g., [0, 50] or [-50, 0]), adjust to ensure 3 points for color scale
    if len(domain) == 2:
        if 0 in domain: # e.g. [0, 50] -> [0, 0, 50] or [-50, 0] -> [-50, 0, 0]
             domain.insert(domain.index(0), 0)
        else: # e.g. [10, 50] -> [10, (10+50)/2, 50] - add midpoint
             domain.insert(1, (domain[0]+domain[1])/2)


    # Define color range (Red -> Gray/Neutral -> Green)
    color_range = [PERFORMANCE_CATEGORIES[CAT_INTERVENTION]['color'], '#BBBBBB', PERFORMANCE_CATEGORIES[CAT_STAR]['color']]

    diff_chart = alt.Chart(comp_data).mark_bar().encode(
        # Keep sort order consistent with the absolute chart (by Engaged %)
        y=alt.Y(f'{COL_STORE_ID}:N', title='Store', sort=alt.EncodingSortField(field=COL_ENGAGED_PCT, order='descending')),
        x=alt.X('Percentage:Q', title='% Difference from Average'),
        color=alt.Color('Percentage:Q', scale=alt.Scale(domain=domain, range=color_range, type='linear'), legend=None),
        tooltip=[
            alt.Tooltip(COL_STORE_ID, title='Store'),
            alt.Tooltip(f'{COL_ENGAGED_PCT}:Q', format='.2f', title='Engaged %'),
            alt.Tooltip('Percentage:Q', format='+.2f', title='% Diff from Avg')
        ]
    ).properties(
        height=chart_height,
        title="Performance Relative to Average"
    )

    # Add zero line
    center_rule = alt.Chart(pd.DataFrame({'center': [0]})).mark_rule(color='black').encode(x='center:Q')

    return alt.layer(diff_chart, center_rule)


def create_rank_trend_chart(rank_data: pd.DataFrame) -> Optional[alt.Chart]:
    """Creates the line chart for tracking weekly ranks."""
    required_cols = [COL_WEEK, COL_STORE_ID, COL_RANK]
    if rank_data.empty or not all(col in rank_data.columns for col in required_cols):
        return None

    # Ensure rank is numeric for plotting scale
    rank_data = rank_data.dropna(subset=[COL_RANK])
    if rank_data.empty: return None
    rank_data[COL_RANK] = rank_data[COL_RANK].astype(int)

    # Determine rank domain (min/max rank) - reverse scale so 1 is at the top
    min_rank = safe_min(rank_data[COL_RANK])
    max_rank = safe_max(rank_data[COL_RANK])

    # Handle case where min/max are None or equal
    if min_rank is None or max_rank is None:
        rank_domain = [10, 0] # Default range
    elif min_rank == max_rank:
         rank_domain = [max_rank + 1, min_rank - 1] # Ensure some space if only one rank value
    else:
         rank_domain = [max_rank + 1, min_rank - 1] # Add padding


    rank_chart_base = alt.Chart(rank_data).mark_line(point=True).encode(
        x=alt.X(f'{COL_WEEK}:O', title='Week'),
        y=alt.Y(f'{COL_RANK}:Q', title='Rank', scale=alt.Scale(domain=rank_domain, zero=False)), # Reverse scale, don't force include zero
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
    ).interactive()

    return rank_chart_interactive

def create_recent_trend_bar_chart(dir_df: pd.DataFrame) -> Optional[alt.LayerChart]:
    """Creates the bar chart showing total change for recent trends."""
    if dir_df.empty or 'total_change' not in dir_df.columns or 'store' not in dir_df.columns or 'direction' not in dir_df.columns:
        return None

    num_stores = len(dir_df['store'].unique())
    chart_height = max(CHART_HEIGHT_SHORT, COMPARISON_BAR_HEIGHT * num_stores)
    # Get the actual window size used from the data
    trend_window_used = dir_df["weeks"].iloc[0] if not dir_df.empty and "weeks" in dir_df.columns else RECENT_TRENDS_WINDOW


    change_chart = alt.Chart(dir_df).mark_bar().encode(
        x=alt.X('total_change:Q', title=f'Change in Engagement % (Last {trend_window_used} Weeks)'),
        y=alt.Y('store:N', sort=alt.EncodingSortField(field='total_change', order='descending'), title='Store'),
        color=alt.Color('direction:N',
                        scale=alt.Scale(domain=['Improving', 'Stable', 'Declining'],
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
        title="Recent Performance Change"
    )

    # Add zero line for reference
    zero_line = alt.Chart(pd.DataFrame({'x': [0]})).mark_rule(color='white', strokeDash=[3, 3]).encode(x='x:Q')

    return alt.layer(change_chart, zero_line)


# --- UI Display Functions ---

def display_sidebar(df: Optional[pd.DataFrame]) -> Tuple[Any, Any, str, str, List[str], float, bool, int]:
    """Creates and manages the Streamlit sidebar elements."""
    st.sidebar.header("Data Input")
    data_file = st.sidebar.file_uploader("Upload engagement data (Excel or CSV)", type=['csv', 'xlsx'], key="data_uploader")
    comp_file = st.sidebar.file_uploader("Optional: Upload comparison data (prior period)", type=['csv', 'xlsx'], key="comp_uploader")

    # Initialize filter defaults
    quarter_choice = "All"
    week_choice = "All"
    store_choice = []
    z_threshold = DEFAULT_Z_THRESHOLD
    show_ma = DEFAULT_SHOW_MA
    trend_analysis_weeks = DEFAULT_TREND_WINDOW
    store_list = [] # Initialize store list

    if df is not None and not df.empty:
        st.sidebar.header("Filters")

        # Get available stores
        if COL_STORE_ID in df.columns:
             store_list = sorted(df[COL_STORE_ID].unique().tolist())

        # Quarter Filter
        if COL_QUARTER in df.columns:
            quarters = sorted(df[COL_QUARTER].dropna().unique().tolist())
            quarter_options = ["All"] + [f"Q{int(q)}" for q in quarters]
            quarter_choice = st.sidebar.selectbox("Select Quarter", quarter_options, index=0, key="quarter_filter")
        else:
            st.sidebar.markdown("_Quarter information not available._")


        # Week Filter (dependent on Quarter selection)
        if COL_WEEK in df.columns:
            # Determine available weeks based on quarter selection
            if quarter_choice != "All":
                try:
                    q_num = int(quarter_choice.replace('Q', ''))
                    # Ensure Quarter column exists before filtering
                    if COL_QUARTER in df.columns:
                         weeks_in_quarter = sorted(df[df[COL_QUARTER] == q_num][COL_WEEK].unique())
                    else:
                         weeks_in_quarter = sorted(df[COL_WEEK].unique()) # Fallback if quarter column missing
                except (ValueError, KeyError):
                    weeks_in_quarter = sorted(df[COL_WEEK].unique()) # Fallback on error
            else:
                weeks_in_quarter = sorted(df[COL_WEEK].unique())

            # Create week options only if weeks are available
            if weeks_in_quarter:
                 week_options = ["All"] + [str(int(w)) for w in weeks_in_quarter]
                 week_choice = st.sidebar.selectbox("Select Week", week_options, index=0, key="week_filter") # Default to 'All'
            else:
                 st.sidebar.markdown("_No weeks available for selected quarter._")
                 week_options = ["All"] # Ensure options list is not empty
                 week_choice = "All" # Force 'All' if no specific weeks

        else:
             st.sidebar.markdown("_Week information not available._")


        # Store Filter
        if store_list:
            store_choice = st.sidebar.multiselect("Select Store(s)", store_list, default=[], key="store_filter")
        else:
            st.sidebar.markdown("_Store information not available._")


        # Advanced Settings
        with st.sidebar.expander("Advanced Settings", expanded=False):
            z_threshold = st.slider("Anomaly Z-score Threshold", 1.0, 3.0, DEFAULT_Z_THRESHOLD, 0.1, key="z_slider")
            show_ma = st.checkbox(f"Show {DEFAULT_TREND_WINDOW}-week moving average", value=DEFAULT_SHOW_MA, key="ma_checkbox")
            trend_analysis_weeks = st.slider("Trend analysis window (weeks)", MIN_TREND_WINDOW, MAX_TREND_WINDOW, DEFAULT_TREND_WINDOW, key="trend_window_slider")
            st.caption("Adjust sensitivity for anomaly detection and overall trend analysis.")

    # Footer and Help (Always display)
    now = datetime.datetime.now()
    st.sidebar.markdown("---")
    # st.sidebar.caption(f"© Publix Super Markets, Inc. {now.year}") # As per original code - Consider removing if not official
    st.sidebar.caption(f"Dashboard Refactored - Run: {now.strftime('%Y-%m-%d %H:%M')}")
    with st.sidebar.expander("Help & Information"):
        st.markdown("### Using This Dashboard\n- **Upload Data**: Start by uploading your engagement data file\n- **Apply Filters**: Use the filters to focus on specific time periods or stores\n- **Explore Tabs**: Each tab provides different insights:\n    - **Engagement Trends**: Performance over time\n    - **Store Comparison**: Compare stores directly\n    - **Store Performance Categories**: Categories and action plans\n    - **Anomalies & Insights**: Unusual patterns and opportunities\n\n_Disclaimer: This is not an official Publix tool. Use data responsibly._") # Added disclaimer note

    return data_file, comp_file, quarter_choice, week_choice, store_choice, z_threshold, show_ma, trend_analysis_weeks


def display_executive_summary(summary_data: Dict[str, Any]):
    """Displays the executive summary metrics and text."""
    st.subheader("Executive Summary")
    if summary_data['current_week'] is None:
        st.info("Not enough data to generate executive summary based on current filters.")
        return

    col1, col2, col3 = st.columns(3)

    # Metric 1: Average Engagement
    avg_display = format_percentage(summary_data['current_avg'])
    delta_str = format_delta(summary_data['delta_val']) if summary_data['prev_week'] is not None else "N/A"
    col1.metric(f"{summary_data['avg_label']} (Week {summary_data['current_week']})", avg_display, delta_str)

    # Metric 2 & 3: Top/Bottom Performers
    if summary_data['top_store'] is not None:
        top_perf_str = f"Store {summary_data['top_store']} — {format_percentage(summary_data['top_val'])}"
        col2.metric(f"Top Performer (Week {summary_data['current_week']})", top_perf_str, help="Highest average engagement for the week among selected stores.")
        # Add trend for top performer
        top_trend = summary_data['store_trends'].get(summary_data['top_store'], "N/A")
        t_color_class = "highlight-good" if top_trend in ["Upward","Strong Upward"] else "highlight-bad" if top_trend in ["Downward","Strong Downward"] else "highlight-neutral"
        col2.markdown(f"<small>Trend (Period): <span class='{t_color_class}'>{top_trend}</span></small>", unsafe_allow_html=True)

    if summary_data['bottom_store'] is not None:
        bottom_perf_str = f"Store {summary_data['bottom_store']} — {format_percentage(summary_data['bottom_val'])}"
        col3.metric(f"Bottom Performer (Week {summary_data['current_week']})", bottom_perf_str, help="Lowest average engagement for the week among selected stores.")
         # Add trend for bottom performer
        bottom_trend = summary_data['store_trends'].get(summary_data['bottom_store'], "N/A")
        b_color_class = "highlight-good" if bottom_trend in ["Upward","Strong Upward"] else "highlight-bad" if bottom_trend in ["Downward","Strong Downward"] else "highlight-neutral"
        col3.markdown(f"<small>Trend (Period): <span class='{b_color_class}'>{bottom_trend}</span></small>", unsafe_allow_html=True)


    # Summary Text
    if summary_data['delta_val'] is not None and summary_data['prev_week'] is not None:
        st.markdown(f"Week {summary_data['current_week']} average engagement is <span class='{summary_data['trend_class']}'>{abs(summary_data['delta_val']):.2f} points {summary_data['trend_dir']}</span> from Week {summary_data['prev_week']}.", unsafe_allow_html=True)
    elif summary_data['current_avg'] is not None:
        st.markdown(f"Week {summary_data['current_week']} engagement average: <span class='highlight-neutral'>{format_percentage(summary_data['current_avg'])}</span>", unsafe_allow_html=True)
    else:
         st.markdown("_Average engagement could not be calculated for the current week._")


def display_key_insights(insights: List[str]):
    """Displays the generated key insights list."""
    st.subheader("Key Insights")
    if not insights or insights == ["No data available for insights."]:
        st.info("Not enough data to generate key insights for the current selection.")
        return
    # Use columns for better layout if many insights
    # cols = st.columns(len(insights))
    for i, point in enumerate(insights, start=1):
         # with cols[i-1]: # Put each insight in its own column (might be too wide)
         st.markdown(f"{i}. {point}")


def display_engagement_trends_tab(df_filtered: pd.DataFrame, df_comp_filtered: Optional[pd.DataFrame], show_ma: bool, district_trend: pd.DataFrame, district_trend_comp: Optional[pd.DataFrame]):
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
    df_plot = calculate_moving_averages(df_plot)


    # --- Recent Trends Specific Filters & Metrics ---
    stores_to_show_custom = []
    if view_option == "Recent Trends":
        all_weeks = sorted(df_plot[COL_WEEK].unique())
        if len(all_weeks) > 1:
            min_week, max_week = min(all_weeks), max(all_weeks)
            default_start = all_weeks[0] if len(all_weeks) <= 8 else all_weeks[-8]
            default_end = all_weeks[-1]
            # Ensure default_start is not after default_end if few weeks exist
            if default_start > default_end: default_start = default_end

            # Use tuple for value in select_slider
            try:
                recent_weeks_range = st.select_slider(
                    "Select weeks to display:",
                    options=all_weeks,
                    value=(default_start, default_end), # Pass tuple here
                    help="Adjust to show a shorter or longer recent period",
                    key="recent_weeks_slider"
                )
            except st.errors.StreamlitAPIException:
                 # Handle case where default value might be invalid (e.g., only one week)
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

                df_plot = df_plot[(df_plot[COL_WEEK] >= start_week) & (df_plot[COL_WEEK] <= end_week)]
                district_trend = district_trend[(district_trend[COL_WEEK] >= start_week) & (district_trend[COL_WEEK] <= end_week)] if district_trend is not None else None
                district_trend_comp = district_trend_comp[(district_trend_comp[COL_WEEK] >= start_week) & (district_trend_comp[COL_WEEK] <= end_week)] if district_trend_comp is not None else None

            # Display Recent Trend Metrics
            st.markdown("---") # Separator
            st.markdown("##### Metrics for Selected Recent Weeks")
            col1, col2 = st.columns(2)
            with col1:
                if district_trend is not None and len(district_trend) >= 2:
                    last_two = district_trend.sort_values(COL_WEEK).tail(2)
                    cur_val = last_two['Average Engagement %'].iloc[1]
                    prev_val = last_two['Average Engagement %'].iloc[0]
                    change_pct = ((cur_val - prev_val) / prev_val * 100) if prev_val != 0 else 0
                    st.metric("District Trend (Week-over-Week)", f"{cur_val:.2f}%", f"{change_pct:+.1f}%", help="Change between the last two weeks shown in the chart.")
                else:
                     st.metric("District Trend (Week-over-Week)", "N/A", help="Requires at least two weeks in the selected range.")
            with col2:
                 if not df_plot.empty:
                     last_week = safe_max(df_plot[COL_WEEK])
                     if last_week is not None:
                         last_week_data = df_plot[df_plot[COL_WEEK] == last_week]
                         if not last_week_data.empty:
                            best_store_row = last_week_data.loc[safe_idxmax(last_week_data[COL_ENGAGED_PCT])] if safe_idxmax(last_week_data[COL_ENGAGED_PCT]) is not None else None
                            if best_store_row is not None:
                                 st.metric(f"Top Performer (Week {last_week})", f"Store {best_store_row[COL_STORE_ID]}", f"{best_store_row[COL_ENGAGED_PCT]:.2f}%", delta_color="off", help="Best performing store in the last week shown.")
                            else:
                                 st.metric(f"Top Performer (Week {last_week})", "N/A")

                         else:
                            st.metric(f"Top Performer (Week {last_week})", "N/A")

                     else:
                         st.metric("Top Performer", "N/A")
                 else:
                     st.metric("Top Performer", "N/A")
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
             stores_to_show_custom = st.multiselect("Select stores to compare:", options=available_stores, default=[], key="custom_store_select")
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
            caption = "**All Stores View:** Shows all store trends with interactive legend. Black dashed line = district average."
        elif view_option == "Custom Selection":
             caption = "**Custom Selection View:** Shows only selected stores. Black dashed line = district average."
        elif view_option == "Recent Trends":
             caption = "**Recent Trends View:** Focuses on selected weeks. Black dashed line = district average."

        if df_comp_plot is not None and not df_comp_plot.empty:
            caption += " Gray dashed line = comparison period's district average."
        if show_ma:
             caption += " Lighter dashed lines = 4-week moving averages."
        st.caption(caption)

    else:
        st.info("No data available to display engagement trend chart for the current selection.")

    # --- Weekly Engagement Heatmap ---
    st.subheader("Weekly Engagement Heatmap")
    with st.expander("Heatmap Settings", expanded=False):
        colA, colB = st.columns(2)
        with colA:
            sort_method = st.selectbox("Sort stores by:", ["Average Engagement", "Recent Performance"], index=0, key="heatmap_sort")
        with colB:
            color_scheme = st.selectbox("Color scheme:", COLOR_SCHEME_OPTIONS, index=0, key="heatmap_color").lower()

    # Filter heatmap data by week range slider
    heatmap_df_base = df_filtered.copy() # Use original filtered data for heatmap base
    weeks_list = sorted(heatmap_df_base[COL_WEEK].unique()) if COL_WEEK in heatmap_df_base.columns else []


    if len(weeks_list) > 1:
        min_w, max_w = int(min(weeks_list)), int(max(weeks_list))
        # Ensure slider values are within the available range
        slider_min = min_w
        slider_max = max_w
        default_slider_val = (slider_min, slider_max)

        # Handle case where min > max (shouldn't happen with sorted list > 1)
        if slider_min > slider_max:
            slider_min, slider_max = slider_max, slider_min # Swap if needed
            default_slider_val = (slider_min, slider_max)

        # Ensure options are unique if list has duplicates (shouldn't with unique())
        unique_weeks_list = sorted(list(set(weeks_list)))

        try:
            selected_range = st.select_slider(
                "Select week range for heatmap:",
                options=unique_weeks_list,
                value=default_slider_val,
                key="heatmap_week_slider"
            )
        except st.errors.StreamlitAPIException:
             # Fallback if default value is outside options (e.g., single week data)
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


    # --- Recent Performance Trends Section ---
    st.subheader("Recent Performance Trends (within Heatmap Range)")
    with st.expander("About This Section", expanded=False): # Default collapsed
        st.write("This section shows which stores are **improving**, **stable**, or **declining** over the last several weeks *within the date range selected for the heatmap above*, focusing on short-term momentum.")


    col1, col2 = st.columns(2)
    with col1:
        trend_window_recent = st.slider("Number of recent weeks to analyze", MIN_TREND_WINDOW, MAX_TREND_WINDOW, RECENT_TRENDS_WINDOW, key="recent_trend_window")
    with col2:
        sensitivity = st.select_slider("Sensitivity to small changes", options=["Low", "Medium", "High"], value="Medium", key="recent_trend_sensitivity")
        momentum_threshold = RECENT_TRENDS_SENSITIVITY_MAP[sensitivity]

    # Use the heatmap_df as it's already filtered by the week slider
    recent_trends_df = calculate_recent_performance_trends(heatmap_df, trend_window_recent, momentum_threshold)

    if recent_trends_df.empty:
        st.info("Not enough data to analyze recent trends for the selected week range and window.")
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

            color = group.iloc[0]['color']
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

    all_stores = sorted(df_filtered[COL_STORE_ID].unique()) if COL_STORE_ID in df_filtered.columns else []
    if len(all_stores) < 2:
        st.info("Please select at least two stores in the sidebar filters (or clear store selection) to enable comparison.")
        return

    # Prepare comparison data (either single week or period average)
    if week_choice != "All":
        try:
             week_num = int(week_choice)
             comp_data_base = df_filtered[df_filtered[COL_WEEK] == week_num].copy()
             comp_title = f"Store Comparison - Week {week_choice}"
             # Ensure we still group by store in case of duplicate entries for a store in a week (shouldn't happen with clean data)
             comp_data = comp_data_base.groupby(COL_STORE_ID, as_index=False)[COL_ENGAGED_PCT].mean()
        except ValueError:
             st.error(f"Invalid week selected for comparison: {week_choice}. Showing period average instead.")
             comp_data = df_filtered.groupby(COL_STORE_ID, as_index=False)[COL_ENGAGED_PCT].mean()
             comp_title = "Store Comparison - Period Average (Fallback)"

    else: # Period Average
        comp_data = df_filtered.groupby(COL_STORE_ID, as_index=False)[COL_ENGAGED_PCT].mean()
        comp_title = "Store Comparison - Period Average"

    if comp_data.empty or COL_ENGAGED_PCT not in comp_data.columns:
        st.warning("No comparison data available for the selected week/period.")
        return

    comp_data = comp_data.sort_values(COL_ENGAGED_PCT, ascending=False)
    district_avg = safe_mean(comp_data[COL_ENGAGED_PCT])

    # --- Absolute Performance Chart ---
    st.markdown("#### Absolute Engagement Percentage")
    comparison_chart = create_comparison_bar_chart(comp_data, district_avg, comp_title)
    if comparison_chart:
        st.altair_chart(comparison_chart, use_container_width=True)
        avg_text = f"average engagement ({format_percentage(district_avg)})" if district_avg is not None else "average engagement (N/A)"
        st.caption(f"Red dashed line indicates the {avg_text} for the selected stores and period.")

    else:
        st.info("Could not generate absolute comparison chart.")

    # --- Relative Performance Chart ---
    st.markdown("#### Performance Relative to Average")
    relative_chart = create_relative_comparison_chart(comp_data, district_avg)
    if relative_chart:
        st.altair_chart(relative_chart, use_container_width=True)
        st.caption("Green bars = above selected average, red bars = below selected average.")
    else:
        st.info("Could not generate relative comparison chart (requires valid average).")


    # --- Weekly Rank Tracking ---
    if COL_RANK in df_filtered.columns:
        st.markdown("#### Weekly Rank Tracking")
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
        st.info("Weekly Rank column not found in the data.")


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
    cat_order = [CAT_STAR, CAT_IMPROVING, CAT_STABILIZE, CAT_INTERVENTION] # Display order

    for i, cat in enumerate(cat_order):
        if cat in PERFORMANCE_CATEGORIES:
            info = PERFORMANCE_CATEGORIES[cat]
            with cols[i]:
                 card_html = f"""
                 <div class='info-card' style='border-left-color: {info['color']};'>
                     <h4 style='color:{info['color']};'>{info['icon']} {cat}</h4>
                     <p>{info['explanation']}</p>
                     <p><strong>Action:</strong> {info['short_action']}</p>
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

        st.markdown(f"<div style='border-left:5px solid {color}; padding-left:15px; margin: 20px 0 10px 0;'><h4 style='color:{color}; margin-bottom: 0;'>{icon} {cat} ({len(subset)} stores)</h4></div>", unsafe_allow_html=True)


        # Display store cards within the category
        cols = st.columns(min(4, len(subset))) # Max 4 columns for store cards
        # Sort within category: High-to-low for top cats, Low-to-high for bottom cats
        sort_ascending = (cat in [CAT_IMPROVING, CAT_INTERVENTION])
        subset = subset.sort_values(COL_AVG_ENGAGEMENT, ascending=sort_ascending)

        for i, (_, store) in enumerate(subset.iterrows()):
            with cols[i % 4]: # Cycle through columns
                avg_eng_disp = format_percentage(store.get(COL_AVG_ENGAGEMENT)) # Use .get for safety
                trend_corr = store.get(COL_TREND_CORR, 0.0) # Use .get for safety

                # Determine trend icon based on correlation
                if trend_corr > TREND_UP: trend_icon = "📈" # Improving
                elif trend_corr < TREND_DOWN: trend_icon = "📉" # Declining
                else: trend_icon = "➡️" # Stable

                card_html = f"""
                <div class='info-card' style='border-left-color: {color};'>
                    <h4 style='color:{color}; text-align:center;'>Store {store.get(COL_STORE_ID, 'N/A')}</h4>
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

    selected_store = st.selectbox("Select a store:", store_list_options, key="category_store_select")

    if selected_store:
        # Retrieve the row safely
        row_df = store_stats[store_stats[COL_STORE_ID] == selected_store]
        if row_df.empty:
             st.warning(f"Could not find data for selected store: {selected_store}")
             return

        row = row_df.iloc[0] # Get the first (and only) row as a Series
        cat = row.get(COL_CATEGORY, CAT_UNCATEGORIZED)
        color = PERFORMANCE_CATEGORIES.get(cat, {}).get('color', '#757575')
        icon = PERFORMANCE_CATEGORIES.get(cat, {}).get('icon', '')
        avg_val = row.get(COL_AVG_ENGAGEMENT)
        corr = row.get(COL_TREND_CORR, 0.0)
        explanation = row.get(COL_EXPLANATION, "N/A")
        action_plan = row.get(COL_ACTION_PLAN, "N/A")

        # Trend description based on correlation
        if corr > TREND_STRONG_UP: trend_desc, trend_icon = "Strong positive trend", "🔼"
        elif corr > TREND_UP: trend_desc, trend_icon = "Mild positive trend", "↗️"
        elif corr < TREND_STRONG_DOWN: trend_desc, trend_icon = "Strong negative trend", "🔽"
        elif corr < TREND_DOWN: trend_desc, trend_icon = "Mild negative trend", "↘️"
        else: trend_desc, trend_icon = "Stable trend", "➡️"

        detail_html = f"""
        <div class='info-card' style='border-left-color: {color}; padding: 20px;'>
            <h3 style='color:{color}; margin-top:0;'>{icon} Store {selected_store} - {cat}</h3>
            <p><strong>Average Engagement:</strong> {format_percentage(avg_val)}</p>
            <p><strong>Trend:</strong> {trend_icon} {trend_desc} (Correlation: {corr:.2f})</p>
            <p><strong>Explanation:</strong> {explanation}</p>
            <h4 style='color:{color}; margin-top:1em;'>Recommended Action Plan:</h4>
            <p>{action_plan}</p>
        </div>
        """
        st.markdown(detail_html, unsafe_allow_html=True)

        # Additional context for lower-performing categories
        if cat in [CAT_IMPROVING, CAT_INTERVENTION]:
            st.markdown("---")
            st.markdown("##### Improvement Opportunities & Context")
            # Suggest learning partners
            top_stores = store_stats[store_stats[COL_CATEGORY] == CAT_STAR][COL_STORE_ID].tolist()
            if top_stores:
                partners = ", ".join(f"Store {s}" for s in top_stores)
                partner_color = PERFORMANCE_CATEGORIES.get(CAT_STAR, {}).get('color', '#2E7D32')
                st.markdown(f"<div class='info-card' style='border-left-color: {partner_color};'><h4 style='color:{partner_color}; margin-top:0;'>Potential Learning Partners</h4><p>Consider reviewing strategies from top performers: <strong>{partners}</strong></p></div>", unsafe_allow_html=True)

            else:
                 st.markdown("<div class='info-card' style='border-left-color: #ccc;'><h4 style='margin-top:0;'>Potential Learning Partners</h4><p>No stores currently categorized as 'Star Performers' in this period.</p></div>", unsafe_allow_html=True)


            # Show gap to median
            median_eng = store_stats[COL_AVG_ENGAGEMENT].median()
            current_eng = avg_val if avg_val is not None else 0
            # Ensure median is valid before calculating gain
            if pd.notna(median_eng):
                 gain = median_eng - current_eng
                 if gain > 0:
                      gain_color = PERFORMANCE_CATEGORIES.get(CAT_IMPROVING, {}).get('color', '#1976D2')
                      st.markdown(f"<div class='info-card' style='border-left-color: {gain_color}; margin-top:15px;'><h4 style='color:{gain_color}; margin-top:0;'>Gap to Median</h4><p>Current average: <strong>{format_percentage(current_eng)}</strong> | District median: <strong>{format_percentage(median_eng)}</strong> | Potential gain to median: <strong>{gain:.2f}%</strong></p></div>", unsafe_allow_html=True)

                 else:
                      st.markdown(f"<div class='info-card' style='border-left-color: #ccc; margin-top:15px;'><h4 style='margin-top:0;'>Gap to Median</h4><p>Store is already performing at or above the median engagement ({format_percentage(median_eng)}).</p></div>", unsafe_allow_html=True)

            else:
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
        anomalies_display = anomalies_df[[col for col in display_cols if col in anomalies_df.columns]].copy()
        anomalies_display.rename(columns={
            COL_ENGAGED_PCT: 'Engagement %',
            COL_CHANGE_PCT_PTS: 'Change Pts',
            COL_Z_SCORE: 'Z-Score',
            COL_RANK: 'Current Rank',
            COL_PREV_RANK: 'Previous Rank',
            COL_POSSIBLE_EXPLANATION: 'Possible Explanation'
        }, inplace=True)

        # Display with formatting
        st.dataframe(
             anomalies_display,
             column_config={
                  "Engagement %": st.column_config.NumberColumn(format="%.2f%%"),
                  "Change Pts": st.column_config.NumberColumn(format="%+.2f"),
                  "Z-Score": st.column_config.NumberColumn(format="%.2f"),
                  "Current Rank": st.column_config.NumberColumn(format="%d"),
                  "Previous Rank": st.column_config.NumberColumn(format="%d"),
             },
             hide_index=True
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
              },
              hide_index=True
         )


# --- Main Application Flow ---

def main():
    """Main function to run the Streamlit application."""
    # --- Page Config ---
    st.set_page_config(
        page_title=APP_TITLE,
        layout=PAGE_LAYOUT,
        initial_sidebar_state=INITIAL_SIDEBAR_STATE
    )
    st.markdown(APP_CSS, unsafe_allow_html=True)
    st.markdown(f"<h1 class='dashboard-title'>{APP_TITLE}</h1>", unsafe_allow_html=True)
    st.markdown("Analyze **Club Publix** engagement data. Upload weekly data to explore KPIs, trends, and opportunities across stores. Use sidebar filters to refine the view.")

    # --- Sidebar and Data Loading ---
    df_all = None
    df_comp_all = None
    store_list = [] # Initialize empty store list

    # Initial sidebar display (before data is loaded)
    data_file, comp_file, quarter_choice, week_choice, store_choice, z_threshold, show_ma, trend_analysis_weeks = display_sidebar(None)

    if data_file:
        df_all = load_and_process_data(data_file)
        if df_all is None or df_all.empty:
             st.error("Failed to load or process primary data file. Please check the file format, required columns (Store #, Week/Date, Engaged Transaction %), and ensure data exists.")
             st.stop() # Stop execution if primary data fails
        elif COL_STORE_ID in df_all.columns:
             store_list = sorted(df_all[COL_STORE_ID].unique())


        if comp_file:
            df_comp_all = load_and_process_data(comp_file)
            if df_comp_all is None or df_comp_all.empty:
                 st.warning("Failed to load or process comparison data file. It will be ignored.")
                 df_comp_all = None # Ensure it's None if loading failed


        # --- Update Sidebar with Data-Driven Options ---
        # Re-render sidebar now that df_all is loaded and store_list is populated
        _, _, quarter_choice, week_choice, store_choice, z_threshold, show_ma, trend_analysis_weeks = display_sidebar(df_all)


    else:
        st.info("Please upload a primary engagement data file using the sidebar to begin analysis.")
        st.markdown("### Required Columns")
        st.markdown(f"- `{COL_STORE_ID}`\n- `{COL_WEEK}` or `{COL_DATE}`\n- `{COL_ENGAGED_PCT}`")
        st.markdown("### Optional Columns")
        st.markdown(f"- `{COL_RANK}`\n- `{COL_QTD_PCT}`")
        st.stop() # Stop if no file is uploaded


    # --- Data Filtering ---
    df_filtered = filter_dataframe(df_all, quarter_choice, week_choice, store_choice)
    df_comp_filtered = filter_dataframe(df_comp_all, quarter_choice, week_choice, store_choice) if df_comp_all is not None else None


    if df_filtered.empty:
        st.error("No data available for the selected filters. Please adjust filters or check the uploaded data.")
        st.stop()


    # --- Perform Calculations ---
    # Note: Pass df_all for context needed beyond filtered scope (e.g., finding previous week)
    summary_data = get_executive_summary_data(df_filtered, df_all, store_choice, store_list, trend_analysis_weeks)
    store_perf_current = df_filtered.groupby(COL_STORE_ID)[COL_ENGAGED_PCT].mean() if not df_filtered.empty and COL_STORE_ID in df_filtered.columns and COL_ENGAGED_PCT in df_filtered.columns else pd.Series(dtype=float)

    key_insights = generate_key_insights(df_filtered, summary_data['store_trends'], store_perf_current)


    # Calculations needed for tabs (perform these *after* filtering)
    district_trend = calculate_district_trends(df_filtered)
    district_trend_comp = calculate_district_trends(df_comp_filtered) if df_comp_filtered is not None else None
    store_stats = calculate_performance_categories(df_filtered) # Categories based on filtered period
    anomalies_df = find_anomalies(df_filtered, z_threshold) # Anomalies within filtered period
    recommendations_df = generate_recommendations(df_filtered, store_stats, anomalies_df, trend_analysis_weeks)



    # --- Display Main Content ---
    display_executive_summary(summary_data)
    display_key_insights(key_insights)

    # --- Tabs ---
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Engagement Trends",
        "📈 Store Comparison",
        "📋 Store Performance Categories",
        "💡 Anomalies & Insights"
    ])

    with tab1:
        display_engagement_trends_tab(df_filtered, df_comp_filtered, show_ma, district_trend, district_trend_comp)


    with tab2:
        display_store_comparison_tab(df_filtered, week_choice)

    with tab3:
        display_performance_categories_tab(store_stats)

    with tab4:
        display_anomalies_insights_tab(anomalies_df, recommendations_df, z_threshold)



if __name__ == "__main__":
    main()