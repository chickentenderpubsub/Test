import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import datetime
from typing import Optional, Tuple, Dict, Any

# --- Configuration Constants ---

PAGE_CONFIG = {"page_title": "Club Publix Engagement Dashboard", "layout": "wide", "initial_sidebar_state": "expanded"}
# Column Name Constants (using lowercase for flexibility in matching)
COL_QTD = 'quarter to date %'
COL_RANK = 'weekly rank'
COL_DATE = 'date'
COL_WEEK_ENDING = 'week ending' # Alternative date column name
COL_WEEK = 'week'
COL_STORE = 'store #'
COL_ENGAGED = 'engaged transaction %'

# Mapping potential input column names to standardized names
COLUMN_MAP_KEYS = {
    'quarter': COL_QTD, 'qtd': COL_QTD,
    'rank': COL_RANK,
    'date': COL_DATE, 'week ending': COL_DATE,
    'week': COL_WEEK,
    'store': COL_STORE,
    'engaged': COL_ENGAGED, 'engagement': COL_ENGAGED
}

# CSS Styles
CSS_STYLES = """
<style>
    .metric-card { background-color: #f5f5f5; border-radius: 10px; padding: 15px; box-shadow: 2px 2px 5px rgba(0,0,0,0.1); }
    .highlight-good { color: #2E7D32; font-weight: bold; } /* Green */
    .highlight-bad { color: #C62828; font-weight: bold; }  /* Red */
    .highlight-neutral { color: #F57C00; font-weight: bold; } /* Orange */
    .dashboard-title { color: #1565C0; text-align: center; padding-bottom: 20px; } /* Blue */
    .caption-text { font-size: 0.85em; color: #555; }
    .category-card { background-color:#2C2C2C; padding:15px; border-radius:5px; margin-bottom:10px; border-left:5px solid; }
    .category-card h4 { color: inherit; margin-top:0; }
    .category-card p { color:#FFFFFF; margin:0; }
    .category-card strong { font-weight: bold; }
    .store-card { background-color:#2C2C2C; padding:10px; border-radius:5px; margin-bottom:10px; }
    .store-card h4 { text-align:center; color: inherit; margin:5px 0; }
    .store-card p { text-align:center; color:#FFFFFF; margin:5px 0; }
    .store-card strong { font-weight: bold; }
    .styled-div-border { padding-left:10px; margin:20px 0 10px; border-left:5px solid; }
    .styled-div-border h4 { color: inherit; margin-top:0; }
</style>
"""

# Performance Category Configuration
CATEGORY_CONFIG = {
    "Star Performer": {"icon": "⭐", "color": "#2E7D32", "explanation": "High engagement with stable or improving trend", "action": "Maintain current strategies. Share best practices."},
    "Needs Stabilization": {"icon": "⚠️", "color": "#F57C00", "explanation": "High engagement but recent downward trend", "action": "Investigate inconsistencies. Reinforce processes."},
    "Improving": {"icon": "📈", "color": "#1976D2", "explanation": "Below average engagement but trending upward", "action": "Continue positive momentum. Intensify efforts."},
    "Requires Intervention": {"icon": "🚨", "color": "#C62828", "explanation": "Below average engagement with flat or declining trend", "action": "Urgent attention needed. Develop improvement plan."}
}
CATEGORY_ORDER = ["Star Performer", "Needs Stabilization", "Improving", "Requires Intervention"]
TREND_CORR_THRESHOLD_POS = 0.1
TREND_CORR_THRESHOLD_NEG = -0.1
TREND_SLOPE_STRONG_POS = 0.5
TREND_SLOPE_POS = 0.1
TREND_SLOPE_STRONG_NEG = -0.5
TREND_SLOPE_NEG = -0.1

# Recent Trend Analysis Configuration
RECENT_TREND_SENSITIVITY = {
    "Low": 0.5,
    "Medium": 0.3,
    "High": 0.2
}
RECENT_TREND_COLORS = {
    "Improving": "#2E7D32",
    "Stable": "#1976D2",
    "Declining": "#C62828"
}

# Anomaly Detection Defaults
DEFAULT_Z_SCORE_THRESHOLD = 2.0

# Chart Configuration
ALTAIR_COLOR_SCHEMES = ["blues", "greens", "purples", "oranges", "reds", "viridis", "category10"]
CHART_HEIGHT_DEFAULT = 400
HEATMAP_HEIGHT_PER_ROW = 20
BAR_CHART_HEIGHT_PER_ROW = 25
RANK_CHART_HEIGHT = 300

# --- Utility Functions ---

@st.cache_data
def load_data(uploaded_file) -> Tuple[Optional[pd.DataFrame], list]:
    """
    Loads data from an uploaded CSV or Excel file, standardizes columns,
    parses dates and percentages, and reports parsing errors.

    Args:
        uploaded_file: The file object uploaded via st.file_uploader.

    Returns:
        A tuple containing:
            - pd.DataFrame: The loaded and processed DataFrame, or None if loading fails.
            - list: A list of error messages encountered during processing.
    """
    if not uploaded_file:
        return None, ["No file uploaded."]

    errors_log = []
    df = None
    try:
        name = uploaded_file.name.lower()
        if name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif name.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(uploaded_file)
        else:
            return None, [f"Unsupported file type: {uploaded_file.name}. Please use CSV or Excel."]
    except Exception as e:
        return None, [f"Error reading file {uploaded_file.name}: {e}"]

    # Standardize column names
    original_columns = df.columns.tolist()
    df.columns = [col.strip().lower() for col in original_columns]
    col_map = {}
    identified_cols = set()
    for col_in in df.columns:
        for key, target_col in COLUMN_MAP_KEYS.items():
            if key in col_in and target_col not in identified_cols:
                col_map[col_in] = target_col
                identified_cols.add(target_col)
                break # Prioritize first match for a target column

    # Find original case names for mapped columns before renaming
    reverse_col_map = {v: k for k, v in col_map.items()}
    original_mapped_names = {v: next((orig for orig, lower in zip(original_columns, df.columns) if lower == k), k)
                             for k, v in col_map.items()}

    df = df.rename(columns=col_map)

    # --- Data Type Conversion and Validation ---
    processed_rows = len(df)

    # Date Parsing
    if COL_DATE in df.columns:
        original_dtype = df[COL_DATE].dtype
        df[COL_DATE] = pd.to_datetime(df[COL_DATE], errors='coerce')
        null_dates = df[COL_DATE].isnull().sum()
        if null_dates > 0:
            errors_log.append(f"Could not parse {null_dates} values in '{original_mapped_names.get(COL_DATE, COL_DATE)}' as dates. Affected rows ignored for date-based calculations.")
            # Optionally keep original values for inspection: df[f'{COL_DATE}_original'] = df[COL_DATE]
        df = df.dropna(subset=[COL_DATE]) # Drop rows where date parsing failed
        if processed_rows - len(df) > 0:
             errors_log.append(f"{processed_rows - len(df)} rows removed due to invalid dates.")
             processed_rows = len(df)

    elif COL_WEEK not in df.columns:
         errors_log.append(f"Missing required column: Could not find a column mappable to '{COL_DATE}' or '{COL_WEEK}'.")
         # Decide how critical this is - maybe return None? For now, continue.

    # Percentage Parsing
    for percent_col in [COL_ENGAGED, COL_QTD]:
        if percent_col in df.columns:
            original_dtype = df[percent_col].dtype
            initial_nulls = df[percent_col].isnull().sum()
            # Attempt conversion robustly
            numeric_col = pd.to_numeric(df[percent_col].astype(str).str.replace('%', '').str.strip(), errors='coerce')
            conversion_errors = numeric_col.isnull().sum() - initial_nulls

            if conversion_errors > 0:
                 errors_log.append(f"Could not parse {conversion_errors} non-empty values in '{original_mapped_names.get(percent_col, percent_col)}' as numbers. Affected rows ignored.")
            df[percent_col] = numeric_col

            if percent_col == COL_ENGAGED:
                df = df.dropna(subset=[COL_ENGAGED]) # Critical column
                if processed_rows - len(df) > 0:
                    errors_log.append(f"{processed_rows - len(df)} rows removed due to invalid or missing '{original_mapped_names.get(percent_col, percent_col)}'.")
                    processed_rows = len(df)

    # Rank Parsing
    if COL_RANK in df.columns:
        original_dtype = df[COL_RANK].dtype
        initial_nulls = df[COL_RANK].isnull().sum()
        # Use Int64 for nullable integers
        numeric_col = pd.to_numeric(df[COL_RANK], errors='coerce')
        conversion_errors = numeric_col.isnull().sum() - initial_nulls
        if conversion_errors > 0:
            errors_log.append(f"Could not parse {conversion_errors} non-empty values in '{original_mapped_names.get(COL_RANK, COL_RANK)}' as integers.")
        # Convert valid numbers to nullable integers
        df[COL_RANK] = numeric_col.astype('Int64')

    # Store Number as String
    if COL_STORE in df.columns:
        df[COL_STORE] = df[COL_STORE].astype(str).str.strip()

    # Week as Integer and Sorting
    if COL_WEEK in df.columns:
         initial_nulls = df[COL_WEEK].isnull().sum()
         numeric_col = pd.to_numeric(df[COL_WEEK], errors='coerce')
         conversion_errors = numeric_col.isnull().sum() - initial_nulls
         if conversion_errors > 0:
              errors_log.append(f"Could not parse {conversion_errors} non-empty values in '{original_mapped_names.get(COL_WEEK, COL_WEEK)}' as integers.")
         df[COL_WEEK] = numeric_col.astype('Int64') # Use Int64 for consistency
         df = df.dropna(subset=[COL_WEEK])
         if processed_rows - len(df) > 0:
              errors_log.append(f"{processed_rows - len(df)} rows removed due to invalid or missing '{original_mapped_names.get(COL_WEEK, COL_WEEK)}'.")
              processed_rows = len(df)
         # Sort only if both week and store exist
         sort_cols = [col for col in [COL_WEEK, COL_STORE] if col in df.columns]
         if sort_cols:
             df = df.sort_values(sort_cols)

    # Add Quarter if missing
    if 'Quarter' not in df.columns:
        if COL_DATE in df.columns and pd.api.types.is_datetime64_any_dtype(df[COL_DATE]):
            df['Quarter'] = df[COL_DATE].dt.quarter
        elif COL_WEEK in df.columns and pd.api.types.is_numeric_dtype(df[COL_WEEK]):
            # Ensure week numbers are valid for quarter calculation
            valid_weeks = df[COL_WEEK].dropna()
            if not valid_weeks.empty:
                 df['Quarter'] = ((valid_weeks - 1) // 13 + 1).astype(int)
            else:
                 df['Quarter'] = pd.NA # Assign NA if no valid weeks
                 errors_log.append("Could not derive 'Quarter' as 'Week' column has no valid numeric data.")
        else:
            errors_log.append("Could not derive 'Quarter' column. Requires a valid 'Date' or numeric 'Week' column.")
            df['Quarter'] = pd.NA # Assign NA if cannot be derived


    if df.empty:
        errors_log.append("No valid data rows remaining after processing and validation.")
        return None, errors_log

    # Keep only potentially useful columns
    final_cols = [col for col in [COL_STORE, COL_DATE, COL_WEEK, 'Quarter', COL_ENGAGED, COL_QTD, COL_RANK] if col in df.columns]
    df = df[final_cols]


    return df, errors_log

def filter_dataframe(df: pd.DataFrame, quarter_choice: str, week_choice: str, store_choice: list) -> pd.DataFrame:
    """Applies filters for quarter, week, and store to a DataFrame."""
    if df is None or df.empty:
        return pd.DataFrame() # Return empty if input is invalid

    df_filtered = df.copy()

    # Quarter Filter
    if quarter_choice != "All" and 'Quarter' in df_filtered.columns:
        try:
            q_num = int(quarter_choice[1:])
            df_filtered = df_filtered[df_filtered['Quarter'] == q_num]
        except (ValueError, TypeError):
            st.warning(f"Invalid quarter format: {quarter_choice}. Ignoring quarter filter.")


    # Week Filter
    if week_choice != "All" and COL_WEEK in df_filtered.columns:
        try:
            week_num = int(week_choice)
            # Ensure week column is numeric before filtering
            if pd.api.types.is_numeric_dtype(df_filtered[COL_WEEK]):
                 df_filtered = df_filtered[df_filtered[COL_WEEK] == week_num]
            else:
                 st.warning(f"'Week' column is not numeric. Cannot apply week filter for week {week_num}.")
        except (ValueError, TypeError):
            st.warning(f"Invalid week format: {week_choice}. Ignoring week filter.")

    # Store Filter
    if store_choice and COL_STORE in df_filtered.columns:
        # Ensure store IDs in the filter list are strings for accurate matching
        store_choice_str = [str(s) for s in store_choice]
        df_filtered = df_filtered[df_filtered[COL_STORE].astype(str).isin(store_choice_str)]

    return df_filtered

def calculate_trend_stats(df_filtered: pd.DataFrame, trend_analysis_weeks: int) -> Tuple[pd.Series, pd.Series]:
    """Calculates average engagement per store and trend classification."""
    if df_filtered.empty or COL_STORE not in df_filtered.columns or COL_ENGAGED not in df_filtered.columns:
        return pd.Series(dtype=float), pd.Series(dtype=str)

    store_perf = df_filtered.groupby(COL_STORE)[COL_ENGAGED].mean()

    def classify_trend(group, window=4):
        if len(group) < 2 or COL_WEEK not in group.columns or COL_ENGAGED not in group.columns:
            return "Stable"

        data = group.sort_values(COL_WEEK).tail(window)
        if len(data) < 2: return "Stable"

        # Ensure data is numeric and not all NaNs before polyfit
        weeks = pd.to_numeric(data[COL_WEEK], errors='coerce')
        engagement = pd.to_numeric(data[COL_ENGAGED], errors='coerce')
        valid_data = pd.DataFrame({'week':weeks, 'engagement': engagement}).dropna()

        if len(valid_data) < 2: return "Stable"

        # Center weeks before fitting to improve numerical stability
        x = valid_data['week'].values - np.mean(valid_data['week'].values)
        try:
             # Use degree 1 for linear trend
             slope = np.polyfit(x, valid_data['engagement'].values, 1)[0]
        except (np.linalg.LinAlgError, ValueError):
             return "Stable" # Handle cases where polyfit fails


        if slope > TREND_SLOPE_STRONG_POS: return "Strong Upward"
        elif slope > TREND_SLOPE_POS: return "Upward"
        elif slope < TREND_SLOPE_STRONG_NEG: return "Strong Downward"
        elif slope < TREND_SLOPE_NEG: return "Downward"
        else: return "Stable"

    if COL_WEEK in df_filtered.columns:
         store_trends = df_filtered.groupby(COL_STORE).apply(lambda g: classify_trend(g, trend_analysis_weeks))
    else:
         # Cannot calculate trends without week data
         store_trends = pd.Series("N/A", index=store_perf.index)
         st.warning("Cannot calculate store trends without a 'Week' column.", icon="⚠️")


    return store_perf, store_trends

def find_anomalies(df: pd.DataFrame, z_threshold: float) -> pd.DataFrame:
    """
    Detects anomalies in weekly engagement changes based on Z-score.

    Args:
        df (pd.DataFrame): DataFrame containing store, week, engagement %, and optional rank.
                           Must be sorted by Store # and Week.
        z_threshold (float): The Z-score threshold to flag an anomaly.

    Returns:
        pd.DataFrame: DataFrame of detected anomalies with details.
    """
    if df.empty or COL_STORE not in df.columns or COL_WEEK not in df.columns or COL_ENGAGED not in df.columns:
        return pd.DataFrame()

    anomalies = []
    # Ensure correct sorting for diff()
    df_sorted = df.sort_values([COL_STORE, COL_WEEK])

    # Calculate differences within each store group
    df_sorted['Change %pts'] = df_sorted.groupby(COL_STORE)[COL_ENGAGED].diff()
    df_sorted['Prev Rank'] = df_sorted.groupby(COL_STORE)[COL_RANK].shift(1) if COL_RANK in df.columns else pd.NA
    df_sorted['Prev Week'] = df_sorted.groupby(COL_STORE)[COL_WEEK].shift(1)

    # Calculate Z-scores for the differences within each store group
    diff_stats = df_sorted.groupby(COL_STORE)['Change %pts'].agg(['mean', 'std']).reset_index()
    df_sorted = pd.merge(df_sorted, diff_stats, on=COL_STORE, how='left')

    # Avoid division by zero or NaN std dev
    df_sorted['std'] = df_sorted['std'].fillna(0)
    valid_std = df_sorted['std'] != 0

    df_sorted['Z-score'] = np.nan # Initialize column
    # Calculate Z-score only where std is not zero
    df_sorted.loc[valid_std, 'Z-score'] = (df_sorted.loc[valid_std, 'Change %pts'] - df_sorted.loc[valid_std, 'mean']) / df_sorted.loc[valid_std, 'std']


    # Filter anomalies based on threshold
    anomaly_df_raw = df_sorted[df_sorted['Z-score'].abs() >= z_threshold].copy()

    if anomaly_df_raw.empty:
        return pd.DataFrame()

    # Format output DataFrame
    anomaly_df = anomaly_df_raw[[COL_STORE, COL_WEEK, COL_ENGAGED, 'Change %pts', 'Z-score', 'Prev Week', 'Prev Rank', COL_RANK]].copy()
    anomaly_df.rename(columns={COL_STORE: 'Store #', COL_WEEK: 'Week', COL_ENGAGED: 'Engaged Transaction %', COL_RANK:'Rank'}, inplace=True)

    # Round numeric columns
    for col in ['Engaged Transaction %', 'Change %pts', 'Z-score']:
        anomaly_df[col] = anomaly_df[col].round(2)

    # Convertnullable int columns
    for col in ['Prev Rank', 'Rank', 'Prev Week', 'Week']:
         if col in anomaly_df.columns:
              # Convert to float first to handle NA, then to Int64
              anomaly_df[col] = pd.to_numeric(anomaly_df[col], errors='coerce').astype('Int64')


    # Add explanations
    anomaly_df['Possible Explanation'] = np.where(anomaly_df['Change %pts'] >= 0,
                                                 "Engagement spiked significantly. Possible promotion or event impact.",
                                                 "Sharp drop in engagement. Potential system issue or loss of engagement.")

    # Add rank change context if available
    if 'Rank' in anomaly_df.columns and 'Prev Rank' in anomaly_df.columns:
        improve_mask = (anomaly_df['Change %pts'] >= 0) & anomaly_df['Prev Rank'].notna() & anomaly_df['Rank'].notna() & (anomaly_df['Prev Rank'] > anomaly_df['Rank'])
        decline_mask = (anomaly_df['Change %pts'] < 0) & anomaly_df['Prev Rank'].notna() & anomaly_df['Rank'].notna() & (anomaly_df['Prev Rank'] < anomaly_df['Rank'])

        anomaly_df.loc[improve_mask, 'Possible Explanation'] += " (Improved rank from " + anomaly_df.loc[improve_mask, 'Prev Rank'].astype(str) + " to " + anomaly_df.loc[improve_mask, 'Rank'].astype(str) + ".)"
        anomaly_df.loc[decline_mask, 'Possible Explanation'] += " (Dropped rank from " + anomaly_df.loc[decline_mask, 'Prev Rank'].astype(str) + " to " + anomaly_df.loc[decline_mask, 'Rank'].astype(str) + ".)"

    # Sort by absolute Z-score descending
    anomaly_df['Abs Z'] = anomaly_df['Z-score'].abs()
    anomaly_df = anomaly_df.sort_values('Abs Z', ascending=False).drop(columns=['Abs Z'])

    return anomaly_df[['Store #', 'Week', 'Engaged Transaction %', 'Change %pts', 'Z-score', 'Rank', 'Prev Rank', 'Possible Explanation']]


def calculate_performance_categories(df_filtered: pd.DataFrame) -> pd.DataFrame:
    """Categorizes stores based on average engagement and trend correlation."""
    required_cols = [COL_STORE, COL_ENGAGED, COL_WEEK]
    if df_filtered.empty or not all(col in df_filtered.columns for col in required_cols):
        st.warning("Cannot calculate performance categories. Requires 'Store #', 'Engaged Transaction %', and 'Week' data.", icon="⚠️")
        return pd.DataFrame(columns=['Store #', 'Average Engagement', 'Trend Correlation', 'Category', 'Explanation', 'Action Plan'])

    store_stats = df_filtered.groupby(COL_STORE)[COL_ENGAGED].agg(['mean']).reset_index()
    store_stats.rename(columns={'mean': 'Average Engagement'}, inplace=True)

    # Calculate trend correlation (requires at least 3 data points per store)
    def safe_corr(group):
         if len(group) >= 3 and group[COL_ENGAGED].notna().sum() >= 2 and group[COL_WEEK].notna().sum() >= 2 :
              # Check for variance before calculating correlation
              if group[COL_WEEK].nunique() > 1 and group[COL_ENGAGED].nunique() > 1:
                   return group[COL_WEEK].corr(group[COL_ENGAGED])
         return 0 # Return neutral correlation if insufficient data or no variance

    trend_corr = df_filtered.groupby(COL_STORE).apply(safe_corr)
    store_stats['Trend Correlation'] = store_stats[COL_STORE].map(trend_corr).fillna(0)


    if store_stats['Average Engagement'].empty:
         st.warning("No stores found with valid engagement data for categorization.")
         return pd.DataFrame(columns=['Store #', 'Average Engagement', 'Trend Correlation', 'Category', 'Explanation', 'Action Plan'])


    med_eng = store_stats['Average Engagement'].median()

    conditions = [
        (store_stats['Average Engagement'] >= med_eng) & (store_stats['Trend Correlation'] < TREND_CORR_THRESHOLD_NEG), # High Perf, Declining Trend -> Stabilize
        (store_stats['Average Engagement'] >= med_eng), # High Perf, Stable/Improving -> Star
        (store_stats['Average Engagement'] < med_eng) & (store_stats['Trend Correlation'] > TREND_CORR_THRESHOLD_POS), # Low Perf, Improving Trend -> Improving
        (store_stats['Average Engagement'] < med_eng) # Low Perf, Stable/Declining -> Intervention
    ]
    # Order matters here, more specific conditions first
    choices = ["Needs Stabilization", "Star Performer", "Improving", "Requires Intervention"]
    store_stats['Category'] = np.select(conditions, choices, default="Uncategorized").astype(str) # Ensure string type

    # Map explanations and actions
    store_stats['Explanation'] = store_stats['Category'].map({k: v['explanation'] for k, v in CATEGORY_CONFIG.items()})
    store_stats['Action Plan'] = store_stats['Category'].map({k: v['action'] for k, v in CATEGORY_CONFIG.items()})
    store_stats.fillna({'Explanation': 'N/A', 'Action Plan': 'N/A'}, inplace=True) # Handle Uncategorized

    return store_stats[['Store #', 'Average Engagement', 'Trend Correlation', 'Category', 'Explanation', 'Action Plan']]

# --- Altair Charting Functions ---

def create_trend_chart(combined_data: pd.DataFrame, dist_trend: pd.DataFrame, dist_trend_comp: Optional[pd.DataFrame],
                       view_option: str, stores_to_show: list, show_ma: bool) -> Optional[alt.Chart]:
    """Creates the layered Altair trend chart."""
    if combined_data.empty:
        return None

    color_scale = alt.Scale(scheme='category10')
    layers = []

    # Store Lines Layer
    if view_option == "Custom Selection":
        if not stores_to_show:
            st.info("Please select at least one store to display for Custom Selection.")
            return None
        data_sel = combined_data[combined_data[COL_STORE].isin(stores_to_show)]
        if data_sel.empty:
             st.info("No data found for the selected stores in Custom Selection.")
             return None

        base = alt.Chart(data_sel).encode(x=alt.X(f'{COL_WEEK}:O', title='Week'), color=alt.Color(f'{COL_STORE}:N', scale=color_scale, title='Store'))
        layers.append(base.mark_line(strokeWidth=3).encode(y=alt.Y(f'{COL_ENGAGED}:Q', title='Engaged Transaction %'), tooltip=[COL_STORE, COL_WEEK, alt.Tooltip(f'{COL_ENGAGED}:Q', format='.2f')]))
        layers.append(base.mark_point(filled=True, size=80).encode(y=f'{COL_ENGAGED}:Q', tooltip=[COL_STORE, COL_WEEK, alt.Tooltip(f'{COL_ENGAGED}:Q', format='.2f')]))
        if show_ma and 'MA_4W' in data_sel.columns:
            layers.append(base.mark_line(strokeDash=[2, 2], strokeWidth=2).encode(y=alt.Y('MA_4W:Q', title='4W MA'), tooltip=[COL_STORE, COL_WEEK, alt.Tooltip('MA_4W:Q', format='.2f')]))
    else: # All Stores or Recent Trends View
        base = alt.Chart(combined_data).encode(x=alt.X(f'{COL_WEEK}:O', title='Week'), color=alt.Color(f'{COL_STORE}:N', scale=color_scale, title='Store'))
        store_sel = alt.selection_point(fields=[COL_STORE], bind='legend')

        store_line = base.mark_line(strokeWidth=1.5).encode(
            y=alt.Y(f'{COL_ENGAGED}:Q', title='Engaged Transaction %'),
            tooltip=[COL_STORE, COL_WEEK, alt.Tooltip(f'{COL_ENGAGED}:Q', format='.2f')],
            opacity=alt.condition(store_sel, alt.value(1), alt.value(0.2)),
            strokeWidth=alt.condition(store_sel, alt.value(3), alt.value(1))
        ).add_params(store_sel)
        layers.append(store_line)

        if show_ma and 'MA_4W' in combined_data.columns:
            ma_line = base.mark_line(strokeDash=[2, 2], strokeWidth=1.5).encode(
                y=alt.Y('MA_4W:Q', title='4W MA'),
                tooltip=[COL_STORE, COL_WEEK, alt.Tooltip('MA_4W:Q', format='.2f')],
                opacity=alt.condition(store_sel, alt.value(0.8), alt.value(0.1))
            ).add_params(store_sel) # Add selection here too
            layers.append(ma_line)

    # District Average Layer
    if not dist_trend.empty:
        dist_base = alt.Chart(dist_trend).encode(x=f'{COL_WEEK}:O')
        layers.append(dist_base.mark_line(color='black', strokeDash=[4, 2], size=3).encode(
            y=alt.Y(f'Average Engagement %:Q', title='Engaged Transaction %'), # Ensure y-axis title matches store lines
            tooltip=[alt.Tooltip(f'Average Engagement %:Q', format='.2f', title='District Avg')]
        ))
        if show_ma and 'MA_4W' in dist_trend.columns:
            layers.append(dist_base.mark_line(color='black', strokeDash=[1, 1], size=2, opacity=0.7).encode(
                y='MA_4W:Q',
                tooltip=[alt.Tooltip('MA_4W:Q', format='.2f', title='District 4W MA')]
            ))

    # Comparison Period District Average Layer
    if dist_trend_comp is not None and not dist_trend_comp.empty:
        comp_base = alt.Chart(dist_trend_comp).encode(x=f'{COL_WEEK}:O')
        layers.append(comp_base.mark_line(color='#555', strokeDash=[4, 2], size=2).encode(
            y=alt.Y(f'{COL_ENGAGED}:Q'), # Use original column name here as it wasn't renamed
            tooltip=[alt.Tooltip(f'{COL_ENGAGED}:Q', format='.2f', title="Last Period Avg")]
        ))
        if show_ma and 'MA_4W' in dist_trend_comp.columns:
             layers.append(comp_base.mark_line(color='#555', strokeDash=[1,1], size=1.5, opacity=0.7).encode(
                 y='MA_4W:Q',
                 tooltip=[alt.Tooltip('MA_4W:Q', format='.2f', title="Last Period 4W MA")]
             ))


    if not layers:
         return None

    # Combine layers and resolve scales
    final_chart = alt.layer(*layers).resolve_scale(
        y='shared' # Ensure all y-axes use the same scale
    ).properties(
        height=CHART_HEIGHT_DEFAULT
    )

    return final_chart

def create_heatmap(heatmap_df: pd.DataFrame, sort_method: str, color_scheme: str) -> Optional[alt.Chart]:
    """Creates the engagement heatmap."""
    if heatmap_df.empty or 'EngagedPct' not in heatmap_df.columns or heatmap_df['EngagedPct'].dropna().empty:
        return None

    if sort_method == "Average Engagement":
        store_order = heatmap_df.groupby('StoreID')['EngagedPct'].mean().sort_values(ascending=False).index.tolist()
    else: # Recent Performance
        most_recent_week = heatmap_df[COL_WEEK].max()
        store_order = heatmap_df[heatmap_df[COL_WEEK] == most_recent_week].sort_values('EngagedPct', ascending=False)['StoreID'].tolist()

    heatmap_chart = alt.Chart(heatmap_df).mark_rect().encode(
        x=alt.X(f'{COL_WEEK}:O', title='Week'),
        y=alt.Y('StoreID:O', title='Store', sort=store_order),
        color=alt.Color('EngagedPct:Q', title='Engaged %', scale=alt.Scale(scheme=color_scheme), legend=alt.Legend(orient='right')),
        tooltip=['StoreID', alt.Tooltip(f'{COL_WEEK}:O', title='Week'), alt.Tooltip('EngagedPct:Q', format='.2f')]
    ).properties(
        height=max(250, HEATMAP_HEIGHT_PER_ROW * len(store_order)) # Dynamic height
    )
    return heatmap_chart

def create_comparison_bar_chart(comp_data: pd.DataFrame, title: str) -> Optional[alt.Chart]:
    """Creates the main store comparison bar chart."""
    if comp_data.empty: return None

    bar_chart = alt.Chart(comp_data).mark_bar().encode(
        y=alt.Y(f'{COL_STORE}:N', title='Store', sort='-x'), # Sort descending by engagement
        x=alt.X(f'{COL_ENGAGED}:Q', title='Engaged Transaction %'),
        color=alt.Color(f'{COL_ENGAGED}:Q', scale=alt.Scale(scheme='blues'), legend=None), # Remove redundant legend
        tooltip=[COL_STORE, alt.Tooltip(f'{COL_ENGAGED}:Q', format='.2f')]
    ).properties(
        title=title,
        height=max(150, BAR_CHART_HEIGHT_PER_ROW * len(comp_data)) # Dynamic height
    )
    return bar_chart

def create_difference_bar_chart(comp_data: pd.DataFrame) -> Optional[alt.Chart]:
    """Creates the bar chart showing difference from average."""
    if comp_data.empty or 'Percentage' not in comp_data.columns: return None

    min_perc, max_perc = comp_data['Percentage'].min(), comp_data['Percentage'].max()
    # Ensure domain includes zero for the color scale midpoint
    color_domain = [min(min_perc, -0.01), 0, max(max_perc, 0.01)] # Use small offset if all values are same sign
    color_range = [RECENT_TREND_COLORS["Declining"], '#BBBBBB', RECENT_TREND_COLORS["Improving"]] # Red, Grey, Green

    diff_chart = alt.Chart(comp_data).mark_bar().encode(
        y=alt.Y(f'{COL_STORE}:N', title='Store', sort='-x'), # Sort descending by percentage diff
        x=alt.X('Percentage:Q', title='% Difference from Average'),
        color=alt.Color('Percentage:Q', scale=alt.Scale(domain=color_domain, range=color_range, type='linear'), legend=None),
        tooltip=[COL_STORE, alt.Tooltip(f'{COL_ENGAGED}:Q', format='.2f', title='Engaged %'), alt.Tooltip('Percentage:Q', format='+.1f', title='% Diff from Avg')]
    ).properties(
        height=max(150, BAR_CHART_HEIGHT_PER_ROW * len(comp_data)) # Dynamic height
    )
    return diff_chart

def create_rank_chart(rank_data: pd.DataFrame) -> Optional[alt.Chart]:
    """Creates the weekly rank tracking line chart."""
    if rank_data.empty or COL_RANK not in rank_data.columns: return None

    rank_chart = alt.Chart(rank_data).mark_line(point=True).encode(
        x=alt.X(f'{COL_WEEK}:O', title='Week'),
        y=alt.Y(f'{COL_RANK}:Q', title='Rank', scale=alt.Scale(reverse=True)), # Reverse scale so 1 is at top
        color=alt.Color(f'{COL_STORE}:N', scale=alt.Scale(scheme='category10'), title='Store'),
        tooltip=[COL_STORE, alt.Tooltip(f'{COL_WEEK}:O', title='Week'), alt.Tooltip(f'{COL_RANK}:Q', title='Rank')]
    ).properties(
        height=RANK_CHART_HEIGHT
    )

    # Add interactive legend
    rank_sel = alt.selection_point(fields=[COL_STORE], bind='legend')
    rank_chart = rank_chart.add_params(rank_sel).encode(
        opacity=alt.condition(rank_sel, alt.value(1), alt.value(0.2)),
        strokeWidth=alt.condition(rank_sel, alt.value(3), alt.value(1))
    )
    return rank_chart

def create_recent_trend_change_chart(dir_df: pd.DataFrame) -> Optional[alt.Chart]:
     """ Creates the bar chart showing total engagement change for recent trends. """
     if dir_df.empty or 'total_change' not in dir_df.columns: return None

     change_chart = alt.Chart(dir_df).mark_bar().encode(
         x=alt.X('total_change:Q', title=f'Change in Engagement % (Last {dir_df["weeks"].iloc[0]} Weeks)'),
         y=alt.Y('store:N', sort=alt.EncodingSortField(field='total_change', order='descending'), title='Store'),
         color=alt.Color('direction:N',
                         scale=alt.Scale(domain=['Improving', 'Stable', 'Declining'],
                                         range=[RECENT_TREND_COLORS['Improving'], RECENT_TREND_COLORS['Stable'], RECENT_TREND_COLORS['Declining']]),
                         title='Trend Direction'),
         tooltip=[alt.Tooltip('store:N', title='Store'),
                  alt.Tooltip('direction:N', title='Direction'),
                  alt.Tooltip('strength:N', title='Performance Detail'),
                  alt.Tooltip('start_value:Q', format='.2f', title='Starting Value'),
                  alt.Tooltip('current_value:Q', format='.2f', title='Current Value'),
                  alt.Tooltip('total_change:Q', format='+.2f', title='Total Change')]
     ).properties(
         height=max(250, BAR_CHART_HEIGHT_PER_ROW * len(dir_df))
     )
     # Add a rule at zero for reference
     zero_line = alt.Chart(pd.DataFrame({'x': [0]})).mark_rule(color='white', strokeDash=[3, 3]).encode(x='x:Q')

     return change_chart + zero_line


# --- UI Rendering Functions ---

def display_sidebar(df: pd.DataFrame) -> Tuple[str, str, list, float, bool, int, Optional[st.runtime.uploaded_file_manager.UploadedFile]]:
    """Displays the sidebar controls and returns their values."""
    st.sidebar.header("Data Input")
    data_file = st.sidebar.file_uploader("Upload engagement data (Excel or CSV)", type=['csv', 'xlsx', 'xls'])
    comp_file = st.sidebar.file_uploader("Optional: Upload comparison data (prior period)", type=['csv', 'xlsx', 'xls'])

    # Filters (only show if primary data is loaded)
    quarter_choice = "All"
    week_choice = "All"
    store_choice = []
    z_threshold = DEFAULT_Z_SCORE_THRESHOLD
    show_ma = True
    trend_analysis_weeks = 4

    if df is not None and not df.empty:
        st.sidebar.header("Filters")

        # Quarter Filter
        if 'Quarter' in df.columns and df['Quarter'].notna().any():
            quarters = sorted(df['Quarter'].dropna().unique().astype(int).tolist())
            quarter_options = ["All"] + [f"Q{q}" for q in quarters]
            quarter_choice = st.sidebar.selectbox("Select Quarter", quarter_options, index=0)
            # Determine available weeks based on quarter choice
            if quarter_choice != "All" and COL_WEEK in df.columns:
                 try:
                      q_num = int(quarter_choice[1:])
                      weeks_in_quarter = sorted(df[df['Quarter'] == q_num][COL_WEEK].dropna().unique().astype(int))
                 except: # Handle potential errors if conversion fails
                      weeks_in_quarter = sorted(df[COL_WEEK].dropna().unique().astype(int))
            elif COL_WEEK in df.columns:
                 weeks_in_quarter = sorted(df[COL_WEEK].dropna().unique().astype(int))
            else:
                 weeks_in_quarter = []
        else: # No Quarter data
             quarter_choice = "All"
             if COL_WEEK in df.columns:
                  weeks_in_quarter = sorted(df[COL_WEEK].dropna().unique().astype(int))
             else:
                  weeks_in_quarter = []


        # Week Filter
        if weeks_in_quarter:
             week_options = ["All"] + [str(w) for w in weeks_in_quarter]
             week_choice = st.sidebar.selectbox("Select Week", week_options, index=0)
        else:
             week_choice = "All"
             st.sidebar.info("No 'Week' data available for filtering.")


        # Store Filter
        if COL_STORE in df.columns and df[COL_STORE].notna().any():
            store_list = sorted(df[COL_STORE].dropna().unique().tolist())
            store_choice = st.sidebar.multiselect("Select Store(s)", store_list, default=[])
        else:
             store_choice = []
             st.sidebar.info("No 'Store #' data available for filtering.")


        # Advanced Settings
        with st.sidebar.expander("Advanced Settings", expanded=False):
            z_threshold = st.slider("Anomaly Z-score Threshold", 1.0, 3.0, DEFAULT_Z_SCORE_THRESHOLD, 0.1, help="Sensitivity for anomaly detection. Lower values detect smaller changes.")
            show_ma = st.checkbox("Show 4-week moving average", value=True)
            trend_analysis_weeks = st.slider("Trend analysis window (weeks)", 3, 8, 4, help="Number of recent weeks used to determine store trend direction.")
            st.caption("Adjust sensitivity for anomaly detection and trend analysis.")
    else:
         st.sidebar.info("Upload primary data to enable filters.")


    # Sidebar Footer
    now = datetime.datetime.now()
    st.sidebar.markdown("---")
    st.sidebar.caption(f"© Publix Super Markets, Inc. {now.year}")
    st.sidebar.caption(f"Last updated: {now.strftime('%Y-%m-%d')}")
    with st.sidebar.expander("Help & Information"):
        st.markdown("""
            ### Using This Dashboard
            - **Upload Data**: Start by uploading your primary engagement data file (CSV or Excel). Optionally, upload a comparison period file.
            - **Data Format**: Ensure columns like 'Store #', 'Week' (or 'Date'), and 'Engaged Transaction %' are present. The tool attempts to map common variations.
            - **Apply Filters**: Use the sidebar filters to focus on specific time periods or stores.
            - **Explore Tabs**: Each tab provides different insights:
                - **Engagement Trends**: Performance over time, heatmap, and recent directional trends.
                - **Store Comparison**: Compare stores directly via bar charts and rank tracking.
                - **Store Performance Categories**: Automatic categorization with action plans.
                - **Anomalies & Insights**: Unusual weekly changes and specific recommendations.
            - **Troubleshooting**: If data doesn't load, check column names and formats. Error messages may appear below the file uploader or within specific sections.

            For technical support, contact Reid.
        """)


    return quarter_choice, week_choice, store_choice, z_threshold, show_ma, trend_analysis_weeks, comp_file


def render_executive_summary(df_filtered: pd.DataFrame, store_perf: pd.Series, store_trends: pd.Series,
                             week_choice: str, store_choice: list, store_list: list):
    """Displays the executive summary metrics and insights."""
    st.subheader("Executive Summary")

    if df_filtered.empty or store_perf.empty:
        st.warning("No data available for the selected filters to generate summary.", icon="⚠️")
        return

    # Determine current and previous week for comparison
    current_week = None
    prev_week = None
    if COL_WEEK in df_filtered.columns and df_filtered[COL_WEEK].notna().any():
         valid_weeks = df_filtered[COL_WEEK].dropna().astype(int)
         if not valid_weeks.empty:
              current_week = int(valid_weeks.max())
              prev_weeks_data = valid_weeks[valid_weeks < current_week]
              if not prev_weeks_data.empty:
                   prev_week = int(prev_weeks_data.max())


    # Calculate overall average and change
    current_avg = df_filtered[df_filtered[COL_WEEK] == current_week][COL_ENGAGED].mean() if current_week is not None else df_filtered[COL_ENGAGED].mean()
    prev_avg = df_filtered[df_filtered[COL_WEEK] == prev_week][COL_ENGAGED].mean() if prev_week is not None else None

    delta_val = current_avg - prev_avg if current_avg is not None and prev_avg is not None else None
    delta_str = f"{delta_val:+.2f}%" if delta_val is not None else None # Display N/A handled by st.metric


    # Top/Bottom Performers
    top_store = store_perf.idxmax() if not store_perf.empty else "N/A"
    bottom_store = store_perf.idxmin() if not store_perf.empty else "N/A"
    top_val = store_perf.max() if not store_perf.empty else None
    bottom_val = store_perf.min() if not store_perf.empty else None

    # Display Metrics
    col1, col2, col3 = st.columns(3)

    # Determine label for the main metric
    if store_choice and len(store_choice) == 1:
        avg_label = f"Store {store_choice[0]} Engagement"
    elif store_choice and len(store_choice) < len(store_list):
        avg_label = "Selected Stores Avg Engagement"
    else:
        avg_label = "District Avg Engagement"

    week_label = f"(Week {current_week})" if current_week is not None else "(Period Avg)"
    col1.metric(f"{avg_label} {week_label}", f"{current_avg:.2f}%" if current_avg is not None else "N/A", delta=delta_str)

    col2.metric(f"Top Performer {week_label}", f"Store {top_store}" if top_store != "N/A" else "N/A", f"{top_val:.2f}%" if top_val is not None else None, delta_color="off")
    col3.metric(f"Bottom Performer {week_label}", f"Store {bottom_store}" if bottom_store != "N/A" else "N/A", f"{bottom_val:.2f}%" if bottom_val is not None else None, delta_color="off")


    # Summary Sentence
    if delta_val is not None and prev_week is not None:
        trend_dir = "up" if delta_val > 0 else "down" if delta_val < 0 else "flat"
        trend_class = "highlight-good" if delta_val > 0 else "highlight-bad" if delta_val < 0 else "highlight-neutral"
        st.markdown(f"Week {current_week} average engagement is <span class='{trend_class}'>{abs(delta_val):.2f} percentage points {trend_dir}</span> from Week {prev_week}.", unsafe_allow_html=True)
    elif current_avg is not None and current_week is not None:
        st.markdown(f"Week {current_week} engagement average: <span class='highlight-neutral'>{current_avg:.2f}%</span>.", unsafe_allow_html=True)
    elif current_avg is not None:
         st.markdown(f"Period engagement average: <span class='highlight-neutral'>{current_avg:.2f}%</span>.", unsafe_allow_html=True)


    # Top & Bottom Trend Labels (if trends are available)
    if not store_trends.empty and top_store != "N/A" and bottom_store != "N/A":
         colA, colB = st.columns(2)
         top_trend = store_trends.get(top_store, "N/A")
         bottom_trend = store_trends.get(bottom_store, "N/A")

         tcolor_class = "highlight-good" if top_trend in ["Upward", "Strong Upward"] else "highlight-bad" if top_trend in ["Downward", "Strong Downward"] else "highlight-neutral"
         bcolor_class = "highlight-good" if bottom_trend in ["Upward", "Strong Upward"] else "highlight-bad" if bottom_trend in ["Downward", "Strong Downward"] else "highlight-neutral"

         colA.markdown(f"**Store {top_store}** trend: <span class='{tcolor_class}'>{top_trend}</span>", unsafe_allow_html=True)
         colB.markdown(f"**Store {bottom_store}** trend: <span class='{bcolor_class}'>{bottom_trend}</span>", unsafe_allow_html=True)

    st.divider() # Add a visual separator

    # Key Insights Section
    st.subheader("Key Insights")
    insights = []
    # Consistency (requires std dev)
    if COL_ENGAGED in df_filtered.columns and len(df_filtered) > 1:
        store_std = df_filtered.groupby(COL_STORE)[COL_ENGAGED].std().fillna(0)
        if not store_std.empty and len(store_std) > 1:
            most_consistent = store_std.idxmin()
            least_consistent = store_std.idxmax()
            insights.append(f"**Store {most_consistent}** shows the most consistent engagement (std dev: {store_std.min():.2f}).")
            insights.append(f"**Store {least_consistent}** shows the most variable engagement (std dev: {store_std.max():.2f}).")

    # Trending stores insights (if trends available)
    if not store_trends.empty:
        trending_up = [str(s) for s, t in store_trends.items() if t in ["Upward", "Strong Upward"]]
        trending_down = [str(s) for s, t in store_trends.items() if t in ["Downward", "Strong Downward"]]
        if trending_up:
            insights.append("Stores showing positive trends: " + ", ".join(f"**{s}**" for s in trending_up))
        if trending_down:
            insights.append("Stores needing attention (negative trends): " + ", ".join(f"**{s}**" for s in trending_down))

    # Gap insights
    if top_val is not None and bottom_val is not None and len(store_perf) > 1:
        gap = top_val - bottom_val
        insights.append(f"Gap between highest and lowest performing stores: **{gap:.2f}%**.")
        if gap > 10: # Example threshold for large gap
            insights.append("🚨 Large performance gap detected. Consider sharing best practices from top performers.")

    # Display first up to 5 insights
    if insights:
        for i, point in enumerate(insights[:5], start=1):
            st.markdown(f"{i}. {point}")
    else:
        st.info("Not enough data or variation to generate specific insights for the current selection.")


def render_engagement_trends_tab(df_filtered: pd.DataFrame, df_comp_filtered: Optional[pd.DataFrame],
                                 store_list: list, show_ma: bool):
    """Renders the content for the Engagement Trends tab."""
    st.subheader("Engagement Trends Over Time")

    if df_filtered.empty or COL_WEEK not in df_filtered.columns or COL_ENGAGED not in df_filtered.columns:
        st.warning("Engagement trend analysis requires 'Week' and 'Engaged Transaction %' data for the selected filters.", icon="⚠️")
        return

    view_option = st.radio("View mode:", ["All Stores", "Custom Selection", "Recent Trends"], horizontal=True, key="trend_view",
                           help="All Stores: View all selected stores | Custom Selection: Pick specific stores | Recent Trends: Focus on a specific range of weeks")

    # --- Data Preparation for Trends ---
    # Calculate district average trend and moving averages
    df_processed = df_filtered.sort_values([COL_STORE, COL_WEEK])
    dist_trend = df_processed.groupby(COL_WEEK, as_index=False)[COL_ENGAGED].mean().rename(columns={COL_ENGAGED:'Average Engagement %'})

    # Calculate Moving Averages if selected
    if show_ma:
        dist_trend['MA_4W'] = dist_trend['Average Engagement %'].rolling(window=4, min_periods=1).mean()
        df_processed['MA_4W'] = df_processed.groupby(COL_STORE)[COL_ENGAGED].transform(lambda s: s.rolling(4, min_periods=1).mean())

    # Combine with comparison data if available
    dist_trend_comp = None
    if df_comp_filtered is not None and not df_comp_filtered.empty:
        df_processed['Period'] = 'Current'
        df_comp_processed = df_comp_filtered.sort_values([COL_STORE, COL_WEEK]).copy()
        df_comp_processed['Period'] = 'Comparison'
        if show_ma:
             # Calculate MA for comparison period stores separately
             df_comp_processed['MA_4W'] = df_comp_processed.groupby(COL_STORE)[COL_ENGAGED].transform(lambda s: s.rolling(4, min_periods=1).mean())
             # Calculate district MA for comparison period
             dist_trend_comp_raw = df_comp_processed.groupby(COL_WEEK, as_index=False)[COL_ENGAGED].mean().sort_values(COL_WEEK)
             dist_trend_comp_raw['MA_4W'] = dist_trend_comp_raw[COL_ENGAGED].rolling(4, min_periods=1).mean()
             dist_trend_comp = dist_trend_comp_raw # Assign calculated df
        else:
             dist_trend_comp = df_comp_processed.groupby(COL_WEEK, as_index=False)[COL_ENGAGED].mean().sort_values(COL_WEEK)


        combined = pd.concat([df_processed, df_comp_processed], ignore_index=True)
    else:
        combined = df_processed.copy()
        combined['Period'] = 'Current' # Add period column even if no comparison

    # Filter for Recent Trends view
    stores_to_show_custom = [] # Initialize for custom view
    if view_option == "Recent Trends":
        all_weeks = sorted(combined[COL_WEEK].dropna().unique().astype(int))
        if len(all_weeks) >= 2:
            default_start = all_weeks[0] if len(all_weeks) <= 8 else all_weeks[-8]
            default_end = all_weeks[-1]
            recent_weeks_range = st.select_slider("Select weeks to display:", options=all_weeks, value=(default_start, default_end),
                                                  help="Adjust the slider to show a shorter or longer recent period")
            recent_weeks = [w for w in all_weeks if w >= recent_weeks_range[0] and w <= recent_weeks_range[1]]
            combined = combined[combined[COL_WEEK].isin(recent_weeks)]
            dist_trend = dist_trend[dist_trend[COL_WEEK].isin(recent_weeks)]
            if dist_trend_comp is not None:
                dist_trend_comp = dist_trend_comp[dist_trend_comp[COL_WEEK].isin(recent_weeks)]
        elif len(all_weeks) == 1:
            st.info("Only one week of data available in selection, showing that week.")
        else:
            st.warning("No weekly data available for 'Recent Trends' view in the current selection.", icon="⚠️")
            return # Cannot proceed with recent trends view

    elif view_option == "Custom Selection":
         available_stores = sorted(df_processed[COL_STORE].unique())
         if available_stores:
              default_selection = [available_stores[0]] if available_stores else []
              stores_to_show_custom = st.multiselect("Select stores to compare:", options=available_stores, default=default_selection, key="trend_store_select")
              if not stores_to_show_custom:
                   st.info("Please select at least one store for 'Custom Selection' view.")
                   # Don't return yet, let the chart function handle empty selection later
         else:
              st.info("No stores available in the current filter selection.")
              return # Cannot proceed


    # --- Display Trend Chart ---
    trend_chart = create_trend_chart(combined, dist_trend, dist_trend_comp, view_option, stores_to_show_custom, show_ma)

    if trend_chart:
         # Display extra metrics for Recent Trends view
         if view_option == "Recent Trends":
              c1, c2 = st.columns(2)
              with c1:
                   # Calculate Week-over-Week District Trend
                   if not dist_trend.empty and len(dist_trend[COL_WEEK].unique()) >= 2:
                        last_two_weeks = sorted(dist_trend[COL_WEEK].unique())[-2:]
                        cur_val_series = dist_trend[dist_trend[COL_WEEK] == last_two_weeks[1]]['Average Engagement %']
                        prev_val_series = dist_trend[dist_trend[COL_WEEK] == last_two_weeks[0]]['Average Engagement %']

                        if not cur_val_series.empty and not prev_val_series.empty:
                             cur_val = float(cur_val_series.iloc[0])
                             prev_val = float(prev_val_series.iloc[0])
                             change_pct = ((cur_val - prev_val) / prev_val * 100) if prev_val != 0 else 0
                             st.metric("District Trend (Week-over-Week)", f"{cur_val:.2f}%", f"{change_pct:+.1f}%")
                        else:
                             st.metric("District Trend (Week-over-Week)", "N/A")

              with c2:
                   # Display Top Performer for the last week in the selection
                   last_week_data = combined[combined[COL_WEEK] == combined[COL_WEEK].max()]
                   if not last_week_data.empty:
                        best_store_row = last_week_data.loc[last_week_data[COL_ENGAGED].idxmax()]
                        st.metric(f"Top Performer (Week {int(best_store_row[COL_WEEK])})", f"Store {best_store_row[COL_STORE]}", f"{best_store_row[COL_ENGAGED]:.2f}%", delta_color="off")
                   else:
                        st.metric("Top Performer", "N/A")

         st.altair_chart(trend_chart, use_container_width=True)

         # Caption explaining the view
         caption_base = ""
         if view_option == "All Stores":
              caption_base = "**All Stores View:** Shows trends for all selected stores. Use the interactive legend to highlight specific stores. Black dashed line = district average."
         elif view_option == "Custom Selection":
              caption_base = "**Custom Selection View:** Shows only the stores selected in the multiselect box above, with thicker lines and points."
         elif view_option == "Recent Trends":
              caption_base = "**Recent Trends View:** Focuses on the weeks selected in the slider. Week-over-week district trend and top performer for the last selected week are shown above the chart."

         if dist_trend_comp is not None:
              caption_base += " Gray dashed line = comparison period's district average."
         st.caption(caption_base)

    else:
        st.info("No data available to display the trend chart for the current selection and view mode.")

    st.divider()

    # --- Weekly Engagement Heatmap ---
    st.subheader("Weekly Engagement Heatmap")

    # Rename columns for heatmap function
    heatmap_data_input = df_filtered[[COL_STORE, COL_WEEK, COL_ENGAGED]].copy()
    heatmap_data_input.rename(columns={COL_STORE: 'StoreID', COL_ENGAGED: 'EngagedPct'}, inplace=True)


    if heatmap_data_input.empty or 'EngagedPct' not in heatmap_data_input.columns or heatmap_data_input['EngagedPct'].dropna().empty:
         st.info("No data available for the heatmap with the current filters.")
    else:
         with st.expander("Heatmap Settings", expanded=False):
              colA, colB, colC = st.columns(3)
              with colA:
                   sort_method = st.selectbox("Sort stores by:", ["Average Engagement", "Recent Performance"], index=0, key="heatmap_sort")
              with colB:
                   color_scheme = st.selectbox("Color scheme:", ALTAIR_COLOR_SCHEMES, index=0, key="heatmap_color").lower()
              # Week Range Slider for Heatmap
              weeks_list = sorted(heatmap_data_input[COL_WEEK].dropna().unique().astype(int))
              if len(weeks_list) > 1:
                   with colC:
                       selected_range = st.select_slider("Select week range for heatmap:", options=weeks_list, value=(min(weeks_list), max(weeks_list)), key="heatmap_week_range")
                   heatmap_df_filtered = heatmap_data_input[(heatmap_data_input[COL_WEEK] >= selected_range[0]) & (heatmap_data_input[COL_WEEK] <= selected_range[1])].copy()
              else:
                   heatmap_df_filtered = heatmap_data_input.copy() # Use all data if only one week
                   st.caption("Only one week selected, showing heatmap for that week.")


         heatmap_chart = create_heatmap(heatmap_df_filtered, sort_method, color_scheme)
         if heatmap_chart:
              st.altair_chart(heatmap_chart, use_container_width=True)
              min_w, max_w = int(heatmap_df_filtered[COL_WEEK].min()), int(heatmap_df_filtered[COL_WEEK].max())
              week_range_text = f"Week {min_w}" if min_w == max_w else f"Weeks {min_w} to {max_w}"
              st.caption(f"Showing data from {week_range_text}. Stores sorted by {sort_method.lower()}. Darker colors indicate higher engagement.")
         else:
              st.info("Not enough data to generate heatmap for the selected week range.")

         st.divider()

         # --- Recent Performance Trends Section ---
         st.subheader("Recent Performance Direction")
         with st.expander("About This Section", expanded=False):
             st.write("""
                 This section analyzes short-term **directional trends** over the last few weeks for each store within the selected heatmap range.
                 It helps identify stores that are recently improving, declining, or holding steady, complementing the longer-term categories on Tab 3.
                 - **Improving**: Engagement showing recent upward momentum.
                 - **Stable**: Engagement relatively consistent recently.
                 - **Declining**: Engagement showing recent downward momentum.
             """)

         # Use the data filtered by the heatmap's week range
         trend_df_input = heatmap_df_filtered.copy()

         if trend_df_input.empty:
              st.info("No data available to analyze recent trends based on heatmap selection.")
         else:
              col1, col2 = st.columns(2)
              with col1:
                   trend_window = st.slider("Number of recent weeks to analyze", min_value=2, max_value=min(8, len(trend_df_input[COL_WEEK].unique())), value=min(4, len(trend_df_input[COL_WEEK].unique())), key="recent_trend_window", help="How many trailing weeks (within the heatmap selection) to use for calculating direction.")
              with col2:
                   sensitivity = st.select_slider("Sensitivity to change", options=["Low", "Medium", "High"], value="Medium", key="recent_trend_sensitivity", help="How much change is needed to be classified as 'Improving' or 'Declining'. High sensitivity flags smaller changes.")
                   momentum_threshold = RECENT_TREND_SENSITIVITY[sensitivity]

              directions = []
              for store_id, data in trend_df_input.groupby('StoreID'):
                   if len(data) < trend_window: continue # Need enough data points for the window

                   recent = data.sort_values(COL_WEEK).tail(trend_window)
                   vals = recent['EngagedPct'].values
                   weeks_for_slope = np.arange(len(vals)) # Use simple index for slope

                   # Check for NaNs before calculations
                   valid_mask = ~np.isnan(vals)
                   vals_clean = vals[valid_mask]
                   weeks_clean = weeks_for_slope[valid_mask]

                   if len(vals_clean) < 2: continue # Need at least two points

                   # Calculate change (simple start vs end of window average)
                   first_half_avg = vals_clean[0] if len(vals_clean) <= 3 else vals_clean[:len(vals_clean)//2].mean()
                   second_half_avg = vals_clean[-1] if len(vals_clean) <= 3 else vals_clean[-(len(vals_clean)//2):].mean()
                   change = second_half_avg - first_half_avg

                   # Get actual start and end values for display
                   start_val = vals_clean[0]
                   current_val = vals_clean[-1]
                   total_change = current_val - start_val

                   # Calculate slope using polyfit on cleaned data
                   try:
                       slope = np.polyfit(weeks_clean, vals_clean, 1)[0]
                   except (np.linalg.LinAlgError, ValueError):
                       slope = 0 # Default to stable if fit fails


                   # Classify Direction and Strength
                   if abs(change) < momentum_threshold and abs(slope) < TREND_SLOPE_POS: # Require both low change and low slope for stable
                        direction, strength = "Stable", "Holding Steady"
                   elif change > 0 or slope > TREND_SLOPE_POS: # Improving if change is positive OR slope is positive enough
                        direction = "Improving"
                        strength = "Strong Improvement" if change > 2 * momentum_threshold or slope > TREND_SLOPE_STRONG_POS else "Gradual Improvement"
                   else: # Declining otherwise
                        direction = "Declining"
                        strength = "Significant Decline" if change < -2 * momentum_threshold or slope < TREND_SLOPE_STRONG_NEG else "Gradual Decline"

                   indicator_map = {"Improving": "↗️", "Stable": "➡️", "Declining": "↘️"}
                   indicator_strength_map = {"Strong Improvement": "🔼", "Significant Decline": "🔽"}
                   indicator = indicator_strength_map.get(strength, indicator_map.get(direction, "❔"))

                   directions.append({
                       'store': store_id,
                       'direction': direction,
                       'strength': strength,
                       'indicator': indicator,
                       'start_value': start_val,
                       'current_value': current_val,
                       'total_change': total_change,
                       'color': RECENT_TREND_COLORS[direction],
                       'weeks': trend_window,
                       'slope': slope
                   })

              dir_df = pd.DataFrame(directions)

              if dir_df.empty:
                   st.info(f"Not enough consecutive weekly data (requires {trend_window}) within the selected heatmap range to analyze recent trends.")
              else:
                   # Display summary metrics
                   col_imp, col_stab, col_dec = st.columns(3)
                   imp_count = (dir_df['direction'] == 'Improving').sum()
                   stab_count = (dir_df['direction'] == 'Stable').sum()
                   dec_count = (dir_df['direction'] == 'Declining').sum()

                   col_imp.metric("Improving Recently", f"{imp_count} stores", delta="↗️", delta_color="normal")
                   col_stab.metric("Stable Recently", f"{stab_count} stores", delta="➡️", delta_color="off")
                   col_dec.metric("Declining Recently", f"{dec_count} stores", delta="↘️", delta_color="inverse")

                   st.markdown("---") # Separator

                   # Display stores grouped by direction
                   cols_per_row = 3 # Adjust as needed
                   for direction, group in dir_df.groupby('direction'):
                       color = RECENT_TREND_COLORS[direction]
                       st.markdown(f"<div class='styled-div-border' style='border-left-color:{color};'><h4 style='color:{color};'>{direction} ({len(group)} stores)</h4></div>", unsafe_allow_html=True)

                       cols = st.columns(min(cols_per_row, len(group)))
                       group = group.sort_values('total_change', ascending=(direction == 'Declining')) # Sort by magnitude within group
                       for i, (_, store_data) in enumerate(group.iterrows()):
                           with cols[i % cols_per_row]:
                               change_disp = f"{store_data['total_change']:+.2f}%"
                               # Use markdown with inline styles for card appearance
                               st.markdown(f"""
                               <div class='store-card' style='border-left: 5px solid {store_data['color']};'>
                                   <h4 style='color:{store_data['color']};'>{store_data['indicator']} Store {store_data['store']}</h4>
                                   <p><strong>{store_data['strength']}</strong><br>
                                   <span style='font-size:0.9em;'><strong>{change_disp}</strong> over {store_data['weeks']} weeks</span><br>
                                   <span style='font-size:0.85em; color:#BBBBBB;'>{store_data['start_value']:.2f}% → {store_data['current_value']:.2f}%</span></p>
                               </div>
                               """, unsafe_allow_html=True)
                       st.markdown("<br>", unsafe_allow_html=True) # Add space between groups


                   st.markdown("---")
                   st.subheader("Overall Change Distribution (Recent Trend Window)")
                   change_chart = create_recent_trend_change_chart(dir_df)
                   if change_chart:
                        st.altair_chart(change_chart, use_container_width=True)
                        st.caption("This chart shows the total percentage point change from the start to the end of the selected recent trend window for each store.")
                   else:
                        st.info("Could not generate the change distribution chart.")


def render_store_comparison_tab(df_filtered: pd.DataFrame, week_choice: str, store_list: list):
    """Renders the content for the Store Comparison tab."""
    st.subheader("Store Performance Comparison")

    if df_filtered.empty or COL_STORE not in df_filtered.columns or COL_ENGAGED not in df_filtered.columns:
        st.warning("Store comparison requires 'Store #' and 'Engaged Transaction %' data for the selected filters.", icon="⚠️")
        return

    if len(store_list) <= 1:
        st.info("Please select at least two stores in the sidebar filters to enable comparison view.")
        return

    # Prepare comparison data (either single week or period average)
    if week_choice != "All" and COL_WEEK in df_filtered.columns:
        try:
             week_num = int(week_choice)
             comp_data = df_filtered[df_filtered[COL_WEEK] == week_num].copy()
             # If no data for that specific week, fallback to average? Or just show error?
             if comp_data.empty:
                  st.warning(f"No data found for Week {week_num}. Showing period average instead.", icon="⚠️")
                  comp_data = df_filtered.groupby(COL_STORE, as_index=False)[COL_ENGAGED].mean()
                  comp_title = f"Store Comparison - Period Average (Week {week_num} not found)"
             else:
                  comp_title = f"Store Comparison - Week {week_num}"
        except ValueError:
             st.warning(f"Invalid week selected ({week_choice}). Showing period average.", icon="⚠️")
             comp_data = df_filtered.groupby(COL_STORE, as_index=False)[COL_ENGAGED].mean()
             comp_title = "Store Comparison - Period Average"
    else: # All weeks selected or no week column
        comp_data = df_filtered.groupby(COL_STORE, as_index=False)[COL_ENGAGED].mean()
        comp_title = "Store Comparison - Period Average"


    if comp_data.empty:
         st.warning("No comparison data could be prepared for the current selection.", icon="⚠️")
         return

    # Sort data for charts
    comp_data = comp_data.sort_values(COL_ENGAGED, ascending=False)

    # --- Bar Chart: Absolute Performance ---
    bar_chart = create_comparison_bar_chart(comp_data, comp_title)
    if bar_chart:
        # Add average line
        avg_val = comp_data[COL_ENGAGED].mean()
        avg_rule = alt.Chart(pd.DataFrame({'avg': [avg_val]})).mark_rule(
            color='red', strokeDash=[4, 4], size=2
        ).encode(
            x='avg:Q',
            tooltip=[alt.Tooltip('avg:Q', title='District Average', format='.2f')]
        )
        st.altair_chart(bar_chart + avg_rule, use_container_width=True)
    else:
         st.info("Could not generate the main comparison bar chart.")


    st.divider()

    # --- Bar Chart: Performance Relative to Average ---
    st.subheader("Performance Relative to District Average")
    if avg_val is not None and avg_val != 0:
         comp_data['Difference'] = comp_data[COL_ENGAGED] - avg_val
         comp_data['Percentage'] = comp_data['Difference'] / avg_val * 100
         diff_chart = create_difference_bar_chart(comp_data)
         if diff_chart:
              # Add zero line for reference
              center_rule = alt.Chart(pd.DataFrame({'center': [0]})).mark_rule(color='black').encode(x='center:Q')
              st.altair_chart(diff_chart + center_rule, use_container_width=True)
              st.caption("Green bars indicate performance above the average of the selected stores/period; red bars indicate performance below average.")
         else:
              st.info("Could not generate the relative performance bar chart.")

    elif avg_val == 0:
         st.info("District average is zero, cannot calculate relative performance.")
    else:
         st.info("Could not calculate district average.")


    st.divider()

    # --- Rank Tracking Chart ---
    if COL_RANK in df_filtered.columns and COL_WEEK in df_filtered.columns:
        st.subheader("Weekly Rank Tracking")
        rank_data = df_filtered[[COL_WEEK, COL_STORE, COL_RANK]].dropna()
        rank_chart = create_rank_chart(rank_data)
        if rank_chart:
            st.altair_chart(rank_chart, use_container_width=True)
            st.caption("Tracks store rank over the selected period (lower rank number = better performance). Use the interactive legend to highlight stores.")
        else:
            st.info("Not enough data or no rank data available to display the rank tracking chart for the selected period.")
    else:
        st.info("Weekly rank tracking requires both 'Week' and 'Weekly Rank' columns in the data.")


def render_performance_categories_tab(store_stats: pd.DataFrame):
    """Renders the content for the Store Performance Categories tab."""
    st.subheader("Store Performance Categories")
    st.write("Stores are categorized based on their average engagement level relative to the median and their recent performance trend (correlation between week and engagement).")

    if store_stats.empty:
        st.warning("Performance categories could not be calculated. Check if data includes 'Store #', 'Engaged Transaction %', and 'Week'.", icon="⚠️")
        return

    # --- Category Definitions ---
    st.markdown("#### Category Definitions")
    col_defs1, col_defs2 = st.columns(2)
    cols_map = {0: col_defs1, 1: col_defs2}
    for i, cat in enumerate(CATEGORY_ORDER):
        col_idx = i % 2 # Alternate columns
        info = CATEGORY_CONFIG[cat]
        with cols_map[col_idx]:
             st.markdown(f"""
             <div class='category-card' style='border-left-color: {info['color']};'>
                 <h4 style='color:{info['color']};'>{info['icon']} {cat}</h4>
                 <p><strong>Definition:</strong> {info['explanation']}</p>
                 <p><strong>Recommended Focus:</strong> {info['action']}</p>
             </div>
             """, unsafe_allow_html=True)
    st.markdown("---") # Separator

    # --- Store Categorization Results ---
    st.markdown("#### Store Categorization Results")
    cols_per_row = 4 # Adjust as needed
    for cat in CATEGORY_ORDER:
        subset = store_stats[store_stats['Category'] == cat]
        if subset.empty: continue

        info = CATEGORY_CONFIG[cat]
        color = info['color']
        st.markdown(f"<div class='styled-div-border' style='border-left-color:{color};'><h4 style='color:{color};'>{info['icon']} {cat} ({len(subset)} stores)</h4></div>", unsafe_allow_html=True)

        cols = st.columns(min(cols_per_row, len(subset)))
        # Sort subset for consistent display, e.g., by engagement or store #
        subset = subset.sort_values('Average Engagement', ascending=False)
        for i, (_, store_row) in enumerate(subset.iterrows()):
            with cols[i % cols_per_row]:
                corr = store_row['Trend Correlation']
                # Define trend icons based on correlation strength
                trend_icon = "🔼" if corr > 0.3 else "↗️" if corr > TREND_CORR_THRESHOLD_POS else "🔽" if corr < -0.3 else "↘️" if corr < TREND_CORR_THRESHOLD_NEG else "➡️"

                st.markdown(f"""
                <div class='store-card' style='border-left: 5px solid {color};'>
                    <h4 style='color:{color};'>Store {store_row['Store #']}</h4>
                    <p><strong>Avg Engagement:</strong> {store_row['Average Engagement']:.2f}%<br>
                    <strong>Trend Corr:</strong> {trend_icon} {corr:.2f}</p>
                </div>
                """, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True) # Add space between categories

    st.markdown("---") # Separator

    # --- Store-Specific Action Plan ---
    st.subheader("Detailed Action Plan per Store")
    available_stores = sorted(store_stats['Store #'].unique())
    if not available_stores:
         st.info("No stores available to select for detailed action plan.")
         return

    selected_store = st.selectbox("Select a store:", available_stores, key="category_store_select")

    if selected_store:
        row = store_stats[store_stats['Store #'] == selected_store].iloc[0]
        cat = row['Category']

        if cat == "Uncategorized":
             st.warning(f"Store {selected_store} could not be categorized based on available data.")
        else:
             info = CATEGORY_CONFIG[cat]
             color = info['color']
             corr = row['Trend Correlation']
             avg_val = row['Average Engagement']

             # Trend Description and Icon
             trend_desc = ("Strong positive trend" if corr > 0.3 else "Mild positive trend" if corr > TREND_CORR_THRESHOLD_POS
                          else "Strong negative trend" if corr < -0.3 else "Mild negative trend" if corr < TREND_CORR_THRESHOLD_NEG
                          else "Stable trend")
             trend_icon = "🔼" if corr > 0.3 else "↗️" if corr > TREND_CORR_THRESHOLD_POS else "🔽" if corr < -0.3 else "↘️" if corr < TREND_CORR_THRESHOLD_NEG else "➡️"

             # Display Card
             st.markdown(f"""
             <div class='category-card' style='border-left-color: {color};'>
                 <h3 style='color:{color}; margin-top:0;'>Store {selected_store} - {cat}</h3>
                 <p><strong>Average Engagement:</strong> {avg_val:.2f}%</p>
                 <p><strong>Trend:</strong> {trend_icon} {trend_desc} (Correlation: {corr:.2f})</p>
                 <p><strong>Category Explanation:</strong> {row['Explanation']}</p>
                 <h4 style='color:{color}; margin-top:1em;'>Recommended Action Plan:</h4>
                 <p>{row['Action Plan']}</p>
             </div>
             """, unsafe_allow_html=True)

             # Improvement Opportunities / Learning Partners
             if cat in ["Improving", "Requires Intervention"]:
                  st.markdown("---")
                  st.subheader(f"Improvement Opportunities for Store {selected_store}")
                  # Suggest learning partners (Star Performers)
                  top_stores = store_stats[store_stats['Category'] == "Star Performer"]['Store #'].tolist()
                  if top_stores:
                       partners = ", ".join(f"**Store {s}**" for s in top_stores if s != selected_store) # Exclude self
                       if partners:
                            st.markdown(f"""
                            <div class='category-card' style='border-left-color: {CATEGORY_CONFIG['Star Performer']['color']};'>
                                <h4 style='color:{CATEGORY_CONFIG['Star Performer']['color']};'>Potential Learning Partners</h4>
                                <p>Consider discussing strategies with: {partners}</p>
                            </div>
                            """, unsafe_allow_html=True)

                  # Show potential gain to median
                  med = store_stats['Average Engagement'].median()
                  gain = med - avg_val
                  if gain > 0:
                       st.markdown(f"""
                       <div class='category-card' style='border-left-color: {CATEGORY_CONFIG['Improving']['color']}; margin-top:15px;'>
                           <h4 style='color:{CATEGORY_CONFIG['Improving']['color']};'>Potential Improvement Target</h4>
                           <p>Current Average: <strong>{avg_val:.2f}%</strong> | District Median: <strong>{med:.2f}%</strong> | Potential Gain to Median: <strong>{gain:.2f}%</strong></p>
                       </div>
                       """, unsafe_allow_html=True)


def render_anomalies_insights_tab(df_filtered: pd.DataFrame, store_stats: pd.DataFrame, z_threshold: float, trend_analysis_weeks: int):
    """Renders the content for the Anomalies & Insights tab."""
    st.subheader("Anomaly Detection")
    st.write(f"This section identifies significant week-over-week changes in engagement for individual stores, exceeding a Z-score threshold of **{z_threshold:.1f}**. A high Z-score indicates a change much larger than that store's typical weekly variation.")

    anomalies_df = find_anomalies(df_filtered, z_threshold)

    if anomalies_df.empty:
        st.info(f"No significant anomalies (Z-score > {z_threshold:.1f}) detected for the selected data and threshold.")
    else:
        st.write(f"Found {len(anomalies_df)} instance(s) where weekly engagement change exceeded the threshold:")
        with st.expander("View Anomaly Details", expanded=True):
            # Select and rename columns for display
            display_cols = ['Store #', 'Week', 'Engaged Transaction %', 'Change %pts', 'Z-score', 'Rank', 'Prev Rank', 'Possible Explanation']
            anomalies_to_display = anomalies_df[[col for col in display_cols if col in anomalies_df.columns]].copy()
            st.dataframe(anomalies_to_display, hide_index=True, use_container_width=True)
            st.caption("Z-score measures how many standard deviations a change is from the store's average weekly change. Rank/Prev Rank shown if available.")

    st.divider()

    # --- Store-Specific Recommendations ---
    st.subheader("Store-Specific Summary & Recommendations")
    st.write("Combines performance category, recent trend, and detected anomalies for each store.")

    recommendations = []
    # Use store list from store_stats if available and categorized, otherwise from df_filtered
    store_list_rec = sorted(store_stats['Store #'].unique()) if not store_stats.empty else sorted(df_filtered[COL_STORE].unique()) if COL_STORE in df_filtered else []


    if not store_list_rec:
        st.warning("No store data available to generate recommendations.", icon="⚠️")
        return

    # Recalculate trends specifically for recommendation context if needed (e.g., using full filtered data)
    _, store_trends_rec = calculate_trend_stats(df_filtered, trend_analysis_weeks)


    for store_id in store_list_rec:
        store_data = df_filtered[df_filtered[COL_STORE] == store_id]
        if store_data.empty: continue

        avg_eng = store_data[COL_ENGAGED].mean() if COL_ENGAGED in store_data else None
        trend = store_trends_rec.get(store_id, "N/A") # Get trend calculated earlier

        # Get category from store_stats DataFrame
        category = "N/A"
        if not store_stats.empty:
            cat_row = store_stats[store_stats['Store #'] == store_id]
            if not cat_row.empty:
                category = cat_row.iloc[0]['Category']

        # Get anomalies for this store
        store_anoms = anomalies_df[anomalies_df['Store #'] == store_id] if not anomalies_df.empty else pd.DataFrame()

        # --- Recommendation Logic ---
        rec = "Review performance category details and recent trends." # Default
        if category != "N/A" and category != "Uncategorized":
             base_rec = CATEGORY_CONFIG[category]['action']
             # Refine based on trend if it conflicts with category expectation
             if category == "Star Performer" and trend in ["Downward", "Strong Downward"]:
                  rec = f"{base_rec} However, monitor recent downward trend ({trend}). Investigate potential causes."
             elif category == "Requires Intervention" and trend in ["Upward", "Strong Upward"]:
                  rec = f"{base_rec} Positive signs noted: recent trend is {trend}. Reinforce actions driving improvement."
             elif category == "Needs Stabilization" and trend not in ["Downward", "Strong Downward"]:
                  rec = f"{base_rec} Focus on maintaining stability. Recent trend ({trend}) seems stable or positive, which is good."
             elif category == "Improving" and trend in ["Downward", "Strong Downward"]:
                  rec = f"{base_rec} Warning: Recent trend is {trend}. Identify obstacles and refocus efforts."
             else:
                  rec = base_rec # Use default category action if trend aligns
        elif avg_eng is not None: # Basic recommendation if no category
             rec = f"Average engagement: {avg_eng:.2f}%. Recent trend: {trend}. Focus on general best practices."
        else:
             rec = "Insufficient data for specific recommendation."


        # Append anomaly note if exists
        if not store_anoms.empty:
            # Sort anomalies by week for the note, take the most recent one
            biggest = store_anoms.sort_values('Week', ascending=False).iloc[0]
            change_type = 'positive spike' if biggest['Change %pts'] > 0 else 'negative drop'
            rec += f" **Anomaly Alert:** Investigate significant {change_type} ({biggest['Change %pts']:+.2f}%) in Week {int(biggest['Week'])} (Z={biggest['Z-score']:.2f})."

        recommendations.append({
            'Store #': store_id,
            'Category': category,
            'Recent Trend': trend,
            'Avg Engagement (%)': round(avg_eng, 2) if avg_eng is not None else None,
            'Recommendation': rec
        })

    if recommendations:
        rec_df = pd.DataFrame(recommendations)
        # Improve display using st.dataframe with configuration
        st.dataframe(rec_df,
                     hide_index=True,
                     use_container_width=True,
                     column_config={
                         "Avg Engagement (%)": st.column_config.NumberColumn(format="%.2f%%"),
                         "Recommendation": st.column_config.TextColumn(width="large") # Make recommendation column wider
                     }
                    )
    else:
        st.info("No data available to generate recommendations for the selected stores.")


# --- Main Application Flow ---

def main():
    st.set_page_config(**PAGE_CONFIG)
    st.markdown(CSS_STYLES, unsafe_allow_html=True)

    # Title and introduction
    st.markdown("<h1 class='dashboard-title'>Club Publix Engagement Dashboard</h1>", unsafe_allow_html=True)
    st.markdown("Analyze Club Publix engagement data. Upload weekly store data (CSV/Excel) and use the sidebar filters to explore trends, compare performance, and gain insights.")
    st.info("Ensure your data includes columns like 'Store #', 'Week' (or 'Date'), and 'Engaged Transaction %'. See Help section in sidebar for details.")


    # --- Load Data ---
    # Display sidebar first to get file uploaders
    # We pass None initially to display sidebar even without data
    quarter_choice, week_choice, store_choice, z_threshold, show_ma, trend_analysis_weeks, comp_file_obj = display_sidebar(None)

    # Get the uploaded primary file from the sidebar state (it was created in display_sidebar)
    # Need to access sidebar's widget state if not passed back directly - tricky.
    # Re-calling file_uploader here is simpler for now, but less ideal.
    # A better approach might involve session state or passing the uploader widget itself.
    data_file_obj = st.session_state.get('FormSubmitter:Data Input-Upload engagement data (Excel or CSV)', None) # Example key, inspect element to find correct key
    # Let's stick to the simpler approach of just re-getting it if not passed back
    # This part is awkward due to Streamlit's execution model. Ideally sidebar returns the file object.
    # Re-calling might reset state. Let's assume the initial call in display_sidebar works.

    df = None
    error_messages = []
    # Attempt to access the file object uploaded in the sidebar
    # NOTE: Accessing widget state directly is fragile. Best practice is evolving.
    # Check Streamlit documentation for recommended ways to handle inter-widget communication or state.
    # For now, let's re-render the uploader invisibly or rely on the passed-back object if possible.
    # Because display_sidebar was called with df=None, it created the uploader. Let's try getting file object AFTER sidebar render.

    # Re-access the file uploader result AFTER the sidebar has rendered
    data_file_obj = st.sidebar.file_uploader("Re-access data file", type=['csv', 'xlsx', 'xls'], key="data_file_main", label_visibility="collapsed") # Hidden re-access
    comp_file_obj = st.sidebar.file_uploader("Re-access comp file", type=['csv', 'xlsx', 'xls'], key="comp_file_main", label_visibility="collapsed") # Hidden re-access

    if data_file_obj:
        df, error_messages = load_data(data_file_obj)
    else:
        st.info("Please upload a primary engagement data file using the sidebar to begin analysis.")
        # Display expected format if no file uploaded
        st.markdown("### Expected Data Format")
        st.markdown("- `Store #` (or similar like Store ID)\n- `Week` (numeric) or `Date`\n- `Engaged Transaction %` (numeric or percentage string)\n- *Optional:* `Weekly Rank`, `Quarter to Date %`")
        st.markdown("---")
        st.stop() # Stop execution if no primary file


    # Display Loading Errors/Warnings
    if error_messages:
        for msg in error_messages:
            if "Error reading file" in msg or "No valid data rows" in msg or "Missing required column" in msg:
                st.error(msg, icon="🚨")
            else:
                st.warning(msg, icon="⚠️")
        if df is None or df.empty:
             st.error("Failed to load valid data from the primary file. Please check the file format and content.", icon="🚨")
             st.stop() # Stop if loading completely failed


    # Load Comparison Data
    df_comp = None
    if comp_file_obj:
         df_comp, comp_errors = load_data(comp_file_obj)
         if comp_errors:
              for msg in comp_errors: st.warning(f"Comparison Data: {msg}", icon="⚠️")
         if df_comp is None or df_comp.empty:
              st.warning("Could not load valid data from the comparison file.", icon="⚠️")


    # --- Apply Filters ---
    # Now that df is loaded, re-render sidebar to get actual filter values based on df content
    quarter_choice, week_choice, store_choice, z_threshold, show_ma, trend_analysis_weeks, _ = display_sidebar(df)

    # Apply filters using the function
    df_filtered = filter_dataframe(df, quarter_choice, week_choice, store_choice)
    df_comp_filtered = filter_dataframe(df_comp, quarter_choice, week_choice, store_choice) if df_comp is not None else None


    if df_filtered.empty:
        st.error("No data available for the selected filters. Please adjust filters in the sidebar.", icon="🚨")
        st.stop()

    # --- Pre-calculate Stats ---
    store_list = sorted(df[COL_STORE].unique()) if COL_STORE in df else []
    store_perf, store_trends = calculate_trend_stats(df_filtered, trend_analysis_weeks)
    store_stats = calculate_performance_categories(df_filtered) # Used in Categories and Recommendations


    # --- Render Executive Summary ---
    render_executive_summary(df_filtered, store_perf, store_trends, week_choice, store_choice, store_list)
    st.divider()


    # --- Render Tabs ---
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Engagement Trends",
        "⚖️ Store Comparison",
        "📈 Store Performance Categories",
        "💡 Anomalies & Insights"
    ])

    with tab1:
        render_engagement_trends_tab(df_filtered, df_comp_filtered, store_list, show_ma)

    with tab2:
        render_store_comparison_tab(df_filtered, week_choice, store_list)

    with tab3:
        render_performance_categories_tab(store_stats)

    with tab4:
        render_anomalies_insights_tab(df_filtered, store_stats, z_threshold, trend_analysis_weeks)


if __name__ == "__main__":
    main()
