import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from sklearn.preprocessing import StandardScaler # Note: KMeans was imported but not used
import datetime

# --- Configuration & Styling ---

PAGE_TITLE = "Publix District 20 Engagement Dashboard"
DARK_BG_COLOR = "#2C2C2C"
LIGHT_TEXT_COLOR = "#FFFFFF"

# Moved CSS here for better organization
CUSTOM_CSS = """
<style>
    .metric-card {
        background-color: #f5f5f5;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }
    .highlight-good { color: #2E7D32; font-weight: bold; }
    .highlight-bad { color: #C62828; font-weight: bold; }
    .highlight-neutral { color: #F57C00; font-weight: bold; }
    .dashboard-title { color: #1565C0; text-align: center; padding-bottom: 20px; }
    .caption-text { font-size: 0.85em; color: #555; }
    .category-card-base {
        background-color: """ + DARK_BG_COLOR + """;
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 10px;
        color: """ + LIGHT_TEXT_COLOR + """;
    }
    .category-card-accent {
        border-left: 5px solid; /* Color set dynamically */
    }
    .trend-card-base {
        background-color: """ + DARK_BG_COLOR + """;
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 10px;
        border-left: 5px solid; /* Color set dynamically */
        color: """ + LIGHT_TEXT_COLOR + """;
    }
    .trend-card-title { text-align: center; margin: 5px 0; }
    .trend-card-text { text-align: center; margin: 5px 0; }
    .trend-card-subtext { font-size: 0.85em; color: #BBBBBB; }
</style>
"""

CATEGORY_COLORS = {
    "Star Performer": "#2E7D32", # Green
    "Needs Stabilization": "#F57C00", # Orange
    "Improving": "#1976D2", # Blue
    "Requires Intervention": "#C62828" # Red
}

# --------------------------------------------------------
# Helper Functions: Data Loading & Processing
# --------------------------------------------------------

def standardize_columns_integrated(columns):
    """Standardizes column names directly."""
    new_cols = []
    for col in columns:
        cl = col.strip().lower()
        if 'quarter' in cl or 'qtd' in cl:
            new_cols.append('Quarter to Date %')
        elif 'rank' in cl:
            new_cols.append('Weekly Rank')
        elif ('week' in cl and 'ending' in cl) or cl == 'date' or cl == 'week ending':
            new_cols.append('Date') # Prefer 'Date' if available
        elif cl.startswith('week') and 'rank' not in cl: # Avoid matching 'Weekly Rank'
             new_cols.append('Week')
        elif 'store' in cl:
            new_cols.append('Store #')
        elif 'engaged' in cl or 'engagement' in cl:
            new_cols.append('Engaged Transaction %')
        else:
            new_cols.append(col.strip()) # Keep original but stripped
    # Ensure essential columns exist if possible alternates were found
    if 'Date' not in new_cols and 'Week' not in new_cols:
        # Add placeholder or raise error depending on strictness needed
        pass # Or: raise ValueError("Missing required 'Week' or 'Date' column")
    if 'Store #' not in new_cols:
        pass # Or: raise ValueError("Missing required 'Store #' column")
    if 'Engaged Transaction %' not in new_cols:
        pass # Or: raise ValueError("Missing required 'Engaged Transaction %' column")
    return new_cols

@st.cache_data
def load_data(uploaded_file):
    """
    Reads, standardizes, cleans, and adds Quarter to the uploaded data.
    """
    if uploaded_file is None:
        return None

    filename = uploaded_file.name.lower()
    try:
        if filename.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif filename.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(uploaded_file)
        else:
            st.error(f"Unsupported file type: {filename}. Please use CSV or Excel.")
            return None
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return None

    # --- Standardization ---
    df.columns = standardize_columns_integrated(df.columns)

    # --- Cleaning & Type Conversion ---
    required_cols = ['Store #', 'Engaged Transaction %']
    if not all(col in df.columns for col in required_cols):
        st.error(f"Missing required columns. Need: {', '.join(required_cols)}. Found: {', '.join(df.columns)}")
        return None
    if 'Week' not in df.columns and 'Date' not in df.columns:
         st.error("Data must contain either a 'Week' or 'Date' column.")
         return None

    # Handle Dates and Weeks
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df.dropna(subset=['Date'], inplace=True)
        # Ensure 'Week' column exists if Date is primary
        if 'Week' not in df.columns:
             # Attempt to calculate ISO week or handle as needed
             # Example: df['Week'] = df['Date'].dt.isocalendar().week
             # For simplicity here, we'll assume 'Week' is the primary identifier if 'Date' is missing later
             pass # Add week calculation logic if strictly needed based on Date
    elif 'Week' in df.columns:
         df['Week'] = pd.to_numeric(df['Week'], errors='coerce').astype('Int64') # Allow NA temporarily
         df.dropna(subset=['Week'], inplace=True)
         df['Week'] = df['Week'].astype(int) # Convert back to int after NA drop

    # Convert percentages
    percent_cols = ['Engaged Transaction %', 'Quarter to Date %']
    for col in percent_cols:
        if col in df.columns:
            # Robust conversion: handle strings, percentages, and numbers
            df[col] = df[col].astype(str).str.replace('%', '', regex=False).str.strip()
            df[col] = pd.to_numeric(df[col], errors='coerce')
            # Optional: Divide by 100 if percentages are > 1 (e.g., 85 instead of 0.85)
            # if df[col].max() > 1.5: # Heuristic threshold
            #     df[col] = df[col] / 100.0

    df = df.dropna(subset=['Engaged Transaction %']) # Crucial KPI

    # Other types
    if 'Weekly Rank' in df.columns:
        df['Weekly Rank'] = pd.to_numeric(df['Weekly Rank'], errors='coerce').astype('Int64')
    df['Store #'] = df['Store #'].astype(str)

    # --- Add Quarter ---
    if 'Date' in df.columns:
        df['Quarter'] = df['Date'].dt.quarter
    elif 'Week' in df.columns:
        # Standard quarter calculation (adjust if fiscal year differs)
        df['Quarter'] = ((df['Week'] - 1) // 13 + 1).astype(int)
    else:
        df['Quarter'] = None # Should not happen based on earlier checks

    # --- Sort ---
    sort_cols = ['Week', 'Store #'] if 'Week' in df.columns else ['Date', 'Store #']
    df = df.sort_values(sort_cols)

    return df

def apply_filters(dataframe, quarter_choice, week_choice, store_choice):
    """Applies selected filters to a dataframe."""
    if dataframe is None:
        return pd.DataFrame() # Return empty DF if input is None
    df_out = dataframe.copy()
    if quarter_choice != "All":
        q_num = int(quarter_choice[1:])
        df_out = df_out[df_out['Quarter'] == q_num]
    if week_choice != "All":
        week_num = int(week_choice)
        df_out = df_out[df_out['Week'] == week_num] # Assumes 'Week' column exists
    if store_choice: # If list is not empty
        df_out = df_out[df_out['Store #'].isin([str(s) for s in store_choice])]
    return df_out

def calculate_trend(group, window=4):
    """Calculates trend label based on recent performance slope."""
    if len(group) < 2:
        return "Insufficient Data"
    # Ensure sorting by time (Week or Date)
    time_col = 'Week' if 'Week' in group.columns else 'Date'
    sorted_data = group.sort_values(time_col, ascending=True).tail(window)
    if len(sorted_data) < 2:
        return "Insufficient Data"

    # Use numeric representation of time for regression
    if time_col == 'Date':
         # Convert date to ordinal number for linear regression
         x = sorted_data[time_col].apply(lambda date: date.toordinal()).values
    else: # Week
         x = sorted_data[time_col].values

    y = sorted_data['Engaged Transaction %'].values

    # Basic linear regression slope
    # Center X to improve numerical stability, especially with large date ordinals
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    x_centered = x - x_mean

    numerator = np.sum(x_centered * (y - y_mean))
    denominator = np.sum(x_centered**2)

    if denominator == 0:
        return "Stable" # No change in time variable within the window

    slope = numerator / denominator

    # Adjust thresholds as needed based on typical % scales
    if slope > 0.5: return "Strong Upward"
    elif slope > 0.1: return "Upward"
    elif slope < -0.5: return "Strong Downward"
    elif slope < -0.1: return "Downward"
    else: return "Stable"

def find_anomalies(df, z_threshold=2.0):
    """Identifies significant week-over-week changes."""
    if df is None or df.empty or 'Week' not in df.columns:
        return pd.DataFrame()

    anomalies_list = []
    df_sorted = df.sort_values(['Store #', 'Week'])
    # Calculate differences *within* each store group
    df_sorted['Change %pts'] = df_sorted.groupby('Store #')['Engaged Transaction %'].diff()

    # Calculate Z-score for the changes within each store
    # Using transform allows broadcasting the group mean/std back to the original shape
    mean_diff = df_sorted.groupby('Store #')['Change %pts'].transform('mean')
    std_diff = df_sorted.groupby('Store #')['Change %pts'].transform('std', ddof=0)

    # Avoid division by zero or NaN standard deviation
    # Replace std_diff of 0 or NaN with NaN to prevent Z-score calculation errors
    std_diff_safe = std_diff.replace(0, np.nan)

    df_sorted['Z-score'] = (df_sorted['Change %pts'] - mean_diff) / std_diff_safe

    # Filter for anomalies
    anomalies_df = df_sorted[df_sorted['Z-score'].abs() >= z_threshold].copy()

    if anomalies_df.empty:
        return pd.DataFrame()

    # Prepare output dataframe - get previous week/rank info efficiently
    anomalies_df['Prev Week'] = anomalies_df['Week'] - 1 # Simple assumption
    anomalies_df = anomalies_df.rename(columns={'Weekly Rank': 'Rank'}) # Rename for clarity

    # Get previous rank (requires merging or careful lookup)
    # Shift rank within each group to get previous week's rank
    if 'Rank' in anomalies_df.columns:
        anomalies_df['Prev Rank'] = anomalies_df.groupby('Store #')['Rank'].shift(1)
    else:
         anomalies_df['Rank'] = None
         anomalies_df['Prev Rank'] = None


    # Add Explanations
    def get_explanation(row):
        reason = ""
        if row['Change %pts'] >= 0:
            reason = "Engagement spiked significantly. Possible promotion or event impact."
            if pd.notna(row['Prev Rank']) and pd.notna(row['Rank']) and row['Prev Rank'] > row['Rank']:
                reason += f" (Improved from rank {int(row['Prev Rank'])} to {int(row['Rank'])}.)"
        else:
            reason = "Sharp drop in engagement. Potential system issue or loss of engagement."
            if pd.notna(row['Prev Rank']) and pd.notna(row['Rank']) and row['Prev Rank'] < row['Rank']:
                reason += f" (Dropped from rank {int(row['Prev Rank'])} to {int(row['Rank'])}.)"
        return reason

    anomalies_df['Possible Explanation'] = anomalies_df.apply(get_explanation, axis=1)

    # Select and format columns
    output_cols = [
        'Store #', 'Week', 'Engaged Transaction %', 'Change %pts',
        'Z-score', 'Prev Week', 'Rank', 'Prev Rank', 'Possible Explanation'
    ]
    # Ensure all columns exist before selecting
    final_output_cols = [col for col in output_cols if col in anomalies_df.columns]
    anomalies_final = anomalies_df[final_output_cols].copy()


    # Round numeric columns
    for col in ['Engaged Transaction %', 'Change %pts', 'Z-score']:
        if col in anomalies_final:
            anomalies_final[col] = anomalies_final[col].round(2)

    # Convert ranks/weeks back to Int64 (nullable integer)
    for col in ['Week', 'Prev Week', 'Rank', 'Prev Rank']:
         if col in anomalies_final:
              anomalies_final[col] = pd.to_numeric(anomalies_final[col], errors='coerce').astype('Int64')


    # Sort by absolute Z-score magnitude
    anomalies_final['Abs Z'] = anomalies_final['Z-score'].abs()
    anomalies_final = anomalies_final.sort_values('Abs Z', ascending=False).drop(columns=['Abs Z'])

    return anomalies_final

def calculate_store_stats(df_filtered, trend_analysis_weeks):
    """Calculates average engagement, consistency, trend, and category for each store."""
    if df_filtered is None or df_filtered.empty:
        return pd.DataFrame(), None

    store_stats = df_filtered.groupby('Store #')['Engaged Transaction %'].agg(
        AverageEngagement='mean',
        StdDev='std'
    ).reset_index()
    store_stats['StdDev'] = store_stats['StdDev'].fillna(0.0) # Consistency (lower is better)

    # Calculate trend for each store (using correlation as proxy)
    trends = df_filtered.groupby('Store #').apply(
        lambda x: calculate_trend(x, trend_analysis_weeks)
    ).reset_index(name='Trend')

    store_stats = store_stats.merge(trends, on='Store #', how='left')
    store_stats['Trend'] = store_stats['Trend'].fillna("Stable")

    # --- Assign Performance Category ---
    if store_stats.empty:
         return store_stats, None

    median_engagement = store_stats['AverageEngagement'].median()
    if pd.isna(median_engagement): # Handle case with only one store or all NaN
         median_engagement = 0

    def assign_category(row):
        is_above_median = row['AverageEngagement'] >= median_engagement
        trend = row['Trend']

        if is_above_median:
            if trend in ["Downward", "Strong Downward"]:
                return "Needs Stabilization"
            else: # Stable, Upward, Strong Upward
                return "Star Performer"
        else: # Below median
            if trend in ["Upward", "Strong Upward"]:
                return "Improving"
            else: # Stable, Downward, Strong Downward
                return "Requires Intervention"

    store_stats['Category'] = store_stats.apply(assign_category, axis=1)

    # --- Add Explanations and Action Plans ---
    action_plans = {
        "Star Performer": "Maintain current strategies. Document and share best practices.",
        "Needs Stabilization": "Investigate recent changes. Reinforce successful processes.",
        "Improving": "Continue positive momentum. Identify what's working.",
        "Requires Intervention": "Comprehensive review needed. Create action plan with district support."
    }
    cat_explanations = {
        "Star Performer": "High engagement, stable/improving trend",
        "Needs Stabilization": "High engagement, but downward trend",
        "Improving": "Below average, but positive trend",
        "Requires Intervention": "Below average, flat/declining trend"
    }
    store_stats['Action Plan'] = store_stats['Category'].map(action_plans)
    store_stats['Explanation'] = store_stats['Category'].map(cat_explanations)

    return store_stats, median_engagement


# --------------------------------------------------------
# Helper Functions: UI Components
# --------------------------------------------------------

def display_metric_card(column, label, value, delta=None, delta_color="normal"):
    """Displays a Streamlit metric in a styled column."""
    if value is None or pd.isna(value):
        value_display = "N/A"
        delta_display = None
    else:
        value_display = f"{value:.2f}%"
        if delta is not None and pd.notna(delta):
             # Display delta as percentage points change
             delta_display = f"{delta:+.2f} %pts"
        else:
             delta_display = None

    column.metric(label, value_display, delta_display, delta_color=delta_color)

def display_category_info_card(column, category_name, icon, description, action, accent_color):
     """Displays the explanation card for a performance category."""
     column.markdown(f"""
     <div class="category-card-base category-card-accent" style="border-left-color: {accent_color};">
         <h4 style="color: {accent_color}; margin-top: 0;">{icon} {category_name}</h4>
         <p style="margin: 0;">{description}</p>
         <p style="margin: 0;"><strong>Action:</strong> {action}</p>
     </div>
     """, unsafe_allow_html=True)

def display_store_category_card(column, store_id, avg_engagement, trend, category, accent_color):
    """Displays a small card for a store within its category."""
    if trend in ["Upward", "Strong Upward"]: trend_icon = "🔼"
    elif trend in ["Downward", "Strong Downward"]: trend_icon = "🔽"
    else: trend_icon = "➡️"

    column.markdown(f"""
    <div class="category-card-base" style="padding: 10px;">
        <h4 style="text-align: center; margin: 5px 0; color: {accent_color};">Store {store_id}</h4>
        <p style="text-align: center; margin: 5px 0;">
            Avg: {avg_engagement:.2f}%<br>
            Trend: {trend_icon} {trend}
        </p>
    </div>
    """, unsafe_allow_html=True)

def display_trend_card(column, store_info):
     """Displays a card for the Recent Performance Trends section."""
     change_display = f"{store_info['total_change']:+.2f}%"

     column.markdown(f"""
     <div class="trend-card-base" style="border-left-color: {store_info['color']};">
         <h4 class="trend-card-title" style="color: {store_info['color']};">
             {store_info['indicator']} Store {store_info['store']}
         </h4>
         <p class="trend-card-text">
             <strong>{store_info['strength']}</strong><br>
             <span style="font-size: 0.9em;">
                 <strong>{change_display}</strong> over {store_info['weeks']} weeks
             </span><br>
             <span class="trend-card-subtext">
                 {store_info['start_value']:.2f}% → {store_info['current_value']:.2f}%
             </span>
         </p>
     </div>
     """, unsafe_allow_html=True)


# --------------------------------------------------------
# Helper Functions: Altair Charts
# --------------------------------------------------------

def create_base_chart(data, x_col='Week:O', x_title='Week'):
    """Creates a base Altair chart with X encoding."""
    return alt.Chart(data).encode(
        x=alt.X(x_col, title=x_title, axis=alt.Axis(labelAngle=-45)) # Angled labels
    )

def create_line_layer(base_chart, y_col, y_title, color_col, color_title, tooltip_cols,
                      stroke_width=1.5, add_interactive_legend=False):
    """Creates a line layer for an Altair chart."""
    line = base_chart.mark_line(strokeWidth=stroke_width, point=False).encode(
        y=alt.Y(y_col, title=y_title),
        color=alt.Color(color_col, title=color_title, scale=alt.Scale(scheme='category10')),
        tooltip=tooltip_cols
    )
    if add_interactive_legend:
        selection = alt.selection_point(fields=[color_col], bind='legend')
        line = line.add_params(selection).encode(
            opacity=alt.condition(selection, alt.value(1), alt.value(0.2)),
            strokeWidth=alt.condition(selection, alt.value(stroke_width + 1), alt.value(stroke_width))
        )
    return line

def create_point_layer(base_chart, y_col, color_col, tooltip_cols, size=80):
     """Creates a point layer for highlighting specific data points."""
     points = base_chart.mark_point(filled=True, size=size).encode(
         y=alt.Y(y_col), # Title inherited usually
         color=alt.Color(color_col, scale=alt.Scale(scheme='category10')), # Legend inherited
         tooltip=tooltip_cols
     )
     return points


def create_ma_layer(base_chart, y_col, y_title, color_col, color_title, tooltip_cols,
                    stroke_dash=[2,2], stroke_width=1.5, opacity=0.7, selection=None):
    """Creates a moving average line layer."""
    ma_line = base_chart.mark_line(strokeDash=stroke_dash, strokeWidth=stroke_width, opacity=opacity).encode(
        y=alt.Y(y_col, title=y_title),
        color=alt.Color(color_col, title=color_title, scale=alt.Scale(scheme='category10')),
        tooltip=tooltip_cols
    )
    if selection: # Apply opacity linked to main line selection
        ma_line = ma_line.encode(
             opacity=alt.condition(selection, alt.value(opacity), alt.value(0.1))
        )
    return ma_line

def create_bar_chart(data, x_col, x_title, y_col, y_title, color_col=None, color_scale=None,
                     sort_y='-x', tooltip_cols=None, title="", dynamic_height=True):
    """Creates a standard bar chart."""
    if tooltip_cols is None:
        tooltip_cols = [y_col, x_col]

    bar = alt.Chart(data, title=title).mark_bar().encode(
        x=alt.X(x_col, title=x_title),
        y=alt.Y(y_col, title=y_title, sort=sort_y),
        tooltip=tooltip_cols
    )
    if color_col:
        bar = bar.encode(color=alt.Color(color_col, scale=color_scale, legend=None)) # Legend often redundant

    if dynamic_height:
         height = max(200, 25 * len(data)) # Adjust multiplier as needed
         bar = bar.properties(height=height)

    return bar

def create_rule_layer(value, color='red', stroke_dash=[4, 4], size=2, tooltip_text=None):
    """Creates a horizontal or vertical rule (line) on a chart."""
    # Create a dummy dataframe for the rule
    rule_data = pd.DataFrame({'value': [value]})
    rule = alt.Chart(rule_data).mark_rule(
        color=color, strokeDash=stroke_dash, size=size
    ).encode(
        x='value:Q' # Assume rule is on X-axis by default
        # Use y='value:Q' for horizontal rule
    )
    if tooltip_text:
        rule = rule.encode(tooltip=[alt.Tooltip('value:Q', title=tooltip_text, format='.2f')])
    return rule


# --------------------------------------------------------
# Streamlit Page Config & Main Layout
# --------------------------------------------------------

st.set_page_config(
    page_title=PAGE_TITLE,
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

st.markdown(f"<h1 class='dashboard-title'>{PAGE_TITLE}</h1>", unsafe_allow_html=True)
st.markdown("Upload weekly engagement data to explore KPIs, trends, and opportunities. Use filters on the left.")

# --------------------------------------------------------
# Sidebar for Data Upload & Filters
# --------------------------------------------------------

st.sidebar.header("Data Input")
data_file = st.sidebar.file_uploader("Upload primary engagement data (Excel or CSV)", type=['csv', 'xlsx'])
comp_file = st.sidebar.file_uploader("Optional: Upload comparison data (prior period)", type=['csv', 'xlsx'])

# Load data
df = load_data(data_file)
df_comp = load_data(comp_file) # Will be None if comp_file is None

if df is None:
    st.info("Please upload a primary engagement data file to begin.")
    st.markdown("### Expected Data Format")
    st.markdown("""
    - `Store #` or `Store ID`
    - `Week` (numeric) or `Date`
    - `Engaged Transaction %` (as number or % string)
    - Optional: `Weekly Rank`, `Quarter to Date %`, etc.
    """)
    st.stop()

# --- Sidebar Filters ---
st.sidebar.header("Filters")

# Get available options from the main dataframe
available_quarters = sorted(df['Quarter'].dropna().unique().tolist())
quarter_options = ["All"] + [f"Q{int(q)}" for q in available_quarters]
quarter_choice = st.sidebar.selectbox("Select Quarter", quarter_options, index=0)

# Dynamically update week options based on quarter selection
if quarter_choice != "All":
    q_num_filter = int(quarter_choice[1:])
    available_weeks = sorted(df[df['Quarter'] == q_num_filter]['Week'].unique().tolist())
else:
    available_weeks = sorted(df['Week'].unique().tolist())

week_options = ["All"] + [str(int(w)) for w in available_weeks]
week_choice = st.sidebar.selectbox("Select Week", week_options, index=0 if "All" in week_options else 0) # Handle empty lists

available_stores = sorted(df['Store #'].unique().tolist())
store_choice = st.sidebar.multiselect("Select Store(s)", available_stores, default=[])

# --- Advanced Settings ---
with st.sidebar.expander("Advanced Settings", expanded=False):
    z_threshold = st.slider("Anomaly Z-score Threshold", 1.0, 3.0, 2.0, 0.1, help="Sensitivity for anomaly detection (higher = fewer anomalies)")
    show_ma_global = st.checkbox("Show 4-week moving average on charts", value=True)
    trend_analysis_weeks = st.slider("Trend analysis window (weeks)", 3, 8, 4, help="Number of recent weeks to consider for trend calculation")
    # Removed highlight options as they are often better handled dynamically in charts/tables

# --- Apply Filters ---
df_filtered = apply_filters(df, quarter_choice, week_choice, store_choice)
df_comp_filtered = apply_filters(df_comp, quarter_choice, week_choice, store_choice)

if df_filtered.empty:
    st.warning("No data available for the selected filters. Please adjust filters or check data.")
    st.stop()

# --------------------------------------------------------
# Core Calculations (after filtering)
# --------------------------------------------------------

# --- Determine Current/Previous Weeks ---
if 'Week' in df_filtered.columns:
    all_weeks_in_filter = sorted(df_filtered['Week'].unique())
    if week_choice != "All":
        current_week = int(week_choice)
        # Find previous week *within the filtered dataset*
        prev_week_options = [w for w in all_weeks_in_filter if w < current_week]
        prev_week = max(prev_week_options) if prev_week_options else None
    else: # All weeks selected
        current_week = max(all_weeks_in_filter) if all_weeks_in_filter else None
        prev_week_options = [w for w in all_weeks_in_filter if w < current_week] if current_week else []
        prev_week = max(prev_week_options) if prev_week_options else None
else: # Date based - logic would need adjustment if week filters are primary
     current_week = None # Placeholder if only Date is present
     prev_week = None

# --- Calculate Summary Stats ---
current_avg = None
prev_avg = None
delta_avg = None
if current_week is not None:
    current_data = df_filtered[df_filtered['Week'] == current_week]
    if not current_data.empty:
        current_avg = current_data['Engaged Transaction %'].mean()

    if prev_week is not None:
        prev_data = df_filtered[df_filtered['Week'] == prev_week]
        if not prev_data.empty:
            prev_avg = prev_data['Engaged Transaction %'].mean()

    if current_avg is not None and prev_avg is not None:
        delta_avg = current_avg - prev_avg

# --- Top/Bottom Performers (based on filtered period average) ---
store_perf = df_filtered.groupby('Store #')['Engaged Transaction %'].mean()
top_store = store_perf.idxmax() if not store_perf.empty else "N/A"
bottom_store = store_perf.idxmin() if not store_perf.empty else "N/A"
top_val = store_perf.max() if not store_perf.empty else None
bottom_val = store_perf.min() if not store_perf.empty else None

# --- Store Categories & Trends ---
# Calculated once here for use in multiple tabs
store_stats_df, district_median_engagement = calculate_store_stats(df_filtered, trend_analysis_weeks)

# --- Anomalies ---
anomalies_df = find_anomalies(df_filtered, z_threshold)

# --------------------------------------------------------
# Executive Summary Display
# --------------------------------------------------------

st.subheader("Executive Summary")
col1, col2, col3 = st.columns(3)

# Label for average engagement metric
if store_choice and len(store_choice) == 1:
    avg_label = f"Store {store_choice[0]} Avg"
elif store_choice and len(store_choice) < len(available_stores):
    avg_label = "Selected Stores Avg"
else:
    avg_label = "District Avg"
avg_label += f" (Wk {current_week})" if current_week is not None else ""

display_metric_card(col1, avg_label, current_avg, delta_avg)
display_metric_card(col2, f"Top Performer (Avg)", top_val, delta=None, label_suffix=f"Store {top_store}") # Custom display needed
col2.metric(f"Top Performer (Avg)", f"Store {top_store} ({top_val:.2f}%)" if top_val is not None else "N/A")
col3.metric(f"Bottom Performer (Avg)", f"Store {bottom_store} ({bottom_val:.2f}%)" if bottom_val is not None else "N/A")


# Trend indicator text
if delta_avg is not None:
    delta_abs = abs(delta_avg)
    trend_word = "up" if delta_avg > 0 else "down" if delta_avg < 0 else "flat"
    trend_class = "highlight-good" if delta_avg > 0 else "highlight-bad" if delta_avg < 0 else "highlight-neutral"
    st.markdown(
        f"Week {current_week} average engagement is <span class='{trend_class}'>{delta_abs:.2f} %pts {trend_word}</span> from Week {prev_week}.",
        unsafe_allow_html=True
    )
elif current_avg is not None:
     st.markdown(f"Average engagement for Week {current_week}: <span class='highlight-neutral'>{current_avg:.2f}%</span>", unsafe_allow_html=True)


# Top & Bottom store trends (using calculated trends)
col1a, col2a = st.columns(2)
if not store_stats_df.empty:
    top_store_info = store_stats_df[store_stats_df['Store #'] == top_store].iloc[0] if top_store != "N/A" else None
    bottom_store_info = store_stats_df[store_stats_df['Store #'] == bottom_store].iloc[0] if bottom_store != "N/A" else None

    if top_store_info is not None:
        t_trend = top_store_info['Trend']
        t_color = "highlight-good" if "Upward" in t_trend else "highlight-bad" if "Downward" in t_trend else "highlight-neutral"
        col1a.markdown(f"**Store {top_store}** trend: <span class='{t_color}'>{t_trend}</span>", unsafe_allow_html=True)

    if bottom_store_info is not None:
        b_trend = bottom_store_info['Trend']
        b_color = "highlight-good" if "Upward" in b_trend else "highlight-bad" if "Downward" in b_trend else "highlight-neutral"
        col2a.markdown(f"**Store {bottom_store}** trend: <span class='{b_color}'>{b_trend}</span>", unsafe_allow_html=True)


# --------------------------------------------------------
# Key Insights (Derived from calculations)
# --------------------------------------------------------
st.subheader("Key Insights")
insights = []
if not store_stats_df.empty:
    # Consistency (using StdDev calculated earlier)
    store_std = store_stats_df.set_index('Store #')['StdDev']
    most_consistent = store_std.idxmin()
    least_consistent = store_std.idxmax()
    insights.append(f"**Store {most_consistent}** shows the most consistent engagement (std dev: {store_std.min():.2f}).")
    insights.append(f"**Store {least_consistent}** has the most variable engagement (std dev: {store_std.max():.2f}).")

    # Trend analysis (using Trend calculated earlier)
    trending_up = store_stats_df[store_stats_df['Trend'].str.contains("Upward", na=False)]['Store #'].tolist()
    trending_down = store_stats_df[store_stats_df['Trend'].str.contains("Downward", na=False)]['Store #'].tolist()
    if trending_up: insights.append(f"Stores showing positive trends: {', '.join([f'**{s}**' for s in trending_up])}")
    if trending_down: insights.append(f"Stores needing attention (downward trend): {', '.join([f'**{s}**' for s in trending_down])}")

# Gap analysis
if top_val is not None and bottom_val is not None and len(store_perf) > 1:
    engagement_gap = top_val - bottom_val
    insights.append(f"Gap between highest and lowest performing stores: **{engagement_gap:.2f}%**")
    if engagement_gap > 10: # Example threshold
        insights.append("🚨 Large performance gap indicates opportunity for knowledge sharing.")

# Display Insights
if not insights:
     st.info("No specific insights generated based on current filters.")
else:
     for i, insight in enumerate(insights[:5], start=1): # Limit insights displayed
         st.markdown(f"{i}. {insight}")

# --------------------------------------------------------
# Main Tabs
# --------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Engagement Trends",
    "📈 Store Comparison",
    "📋 Store Categories",
    "💡 Anomalies & Insights"
])

# ----------------- TAB 1: Engagement Trends -----------------
with tab1:
    st.subheader("Engagement Trends Over Time")

    view_option = st.radio(
        "View mode:",
        ["All Stores", "Custom Selection", "Recent Trends"],
        horizontal=True,
        key="tab1_view_mode",
        help="All Stores: View all | Custom: Pick specific stores | Recent: Focus on recent weeks"
    )

    # --- Prepare data for charts ---
    trend_data = df_filtered.copy()

    # Calculate Moving Averages if needed
    if show_ma_global:
        trend_data = trend_data.sort_values(['Store #', 'Week'])
        trend_data['MA_4W'] = trend_data.groupby('Store #')['Engaged Transaction %']\
            .transform(lambda s: s.rolling(window=4, min_periods=1).mean())

    # District average trend
    dist_trend = trend_data.groupby('Week', as_index=False)['Engaged Transaction %'].mean().sort_values('Week')
    dist_trend.rename(columns={'Engaged Transaction %': 'Average Engagement %'}, inplace=True)
    if show_ma_global:
        dist_trend['MA_4W'] = dist_trend['Average Engagement %'].rolling(window=4, min_periods=1).mean()

    # Comparison period data
    dist_trend_comp = None
    if df_comp_filtered is not None and not df_comp_filtered.empty:
        dist_trend_comp = df_comp_filtered.groupby('Week', as_index=False)['Engaged Transaction %'].mean().sort_values('Week')
        dist_trend_comp.rename(columns={'Engaged Transaction %': 'Average Engagement %'}, inplace=True)
        if show_ma_global:
             dist_trend_comp['MA_4W'] = dist_trend_comp['Average Engagement %'].rolling(window=4, min_periods=1).mean()


    # Filter for Recent Trends view
    recent_weeks = None
    if view_option == "Recent Trends":
        all_weeks_trend = sorted(trend_data['Week'].unique())
        if len(all_weeks_trend) > 1:
            default_start_trend = all_weeks_trend[0] if len(all_weeks_trend) <= 8 else all_weeks_trend[-8]
            default_end_trend = all_weeks_trend[-1]
            recent_weeks_range = st.select_slider(
                "Select weeks to display:",
                options=all_weeks_trend,
                value=(default_start_trend, default_end_trend),
                key="tab1_recent_slider"
            )
            recent_weeks = list(range(recent_weeks_range[0], recent_weeks_range[1] + 1)) # Include end week
            trend_data = trend_data[trend_data['Week'].isin(recent_weeks)]
            dist_trend = dist_trend[dist_trend['Week'].isin(recent_weeks)]
            if dist_trend_comp is not None:
                dist_trend_comp = dist_trend_comp[dist_trend_comp['Week'].isin(recent_weeks)]
        else:
            st.info("Not enough historical data for Recent Trends view.")
            st.stop() # Stop rendering this tab if not enough data


    # --- Build Chart Layers ---
    layers = []
    interactive_selection = None # For linking MA opacity

    base_chart = create_base_chart(trend_data) # Pass data here
    tooltip_trend = ['Store #', 'Week', alt.Tooltip('Engaged Transaction %', format='.2f')]
    tooltip_ma = ['Store #', 'Week', alt.Tooltip('MA_4W', format='.2f', title='4W MA')]

    if view_option == "Custom Selection":
        selected_stores = st.multiselect(
             "Select stores to compare:",
             options=available_stores,
             default=available_stores[:1] if available_stores else [], # Default to first store if exists
             key="tab1_store_select"
        )
        if selected_stores:
            selected_data = trend_data[trend_data['Store #'].isin(selected_stores)]
            if not selected_data.empty:
                base_chart_selected = create_base_chart(selected_data) # Use filtered data for this chart base
                layers.append(create_line_layer(base_chart_selected, 'Engaged Transaction %', 'Engagement %', 'Store #', 'Store', tooltip_trend, stroke_width=3))
                layers.append(create_point_layer(base_chart_selected, 'Engaged Transaction %', 'Store #', tooltip_trend))
                if show_ma_global and 'MA_4W' in selected_data.columns:
                    layers.append(create_ma_layer(base_chart_selected, 'MA_4W', '4W MA', 'Store #', 'Store', tooltip_ma))
            else:
                st.info("No data for selected stores in the filtered period.")
        else:
            st.info("Select at least one store to display.")

    else: # All Stores or Recent Trends
        if not trend_data.empty:
             interactive_selection = alt.selection_point(fields=['Store #'], bind='legend') # Define selection for opacity link
             line_layer = create_line_layer(base_chart, 'Engaged Transaction %', 'Engagement %', 'Store #', 'Store', tooltip_trend, add_interactive_legend=False) # Add selection manually below
             line_layer = line_layer.add_params(interactive_selection).encode(
                  opacity=alt.condition(interactive_selection, alt.value(1), alt.value(0.2)),
                  strokeWidth=alt.condition(interactive_selection, alt.value(2.5), alt.value(1.5))
             )
             layers.append(line_layer)

             if show_ma_global and 'MA_4W' in trend_data.columns:
                  layers.append(create_ma_layer(base_chart, 'MA_4W', '4W MA', 'Store #', 'Store', tooltip_ma, selection=interactive_selection))
        else:
             st.info("No trend data to display for this selection.")


    # Add District Average Line(s)
    if not dist_trend.empty:
        base_dist_chart = create_base_chart(dist_trend)
        layers.append(create_line_layer(base_dist_chart, 'Average Engagement %', 'Engagement %', alt.value('black'), 'District Avg', [alt.Tooltip('Average Engagement %', format='.2f')], stroke_width=3, stroke_dash=[4,2]))
        if show_ma_global and 'MA_4W' in dist_trend.columns:
            layers.append(create_ma_layer(base_dist_chart, 'MA_4W', '4W MA', alt.value('black'), 'District 4W MA', [alt.Tooltip('MA_4W', format='.2f')], stroke_dash=[1,1], stroke_width=2, opacity=0.7))

    # Add Comparison Period Line(s)
    if dist_trend_comp is not None and not dist_trend_comp.empty:
        base_dist_comp_chart = create_base_chart(dist_trend_comp)
        layers.append(create_line_layer(base_dist_comp_chart, 'Average Engagement %', 'Engagement %', alt.value('#555555'), 'Prior Period Avg', [alt.Tooltip('Average Engagement %', format='.2f')], stroke_width=2, stroke_dash=[4,2]))
        if show_ma_global and 'MA_4W' in dist_trend_comp.columns:
             layers.append(create_ma_layer(base_dist_comp_chart, 'MA_4W', '4W MA', alt.value('#555555'), 'Prior 4W MA', [alt.Tooltip('MA_4W', format='.2f')], stroke_dash=[1,1], stroke_width=1.5, opacity=0.7))


    # --- Display Chart & Caption ---
    if layers:
        # Specific metrics for Recent Trends view
        if view_option == "Recent Trends" and not dist_trend.empty and len(dist_trend) >= 2:
             col1t, col2t = st.columns(2)
             # District Trend W-o-W
             last_two_weeks = dist_trend.tail(2)
             current_recent_avg = last_two_weeks['Average Engagement %'].iloc[-1]
             prev_recent_avg = last_two_weeks['Average Engagement %'].iloc[0]
             recent_delta = current_recent_avg - prev_recent_avg
             display_metric_card(col1t, "District Trend (WoW)", current_recent_avg, recent_delta)

             # Top performer last week
             last_week_num = trend_data['Week'].max()
             last_week_data = trend_data[trend_data['Week'] == last_week_num]
             if not last_week_data.empty:
                  best_last_week = last_week_data.loc[last_week_data['Engaged Transaction %'].idxmax()]
                  col2t.metric(f"Top Performer (Wk {last_week_num})", f"Store {best_last_week['Store #']}", f"{best_last_week['Engaged Transaction %']:.2f}%", delta_color="off")


        final_chart = alt.layer(*layers).resolve_scale(y='shared').properties(height=400).interactive()
        st.altair_chart(final_chart, use_container_width=True)

        # Caption
        captions = {
            "All Stores": "Shows all store trends. Black dashed = district avg.",
            "Custom Selection": "Shows selected stores. Black dashed = district avg.",
            "Recent Trends": "Focuses on recent weeks. Black dashed = district avg."
        }
        caption = captions.get(view_option, "")
        if dist_trend_comp is not None and not dist_trend_comp.empty:
            caption += " Gray dashed = prior period district avg."
        st.caption(caption)
    else:
         st.info("No chart layers generated. Adjust filters or view options.")


    # ----------------- Weekly Engagement Heatmap -----------------
    st.subheader("Weekly Engagement Heatmap")

    heatmap_df = df_filtered[['Week', 'Store #', 'Engaged Transaction %']].copy()
    heatmap_df = heatmap_df.rename(columns={'Store #': 'StoreID', 'Engaged Transaction %': 'EngagedPct'})

    if heatmap_df.empty or heatmap_df['EngagedPct'].isnull().all():
        st.info("No data available for heatmap with current filters.")
    else:
        with st.expander("Heatmap Settings", expanded=False):
            col1h, col2h = st.columns(2)
            sort_method = col1h.selectbox("Sort stores by:", ["Average Engagement", "Recent Performance"], index=0, key="heatmap_sort")
            color_scheme = col2h.selectbox("Color scheme:", ["blues", "greens", "purples", "oranges", "reds", "viridis"], index=0, key="heatmap_color")
            # normalize_colors = col2h.checkbox("Normalize colors by week", value=False, key="heatmap_norm") # Normalization logic complex, removed for brevity in refactor, can be added back

        # Week range slider
        weeks_list_hm = sorted(heatmap_df['Week'].unique())
        if len(weeks_list_hm) > 1:
            selected_weeks_hm = st.select_slider(
                "Select week range for heatmap:",
                options=weeks_list_hm,
                value=(min(weeks_list_hm), max(weeks_list_hm)),
                key="heatmap_week_slider"
            )
            heatmap_df = heatmap_df[(heatmap_df['Week'] >= selected_weeks_hm[0]) & (heatmap_df['Week'] <= selected_weeks_hm[1])]

        if not heatmap_df.empty:
            # Determine sort order
            if sort_method == "Average Engagement":
                store_avg = heatmap_df.groupby('StoreID')['EngagedPct'].mean()
                store_order = store_avg.sort_values(ascending=False).index.tolist()
            else: # Recent Performance
                most_recent_week_hm = heatmap_df['Week'].max()
                recent_perf = heatmap_df[heatmap_df['Week'] == most_recent_week_hm].set_index('StoreID')['EngagedPct']
                store_order = recent_perf.sort_values(ascending=False).index.tolist()

            # Create heatmap
            heatmap_chart = alt.Chart(heatmap_df).mark_rect().encode(
                x=alt.X('Week:O', title='Week'),
                y=alt.Y('StoreID:O', title='Store', sort=store_order), # Apply sort order
                color=alt.Color('EngagedPct:Q', title='Engaged %', scale=alt.Scale(scheme=color_scheme)),
                tooltip=['StoreID', 'Week:O', alt.Tooltip('EngagedPct:Q', format='.2f')]
            ).properties(
                height=max(250, len(store_order)*20) # Dynamic height
            )
            st.altair_chart(heatmap_chart, use_container_width=True)
            st.caption(f"Stores sorted by {sort_method.lower()}. Darker colors = higher engagement.")
        else:
             st.info("No data within selected heatmap week range.")

# ----------------- TAB 2: Store Comparison -----------------
with tab2:
    st.subheader("Store Performance Comparison")

    if len(available_stores) <= 1:
        st.info("Select more than one store in the filters to enable comparison.")
    else:
        # --- Bar chart: Current Performance ---
        if week_choice != "All":
            comp_data = df_filtered[df_filtered['Week'] == int(week_choice)][['Store #', 'Engaged Transaction %']]
            comp_title = f"Store Comparison - Week {week_choice}"
        else: # Period Average
            comp_data = store_perf.reset_index() # Use pre-calculated average performance
            comp_data.columns = ['Store #', 'Engaged Transaction %']
            comp_title = "Store Comparison - Period Average"

        if not comp_data.empty:
             bar_chart = create_bar_chart(
                  comp_data,
                  x_col='Engaged Transaction %:Q', x_title='Engaged Transaction %',
                  y_col='Store #:N', y_title='Store',
                  color_col='Engaged Transaction %:Q', color_scale=alt.Scale(scheme='blues'),
                  tooltip_cols=['Store #', alt.Tooltip('Engaged Transaction %', format='.2f')],
                  title=comp_title
             )
             # Add average line
             district_avg_comp = comp_data['Engaged Transaction %'].mean()
             avg_rule = create_rule_layer(district_avg_comp, color='red', tooltip_text='District Average')
             st.altair_chart(bar_chart + avg_rule.encode(x='value:Q'), use_container_width=True) # Specify axis for rule

        else:
             st.info("No comparison data available for this selection.")


        # --- Bar chart: Difference from Average ---
        st.subheader("Performance Relative to District Average")
        if not comp_data.empty and district_avg_comp is not None:
            comp_data['Difference'] = comp_data['Engaged Transaction %'] - district_avg_comp
            comp_data['Percentage'] = (comp_data['Difference'] / district_avg_comp) * 100 if district_avg_comp != 0 else 0

            # Determine color range dynamically
            min_perc = comp_data['Percentage'].min()
            max_perc = comp_data['Percentage'].max()
            # Ensure 0 is included in the domain for the midpoint color
            color_domain = sorted(list(set([min_perc, 0, max_perc])))
            color_range = ['#C62828', '#BBBBBB', '#2E7D32'] # Red, Grey, Green
            if len(color_domain) == 2: # Handle cases where all values are positive or negative
                 if 0 not in color_domain: # Add 0 if missing
                      color_domain.append(0)
                      color_domain.sort()
                 # Adjust range if needed, e.g., only two colors if all > 0 or all < 0
                 idx_zero = color_domain.index(0)
                 if idx_zero == 0: color_range = color_range[1:] # All positive, use Grey/Green
                 elif idx_zero == len(color_domain) -1: color_range = color_range[:2] # All negative, use Red/Grey


            diff_chart = create_bar_chart(
                 comp_data,
                 x_col='Percentage:Q', x_title='% Difference from Average',
                 y_col='Store #:N', y_title='Store',
                 color_col='Percentage:Q',
                 color_scale=alt.Scale(domain=color_domain, range=color_range),
                 tooltip_cols=['Store #', alt.Tooltip('Engaged Transaction %', format='.2f'), alt.Tooltip('Percentage:Q', format='+.2f', title='% Diff')],
                 dynamic_height=True
            )
            center_rule = create_rule_layer(0, color='black', size=1)
            st.altair_chart(diff_chart + center_rule.encode(x='value:Q'), use_container_width=True)
            st.caption("Green > average, Red < average.")
        else:
             st.info("Cannot calculate difference from average.")


        # --- Weekly Rank Tracking ---
        if 'Weekly Rank' in df_filtered.columns:
            st.subheader("Weekly Rank Tracking")
            rank_data = df_filtered[['Week', 'Store #', 'Weekly Rank']].dropna()
            if not rank_data.empty:
                 base_rank_chart = create_base_chart(rank_data)
                 # Invert rank scale so 1 is at the top
                 rank_scale = alt.Scale(domain=[rank_data['Weekly Rank'].max(), 1]) # Max rank to 1

                 rank_chart_lines = create_line_layer(
                      base_rank_chart,
                      y_col='Weekly Rank:Q', y_title='Rank', color_col='Store #:N', color_title='Store',
                      tooltip_cols=['Store #', 'Week:O', alt.Tooltip('Weekly Rank:Q', title='Rank')],
                      add_interactive_legend=True
                 ).encode(y=alt.Y(scale=rank_scale)) # Apply inverted scale

                 rank_chart_points = create_point_layer(
                      base_rank_chart,
                      y_col='Weekly Rank:Q', color_col='Store #:N',
                      tooltip_cols=['Store #', 'Week:O', alt.Tooltip('Weekly Rank:Q', title='Rank')]
                 ).encode(y=alt.Y(scale=rank_scale)) # Apply inverted scale


                 # Link points opacity to legend selection if line layer has it
                 if hasattr(rank_chart_lines, 'params'):
                      selection_param = next((p for p in rank_chart_lines.params if isinstance(p, alt.SelectionParameter)), None)
                      if selection_param:
                           rank_chart_points = rank_chart_points.encode(opacity=alt.condition(selection_param, alt.value(1), alt.value(0.2)))


                 final_rank_chart = (rank_chart_lines + rank_chart_points).properties(height=350).interactive()
                 st.altair_chart(final_rank_chart, use_container_width=True)
                 st.caption("Lower rank number = better performance (Rank 1 is best). Click legend to highlight.")
            else:
                st.info("Weekly rank data not available or empty for the selected period/filters.")

# ----------------- TAB 3: Store Performance Categories -----------------
with tab3:
    st.subheader("Store Performance Categories")

    if store_stats_df.empty:
        st.warning("Store performance categories could not be calculated. Check data and filters.")
    else:
        st.write("Stores are categorized based on average engagement (vs. median) and recent performance trend:")

        # --- Category Explanation Cards ---
        colA, colB = st.columns(2)
        category_info = {
             "Star Performer": {"icon": "⭐", "desc": "High engagement, stable/improving trend", "action": "Share best practices", "col": colA},
             "Needs Stabilization": {"icon": "⚠️", "desc": "High engagement, but downward trend", "action": "Reinforce processes", "col": colA},
             "Improving": {"icon": "📈", "desc": "Below average, but positive trend", "action": "Continue momentum", "col": colB},
             "Requires Intervention": {"icon": "🚨", "desc": "Below average, flat/declining trend", "action": "Needs support", "col": colB}
        }
        for cat, info in category_info.items():
             display_category_info_card(info["col"], cat, info["icon"], info["desc"], info["action"], CATEGORY_COLORS[cat])

        # --- Display Stores by Category ---
        st.subheader("Store Category Results")
        for cat in CATEGORY_COLORS.keys(): # Iterate in defined order
            cat_stores = store_stats_df[store_stats_df['Category'] == cat]
            if not cat_stores.empty:
                 accent = CATEGORY_COLORS[cat]
                 st.markdown(f"""
                 <div style="border-left: 5px solid {accent}; padding-left: 15px; margin-bottom: 15px;">
                     <h4 style="color: {accent}; margin-top: 0;">{cat} ({len(cat_stores)} stores)</h4>
                 </div>
                 """, unsafe_allow_html=True)

                 # Display store cards in columns
                 cols_per_row = 4 # Adjust as needed
                 num_rows = (len(cat_stores) + cols_per_row - 1) // cols_per_row
                 store_list_cat = cat_stores.to_dict('records')

                 for r in range(num_rows):
                      cols = st.columns(cols_per_row)
                      for i in range(cols_per_row):
                           store_index = r * cols_per_row + i
                           if store_index < len(store_list_cat):
                                store = store_list_cat[store_index]
                                display_store_category_card(
                                     cols[i], store['Store #'], store['AverageEngagement'],
                                     store['Trend'], store['Category'], accent
                                )


        # --- Store-Specific Action Plans ---
        st.subheader("Store-Specific Details & Action Plan")
        store_options_cat = sorted(store_stats_df['Store #'].tolist())
        if not store_options_cat:
             st.info("No stores found after filtering.")
        else:
             selected_store_cat = st.selectbox(
                  "Select a store:", options=store_options_cat, key="tab3_store_select"
             )
             if selected_store_cat:
                  store_details = store_stats_df[store_stats_df['Store #'] == selected_store_cat].iloc[0]
                  accent = CATEGORY_COLORS[store_details['Category']]
                  trend = store_details['Trend']
                  if "Upward" in trend: trend_icon = "🔼"
                  elif "Downward" in trend: trend_icon = "🔽"
                  else: trend_icon = "➡️"

                  # Display details card
                  st.markdown(f"""
                  <div class="category-card-base category-card-accent" style="border-left-color: {accent}; padding: 20px;">
                      <h3 style="color: {accent}; margin-top: 0;">
                          Store {selected_store_cat} - {store_details['Category']}
                      </h3>
                      <p><strong>Average Engagement:</strong> {store_details['AverageEngagement']:.2f}%</p>
                      <p><strong>Trend ({trend_analysis_weeks} wks):</strong> {trend_icon} {trend}</p>
                      <p><strong>Explanation:</strong> {store_details['Explanation']}</p>
                      <h4 style="color: {accent}; margin-top: 1em;">Recommended Action:</h4>
                      <p>{store_details['Action Plan']}</p>
                  </div>
                  """, unsafe_allow_html=True)

                  # Display Trend Chart for the selected store
                  st.markdown(f"##### Store {selected_store_cat} Engagement Trend")
                  store_trend_data = df_filtered[df_filtered['Store #'] == selected_store_cat].sort_values('Week')
                  if not store_trend_data.empty:
                       base_store_chart = create_base_chart(store_trend_data)
                       store_line = create_line_layer(base_store_chart, 'Engaged Transaction %', 'Engagement %', alt.value(accent), 'Store', ['Week:O', alt.Tooltip('Engaged Transaction %:Q', format='.2f')], stroke_width=3)
                       store_points = create_point_layer(base_store_chart, 'Engaged Transaction %', alt.value(accent), ['Week:O', alt.Tooltip('Engaged Transaction %:Q', format='.2f')])

                       # Add district average line for context
                       dist_avg_line_store = create_line_layer(create_base_chart(dist_trend), 'Average Engagement %', 'Engagement %', alt.value('gray'), 'District Avg', [alt.Tooltip('Average Engagement %', format='.2f')], stroke_width=2, stroke_dash=[2,2])

                       final_store_chart = (store_line + store_points + dist_avg_line_store).properties(height=300).interactive()
                       st.altair_chart(final_store_chart, use_container_width=True)
                       st.caption(f"Store {selected_store_cat} trend (color), District average (gray dashed).")
                  else:
                       st.info("No trend data to display for this store in the filtered period.")

                  # --- Improvement Opportunities (Simplified) ---
                  if store_details['Category'] in ["Improving", "Requires Intervention"]:
                        st.markdown("##### Improvement Opportunities")
                        if district_median_engagement is not None:
                             improvement_potential = district_median_engagement - store_details['AverageEngagement']
                             if improvement_potential > 0:
                                  st.markdown(f"- Potential gain to reach district median ({district_median_engagement:.2f}%): **{improvement_potential:.2f}%**")

                        top_stores = store_stats_df[store_stats_df['Category'] == "Star Performer"]['Store #'].tolist()
                        if top_stores:
                             st.markdown(f"- Consider learning from Star Performers: **{', '.join(top_stores)}**")


# ----------------- TAB 4: Anomalies & Insights -----------------
with tab4:
    st.subheader("Anomaly Detection")
    st.write(f"Identifying significant week-over-week changes (Z-score > {z_threshold:.1f}).")

    if anomalies_df.empty:
        st.info("No significant anomalies detected for the selected criteria.")
    else:
        with st.expander("View Anomaly Details", expanded=True):
            display_cols_anom = [
                'Store #', 'Week', 'Engaged Transaction %', 'Change %pts',
                'Z-score', 'Rank', 'Prev Rank', 'Possible Explanation'
            ]
            # Filter df to only include columns that actually exist
            display_cols_anom_exist = [col for col in display_cols_anom if col in anomalies_df.columns]
            st.dataframe(anomalies_df[display_cols_anom_exist], hide_index=True)

    # --- Additional Insights ---
    st.subheader("Additional Insights")
    insight_tabs = st.tabs(["Period Performance", "Opportunities"])

    with insight_tabs[0]: # Period Performance
        st.markdown("##### Performance Summary (Filtered Period)")
        if not store_stats_df.empty:
            # Display key stats calculated earlier
            stats_display = store_stats_df[['Store #', 'AverageEngagement', 'StdDev', 'Trend', 'Category']].copy()
            stats_display.rename(columns={'AverageEngagement': 'Avg Eng %', 'StdDev': 'Consistency (StdDev)'}, inplace=True)
            stats_display['Avg Eng %'] = stats_display['Avg Eng %'].round(2)
            stats_display['Consistency (StdDev)'] = stats_display['Consistency (StdDev)'].round(2)
            st.dataframe(stats_display.sort_values('Avg Eng %', ascending=False), hide_index=True)
        else:
            st.info("Store statistics not available.")

        # Trend Correlation Chart (Optional alternative view)
        # Trend calculation already done and stored in store_stats_df['Trend']
        # Could visualize the distribution of trends if needed


    with insight_tabs[1]: # Opportunities
        st.markdown("##### Improvement Opportunities")
        if not store_perf.empty and len(store_perf) > 1 and district_median_engagement is not None:
            current_district_avg = store_perf.mean() # Use filtered average

            # --- Scenario 1: Bottom performer to median ---
            if bottom_store != "N/A" and bottom_val is not None:
                scenario_perf = store_perf.copy()
                scenario_perf[bottom_store] = district_median_engagement # Target median
                new_avg_scenario1 = scenario_perf.mean()
                improvement1 = new_avg_scenario1 - current_district_avg
                st.markdown(f"""
                **Scenario 1: Bottom Performer to Median**
                - If Store **{bottom_store}** ({bottom_val:.2f}%) reached the median ({district_median_engagement:.2f}%):
                - District average could increase by **{improvement1:.2f} pts** (to {new_avg_scenario1:.2f}%)
                """)

            # --- Gap to Top Performer ---
            st.markdown("##### Gap to Top Performer")
            if top_store != "N/A" and top_val is not None:
                gap_df = pd.DataFrame({
                    'Store #': store_perf.index,
                    'Current %': store_perf.values,
                    'Gap to Top': top_val - store_perf.values
                }).round(2)
                gap_df = gap_df[gap_df['Gap to Top'] > 0.01].sort_values('Gap to Top', ascending=False) # Exclude top store, small gaps

                if not gap_df.empty:
                     gap_chart = create_bar_chart(
                          gap_df,
                          x_col='Gap to Top:Q', x_title='Gap to Top Performer (%)',
                          y_col='Store #:N', y_title='Store', sort_y='-x',
                          color_col='Gap to Top:Q', color_scale=alt.Scale(scheme='reds'),
                          tooltip_cols=['Store #', 'Current %', 'Gap to Top']
                     )
                     st.altair_chart(gap_chart, use_container_width=True)
                     st.caption("Shows how far each store's average is from the top performer's average.")
                else:
                     st.info("No significant gaps to the top performer found.")
            else:
                 st.info("Top performer information not available.")

        else:
            st.info("Insufficient data or stores for opportunity analysis.")


# --------------------------------------------------------
# Footer & Help
# --------------------------------------------------------
now = datetime.datetime.now()
st.sidebar.markdown("---")
st.sidebar.caption(f"© Publix Supermarkets, Inc. {now.year}")
st.sidebar.caption(f"Last updated: {now.strftime('%Y-%m-%d %H:%M')}")

with st.sidebar.expander("Help & Information"):
    st.markdown("""
    **Using This Dashboard**
    1.  **Upload Data**: Use the sidebar to upload primary (and optionally comparison) data.
    2.  **Apply Filters**: Refine by Quarter, Week, or Store(s).
    3.  **Explore Tabs**:
        - **📊 Engagement Trends**: Performance over time.
        - **📈 Store Comparison**: Direct store-vs-store views.
        - **📋 Store Categories**: Performance segments & action plans.
        - **💡 Anomalies & Insights**: Outliers and opportunities.
    4.  **Advanced Settings**: Adjust anomaly sensitivity, trend window, etc.

    *Contact Reid for support.*
    """)