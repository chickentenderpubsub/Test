import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import datetime

# --------------------------------------------------------
# Helper Functions
# --------------------------------------------------------

@st.cache_data
def load_data(uploaded_file):
    """
    Reads CSV/XLSX file into a pandas DataFrame, standardizes key columns,
    and sorts by week/store.
    """
    filename = uploaded_file.name.lower()
    if filename.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
    # Standardize column names
    df.columns = standardize_columns(df.columns)
    # Parse dates
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    # Convert percentage columns to numeric
    percent_cols = ['Engaged Transaction %', 'Quarter to Date %']
    for col in percent_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace('%', ''), errors='coerce')
    # Drop rows with no engagement data
    df = df.dropna(subset=['Engaged Transaction %'])
    # Convert data types
    if 'Weekly Rank' in df.columns:
        df['Weekly Rank'] = pd.to_numeric(df['Weekly Rank'], errors='coerce')
        df['Weekly Rank'] = df['Weekly Rank'].astype('Int64')  # integer rank (allow NA)
    if 'Store #' in df.columns:
        df['Store #'] = df['Store #'].astype(str)
    # Ensure Week is integer if present
    if 'Week' in df.columns:
        df['Week'] = df['Week'].astype(int)
        df = df.sort_values(['Week', 'Store #'])
    return df

def standardize_columns(columns):
    """
    Renames columns to standard internal names for consistency.
    """
    new_cols = []
    for col in columns:
        cl = col.strip().lower()
        if 'quarter' in cl or 'qtd' in cl:
            new_cols.append('Quarter to Date %')
        elif 'rank' in cl:
            new_cols.append('Weekly Rank')
        elif ('week' in cl and 'ending' in cl) or cl == 'date' or cl == 'week ending':
            new_cols.append('Date')
        elif cl.startswith('week'):
            new_cols.append('Week')
        elif 'store' in cl:
            new_cols.append('Store #')
        elif 'engaged' in cl or 'engagement' in cl:
            new_cols.append('Engaged Transaction %')
        else:
            new_cols.append(col)
    return new_cols

def calculate_trend(group, window=4):
    """
    Calculates a trend label (Upward, Downward, etc.) based on a linear slope 
    of the last `window` data points in 'Engaged Transaction %'.
    """
    if len(group) < 2:
        return "Stable"
    sorted_data = group.sort_values('Week', ascending=True).tail(window)
    if len(sorted_data) < 2:
        return "Insufficient Data"
    x = sorted_data['Week'].values
    y = sorted_data['Engaged Transaction %'].values
    # Center X to avoid numeric issues
    x = x - np.mean(x)
    if np.sum(x**2) == 0:
        return "Stable"
    slope = np.sum(x * y) / np.sum(x**2)
    if slope > 0.5:
        return "Strong Upward"
    elif slope > 0.1:
        return "Upward"
    elif slope < -0.5:
        return "Strong Downward"
    elif slope < -0.1:
        return "Downward"
    else:
        return "Stable"

def find_anomalies(df, z_threshold=2.0):
    """
    Calculates week-over-week changes in Engaged Transaction % for each store
    and flags any changes whose Z-score exceeds the given threshold.
    Returns a DataFrame of anomalies with potential explanations.
    """
    anomalies_list = []
    for store_id, grp in df.groupby('Store #'):
        grp = grp.sort_values('Week')
        diffs = grp['Engaged Transaction %'].diff().dropna()
        if diffs.empty:
            continue
        mean_diff = diffs.mean()
        std_diff = diffs.std(ddof=0)
        if std_diff == 0 or np.isnan(std_diff):
            continue
        for idx, diff_val in diffs.items():
            z = (diff_val - mean_diff) / std_diff
            if abs(z) >= z_threshold:
                week_cur = grp.loc[idx, 'Week']
                prev_idx = grp.index[grp.index.get_indexer([idx]) - 1][0] if grp.index.get_indexer([idx])[0] - 1 >= 0 else None
                week_prev = grp.loc[prev_idx, 'Week'] if prev_idx is not None else None
                val_cur = grp.loc[idx, 'Engaged Transaction %']
                rank_cur = grp.loc[idx, 'Weekly Rank'] if 'Weekly Rank' in grp.columns else None
                rank_prev = grp.loc[prev_idx, 'Weekly Rank'] if prev_idx is not None and 'Weekly Rank' in grp.columns else None
                anomalies_list.append({
                    'Store #': store_id,
                    'Week': int(week_cur),
                    'Engaged Transaction %': val_cur,
                    'Change %pts': diff_val,
                    'Z-score': z,
                    'Prev Week': int(week_prev) if week_prev is not None else None,
                    'Prev Rank': int(rank_prev) if rank_prev is not None and pd.notna(rank_prev) else None,
                    'Rank': int(rank_cur) if pd.notna(rank_cur) else None
                })
    anomalies_df = pd.DataFrame(anomalies_list)
    if not anomalies_df.empty:
        anomalies_df['Abs Z'] = anomalies_df['Z-score'].abs()
        anomalies_df = anomalies_df.sort_values('Abs Z', ascending=False).drop(columns=['Abs Z'])
        anomalies_df['Engaged Transaction %'] = anomalies_df['Engaged Transaction %'].round(2)
        anomalies_df['Z-score'] = anomalies_df['Z-score'].round(2)
        anomalies_df['Change %pts'] = anomalies_df['Change %pts'].round(2)
        # Add quick textual explanation
        explanations = []
        for _, row in anomalies_df.iterrows():
            if row['Change %pts'] >= 0:
                reason = "Engagement spiked significantly. Possible promotion or event impact."
            else:
                reason = "Engagement dropped significantly. Investigate potential issues (e.g., staffing, inventory)."
            explanations.append(reason)
        anomalies_df['Potential Cause'] = explanations
    return anomalies_df

# --------------------------------------------------------
# Page Configuration and Custom CSS
# --------------------------------------------------------
st.set_page_config(
    page_title="Publix District 20 Engagement Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Publix branding
st.markdown("""
<style>
/* Global font and base text color */
.stApp {
    font-family: 'Futura', 'Helvetica Neue', Arial, sans-serif;
    color: #2E2E2E;
    background-color: #FFFFFF;
}
/* Sidebar background */
section[data-testid="stSidebar"] {
    background-color: #F5F5F5;
}
/* Dashboard title styling */
.dashboard-title {
    color: #007749;
    text-align: center;
    padding-bottom: 20px;
}
/* KPI Metric card styling */
div[data-testid="metric-container"] {
    background-color: #FFFFFF;
    border: 1px solid #e0e0e0;
    border-radius: 10px;
    padding: 15px;
    box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
}
div[data-testid="metric-container"] > label[data-testid="stMetricLabel"] > div {
    color: #007749;
}
div[data-testid="metric-container"] > div[data-testid="stMetricValue"] {
    color: #2E2E2E;
    font-weight: bold;
}
/* Highlight classes for trend text */
.highlight-good {
    color: #43B02A;
    font-weight: bold;
}
.highlight-bad {
    color: #C62828;
    font-weight: bold;
}
.highlight-neutral {
    color: #808080;
    font-weight: bold;
}
/* Caption and subtext */
.caption-text {
    font-size: 0.85em;
    color: #555555;
}
/* Filter bar container */
.filter-bar {
    background-color: #F5F5F5;
    padding: 10px;
    border-radius: 5px;
    margin-bottom: 20px;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------------
# Title & Introduction
# --------------------------------------------------------
st.markdown("<h1 class='dashboard-title'>Publix District 20 Engagement Dashboard</h1>", unsafe_allow_html=True)
st.markdown("**Publix Supermarkets – District 20** engagement analysis dashboard. "
            "Upload weekly engagement data to explore key performance indicators, trends, and opportunities across 10 stores. "
            "Use the filters above to drill down by time period or store.")

# --------------------------------------------------------
# Sidebar for Data Input
# --------------------------------------------------------
st.sidebar.header("Data Input")
data_file = st.sidebar.file_uploader("Upload engagement data (Excel or CSV)", type=['csv', 'xlsx'])
comp_file = st.sidebar.file_uploader("Optional: Upload comparison data (prior period)", type=['csv', 'xlsx'])

if not data_file:
    st.info("Please upload a primary engagement data file to begin.")
    st.markdown("### Expected Data Format")
    st.markdown("""
    Your data file should contain the following columns:
    - Store # or Store ID  
    - Week or Date  
    - Engaged Transaction % (the main KPI)  
    - Optional: Weekly Rank, Quarter to Date %, etc.
    
    **Example formats supported:**  
    - CSV with headers  
    - Excel file with data in the first sheet
    """)
    st.stop()

df = load_data(data_file)
df_comp = load_data(comp_file) if comp_file else None

# Derive Quarter from Date or Week
if 'Date' in df.columns:
    df['Quarter'] = df['Date'].dt.quarter
elif 'Week' in df.columns:
    df['Quarter'] = ((df['Week'] - 1) // 13 + 1).astype(int)

if df_comp is not None:
    if 'Date' in df_comp.columns:
        df_comp['Quarter'] = df_comp['Date'].dt.quarter
    elif 'Week' in df_comp.columns:
        df_comp['Quarter'] = ((df_comp['Week'] - 1) // 13 + 1).astype(int)

# --------------------------------------------------------
# Filters (Quarter, Week, Store)
# --------------------------------------------------------
st.markdown('<div class="filter-bar">', unsafe_allow_html=True)
col1, col2, col3 = st.columns([1, 1, 2])
with col1:
    quarters = sorted(df['Quarter'].dropna().unique().tolist())
    quarter_options = ["All"] + [f"Q{int(q)}" for q in quarters]
    quarter_choice = st.selectbox("Select Quarter", quarter_options, index=0)
# Determine available weeks based on quarter selection
if quarter_choice != "All":
    q_num = int(quarter_choice[1:])
    available_weeks = sorted(df[df['Quarter'] == q_num]['Week'].unique().tolist())
else:
    available_weeks = sorted(df['Week'].unique().tolist())
week_options = ["All"] + [str(int(w)) for w in available_weeks]
with col2:
    week_choice = st.selectbox("Select Week", week_options, index=0)
store_list = sorted(df['Store #'].unique().tolist())
with col3:
    store_choice = st.multiselect("Select Store(s)", store_list, default=[])
st.markdown("</div>", unsafe_allow_html=True)

# Advanced settings (in sidebar)
with st.sidebar.expander("Advanced Settings", expanded=False):
    z_threshold = st.slider("Anomaly Z-score Threshold", 1.0, 3.0, 2.0, 0.1)
    show_ma = st.checkbox("Show 4-week moving average", value=True)
    highlight_top = st.checkbox("Highlight top performer", value=True)
    highlight_bottom = st.checkbox("Highlight bottom performer", value=True)
    trend_analysis_weeks = st.slider("Trend analysis window (weeks)", 3, 8, 4)
    st.caption("Adjust the sensitivity for anomaly detection. (Higher = fewer anomalies)")

# Filter main dataframe based on selections
df_filtered = df.copy()
if quarter_choice != "All":
    df_filtered = df_filtered[df_filtered['Quarter'] == q_num]
if week_choice != "All":
    week_num = int(week_choice)
    df_filtered = df_filtered[df_filtered['Week'] == week_num]
if store_choice:
    df_filtered = df_filtered[df_filtered['Store #'].isin([str(s) for s in store_choice])]

# Filter comparison dataframe similarly
df_comp_filtered = None
if df_comp is not None:
    df_comp_filtered = df_comp.copy()
    if quarter_choice != "All":
        df_comp_filtered = df_comp_filtered[df_comp_filtered['Quarter'] == q_num]
    if week_choice != "All":
        df_comp_filtered = df_comp_filtered[df_comp_filtered['Week'] == week_num]
    if store_choice:
        df_comp_filtered = df_comp_filtered[df_comp_filtered['Store #'].isin([str(s) for s in store_choice])]

if df_filtered.empty:
    st.error("No data available for the selected filters. Please adjust your filters.")
    st.stop()

# --------------------------------------------------------
# Executive Summary Calculations
# --------------------------------------------------------
# Identify current and previous week
if week_choice != "All":
    current_week = int(week_choice)
    prev_week = current_week - 1
    # If prev_week not in the data, pick the largest week < current_week
    if prev_week not in df_filtered['Week'].values:
        prev_df = df[(df['Week'] < current_week) & ((quarter_choice == "All") or (df['Quarter'] == q_num))]
        prev_week = int(prev_df['Week'].max()) if not prev_df.empty else None
else:
    current_week = int(df_filtered['Week'].max())
    prev_weeks = df_filtered['Week'][df_filtered['Week'] < current_week]
    prev_week = int(prev_weeks.max()) if not prev_weeks.empty else None

# District/Selection average for current/prev week
current_avg = df_filtered[df_filtered['Week'] == current_week]['Engaged Transaction %'].mean() if current_week else None
prev_avg = df_filtered[df_filtered['Week'] == prev_week]['Engaged Transaction %'].mean() if prev_week else None

# Top/Bottom performer over the filtered period
store_perf = df_filtered.groupby('Store #')['Engaged Transaction %'].mean()
top_store = store_perf.idxmax()
bottom_store = store_perf.idxmin()
top_val = store_perf.max()
bottom_val = store_perf.min()

# Calculate trend for each store (using the helper function)
store_trends = df_filtered.groupby('Store #').apply(lambda x: calculate_trend(x, trend_analysis_weeks))
top_store_trend = store_trends.get(top_store, "Stable")
bottom_store_trend = store_trends.get(bottom_store, "Stable")

# --------------------------------------------------------
# Executive Summary Display
# --------------------------------------------------------
st.subheader("Executive Summary")
col1, col2, col3 = st.columns(3)

# Label for average engagement metric
if store_choice and len(store_choice) == 1:
    avg_label = f"Store {store_choice[0]} Engagement"
elif store_choice and len(store_choice) < len(store_list):
    avg_label = "Selected Stores Avg Engagement"
else:
    avg_label = "District Avg Engagement"

avg_display = f"{current_avg:.2f}%" if current_avg is not None else "N/A"
if current_avg is not None and prev_avg is not None:
    delta_val = current_avg - prev_avg
    delta_str = f"{delta_val:+.2f}%"
else:
    delta_str = "N/A"

col1.metric(avg_label + f" (Week {current_week})", avg_display, delta_str)
col2.metric(f"Top Performer (Week {current_week})", f"Store {top_store} — {top_val:.2f}%")
col3.metric(f"Bottom Performer (Week {current_week})", f"Store {bottom_store} — {bottom_val:.2f}%")

# Trend indicator sentence
if current_avg is not None and prev_avg is not None:
    delta_abs = abs(delta_val)
    if delta_val > 0:
        trend = "up"
        trend_class = "highlight-good"
    elif delta_val < 0:
        trend = "down"
        trend_class = "highlight-bad"
    else:
        trend = "flat"
        trend_class = "highlight-neutral"
    st.markdown(
        f"Week {current_week} average engagement is "
        f"<span class='{trend_class}'>{delta_abs:.2f} percentage points {trend}</span> "
        f"from Week {prev_week}.",
        unsafe_allow_html=True
    )
elif current_avg is not None:
    st.markdown(
        f"Week {current_week} engagement average: "
        f"<span class='highlight-neutral'>{current_avg:.2f}%</span>",
        unsafe_allow_html=True
    )

# Top & Bottom store trends
col1, col2 = st.columns(2)
with col1:
    tcolor = "highlight-good" if top_store_trend in ["Upward", "Strong Upward"] else \
             "highlight-bad" if top_store_trend in ["Downward", "Strong Downward"] else "highlight-neutral"
    st.markdown(f"**Store {top_store}** trend: <span class='{tcolor}'>{top_store_trend}</span>", unsafe_allow_html=True)
with col2:
    bcolor = "highlight-good" if bottom_store_trend in ["Upward", "Strong Upward"] else \
             "highlight-bad" if bottom_store_trend in ["Downward", "Strong Downward"] else "highlight-neutral"
    st.markdown(f"**Store {bottom_store}** trend: <span class='{bcolor}'>{bottom_store_trend}</span>", unsafe_allow_html=True)

# --------------------------------------------------------
# Key Insights
# --------------------------------------------------------
st.subheader("Key Insights")
insights = []

# 1) Consistency
store_std = df_filtered.groupby('Store #')['Engaged Transaction %'].std().fillna(0)
most_consistent = store_std.idxmin()
least_consistent = store_std.idxmax()
insights.append(f"**Store {most_consistent}** shows the most consistent engagement (lowest variability).")
insights.append(f"**Store {least_consistent}** has the most variable engagement performance.")

# 2) Trend analysis
trending_up = store_trends[store_trends.isin(["Upward", "Strong Upward"])].index.tolist()
trending_down = store_trends[store_trends.isin(["Downward", "Strong Downward"])].index.tolist()
if trending_up:
    insights.append(f"Stores showing positive trends: {', '.join([f'**{s}**' for s in trending_up])}")
if trending_down:
    insights.append(f"Stores needing attention: {', '.join([f'**{s}**' for s in trending_down])}")

# 3) Gap analysis
if len(store_perf) > 1:
    engagement_gap = top_val - bottom_val
    insights.append(f"Gap between highest and lowest performing stores: **{engagement_gap:.2f}%**")
    if engagement_gap > 10:
        insights.append("🚨 Large performance gap indicates opportunity for knowledge sharing.")

for i, insight in enumerate(insights[:5], start=1):
    st.markdown(f"{i}. {insight}")

# --------------------------------------------------------
# Main Tabs
# --------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "Engagement Trends",
    "Store Comparison",
    "Store Performance Categories",
    "Anomalies & Insights"
])

# ----------------- TAB 1: Engagement Trends -----------------
with tab1:
    st.subheader("Engagement Trends Over Time")

    view_option = st.radio(
        "View mode:",
        ["All Stores", "Custom Selection", "Recent Trends"],
        horizontal=True,
        help="All Stores: View all stores at once | Custom Selection: Pick specific stores to compare | Recent Trends: Focus on recent weeks"
    )

    # 1) Compute district average over time
    dist_trend = df_filtered.groupby('Week', as_index=False)['Engaged Transaction %'].mean().sort_values('Week')
    dist_trend.rename(columns={'Engaged Transaction %': 'Average Engagement %'}, inplace=True)
    dist_trend['MA_4W'] = dist_trend['Average Engagement %'].rolling(window=4, min_periods=1).mean()

    # 2) Compute 4-week moving average for each store
    df_filtered = df_filtered.sort_values(['Store #', 'Week'])
    df_filtered['MA_4W'] = df_filtered.groupby('Store #')['Engaged Transaction %']\
        .transform(lambda s: s.rolling(window=4, min_periods=1).mean())

    # 3) Combine current and comparison period if available
    if df_comp_filtered is not None and not df_comp_filtered.empty:
        df_filtered['Period'] = 'Current'
        df_comp_filtered['Period'] = 'Comparison'
        combined = pd.concat([df_filtered, df_comp_filtered], ignore_index=True)
        combined = combined.sort_values(['Store #', 'Period', 'Week'])
        combined['MA_4W'] = combined.groupby(['Store #', 'Period'])['Engaged Transaction %']\
            .transform(lambda s: s.rolling(window=4, min_periods=1).mean())
    else:
        combined = df_filtered.copy()
        combined['Period'] = 'Current'

    # For Recent Trends view, allow adjustable week range
    if view_option == "Recent Trends":
        all_weeks = sorted(combined['Week'].unique())
        default_start = all_weeks[0] if len(all_weeks) <= 8 else all_weeks[-8]
        default_end = all_weeks[-1]
        recent_weeks_range = st.select_slider(
            "Select weeks to display:",
            options=all_weeks,
            value=(default_start, default_end),
            help="Adjust to show more or fewer weeks in the trend view"
        )
        recent_weeks = [week for week in all_weeks if week >= recent_weeks_range[0] and week <= recent_weeks_range[1]]
        combined = combined[combined['Week'].isin(recent_weeks)]
        dist_trend = dist_trend[dist_trend['Week'].isin(recent_weeks)]

    # Base chart configuration
    base = alt.Chart(combined).encode(
        x=alt.X('Week:O', title='Week (Ordinal)')
    )
    # Custom color scale for stores (Publix palette shades)
    color_scale = alt.Scale(range=[
        "#007749", "#43B02A", "#388E3C", "#66BB6A", "#8BC34A",
        "#2E2E2E", "#636363", "#9E9E9E", "#C62828"
    ])

    # Initialize list to collect chart layers
    layers = []

    # 4) Handle different view modes
    if view_option == "Custom Selection":
        # Let user select specific stores to compare
        store_list = sorted(df_filtered['Store #'].unique().tolist())
        selected_stores = st.multiselect(
            "Select stores to compare:",
            options=store_list,
            default=[store_list[0]] if store_list else [],
            help="Choose specific stores to highlight in the chart"
        )
        if selected_stores:
            selected_data = combined[combined['Store #'].isin(selected_stores)]
            # Draw lines for selected stores
            store_lines = alt.Chart(selected_data).mark_line(strokeWidth=3).encode(
                x='Week:O',
                y=alt.Y('Engaged Transaction %:Q', title='Engaged Transaction %'),
                color=alt.Color('Store #:N', scale=color_scale, title='Store'),
                tooltip=['Store #', 'Week', alt.Tooltip('Engaged Transaction %', format='.2f')]
            )
            layers.append(store_lines)
            # Add point markers for selected stores
            store_points = alt.Chart(selected_data).mark_point(filled=True, size=80).encode(
                x='Week:O',
                y='Engaged Transaction %:Q',
                color=alt.Color('Store #:N', scale=color_scale),
                tooltip=['Store #', 'Week', alt.Tooltip('Engaged Transaction %', format='.2f')]
            )
            layers.append(store_points)
            # Optional moving average lines for selected stores
            if show_ma:
                ma_lines = alt.Chart(selected_data).mark_line(strokeDash=[2,2], strokeWidth=2).encode(
                    x='Week:O',
                    y=alt.Y('MA_4W:Q', title='4W MA'),
                    color=alt.Color('Store #:N', scale=color_scale),
                    tooltip=['Store #', 'Week', alt.Tooltip('MA_4W', format='.2f')]
                )
                layers.append(ma_lines)
        else:
            st.info("Please select at least one store to display.")
    else:
        # All Stores or Recent Trends view
        store_line_chart = base.mark_line(strokeWidth=1.5).encode(
            y=alt.Y('Engaged Transaction %:Q', title='Engaged Transaction %'),
            color=alt.Color('Store #:N', scale=color_scale, title='Store'),
            tooltip=['Store #', 'Week', alt.Tooltip('Engaged Transaction %', format='.2f')]
        )
        # Add interactive legend selection (opacity and line thickness toggle)
        store_selection = alt.selection_point(fields=['Store #'], bind='legend')
        store_line_chart = store_line_chart.add_params(store_selection).encode(
            opacity=alt.condition(store_selection, alt.value(1), alt.value(0.2)),
            strokeWidth=alt.condition(store_selection, alt.value(3), alt.value(1))
        )
        layers.append(store_line_chart)
        # Optional moving average lines for all stores
        if show_ma:
            ma_line_chart = base.mark_line(strokeDash=[2,2], strokeWidth=1.5).encode(
                y=alt.Y('MA_4W:Q', title='4W MA'),
                color=alt.Color('Store #:N', scale=color_scale, title='Store'),
                opacity=alt.condition(store_selection, alt.value(0.8), alt.value(0.1)),
                tooltip=['Store #', 'Week', alt.Tooltip('MA_4W', format='.2f')]
            )
            layers.append(ma_line_chart)

    # 5) District average line for the current period
    if not dist_trend.empty:
        dist_line_curr = alt.Chart(dist_trend).mark_line(
            color='#2E2E2E', strokeDash=[4,2], size=3
        ).encode(
            x='Week:O',
            y=alt.Y('Average Engagement %:Q', title='Engaged Transaction %'),
            tooltip=[alt.Tooltip('Average Engagement %:Q', format='.2f', title='District Avg')]
        )
        layers.append(dist_line_curr)
        if show_ma:
            dist_line_curr_ma = alt.Chart(dist_trend).mark_line(
                color='#2E2E2E', strokeDash=[1,1], size=2, opacity=0.7
            ).encode(
                x='Week:O',
                y='MA_4W:Q',
                tooltip=[alt.Tooltip('MA_4W:Q', format='.2f', title='District 4W MA')]
            )
            layers.append(dist_line_curr_ma)

    # 6) Comparison period line if available
    if df_comp_filtered is not None and not df_comp_filtered.empty:
        # For Recent Trends, use filtered comparison data
        if view_option == "Recent Trends":
            df_comp_filtered_view = df_comp_filtered[df_comp_filtered['Week'].isin(recent_weeks)]
        else:
            df_comp_filtered_view = df_comp_filtered
        dist_trend_comp = df_comp_filtered_view.groupby('Week', as_index=False)['Engaged Transaction %'].mean().sort_values('Week')
        dist_trend_comp['MA_4W'] = dist_trend_comp['Engaged Transaction %'].rolling(window=4, min_periods=1).mean()
        if not dist_trend_comp.empty:
            dist_line_comp = alt.Chart(dist_trend_comp).mark_line(
                color='#555555', strokeDash=[4,2], size=2
            ).encode(
                x='Week:O',
                y='Engaged Transaction %:Q',
                tooltip=[alt.Tooltip('Engaged Transaction %:Q', format='.2f', title="Last Period District Avg")]
            )
            layers.append(dist_line_comp)
            if show_ma:
                dist_line_comp_ma = alt.Chart(dist_trend_comp).mark_line(
                    color='#555555', strokeDash=[1,1], size=1.5, opacity=0.7
                ).encode(
                    x='Week:O',
                    y='MA_4W:Q',
                    tooltip=[alt.Tooltip('MA_4W:Q', format='.2f', title="Last Period 4W MA")]
                )
                layers.append(dist_line_comp_ma)

    # 7) Create the layered chart
    if layers:
        # Add WoW metric for Recent Trends view
        if view_option == "Recent Trends":
            col_a, col_b = st.columns(2)
            with col_a:
                last_weeks = sorted(dist_trend['Week'].unique())[-2:]
                if len(last_weeks) >= 2:
                    current = dist_trend[dist_trend['Week'] == last_weeks[1]]['Average Engagement %'].values[0]
                    previous = dist_trend[dist_trend['Week'] == last_weeks[0]]['Average Engagement %'].values[0]
                    change = current - previous
                    change_pct = (change / previous * 100) if previous != 0 else 0
                    st.metric(
                        "District Trend (Week-over-Week)",
                        f"{current:.2f}%",
                        f"{change_pct:.1f}%",
                        delta_color="normal"
                    )
            with col_b:
                last_week = max(combined['Week'])
                last_week_data = combined[combined['Week'] == last_week]
                if not last_week_data.empty:
                    best_store = last_week_data.loc[last_week_data['Engaged Transaction %'].idxmax()]
                    st.metric(
                        f"Top Performer (Week {last_week})",
                        f"Store {best_store['Store #']}",
                        f"{best_store['Engaged Transaction %']:.2f}%",
                        delta_color="off"
                    )
        # Display the final layered chart
        final_chart = alt.layer(*layers).resolve_scale(y='shared').properties(height=400)
        st.altair_chart(final_chart, use_container_width=True)
    else:
        st.info("No data available to display in the chart.")

    # 9) Descriptive caption based on view mode
    if view_option == "All Stores":
        caption = "**All Stores View:** Shows all store trends with interactive legend selection. The black dashed line represents the district average."
    elif view_option == "Custom Selection":
        caption = "**Custom Selection View:** Shows only selected stores with emphasized lines and markers for better comparison."
    else:  # Recent Trends
        caption = "**Recent Trends View:** Focuses on selected weeks with additional trend metrics above the chart."
    if df_comp_filtered is not None and not df_comp_filtered.empty:
        caption += " The gray dashed line represents the previous period's district average."
    st.caption(caption)

    # ----------------- Heatmap with Fixed Controls -----------------
    st.subheader("Weekly Engagement Heatmap")
    # Controls for heatmap (in expander for cleanliness)
    with st.container():
        with st.expander("Heatmap Settings", expanded=False):
            col_h1, col_h2 = st.columns([1, 1])
            with col_h1:
                sort_method = st.selectbox(
                    "Sort stores by:",
                    ["Average Engagement", "Recent Performance"],
                    index=0,
                    help="Choose how to order stores in the heatmap"
                )
            with col_h2:
                color_scheme = st.selectbox(
                    "Color scheme:",
                    ["Blues", "Greens", "Purples", "Oranges", "Reds", "Viridis"],
                    index=0,
                    help="Choose the color gradient for the heatmap"
                )
                normalize_colors = st.checkbox(
                    "Normalize colors by week",
                    value=False,
                    help="When checked, color intensity is relative to each week instead of across all weeks"
                )
        # Week range slider outside expander for easy access
        weeks_list = sorted(df_filtered['Week'].unique())
        if len(weeks_list) > 4:
            selected_weeks = st.select_slider(
                "Select week range for heatmap:",
                options=weeks_list,
                value=(min(weeks_list), max(weeks_list))
            )
            heatmap_df = df_filtered[(df_filtered['Week'] >= selected_weeks[0]) & (df_filtered['Week'] <= selected_weeks[1])].copy()
        else:
            heatmap_df = df_filtered.copy()

    # Prepare data for heatmap
    heatmap_df = heatmap_df.rename(columns={'Store #': 'StoreID', 'Engaged Transaction %': 'EngagedPct'}).copy()
    if heatmap_df.empty or heatmap_df['EngagedPct'].dropna().empty:
        st.info("No data available for the heatmap.")
    else:
        # Sort stores for heatmap Y-axis order
        if sort_method == "Average Engagement":
            store_avg = heatmap_df.groupby('StoreID')['EngagedPct'].mean().reset_index()
            store_order = store_avg.sort_values('EngagedPct', ascending=False)['StoreID'].tolist()
        else:  # Recent Performance
            most_recent_week = max(heatmap_df['Week'])
            recent_perf = heatmap_df[heatmap_df['Week'] == most_recent_week]
            store_order = recent_perf.sort_values('EngagedPct', ascending=False)['StoreID'].tolist()
        domain = store_order

        # Normalize colors option
        if normalize_colors:
            week_stats = heatmap_df.groupby('Week')['EngagedPct'].agg(['min', 'max']).reset_index()
            heatmap_df = pd.merge(heatmap_df, week_stats, on='Week')
            heatmap_df['NormalizedPct'] = heatmap_df.apply(
                lambda row: 0 if row['min'] == row['max'] else 100 * (row['EngagedPct'] - row['min']) / (row['max'] - row['min']),
                axis=1
            )
            color_field = 'NormalizedPct:Q'
            color_title = 'Normalized %'
        else:
            color_field = 'EngagedPct:Q'
            color_title = 'Engaged %'

        # Heatmap chart
        heatmap_chart = alt.Chart(heatmap_df).mark_rect().encode(
            x=alt.X('Week:O', title='Week'),
            y=alt.Y('StoreID:O', title='Store', sort=domain),
            color=alt.Color(color_field, title=color_title, scale=alt.Scale(scheme=color_scheme.lower()), legend=alt.Legend(orient='right')),
            tooltip=['StoreID', 'Week:O', alt.Tooltip('EngagedPct:Q', format='.2f')]
        ).properties(height=max(250, len(store_order) * 20))
        st.altair_chart(heatmap_chart, use_container_width=True)

        # Heatmap details caption
        st.caption(
            f"**Heatmap Details:** Showing engagement data from Week {min(heatmap_df['Week'])} to Week {max(heatmap_df['Week'])}. "
            + f"Stores sorted by {sort_method.lower()}. "
            + ("Colors normalized within each week. " if normalize_colors else "Global color scale across all weeks. ")
            + "Darker colors represent higher engagement values."
        )

        # ----------------- Streak Analysis Visualization -----------------
        st.subheader("Recent Performance Trends")
        with st.expander("About This Section", expanded=True):
            st.write("""
            This section shows which stores are **improving**, **stable**, or **declining** over the last several weeks.

            While the Store Performance Categories tab shows overall long-term performance, 
            this analysis focuses specifically on recent short-term trends to help identify emerging patterns.
            """)

        # Define light theme variables for streak analysis cards
        dark_bg = "#F5F5F5"     # light gray background for cards
        light_text = "#2E2E2E"  # dark text for contrast

        # Controls for trend analysis
        col_t1, col_t2 = st.columns(2)
        with col_t1:
            trend_window = st.slider(
                "Number of recent weeks to analyze",
                min_value=3,
                max_value=8,
                value=4,
                help="Focus on more recent weeks (e.g., 4) or a longer period (e.g., 8)"
            )
        with col_t2:
            sensitivity = st.select_slider(
                "Sensitivity to small changes",
                options=["Low", "Medium", "High"],
                value="Medium",
                help="High sensitivity will detect smaller changes in performance"
            )
            if sensitivity == "Low":
                momentum_threshold = 0.5
            elif sensitivity == "High":
                momentum_threshold = 0.2
            else:  # Medium
                momentum_threshold = 0.3

        # Calculate performance direction for each store over the selected window
        store_directions = []
        for store_id, store_data in heatmap_df.groupby('StoreID'):
            if len(store_data) < trend_window:
                continue
            store_data = store_data.sort_values('Week')
            recent_data = store_data.tail(trend_window)
            # Average first half vs second half
            half_point = trend_window // 2
            if trend_window <= 3:
                first_half = recent_data.iloc[0:1]['EngagedPct'].mean()
                second_half = recent_data.iloc[-1:]['EngagedPct'].mean()
            else:
                first_half = recent_data.iloc[0:half_point]['EngagedPct'].mean()
                second_half = recent_data.iloc[-half_point:]['EngagedPct'].mean()
            change = second_half - first_half
            # Start and end values
            start_value = recent_data.iloc[0]['EngagedPct']
            current_value = recent_data.iloc[-1]['EngagedPct']
            total_change = current_value - start_value
            # Simple trend calculation for sorting
            x = np.array(range(len(recent_data)))
            y = recent_data['EngagedPct'].values
            slope, _ = np.polyfit(x, y, 1)
            # Determine direction and strength labels
            if abs(change) < momentum_threshold:
                direction = "Stable"
                strength = "Holding Steady"
                color = "#808080"  # Gray
            elif change > 0:
                direction = "Improving"
                strength = "Strong Improvement" if change > momentum_threshold * 2 else "Gradual Improvement"
                color = "#43B02A"  # Green
            else:
                direction = "Declining"
                strength = "Significant Decline" if change < -momentum_threshold * 2 else "Gradual Decline"
                color = "#C62828"  # Red
            # Indicator arrows
            if direction == "Improving":
                indicator = "↗️" if strength == "Gradual Improvement" else "🔼"
            elif direction == "Declining":
                indicator = "↘️" if strength == "Gradual Decline" else "🔽"
            else:
                indicator = "➡️"
            # Store the result
            store_directions.append({
                'store': store_id,
                'direction': direction,
                'strength': strength,
                'indicator': indicator,
                'start_value': start_value,
                'current_value': current_value,
                'total_change': total_change,
                'half_change': change,
                'color': color,
                'weeks': trend_window,
                'slope': slope
            })
        direction_df = pd.DataFrame(store_directions)
        if direction_df.empty:
            st.info("Not enough data to analyze recent trends. Try selecting a larger date range.")
        else:
            # Sort stores by direction and trend slope
            direction_order = {"Improving": 0, "Stable": 1, "Declining": 2}
            direction_df['direction_order'] = direction_df['direction'].map(direction_order)
            sorted_stores = direction_df.sort_values(['direction_order', 'slope'], ascending=[True, False])

            # Display summary metrics (counts by direction)
            col_i1, col_i2, col_i3 = st.columns(3)
            improving_count = len(direction_df[direction_df['direction'] == 'Improving'])
            stable_count = len(direction_df[direction_df['direction'] == 'Stable'])
            declining_count = len(direction_df[direction_df['direction'] == 'Declining'])
            with col_i1:
                st.metric("Improving", f"{improving_count} stores", delta="↗️", delta_color="normal")
            with col_i2:
                st.metric("Stable", f"{stable_count} stores", delta="➡️", delta_color="off")
            with col_i3:
                st.metric("Declining", f"{declining_count} stores", delta="↘️", delta_color="inverse")

            # Display each category of stores with details
            for direction in ['Improving', 'Stable', 'Declining']:
                direction_data = sorted_stores[sorted_stores['direction'] == direction]
                if direction_data.empty:
                    continue
                color = direction_data.iloc[0]['color']
                st.markdown(f"""
                <div style="
                    border-left: 5px solid {color};
                    padding-left: 10px;
                    margin-top: 20px;
                    margin-bottom: 10px;
                ">
                    <h4 style="color: {color};">{direction} ({len(direction_data)} stores)</h4>
                </div>
                """, unsafe_allow_html=True)
                cols_per_row = 3
                num_rows = (len(direction_data) + cols_per_row - 1) // cols_per_row
                for row in range(num_rows):
                    cols = st.columns(cols_per_row)
                    for i in range(cols_per_row):
                        idx = row * cols_per_row + i
                        if idx < len(direction_data):
                            store_data = direction_data.iloc[idx]
                            with cols[i]:
                                change_display = f"{store_data['total_change']:.2f}%"
                                change_sign = "+" if store_data['total_change'] > 0 else ""
                                st.markdown(f"""
                                <div style="
                                    background-color: {dark_bg};
                                    padding: 10px;
                                    border-radius: 5px;
                                    margin-bottom: 10px;
                                    border-left: 5px solid {store_data['color']};
                                ">
                                    <h4 style="text-align: center; margin: 5px 0; color: {store_data['color']};">
                                        {store_data['indicator']} Store {store_data['store']}
                                    </h4>
                                    <p style="text-align: center; margin: 5px 0; color: {light_text};">
                                        <strong>{store_data['strength']}</strong><br>
                                        <span style="font-size: 0.9em;">
                                            <strong>{change_sign}{change_display}</strong> over {store_data['weeks']} weeks
                                        </span><br>
                                        <span style="font-size: 0.85em; color: #BBBBBB;">
                                            {store_data['start_value']:.2f}% → {store_data['current_value']:.2f}%
                                        </span>
                                    </p>
                                </div>
                                """, unsafe_allow_html=True)

            # Simplified bar chart of recent changes
            st.subheader("Recent Engagement Change")
            st.write("This chart shows how much each store's engagement has changed during the selected analysis period.")
            chart_data = direction_df.copy()
            change_chart = alt.Chart(chart_data).mark_bar().encode(
                x=alt.X('total_change:Q', title='Change in Engagement % (Over Selected Weeks)'),
                y=alt.Y('store:N', title='Store', sort=alt.EncodingSortField(field='total_change', order='descending')),
                color=alt.Color('direction:N',
                                scale=alt.Scale(domain=['Improving', 'Stable', 'Declining'],
                                                range=['#43B02A', '#808080', '#C62828'])),
                tooltip=[
                    alt.Tooltip('store:N', title='Store'),
                    alt.Tooltip('direction:N', title='Direction'),
                    alt.Tooltip('strength:N', title='Performance'),
                    alt.Tooltip('start_value:Q', title='Starting Value', format='.2f'),
                    alt.Tooltip('current_value:Q', title='Current Value', format='.2f'),
                    alt.Tooltip('total_change:Q', title='Total Change', format='+.2f')
                ]
            ).properties(height=max(250, len(chart_data) * 25))
            zero_line = alt.Chart(pd.DataFrame({'x': [0]})).mark_rule(color='#BBBBBB', strokeDash=[3, 3]).encode(x='x:Q')
            final_chart = change_chart + zero_line
            st.altair_chart(final_chart, use_container_width=True)

            st.subheader("How to Use This Analysis")
            st.markdown("""
            **This section complements the Store Performance Categories tab:**

            - **Store Performance Categories** focuses on overall, longer-term store performance  
            - **Recent Performance Trends** highlights short-term movement that might not yet be reflected in the categories

            **When to take action:**

            - A "Star Performer" showing a "Declining" trend may need attention before performance drops  
            - A "Requires Intervention" store showing an "Improving" trend indicates your actions may be working  
            - Stores showing opposite trends from their category deserve the most attention
            """)

            st.subheader("Key Insights")
            insight_points = []
            # Identify stores where category and recent trend conflict
            if 'Category' in df_filtered.columns:
                category_conflict = []
                for _, store in direction_df.iterrows():
                    store_id = store['store']
                    store_cat = df_filtered[df_filtered['Store #'] == store_id]['Category'].iloc[0] if not df_filtered[df_filtered['Store #'] == store_id].empty else None
                    if store_cat == "Star Performer" and store['direction'] == "Declining":
                        category_conflict.append({
                            'store': store_id,
                            'conflict': "Star performer with recent decline",
                            'color': "#F57C00"  # Orange
                        })
                    elif store_cat == "Requires Intervention" and store['direction'] == "Improving":
                        category_conflict.append({
                            'store': store_id,
                            'conflict': "Struggling store showing improvement",
                            'color': "#43B02A"  # Green
                        })
                if category_conflict:
                    insight_points.append("**Stores with changing performance:**")
                    for conflict in category_conflict:
                        insight_points.append(f"- Store **{conflict['store']}**: {conflict['conflict']}")

            # (Additional insight calculations could go here...)

            for insight in insight_points:
                st.write(insight)

# ----------------- TAB 2: Store Comparison -----------------
with tab2:
    # (Placeholder or additional content for Store Comparison tab if needed)
    st.write("## Store Comparison")
    st.write("This tab can include comparative analyses such as side-by-side metrics, etc.")
    # (Assuming detailed content is implemented similarly with branding in mind)
    # For brevity, not fully expanded here.

# ----------------- TAB 3: Store Performance Categories -----------------
with tab3:
    # Light theme for category cards
    dark_bg = "#F5F5F5"     # light gray background for cards
    light_text = "#2E2E2E"  # dark text for contrast

    st.write("### Store Categories")
    st.write("Each store is placed into one of four categories based on their engagement level and performance trend:")

    # Category Cards (two columns)
    colA, colB = st.columns(2)
    category_cards = {
        "Star Performer": {
            "accent": "#007749",
            "icon": "⭐",
            "description": "High engagement with stable or improving trend",
            "action": "Share best practices with other stores"
        },
        "Needs Stabilization": {
            "accent": "#F57C00",
            "icon": "⚠️",
            "description": "High engagement but showing a concerning downward trend",
            "action": "Reinforce successful processes"
        },
        "Improving": {
            "accent": "#43B02A",
            "icon": "📈",
            "description": "Below average but showing positive improvement trend",
            "action": "Continue positive momentum"
        },
        "Requires Intervention": {
            "accent": "#C62828",
            "icon": "🚨",
            "description": "Below average with flat or declining trend",
            "action": "Needs comprehensive support"
        }
    }
    for cat in ["Star Performer", "Needs Stabilization"]:
        info = category_cards[cat]
        with colA:
            st.markdown(f"""
            <div style="
                background-color: {dark_bg};
                padding: 15px;
                border-radius: 5px;
                margin-bottom: 10px;
                border-left: 5px solid {info['accent']};
            ">
                <h4 style="color: {info['accent']}; margin-top: 0;">{info['icon']} {cat}</h4>
                <p style="color: {light_text}; margin: 0;">{info['description']}</p>
                <p style="color: {light_text}; margin: 0;"><strong>Action:</strong> {info['action']}</p>
            </div>
            """, unsafe_allow_html=True)
    for cat in ["Improving", "Requires Intervention"]:
        info = category_cards[cat]
        with colB:
            st.markdown(f"""
            <div style="
                background-color: {dark_bg};
                padding: 15px;
                border-radius: 5px;
                margin-bottom: 10px;
                border-left: 5px solid {info['accent']};
            ">
                <h4 style="color: {info['accent']}; margin-top: 0;">{info['icon']} {cat}</h4>
                <p style="color: {light_text}; margin: 0;">{info['description']}</p>
                <p style="color: {light_text}; margin: 0;"><strong>Action:</strong> {info['action']}</p>
            </div>
            """, unsafe_allow_html=True)

    # 4) Display Category Results (if any analysis or chart follows)
    st.subheader("Store Category Results")
    category_colors = {
        "Star Performer": "#007749",
        "Needs Stabilization": "#F57C00",
        "Improving": "#43B02A",
        "Requires Intervention": "#C62828"
    }
    # (Placeholder for any charts or outputs using category_colors, ensuring branding colors are applied)
    # e.g., bar charts comparing categories, etc.

# ----------------- TAB 4: Anomalies & Insights -----------------
with tab4:
    # (Placeholder or additional content for anomalies and insights)
    st.write("## Anomalies & Insights")
    st.write("This tab can include data anomaly detection results, insights and recommendations.")
    # (Assuming detailed content implemented with consistent branding)
    
# Sidebar footer (e.g., copyright, last updated)
now = datetime.datetime.now()
st.sidebar.markdown("---")
st.sidebar.caption(f"© Publix Super Markets, Inc. {now.year}")
st.sidebar.caption(f"Last updated: {now.strftime('%Y-%m-%d')}")
