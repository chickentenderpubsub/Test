import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from st_aggrid import AgGrid, GridOptionsBuilder
import datetime

# --------------------------------------------------------
# Theme Toggle CSS
# --------------------------------------------------------
# Define light and dark CSS styles for custom elements
light_css = """
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
    body { background-color: #FFFFFF; color: #000000; }
    .stApp { background-color: #FFFFFF; }
</style>
"""
dark_css = """
<style>
    .metric-card {
        background-color: #333333;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 2px 2px 5px rgba(255,255,255,0.1);
    }
    .highlight-good { color: #66BB6A; font-weight: bold; }
    .highlight-bad { color: #E57373; font-weight: bold; }
    .highlight-neutral { color: #FFB74D; font-weight: bold; }
    .dashboard-title { color: #90CAF9; text-align: center; padding-bottom: 20px; }
    .caption-text { font-size: 0.85em; color: #BBBBBB; }
    body, .stApp { background-color: #0E1117; color: #FFFFFF; }
</style>
"""

# --------------------------------------------------------
# Helper Functions
# --------------------------------------------------------
@st.cache_data
def load_data(uploaded_file):
    """Load CSV or Excel data and standardize columns."""
    filename = uploaded_file.name.lower()
    if filename.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
    # Standardize column names
    df.columns = standardize_columns(df.columns)
    # Parse dates if present
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    # Convert percentage columns from strings to numeric
    percent_cols = ['Engaged Transaction %', 'Quarter to Date %']
    for col in percent_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace('%', ''), errors='coerce')
    # Drop rows without engagement data
    df = df.dropna(subset=['Engaged Transaction %'])
    # Convert Weekly Rank to integer if exists
    if 'Weekly Rank' in df.columns:
        df['Weekly Rank'] = pd.to_numeric(df['Weekly Rank'], errors='coerce').astype('Int64')
    # Ensure Store # is string
    if 'Store #' in df.columns:
        df['Store #'] = df['Store #'].astype(str)
    # Ensure Week is int and sort by Week, Store
    if 'Week' in df.columns:
        df['Week'] = pd.to_numeric(df['Week'], errors='coerce').astype(int)
        df = df.sort_values(['Week', 'Store #'])
    return df

def standardize_columns(columns):
    """Rename columns to standard names (Store #, Week, Engaged Transaction %, etc.)."""
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

def calculate_trend_label(group, window=4):
    """
    Calculate a trend label (Upward, Downward, Stable, etc.) based on linear slope 
    of last `window` points of Engaged Transaction %.
    """
    if len(group) < 2:
        return "Stable"
    sorted_data = group.sort_values('Week').tail(window)
    if len(sorted_data) < 2:
        return "Stable"
    x = sorted_data['Week'].values.astype(float)
    y = sorted_data['Engaged Transaction %'].values.astype(float)
    # Center x for numeric stability
    x -= x.mean()
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
    Identify week-over-week engagement changes with Z-score beyond the threshold.
    Returns a DataFrame of anomalies with possible explanations.
    """
    anomalies_list = []
    for store_id, grp in df.groupby('Store #'):
        grp = grp.sort_values('Week')
        changes = grp['Engaged Transaction %'].diff().dropna()
        if changes.empty or changes.std(ddof=0) == 0:
            continue
        mean_diff = changes.mean()
        std_diff = changes.std(ddof=0)
        for idx, diff_val in changes.iteritems():
            z = (diff_val - mean_diff) / std_diff
            if abs(z) >= z_threshold:
                week_cur = int(grp.loc[idx, 'Week'])
                prev_idx = grp.index[grp.index.get_indexer([idx]) - 1][0] if grp.index.get_indexer([idx])[0] - 1 >= 0 else None
                week_prev = int(grp.loc[prev_idx, 'Week']) if prev_idx is not None else None
                val_cur = grp.loc[idx, 'Engaged Transaction %']
                rank_cur = grp.loc[idx, 'Weekly Rank'] if 'Weekly Rank' in grp.columns else None
                rank_prev = grp.loc[prev_idx, 'Weekly Rank'] if prev_idx is not None and 'Weekly Rank' in grp.columns else None
                anomalies_list.append({
                    'Store #': store_id,
                    'Week': week_cur,
                    'Engaged Transaction %': round(val_cur, 2),
                    'Change %pts': round(diff_val, 2),
                    'Z-score': round(z, 2),
                    'Prev Week': week_prev,
                    'Prev Rank': int(rank_prev) if pd.notna(rank_prev) else None,
                    'Rank': int(rank_cur) if pd.notna(rank_cur) else None
                })
    anomalies_df = pd.DataFrame(anomalies_list)
    if not anomalies_df.empty:
        # Add a quick explanation for each anomaly
        explanations = []
        for _, row in anomalies_df.iterrows():
            if row['Change %pts'] >= 0:
                reason = "Engagement spiked significantly. Possible promotion or event impact."
                if row['Prev Rank'] and row['Rank'] and row['Prev Rank'] > row['Rank']:
                    reason += f" (Improved from rank {int(row['Prev Rank'])} to {int(row['Rank'])}.)"
            else:
                reason = "Sharp drop in engagement. Potential system issue or loss of engagement."
                if row['Prev Rank'] and row['Rank'] and row['Prev Rank'] < row['Rank']:
                    reason += f" (Dropped from rank {int(row['Prev Rank'])} to {int(row['Rank'])}.)"
            explanations.append(reason)
        anomalies_df['Possible Explanation'] = explanations
    return anomalies_df

# --------------------------------------------------------
# Page Configuration & Theme Selection
# --------------------------------------------------------
st.set_page_config(page_title="Publix District 20 Engagement Dashboard", layout="wide", initial_sidebar_state="expanded")

# Sidebar Role Selection
st.sidebar.header("User Role")
role = st.sidebar.selectbox("Select Role", ["District Manager", "Store Manager", "Assistant Customer Service Manager"], index=0)

# Apply theme CSS based on selection
theme_choice = st.sidebar.radio("Theme", ["Light", "Dark"], index=0)
if theme_choice == "Dark":
    st.markdown(dark_css, unsafe_allow_html=True)
else:
    st.markdown(light_css, unsafe_allow_html=True)

# Title & description
st.markdown(f"<h1 class='dashboard-title'>Publix District 20 Engagement Dashboard</h1>", unsafe_allow_html=True)
st.markdown("**Publix Supermarkets – District 20** engagement analysis dashboard. Upload weekly engagement data to explore key performance indicators, trends, and opportunities across 10 stores. Use the filters on the left to drill down by time period or store.")

# Sidebar Data Input
st.sidebar.header("Data Input")
data_file = st.sidebar.file_uploader("Upload engagement data (Excel or CSV)", type=['csv', 'xlsx'])
comp_file = st.sidebar.file_uploader("Optional: Upload comparison data (prior period)", type=['csv', 'xlsx'])

# If no data file, prompt user and stop
if not data_file:
    st.info("Please upload a primary engagement data file to begin.")
    st.markdown("### Expected Data Format")
    st.markdown("""
    Your data file should contain the following columns:
    - Store # or Store ID
    - Week or Date
    - Engaged Transaction % (the main KPI)
    - Optional: Weekly Rank, Quarter to Date %, etc.
    \nExample formats supported:
    - CSV with headers
    - Excel file with data in the first sheet
    """)
    st.stop()

# Load data
df = load_data(data_file)
df_comp = load_data(comp_file) if comp_file else None

# Derive Quarter from Date or Week for both datasets
if 'Date' in df.columns:
    df['Quarter'] = df['Date'].dt.quarter
elif 'Week' in df.columns:
    df['Quarter'] = ((df['Week'] - 1) // 13 + 1).astype(int)
if df_comp is not None:
    if 'Date' in df_comp.columns:
        df_comp['Quarter'] = df_comp['Date'].dt.quarter
    elif 'Week' in df_comp.columns:
        df_comp['Quarter'] = ((df_comp['Week'] - 1) // 13 + 1).astype(int)

# Sidebar Filters for time period and store
st.sidebar.header("Filters")
quarters = sorted(df['Quarter'].dropna().unique().tolist())
quarter_options = ["All"] + [f"Q{int(q)}" for q in quarters]
quarter_choice = st.sidebar.selectbox("Select Quarter", quarter_options, index=0)
if quarter_choice != "All":
    q_num = int(quarter_choice[1:])
    available_weeks = sorted(df[df['Quarter'] == q_num]['Week'].unique().tolist())
else:
    available_weeks = sorted(df['Week'].unique().tolist())
week_options = ["All"] + [str(int(w)) for w in available_weeks]
week_choice = st.sidebar.selectbox("Select Week", week_options, index=0)
store_list = sorted(df['Store #'].unique().tolist())
store_choice = st.sidebar.multiselect("Select Store(s)", store_list, default=[] if role == "District Manager" else None)

# Advanced settings (anomaly threshold, moving average toggle) in expander
with st.sidebar.expander("Advanced Settings", expanded=False):
    z_threshold = st.slider("Anomaly Z-score Threshold", 1.0, 3.0, 2.0, 0.1)
    show_ma = st.checkbox("Show 4-week moving average", value=True)
    highlight_top = st.checkbox("Highlight top performer", value=True)
    highlight_bottom = st.checkbox("Highlight bottom performer", value=True)
    trend_analysis_weeks = st.slider("Trend analysis window (weeks) for trend labels", 3, 8, 4)
    st.caption("Adjust sensitivity for anomaly detection and trend analysis.")

# Filter main DataFrame (df) by selected quarter, week, and store(s)
df_base = df.copy()  # base data filtered only by time (we will use for comparisons/rank if needed)
if quarter_choice != "All":
    df_base = df_base[df_base['Quarter'] == q_num]
if week_choice != "All":
    week_num = int(week_choice)
    df_base = df_base[df_base['Week'] == week_num]
# Now apply store filter to get df_filtered for main content
df_filtered = df_base.copy()
if store_choice:
    df_filtered = df_filtered[df_filtered['Store #'].isin([str(s) for s in store_choice])]

# Filter comparison data similarly
df_comp_filtered = None
if df_comp is not None:
    df_comp_filtered = df_comp.copy()
    if quarter_choice != "All":
        df_comp_filtered = df_comp_filtered[df_comp_filtered['Quarter'] == q_num]
    if week_choice != "All":
        df_comp_filtered = df_comp_filtered[df_comp_filtered['Week'] == (week_num if 'week_num' in locals() else None)]
    if store_choice:
        df_comp_filtered = df_comp_filtered[df_comp_filtered['Store #'].isin([str(s) for s in store_choice])]

# If no data after filters, show error and stop
if df_filtered.empty:
    st.error("No data available for the selected filters. Please adjust your filters.")
    st.stop()

# Identify current and previous week for context
if week_choice != "All":
    current_week = int(week_choice)
    prev_week = current_week - 1
    if prev_week not in df_base['Week'].values:
        prev_df = df_base[df_base['Week'] < current_week]
        prev_week = int(prev_df['Week'].max()) if not prev_df.empty else None
else:
    current_week = int(df_base['Week'].max())
    prev_candidates = df_base['Week'][df_base['Week'] < current_week]
    prev_week = int(prev_candidates.max()) if not prev_candidates.empty else None

# Compute district (or selection) average engagement for current and previous week
current_avg = df_base[df_base['Week'] == current_week]['Engaged Transaction %'].mean() if current_week else None
prev_avg = df_base[df_base['Week'] == prev_week]['Engaged Transaction %'].mean() if prev_week else None

# Top and bottom performer over the filtered period (df_base scope, which is all stores filtered by time)
store_perf_all = df_base.groupby('Store #')['Engaged Transaction %'].mean()
top_store = store_perf_all.idxmax()
bottom_store = store_perf_all.idxmin()
top_val = store_perf_all.max()
bottom_val = store_perf_all.min()

# Calculate trend labels for each store over the specified window (trend_analysis_weeks)
store_trends = df_base.groupby('Store #').apply(lambda x: calculate_trend_label(x, trend_analysis_weeks))

# Executive Summary Metrics Display (varies by role)
st.subheader("Executive Summary")
if role == "District Manager":
    # Three metrics: District Avg Engagement, Top Performer, Bottom Performer for current week
    col1, col2, col3 = st.columns(3)
    avg_label = "District Avg Engagement" if not store_choice or len(store_choice) == len(store_list) or len(store_choice) == 0 else ("Selected Stores Avg Engagement" if len(store_choice) > 1 else f"Store {store_choice[0]} Engagement")
    avg_display = f"{current_avg:.2f}%" if current_avg is not None else "N/A"
    delta_str = f"{(current_avg - prev_avg):+.2f}%" if current_avg is not None and prev_avg is not None else "N/A"
    col1.metric(f"{avg_label} (Week {current_week})", avg_display, delta_str)
    col2.metric(f"Top Performer (Week {current_week})", f"Store {top_store} — {top_val:.2f}%")
    col3.metric(f"Bottom Performer (Week {current_week})", f"Store {bottom_store} — {bottom_val:.2f}%")
elif role in ["Store Manager", "Assistant Customer Service Manager"]:
    # For Store/ACSM: ensure a single store is selected for meaningful stats
    if not store_choice or len(store_choice) != 1:
        st.warning("Please select exactly one store to view store-specific dashboard.")
        st.stop()
    selected_store = str(store_choice[0])
    # Calculate this store's current week value and rank vs district
    store_current_val = df_base[(df_base['Store #'] == selected_store) & (df_base['Week'] == current_week)]['Engaged Transaction %'].mean()
    # Determine rank among all stores for current week
    if 'Weekly Rank' in df_base.columns and not df_base[df_base['Store #'] == selected_store].empty:
        # Use provided Weekly Rank if available (take rank at current_week if exists)
        rank_val = df_base[(df_base['Store #'] == selected_store) & (df_base['Week'] == current_week)]['Weekly Rank']
        rank_val = int(rank_val.iloc[0]) if not rank_val.empty else None
    else:
        # Compute rank manually for current week
        week_df = df_base[df_base['Week'] == current_week]
        week_df = week_df.groupby('Store #', as_index=False)['Engaged Transaction %'].mean()  # average in case multiple entries per store-week
        week_df = week_df.sort_values('Engaged Transaction %', ascending=False).reset_index(drop=True)
        rank_val = week_df.index[week_df['Store #'] == selected_store].tolist()
        rank_val = rank_val[0] + 1 if rank_val else None
    # Compute gap to top performer for current week
    gap_to_top = None
    if store_current_val is not None and not np.isnan(store_current_val):
        gap_to_top = top_val - store_current_val  # positive means how many points behind the top
    # Display metrics: Store Engagement (with delta), Rank, Gap to Top
    col1, col2, col3 = st.columns(3)
    val_display = f"{store_current_val:.2f}%" if store_current_val is not None and not np.isnan(store_current_val) else "N/A"
    delta_str = f"{(store_current_val - prev_avg):+.2f}%" if store_current_val is not None and prev_avg is not None else "N/A"
    col1.metric(f"Store {selected_store} Engagement (Week {current_week})", val_display, delta_str)
    rank_text = f"{rank_val} / {store_perf_all.count()}" if rank_val else "N/A"
    col2.metric(f"Rank in District (Week {current_week})", rank_text)
    if gap_to_top is not None:
        col3.metric(f"Gap to Top (Week {current_week})", f"{gap_to_top:.2f}%")
    else:
        col3.metric(f"Gap to Top (Week {current_week})", "N/A")

# Generate a few high-level insights for District Manager view
if role == "District Manager":
    insights = []
    # Consistency analysis: find store with highest engagement variability (std)
    store_consistency = df_base.groupby('Store #')['Engaged Transaction %'].std().fillna(0)
    least_consistent = store_consistency.idxmax()
    if store_consistency.max() > 0:
        insights.append(f"**Store {least_consistent}** has the most variable engagement performance.")
    # Trend analysis: stores trending up or down
    trending_up = store_trends[store_trends.isin(["Upward", "Strong Upward"])].index.tolist()
    trending_down = store_trends[store_trends.isin(["Downward", "Strong Downward"])].index.tolist()
    if trending_up:
        insights.append("Stores showing positive trends: " + ", ".join([f"**{s}**" for s in trending_up]))
    if trending_down:
        insights.append("Stores needing attention: " + ", ".join([f"**{s}**" for s in trending_down]))
    # Performance gap analysis
    if store_perf_all.size > 1:
        engagement_gap = top_val - bottom_val
        insights.append(f"Gap between highest and lowest performing stores: **{engagement_gap:.2f}%**")
        if engagement_gap > 10:
            insights.append("🚨 Large performance gap indicates opportunity for knowledge sharing.")
    # Display top 5 insights as a list
    if insights:
        st.markdown("**Key Insights:**")
        for i, insight in enumerate(insights[:5], start=1):
            st.markdown(f"{i}. {insight}")

# Role-specific main content
if role == "District Manager":
    # Use tabs for detailed analysis sections
    tab1, tab2, tab3, tab4 = st.tabs(["Engagement Trends", "Store Comparison", "Store Performance Categories", "Anomalies & Insights"])

    # ----------------- TAB 1: Engagement Trends -----------------
    with tab1:
        st.subheader("Engagement Trends Over Time")
        # Radio to choose view mode for trends
        view_option = st.radio("View mode:", ["All Stores", "Custom Selection", "Recent Trends"], horizontal=True,
                               help="All Stores: View all stores at once | Custom Selection: Pick specific stores to compare | Recent Trends: Focus on recent weeks")
        # Compute district average trend over time (current period)
        dist_trend = df_filtered.groupby('Week', as_index=False)['Engaged Transaction %'].mean().sort_values('Week')
        dist_trend.rename(columns={'Engaged Transaction %': 'Average Engagement %'}, inplace=True)
        dist_trend['MA_4W'] = dist_trend['Average Engagement %'].rolling(window=4, min_periods=1).mean()
        # Compute 4-week moving average for each store in current period data
        df_filtered = df_filtered.sort_values(['Store #', 'Week'])
        df_filtered['MA_4W'] = df_filtered.groupby('Store #')['Engaged Transaction %'].transform(lambda s: s.rolling(window=4, min_periods=1).mean())
        # Combine current and comparison data for plotting (to overlay if needed)
        combined = df_filtered.copy()
        combined['Period'] = 'Current'
        if df_comp_filtered is not None and not df_comp_filtered.empty:
            df_comp_filtered['Period'] = 'Comparison'
            combined = pd.concat([combined, df_comp_filtered], ignore_index=True)
            combined = combined.sort_values(['Store #', 'Period', 'Week'])
            # Recompute 4W MA in combined context (separately for current vs comparison periods)
            combined['MA_4W'] = combined.groupby(['Store #', 'Period'])['Engaged Transaction %'].transform(lambda s: s.rolling(window=4, min_periods=1).mean())
        # If "Recent Trends" view, allow selecting a range of recent weeks via slider
        if view_option == "Recent Trends":
            all_weeks = sorted(combined['Week'].unique().tolist())
            if len(all_weeks) <= 8:
                start_week = all_weeks[0]
            else:
                start_week = all_weeks[-8]  # default start as 8th from last
            end_week = all_weeks[-1]
            recent_weeks_range = st.select_slider("Select weeks to display:", options=all_weeks, value=(start_week, end_week),
                                                  help="Adjust to show more or fewer weeks in the trend view")
            recent_weeks = [w for w in all_weeks if w >= recent_weeks_range[0] and w <= recent_weeks_range[1]]
            combined_view = combined[combined['Week'].isin(recent_weeks)]
            dist_trend_view = dist_trend[dist_trend['Week'].isin(recent_weeks)]
        else:
            combined_view = combined
            dist_trend_view = dist_trend
        # Prepare Plotly line chart
        fig_trend = go.Figure()
        if view_option == "Custom Selection":
            # Let user pick specific stores to compare
            store_options = sorted(df_filtered['Store #'].unique().tolist())
            selected_stores = st.multiselect("Select stores to compare:", options=store_options,
                                             default=[store_options[0]] if store_options else [],
                                             help="Choose specific stores to highlight in the chart")
            if selected_stores:
                for store in selected_stores:
                    store_data = combined_view[combined_view['Store #'] == store]
                    # Engaged % line + markers
                    fig_trend.add_trace(go.Scatter(x=store_data['Week'], y=store_data['Engaged Transaction %'],
                                                   mode='lines+markers', name=f"Store {store}",
                                                   legendgroup=f"Store {store}", showlegend=True,
                                                   marker=dict(size=8), line=dict(width=3)))
                    # Optional moving average dashed line
                    if show_ma:
                        fig_trend.add_trace(go.Scatter(x=store_data['Week'], y=store_data['MA_4W'],
                                                       mode='lines', name=f"MA_{store}", legendgroup=f"Store {store}",
                                                       showlegend=False, line=dict(width=2, dash='dash')))
            else:
                st.info("Please select at least one store to display.")
            # District average for current period (dashed black line)
            if not dist_trend_view.empty:
                fig_trend.add_trace(go.Scatter(x=dist_trend_view['Week'], y=dist_trend_view['Average Engagement %'],
                                               mode='lines', name='District Avg', legendgroup='District Avg', showlegend=True,
                                               line=dict(color='black', width=3, dash='dash')))
                if show_ma:
                    fig_trend.add_trace(go.Scatter(x=dist_trend_view['Week'], y=dist_trend_view['MA_4W'],
                                                   mode='lines', name='District MA', legendgroup='District Avg', showlegend=False,
                                                   line=dict(color='black', width=2, dash='dot')))
            # Previous period district average (if available, grey dashed)
            if df_comp_filtered is not None and not df_comp_filtered.empty:
                # Compute district avg for comp period (for overlay)
                dist_comp = df_comp_filtered.groupby('Week', as_index=False)['Engaged Transaction %'].mean().sort_values('Week')
                fig_trend.add_trace(go.Scatter(x=dist_comp['Week'], y=dist_comp['Engaged Transaction %'],
                                               mode='lines', name='Prev Avg', legendgroup='Prev Avg', showlegend=True,
                                               line=dict(color='gray', width=2, dash='dash')))
                if show_ma:
                    dist_comp['MA_4W'] = dist_comp['Engaged Transaction %'].rolling(window=4, min_periods=1).mean()
                    fig_trend.add_trace(go.Scatter(x=dist_comp['Week'], y=dist_comp['MA_4W'],
                                                   mode='lines', name='Prev MA', legendgroup='Prev Avg', showlegend=False,
                                                   line=dict(color='gray', width=1.5, dash='dot')))
        else:
            # All Stores or Recent Trends: plot all stores lines with interactive legend toggling
            # One trace per store for Engaged %
            for store, group in combined_view.groupby('Store #'):
                fig_trend.add_trace(go.Scatter(x=group['Week'], y=group['Engaged Transaction %'],
                                               mode='lines', name=f"Store {store}",
                                               legendgroup=f"Store {store}", showlegend=True,
                                               line=dict(width=1.5)))
                if show_ma:
                    fig_trend.add_trace(go.Scatter(x=group['Week'], y=group['MA_4W'],
                                                   mode='lines', name=f"MA_{store}",
                                                   legendgroup=f"Store {store}", showlegend=False,
                                                   line=dict(width=1.5, dash='dash')))
            # District average (current period) as bold black dashed line
            if not dist_trend_view.empty:
                fig_trend.add_trace(go.Scatter(x=dist_trend_view['Week'], y=dist_trend_view['Average Engagement %'],
                                               mode='lines', name='District Avg',
                                               line=dict(color='black', width=3, dash='dash')))
                if show_ma:
                    fig_trend.add_trace(go.Scatter(x=dist_trend_view['Week'], y=dist_trend_view['MA_4W'],
                                                   mode='lines', name='District MA',
                                                   line=dict(color='black', width=2, dash='dot'), showlegend=False))
            # Previous period district average (grey dashed line)
            if df_comp_filtered is not None and not df_comp_filtered.empty:
                dist_comp = df_comp_filtered.groupby('Week', as_index=False)['Engaged Transaction %'].mean().sort_values('Week')
                fig_trend.add_trace(go.Scatter(x=dist_comp['Week'], y=dist_comp['Engaged Transaction %'],
                                               mode='lines', name='Prev Avg',
                                               line=dict(color='gray', width=2, dash='dash')))
                if show_ma:
                    dist_comp['MA_4W'] = dist_comp['Engaged Transaction %'].rolling(window=4, min_periods=1).mean()
                    fig_trend.add_trace(go.Scatter(x=dist_comp['Week'], y=dist_comp['MA_4W'],
                                                   mode='lines', name='Prev MA',
                                                   line=dict(color='gray', width=1.5, dash='dot'), showlegend=False))
        # Update layout for readability
        fig_trend.update_layout(height=400, margin=dict(l=20, r=20, t=30, b=20), legend_title_text='Store', legend=dict(itemsizing='constant'))
        # Show the chart
        st.plotly_chart(fig_trend, use_container_width=True)
        # Additional metrics for Recent Trends view
        if view_option == "Recent Trends":
            colA, colB = st.columns(2)
            with colA:
                # District week-over-week trend metric
                if len(dist_trend_view) >= 2:
                    last_two = dist_trend_view.sort_values('Week').tail(2)
                    cur_val = last_two['Average Engagement %'].iloc[-1]
                    prev_val = last_two['Average Engagement %'].iloc[0]
                    change_pct = (cur_val - prev_val) / prev_val * 100 if prev_val != 0 else 0
                    st.metric("District Trend (Week-over-Week)", f"{cur_val:.2f}%", f"{change_pct:.1f}%", delta_color="normal")
            with colB:
                # Top performer of the last week in view
                last_week_num = combined_view['Week'].max()
                last_week_data = combined_view[combined_view['Week'] == last_week_num]
                if not last_week_data.empty:
                    top_entry = last_week_data.loc[last_week_data['Engaged Transaction %'].idxmax()]
                    st.metric(f"Top Performer (Week {int(last_week_num)})", f"Store {top_entry['Store #']}", f"{top_entry['Engaged Transaction %']:.2f}%", delta_color="off")
        # Caption describing view mode and line styles
        caption = ""
        if view_option == "All Stores":
            caption = "**All Stores View:** Shows all store trends. The black dashed line is the district average."
        elif view_option == "Custom Selection":
            caption = "**Custom Selection View:** Only selected stores are shown with emphasized lines and markers."
        else:
            caption = "**Recent Trends View:** Focuses on recent weeks with added week-over-week metrics."
        if df_comp_filtered is not None and not df_comp_filtered.empty:
            caption += " The gray dashed line represents the previous period's district average."
        st.caption(caption)

    # ----------------- TAB 2: Store Comparison -----------------
    with tab2:
        st.subheader("Weekly Engagement Heatmap")
        # Heatmap settings in expander
        with st.container():
            with st.expander("Heatmap Settings", expanded=False):
                col1, col2 = st.columns([1, 1])
                with col1:
                    sort_method = st.selectbox("Sort stores by:", ["Average Engagement", "Recent Performance"], index=0,
                                               help="Order the heatmap by overall average or by the latest week performance")
                with col2:
                    color_scheme = st.selectbox("Color scheme:", ["Blues", "Greens", "Purples", "Oranges", "Reds", "Viridis"], index=0,
                                                help="Color palette for the heatmap")
                    normalize_colors = st.checkbox("Normalize colors by week", value=False,
                                                   help="If checked, color intensity is relative within each week (to highlight weekly outliers)")
            # Week range slider outside expander for easy access
            weeks_list = sorted(df_filtered['Week'].unique().tolist())
            if len(weeks_list) > 4:
                selected_weeks = st.select_slider("Select week range for heatmap:", options=weeks_list,
                                                  value=(min(weeks_list), max(weeks_list)))
                heatmap_data = df_filtered[(df_filtered['Week'] >= selected_weeks[0]) & (df_filtered['Week'] <= selected_weeks[1])].copy()
            else:
                heatmap_data = df_filtered.copy()
        # Prepare data for heatmap
        hm_df = heatmap_data.rename(columns={'Store #': 'StoreID', 'Engaged Transaction %': 'EngagedPct'}).copy()
        # Determine store order based on sorting method
        if sort_method == "Average Engagement":
            store_avg = hm_df.groupby('StoreID')['EngagedPct'].mean().reset_index()
            store_order = store_avg.sort_values('EngagedPct', ascending=False)['StoreID'].tolist()
        else:  # Recent Performance (use last week available in hm_df)
            most_recent_week = hm_df['Week'].max()
            recent_perf = hm_df[hm_df['Week'] == most_recent_week]
            store_order = recent_perf.sort_values('EngagedPct', ascending=False)['StoreID'].tolist()
        # Normalize per-week if selected
        color_field = 'EngagedPct'
        color_title = 'Engaged %'
        if normalize_colors:
            week_stats = hm_df.groupby('Week')['EngagedPct'].agg(['min', 'max']).reset_index()
            hm_df = hm_df.merge(week_stats, on='Week')
            # Compute normalized percentage (0-100) for each value within its week range
            hm_df['Norm'] = hm_df.apply(lambda row: 0 if row['min'] == row['max'] else 100 * (row['EngagedPct'] - row['min']) / (row['max'] - row['min']), axis=1)
            color_field = 'Norm'
            color_title = 'Normalized %'
        # Pivot data to matrix form for heatmap
        pivot_df = hm_df.pivot(index='StoreID', columns='Week', values=color_field)
        # Reorder rows (StoreID) according to chosen sort order
        pivot_df = pivot_df.reindex(store_order)
        # Create heatmap figure
        fig_heatmap = go.Figure(data=go.Heatmap(
            z=pivot_df.values,
            x=pivot_df.columns.astype(str),
            y=pivot_df.index,
            colorscale=color_scheme,
            colorbar=dict(title=color_title)
        ))
        # Adjust layout (reverse Y-axis to have first in list at top)
        fig_heatmap.update_yaxes(autorange="reversed")
        fig_heatmap.update_layout(height=max(250, 20 * len(store_order)), margin=dict(l=20, r=100, t=20, b=20))
        st.plotly_chart(fig_heatmap, use_container_width=True)
        # Caption for heatmap details
        min_week = heatmap_data['Week'].min() if not heatmap_data.empty else None
        max_week = heatmap_data['Week'].max() if not heatmap_data.empty else None
        norm_note = "Colors normalized within each week." if normalize_colors else "Global color scale across all weeks."
        if min_week is not None and max_week is not None:
            st.caption(f"**Heatmap Details:** Showing engagement from Week {int(min_week)} to Week {int(max_week)}. Stores sorted by {sort_method.lower()}. {norm_note} Darker colors represent higher engagement.")
        # Recent Performance Trends (Streak Analysis)
        st.subheader("Recent Performance Trends")
        with st.expander("About This Section", expanded=True):
            st.write("""
            This section shows which stores are **improving**, **stable**, or **declining** over the last several weeks.
            While the Store Performance Categories tab shows long-term performance, this analysis focuses on recent short-term trends to identify emerging patterns.
            """)
        # Controls for streak analysis
        colA, colB = st.columns(2)
        with colA:
            trend_window = st.slider("Number of recent weeks to analyze", min_value=3, max_value=8, value=4,
                                     help="Choose a window of recent weeks (e.g., 4) to assess short-term trends")
        with colB:
            sensitivity = st.select_slider("Sensitivity to small changes", options=["Low", "Medium", "High"], value="Medium",
                                           help="Higher sensitivity detects smaller changes; lower sensitivity focuses on larger swings")
            if sensitivity == "Low":
                momentum_threshold = 0.5
            elif sensitivity == "High":
                momentum_threshold = 0.2
            else:
                momentum_threshold = 0.3
        # Calculate recent trend direction for each store
        store_directions = []
        for store_id, data in heatmap_data.groupby('Store #'):
            # Use heatmap_data (which is within selected week range) for consistency
            data = data.sort_values('Week')
            if len(data) < trend_window:
                continue
            recent_data = data.tail(trend_window)
            half = trend_window // 2
            if trend_window <= 3:
                first_half = recent_data.iloc[0:1]['Engaged Transaction %'].mean()
                second_half = recent_data.iloc[-1:]['Engaged Transaction %'].mean()
            else:
                first_half = recent_data.iloc[0:half]['Engaged Transaction %'].mean()
                second_half = recent_data.iloc[-half:]['Engaged Transaction %'].mean()
            change = second_half - first_half
            start_val = recent_data.iloc[0]['Engaged Transaction %']
            end_val = recent_data.iloc[-1]['Engaged Transaction %']
            total_change = end_val - start_val
            # Simple linear trend (slope) for additional sorting
            if len(recent_data) >= 2:
                x = np.arange(len(recent_data))
                y = recent_data['Engaged Transaction %'].values
                slope, _ = np.polyfit(x, y, 1)
            else:
                slope = 0
            if abs(change) < momentum_threshold:
                direction = "Stable"; strength = "Holding Steady"; color = "#1976D2"
            elif change > 0:
                direction = "Improving"; strength = "Strong Improvement" if change > momentum_threshold*2 else "Gradual Improvement"; color = "#2E7D32"
            else:
                direction = "Declining"; strength = "Significant Decline" if change < -momentum_threshold*2 else "Gradual Decline"; color = "#C62828"
            indicator = "➡️"
            if direction == "Improving":
                indicator = "🔼" if "Strong" in strength else "↗️"
            elif direction == "Declining":
                indicator = "🔽" if "Significant" in strength else "↘️"
            store_directions.append({
                'Store': store_id, 'direction': direction, 'strength': strength, 'indicator': indicator,
                'start_val': start_val, 'end_val': end_val, 'total_change': total_change,
                'color': color, 'weeks': trend_window, 'slope': slope
            })
        direction_df = pd.DataFrame(store_directions)
        if direction_df.empty:
            st.info("Not enough data to analyze recent trends. Try a different date range or a smaller analysis window.")
        else:
            # Sort stores by direction category (Improving, Stable, Declining) and slope (within category)
            direction_order = {"Improving": 0, "Stable": 1, "Declining": 2}
            direction_df['order'] = direction_df['direction'].map(direction_order)
            sorted_df = direction_df.sort_values(['order', 'slope'], ascending=[True, False])
            # Summary metrics: count of improving/stable/declining stores
            col1, col2, col3 = st.columns(3)
            improving_count = (direction_df['direction'] == 'Improving').sum()
            stable_count = (direction_df['direction'] == 'Stable').sum()
            declining_count = (direction_df['direction'] == 'Declining').sum()
            col1.metric("Improving", f"{improving_count} stores", delta="↗️", delta_color="normal")
            col2.metric("Stable", f"{stable_count} stores", delta="➡️", delta_color="off")
            col3.metric("Declining", f"{declining_count} stores", delta="↘️", delta_color="inverse")
            # Detailed listing per category
            for direction in ["Improving", "Stable", "Declining"]:
                group_data = sorted_df[sorted_df['direction'] == direction]
                if group_data.empty:
                    continue
                group_color = group_data.iloc[0]['color']
                st.markdown(f"""<div style="border-left: 5px solid {group_color}; padding-left: 10px; margin-top: 20px; margin-bottom: 10px;">
                                <h4 style="color: {group_color};">{direction} ({len(group_data)} stores)</h4>
                                </div>""", unsafe_allow_html=True)
                cols_per_row = 3
                rows = (len(group_data) + cols_per_row - 1) // cols_per_row
                for r in range(rows):
                    colset = st.columns(cols_per_row)
                    for i in range(cols_per_row):
                        idx = r * cols_per_row + i
                        if idx < len(group_data):
                            st_info = group_data.iloc[idx]
                            with colset[i]:
                                change_display = f"{st_info['total_change']:.2f}%"
                                sign = "+" if st_info['total_change'] > 0 else ""
                                st.markdown(f"""<div style="background-color: #2C2C2C; padding: 10px; border-radius: 5px; border-left: 5px solid {st_info['color']}; margin-bottom: 10px;">
                                                <h4 style="text-align: center; color: {st_info['color']}; margin: 5px 0;">{st_info['indicator']} Store {st_info['Store']}</h4>
                                                <p style="text-align: center; color: #FFFFFF; margin: 5px 0;">
                                                    <strong>{st_info['strength']}</strong><br/>
                                                    <span style="font-size: 0.9em;"><strong>{sign}{change_display}</strong> over {st_info['weeks']} weeks</span><br/>
                                                    <span style="font-size: 0.85em; color: #BBBBBB;">{st_info['start_val']:.2f}% → {st_info['end_val']:.2f}%</span>
                                                </p>
                                                </div>""", unsafe_allow_html=True)
            # Bar chart of total change for each store
            st.subheader("Recent Engagement Change")
            st.write("This chart shows how much each store's engagement has changed during the selected analysis period.")
            change_chart_data = direction_df.copy()
            change_chart_data = change_chart_data.sort_values('total_change', ascending=False)
            fig_change = px.bar(change_chart_data, x='total_change', y='Store', color='direction',
                                 color_discrete_map={'Improving': '#2E7D32', 'Stable': '#1976D2', 'Declining': '#C62828'},
                                 labels={'total_change': 'Change in Engagement %', 'Store': 'Store'}, orientation='h')
            fig_change.update_yaxes(autorange="reversed")  # largest change at top
            fig_change.add_vline(x=0, line_width=1, line_dash='dash', line_color='#FFFFFF')
            fig_change.update_layout(height=max(250, 25 * len(change_chart_data)), margin=dict(l=100, r=20, t=20, b=20))
            st.plotly_chart(fig_change, use_container_width=True)
            st.subheader("How to Use This Analysis")
            st.markdown("""
            **This section complements the Store Performance Categories tab:**
            - **Store Performance Categories** focuses on overall, longer-term performance vs. district.
            - **Recent Performance Trends** highlights short-term momentum or issues.
            
            Use both views together: for example, a store might be classified as "Needs Stabilization" overall, but this section can show if it's currently trending back up or continuing to decline, informing where to prioritize immediate attention.
            """)

    # ----------------- TAB 3: Store Performance Categories -----------------
    with tab3:
        st.subheader("Store Performance Categories")
        # Calculate average engagement and consistency (std) for each store (using df_base which is all stores in filtered period)
        store_stats = df_base.groupby('Store #')['Engaged Transaction %'].agg(['mean', 'std']).reset_index()
        store_stats.columns = ['Store #', 'Average Engagement', 'Consistency']
        store_stats['Consistency'] = store_stats['Consistency'].fillna(0.0)
        # Calculate trend correlation for each store (Week vs Engagement)
        trend_data = []
        for store_id, grp in df_base.groupby('Store #'):
            if len(grp) >= 3:
                corr_val = grp[['Week', 'Engaged Transaction %']].corr().iloc[0, 1]
                trend_data.append({'Store #': store_id, 'Trend Correlation': corr_val})
        trend_df = pd.DataFrame(trend_data)
        if not trend_df.empty:
            store_stats = store_stats.merge(trend_df, on='Store #', how='left')
        else:
            store_stats['Trend Correlation'] = 0
        store_stats['Trend Correlation'] = store_stats['Trend Correlation'].fillna(0.0)
        # Determine median engagement for category threshold
        med_eng = store_stats['Average Engagement'].median() if not store_stats.empty else 0
        # Categorize each store
        def assign_category(row):
            has_positive_trend = row['Trend Correlation'] > 0.1
            has_negative_trend = row['Trend Correlation'] < -0.1
            if row['Average Engagement'] >= med_eng:
                return "Needs Stabilization" if has_negative_trend else "Star Performer"
            else:
                return "Improving" if has_positive_trend else "Requires Intervention"
        store_stats['Category'] = store_stats.apply(assign_category, axis=1)
        # Define action plans and explanations for each category
        action_plans = {
            "Star Performer": "Maintain current strategies. Share best practices with other stores.",
            "Needs Stabilization": "Investigate recent changes. Reinforce successful processes that may be slipping.",
            "Improving": "Continue positive momentum. Intensify what's working to boost engagement.",
            "Requires Intervention": "Immediate attention needed. Create a detailed action plan with district support."
        }
        cat_explanations = {
            "Star Performer": "High engagement with stable or improving performance",
            "Needs Stabilization": "High engagement but a concerning downward trend",
            "Improving": "Below average engagement but trending upward",
            "Requires Intervention": "Below average engagement with flat or declining trend"
        }
        # Display each category and the stores in it
        category_colors = {
            "Star Performer": "#2E7D32",
            "Needs Stabilization": "#F57C00",
            "Improving": "#1976D2",
            "Requires Intervention": "#C62828"
        }
        cats = ["Star Performer", "Needs Stabilization", "Improving", "Requires Intervention"]
        colA, colB = st.columns(2)
        # Split categories into two columns for layout
        for idx, cat in enumerate(cats):
            target_col = colA if idx % 2 == 0 else colB
            with target_col:
                cat_df = store_stats[store_stats['Category'] == cat]
                if cat_df.empty:
                    continue
                c_color = category_colors.get(cat, "#333")
                st.markdown(f"""<div style="border-left: 5px solid {c_color}; padding-left: 10px; margin: 10px 0 5px 0;">
                                <h4 style="color: {c_color};">{cat} ({len(cat_df)} stores)</h4>
                                </div>""", unsafe_allow_html=True)
                # List stores in this category
                store_list_cat = ", ".join([f"Store {s}" for s in cat_df['Store #'].tolist()])
                st.markdown(f"*Stores:* {store_list_cat}")
                # Show explanation and action plan for this category
                st.markdown(f"_**Meaning:**_ {cat_explanations[cat]}\n\n_**Action Plan:**_ {action_plans[cat]}")
        # Provide an interactive way to view a specific store's details
        st.markdown("----")
        st.markdown("### Store-Specific Details")
        selected_store = st.selectbox("Select a store to view detailed plan:", options=sorted(store_stats['Store #'].tolist()))
        if selected_store:
            row = store_stats[store_stats['Store #'] == selected_store].iloc[0]
            cat = row['Category']
            c_color = category_colors.get(cat, "#333")
            # Display store's category, explanation, and action plan in a modal-like container
            st.markdown(f"""<div style="background-color: #f0f0f0; padding: 15px; border-left: 5px solid {c_color}; border-radius: 5px;">
                            <h4 style="margin-top: 0; color: {c_color};">Store {selected_store}: {cat}</h4>
                            <p style="margin-bottom: 5px;"><strong>Explanation:</strong> {cat_explanations[cat]}</p>
                            <p style="margin-bottom: 5px;"><strong>Recommended Action:</strong> {action_plans[cat]}</p>
                            </div>""", unsafe_allow_html=True)
            # If the store requires improvement or intervention, suggest learning partners (top stores)
            if cat in ["Improving", "Requires Intervention"]:
                top_stores = store_stats[store_stats['Category'] == "Star Performer"]['Store #'].tolist()
                if top_stores:
                    partners = ", ".join([f"Store {s}" for s in top_stores])
                    st.info(f"**Recommended Learning Partners:** Consider visits or knowledge sharing with top performers: {partners}")

    # ----------------- TAB 4: Anomalies & Insights -----------------
    with tab4:
        st.subheader("Anomaly Detection")
        st.write(f"Stores with weekly engagement changes exceeding a Z-score threshold of **{z_threshold:.1f}** are flagged below:")
        anomalies_df = find_anomalies(df_filtered, z_threshold)
        if anomalies_df.empty:
            st.write("No significant anomalies detected for the selected criteria.")
        else:
            expander = st.expander("View anomaly details", expanded=True)
            with expander:
                # Use AgGrid for interactive anomaly table
                anom_display_df = anomalies_df[['Store #', 'Week', 'Engaged Transaction %', 'Change %pts', 'Z-score', 'Possible Explanation']]
                gb = GridOptionsBuilder.from_dataframe(anom_display_df)
                gb.configure_pagination(enabled=True, paginationAutoPageSize=True)
                gb.configure_side_bar(filters_panel=True, columns_panel=False)
                gb.configure_default_column(groupable=False, filter=True, sortable=True)
                grid_options = gb.build()
                AgGrid(anom_display_df, gridOptions=grid_options, height=min(300, 30 * (len(anom_display_df) + 1)), fit_columns_on_grid_load=True)
        # Advanced Analytics Tabs
        st.subheader("Advanced Analytics")
        subtab1, subtab2, subtab3 = st.tabs(["YTD Performance", "Recommendations", "Opportunities"])
        # Subtab 1: YTD Performance
        with subtab1:
            st.subheader("Year-To-Date Performance")
            # Summary statistics per store
            ytd_data = df_base.groupby('Store #')['Engaged Transaction %'].agg(['mean', 'std', 'min', 'max']).reset_index()
            ytd_data.columns = ['Store #', 'Average', 'StdDev', 'Min', 'Max']
            ytd_data['Range'] = ytd_data['Max'] - ytd_data['Min']
            ytd_data = ytd_data.sort_values('Average', ascending=False)
            ytd_data[['Average', 'StdDev', 'Min', 'Max', 'Range']] = ytd_data[['Average', 'StdDev', 'Min', 'Max', 'Range']].round(2)
            st.dataframe(ytd_data, hide_index=True)
            st.subheader("Engagement Trend Direction")
            st.write("Correlation between Week and Engagement % (positive = upward trend):")
            corr_data = []
            for store_id, grp in df_base.groupby('Store #'):
                if len(grp) >= 3:
                    corr_val = grp[['Week', 'Engaged Transaction %']].corr().iloc[0, 1]
                    corr_data.append({'Store #': store_id, 'Trend Correlation': round(corr_val, 3)})
            if corr_data:
                corr_df = pd.DataFrame(corr_data).sort_values('Trend Correlation', ascending=False)
                fig_corr = px.bar(corr_df, x='Trend Correlation', y='Store #',
                                   color='Trend Correlation', color_continuous_scale=['#C62828', '#BBBBBB', '#2E7D32'],
                                   range_color=(-1, 1), orientation='h',
                                   labels={'Trend Correlation': 'Week-Engagement Correlation', 'Store #': 'Store'})
                fig_corr.update_yaxes(autorange="reversed")  # highest correlation at top
                fig_corr.update_layout(height=30 * len(corr_df) + 100 if not corr_df.empty else 200)
                st.plotly_chart(fig_corr, use_container_width=True)
                st.caption("Green = improving trend, Red = declining trend. Values closer to ±1 indicate stronger trends.")
            else:
                st.info("Not enough data points for trend analysis. Try expanding the date range.")
        # Subtab 2: Recommendations
        with subtab2:
            st.subheader("Store-Specific Recommendations")
            recommendations = []
            for store_id in store_stats['Store #']:
                # Use computed category and short-term trend label for each store
                store_row = store_stats[store_stats['Store #'] == store_id].iloc[0]
                category = store_row['Category']
                trend_label = store_trends.get(store_id, "Stable")
                store_anoms = anomalies_df[anomalies_df['Store #'] == store_id] if not anomalies_df.empty else pd.DataFrame()
                # Basic recommendation logic based on category and recent trend
                if category == "Star Performer":
                    if trend_label in ["Upward", "Strong Upward"]:
                        rec = "Continue current strategies and share best practices with others."
                    elif trend_label in ["Downward", "Strong Downward"]:
                        rec = "Investigate any recent changes that might be affecting performance."
                    else:
                        rec = "Maintain consistency and monitor for any changes."
                elif category == "Needs Stabilization":
                    rec = "Focus on regaining stability. Identify causes of recent declines and address them."
                elif category == "Improving":
                    if trend_label in ["Upward", "Strong Upward"]:
                        rec = "Keep up the improvements and try to accelerate the positive momentum."
                    else:
                        rec = "Implement new engagement strategies; consider learning from top stores."
                elif category == "Requires Intervention":
                    rec = "Urgent attention needed. Conduct a thorough review and develop a turnaround plan."
                else:
                    rec = "Maintain standard practices and seek incremental improvements."
                # Add notes if anomalies present for this store
                if not store_anoms.empty:
                    biggest = store_anoms.iloc[0]
                    if biggest['Change %pts'] > 0:
                        rec += f" Investigate the positive spike in week {int(biggest['Week'])} for repeatable actions."
                    else:
                        rec += f" Investigate the drop in week {int(biggest['Week'])} and address the root cause."
                recommendations.append({
                    'Store #': store_id,
                    'Category': category,
                    'Recent Trend': trend_label,
                    'Recommendation': rec
                })
            rec_df = pd.DataFrame(recommendations)
            st.dataframe(rec_df, hide_index=True)
            st.markdown("You can filter or sort the above recommendations table to focus on specific stores or issues.")
            # Optionally open recommendations in a modal dialog for better focus
            if st.button("Open All Recommendations in Modal"):
                dialog = st.dialog("Recommendations for All Stores")
                with dialog:
                    st.dataframe(rec_df, hide_index=True, width=700)
                    st.button("Close", on_click=lambda: None)
        # Subtab 3: Opportunities
        with subtab3:
            st.subheader("Improvement Opportunities")
            if store_perf_all.size > 1:
                current_district_avg = store_perf_all.mean()
                # Scenario 1: improve bottom performer to median
                bottom_store_id = bottom_store
                bottom_current = store_perf_all[bottom_store_id]
                median_value = store_perf_all.median()
                scenario_perf = store_perf_all.copy()
                scenario_perf[bottom_store_id] = median_value
                scenario_avg = scenario_perf.mean()
                improvement = scenario_avg - current_district_avg
                st.markdown(f"#### Scenario 1: Improve Bottom Performer")
                st.markdown(f"If Store **{bottom_store_id}** improved from **{bottom_current:.2f}%** to the median of **{median_value:.2f}%**:")
                st.markdown(f"- District average would increase by **{improvement:.2f}** points")
                st.markdown(f"- New district average would be **{scenario_avg:.2f}%**")
                # Scenario 2: improve bottom 3 stores by 2 points each
                if store_perf_all.size >= 3:
                    bottom3 = store_perf_all.nsmallest(3)
                    scenario2 = store_perf_all.copy()
                    for s, val in bottom3.items():
                        scenario2[s] = min(val + 2, 100)  # assume max 100%
                    scenario_avg2 = scenario2.mean()
                    improvement2 = scenario_avg2 - current_district_avg
                    st.markdown(f"#### Scenario 2: Improve Bottom 3 Performers")
                    st.markdown(f"If each of the bottom 3 stores improved by 2 percentage points:")
                    st.markdown(f"- District average would increase by **{improvement2:.2f}** points")
                    st.markdown(f"- New district average would be **{scenario_avg2:.2f}%**")
                    b3_list = ", ".join([f"**{s}** ({bottom3[s]:.2f}%)" for s in bottom3.index])
                    st.markdown(f"*Bottom 3 stores:* {b3_list}")
                # Gap to top performer chart
                st.subheader("Gap to Top Performer")
                gap_df = pd.DataFrame({
                    'Store #': store_perf_all.index,
                    'Current %': store_perf_all.values,
                    'Gap to Top': top_val - store_perf_all.values
                })
                gap_df = gap_df[gap_df['Gap to Top'] > 0].sort_values('Gap to Top', ascending=False)
                if gap_df.empty:
                    st.write("No gap to top performer (all stores are tied for top performance).")
                else:
                    gap_df[['Current %', 'Gap to Top']] = gap_df[['Current %', 'Gap to Top']].round(2)
                    fig_gap = px.bar(gap_df, x='Gap to Top', y='Store #',
                                      color='Gap to Top', color_continuous_scale='Reds',
                                      labels={'Gap to Top': 'Gap to Top Performer (%)', 'Store #': 'Store'}, orientation='h')
                    fig_gap.update_yaxes(autorange="reversed")
                    fig_gap.update_layout(height=25 * len(gap_df) + 150, coloraxis_showscale=False)
                    st.plotly_chart(fig_gap, use_container_width=True)
            else:
                st.info("Insufficient data for opportunity analysis.")
elif role in ["Store Manager", "Assistant Customer Service Manager"]:
    # Store Manager and ACSM view: focus on single store's details
    # (We ensured exactly one store is selected above or stopped if not)
    selected_store = str(store_choice[0])
    st.subheader(f"Store {selected_store} Engagement Trend")
    # Compute store-specific data and district average for context
    store_data = df[df['Store #'] == selected_store].copy()
    if quarter_choice != "All":
        store_data = store_data[store_data['Quarter'] == q_num]
    if week_choice != "All":
        store_data = store_data[store_data['Week'] == week_num]
    store_data = store_data.sort_values('Week')
    # District average over same weeks as store_data for comparison
    dist_compare = df_base.groupby('Week', as_index=False)['Engaged Transaction %'].mean().sort_values('Week')
    # 4-week moving average for store
    store_data['MA_4W'] = store_data['Engaged Transaction %'].rolling(window=4, min_periods=1).mean()
    # Plot store trend vs district average
    fig_store = go.Figure()
    fig_store.add_trace(go.Scatter(x=store_data['Week'], y=store_data['Engaged Transaction %'],
                                   mode='lines+markers', name=f"Store {selected_store}",
                                   marker=dict(size=8), line=dict(width=3)))
    if show_ma:
        fig_store.add_trace(go.Scatter(x=store_data['Week'], y=store_data['MA_4W'],
                                       mode='lines', name="Store 4W MA", line=dict(width=2, dash='dash')))
    if not dist_compare.empty:
        fig_store.add_trace(go.Scatter(x=dist_compare['Week'], y=dist_compare['Engaged Transaction %'],
                                       mode='lines', name="District Avg", line=dict(color='black', width=3, dash='dash')))
        if show_ma:
            dist_compare['MA_4W'] = dist_compare['Engaged Transaction %'].rolling(window=4, min_periods=1).mean()
            fig_store.add_trace(go.Scatter(x=dist_compare['Week'], y=dist_compare['MA_4W'],
                                           mode='lines', name="District 4W MA", line=dict(color='black', width=2, dash='dot')))
    # If comparison data provided, overlay this store's previous period trend as gray line
    if df_comp_filtered is not None and not df_comp_filtered.empty:
        store_comp = df_comp_filtered[df_comp_filtered['Store #'] == selected_store].sort_values('Week')
        if not store_comp.empty:
            fig_store.add_trace(go.Scatter(x=store_comp['Week'], y=store_comp['Engaged Transaction %'],
                                           mode='lines', name="Last Period", line=dict(color='gray', width=2, dash='dash')))
    fig_store.update_layout(height=400, margin=dict(l=20, r=20, t=30, b=20))
    st.plotly_chart(fig_store, use_container_width=True)
    # Identify anomalies for this store (if any)
    st.subheader("Anomalies for Store " + selected_store)
    anomalies_df = find_anomalies(store_data, z_threshold)
    if anomalies_df.empty:
        st.write("No significant week-over-week anomalies detected for this store.")
    else:
        AgGrid(anomalies_df[['Week', 'Engaged Transaction %', 'Change %pts', 'Z-score', 'Possible Explanation']],
               height=min(300, 30 * (len(anomalies_df) + 1)), fit_columns_on_grid_load=True)
    # Determine this store's category relative to district and provide insights
    # Compute overall performance vs district median
    df_all_current = df_base.copy()  # all stores (time-filtered) for context
    store_avg = store_data['Engaged Transaction %'].mean()
    median_eng = df_all_current.groupby('Store #')['Engaged Transaction %'].mean().median()
    # Compute trend correlation for store (long-term)
    if len(store_data) >= 3:
        corr_val = store_data[['Week', 'Engaged Transaction %']].corr().iloc[0, 1]
    else:
        corr_val = 0.0
    # Assign category using same logic as DM categories
    if store_avg >= median_eng:
        store_category = "Needs Stabilization" if corr_val < -0.1 else "Star Performer"
    else:
        store_category = "Improving" if corr_val > 0.1 else "Requires Intervention"
    action_plans = {
        "Star Performer": "Maintain current strategies and share best practices with others.",
        "Needs Stabilization": "Investigate and address recent declines to prevent further slippage.",
        "Improving": "Capitalize on upward momentum and reinforce successful tactics.",
        "Requires Intervention": "Work on a detailed improvement plan with support from the district team."
    }
    cat_explanations = {
        "Star Performer": "Engagement is high and trend is healthy.",
        "Needs Stabilization": "Engagement is high but has shown a recent decline.",
        "Improving": "Engagement is below district median but trending upward.",
        "Requires Intervention": "Engagement is below district median with no upward trend."
    }
    category_colors = {"Star Performer": "#2E7D32", "Needs Stabilization": "#F57C00", "Improving": "#1976D2", "Requires Intervention": "#C62828"}
    st.subheader(f"Overall Status: {store_category}")
    c_color = category_colors.get(store_category, "#333")
    st.markdown(f"""<div style="background-color: #f0f0f0; padding: 15px; border-left: 5px solid {c_color}; border-radius: 5px;">
                    <h4 style="color: {c_color}; margin-top: 0;">{store_category}</h4>
                    <p style="color: #000; margin-bottom: 5px;"><strong>Explanation:</strong> {cat_explanations.get(store_category, "")}</p>
                    <p style="color: #000; margin-bottom: 5px;"><strong>Recommended Action:</strong> {action_plans.get(store_category, "")}</p>
                    </div>""", unsafe_allow_html=True)
    # If the store is not a Star, suggest top performers as mentors
    if store_category in ["Improving", "Requires Intervention", "Needs Stabilization"]:
        # Identify top performing stores in district (above median and not declining)
        top_stores = df_all_current.groupby('Store #')['Engaged Transaction %'].mean().sort_values(ascending=False).head(2).index.tolist()
        if selected_store in top_stores:
            top_stores = [s for s in top_stores if s != selected_store]
        if top_stores:
            partners = ", ".join([f"Store {s}" for s in top_stores])
            st.info(f"**Learning Opportunity:** Consider observing {partners} to adopt best practices.")
    # ACSM specific: detailed data view
    if role == "Assistant Customer Service Manager":
        st.subheader(f"Detailed Data for Store {selected_store}")
        detail_df = store_data.copy()
        # Select columns to show
        cols_to_show = ['Week', 'Engaged Transaction %']
        if 'Weekly Rank' in detail_df.columns:
            cols_to_show.append('Weekly Rank')
        if 'Quarter to Date %' in detail_df.columns:
            cols_to_show.append('Quarter to Date %')
        detail_df = detail_df[cols_to_show].reset_index(drop=True)
        gb = GridOptionsBuilder.from_dataframe(detail_df)
        gb.configure_default_column(filter=True, sortable=True)
        gb.configure_grid_options(domLayout='normal')
        grid_options = gb.build()
        AgGrid(detail_df, gridOptions=grid_options, height=min(400, 35 * (len(detail_df) + 1)), fit_columns_on_grid_load=True)