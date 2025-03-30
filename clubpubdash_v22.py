import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import datetime

# Page configuration and custom CSS
st.set_page_config(page_title="Club Publix Engagement Dashboard", layout="wide", initial_sidebar_state="expanded")
st.markdown("""
<style>
    .metric-card { background-color: #f5f5f5; border-radius: 10px; padding: 15px; box-shadow: 2px 2px 5px rgba(0,0,0,0.1); }
    .highlight-good { color: #2E7D32; font-weight: bold; }
    .highlight-bad { color: #C62828; font-weight: bold; }
    .highlight-neutral { color: #F57C00; font-weight: bold; }
    .dashboard-title { color: #1565C0; text-align: center; padding-bottom: 20px; }
    .caption-text { font-size: 0.85em; color: #555; }
</style>
""", unsafe_allow_html=True)

# Title and introduction
st.markdown("<h1 class='dashboard-title'>Club Publix Engagement Dashboard</h1>", unsafe_allow_html=True)
st.markdown("**Club Publix** engagement analysis dashboard. Upload weekly engagement data to explore key performance indicators, trends, and opportunities across your stores. Use the filters on the left to drill down by time period or store.")

@st.cache_data
def load_data(uploaded_file):
    name = uploaded_file.name.lower()
    if name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
    # Standardize column names
    df.columns = [col.strip() for col in df.columns]
    col_map = {}
    for col in df.columns:
        cl = col.strip().lower()
        if 'quarter' in cl or 'qtd' in cl: col_map[col] = 'Quarter to Date %'
        elif 'rank' in cl: col_map[col] = 'Weekly Rank'
        elif ('week' in cl and 'ending' in cl) or cl == 'date' or cl == 'week ending': col_map[col] = 'Date'
        elif cl.startswith('week'): col_map[col] = 'Week'
        elif 'store' in cl: col_map[col] = 'Store #'
        elif 'engaged' in cl or 'engagement' in cl: col_map[col] = 'Engaged Transaction %'
    df = df.rename(columns=col_map)
    # Parse dates and convert percentage strings to numeric
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    for percent_col in ['Engaged Transaction %', 'Quarter to Date %']:
        if percent_col in df.columns:
            df[percent_col] = pd.to_numeric(df[percent_col].astype(str).str.replace('%', ''), errors='coerce')
    df = df.dropna(subset=['Engaged Transaction %'])
    if 'Weekly Rank' in df.columns:
        df['Weekly Rank'] = pd.to_numeric(df['Weekly Rank'], errors='coerce').astype('Int64')
    if 'Store #' in df.columns:
        df['Store #'] = df['Store #'].astype(str)
    if 'Week' in df.columns:
        df['Week'] = df['Week'].astype(int)
        df = df.sort_values(['Week', 'Store #'])
    return df
# Sidebar for data upload
st.sidebar.header("Data Input")
data_file = st.sidebar.file_uploader("Upload engagement data (Excel or CSV)", type=['csv', 'xlsx'])
comp_file = st.sidebar.file_uploader("Optional: Upload comparison data (prior period)", type=['csv', 'xlsx'])
if not data_file:
    st.info("Please upload a primary engagement data file to begin.")
    st.markdown("### Expected Data Format")
    st.markdown("- Store # or Store ID\n- Week or Date\n- Engaged Transaction % (the main KPI)\n- Optional: Weekly Rank, Quarter to Date %, etc.\nExample: CSV or Excel with these columns.")
    st.stop()

df = load_data(data_file)
df_comp = load_data(comp_file) if comp_file else None

# Derive quarter in dataframes
if 'Date' in df.columns:
    df['Quarter'] = df['Date'].dt.quarter
else:
    df['Quarter'] = ((df['Week'] - 1) // 13 + 1).astype(int)
if df_comp is not None:
    if 'Date' in df_comp.columns:
        df_comp['Quarter'] = df_comp['Date'].dt.quarter
    else:
        df_comp['Quarter'] = ((df_comp['Week'] - 1) // 13 + 1).astype(int)

# Sidebar filters
st.sidebar.header("Filters")
quarters = sorted(df['Quarter'].dropna().unique().tolist())
quarter_options = ["All"] + [f"Q{int(q)}" for q in quarters]
quarter_choice = st.sidebar.selectbox("Select Quarter", quarter_options, index=0)
if quarter_choice != "All":
    q_num = int(quarter_choice[1:])
    weeks_in_quarter = sorted(df[df['Quarter'] == q_num]['Week'].unique())
else:
    weeks_in_quarter = sorted(df['Week'].unique())
week_options = ["All"] + [str(int(w)) for w in weeks_in_quarter]
week_choice = st.sidebar.selectbox("Select Week", week_options, index=0)
store_list = sorted(df['Store #'].unique().tolist())
store_choice = st.sidebar.multiselect("Select Store(s)", store_list, default=[])
with st.sidebar.expander("Advanced Settings", expanded=False):
    z_threshold = st.slider("Anomaly Z-score Threshold", 1.0, 3.0, 2.0, 0.1)
    show_ma = st.checkbox("Show 4-week moving average", value=True)
    trend_analysis_weeks = st.slider("Trend analysis window (weeks)", 3, 8, 4)
    st.caption("Adjust sensitivity for anomaly detection and trend analysis.")

# Filter primary dataframe
df_filtered = df.copy()
if quarter_choice != "All":
    df_filtered = df_filtered[df_filtered['Quarter'] == q_num]
if week_choice != "All":
    week_num = int(week_choice)
    df_filtered = df_filtered[df_filtered['Week'] == week_num]
if store_choice:
    df_filtered = df_filtered[df_filtered['Store #'].isin([str(s) for s in store_choice])]
# Filter comparison dataframe
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
    st.error("No data available for the selected filters.")
    st.stop()

# Executive Summary
if week_choice != "All":
    current_week = int(week_choice)
    prev_week = current_week - 1
    if prev_week not in df_filtered['Week'].values:
        prev_vals = df[(df['Week'] < current_week) & ((quarter_choice == "All") | (df['Quarter'] == q_num))]
        prev_week = int(prev_vals['Week'].max()) if not prev_vals.empty else None
    else:
        current_week = int(df_filtered['Week'].max())
        prev_vals = df_filtered[df_filtered['Week'] < current_week]['Week']
        prev_week = int(prev_vals.max()) if not prev_vals.empty else None
else:
    current_week = int(df_filtered['Week'].max())
    prev_vals = df_filtered[df_filtered['Week'] < current_week]['Week']
    prev_week = int(prev_vals.max()) if not prev_vals.empty else None

current_avg = df_filtered[df_filtered['Week'] == current_week]['Engaged Transaction %'].mean() if current_week else None
prev_avg = df_filtered[df_filtered['Week'] == prev_week]['Engaged Transaction %'].mean() if prev_week else None
store_perf = df_filtered.groupby('Store #')['Engaged Transaction %'].mean()
top_store = store_perf.idxmax()
bottom_store = store_perf.idxmin()
top_val = store_perf.max()
bottom_val = store_perf.min()
# Trend classification for each store
def calculate_trend(group, window=4):
    if len(group) < 2: return "Stable"
    data = group.sort_values('Week').tail(window)
    if len(data) < 2: return "Stable"
    x = data['Week'].values - np.mean(data['Week'].values)
    slope = np.polyfit(x, data['Engaged Transaction %'].values, 1)[0]
    if slope > 0.5: return "Strong Upward"
    elif slope > 0.1: return "Upward"
    elif slope < -0.5: return "Strong Downward"
    elif slope < -0.1: return "Downward"
    else: return "Stable"
store_trends = df_filtered.groupby('Store #').apply(lambda g: calculate_trend(g, trend_analysis_weeks))
# Executive summary display
st.subheader("Executive Summary")
col1, col2, col3 = st.columns(3)
avg_label = (f"Store {store_choice[0]} Engagement" if store_choice and len(store_choice) == 1 
             else "Selected Stores Avg Engagement" if store_choice and len(store_choice) < len(store_list) 
             else "District Avg Engagement")
avg_display = f"{current_avg:.2f}%" if current_avg is not None else "N/A"
delta_str = f"{current_avg - prev_avg:+.2f}%" if (current_avg is not None and prev_avg is not None) else "N/A"
col1.metric(f"{avg_label} (Week {current_week})", avg_display, delta_str)
col2.metric(f"Top Performer (Week {current_week})", f"Store {top_store} — {top_val:.2f}%")
col3.metric(f"Bottom Performer (Week {current_week})", f"Store {bottom_store} — {bottom_val:.2f}%")
if current_avg is not None and prev_avg is not None:
    delta_val = current_avg - prev_avg
    trend_dir = "up" if delta_val > 0 else "down" if delta_val < 0 else "flat"
    trend_class = "highlight-good" if delta_val > 0 else "highlight-bad" if delta_val < 0 else "highlight-neutral"
    st.markdown(f"Week {current_week} average engagement is <span class='{trend_class}'>{abs(delta_val):.2f} percentage points {trend_dir}</span> from Week {prev_week}.", unsafe_allow_html=True)
elif current_avg is not None:
    st.markdown(f"Week {current_week} engagement average: <span class='highlight-neutral'>{current_avg:.2f}%</span>", unsafe_allow_html=True)
# Top & bottom trend labels
colA, colB = st.columns(2)
tcolor = "highlight-good" if store_trends[top_store] in ["Upward","Strong Upward"] else "highlight-bad" if store_trends[top_store] in ["Downward","Strong Downward"] else "highlight-neutral"
bcolor = "highlight-good" if store_trends[bottom_store] in ["Upward","Strong Upward"] else "highlight-bad" if store_trends[bottom_store] in ["Downward","Strong Downward"] else "highlight-neutral"
colA.markdown(f"**Store {top_store}** trend: <span class='{tcolor}'>{store_trends[top_store]}</span>", unsafe_allow_html=True)
colB.markdown(f"**Store {bottom_store}** trend: <span class='{bcolor}'>{store_trends[bottom_store]}</span>", unsafe_allow_html=True)

st.subheader("Key Insights")
insights = []
# Consistency insights
store_std = df_filtered.groupby('Store #')['Engaged Transaction %'].std().fillna(0)
most_consistent = store_std.idxmin()
least_consistent = store_std.idxmax()
insights.append(f"**Store {most_consistent}** shows the most consistent engagement (least variability).")
insights.append(f"**Store {least_consistent}** has the most variable engagement performance.")
# Trending stores insights
trending_up = [str(s) for s, t in store_trends.items() if t in ["Upward","Strong Upward"]]
trending_down = [str(s) for s, t in store_trends.items() if t in ["Downward","Strong Downward"]]
if trending_up:
    insights.append("Stores showing positive trends: " + ", ".join(f"**{s}**" for s in trending_up))
if trending_down:
    insights.append("Stores needing attention: " + ", ".join(f"**{s}**" for s in trending_down))
# Gap insights
if len(store_perf) > 1:
    gap = top_val - bottom_val
    insights.append(f"Gap between highest and lowest performing stores: **{gap:.2f}%**")
    if gap > 10:
        insights.append("🚨 Large performance gap indicates opportunity for knowledge sharing.")
# Display first up to 5 insights
for i, point in enumerate(insights[:5], start=1):
    st.markdown(f"{i}. {point}")

# Main tabs
tab1, tab2, tab3, tab4 = st.tabs(["Engagement Trends", "Store Comparison", "Store Performance Categories", "Anomalies & Insights"])

# Tab 1: Engagement Trends
with tab1:
    st.subheader("Engagement Trends Over Time")
    view_option = st.radio("View mode:", ["All Stores", "Custom Selection", "Recent Trends"], horizontal=True,
                           help="All Stores: View all stores | Custom Selection: Pick specific stores | Recent Trends: Focus on recent weeks")
    # Calculate district average trend and 4-week moving averages
    dist_trend = df_filtered.groupby('Week', as_index=False)['Engaged Transaction %'].mean().rename(columns={'Engaged Transaction %':'Average Engagement %'})
    dist_trend['MA_4W'] = dist_trend['Average Engagement %'].rolling(window=4, min_periods=1).mean()
    df_filtered = df_filtered.sort_values(['Store #','Week'])
    df_filtered['MA_4W'] = df_filtered.groupby('Store #')['Engaged Transaction %'].transform(lambda s: s.rolling(4, min_periods=1).mean())
    if df_comp_filtered is not None and not df_comp_filtered.empty:
        df_filtered['Period'] = 'Current'
        df_comp_filtered['Period'] = 'Comparison'
        combined = pd.concat([df_filtered, df_comp_filtered], ignore_index=True).sort_values(['Store #','Period','Week'])
        combined['MA_4W'] = combined.groupby(['Store #','Period'])['Engaged Transaction %'].transform(lambda s: s.rolling(4, min_periods=1).mean())
    else:
        combined = df_filtered.copy()
        combined['Period'] = 'Current'
    # Recent Trends view filter
    if view_option == "Recent Trends":
        all_weeks = sorted(combined['Week'].unique())
        default_start = all_weeks[0] if len(all_weeks) <= 8 else all_weeks[-8]
        default_end = all_weeks[-1]
        recent_weeks_range = st.select_slider("Select weeks to display:", options=all_weeks, value=(default_start, default_end),
                                              help="Adjust to show a shorter or longer recent period")
        recent_weeks = [w for w in all_weeks if w >= recent_weeks_range[0] and w <= recent_weeks_range[1]]
        combined = combined[combined['Week'].isin(recent_weeks)]
        dist_trend = dist_trend[dist_trend['Week'].isin(recent_weeks)]
    # Build trend chart layers
    color_scale = alt.Scale(scheme='category10')
    layers = []
    if view_option == "Custom Selection":
        stores_to_show = st.multiselect("Select stores to compare:", options=sorted(df_filtered['Store #'].unique()),
                                        default=[store_list[0]] if store_list else [])
        if stores_to_show:
            data_sel = combined[combined['Store #'].isin(stores_to_show)]
            layers.append(alt.Chart(data_sel).mark_line(strokeWidth=3).encode(
                x='Week:O', y=alt.Y('Engaged Transaction %:Q', title='Engaged Transaction %'),
                color=alt.Color('Store #:N', scale=color_scale), tooltip=['Store #','Week', alt.Tooltip('Engaged Transaction %:Q', format='.2f')]))
            layers.append(alt.Chart(data_sel).mark_point(filled=True, size=80).encode(
                x='Week:O', y='Engaged Transaction %:Q',
                color=alt.Color('Store #:N', scale=color_scale),
                tooltip=['Store #','Week', alt.Tooltip('Engaged Transaction %:Q', format='.2f')]))
            if show_ma:
                layers.append(alt.Chart(data_sel).mark_line(strokeDash=[2,2], strokeWidth=2).encode(
                    x='Week:O', y=alt.Y('MA_4W:Q', title='4W MA'),
                    color=alt.Color('Store #:N', scale=color_scale),
                    tooltip=['Store #','Week', alt.Tooltip('MA_4W:Q', format='.2f')]))
        else:
            st.info("Please select at least one store to display.")
    else:
        store_line = alt.Chart(combined).mark_line(strokeWidth=1.5).encode(
            x='Week:O', y=alt.Y('Engaged Transaction %:Q', title='Engaged Transaction %'),
            color=alt.Color('Store #:N', scale=color_scale), tooltip=['Store #','Week', alt.Tooltip('Engaged Transaction %:Q', format='.2f')])
        store_sel = alt.selection_point(fields=['Store #'], bind='legend')
        store_line = store_line.add_params(store_sel).encode(
            opacity=alt.condition(store_sel, alt.value(1), alt.value(0.2)),
            strokeWidth=alt.condition(store_sel, alt.value(3), alt.value(1))
        )
        layers.append(store_line)
        if show_ma:
            layers.append(alt.Chart(combined).mark_line(strokeDash=[2,2], strokeWidth=1.5).encode(
                x='Week:O', y=alt.Y('MA_4W:Q', title='4W MA'),
                color=alt.Color('Store #:N', scale=color_scale),
                opacity=alt.condition(store_sel, alt.value(0.8), alt.value(0.1)),
                tooltip=['Store #','Week', alt.Tooltip('MA_4W:Q', format='.2f')]))
    # District average trend lines
    layers.append(alt.Chart(dist_trend).mark_line(color='black', strokeDash=[4,2], size=3).encode(
        x='Week:O', y=alt.Y('Average Engagement %:Q', title='Engaged Transaction %'),
        tooltip=[alt.Tooltip('Average Engagement %:Q', format='.2f', title='District Avg')]
    ))
    if show_ma:
        layers.append(alt.Chart(dist_trend).mark_line(color='black', strokeDash=[1,1], size=2, opacity=0.7).encode(
            x='Week:O', y='MA_4W:Q', tooltip=[alt.Tooltip('MA_4W:Q', format='.2f', title='District 4W MA')]
        ))
    # Comparison period trend lines
    if df_comp_filtered is not None and not df_comp_filtered.empty:
        comp_view = df_comp_filtered[df_comp_filtered['Week'].isin(combined['Week'].unique())] if view_option == "Recent Trends" else df_comp_filtered
        dist_trend_comp = comp_view.groupby('Week', as_index=False)['Engaged Transaction %'].mean().sort_values('Week')
        dist_trend_comp['MA_4W'] = dist_trend_comp['Engaged Transaction %'].rolling(4, min_periods=1).mean()
        layers.append(alt.Chart(dist_trend_comp).mark_line(color='#555', strokeDash=[4,2], size=2).encode(
            x='Week:O', y='Engaged Transaction %:Q',
            tooltip=[alt.Tooltip('Engaged Transaction %:Q', format='.2f', title="Last Period Avg")]
        ))
        if show_ma:
            layers.append(alt.Chart(dist_trend_comp).mark_line(color='#555', strokeDash=[1,1], size=1.5, opacity=0.7).encode(
                x='Week:O', y='MA_4W:Q',
                tooltip=[alt.Tooltip('MA_4W:Q', format='.2f', title="Last Period 4W MA")]
            ))
    if layers:
        if view_option == "Recent Trends":
            c1, c2 = st.columns(2)
            with c1:
                if len(dist_trend['Week'].unique()) >= 2:
                    last_two = sorted(dist_trend['Week'].unique())[-2:]
                    cur_val = float(dist_trend[dist_trend['Week'] == last_two[1]]['Average Engagement %'])
                    prev_val = float(dist_trend[dist_trend['Week'] == last_two[0]]['Average Engagement %'])
                    change_pct = (cur_val - prev_val) / prev_val * 100 if prev_val != 0 else 0
                    st.metric("District Trend (Week-over-Week)", f"{cur_val:.2f}%", f"{change_pct:.1f}%", delta_color="normal")
            with c2:
                last_week = combined['Week'].max()
                last_week_data = combined[combined['Week'] == last_week]
                if not last_week_data.empty:
                    best_store = last_week_data.loc[last_week_data['Engaged Transaction %'].idxmax()]
                    st.metric(f"Top Performer (Week {last_week})", f"Store {best_store['Store #']}", f"{best_store['Engaged Transaction %']:.2f}%", delta_color="off")
        st.altair_chart(alt.layer(*layers).resolve_scale(y='shared').properties(height=400), use_container_width=True)
    else:
        st.info("No data available to display in the chart.")
    # Caption for view mode
    caption = ("**All Stores View:** Shows all store trends with interactive legend. The black dashed line = district average."
               if view_option == "All Stores" else 
               "**Custom Selection View:** Shows only selected stores with highlighted lines and markers." 
               if view_option == "Custom Selection" else 
               "**Recent Trends View:** Focuses on selected weeks with additional trend metrics above.")
    if df_comp_filtered is not None and not df_comp_filtered.empty:
        caption += " Gray dashed line = last period's district average."
    st.caption(caption)
    # Weekly Engagement Heatmap
    st.subheader("Weekly Engagement Heatmap")
    with st.expander("Heatmap Settings", expanded=False):
        colA, colB = st.columns(2)
        with colA:
            sort_method = st.selectbox("Sort stores by:", ["Average Engagement", "Recent Performance"], index=0)
        with colB:
            color_scheme = st.selectbox("Color scheme:", ["Blues","Greens","Purples","Oranges","Reds","Viridis"], index=0).lower()
    weeks_list = sorted(df_filtered['Week'].unique())
    if len(weeks_list) > 4:
        selected_range = st.select_slider("Select week range for heatmap:", options=weeks_list, value=(min(weeks_list), max(weeks_list)))
        heatmap_df = df_filtered[(df_filtered['Week'] >= selected_range[0]) & (df_filtered['Week'] <= selected_range[1])].copy()
    else:
        heatmap_df = df_filtered.copy()
    heatmap_df = heatmap_df.rename(columns={'Store #': 'StoreID','Engaged Transaction %': 'EngagedPct'})
    if heatmap_df.empty or heatmap_df['EngagedPct'].dropna().empty:
        st.info("No data available for the heatmap.")
    else:
        if sort_method == "Average Engagement":
            store_order = heatmap_df.groupby('StoreID')['EngagedPct'].mean().sort_values(ascending=False).index.tolist()
        else:
            most_recent_week = heatmap_df['Week'].max()
            store_order = heatmap_df[heatmap_df['Week'] == most_recent_week].sort_values('EngagedPct', ascending=False)['StoreID'].tolist()
        color_field, color_title = 'EngagedPct:Q', 'Engaged %'
        heatmap_chart = alt.Chart(heatmap_df).mark_rect().encode(
            x=alt.X('Week:O', title='Week'),
            y=alt.Y('StoreID:O', title='Store', sort=store_order),
            color=alt.Color(color_field, title=color_title, scale=alt.Scale(scheme=color_scheme), legend=alt.Legend(orient='right')),
            tooltip=['StoreID','Week:O', alt.Tooltip('EngagedPct:Q', format='.2f')]
        ).properties(height=max(250, 20 * len(store_order)))
        st.altair_chart(heatmap_chart, use_container_width=True)
        st.caption(f"Showing data from Week {int(heatmap_df['Week'].min())} to Week {int(heatmap_df['Week'].max())}. Stores sorted by {sort_method.lower()}. " + ("Colors normalized by week." if False else "Global color scale across all weeks.") + " Darker colors = higher engagement.")
        st.subheader("Recent Performance Trends")
        with st.expander("About This Section", expanded=True):
            st.write("This section shows which stores are **improving**, **stable**, or **declining** over the last several weeks, focusing on short-term trends.")
        c1, c2 = st.columns(2)
        with c1:
            trend_window = st.slider("Number of recent weeks to analyze", 3, 8, 4)
        with c2:
            sensitivity = st.select_slider("Sensitivity to small changes", options=["Low","Medium","High"], value="Medium")
            momentum_threshold = 0.5 if sensitivity == "Low" else 0.2 if sensitivity == "High" else 0.3
        # Compute short-term direction for each store
        directions = []
        for store_id, data in heatmap_df.groupby('StoreID'):
            if len(data) < trend_window: continue
            recent = data.sort_values('Week').tail(trend_window)
            vals = recent['EngagedPct'].values
            first_half = vals[0] if trend_window <= 3 else vals[:trend_window//2].mean()
            second_half = vals[-1] if trend_window <= 3 else vals[-(trend_window//2):].mean()
            change = second_half - first_half
            start_val = recent.iloc[0]['EngagedPct']
            current_val = recent.iloc[-1]['EngagedPct']
            total_change = current_val - start_val
            slope = np.polyfit(np.arange(len(vals)), vals, 1)[0]
            if abs(change) < momentum_threshold:
                direction, strength, color = "Stable", "Holding Steady", "#1976D2"
            elif change > 0:
                direction = "Improving"
                strength = "Strong Improvement" if change > 2 * momentum_threshold else "Gradual Improvement"
                color = "#2E7D32"
            else:
                direction = "Declining"
                strength = "Significant Decline" if change < -2 * momentum_threshold else "Gradual Decline"
                color = "#C62828"
            indicator = "↗️" if direction == "Improving" and "Gradual" in strength else "🔼" if direction == "Improving" else "↘️" if direction == "Declining" and "Gradual" in strength else "🔽" if direction == "Declining" else "➡️"
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
        if dir_df.empty:
            st.info("Not enough data to analyze recent trends.")
        else:
            col_imp, col_stab, col_dec = st.columns(3)
            imp_count = (dir_df['direction'] == 'Improving').sum()
            stab_count = (dir_df['direction'] == 'Stable').sum()
            dec_count = (dir_df['direction'] == 'Declining').sum()
            col_imp.metric("Improving", f"{imp_count} stores", delta="↗️", delta_color="normal")
            col_stab.metric("Stable", f"{stab_count} stores", delta="➡️", delta_color="off")
            col_dec.metric("Declining", f"{dec_count} stores", delta="↘️", delta_color="inverse")
            # Group results by direction
            for direction, group in dir_df.groupby('direction'):
                color = group.iloc[0]['color']
                st.markdown(f"<div style='border-left:5px solid {color}; padding-left:10px; margin:20px 0 10px;'><h4 style='color:{color};'>{direction} ({len(group)} stores)</h4></div>", unsafe_allow_html=True)
                cols = st.columns(min(3, len(group)))
                for i, (_, store_data) in enumerate(group.iterrows()):
                    with cols[i % len(cols)]:
                        change_disp = f"{store_data['total_change']:+.2f}%"
                        st.markdown(f"<div style='background-color:#2C2C2C; padding:10px; border-radius:5px; border-left:5px solid {store_data['color']}; margin-bottom:10px;'>"
                                    f"<h4 style='text-align:center; color:{store_data['color']}; margin:5px 0;'>{store_data['indicator']} Store {store_data['store']}</h4>"
                                    f"<p style='text-align:center; color:#FFFFFF; margin:5px 0;'><strong>{store_data['strength']}</strong><br>"
                                    f"<span style='font-size:0.9em;'><strong>{change_disp}</strong> over {store_data['weeks']} weeks</span><br>"
                                    f"<span style='font-size:0.85em; color:#BBBBBB;'>{store_data['start_value']:.2f}% → {store_data['current_value']:.2f}%</span></p>"
                                    "</div>", unsafe_allow_html=True)
            # Bar chart of total changes
            change_chart = alt.Chart(dir_df).mark_bar().encode(
                x=alt.X('total_change:Q', title='Change in Engagement % (Selected Weeks)'),
                y=alt.Y('store:N', sort=alt.EncodingSortField(field='total_change', order='descending'), title='Store'),
                color=alt.Color('direction:N', scale=alt.Scale(domain=['Improving','Stable','Declining'], range=['#2E7D32','#1976D2','#C62828'])),
                tooltip=[alt.Tooltip('store:N', title='Store'), alt.Tooltip('direction:N', title='Direction'),
                         alt.Tooltip('strength:N', title='Performance'), alt.Tooltip('start_value:Q', format='.2f', title='Starting Value'),
                         alt.Tooltip('current_value:Q', format='.2f', title='Current Value'), alt.Tooltip('total_change:Q', format='+.2f', title='Total Change')]
            ).properties(height=max(250, 25 * len(dir_df)))
            zero_line = alt.Chart(pd.DataFrame({'x':[0]})).mark_rule(color='white', strokeDash=[3,3]).encode(x='x:Q')
            st.altair_chart(change_chart + zero_line, use_container_width=True)
            st.subheader("How to Use This Analysis")
            st.markdown("**This section complements the Store Performance Categories tab:**\n- **Store Performance Categories** focus on longer-term performance, while **Recent Performance Trends** highlight short-term movement that might not yet be reflected in the categories.\n\n**When to take action:**\n- A \"Star Performer\" showing a \"Declining\" trend may need attention before performance drops\n- A \"Requires Intervention\" store showing an \"Improving\" trend indicates recent progress\n- Stores showing opposite trends from their category deserve the most attention")
            st.subheader("Key Insights")
            insight_points = []
            if 'Category' in df_filtered.columns:
                category_conflicts = []
                for _, store in dir_df.iterrows():
                    store_id = store['store']
                    store_cat = df_filtered[df_filtered['Store #'] == store_id]['Category'].iloc[0] if not df_filtered[df_filtered['Store #'] == store_id].empty else None
                    if store_cat == "Star Performer" and store['direction'] == "Declining":
                        category_conflicts.append({'store': store_id, 'conflict': "Star performer with recent decline"})
                    elif store_cat == "Requires Intervention" and store['direction'] == "Improving":
                        category_conflicts.append({'store': store_id, 'conflict': "Struggling store showing improvement"})
                if category_conflicts:
                    insight_points.append("**Stores with changing performance:**")
                    for conflict in category_conflicts:
                        insight_points.append(f"- Store {conflict['store']}: {conflict['conflict']}")
            if not dir_df.empty:
                top_improver = dir_df.loc[dir_df['total_change'].idxmax()]
                insight_points.append(f"**Most improved store:** Store {top_improver['store']} with {top_improver['total_change']:.2f}% increase")
                top_decliner = dir_df.loc[dir_df['total_change'].idxmin()]
                insight_points.append(f"**Largest decline:** Store {top_decliner['store']} with {top_decliner['total_change']:.2f}% decrease")
            if insight_points:
                for pt in insight_points:
                    st.markdown(pt)
            else:
                st.info("No significant insights detected in recent performance data.")

# Tab 2: Store Comparison
with tab2:
    st.subheader("Store Performance Comparison")
    if len(store_list) > 1:
        comp_data = df_filtered[df_filtered['Week'] == int(week_choice)] if week_choice != "All" else df_filtered.groupby('Store #', as_index=False)['Engaged Transaction %'].mean()
        comp_title = f"Store Comparison - Week {week_choice}" if week_choice != "All" else "Store Comparison - Period Average"
        comp_data = comp_data.sort_values('Engaged Transaction %', ascending=False)
        bar_chart = alt.Chart(comp_data).mark_bar().encode(
            y=alt.Y('Store #:N', title='Store', sort='-x'),
            x=alt.X('Engaged Transaction %:Q', title='Engaged Transaction %'),
            color=alt.Color('Engaged Transaction %:Q', scale=alt.Scale(scheme='blues')),
            tooltip=['Store #', alt.Tooltip('Engaged Transaction %:Q', format='.2f')]
        ).properties(title=comp_title, height=25 * len(comp_data))
        avg_val = comp_data['Engaged Transaction %'].mean()
        avg_rule = alt.Chart(pd.DataFrame({'avg':[avg_val]})).mark_rule(color='red', strokeDash=[4,4], size=2).encode(
            x='avg:Q', tooltip=[alt.Tooltip('avg:Q', title='District Average', format='.2f')]
        )
        st.altair_chart(bar_chart + avg_rule, use_container_width=True)
        st.subheader("Performance Relative to District Average")
        comp_data['Difference'] = comp_data['Engaged Transaction %'] - avg_val
        comp_data['Percentage'] = comp_data['Difference'] / avg_val * 100
        min_perc, max_perc = comp_data['Percentage'].min(), comp_data['Percentage'].max()
        diff_chart = alt.Chart(comp_data).mark_bar().encode(
            y=alt.Y('Store #:N', title='Store', sort='-x'),
            x=alt.X('Percentage:Q', title='% Difference from Average'),
            color=alt.Color('Percentage:Q', scale=alt.Scale(domain=[min_perc, 0, max_perc], range=['#C62828','#BBBBBB','#2E7D32'])),
            tooltip=['Store #', alt.Tooltip('Engaged Transaction %:Q', format='.2f'), alt.Tooltip('Percentage:Q', format='+.2f', title='% Diff from Avg')]
        ).properties(height=25 * len(comp_data))
        center_rule = alt.Chart(pd.DataFrame({'center':[0]})).mark_rule(color='black').encode(x='center:Q')
        st.altair_chart(diff_chart + center_rule, use_container_width=True)
        st.caption("Green bars = above average, red bars = below average.")
        if 'Weekly Rank' in df_filtered.columns:
            st.subheader("Weekly Rank Tracking")
            rank_data = df_filtered[['Week','Store #','Weekly Rank']].dropna()
            if not rank_data.empty:
                rank_chart = alt.Chart(rank_data).mark_line(point=True).encode(
                    x=alt.X('Week:O', title='Week'),
                    y=alt.Y('Weekly Rank:Q', title='Rank', scale=alt.Scale(domain=[10,1])),
                    color=alt.Color('Store #:N', scale=alt.Scale(scheme='category10')),
                    tooltip=['Store #','Week:O', alt.Tooltip('Weekly Rank:Q', title='Rank')]
                ).properties(height=300)
                rank_sel = alt.selection_point(fields=['Store #'], bind='legend')
                rank_chart = rank_chart.add_params(rank_sel).encode(
                    opacity=alt.condition(rank_sel, alt.value(1), alt.value(0.2)),
                    strokeWidth=alt.condition(rank_sel, alt.value(3), alt.value(1))
                )
                st.altair_chart(rank_chart, use_container_width=True)
                st.caption("Higher rank = better. Click legend to highlight.")
            else:
                st.info("Weekly rank data not available for the selected period.")
    else:
        st.info("Please select at least two stores to enable comparison view.")

# Tab 3: Store Performance Categories
with tab3:
    st.subheader("Store Performance Categories")
    st.write("Each store is placed into one of four categories based on their engagement level and performance trend:")
    # Calculate store stats and categories
    store_stats = df_filtered.groupby('Store #')['Engaged Transaction %'].agg(['mean','std']).reset_index()
    store_stats.columns = ['Store #','Average Engagement','Consistency']
    store_stats['Consistency'] = store_stats['Consistency'].fillna(0.0)
    trend_corr = df_filtered.groupby('Store #').apply(lambda g: g['Week'].corr(g['Engaged Transaction %']) if len(g) >= 3 else 0)
    store_stats['Trend Correlation'] = store_stats['Store #'].map(trend_corr).fillna(0)
    med_eng = store_stats['Average Engagement'].median()
    # Assign categories
    conditions = [
        (store_stats['Average Engagement'] >= med_eng) & (store_stats['Trend Correlation'] < -0.1),
        (store_stats['Average Engagement'] >= med_eng),
        (store_stats['Average Engagement'] < med_eng) & (store_stats['Trend Correlation'] > 0.1),
        (store_stats['Average Engagement'] < med_eng)
    ]
    choices = ["Needs Stabilization","Star Performer","Improving","Requires Intervention"]
    store_stats['Category'] = np.select(conditions, choices, default="Uncategorized").astype(str)
    # Define action plans and explanations
    action_plans = {
        "Star Performer": "Maintain current strategies. Document and share best practices with other stores.",
        "Needs Stabilization": "Investigate recent changes or inconsistencies. Reinforce processes to prevent decline.",
        "Improving": "Continue positive momentum. Intensify efforts driving improvement.",
        "Requires Intervention": "Urgent attention needed. Develop a comprehensive improvement plan."
    }
    explanations = {
        "Star Performer": "High engagement with stable or improving trend",
        "Needs Stabilization": "High engagement but recent downward trend",
        "Improving": "Below average engagement but trending upward",
        "Requires Intervention": "Below average engagement with flat or declining trend"
    }
    store_stats['Action Plan'] = store_stats['Category'].map(action_plans)
    store_stats['Explanation'] = store_stats['Category'].map(explanations)
    # Category overview cards
    category_info = {
        "Star Performer": {"icon": "⭐", "color": "#2E7D32"},
        "Needs Stabilization": {"icon": "⚠️", "color": "#F57C00"},
        "Improving": {"icon": "📈", "color": "#1976D2"},
        "Requires Intervention": {"icon": "🚨", "color": "#C62828"}
    }
    short_actions = {
        "Star Performer": "Share best practices with other stores",
        "Needs Stabilization": "Reinforce successful processes",
        "Improving": "Continue positive momentum",
        "Requires Intervention": "Needs comprehensive support"
    }
    colA, colB = st.columns(2)
    for cat in ["Star Performer","Needs Stabilization"]:
        info = category_info[cat]
        colA.markdown(f"<div style='background-color:#2C2C2C; padding:15px; border-radius:5px; margin-bottom:10px; border-left:5px solid {info['color']};'><h4 style='color:{info['color']}; margin-top:0;'>{info['icon']} {cat}</h4><p style='color:#FFFFFF; margin:0;'>{explanations[cat]}</p><p style='color:#FFFFFF; margin:0;'><strong>Action:</strong> {short_actions[cat]}</p></div>", unsafe_allow_html=True)
    for cat in ["Improving","Requires Intervention"]:
        info = category_info[cat]
        colB.markdown(f"<div style='background-color:#2C2C2C; padding:15px; border-radius:5px; margin-bottom:10px; border-left:5px solid {info['color']};'><h4 style='color:{info['color']}; margin-top:0;'>{info['icon']} {cat}</h4><p style='color:#FFFFFF; margin:0;'>{explanations[cat]}</p><p style='color:#FFFFFF; margin:0;'><strong>Action:</strong> {short_actions[cat]}</p></div>", unsafe_allow_html=True)
    # Category results per store group
    for cat in ["Star Performer","Needs Stabilization","Improving","Requires Intervention"]:
        subset = store_stats[store_stats['Category'] == cat]
        if subset.empty: continue
        color = category_info[cat]['color']
        st.markdown(f"<div style='border-left:5px solid {color}; padding-left:15px; margin-bottom:15px;'><h4 style='color:{color}; margin-top:0;'>{cat} ({len(subset)} stores)</h4></div>", unsafe_allow_html=True)
        subset = subset.copy()
        subset['Average Engagement'] = subset['Average Engagement'].round(2)
        subset['Trend Correlation'] = subset['Trend Correlation'].round(2)
        cols = st.columns(min(4, len(subset)))
        for i, (_, store) in enumerate(subset.iterrows()):
            with cols[i % len(cols)]:
                corr = store['Trend Correlation']
                trend_icon = "🔼" if corr > 0.3 else "⬆️" if corr > 0.1 else "🔽" if corr < -0.3 else "⬇️" if corr < -0.1 else "➡️"
                st.markdown(f"<div style='background-color:#2C2C2C; padding:10px; border-radius:5px; margin-bottom:10px;'><h4 style='text-align:center; color:{color}; margin:5px 0;'>Store {store['Store #']}</h4><p style='text-align:center; color:#FFFFFF; margin:5px 0;'><strong>Engagement:</strong> {store['Average Engagement']:.2f}%<br><strong>Trend:</strong> {trend_icon} {store['Trend Correlation']:.2f}</p></div>", unsafe_allow_html=True)
    # Store-specific action plan
    selected_store = st.selectbox("Select a store to view detailed action plan:", sorted(store_stats['Store #']))
    if selected_store:
        row = store_stats[store_stats['Store #'] == selected_store].iloc[0]
        cat = row['Category']; color = category_info[cat]['color']
        corr = row['Trend Correlation']; avg_val = row['Average Engagement']
        trend_desc, trend_icon = ("Strong positive trend","🔼") if corr > 0.3 else ("Mild positive trend","⬆️") if corr > 0.1 else ("Strong negative trend","🔽") if corr < -0.3 else ("Mild negative trend","⬇️") if corr < -0.1 else ("Stable trend","➡️")
        st.markdown(f"<div style='background-color:#2C2C2C; padding:20px; border-radius:10px; border-left:5px solid {color}; margin-bottom:20px;'><h3 style='color:{color}; margin-top:0;'>Store {selected_store} - {cat}</h3><p style='color:#FFFFFF;'><strong>Average Engagement:</strong> {avg_val:.2f}%</p><p style='color:#FFFFFF;'><strong>Trend:</strong> {trend_icon} {trend_desc} ({corr:.2f})</p><p style='color:#FFFFFF;'><strong>Explanation:</strong> {row['Explanation']}</p><h4 style='color:{color}; margin-top:1em;'>Recommended Action Plan:</h4><p style='color:#FFFFFF;'>{row['Action Plan']}</p></div>", unsafe_allow_html=True)
        if selected_store:
            
            st.subheader(f"Store {selected_store} Engagement Trend")
            
            # Filter data for just this store
            store_data = df_filtered[df_filtered['Store #'] == selected_store].sort_values('Week')
            
            if not store_data.empty:
                # Create a dataframe for district average to compare
                dist_avg = df_filtered.groupby('Week', as_index=False)['Engaged Transaction %'].mean()
                dist_avg.rename(columns={'Engaged Transaction %': 'District Average'}, inplace=True)
                
                # Prepare chart layers
                store_line = alt.Chart(store_data).mark_line(color='#1565C0', strokeWidth=3).encode(
                    x='Week:O',
                    y=alt.Y('Engaged Transaction %:Q', title='Engagement %'),
                    tooltip=['Week:O', alt.Tooltip('Engaged Transaction %:Q', format='.2f')]
                )
                
                store_points = alt.Chart(store_data).mark_circle(size=60).encode(
                    x='Week:O',
                    y='Engaged Transaction %:Q',
                    tooltip=['Week:O', alt.Tooltip('Engaged Transaction %:Q', format='.2f')]
                )
                
                # Add regression line to show trend
                regression = alt.Chart(store_data).transform_regression(
                    'Week', 'Engaged Transaction %'
                ).mark_line(color='white', strokeDash=[3,3], strokeWidth=2).encode(
                    x='Week:O',
                    y='Engaged Transaction %:Q'
                )
                
                # Add district average line
                dist_line = alt.Chart(dist_avg).mark_line(color='gray', strokeDash=[4,2], strokeWidth=2).encode(
                    x='Week:O',
                    y=alt.Y('District Average:Q'),
                    tooltip=[alt.Tooltip('District Average:Q', format='.2f', title='District Avg')]
                )
                
                # Combine chart elements
                chart = alt.layer(store_line, store_points, regression, dist_line).properties(
                    height=300
                ).resolve_scale(
                    y='shared'
                )
                
                st.altair_chart(chart, use_container_width=True)
                st.caption("Colored line = Store trend, white dashed line = regression fit, gray dashed line = district average.")
            else:
                st.info("Not enough data available to display the trend.")
        if cat in ["Improving","Requires Intervention"]:
            st.subheader("Improvement Opportunities")
            top_stores = store_stats[store_stats['Category'] == "Star Performer"]['Store #'].tolist()
            if top_stores:
                partners = ", ".join(f"Store {s}" for s in top_stores)
                st.markdown(f"<div style='background-color:#2C2C2C; padding:15px; border-radius:5px; border-left:5px solid #2E7D32;'><h4 style='color:#2E7D32; margin-top:0;'>Recommended Learning Partners</h4><p style='color:#FFFFFF;'>Consider scheduling visits with: <strong>{partners}</strong></p></div>", unsafe_allow_html=True)
            curr = row['Average Engagement']; med = store_stats['Average Engagement'].median(); gain = med - curr
            if gain > 0:
                st.markdown(f"<div style='background-color:#2C2C2C; padding:15px; border-radius:5px; border-left:5px solid #1976D2; margin-top:15px;'><h4 style='color:#1976D2; margin-top:0;'>Potential Improvement Target</h4><p style='color:#FFFFFF;'>Current average: <strong>{curr:.2f}%</strong></p><p style='color:#FFFFFF;'>District median: <strong>{med:.2f}%</strong></p><p style='color:#FFFFFF;'>Possible gain: <strong>{gain:.2f}%</strong></p></div>", unsafe_allow_html=True)
    st.markdown("### Understanding These Categories\nWe measure two key factors for each store:\n1. **Average Engagement** (current performance)\n2. **Trend Correlation** (direction of change)\n\nEach store is placed into one of four categories:\n- **Star Performer**: High engagement with stable or improving trend\n- **Needs Stabilization**: High engagement but showing a downward trend\n- **Improving**: Below average but trending upward\n- **Requires Intervention**: Below average with flat or declining trend")

# Tab 4: Anomalies & Insights
with tab4:
    st.subheader("Anomaly Detection")
    def find_anomalies(df, z_threshold=2.0):
        anomalies = []
        for store_id, grp in df.groupby('Store #'):
            grp = grp.sort_values('Week')
            diffs = grp['Engaged Transaction %'].diff().dropna()
            if diffs.empty: continue
            mean_diff = diffs.mean()
            std_diff = diffs.std(ddof=0)
            if std_diff == 0 or np.isnan(std_diff): continue
            for idx, diff_val in diffs.items():
                z = (diff_val - mean_diff) / std_diff
                if abs(z) >= z_threshold:
                    prev_idx = grp.index[grp.index.get_indexer([idx])[0] - 1] if grp.index.get_indexer([idx])[0] - 1 >= 0 else None
                    week_cur = int(grp.loc[idx, 'Week'])
                    week_prev = int(grp.loc[prev_idx, 'Week']) if prev_idx is not None else None
                    val_cur = grp.loc[idx, 'Engaged Transaction %']
                    rank_cur = grp.loc[idx, 'Weekly Rank'] if 'Weekly Rank' in grp.columns else None
                    rank_prev = grp.loc[prev_idx, 'Weekly Rank'] if prev_idx is not None and 'Weekly Rank' in grp.columns else None
                    anomalies.append({
                        'Store #': store_id,
                        'Week': week_cur,
                        'Engaged Transaction %': val_cur,
                        'Change %pts': diff_val,
                        'Z-score': z,
                        'Prev Week': week_prev,
                        'Prev Rank': int(rank_prev) if pd.notna(rank_prev) else None,
                        'Rank': int(rank_cur) if pd.notna(rank_cur) else None
                    })
        anomalies_df = pd.DataFrame(anomalies)
        if not anomalies_df.empty:
            anomalies_df['Abs Z'] = anomalies_df['Z-score'].abs()
            anomalies_df = anomalies_df.sort_values('Abs Z', ascending=False).drop(columns=['Abs Z'])
            anomalies_df[['Engaged Transaction %','Z-score','Change %pts']] = anomalies_df[['Engaged Transaction %','Z-score','Change %pts']].round(2)
            anomalies_df['Possible Explanation'] = np.where(anomalies_df['Change %pts'] >= 0,
                                                           "Engagement spiked significantly. Possible promotion or event impact.",
                                                           "Sharp drop in engagement. Potential system issue or loss of engagement.")
            improve_mask = (anomalies_df['Change %pts'] >= 0) & anomalies_df['Prev Rank'].notna() & anomalies_df['Rank'].notna() & (anomalies_df['Prev Rank'] > anomalies_df['Rank'])
            decline_mask = (anomalies_df['Change %pts'] < 0) & anomalies_df['Prev Rank'].notna() & anomalies_df['Rank'].notna() & (anomalies_df['Prev Rank'] < anomalies_df['Rank'])
            anomalies_df.loc[improve_mask, 'Possible Explanation'] += " (Improved from rank " + anomalies_df.loc[improve_mask, 'Prev Rank'].astype(int).astype(str) + " to " + anomalies_df.loc[improve_mask, 'Rank'].astype(int).astype(str) + ".)"
            anomalies_df.loc[decline_mask, 'Possible Explanation'] += " (Dropped from rank " + anomalies_df.loc[decline_mask, 'Prev Rank'].astype(int).astype(str) + " to " + anomalies_df.loc[decline_mask, 'Rank'].astype(int).astype(str) + ".)"
        return anomalies_df
    anomalies_df = find_anomalies(df_filtered, z_threshold)
    if anomalies_df.empty:
        st.write("No significant anomalies detected for the selected criteria.")
    else:
        st.write(f"Stores with weekly engagement changes exceeding a Z-score threshold of **{z_threshold:.1f}**:")
        exp = st.expander("View anomaly details", expanded=True)
        with exp:
            st.dataframe(anomalies_df[['Store #','Week','Engaged Transaction %','Change %pts','Z-score','Possible Explanation']], hide_index=True)
    st.subheader("Store-Specific Recommendations")
    recommendations = []
    for store_id in store_list:
        store_data = df_filtered[df_filtered['Store #'] == store_id]
        if store_data.empty: continue
        avg_eng = store_data['Engaged Transaction %'].mean()
        trend = calculate_trend(store_data, trend_analysis_weeks)
        store_anoms = anomalies_df[anomalies_df['Store #'] == store_id] if not anomalies_df.empty else pd.DataFrame()
        # Find category from store_stats (if available)
        category = "Unknown"
        if not store_stats.empty:
            cat_row = store_stats[store_stats['Store #'] == store_id]
            if not cat_row.empty:
                category = cat_row.iloc[0]['Category']
        # Recommendation logic by category and trend
        if category == "Star Performer":
            if trend in ["Upward","Strong Upward"]:
                rec = "Continue current strategies. Share best practices with other stores."
            elif trend in ["Downward","Strong Downward"]:
                rec = "Investigate recent changes that may have affected performance."
            else:
                rec = "Maintain consistency. Document successful practices."
        elif category == "Needs Stabilization":
            rec = "Focus on consistency. Identify factors causing variability in engagement."
        elif category == "Improving":
            if trend in ["Upward","Strong Upward"]:
                rec = "Continue improvement trajectory. Accelerate current initiatives."
            else:
                rec = "Implement new engagement strategies. Consider learning from top stores."
        elif category == "Requires Intervention":
            rec = "Urgent attention needed. Detailed store audit recommended."
        else:
            rec = "No specific category assigned. General best practices recommended."
        # Append anomaly note if exists
        if not store_anoms.empty:
            biggest = store_anoms.iloc[0]
            rec += f" Investigate {'positive spike' if biggest['Change %pts'] > 0 else 'negative drop'} in Week {int(biggest['Week'])}."
        recommendations.append({'Store #': store_id, 'Category': category, 'Trend': trend, 'Avg Engagement': round(avg_eng,2), 'Recommendation': rec})
    if recommendations:
        rec_df = pd.DataFrame(recommendations)
        st.dataframe(rec_df, hide_index=True)
    else:
        st.info("No data available for recommendations.")

# Sidebar footer and help
now = datetime.datetime.now()
st.sidebar.markdown("---")
st.sidebar.caption(f"© Publix Super Markets, Inc. {now.year}")
st.sidebar.caption(f"Last updated: {now.strftime('%Y-%m-%d')}")
with st.sidebar.expander("Help & Information"):
    st.markdown("### Using This Dashboard\n- **Upload Data**: Start by uploading your engagement data file\n- **Apply Filters**: Use the filters to focus on specific time periods or stores\n- **Explore Tabs**: Each tab provides different insights:\n    - **Engagement Trends**: Performance over time\n    - **Store Comparison**: Compare stores directly\n    - **Store Performance Categories**: Categories and action plans\n    - **Anomalies & Insights**: Unusual patterns and opportunities\n\nFor technical support, contact Reid.")