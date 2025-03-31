import streamlit as st
import pandas as pd
import numpy as np
import io
import matplotlib.pyplot as plt
import altair as alt

# Set page configuration
st.set_page_config(
    page_title="Publix Engagement Dashboard",
    page_icon="🛒",
    layout="wide"
)

# Custom CSS for styling
st.markdown("""
<style>
    /* Overall page styling */
    .reportview-container {
        background-color: #f5f5f0;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #1e5631;
        color: white;
    }
    
    [data-testid="stSidebar"] .sidebar-content {
        background-color: #1e5631;
    }
    
    .sidebar-text {
        color: white !important;
        font-size: 18px;
        padding: 15px 0 15px 10px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        margin: 0;
        cursor: pointer;
    }
    
    .sidebar-text:hover {
        background-color: rgba(255, 255, 255, 0.1);
    }
    
    /* Main content styling */
    .main-content {
        background-color: #f5f5f0;
        padding: 20px;
    }
    
    /* Filter controls styling */
    .stSelectbox > div > div {
        background-color: white;
        border-radius: 4px;
        border: 1px solid #ddd;
        padding: 0.5rem;
    }
    
    .stSelectbox > div {
        background-color: white;
        border-radius: 4px;
        border: 1px solid #ddd;
    }
    
    /* Change dropdown arrow color */
    .stSelectbox > div > div > div > div > svg {
        color: #1e5631;
    }
    
    /* Add space between filters */
    .filter-container {
        margin: 10px 0 30px 0;
    }
    
    /* Metric card styling */
    .metric-card {
        background-color: white;
        border-radius: 5px;
        padding: 20px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.12);
        height: 100%;
    }
    
    /* Chart styling */
    .chart-container {
        background-color: white;
        border-radius: 5px;
        padding: 20px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.12);
        height: 100%;
        margin-bottom: 20px;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Green headings */
    h3, h4 {
        color: #1e5631;
    }
    
    /* Custom header spacing */
    h3 {
        margin-top: 30px;
        margin-bottom: 15px;
    }
    
    /* Styling for the green Publix logo */
    .logo-container {
        display: flex;
        justify-content: center;
        padding: 20px 0;
    }
    
    .logo {
        background-color: #1e5631;
        width: 80px;
        height: 80px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-size: 50px;
        font-weight: bold;
    }
    
    /* Active nav item */
    .active-nav {
        background-color: rgba(255, 255, 255, 0.2);
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar with Publix styling
with st.sidebar:
    # Publix logo
    st.markdown(
        '<div class="logo-container">'
        '<div style="background-color: white; width: 80px; height: 80px; border-radius: 50%; display: flex; align-items: center; justify-content: center;">'
        '<p style="color: #1e5631; font-size: 50px; font-weight: bold; margin: 0;">p</p>'
        '</div></div>',
        unsafe_allow_html=True
    )
    
    # Navigation menu
    st.markdown('<div class="sidebar-text active-nav">Overview</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-text">Sales</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-text">Labor</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-text">Engagement</div>', unsafe_allow_html=True)
    
    # Add spacer
    st.markdown('<div style="flex-grow: 1;"></div>', unsafe_allow_html=True)
    
    # File uploader in sidebar
    st.markdown("### Upload Data")
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    
    # Add sample data option
    st.markdown("### Sample Data")
    use_sample_data = st.checkbox("Use sample data", value=not bool(uploaded_file))
    
    # Add disclaimer about data security
    st.markdown("---")
    st.markdown("**Note:** Data is not stored permanently and will be cleared when you refresh the page.")

# Main content area
st.markdown('<div class="main-content">', unsafe_allow_html=True)

# Process data based on upload or sample
sample_data = """Week #,Date,Store #,Engaged Transaction %,Quarter To Date %,Weekly Rank
1,01/01/2025,1001,45.2%,45.2%,3.0
1,01/01/2025,1002,42.1%,42.1%,7.0
1,01/01/2025,1003,47.5%,47.5%,1.0
1,01/01/2025,1004,44.3%,44.3%,4.0
1,01/01/2025,1005,41.8%,41.8%,8.0
1,01/01/2025,1006,46.2%,46.2%,2.0
1,01/01/2025,1007,39.5%,39.5%,10.0
1,01/01/2025,1008,43.7%,43.7%,5.0
1,01/01/2025,1009,40.2%,40.2%,9.0
1,01/01/2025,1010,43.0%,43.0%,6.0
2,01/08/2025,1001,46.3%,45.8%,4.0
2,01/08/2025,1002,43.5%,42.8%,6.0
2,01/08/2025,1003,48.2%,47.9%,1.0
2,01/08/2025,1004,44.1%,44.2%,5.0
2,01/08/2025,1005,40.9%,41.4%,9.0
2,01/08/2025,1006,47.1%,46.7%,2.0
2,01/08/2025,1007,38.7%,39.1%,10.0
2,01/08/2025,1008,42.8%,43.3%,7.0
2,01/08/2025,1009,41.5%,40.9%,8.0
2,01/08/2025,1010,46.9%,45.0%,3.0
3,01/15/2025,1001,48.1%,46.5%,2.0
3,01/15/2025,1002,44.0%,43.2%,6.0
3,01/15/2025,1003,47.3%,47.7%,3.0
3,01/15/2025,1004,45.2%,44.5%,5.0
3,01/15/2025,1005,41.3%,41.3%,9.0
3,01/15/2025,1006,48.5%,47.3%,1.0
3,01/15/2025,1007,39.0%,39.1%,10.0
3,01/15/2025,1008,43.5%,43.3%,7.0
3,01/15/2025,1009,42.1%,41.3%,8.0
3,01/15/2025,1010,46.0%,45.3%,4.0
4,01/22/2025,1001,49.2%,47.2%,2.0
4,01/22/2025,1002,45.1%,43.7%,6.0
4,01/22/2025,1003,48.8%,48.0%,3.0
4,01/22/2025,1004,46.5%,45.0%,5.0
4,01/22/2025,1005,42.8%,41.7%,8.0
4,01/22/2025,1006,50.2%,48.0%,1.0
4,01/22/2025,1007,40.5%,39.4%,10.0
4,01/22/2025,1008,44.7%,43.7%,7.0
4,01/22/2025,1009,42.9%,41.7%,9.0
4,01/22/2025,1010,47.3%,45.8%,4.0
5,01/29/2025,1001,51.0%,48.0%,1.0
5,01/29/2025,1002,46.2%,44.2%,6.0
5,01/29/2025,1003,49.5%,48.3%,3.0
5,01/29/2025,1004,47.3%,45.5%,5.0
5,01/29/2025,1005,43.4%,42.0%,8.0
5,01/29/2025,1006,50.5%,48.5%,2.0
5,01/29/2025,1007,41.2%,39.8%,10.0
5,01/29/2025,1008,45.9%,44.1%,7.0
5,01/29/2025,1009,43.5%,42.0%,9.0
5,01/29/2025,1010,48.8%,46.4%,4.0
6,02/05/2025,1001,52.3%,48.7%,1.0
6,02/05/2025,1002,47.1%,44.7%,6.0
6,02/05/2025,1003,50.2%,48.6%,3.0
6,02/05/2025,1004,48.1%,45.9%,5.0
6,02/05/2025,1005,44.0%,42.4%,8.0
6,02/05/2025,1006,51.0%,48.9%,2.0
6,02/05/2025,1007,42.0%,40.2%,10.0
6,02/05/2025,1008,46.8%,44.6%,7.0
6,02/05/2025,1009,43.8%,42.3%,9.0
6,02/05/2025,1010,49.5%,46.9%,4.0"""

# Load data
if uploaded_file is not None:
    # If user uploaded a file
    df = pd.read_csv(uploaded_file)
    use_sample_data = False
elif use_sample_data:
    # Use sample data if no file uploaded or sample data requested
    df = pd.read_csv(io.StringIO(sample_data))
else:
    st.warning("Please upload a CSV file or use sample data to continue.")
    st.stop()

# Data preprocessing
def preprocess_data(df):
    # Convert percentage strings to floats
    for col in ["Engaged Transaction %", "Quarter To Date %"]:
        if isinstance(df[col].iloc[0], str):
            df[col] = df[col].str.rstrip('%').astype(float)
    
    # Ensure Week # is integer
    df["Week #"] = df["Week #"].astype(int)
    
    # Ensure Store # is integer
    df["Store #"] = df["Store #"].astype(int)
    
    return df

df = preprocess_data(df)

# Only show filters if data is loaded
if df is not None:
    # Filter container
    col1, col2, col3 = st.columns(3)

    with col1:
        # Date Range Filter using Week # and Date
        min_week = int(df["Week #"].min())
        max_week = int(df["Week #"].max())
        
        week_range = st.select_slider(
            "Date Range",
            options=list(range(min_week, max_week + 1)),
            value=(min_week, max_week)
        )
        
        # Filter data based on selected week range
        filtered_df = df[(df["Week #"] >= week_range[0]) & (df["Week #"] <= week_range[1])]

    with col2:
        # Store Number Filter
        all_stores = sorted(df["Store #"].unique())
        store_options = ["All Stores"] + [f"Store #{store}" for store in all_stores]
        
        store_selection = st.selectbox(
            "Store Number",
            options=store_options,
            index=0
        )
        
        # Filter data further if a specific store is selected
        if store_selection != "All Stores":
            store_num = int(store_selection.replace("Store #", ""))
            filtered_df = filtered_df[filtered_df["Store #"] == store_num]

    with col3:
        # Replace Department with Rank Range
        rank_options = ["All Ranks", "Top 10%", "Middle 80%", "Bottom 10%"]
        
        rank_selection = st.selectbox(
            "Rank Range",
            options=rank_options,
            index=0
        )
        
        # Filter data based on rank selection
        if rank_selection != "All Ranks" and len(filtered_df) > 10:
            if rank_selection == "Top 10%":
                max_rank = filtered_df["Weekly Rank"].quantile(0.1)
                filtered_df = filtered_df[filtered_df["Weekly Rank"] <= max_rank]
            elif rank_selection == "Bottom 10%":
                min_rank = filtered_df["Weekly Rank"].quantile(0.9)
                filtered_df = filtered_df[filtered_df["Weekly Rank"] >= min_rank]
            elif rank_selection == "Middle 80%":
                lower_rank = filtered_df["Weekly Rank"].quantile(0.1)
                upper_rank = filtered_df["Weekly Rank"].quantile(0.9)
                filtered_df = filtered_df[(filtered_df["Weekly Rank"] > lower_rank) & 
                                        (filtered_df["Weekly Rank"] < upper_rank)]

    # After the filter section, add the metric cards
    if 'filtered_df' in locals() and not filtered_df.empty:
        # Get the most recent week data
        latest_week = filtered_df["Week #"].max()
        latest_data = filtered_df[filtered_df["Week #"] == latest_week]
        
        # Get previous week data for comparison
        prev_week = latest_week - 1
        prev_data = filtered_df[filtered_df["Week #"] == prev_week] if prev_week >= filtered_df["Week #"].min() else None
        
        # Create three metric cards in a row
        metric_cols = st.columns(3)
        
        # --- METRIC CARD 1: District Average Engagement ---
        with metric_cols[0]:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            
            # Card title
            st.markdown("<h4>This Week's District Average</h4>", unsafe_allow_html=True)
            
            # Calculate average engagement for current week
            current_engagement = latest_data["Engaged Transaction %"].mean()
            
            # Calculate week-over-week change
            wow_change = 0
            if prev_data is not None:
                prev_engagement = prev_data["Engaged Transaction %"].mean()
                wow_change = current_engagement - prev_engagement
            
            # Create arrow indicator based on trend
            arrow = "↑" if wow_change > 0 else "↓" if wow_change < 0 else "→"
            arrow_color = "green" if wow_change > 0 else "red" if wow_change < 0 else "gray"
            
            # Display the current engagement with appropriate styling
            st.markdown(
                f'<div style="display: flex; align-items: flex-end;">'
                f'<span style="font-size: 48px; font-weight: bold; color: #1e5631;">{current_engagement:.1f}</span>'
                f'<span style="margin-left: 10px; color: {arrow_color}; font-weight: bold;">'
                f'{arrow} {abs(wow_change):.1f}</span>'
                f'</div>',
                unsafe_allow_html=True
            )
            
            # Create a mini sparkline chart of recent weeks
            weeks_to_show = min(6, len(filtered_df["Week #"].unique()))
            recent_weeks = sorted(filtered_df["Week #"].unique())[-weeks_to_show:]
            
            # Get data for sparkline
            sparkline_data = []
            for week in recent_weeks:
                week_data = filtered_df[filtered_df["Week #"] == week]
                avg_engagement = week_data["Engaged Transaction %"].mean()
                sparkline_data.append({"week": week, "value": avg_engagement})
            
            # Create a bar chart similar to reference image
            sparkline_df = pd.DataFrame(sparkline_data)
            
            # Use Altair for better styling control
            chart = alt.Chart(sparkline_df).mark_bar(color='#1e5631').encode(
                x=alt.X('week:O', axis=None),
                y=alt.Y('value:Q', axis=None)
            ).properties(
                height=100
            )
            
            st.altair_chart(chart, use_container_width=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # --- METRIC CARD 2: QTD Performance ---
        with metric_cols[1]:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            
            # Card title
            st.markdown("<h4>QTD Performance</h4>", unsafe_allow_html=True)
            
            # Calculate average QTD percentage for the latest week
            avg_qtd = latest_data["Quarter To Date %"].mean()
            
            # Calculate QTD change from first week to current week
            qtd_change = 0
            first_week = filtered_df["Week #"].min()
            first_week_data = filtered_df[filtered_df["Week #"] == first_week]
            if not first_week_data.empty and first_week != latest_week:
                first_qtd = first_week_data["Quarter To Date %"].mean()
                qtd_change = avg_qtd - first_qtd
            
            # Display the QTD value with percentage
            st.markdown(
                f'<div style="display: flex; align-items: flex-end;">'
                f'<span style="font-size: 48px; font-weight: bold; color: #1e5631;">{avg_qtd:.1f}</span>'
                f'<span style="margin-left: 10px; font-size: 24px;">%</span>'
                f'</div>',
                unsafe_allow_html=True
            )
            
            # Create a line chart showing QTD trend over weeks
            qtd_trend = []
            for week in sorted(filtered_df["Week #"].unique()):
                week_data = filtered_df[filtered_df["Week #"] == week]
                avg_qtd_week = week_data["Quarter To Date %"].mean()
                qtd_trend.append({"week": week, "qtd": avg_qtd_week})
            
            qtd_df = pd.DataFrame(qtd_trend)
            
            # Use Altair for better styling
            line_chart = alt.Chart(qtd_df).mark_line(color='#1e5631', point=False).encode(
                x=alt.X('week:O', axis=None),
                y=alt.Y('qtd:Q', axis=None)
            ).properties(
                height=100
            )
            
            st.altair_chart(line_chart, use_container_width=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # --- METRIC CARD 3: Top & Bottom Stores ---
        with metric_cols[2]:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            
            # Card title
            st.markdown("<h4>Top & Bottom Stores</h4>", unsafe_allow_html=True)
            
            if not latest_data.empty:
                # Find the top store (lowest rank value = highest ranking)
                top_store_row = latest_data.loc[latest_data["Weekly Rank"].idxmin()]
                top_store_num = int(top_store_row["Store #"])
                top_store_engagement = top_store_row["Engaged Transaction %"]
                
                # Find the bottom store (highest rank value = lowest ranking)
                bottom_store_row = latest_data.loc[latest_data["Weekly Rank"].idxmax()]
                bottom_store_num = int(bottom_store_row["Store #"])
                bottom_store_engagement = bottom_store_row["Engaged Transaction %"]
                
                # Display top store
                st.markdown(
                    f'<div style="margin-bottom: 15px;">'
                    f'<div><strong>Top Store</strong></div>'
                    f'<div style="display: flex; align-items: flex-end;">'
                    f'<span style="font-size: 32px; font-weight: bold; color: #1e5631;">#{top_store_num}</span>'
                    f'<span style="margin-left: 15px; color: green;">{top_store_engagement:.1f}%</span>'
                    f'</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )
                
                # Display bottom store
                st.markdown(
                    f'<div>'
                    f'<div><strong>Bottom Store</strong></div>'
                    f'<div style="display: flex; align-items: flex-end;">'
                    f'<span style="font-size: 32px; font-weight: bold; color: #1e5631;">#{bottom_store_num}</span>'
                    f'<span style="margin-left: 15px; color: red;">{bottom_store_engagement:.1f}%</span>'
                    f'</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )
                
                # Create mini line chart showing trends
                store_trend = []
                for week in sorted(filtered_df["Week #"].unique()):
                    data = filtered_df[filtered_df["Week #"] == week]
                    avg = data["Engaged Transaction %"].mean()
                    store_trend.append({"week": week, "value": avg})
                
                trend_df = pd.DataFrame(store_trend)
                
                # Use Altair for better styling
                trend_chart = alt.Chart(trend_df).mark_line(color='#1e5631').encode(
                    x=alt.X('week:O', axis=None),
                    y=alt.Y('value:Q', axis=None)
                ).properties(
                    height=80
                )
                
                st.altair_chart(trend_chart, use_container_width=True)
            else:
                st.write("No data available for store rankings")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Create the bottom visualizations
        bottom_cols = st.columns(2)
        
        # --- VISUALIZATION 1: Weekly Engagement Trend (left) ---
        with bottom_cols[0]:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            
            # Chart title
            st.markdown("<h4>Weekly Engagement Score</h4>", unsafe_allow_html=True)
            
            # Prepare data for chart
            weekly_avg = []
            for week in sorted(filtered_df["Week #"].unique()):
                week_data = filtered_df[filtered_df["Week #"] == week]
                avg_engagement = week_data["Engaged Transaction %"].mean()
                weekly_avg.append({"Week": f"Week {week}", "Engagement": avg_engagement})
            
            weekly_df = pd.DataFrame(weekly_avg)
            
            # Create a line chart with Altair for better control
            weekly_chart = alt.Chart(weekly_df).mark_line(
                color='#1e5631',
                point=True,
                strokeWidth=3
            ).encode(
                x=alt.X('Week:N', title=None),
                y=alt.Y('Engagement:Q', scale=alt.Scale(domain=[40, 60])),
                tooltip=['Week', 'Engagement']
            ).properties(
                height=350
            )
            
            st.altair_chart(weekly_chart, use_container_width=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # --- VISUALIZATION 2: Store Ranking Distribution (right) ---
        with bottom_cols[1]:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            
            # Chart title
            st.markdown("<h4>Store Ranking Distribution</h4>", unsafe_allow_html=True)
            
            # Only use the most recent week data for this chart
            if not latest_data.empty:
                # Calculate the district average for the latest week
                district_avg = latest_data["Engaged Transaction %"].mean()
                
                # Calculate the percentage of stores above average
                stores_above_avg = latest_data[latest_data["Engaged Transaction %"] > district_avg]
                pct_above_avg = (len(stores_above_avg) / len(latest_data)) * 100
                
                # Create data for donut chart
                donut_data = pd.DataFrame([
                    {'category': 'Above Average', 'value': pct_above_avg},
                    {'category': 'Below Average', 'value': 100 - pct_above_avg}
                ])
                
                # Display the percentage in the center
                st.markdown(
                    f'<div style="text-align: center; margin-top: 20px;">'
                    f'<div style="font-size: 72px; font-weight: bold; color: #1e5631;">{int(pct_above_avg)}%</div>'
                    f'<div>stores above average</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )
                
                # Create a donut-like visualization using matplotlib
                fig, ax = plt.subplots(figsize=(6, 6))
                ax.pie(
                    donut_data['value'],
                    colors=['#1e5631', '#f0f0f0'],
                    startangle=90,
                    wedgeprops=dict(width=0.3)
                )
                ax.axis('equal')
                
                # Make the chart a true donut with a white circle in the middle
                centre_circle = plt.Circle((0, 0), 0.55, fc='white')
                ax.add_patch(centre_circle)
                
                # Remove everything from the chart except the donut
                ax.set_frame_on(False)
                ax.set_xticks([])
                ax.set_yticks([])
                
                # Display the donut chart
                st.pyplot(fig)
                
                # Add a legend
                st.markdown(
                    f'<div style="display: flex; justify-content: space-between; margin-top: 10px;">'
                    f'<div><span style="background-color: #1e5631; width: 10px; height: 10px; display: inline-block; margin-right: 5px;"></span> Above Average ({len(stores_above_avg)} stores)</div>'
                    f'<div><span style="background-color: #f0f0f0; width: 10px; height: 10px; display: inline-block; margin-right: 5px; border: 1px solid #ddd;"></span> Below Average ({len(latest_data) - len(stores_above_avg)} stores)</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )
            else:
                st.write("No data available for store distribution")
            
            st.markdown('</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)
