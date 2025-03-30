import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import io
import numpy as np

# Set page configuration
st.set_page_config(
    page_title="Club Publix Engagement Dashboard",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main {
        background-color: #f5f5f5;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #e6f2ff;
        border-radius: 4px 4px 0 0;
        padding-left: 20px;
        padding-right: 20px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #0066cc;
        color: white;
    }
    .metric-card {
        background-color: white;
        border-radius: 5px;
        padding: 15px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    h1, h2, h3 {
        color: #0066cc;
    }
    .highlight {
        font-weight: bold;
        color: #0066cc;
    }
    .st-emotion-cache-13ln4jf {
        padding-top: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state for storing data and preferences
if 'data' not in st.session_state:
    st.session_state.data = None
    
if 'theme' not in st.session_state:
    st.session_state.theme = 'light'  # Default theme
    
def change_theme():
    if st.session_state.theme == 'light':
        st.session_state.theme = 'dark'
    else:
        st.session_state.theme = 'light'

# Sidebar configuration
with st.sidebar:
    st.image("https://pbs.twimg.com/profile_images/1103364319317950464/xUG42wAm_400x400.png", width=100)
    st.title("Club Publix Dashboard")
    
    # File uploader in sidebar
    uploaded_file = st.file_uploader("Upload your data (CSV)", type="csv")
    
    # Theme toggle
    st.write("Dashboard Theme")
    theme_button = st.button("Toggle Light/Dark Mode", on_click=change_theme)
    st.write(f"Current theme: {st.session_state.theme}")
    
    # User role selector
    user_role = st.selectbox(
        "Select your role:",
        ["District Manager", "Store Manager", "Customer Service Manager", "ACSM"]
    )
    
    # Date range filter (will be used if data is loaded)
    if st.session_state.data is not None:
        try:
            min_date = st.session_state.data['Date'].min()
            max_date = st.session_state.data['Date'].max()
            date_range = st.date_input(
                "Select date range",
                [min_date, max_date]
            )
        except:
            st.write("Date filtering not available")
    
    # Show information about the dashboard
    with st.expander("About this Dashboard"):
        st.write("""
        This dashboard visualizes Club Publix membership engagement data across stores.
        It's designed to help Publix managers track engagement metrics, identify trends, 
        and make data-driven decisions to improve customer engagement.
        """)
    
    st.write("---")
    st.caption("© 2025 Publix Dashboard")

# Main area starts here
st.title("Club Publix Engagement Dashboard")

# Function to process the uploaded data
def process_uploaded_data(uploaded_file):
    try:
        # Read the CSV file
        df = pd.read_csv(uploaded_file)
        
        # Convert 'Date' to datetime if it exists
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        
        # Convert percentage columns to numeric
        percentage_columns = [col for col in df.columns if '%' in col]
        for col in percentage_columns:
            df[col] = df[col].str.rstrip('%').astype('float') / 100.0
            
        return df
    except Exception as e:
        st.error(f"Error processing the data: {e}")
        return None

# Main dashboard content
if uploaded_file is not None:
    try:
        df = process_uploaded_data(uploaded_file)
        if df is not None:
            st.session_state.data = df
            
            # Display data overview
            with st.expander("Data Overview"):
                st.dataframe(df.head(10))
                st.write(f"Total records: {len(df)}")
                
            # Create dashboard tabs
            tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Store Comparison", "Trend Analysis", "Rankings"])
            
            # Tab 1: Overview
            with tab1:
                st.header("Engagement Overview")
                
                # Key metrics
                col1, col2, col3 = st.columns(3)
                
                # Calculate metrics based on the data
                avg_engagement = df['Engaged Transaction %'].mean() * 100
                best_store = df.loc[df['Engaged Transaction %'].idxmax()]
                best_store_num = int(best_store['Store #']) if 'Store #' in best_store else "N/A"
                best_store_eng = best_store['Engaged Transaction %'] * 100 if 'Engaged Transaction %' in best_store else 0
                
                # Last week's average if data has Week #
                if 'Week #' in df.columns:
                    last_week = df['Week #'].max()
                    last_week_avg = df[df['Week #'] == last_week]['Engaged Transaction %'].mean() * 100
                else:
                    last_week_avg = 0
                
                with col1:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.metric("Average Engagement", f"{avg_engagement:.2f}%", f"{avg_engagement - last_week_avg:.2f}%")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col2:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.metric("Top Performing Store", f"Store #{best_store_num}", f"{best_store_eng:.2f}%")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col3:
                    # Calculate quarter to date average if available
                    if 'Quarter To Date %' in df.columns:
                        qtd_avg = df['Quarter To Date %'].mean() * 100
                        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                        st.metric("Quarter To Date Avg", f"{qtd_avg:.2f}%")
                        st.markdown('</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                        st.metric("Total Weeks", df['Week #'].nunique() if 'Week #' in df.columns else "N/A")
                        st.markdown('</div>', unsafe_allow_html=True)
                
                # Create a line chart for overall trend
                if 'Week #' in df.columns and 'Engaged Transaction %' in df.columns:
                    # Aggregate data by week
                    weekly_avg = df.groupby('Week #')['Engaged Transaction %'].mean().reset_index()
                    
                    # Create line chart
                    fig = px.line(
                        weekly_avg, 
                        x='Week #', 
                        y='Engaged Transaction %',
                        title='Weekly Engagement Trend',
                        labels={'Engaged Transaction %': 'Engagement %', 'Week #': 'Week Number'}
                    )
                    
                    # Customize the chart
                    fig.update_traces(line=dict(color='#0066cc', width=3))
                    fig.update_layout(
                        title_font_size=18,
                        xaxis_title_font_size=14,
                        yaxis_title_font_size=14,
                        yaxis_tickformat='.1%',
                        height=400,
                        margin=dict(l=40, r=40, t=50, b=40)
                    )
                    
                    # Add target line if appropriate
                    fig.add_shape(
                        type='line',
                        x0=weekly_avg['Week #'].min(),
                        y0=0.7,  # 70% target
                        x1=weekly_avg['Week #'].max(),
                        y1=0.7,
                        line=dict(color='red', width=2, dash='dash'),
                    )
                    
                    # Add annotation for target line
                    fig.add_annotation(
                        x=weekly_avg['Week #'].max(),
                        y=0.7,
                        text="Target (70%)",
                        showarrow=False,
                        yshift=10,
                        font=dict(color="red")
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                # Create a heatmap to visualize engagement across stores and weeks
                if 'Week #' in df.columns and 'Store #' in df.columns and 'Engaged Transaction %' in df.columns:
                    # Create a pivot table for the heatmap
                    pivot_df = df.pivot_table(
                        index='Store #', 
                        columns='Week #', 
                        values='Engaged Transaction %', 
                        aggfunc='mean'
                    ).fillna(0)
                    
                    # Create the heatmap
                    fig = px.imshow(
                        pivot_df,
                        labels=dict(x="Week #", y="Store #", color="Engagement %"),
                        x=pivot_df.columns,
                        y=pivot_df.index,
                        color_continuous_scale='blues',
                        aspect="auto",
                        title="Engagement Heatmap (Store × Week)"
                    )
                    
                    fig.update_layout(
                        title_font_size=18,
                        height=500,
                        margin=dict(l=40, r=40, t=50, b=40),
                        coloraxis_colorbar=dict(
                            title="Engagement %",
                            tickformat='.1%'
                        )
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
            
            # Tab 2: Store Comparison 
            with tab2:
                st.header("Store Comparison")
                
                # Create a bar chart to compare store performance
                if 'Store #' in df.columns and 'Engaged Transaction %' in df.columns:
                    # Get the most recent week if available
                    if 'Week #' in df.columns:
                        recent_week = df['Week #'].max()
                        filtered_df = df[df['Week #'] == recent_week]
                    else:
                        filtered_df = df
                    
                    # Aggregate by store
                    store_avg = filtered_df.groupby('Store #')['Engaged Transaction %'].mean().reset_index()
                    store_avg = store_avg.sort_values('Engaged Transaction %', ascending=False)
                    
                    # Calculate district average for comparison
                    district_avg = store_avg['Engaged Transaction %'].mean()
                    
                    # Create bar chart
                    fig = px.bar(
                        store_avg,
                        x='Store #',
                        y='Engaged Transaction %',
                        title=f'Store Engagement Comparison {"(Week " + str(recent_week) + ")" if "Week #" in df.columns else ""}',
                        labels={'Engaged Transaction %': 'Engagement %', 'Store #': 'Store Number'},
                        color='Engaged Transaction %',
                        color_continuous_scale=[(0, "lightblue"), (0.5, "blue"), (1, "darkblue")]
                    )
                    
                    # Add district average line
                    fig.add_shape(
                        type='line',
                        x0=-0.5,
                        y0=district_avg,
                        x1=len(store_avg)-0.5,
                        y1=district_avg,
                        line=dict(color='red', width=2, dash='dash'),
                    )
                    
                    # Add annotation for district average
                    fig.add_annotation(
                        x=len(store_avg)-1,
                        y=district_avg,
                        text=f"District Avg: {district_avg:.1%}",
                        showarrow=False,
                        yshift=10,
                        font=dict(color="red")
                    )
                    
                    fig.update_layout(
                        title_font_size=18,
                        xaxis_title_font_size=14,
                        yaxis_title_font_size=14,
                        yaxis_tickformat='.1%',
                        height=500,
                        margin=dict(l=40, r=40, t=50, b=40)
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                # Store selector for detailed view
                if 'Store #' in df.columns:
                    # Get list of stores
                    stores = sorted(df['Store #'].unique())
                    
                    # Create two columns
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        selected_store = st.selectbox("Select a store for detailed view:", stores)
                    
                    with col2:
                        comparison_store = st.selectbox("Select a store to compare with:", [None] + [s for s in stores if s != selected_store])
                    
                    # Filter data for selected store
                    store_data = df[df['Store #'] == selected_store]
                    
                    # Create line chart to show store's trend
                    if 'Week #' in df.columns and 'Engaged Transaction %' in df.columns:
                        # Prepare data
                        fig = go.Figure()
                        
                        # Add selected store line
                        fig.add_trace(go.Scatter(
                            x=store_data['Week #'],
                            y=store_data['Engaged Transaction %'],
                            mode='lines+markers',
                            name=f'Store #{selected_store}',
                            line=dict(color='blue', width=3)
                        ))
                        
                        # Add comparison store if selected
                        if comparison_store:
                            comparison_data = df[df['Store #'] == comparison_store]
                            fig.add_trace(go.Scatter(
                                x=comparison_data['Week #'],
                                y=comparison_data['Engaged Transaction %'],
                                mode='lines+markers',
                                name=f'Store #{comparison_store}',
                                line=dict(color='green', width=3)
                            ))
                        
                        # Add district average for context
                        district_data = df.groupby('Week #')['Engaged Transaction %'].mean().reset_index()
                        fig.add_trace(go.Scatter(
                            x=district_data['Week #'],
                            y=district_data['Engaged Transaction %'],
                            mode='lines',
                            name='District Average',
                            line=dict(color='red', width=2, dash='dash')
                        ))
                        
                        # Set layout
                        fig.update_layout(
                            title=f'Weekly Engagement Trend - Store #{selected_store}',
                            xaxis_title='Week Number',
                            yaxis_title='Engagement %',
                            yaxis_tickformat='.1%',
                            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                            height=400,
                            margin=dict(l=40, r=40, t=50, b=40)
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Show store details
                    if len(store_data) > 0:
                        st.subheader(f"Store #{selected_store} Details")
                        
                        # Create metrics
                        c1, c2, c3 = st.columns(3)
                        
                        with c1:
                            current_eng = store_data['Engaged Transaction %'].iloc[-1] if len(store_data) > 0 else 0
                            avg_eng = store_data['Engaged Transaction %'].mean()
                            st.metric(
                                "Current Engagement", 
                                f"{current_eng:.1%}", 
                                f"{(current_eng - avg_eng) * 100:.1f}pp from avg"
                            )
                        
                        with c2:
                            if 'Quarter To Date %' in store_data.columns:
                                current_qtd = store_data['Quarter To Date %'].iloc[-1] if len(store_data) > 0 else 0
                                st.metric("Quarter To Date", f"{current_qtd:.1%}")
                            else:
                                st.metric("Avg Engagement", f"{avg_eng:.1%}")
                        
                        with c3:
                            if 'Weekly Rank' in store_data.columns:
                                current_rank = store_data['Weekly Rank'].iloc[-1] if len(store_data) > 0 else 0
                                st.metric("Current Rank", f"#{int(current_rank) if not pd.isna(current_rank) else 'N/A'}")
                            else:
                                # Calculate rank manually
                                last_week = df['Week #'].max() if 'Week #' in df.columns else None
                                if last_week:
                                    last_week_data = df[df['Week #'] == last_week]
                                    sorted_stores = last_week_data.sort_values('Engaged Transaction %', ascending=False)
                                    rank = sorted_stores[sorted_stores['Store #'] == selected_store].index[0] + 1
                                    st.metric("Current Rank", f"#{rank}")
                                else:
                                    st.metric("Data Points", len(store_data))
            
            # Tab 3: Trend Analysis
            with tab3:
                st.header("Trend Analysis")
                
                # Create trend visualization
                if 'Week #' in df.columns and 'Engaged Transaction %' in df.columns:
                    # Calculate week-over-week changes
                    weekly_avg = df.groupby(['Week #', 'Store #'])['Engaged Transaction %'].mean().reset_index()
                    
                    # For each store, calculate week-over-week change
                    stores = weekly_avg['Store #'].unique()
                    wow_changes = []
                    
                    for store in stores:
                        store_data = weekly_avg[weekly_avg['Store #'] == store].sort_values('Week #')
                        if len(store_data) > 1:
                            for i in range(1, len(store_data)):
                                current = store_data.iloc[i]
                                previous = store_data.iloc[i-1]
                                wow_changes.append({
                                    'Store #': store,
                                    'Week #': current['Week #'],
                                    'Engagement %': current['Engaged Transaction %'],
                                    'Change': current['Engaged Transaction %'] - previous['Engaged Transaction %'],
                                    'Pct Change': (current['Engaged Transaction %'] / previous['Engaged Transaction %'] - 1) if previous['Engaged Transaction %'] > 0 else 0
                                })
                    
                    if wow_changes:
                        wow_df = pd.DataFrame(wow_changes)
                        
                        # Create scatter plot of week-over-week changes
                        fig = px.scatter(
                            wow_df,
                            x='Week #',
                            y='Change',
                            size='Engagement %',
                            color='Store #',
                            hover_name='Store #',
                            hover_data=['Engagement %', 'Pct Change'],
                            title='Week-over-Week Engagement Changes by Store',
                            labels={'Change': 'WoW Change (pp)', 'Week #': 'Week Number'}
                        )
                        
                        fig.update_layout(
                            title_font_size=18,
                            xaxis_title_font_size=14,
                            yaxis_title_font_size=14,
                            yaxis_tickformat='.1%',
                            height=500,
                            margin=dict(l=40, r=40, t=50, b=40)
                        )
                        
                        # Add a horizontal line at y=0 to indicate no change
                        fig.add_shape(
                            type='line',
                            x0=wow_df['Week #'].min(),
                            y0=0,
                            x1=wow_df['Week #'].max(),
                            y1=0,
                            line=dict(color='gray', width=1, dash='dash'),
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                
                # Create a box plot to show distribution of engagement by week
                if 'Week #' in df.columns and 'Engaged Transaction %' in df.columns:
                    fig = px.box(
                        df,
                        x='Week #',
                        y='Engaged Transaction %',
                        title='Distribution of Store Engagement by Week',
                        labels={'Engaged Transaction %': 'Engagement %', 'Week #': 'Week Number'}
                    )
                    
                    fig.update_layout(
                        title_font_size=18,
                        xaxis_title_font_size=14,
                        yaxis_title_font_size=14,
                        yaxis_tickformat='.1%',
                        height=400,
                        boxmode='group',
                        margin=dict(l=40, r=40, t=50, b=40)
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                # Anomaly detection section
                st.subheader("Anomaly Detection")
                
                if 'Store #' in df.columns and 'Engaged Transaction %' in df.columns:
                    # Calculate z-scores for engagement metrics
                    if 'Week #' in df.columns:
                        # Group by week and calculate stats
                        weekly_stats = df.groupby('Week #')['Engaged Transaction %'].agg(['mean', 'std']).reset_index()
                        
                        # Merge stats back to original data
                        merged_df = pd.merge(df, weekly_stats, on='Week #')
                        
                        # Calculate z-scores
                        merged_df['z_score'] = (merged_df['Engaged Transaction %'] - merged_df['mean']) / merged_df['std'].replace(0, 1)  # Avoid division by zero
                        
                        # Identify outliers (z-score > 2 or < -2)
                        outliers = merged_df[(merged_df['z_score'] > 2) | (merged_df['z_score'] < -2)]
                        
                        if len(outliers) > 0:
                            # Sort by absolute z-score descending
                            outliers['abs_z_score'] = outliers['z_score'].abs()
                            outliers = outliers.sort_values('abs_z_score', ascending=False)
                            
                            # Display top outliers
                            st.write(f"Found {len(outliers)} potential anomalies (stores with unusual engagement levels)")
                            
                            # Format for display
                            display_df = outliers[['Week #', 'Store #', 'Engaged Transaction %', 'z_score']].head(10)
                            display_df['Engaged Transaction %'] = display_df['Engaged Transaction %'].apply(lambda x: f"{x:.1%}")
                            display_df['z_score'] = display_df['z_score'].round(2)
                            display_df.columns = ['Week', 'Store', 'Engagement %', 'Z-Score']
                            
                            st.dataframe(display_df)
                        else:
                            st.write("No significant anomalies detected in the data.")
                    else:
                        st.write("Week data is required for anomaly detection.")
            
            # Tab 4: Rankings
            with tab4:
                st.header("Store Rankings")
                
                # Create ranking visualization
                if 'Store #' in df.columns and 'Engaged Transaction %' in df.columns:
                    # Determine the most recent week
                    if 'Week #' in df.columns:
                        recent_week = df['Week #'].max()
                        st.write(f"Rankings for Week #{recent_week}")
                        
                        # Filter for the most recent week
                        recent_data = df[df['Week #'] == recent_week]
                    else:
                        st.write("Rankings (All Data)")
                        recent_data = df
                    
                    # Calculate store averages and rank them
                    store_ranks = recent_data.groupby('Store #')['Engaged Transaction %'].mean().reset_index()
                    store_ranks = store_ranks.sort_values('Engaged Transaction %', ascending=False)
                    store_ranks['Rank'] = range(1, len(store_ranks) + 1)
                    
                    # Format for display
                    store_ranks['Engaged Transaction %'] = store_ranks['Engaged Transaction %'].apply(lambda x: f"{x:.2%}")
                    store_ranks.columns = ['Store #', 'Engagement %', 'Rank']
                    
                    # Create a simple bar chart for rankings
                    fig = px.bar(
                        store_ranks.head(10),  # Top 10 stores
                        x='Store #',
                        y='Engagement %',
                        title='Top 10 Stores by Engagement',
                        text='Rank',
                        color='Engagement %',
                        color_continuous_scale=[(0, "blue"), (1, "darkblue")]
                    )
                    
                    fig.update_traces(texttemplate='#%{text}', textposition='outside')
                    
                    fig.update_layout(
                        title_font_size=18,
                        xaxis_title_font_size=14,
                        yaxis_title_font_size=14,
                        showlegend=False,
                        height=400,
                        margin=dict(l=40, r=40, t=50, b=40)
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show the full rankings table
                    st.subheader("Complete Store Rankings")
                    st.dataframe(store_ranks, height=400)
                    
                    # Add historical ranking trends if weekly data available
                    if 'Week #' in df.columns and len(df['Week #'].unique()) > 1:
                        st.subheader("Ranking Trends")
                        
                        # Let user select stores to view ranking trends
                        selected_stores = st.multiselect(
                            "Select stores to view ranking trends:",
                            options=sorted(df['Store #'].unique()),
                            default=sorted(df['Store #'].unique())[:5]  # Default to first 5 stores
                        )
                        
                        if selected_stores:
                            # Calculate weekly rankings for all weeks
                            weekly_ranks = []
                            
                            for week in sorted(df['Week #'].unique()):
                                week_data = df[df['Week #'] == week]
                                week_ranks = week_data.groupby('Store #')['Engaged Transaction %'].mean().reset_index()
                                week_ranks = week_ranks.sort_values('Engaged Transaction %', ascending=False)
                                week_ranks['Rank'] = range(1, len(week_ranks) + 1)
                                week_ranks['Week #'] = week
                                weekly_ranks.append(week_ranks)
                            
                            # Combine all weekly rankings
                            all_ranks = pd.concat(weekly_ranks)
                            
                            # Filter for selected stores
                            selected_ranks = all_ranks[all_ranks['Store #'].isin(selected_stores)]
                            
                            # Create line chart for ranking trends
                            fig = px.line(
                                selected_ranks,
                                x='Week #',
                                y='Rank',
                                color='Store #',
                                title='Weekly Ranking Trends',
                                labels={'Rank': 'Store Rank', 'Week #': 'Week Number'}
                            )
                            
                            # Invert y-axis so rank 1 is at the top
                            fig.update_layout(
                                yaxis=dict(autorange="reversed"),
                                title_font_size=18,
                                xaxis_title_font_size=14,
                                yaxis_title_font_size=14,
                                height=500,
                                margin=dict(l=40, r=40, t=50, b=40)
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                
                # Add a section for most improved stores
                if 'Week #' in df.columns and len(df['Week #'].unique()) > 1:
                    st.subheader("Most Improved Stores")
                    
                    # Get first and last week
                    first_week = df['Week #'].min()
                    last_week = df['Week #'].max()
                    
                    # Filter data for first and last week
                    first_week_data = df[df['Week #'] == first_week].groupby('Store #')['Engaged Transaction %'].mean().reset_index()
                    last_week_data = df[df['Week #'] == last_week].groupby('Store #')['Engaged Transaction %'].mean().reset_index()
                    
                    # Merge data and calculate improvement
                    merged = pd.merge(first_week_data, last_week_data, on='Store #', suffixes=('_first', '_last'))
                    merged['Improvement'] = merged['Engaged Transaction %_last'] - merged['Engaged Transaction %_first']
                    merged['Improvement %'] = (merged['Engaged Transaction %_last'] / merged['Engaged Transaction %_first'] - 1) * 100
                    
                    # Sort by improvement
                    improved = merged.sort_values('Improvement', ascending=False)
                    
                    # Format for display
                    display_improved = improved.copy()
                    display_improved['Engaged Transaction %_first'] = display_improved['Engaged Transaction %_first'].apply(lambda x: f"{x:.2%}")
                    display_improved['Engaged Transaction %_last'] = display_improved['Engaged Transaction %_last'].apply(lambda x: f"{x:.2%}")
                    display_improved['Improvement'] = display_improved['Improvement'].apply(lambda x: f"{x:.2%}")
                    display_improved['Improvement %'] = display_improved['Improvement %'].round(1)
                    
                    # Rename columns for display
                    display_improved.columns = ['Store #', 'First Week', 'Last Week', 'Improvement', 'Improvement %']
                    
                    # Create two columns
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Show most improved stores
                        st.write("Most Improved Stores")
                        st.dataframe(display_improved.head(5), height=200)
                    
                    with col2:
                        # Show least improved stores
                        st.write("Least Improved Stores")
                        st.dataframe(display_improved.tail(5).iloc[::-1], height=200)
                    
                    # Create a horizontal bar chart for most improved stores
                    fig = px.bar(
                        display_improved.head(10),
                        y='Store #',
                        x='Improvement %',
                        orientation='h',
                        title='Top 10 Most Improved Stores',
                        labels={'Improvement %': '% Improvement', 'Store #': 'Store Number'},
                        color='Improvement %',
                        color_continuous_scale=[(0, "lightgreen"), (1, "darkgreen")]
                    )
                    
                    fig.update_layout(
                        title_font_size=18,
                        xaxis_title_font_size=14,
                        yaxis_title_font_size=14,
                        yaxis=dict(autorange="reversed"),  # Reverse y-axis to show highest at top
                        height=400,
                        margin=dict(l=40, r=40, t=50, b=40)
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
    else:
        st.error("There was an error processing the uploaded file. Please check the file format and try again.")
else:
    # Display instructions when no file is uploaded
    st.info("Welcome to the Club Publix Engagement Dashboard!")
    
    # Create columns for the info cards
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div style="border:1px solid #ddd; padding:15px; border-radius:5px;">
            <h3>📊 Upload Your Data</h3>
            <p>Start by uploading a CSV file using the uploader in the sidebar. The file should contain Club Publix engagement data.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="border:1px solid #ddd; padding:15px; border-radius:5px;">
            <h3>👁️ Visualize Trends</h3>
            <p>Once your data is uploaded, you'll see multiple tabs to analyze engagement trends, compare stores, and view rankings.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style="border:1px solid #ddd; padding:15px; border-radius:5px;">
            <h3>🔍 Gain Insights</h3>
            <p>Identify top performing stores, track improvements, and detect engagement anomalies to drive strategic decisions.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Display example data structure
    st.subheader("Expected Data Format")
    st.write("Your CSV file should have the following columns:")
    
    example_data = {
        'Week #': [1, 1, 1, 2, 2, 2],
        'Date': ['01/01/2025', '01/01/2025', '01/01/2025', '01/08/2025', '01/08/2025', '01/08/2025'],
        'Store #': [1001, 1002, 1003, 1001, 1002, 1003],
        'Engaged Transaction %': ['65.2%', '70.1%', '68.3%', '66.5%', '71.2%', '69.0%'],
        'Quarter To Date %': ['65.2%', '70.1%', '68.3%', '65.9%', '70.7%', '68.7%'],
        'Weekly Rank': [3, 1, 2, 3, 1, 2]
    }
    
    example_df = pd.DataFrame(example_data)
    st.dataframe(example_df)
    
    # Add some placeholder visualizations
    st.subheader("Sample Visualizations")
    
    # Create a sample line chart
    weeks = list(range(1, 13))
    engagement = [0.65 + 0.01 * i + 0.02 * np.random.random() for i in range(12)]
    
    sample_data = pd.DataFrame({
        'Week #': weeks,
        'Engagement %': engagement
    })
    
    fig = px.line(
        sample_data, 
        x='Week #', 
        y='Engagement %',
        title='Sample Weekly Engagement Trend',
        labels={'Engagement %': 'Engagement %', 'Week #': 'Week Number'}
    )
    
    fig.update_traces(line=dict(color='#0066cc', width=3))
    fig.update_layout(
        title_font_size=18,
        xaxis_title_font_size=14,
        yaxis_title_font_size=14,
        yaxis_tickformat='.1%',
        height=300,
        margin=dict(l=40, r=40, t=50, b=40)
    )
    
    st.plotly_chart(fig, use_container_width=True)