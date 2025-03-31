import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# --- Page Configuration ---
st.set_page_config(
    page_title="Publix Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Publix Brand Colors ---
publix_green = "#007749"
off_white_background = "#FAFAF7"
white_panel = "#FFFFFF"
dark_gray_text = "#2E2E2E"
muted_gray_lines = "#E0E0E0"

# --- Custom CSS Injection ---
st.markdown(f"""
<style>
    .stApp {{
        background-color: {off_white_background};
        color: {dark_gray_text};
        font-family: "Helvetica Neue", sans-serif;
    }}
    
    .st-emotion-cache-1v0mbdj {{
        background-color: {publix_green} !important;
    }}

    .metric-card {{
        background-color: {white_panel};
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        border: 1px solid {muted_gray_lines};
        margin-bottom: 1rem;
    }}

    h4 {{
        margin-bottom: 0.5rem;
        color: {dark_gray_text};
        font-size: 1rem;
        font-weight: 600;
    }}

    .centered-chart {{
        display: flex;
        justify-content: center;
        align-items: center;
        height: 200px;
    }}
</style>
""", unsafe_allow_html=True)

# --- Sidebar ---
with st.sidebar:
    st.markdown(f"<h1 style='color: white;'>P</h1>", unsafe_allow_html=True)
    st.markdown("---")
    page = st.radio("Navigation", ["Overview", "Sales", "Labor", "Engagement"], label_visibility="collapsed")

# --- Filter Bar ---
st.markdown("## Overview")
col1, col2, col3 = st.columns(3)
with col1:
    date_range = st.selectbox("Date Range", ["Last 7 Days", "Last 30 Days", "This Month"])
with col2:
    store_number = st.selectbox("Store Number", ["All Stores", "Store 101", "Store 205"])
with col3:
    department = st.selectbox("District", ["All", "District 1", "District 2"])

st.markdown("---")

# --- Top Metric Cards ---
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.markdown("<h4>Current Engagement</h4>", unsafe_allow_html=True)
    st.metric(label="", value="84", delta="+3")
    spark_data = pd.DataFrame(np.random.rand(6, 1)*20+40, columns=['Score'])
    st.bar_chart(spark_data, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.markdown("<h4>QTD Performance</h4>", unsafe_allow_html=True)
    trend_data = pd.DataFrame(np.random.randn(12, 1).cumsum() + 60, columns=['QTD'])
    st.line_chart(trend_data, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.markdown("<h4>Top Store</h4>", unsafe_allow_html=True)
    st.metric(label="", value="Store 205")
    store_data = pd.DataFrame(np.random.randn(12, 1).cumsum() + 70, columns=['Engagement'])
    st.line_chart(store_data, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# --- Bottom Charts ---
col4, col5 = st.columns([2, 1])

with col4:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.markdown("<h4>Weekly Engagement Score</h4>", unsafe_allow_html=True)
    week_data = pd.DataFrame((np.random.randn(12)/3).cumsum() + 50, columns=['Weekly'])
    st.line_chart(week_data, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col5:
    st.markdown('<div class="metric-card centered-chart">', unsafe_allow_html=True)
    donut_value = 32
    fig = go.Figure(data=[go.Pie(
        values=[donut_value, 100-donut_value],
        hole=0.6,
        marker_colors=[publix_green, muted_gray_lines],
        textinfo='none',
        hoverinfo='none'
    )])
    fig.update_layout(
        showlegend=False,
        margin=dict(t=0, b=0, l=0, r=0),
        annotations=[dict(text=f"<b>{donut_value}%</b>", x=0.5, y=0.5, font_size=24, showarrow=False, font=dict(color=dark_gray_text))],
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
