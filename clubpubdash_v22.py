import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# --- Page Configuration ---
st.set_page_config(
    page_title="Dashboard Clone",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Colors Derived Directly from Image `preview.webp` (Approximations) ---
image_sidebar_green = "#0A382C" # Dark, desaturated green from sidebar
image_main_bg = "#F0F2F6"      # Very light grey background
image_card_bg = "#FFFFFF"      # White card background
image_chart_green = "#6EAA7C"  # Muted green from charts/logo P
image_donut_grey = "#E1E1E1"   # Light grey for donut remainder
image_title_grey = "#6C757D"   # Medium grey for card titles
image_value_dark = "#343A40"   # Dark grey/near-black for main values
image_delta_grey = "#888888"   # Lighter grey for delta text

# --- Custom CSS for Styling ---
# Added !important to rules likely conflicting with dark theme
st.markdown(f"""
<style>
    /* Main container background */
    .main .block-container {{
        padding-top: 2rem;
        padding-bottom: 2rem;
        padding-left: 3rem;
        padding-right: 3rem;
        background-color: {image_main_bg} !important; /* Light Grey BG from image */
        color: {image_value_dark} !important; /* Default text color */
    }}

    /* Sidebar Styling (Likely OK without !important) */
    [data-testid="stSidebar"] {{
        background-color: {image_sidebar_green};
        padding-top: 1.5rem;
    }}
    [data-testid="stSidebar"] h1 {{
        background-color: {image_chart_green}; color: {image_card_bg}; border-radius: 50%;
        width: 50px; height: 50px; display: flex; align-items: center; justify-content: center;
        font-size: 1.8rem; font-weight: bold; margin-left: 1.5rem; margin-bottom: 1.5rem;
    }}
    [data-testid="stSidebar"] .stRadio {{ margin-left: 0.5rem; margin-right: 0.5rem; }}
    [data-testid="stSidebar"] .stRadio [role="radiogroup"] label {{
        padding: 0.6rem 1rem; margin-bottom: 0.3rem; border-radius: 0.375rem;
        color: #EAEAEA; transition: background-color 0.2s ease, color 0.2s ease;
    }}
    [data-testid="stSidebar"] .stRadio [role="radiogroup"] label:hover {{ background-color: rgba(255, 255, 255, 0.1); color: {image_card_bg}; }}
    [data-testid="stSidebar"] .stRadio [role="radiogroup"] label[data-baseweb="radio"]:has(input:checked) {{
        background-color: rgba(255, 255, 255, 0.15); color: {image_card_bg}; font-weight: 600;
    }}
    [data-testid="stSidebar"] .stRadio > label {{ display: none; }}
    [data-testid="stSidebar"] hr {{ display: none; }}

    /* Top Select Box Styling */
    .stSelectbox {{ margin-bottom: 1rem; }}
    .stSelectbox > label {{ color: {image_title_grey} !important; font-weight: 500; font-size: 0.85rem; margin-bottom: 0.3rem; }}
    [data-testid="stSelectbox"] div[data-baseweb="select"] > div {{
        background-color: {image_card_bg} !important; border: 1px solid #CED4DA !important;
        box-shadow: 0 1px 2px rgba(0,0,0,.05) !important; color: {image_value_dark} !important;
    }}
    /* Ensure dropdown options are also light */
    div[data-baseweb="popover"] ul[role="listbox"] li {{
         background-color: {image_card_bg} !important;
         color: {image_value_dark} !important;
    }}
     div[data-baseweb="popover"] ul[role="listbox"] li:hover {{
         background-color: {image_main_bg} !important;
     }}


    /* Card Styling */
    .metric-card {{
        background-color: {image_card_bg} !important; padding: 1.3rem; border-radius: 0.6rem;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.06) !important; margin-bottom: 1rem; height: 100%;
        display: flex; flex-direction: column;
    }}
    .metric-card h4 {{ margin-bottom: 0.5rem; color: {image_title_grey} !important; font-size: 0.9rem; font-weight: 500; }}
    .metric-card .stMetric {{ padding: 0 !important; }}
    .metric-card [data-testid="stMetricLabel"] {{ display: none; }}
    .metric-card [data-testid="stMetricValue"] {{
        font-size: 2.4rem; font-weight: 600; color: {image_value_dark} !important; line-height: 1.1;
    }}
    .metric-card [data-testid="stMetricDelta"] {{
        font-size: 0.9rem; font-weight: 500; color: {image_delta_grey} !important;
        margin-left: 0.3rem; vertical-align: middle;
    }}
    .metric-card [data-testid="stMetricDelta"] svg {{ fill: {image_chart_green}; vertical-align: middle; }}
    .metric-card [data-testid="stMetricDelta"] .stMetricDeltaPositive {{ color: {image_delta_grey} !important; }}
    .metric-card [data-testid="stMetricDelta"] .stMetricDeltaNegative {{ color: {image_delta_grey} !important; }}

    /* Center align the donut chart container */
    .donut-container {{ display: flex; justify-content: center; align-items: center; height: 100%; min-height: 180px; }}

    /* Charts within cards */
    .metric-card .stPlotlyChart,
    .metric-card .stChart {{ margin-top: 1rem; width: 100% !important; }}
    .metric-card .stChart {{ min-height: 80px; }}
    /* Try to force light theme on charts */
    .stChart, .stPlotlyChart {{ background-color: transparent !important; }}


     /* Divider lines */
     hr {{ border: none; border-top: 1px solid #E9ECEF !important; margin-top: 1rem; margin-bottom: 1rem; }}

</style>
""", unsafe_allow_html=True)

# --- Sidebar ---
with st.sidebar:
    st.markdown("<h1>P</h1>", unsafe_allow_html=True)
    page = st.radio(
        "Navigation", ["Overview", "Sales", "Labor", "Engagement"], index=0, label_visibility="collapsed"
    )

# --- Main Content Area ---
col1, col2, col3 = st.columns(3)
with col1:
    date_range = st.selectbox("Date Range", ["Last 7 Days", "Last 30 Days", "This Month", "Custom"])
with col2:
    store_number = st.selectbox("Store Number", ["All Stores", "Store 101", "Store 205", "Store 315"])
with col3:
    department = st.selectbox("Department", ["All Departments", "Grocery", "Produce", "Bakery", "Deli"])

if page == "Overview":
    col1_kpi, col2_kpi, col3_kpi = st.columns(3)
    with col1_kpi:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown("<h4>Customer Satisfaction Score</h4>", unsafe_allow_html=True)
        st.metric(label="", value="84", delta="3", delta_color="off")
        chart_data_bar = pd.DataFrame(np.random.rand(5, 1) * 50 + 30, columns=['a'])
        st.bar_chart(chart_data_bar, height=100, color=image_chart_green)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2_kpi:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown("<h4>Shrink Percentage</h4>", unsafe_allow_html=True)
        st.markdown(f"<div style='font-size: 2.4rem; font-weight: 600; color: {image_value_dark}; line-height: 1.1;'>2,5%</div>", unsafe_allow_html=True)
        chart_data_line1 = pd.DataFrame(np.random.randn(20, 1).cumsum() + 2.5, columns=['a'])
        st.line_chart(chart_data_line1, height=100, color=image_chart_green)
        st.markdown('</div>', unsafe_allow_html=True)

    with col3_kpi:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown("<h4>Club Publix Sign-Ups</h4>", unsafe_allow_html=True)
        st.markdown(f"<div style='font-size: 2.4rem; font-weight: 600; color: {image_value_dark}; line-height: 1.1;'>{12500:,}</div>", unsafe_allow_html=True)
        chart_data_line2 = pd.DataFrame(np.random.randn(20, 1).cumsum() + 12000, columns=['a'])
        st.line_chart(chart_data_line2, height=100, color=image_chart_green)
        st.markdown('</div>', unsafe_allow_html=True)

    col4_chart, col5_chart = st.columns([2, 1])
    with col4_chart:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown("<h4>Weekly Engagement Score</h4>", unsafe_allow_html=True)
        engagement_data = pd.DataFrame((np.random.randn(52)/4).cumsum() + 48, columns=['Score'])
        st.line_chart(engagement_data, height=250, use_container_width=True, color=image_char_green) # ReferenceError: image_char_green is not defined -> image_chart_green
        st.markdown('</div>', unsafe_allow_html=True)

    with col5_chart:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown('<div class="donut-container">', unsafe_allow_html=True)
        percentage_value = 32
        fig = go.Figure(data=[go.Pie(
            values=[percentage_value, 100 - percentage_value], hole=.65,
            marker=dict(colors=[image_chart_green, image_donut_grey], line=dict(color=image_card_bg, width=2)),
            textinfo='none', hoverinfo='none', sort=False, direction='clockwise'
        )])
        fig.update_layout(
            showlegend=False, margin=dict(l=10, r=10, t=10, b=10), height=180, width=180,
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            annotations=[dict(text=f'<b>{percentage_value}%</b>', x=0.5, y=0.5, font_size=22, showarrow=False, font=dict(color=image_value_dark))]
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

# --- Placeholder sections ---
elif page == "Sales": st.header("Sales Data"); st.write("Sales details would appear here.")
elif page == "Labor": st.header("Labor Data"); st.write("Labor details would appear here.")
elif page == "Engagement": st.header("Engagement Data"); st.write("Engagement details would appear here.")
