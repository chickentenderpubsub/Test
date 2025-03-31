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
st.markdown(f"""
<style>
    /* Main container background */
    .main .block-container {{
        padding-top: 2rem;
        padding-bottom: 2rem;
        padding-left: 3rem;
        padding-right: 3rem;
        background-color: {image_main_bg}; /* Light Grey BG from image */
        color: {image_value_dark}; /* Default text color */
    }}

    /* Sidebar Styling */
    [data-testid="stSidebar"] {{
        background-color: {image_sidebar_green}; /* Dark Green from image */
        padding-top: 1.5rem;
    }}
    /* Style for the 'P' logo (using h1) */
    [data-testid="stSidebar"] h1 {{
        background-color: {image_chart_green}; /* Green circle background */
        color: {image_card_bg}; /* White 'P' text */
        border-radius: 50%;      /* Make it circular */
        width: 50px;             /* Fixed width */
        height: 50px;            /* Fixed height */
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.8rem;       /* Adjust font size of 'P' */
        font-weight: bold;
        margin-left: 1.5rem;     /* Adjust positioning */
        margin-bottom: 1.5rem;
    }}
     /* Styling for the radio button options in sidebar */
     [data-testid="stSidebar"] .stRadio {{
        margin-left: 0.5rem; /* Indent radio buttons slightly */
        margin-right: 0.5rem;
     }}
     [data-testid="stSidebar"] .stRadio [role="radiogroup"] label {{
        padding: 0.6rem 1rem; /* Adjust padding */
        margin-bottom: 0.3rem;
        border-radius: 0.375rem;
        color: #EAEAEA; /* Lighter text color on dark green */
        transition: background-color 0.2s ease, color 0.2s ease;
     }}
     /* Style for HOVERED radio button in sidebar */
     [data-testid="stSidebar"] .stRadio [role="radiogroup"] label:hover {{
         background-color: rgba(255, 255, 255, 0.1);
         color: {image_card_bg};
     }}
    /* Style for the SELECTED radio button in sidebar */
    /* In the image, selected item has slightly lighter background */
    [data-testid="stSidebar"] .stRadio [role="radiogroup"] label[data-baseweb="radio"]:has(input:checked) {{
        background-color: rgba(255, 255, 255, 0.15); /* Subtle highlight */
        color: {image_card_bg}; /* White text */
        font-weight: 600; /* Make selected slightly bolder */
    }}
    /* Remove default Streamlit Radio label */
     [data-testid="stSidebar"] .stRadio > label {{
        display: none;
     }}
     /* Sidebar separator */
     [data-testid="stSidebar"] hr {{ display: none; }} /* Hide separator */

    /* Top Select Box Styling */
    .stSelectbox {{
        margin-bottom: 1rem; /* Add space below selectors */
    }}
    .stSelectbox > label {{ /* Select box label */
        color: {image_title_grey};
        font-weight: 500;
        font-size: 0.85rem;
        margin-bottom: 0.3rem;
    }}
     /* Style the dropdown itself */
    [data-testid="stSelectbox"] div[data-baseweb="select"] > div {{
        background-color: {image_card_bg}; /* White background */
        border: 1px solid #CED4DA;       /* Light border similar to image */
        box-shadow: 0 1px 2px rgba(0,0,0,.05); /* Subtle shadow */
        color: {image_value_dark};
    }}

    /* Card Styling */
    .metric-card {{
        background-color: {image_card_bg};       /* White panels */
        padding: 1.3rem;                     /* Adjust padding */
        border-radius: 0.6rem;               /* More pronounced radius */
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.06); /* Slightly more visible shadow */
        margin-bottom: 1rem;
        height: 100%;
        display: flex;
        flex-direction: column;
        /* justify-content: space-between; */ /* Let content flow naturally */
    }}
    .metric-card h4 {{ /* Card titles */
        margin-bottom: 0.5rem;
        color: {image_title_grey}; /* Medium Grey Text */
        font-size: 0.9rem;
        font-weight: 500;
    }}
     /* Streamlit Metric component specific styling */
     .metric-card .stMetric {{
         padding: 0 !important; /* Remove default padding */
     }}
     .metric-card [data-testid="stMetricLabel"] {{
         display: none; /* Hide the default st.metric label */
     }}
     .metric-card [data-testid="stMetricValue"] {{
         font-size: 2.4rem; /* Value font size */
         font-weight: 600;
         color: {image_value_dark}; /* Dark text for value */
         line-height: 1.1;
     }}
    .metric-card [data-testid="stMetricDelta"] {{
         font-size: 0.9rem; /* Delta font size */
         font-weight: 500;
         color: {image_delta_grey}; /* Lighter grey for delta text */
         margin-left: 0.3rem; /* Space between value and delta */
         vertical-align: middle; /* Align delta nicely */
     }}
     /* Style the delta indicator (up/down arrow) */
     .metric-card [data-testid="stMetricDelta"] svg {{
         fill: {image_chart_green}; /* Green arrow color */
         vertical-align: middle;
     }}
     .metric-card [data-testid="stMetricDelta"] .stMetricDeltaPositive {{ color: {image_delta_grey}; }}
     .metric-card [data-testid="stMetricDelta"] .stMetricDeltaNegative {{ color: {image_delta_grey}; }} /* Make negative delta text grey too */


    /* Center align the donut chart container */
     .donut-container {{
        display: flex;
        justify-content: center;
        align-items: center;
        height: 100%;
        min-height: 180px; /* Ensure space for donut */
     }}

    /* Charts within cards */
    .metric-card .stPlotlyChart,
    .metric-card .stChart {{
        margin-top: 1rem; /* Space above chart */
        width: 100% !important;
    }}
    .metric-card .stChart {{
        min-height: 80px; /* Min height for small charts */
    }}

     /* Divider lines */
     hr {{
         border: none; /* Remove default hr */
         border-top: 1px solid #E9ECEF; /* Custom light divider */
         margin-top: 1rem;
         margin-bottom: 1rem;
     }}

</style>
""", unsafe_allow_html=True)


# --- Sidebar ---
with st.sidebar:
    # Recreate the 'P' logo using HTML/CSS within Markdown
    st.markdown("<h1>P</h1>", unsafe_allow_html=True)
    # st.markdown("---") # Removed separator

    page = st.radio(
        "Navigation", # Label technically hidden, but used by Streamlit internally
        ["Overview", "Sales", "Labor", "Engagement"],
        index=0, # Default to Overview
        label_visibility="collapsed"
    )

# --- Main Content Area ---

# Top Row: Filters
col1, col2, col3 = st.columns(3)
with col1:
    date_range = st.selectbox("Date Range", ["Last 7 Days", "Last 30 Days", "This Month", "Custom"])
with col2:
    store_number = st.selectbox("Store Number", ["All Stores", "Store 101", "Store 205", "Store 315"])
with col3:
    department = st.selectbox("Department", ["All Departments", "Grocery", "Produce", "Bakery", "Deli"])

# Removed the main divider hr, spacing handled by CSS margins now

# --- Display Content Based on Sidebar Selection ---
if page == "Overview":

    # Second Row: KPI Cards
    col1_kpi, col2_kpi, col3_kpi = st.columns(3)

    with col1_kpi:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown("<h4>Customer Satisfaction Score</h4>", unsafe_allow_html=True)
        st.metric(label="", value="84", delta="3", delta_color="off") # Use CSS for delta color
        # Dummy data & chart using image's green
        chart_data_bar = pd.DataFrame(np.random.rand(5, 1) * 50 + 30, columns=['a']) # Column name doesn't matter much here
        st.bar_chart(chart_data_bar, height=100, color=image_chart_green)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2_kpi:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown("<h4>Shrink Percentage</h4>", unsafe_allow_html=True)
        # Need to use markdown for '%' formatting if st.metric doesn't handle it well with delta off
        st.markdown(f"<div style='font-size: 2.4rem; font-weight: 600; color: {image_value_dark}; line-height: 1.1;'>2,5%</div>", unsafe_allow_html=True)
        # st.metric(label="", value="2,5%") # Simpler, but delta styling might interfere
        chart_data_line1 = pd.DataFrame(np.random.randn(20, 1).cumsum() + 2.5, columns=['a'])
        st.line_chart(chart_data_line1, height=100, color=image_chart_green)
        st.markdown('</div>', unsafe_allow_html=True)

    with col3_kpi:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown("<h4>Club Publix Sign-Ups</h4>", unsafe_allow_html=True)
        st.markdown(f"<div style='font-size: 2.4rem; font-weight: 600; color: {image_value_dark}; line-height: 1.1;'>{12500:,}</div>", unsafe_allow_html=True)
        # st.metric(label="", value=f"{12500:,}")
        chart_data_line2 = pd.DataFrame(np.random.randn(20, 1).cumsum() + 12000, columns=['a'])
        st.line_chart(chart_data_line2, height=100, color=image_chart_green)
        st.markdown('</div>', unsafe_allow_html=True)

    # Third Row: Larger Charts
    col4_chart, col5_chart = st.columns([2, 1]) # Ratio approx 2:1 from image

    with col4_chart:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown("<h4>Weekly Engagement Score</h4>", unsafe_allow_html=True)
        engagement_data = pd.DataFrame(
            (np.random.randn(52)/4).cumsum() + 48, # Smoother data centered ~50
            columns=['Score']
        )
        # Plot using image's green
        st.line_chart(engagement_data, height=250, use_container_width=True, color=image_chart_green)
        st.markdown('</div>', unsafe_allow_html=True)

    with col5_chart:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown('<div class="donut-container">', unsafe_allow_html=True) # Centering

        percentage_value = 32
        fig = go.Figure(data=[go.Pie(
            values=[percentage_value, 100 - percentage_value],
            hole=.65,
            marker=dict(colors=[image_chart_green, image_donut_grey], line=dict(color=image_card_bg, width=2)), # Add white line between slices like image
            textinfo='none',
            hoverinfo='none',
            sort=False,
            direction='clockwise'
        )])

        fig.update_layout(
            showlegend=False,
            margin=dict(l=10, r=10, t=10, b=10),
            height=180, # Adjusted size
            width=180,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            annotations=[dict(
                text=f'<b>{percentage_value}%</b>',
                x=0.5, y=0.5, font_size=22, showarrow=False, # Adjusted font size
                font=dict(color=image_value_dark) # Dark text color
            )]
        )
        st.plotly_chart(fig, use_container_width=True) # Let container handle sizing

        st.markdown('</div>', unsafe_allow_html=True) # Close donut-container
        st.markdown('</div>', unsafe_allow_html=True) # Close metric-card

# --- Placeholder sections for other pages ---
# (Keep these simple as they are not the focus)
elif page == "Sales":
    st.header("Sales Data")
    st.write("Sales details would appear here.")

elif page == "Labor":
    st.header("Labor Data")
    st.write("Labor details would appear here.")

elif page == "Engagement":
    st.header("Engagement Data")
    st.write("Engagement details would appear here.")
