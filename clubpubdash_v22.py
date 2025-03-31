import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# --- Page Configuration ---
st.set_page_config(
    page_title="Publix Dashboard", # Updated Title
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Publix Brand Colors ---
publix_green = "#007749"
light_green_accent = "#43B02A" # Example lighter green for potential accents
off_white_background = "#FAFAF7"
white_panel = "#FFFFFF"
dark_gray_text = "#2E2E2E"
muted_gray_lines = "#E0E0E0" # Light gray for borders/secondary elements

# --- Custom CSS for Styling ---
st.markdown(f"""
<style>
    /* Main container background */
    .main .block-container {{
        padding-top: 2rem;
        padding-bottom: 2rem;
        padding-left: 3rem;
        padding-right: 3rem;
        background-color: {off_white_background}; /* Soft Off-White/Cream */
    }}

    /* Sidebar Styling */
    [data-testid="stSidebar"] {{
        background-color: {publix_green}; /* Publix Green */
        padding-top: 1.5rem;
    }}
    [data-testid="stSidebar"] .stRadio > label, /* Sidebar labels */
    [data-testid="stSidebar"] .stImage > img,  /* Sidebar image (if used) */
    [data-testid="stSidebar"] h1 {{ /* Sidebar 'P' logo */
        color: {white_panel}; /* White text/elements on green */
    }}
    /* Styling for the radio button options in sidebar */
     [data-testid="stSidebar"] .stRadio [role="radiogroup"] label {{
        padding: 0.5rem 1rem;
        margin-bottom: 0.5rem;
        border-radius: 0.375rem;
        color: {white_panel}; /* White text */
        transition: background-color 0.2s ease; /* Smooth transition */
    }}
     /* Style for HOVERED radio button in sidebar */
     [data-testid="stSidebar"] .stRadio [role="radiogroup"] label:hover {{
         background-color: rgba(255, 255, 255, 0.1); /* Subtle white highlight on hover */
     }}
    /* Style for the SELECTED radio button in sidebar */
    [data-testid="stSidebar"] .stRadio [role="radiogroup"] label[data-baseweb="radio"]:has(input:checked) {{
        background-color: transparent; /* No background color change on selection */
        color: {white_panel}; /* Keep text white */
        font-weight: bold; /* Make selected item bold */
        border: 1px solid {white_panel}; /* Add subtle border to selected */
    }}
    [data-testid="stSidebar"] hr {{ /* Sidebar separator */
        border-top: 1px solid rgba(255, 255, 255, 0.3); /* Lighter separator */
        margin-top: 0.5rem;
        margin-bottom: 1rem;
    }}


    /* Card Styling */
    .metric-card {{
        background-color: {white_panel};       /* Clean white panels */
        padding: 1.2rem;                     /* Slightly reduced padding */
        border-radius: 0.5rem;               /* Soft rounded corners */
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05); /* Subtle shadow for softness */
        border: 1px solid {muted_gray_lines}; /* Optional: very light border */
        margin-bottom: 1rem;
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }}
    .metric-card h4 {{ /* Style for card titles */
        margin-bottom: 0.8rem; /* Slightly more space below title */
        color: {dark_gray_text}; /* Charcoal/Dark Gray Text */
        font-size: 0.95rem; /* Slightly smaller title */
        font-weight: 500; /* Medium weight */
    }}
     /* Streamlit Metric component specific styling */
     .metric-card [data-testid="stMetricLabel"] {{
         color: {dark_gray_text}; /* Ensure label (if shown) is dark gray */
         font-size: 0.9rem;
     }}
     .metric-card [data-testid="stMetricValue"] {{
         font-size: 2.5rem; /* Larger font for main metric value */
         font-weight: 600; /* Semi-bold */
         color: {publix_green}; /* Publix Green for emphasis */
         line-height: 1.2; /* Adjust line height */
         margin-bottom: 0rem; /* Reduce space below value */
     }}
    .metric-card [data-testid="stMetricDelta"] {{
         font-size: 1rem;
         font-weight: normal;
         color: {dark_gray_text}; /* Dark gray for delta */
         margin-top: -0.2rem; /* Pull delta slightly closer */
     }}
     /* Ensure delta up/down icons are appropriately colored if needed */
     .metric-card [data-testid="stMetricDelta"] .stMetricDeltaPositive {{ color: {publix_green}; }}
     .metric-card [data-testid="stMetricDelta"] .stMetricDeltaNegative {{ color: #D32F2F; }} /* Example: Red for negative */

    /* Center align the donut chart container */
     .donut-container {{
        display: flex;
        justify-content: center;
        align-items: center;
        height: 100%;
     }}

    /* Ensure charts within cards don't overflow and have consistent spacing*/
    .metric-card .stPlotlyChart,
    .metric-card .stChart {{
        margin-top: 0.5rem;
        width: 100% !important;
    }}
    /* Style Streamlit Charts elements */
    .metric-card .stChart {{
        min-height: 100px; /* Ensure small charts have some height */
    }}

    /* General Text and Selectors */
    body, .stApp {{ /* Attempt to set base font color */
        color: {dark_gray_text};
    }}
    .stSelectbox label {{ /* Dropdown labels */
        color: {dark_gray_text};
        font-weight: 500;
        font-size: 0.9rem;
    }}
    /* Divider lines */
     hr {{
         border-top: 1px solid {muted_gray_lines};
     }}

</style>
""", unsafe_allow_html=True)


# --- Sidebar ---
with st.sidebar:
    # Publix 'P' logo using Markdown H1
    st.markdown(f"<h1 style='color: {white_panel}; text-align: left; font-size: 2.5rem; margin-left: 1rem; margin-bottom: 0.5rem; font-weight: bold;'>P</h1>", unsafe_allow_html=True)
    st.markdown("---") # Separator line

    page = st.radio(
        "Navigation", # This label is technically hidden by label_visibility
        ["Overview", "Sales", "Labor", "Engagement"],
        label_visibility="collapsed" # Hide the "Navigation" label
    )

# --- Main Content Area ---

# Top Row: Filters
col1, col2, col3 = st.columns(3)
with col1:
    # Using dummy options, replace with actual ones
    date_range = st.selectbox("Date Range", ["Last 7 Days", "Last 30 Days", "This Month", "Custom"], index=0)
with col2:
    store_number = st.selectbox("Store Number", ["All Stores", "Store 101", "Store 205", "Store 315"], index=1)
with col3:
    department = st.selectbox("Department", ["All Departments", "Grocery", "Produce", "Bakery", "Deli"], index=0)

st.markdown("---") # Add a visual separator

# --- Display Content Based on Sidebar Selection ---
if page == "Overview":

    # Second Row: KPI Cards
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown("<h4>Customer Satisfaction Score</h4>", unsafe_allow_html=True)
        # Using an Up arrow unicode character for the delta
        st.metric(label="", value="84", delta="▲ 3") # Empty label because handled by h4
        # Dummy data for the small bar chart - use Publix Green
        chart_data = pd.DataFrame(np.random.rand(5, 1) * 50 + 30, columns=['Score'])
        st.bar_chart(chart_data, height=100, color=publix_green) # Use brand color
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown("<h4>Shrink Percentage</h4>", unsafe_allow_html=True)
        st.metric(label="", value="2.5%")
        # Dummy data for the small line chart - use Publix Green
        chart_data = pd.DataFrame(np.random.randn(20, 1).cumsum() + 2.5, columns=['Shrink'])
        # Use brand color. Convert HEX to RGB tuple for st.line_chart color param if needed, but HEX string often works.
        st.line_chart(chart_data, height=100, color=publix_green)
        st.markdown('</div>', unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown("<h4>Club Publix Sign-Ups</h4>", unsafe_allow_html=True)
        st.metric(label="", value="{:,}".format(12500))
        # Dummy data for the small line chart - use Publix Green
        chart_data = pd.DataFrame(np.random.randn(20, 1).cumsum() + 12000, columns=['SignUps'])
        st.line_chart(chart_data, height=100, color=publix_green)
        st.markdown('</div>', unsafe_allow_html=True)

    # Third Row: Larger Charts
    col4, col5 = st.columns([2, 1]) # Keep line chart wider

    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown("<h4>Weekly Engagement Score</h4>", unsafe_allow_html=True)
        # Dummy data for the larger line chart
        engagement_data = pd.DataFrame(
            # Smoother random walk for a nicer looking line
            (np.random.randn(52)/5).cumsum() + 50, # 52 weeks
            columns=['Score']
        )
        # Plot using Publix Green
        st.line_chart(engagement_data, height=250, use_container_width=True, color=publix_green)
        st.markdown('</div>', unsafe_allow_html=True)

    with col5:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown('<div class="donut-container">', unsafe_allow_html=True) # For centering

        # Donut chart using Plotly with updated colors
        percentage_value = 32
        fig = go.Figure(data=[go.Pie(
            values=[percentage_value, 100 - percentage_value],
            hole=.65, # Slightly larger hole can look cleaner
            marker_colors=[publix_green, muted_gray_lines], # Publix Green and Muted Gray
            textinfo='none',
            hoverinfo='none',
            sort=False,
            direction='clockwise'
        )])

        fig.update_layout(
            showlegend=False,
            margin=dict(l=10, r=10, t=10, b=10),
            height=200, # Keep dimensions or adjust as needed
            width=200,
            paper_bgcolor='rgba(0,0,0,0)', # Transparent background
            plot_bgcolor='rgba(0,0,0,0)',
            # Annotation for the percentage in the center - use dark gray text
            annotations=[dict(
                text=f'<b>{percentage_value}%</b>',
                x=0.5, y=0.5, font_size=24, showarrow=False,
                font=dict(color=dark_gray_text) # Use dark gray text
            )]
        )
        # Use container width can sometimes distort small fixed-size charts, test this
        st.plotly_chart(fig, use_container_width=True)

        st.markdown('</div>', unsafe_allow_html=True) # Close donut-container
        st.markdown('</div>', unsafe_allow_html=True) # Close metric-card

# --- Placeholder sections for other pages ---
elif page == "Sales":
    st.header("Sales Data")
    st.write("Display Sales related charts and data here, styled consistently.")
    # Add relevant components for the Sales page, using brand colors

elif page == "Labor":
    st.header("Labor Data")
    st.write("Display Labor related charts and data here, styled consistently.")
    # Add relevant components for the Labor page, using brand colors

elif page == "Engagement":
    st.header("Engagement Data")
    st.write("Display Engagement related charts and data here, styled consistently.")
    # Add relevant components for the Engagement page, using brand colors
