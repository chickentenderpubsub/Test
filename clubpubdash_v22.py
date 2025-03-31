import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# --- Page Configuration ---
st.set_page_config(
    page_title="Publix Dashboard Clone",
    layout="wide",  # Use wide layout
    initial_sidebar_state="expanded" # Keep sidebar open initially
)

# --- Custom CSS for Styling ---
# (Approximating colors and styles from the image)
st.markdown("""
<style>
    /* Main container background */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        padding-left: 3rem;
        padding-right: 3rem;
        background-color: #f0f2f6; /* Light grey background */
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #004d40; /* Dark green background */
        padding-top: 1.5rem;
    }
    [data-testid="stSidebar"] .stRadio > label,
    [data-testid="stSidebar"] .stImage > img {
        color: #ffffff; /* White text for sidebar items */
    }
    /* Styling for the radio button labels in sidebar */
     [data-testid="stSidebar"] .stRadio [role="radiogroup"] label {
        padding: 0.5rem 1rem;
        margin-bottom: 0.5rem;
        border-radius: 0.375rem; /* Rounded corners for selection */
        color: #ffffff; /* White text */
    }
    /* Style for the selected radio button in sidebar */
    [data-testid="stSidebar"] .stRadio [role="radiogroup"] label[data-baseweb="radio"]:has(input:checked) {
        background-color: #00796b; /* Slightly lighter green for selection */
        color: #ffffff; /* White text */
        font-weight: bold;
    }


    /* Card Styling */
    .metric-card {
        background-color: #ffffff; /* White background for cards */
        padding: 1.5rem;          /* Padding inside cards */
        border-radius: 0.5rem;    /* Rounded corners */
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); /* Subtle shadow */
        margin-bottom: 1rem;      /* Space below cards */
        height: 100%;             /* Make cards in a row equal height */
        display: flex;            /* Use flexbox for content alignment */
        flex-direction: column;   /* Stack content vertically */
        justify-content: space-between; /* Space out content */
    }
    .metric-card h4 { /* Style for card titles */
        margin-bottom: 0.5rem;
        color: #5f6368; /* Greyish color for titles */
        font-size: 1rem;
    }
     .metric-card .stMetric { /* Style for metric values */
         /* background-color: transparent !important; /* Override default Streamlit metric background */ */
         padding: 0 !important; /* Remove default padding if needed */
         border: none !important; /* Remove default border */
     }
     .metric-card [data-testid="stMetricValue"] {
         font-size: 2.2rem; /* Larger font for main metric value */
         font-weight: bold;
         color: #202124; /* Darker color for metric value */
     }
    .metric-card [data-testid="stMetricDelta"] {
         font-size: 1rem; /* Font size for delta */
         font-weight: normal;
     }

    /* Center align the donut chart */
     .donut-container {
        display: flex;
        justify-content: center;
        align-items: center;
        height: 100%; /* Ensure it takes full card height for centering */
     }

    /* Ensure charts within cards don't overflow */
    .metric-card .stPlotlyChart, .metric-card .stChart {
        margin-top: 0.5rem; /* Add some space above charts in cards */
        width: 100% !important; /* Ensure chart takes full width */
    }

     /* Hide default Streamlit headers and footers if desired */
    /*
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    */

</style>
""", unsafe_allow_html=True)


# --- Sidebar ---
with st.sidebar:
    # You would replace 'path/to/your/logo.png' with the actual path or URL
    # Using a placeholder text logo for now
    # st.image("path/to/your/logo.png", width=50) # Example using image file
    st.markdown("<h1 style='color: #4CAF50; text-align: left; font-size: 2.5rem; margin-left: 1rem;'>P</h1>", unsafe_allow_html=True)
    st.markdown("---") # Separator line

    page = st.radio(
        "Navigation",
        ["Overview", "Sales", "Labor", "Engagement"],
        label_visibility="collapsed" # Hide the "Navigation" label itself
    )

# --- Main Content Area ---

# Top Row: Filters
col1, col2, col3 = st.columns(3)
with col1:
    date_range = st.selectbox("Date Range", ["Last 7 Days", "Last 30 Days", "This Month", "Custom"], index=0)
with col2:
    store_number = st.selectbox("Store Number", ["All Stores", "Store 101", "Store 205", "Store 315"], index=1)
with col3:
    department = st.selectbox("Department", ["All Departments", "Grocery", "Produce", "Bakery", "Deli"], index=0)

st.write("---") # Add a visual separator

# --- Display Content Based on Sidebar Selection ---
if page == "Overview":

    # Second Row: KPI Cards
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown("<h4>Customer Satisfaction Score</h4>", unsafe_allow_html=True)
        st.metric(label="", value="84", delta="3") # Label is handled by h4
        # Dummy data for the small bar chart
        chart_data = pd.DataFrame(np.random.rand(5, 1) * 50 + 30, columns=['Score'])
        st.bar_chart(chart_data, height=100)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown("<h4>Shrink Percentage</h4>", unsafe_allow_html=True)
        st.metric(label="", value="2.5%")
        # Dummy data for the small line chart
        chart_data = pd.DataFrame(np.random.randn(20, 1).cumsum() + 2.5, columns=['Shrink'])
        st.line_chart(chart_data, height=100)
        st.markdown('</div>', unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown("<h4>Club Publix Sign-Ups</h4>", unsafe_allow_html=True)
        # Format the number with a comma
        st.metric(label="", value="{:,}".format(12500))
        # Dummy data for the small line chart
        chart_data = pd.DataFrame(np.random.randn(20, 1).cumsum() + 12000, columns=['SignUps'])
        st.line_chart(chart_data, height=100)
        st.markdown('</div>', unsafe_allow_html=True)

    # Third Row: Larger Charts
    col4, col5 = st.columns([2, 1]) # Give more space to the line chart

    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown("<h4>Weekly Engagement Score</h4>", unsafe_allow_html=True)
        # Dummy data for the larger line chart
        engagement_data = pd.DataFrame(
            np.random.randn(52, 1).cumsum() + 50, # 52 weeks
            columns=['Score']
        )
        # Ensure y-axis starts near the data minimum, not necessarily 0 if data is far from it
        min_val = engagement_data['Score'].min()
        st.line_chart(engagement_data, height=250, use_container_width=True) # Let y-axis be auto, or set ylim=[min_val-5, engagement_data['Score'].max()+5] if needed
        st.markdown('</div>', unsafe_allow_html=True)

    with col5:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown('<div class="donut-container">', unsafe_allow_html=True) # Container for centering

        # Donut chart using Plotly
        percentage_value = 32
        fig = go.Figure(data=[go.Pie(
            values=[percentage_value, 100 - percentage_value],
            hole=.6, # This makes it a donut chart
            marker_colors=['#4CAF50', '#E0E0E0'], # Green and grey colors
            textinfo='none', # Hide labels on slices
            hoverinfo='none', # Disable hover info
            sort=False, # Keep the order [value, remainder]
            direction='clockwise'
        )])

        fig.update_layout(
            showlegend=False,
            margin=dict(l=10, r=10, t=10, b=10), # Reduce margins
            height=200, # Adjust height
            width=200,  # Adjust width
            paper_bgcolor='rgba(0,0,0,0)', # Transparent background
            plot_bgcolor='rgba(0,0,0,0)',
            # Add annotation for the percentage in the center
            annotations=[dict(text=f'<b>{percentage_value}%</b>', x=0.5, y=0.5, font_size=24, showarrow=False, font_color='#202124')]
        )
        st.plotly_chart(fig, use_container_width=True) # use_container_width might make it slightly larger than 200x200, adjust if needed

        st.markdown('</div>', unsafe_allow_html=True) # Close donut-container
        st.markdown('</div>', unsafe_allow_html=True) # Close metric-card

elif page == "Sales":
    st.header("Sales Data")
    st.write("Display Sales related charts and data here.")
    # Add relevant components for the Sales page

elif page == "Labor":
    st.header("Labor Data")
    st.write("Display Labor related charts and data here.")
    # Add relevant components for the Labor page

elif page == "Engagement":
    st.header("Engagement Data")
    st.write("Display Engagement related charts and data here.")
    # Add relevant components for the Engagement page
