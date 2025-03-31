import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from sklearn.cluster import KMeans # Note: KMeans is imported but not used in the provided code.
from sklearn.preprocessing import StandardScaler # Note: StandardScaler is imported but not used.
import datetime

# --------------------------------------------------------
# Theme Colors (Publix Inspired)
# --------------------------------------------------------
PUBLIX_GREEN = "#5F8F38"
LIGHT_GREEN = "#E8F5E9"
DARK_TEXT = "#333333"
MEDIUM_TEXT = "#555555"
LIGHT_TEXT_ON_DARK_BG = "#FFFFFF" # Only for elements like buttons
LIGHT_GRAY_BG = "#F8F8F8"
MEDIUM_GRAY_BORDER = "#E0E0E0"
WARNING_RED = "#C62828" # Use sparingly
ACCENT_ORANGE = "#F57C00" # For specific highlights like 'Needs Stabilization'
ACCENT_BLUE = "#1976D2" # For specific highlights like 'Improving'

# --------------------------------------------------------
# Helper Functions
# --------------------------------------------------------

@st.cache_data
def load_data(uploaded_file):
    """
    Reads CSV/XLSX file into a pandas DataFrame, standardizes key columns,
    and sorts by week/store. Expects columns:
      - Store # (or store ID)
      - Week or Date
      - Engaged Transaction %
      - Optional: Weekly Rank, Quarter to Date %, etc.
    """
    if uploaded_file is None:
        return pd.DataFrame() # Return empty dataframe if no file

    filename = uploaded_file.name.lower()
    try:
        if filename.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif filename.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(uploaded_file)
        else:
            st.error("Unsupported file type. Please upload CSV or Excel.")
            return pd.DataFrame()
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return pd.DataFrame()

    # Standardize column names
    df.columns = standardize_columns(df.columns)

    # --- Data Cleaning and Type Conversion ---
    # Ensure required columns exist
    required_cols = ['Store #', 'Engaged Transaction %']
    if 'Date' not in df.columns and 'Week' not in df.columns:
        st.error("Data must contain either a 'Week' or 'Date' column.")
        return pd.DataFrame()
    if not all(col in df.columns for col in required_cols):
        st.error(f"Data must contain columns: {', '.join(required_cols)}")
        return pd.DataFrame()

    # Parse dates
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        # Drop rows where Date could not be parsed
        df = df.dropna(subset=['Date'])
        # Derive Week from Date if Week column doesn't exist
        if 'Week' not in df.columns:
             # Ensure 'Date' is datetime type before using dt accessor
            if pd.api.types.is_datetime64_any_dtype(df['Date']):
                 df['Week'] = df['Date'].dt.isocalendar().week.astype(int)
            else:
                 st.warning("Could not derive 'Week' from 'Date'. Ensure 'Date' column is formatted correctly.")

    # Convert percentage columns to numeric (handle potential errors)
    percent_cols = ['Engaged Transaction %', 'Quarter to Date %']
    for col in percent_cols:
        if col in df.columns:
            # Convert to string first to handle mixed types, remove %, then convert to numeric
            df[col] = pd.to_numeric(df[col].astype(str).str.replace('%', '', regex=False), errors='coerce')

    # Drop rows with missing essential data (Store #, Week/Date, Engagement %)
    essential_cols = ['Store #', 'Engaged Transaction %']
    if 'Week' in df.columns:
        essential_cols.append('Week')
    elif 'Date' in df.columns:
        essential_cols.append('Date')
    df = df.dropna(subset=essential_cols)

    # Convert data types
    if 'Weekly Rank' in df.columns:
        df['Weekly Rank'] = pd.to_numeric(df['Weekly Rank'], errors='coerce').astype('Int64') # integer rank (allow NA)

    df['Store #'] = df['Store #'].astype(str)

    # Ensure Week is integer if present and sort
    if 'Week' in df.columns:
        df['Week'] = pd.to_numeric(df['Week'], errors='coerce').dropna().astype(int)
        df = df.sort_values(['Week', 'Store #'])
    elif 'Date' in df.columns:
         df = df.sort_values(['Date', 'Store #'])


    # Derive Quarter (handle potential errors if Week/Date missing after cleaning)
    if 'Date' in df.columns and pd.api.types.is_datetime64_any_dtype(df['Date']):
        df['Quarter'] = df['Date'].dt.quarter
    elif 'Week' in df.columns and pd.api.types.is_numeric_dtype(df['Week']):
        # Ensure 'Week' is not NaN before calculation
        df['Quarter'] = df['Week'].dropna().apply(lambda w: (int(w) - 1) // 13 + 1 if pd.notna(w) else None).astype('Int64')


    return df


def standardize_columns(columns):
    """
    Renames columns to standard internal names for consistency. More robust checking.
    """
    new_cols = []
    processed_indices = set() # Track indices to avoid renaming twice

    cols_lower = [str(col).strip().lower() for col in columns]

    for i, col_lower in enumerate(cols_lower):
        if i in processed_indices:
            continue

        original_col = columns[i] # Keep original case for fallback

        if 'store' in col_lower and ('#' in col_lower or 'id' in col_lower or 'number' in col_lower):
            new_cols.append('Store #')
        elif 'engage' in col_lower and ('%' in col_lower or 'transaction' in col_lower):
            new_cols.append('Engaged Transaction %')
        elif ('week' in col_lower and 'end' in col_lower) or col_lower == 'date':
             new_cols.append('Date')
        elif col_lower == 'week' or (col_lower.startswith('week') and not any(s in col_lower for s in ['rank', 'end', 'date'])):
            new_cols.append('Week')
        elif 'rank' in col_lower and 'week' in col_lower:
            new_cols.append('Weekly Rank')
        elif 'quarter' in col_lower or 'qtd' in col_lower:
             new_cols.append('Quarter to Date %')
        else:
            new_cols.append(original_col) # Keep original if no match
        processed_indices.add(i)

    # Check for duplicate essential columns after renaming
    from collections import Counter
    col_counts = Counter(new_cols)
    duplicates = [col for col, count in col_counts.items() if count > 1 and col in ['Store #', 'Engaged Transaction %', 'Week', 'Date']]
    if duplicates:
        st.warning(f"Duplicate columns detected after standardization: {', '.join(duplicates)}. Please check your input file headers.")

    return new_cols


def calculate_trend(group, window=4, engagement_col='Engaged Transaction %', week_col='Week'):
    """
    Calculates a trend label based on linear regression slope of the last `window` points.
    More robust calculation and clearer labels. Handles edge cases.
    """
    if engagement_col not in group.columns or week_col not in group.columns:
        return "Missing Data"
    if len(group) < 2:
        return "Insufficient Data" # Need at least 2 points for a trend

    # Ensure data is sorted and get the tail
    sorted_data = group.sort_values(week_col, ascending=True).tail(window)
    # Ensure we still have enough points after tailing
    if len(sorted_data) < 2:
        return "Insufficient Data"

    # Drop rows with NaN engagement in the calculation window
    sorted_data = sorted_data.dropna(subset=[engagement_col])
    if len(sorted_data) < 2:
         return "Insufficient Data" # Not enough valid points

    x = sorted_data[week_col].values
    y = sorted_data[engagement_col].values

    # Check for variance in x and y
    if np.std(x) == 0 or np.std(y) == 0:
        return "Stable" # No change in week or value

    # Perform linear regression
    try:
        # Fit degree 1 polynomial (linear fit)
        coeffs = np.polyfit(x, y, 1)
        slope = coeffs[0]
    except (np.linalg.LinAlgError, ValueError):
        return "Calculation Error" # Handle potential errors in polyfit


    # Define thresholds for trend strength (adjust as needed)
    strong_threshold = 0.5
    mild_threshold = 0.1

    if slope > strong_threshold:
        return "Strong Upward"
    elif slope > mild_threshold:
        return "Upward"
    elif slope < -strong_threshold:
        return "Strong Downward"
    elif slope < -mild_threshold:
        return "Downward"
    else:
        return "Stable"


def find_anomalies(df, z_threshold=2.0, engagement_col='Engaged Transaction %', store_col='Store #', week_col='Week'):
    """
    Calculates week-over-week changes and flags Z-scores exceeding the threshold.
    Returns a DataFrame of anomalies with clearer explanations. Handles potential NaNs.
    """
    if df.empty or not all(col in df.columns for col in [engagement_col, store_col, week_col]):
        return pd.DataFrame() # Return empty if data missing

    anomalies_list = []
    df_sorted = df.sort_values([store_col, week_col])

    for store_id, grp in df_sorted.groupby(store_col):
        # Calculate differences, ensuring alignment and handling NaNs
        grp = grp.dropna(subset=[engagement_col]) # Analyze only valid engagement points
        diffs = grp[engagement_col].diff().dropna() # Calculate differences and remove the initial NaN

        if len(diffs) < 2: # Need at least two differences to calculate std dev reliably
            continue

        mean_diff = diffs.mean()
        std_diff = diffs.std(ddof=0) # Use population std dev if desired, or ddof=1 for sample

        # Skip if standard deviation is zero or NaN (no variability or calculation issue)
        if std_diff == 0 or pd.isna(std_diff):
            continue

        # Iterate through calculated differences
        for idx, diff_val in diffs.items():
            # Check if std_diff is valid before division
            if pd.isna(diff_val) or std_diff == 0 or pd.isna(std_diff):
                continue # Skip if difference or std dev is NaN

            z = (diff_val - mean_diff) / std_diff

            if abs(z) >= z_threshold:
                # Safely get current and previous row data using .loc
                current_row = grp.loc[idx]
                # Find the index of the previous row in the original group 'grp'
                try:
                    prev_idx = diffs.index[diffs.index.get_loc(idx) - 1]
                    # Check if prev_idx is valid and exists in the group index
                    if prev_idx in grp.index:
                         prev_row = grp.loc[prev_idx]
                    else:
                         prev_row = pd.Series(dtype='object') # Empty series if not found
                except (IndexError, KeyError): # Handle index out of bounds or key errors
                    prev_row = pd.Series(dtype='object') # Empty series if no previous index

                # Extract data safely, providing defaults if columns or rows are missing
                week_cur = current_row.get(week_col)
                week_prev = prev_row.get(week_col)
                val_cur = current_row.get(engagement_col)
                val_prev = prev_row.get(engagement_col)
                rank_cur = current_row.get('Weekly Rank') # Optional column
                rank_prev = prev_row.get('Weekly Rank') # Optional column

                # Basic explanation logic
                explanation = ""
                if diff_val >= 0:
                    explanation = "Significant increase in engagement."
                    if pd.notna(rank_prev) and pd.notna(rank_cur) and rank_cur < rank_prev:
                         explanation += f" Rank improved from {int(rank_prev)} to {int(rank_cur)}."
                    elif pd.notna(val_prev):
                         explanation += f" Jumped from {val_prev:.2f}%."
                else:
                    explanation = "Significant decrease in engagement."
                    if pd.notna(rank_prev) and pd.notna(rank_cur) and rank_cur > rank_prev:
                         explanation += f" Rank dropped from {int(rank_prev)} to {int(rank_cur)}."
                    elif pd.notna(val_prev):
                         explanation += f" Dropped from {val_prev:.2f}%."


                anomalies_list.append({
                    'Store #': store_id,
                    'Week': int(week_cur) if pd.notna(week_cur) else None,
                    'Engaged Transaction %': round(val_cur, 2) if pd.notna(val_cur) else None,
                    'Change %pts': round(diff_val, 2) if pd.notna(diff_val) else None,
                    'Z-score': round(z, 2),
                    'Prev Week': int(week_prev) if pd.notna(week_prev) else None,
                    'Rank': int(rank_cur) if pd.notna(rank_cur) else None,
                    'Prev Rank': int(rank_prev) if pd.notna(rank_prev) else None,
                    'Possible Explanation': explanation
                })

    if not anomalies_list:
        return pd.DataFrame()

    anomalies_df = pd.DataFrame(anomalies_list)
    anomalies_df['Abs Z'] = anomalies_df['Z-score'].abs()
    anomalies_df = anomalies_df.sort_values('Abs Z', ascending=False).drop(columns=['Abs Z'])

    return anomalies_df


# --------------------------------------------------------
# Altair Chart Theming Function
# --------------------------------------------------------
def publix_chart_theme():
    """Altair theme settings for Publix branding."""
    font = "sans-serif"
    axis_color = "#BDBDBD" # Lighter gray for axes
    grid_color = "#E0E0E0" # Light grid lines
    primary_color = PUBLIX_GREEN
    secondary_color = "#8BC34A" # Lighter green accent
    gray_color = "#757575"

    return {
        "config": {
            "title": {
                "font": font,
                "fontSize": 16,
                "fontWeight": "bold",
                "color": DARK_TEXT,
                "anchor": "start" # Align title left
            },
            "axis": {
                "labelFont": font,
                "titleFont": font,
                "domainColor": axis_color,
                "tickColor": axis_color,
                "labelColor": MEDIUM_TEXT,
                "titleColor": MEDIUM_TEXT,
                "gridColor": grid_color,
                "labelPadding": 5,
                "titlePadding": 10,
                "titleFontSize": 12,
                "labelFontSize": 11,
            },
             "axisX": {
                 "grid": False # No vertical grid lines typically
            },
            "axisY": {
                "grid": True # Horizontal grid lines okay
             },
            "header": { # For facets/headers in charts
                "labelFont": font,
                "titleFont": font,
                "labelFontSize": 12,
                "titleFontSize": 13,
                "titleColor": DARK_TEXT,
                "labelColor": MEDIUM_TEXT,
            },
            "legend": {
                "labelFont": font,
                "titleFont": font,
                "labelColor": MEDIUM_TEXT,
                "titleColor": MEDIUM_TEXT,
                "padding": 10,
                "symbolSize": 100,
                "titleFontSize": 12,
                "labelFontSize": 11,
                 "orient": "right" # Legend on the right
            },
            "range": { # Define color ranges
                "category": [primary_color, secondary_color, gray_color, ACCENT_ORANGE, ACCENT_BLUE, "#AED581"], # Green/Gray/Accents
                "heatmap": "greens", # Default heatmap to greens
                "ramp": "greens", # Default ramp (sequential) to greens
                "diverging": ["#d73027", "#f7f7f7", primary_color] # Red-White-Green diverging scale
            },
            "view": { # Chart area itself
                "stroke": None # No border around the chart view
            },
            "background": "transparent", # Transparent background
             "mark": { # Default mark properties (can be overridden)
                 "fill": primary_color
             },
             "line": {
                 "stroke": primary_color,
                 "strokeWidth": 2.5
             },
             "bar": {
                  "fill": primary_color,
                  "stroke": None # No outline on bars
              },
            "point": {
                "fill": primary_color,
                "size": 60,
                "stroke": "#FFFFFF", # White outline for visibility
                "strokeWidth": 0.5
            },
            "rule": { # For reference lines
                 "stroke": gray_color,
                 "strokeWidth": 1,
                 "strokeDash": [3, 3]
            }
        }
    }

# Register the theme
alt.themes.register("publix_theme", publix_chart_theme)
alt.themes.enable("publix_theme")


# --------------------------------------------------------
# Streamlit Page Config & Layout
# --------------------------------------------------------

st.set_page_config(
    page_title="Publix District 20 Engagement Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS Injection ---
st.markdown(f"""
<style>
    /* --- Base & Body --- */
    .stApp {{
        background-color: #FFFFFF; /* Force white background */
    }}
    body {{
        font-family: sans-serif; /* Basic, readable font */
        color: {DARK_TEXT}; /* Dark gray for text */
    }}

    /* --- Titles & Headers --- */
    h1, h2, h3, h4, h5, h6 {{
        color: {DARK_TEXT}; /* Consistent dark text for headers */
    }}
    .dashboard-title {{ /* Custom title class */
        color: {PUBLIX_GREEN}; /* Publix Green */
        text-align: center;
        padding-bottom: 20px;
        font-weight: bold;
    }}
    /* Style Streamlit's Subheader */
    [data-testid="stSubheader"] {{
         border-bottom: 2px solid {MEDIUM_GRAY_BORDER}; /* Subtle separator */
         padding-bottom: 8px;
         margin-top: 30px;
         margin-bottom: 20px;
         color: #444444; /* Slightly lighter than main text */
    }}
    /* Target generated h2 elements for consistency if needed */
     div[data-testid="stVerticalBlock"] > div[style*="flex-direction: column;"] > div[data-testid="stVerticalBlock"] > h2 {{
         border-bottom: 2px solid {MEDIUM_GRAY_BORDER};
         padding-bottom: 8px;
         margin-top: 30px;
         margin-bottom: 20px;
         color: #444444;
     }}


    /* --- Sidebar --- */
    [data-testid="stSidebar"] > div:first-child {{
        background-color: {LIGHT_GRAY_BG}; /* Light gray sidebar */
        border-right: 1px solid {MEDIUM_GRAY_BORDER};
    }}
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3, [data-testid="stSidebar"] h4 {{
        color: {DARK_TEXT};
    }}
    /* Sidebar Buttons - Less prominent */
    [data-testid="stSidebar"] .stButton>button {{
        border: 1px solid {PUBLIX_GREEN};
        background-color: transparent;
        color: {PUBLIX_GREEN};
    }}
    [data-testid="stSidebar"] .stButton>button:hover {{
        border-color: #4a752c; /* Darker green */
        color: #4a752c;
        background-color: {LIGHT_GREEN}; /* Light green on hover */
    }}
    [data-testid="stSidebar"] .stButton>button:focus {{
         box-shadow: 0 0 0 2px {LIGHT_GRAY_BG}, 0 0 0 4px {PUBLIX_GREEN}; /* Focus ring for sidebar */
     }}


    /* --- Metric Cards --- */
    [data-testid="stMetric"] {{ /* Style Streamlit's default metric component */
        background-color: #FFFFFF; /* White background */
        border: 1px solid {MEDIUM_GRAY_BORDER};
        border-left: 5px solid {PUBLIX_GREEN}; /* Green accent */
        border-radius: 6px;
        padding: 15px 20px;
        box-shadow: none; /* Cleaner look */
    }}
    [data-testid="stMetric"] label {{ /* Metric label */
        color: {MEDIUM_TEXT};
        font-weight: bold;
        font-size: 0.95em;
    }}
    [data-testid="stMetric"] div:nth-of-type(1) {{ /* Metric value */
        color: {DARK_TEXT};
        font-size: 1.8em; /* Adjust size as needed */
        font-weight: bold;
        line-height: 1.2;
    }}
    [data-testid="stMetric"] div:nth-of-type(2) {{ /* Metric delta */
        font-size: 0.9em;
        color: {MEDIUM_TEXT}; /* Match label color */
    }}
     /* Ensure delta colors work (Streamlit uses inline styles, difficult to override fully) */
    [data-testid="stMetricDelta"] span[style*="color: rgb(46, 125, 50);"] {{ /* Target default green */
        color: {PUBLIX_GREEN} !important;
     }}


    /* --- Custom Highlight Classes (Less necessary with themed metrics/charts) --- */
    .highlight-good {{
        color: {PUBLIX_GREEN};
        font-weight: bold;
    }}
    .highlight-bad {{
        color: {WARNING_RED};
        font-weight: bold;
    }}
    .highlight-neutral {{
        color: {MEDIUM_TEXT};
        font-weight: bold;
    }}

    /* --- Tabs --- */
    [data-testid="stTabs"] [data-baseweb="tab-list"] {{
        border-bottom-color: {MEDIUM_GRAY_BORDER}; /* Lighter tab underline */
        gap: 10px; /* Space between tab headers */
    }}
    [data-testid="stTabs"] [data-baseweb="tab"] {{
        padding: 12px 18px;
        background-color: transparent;
        border-radius: 6px 6px 0 0; /* Slightly rounded top corners */
        color: {MEDIUM_TEXT};
    }}
    [data-testid="stTabs"] [data-baseweb="tab"]:hover {{
        background-color: {LIGHT_GREEN}; /* Light green hover */
        color: {PUBLIX_GREEN};
    }}
    [data-testid="stTabs"] [aria-selected="true"] {{
        border-bottom: 3px solid {PUBLIX_GREEN}; /* Green underline for active tab */
        background-color: #FFFFFF; /* Ensure active tab feels connected to content */
        color: {PUBLIX_GREEN};
        font-weight: bold;
    }}

    /* --- Containers & Cards --- */
    [data-testid="stExpander"] {{
        border: 1px solid {MEDIUM_GRAY_BORDER};
        border-radius: 8px;
        background-color: #FFFFFF; /* White background for expanders */
        box-shadow: 0 1px 3px rgba(0,0,0,0.05); /* Subtle shadow */
    }}
     [data-testid="stExpander"] > details > summary:hover {{
         color: {PUBLIX_GREEN};
     }}
    [data-testid="stExpander"] summary {{
        font-weight: bold;
        color: {MEDIUM_TEXT};
        padding: 10px 15px;
    }}
    div[data-testid="stVerticalBlock"], div[data-testid="stHorizontalBlock"] {{
        gap: 1.2rem; /* Increase gap slightly */
    }}
    /* Custom styled cards (used in Tab 3, Tab 1 Trends) */
    .styled-card {{
         background-color: #FFFFFF;
         border: 1px solid {MEDIUM_GRAY_BORDER};
         padding: 15px;
         border-radius: 6px;
         margin-bottom: 10px;
         box-shadow: 0 1px 2px rgba(0,0,0,0.04);
     }}
     .styled-card h4 {{
         margin-top: 0;
         margin-bottom: 8px;
         font-size: 1.1em;
     }}
      .styled-card p {{
          color: {DARK_TEXT};
          margin-bottom: 5px;
          font-size: 0.95em;
          line-height: 1.4;
      }}
      .styled-card strong {{
          color: {DARK_TEXT}; /* Ensure strong text is also dark */
      }}
      .styled-card .secondary-info {{ /* For less prominent text inside cards */
          font-size: 0.85em;
          color: {MEDIUM_TEXT};
       }}


    /* --- Readability & Misc --- */
    .caption-text {{ /* Your custom caption class */
        font-size: 0.9em;
        color: #666666; /* Medium gray for captions */
    }}
     [data-testid="stCaptionContainer"] {{ /* Style default caption */
         color: #666666;
         font-size: 0.9em;
     }}
    [data-testid="stDataFrame"], [data-testid="stTable"] {{ /* Dataframes */
        border: 1px solid {MEDIUM_GRAY_BORDER};
        border-radius: 4px;
    }}
    [data-testid="stDataFrame"] thead th {{ /* Dataframe header */
        font-weight: bold;
        color: {DARK_TEXT};
        background-color: {LIGHT_GRAY_BG};
     }}

    /* --- Buttons (Main Area) --- */
    .stButton>button {{ /* Default button */
        background-color: {PUBLIX_GREEN};
        color: {LIGHT_TEXT_ON_DARK_BG};
        border: none;
        padding: 10px 20px; /* Slightly larger padding */
        border-radius: 4px;
        font-weight: bold;
    }}
    .stButton>button:hover {{
        background-color: #4a752c; /* Darker green */
        color: {LIGHT_TEXT_ON_DARK_BG};
    }}
    .stButton>button:focus {{
        box-shadow: 0 0 0 2px #FFFFFF, 0 0 0 4px {PUBLIX_GREEN}; /* Focus ring */
        outline: none;
    }}

    /* --- Input Widgets --- */
     [data-testid="stSelectbox"]>div, [data-testid="stMultiselect"]>div,
     [data-testid="stDateInput"]>div, [data-testid="stTextInput"]>div,
     [data-testid="stNumberInput"]>div {{
         border-radius: 4px;
     }}
     [data-testid="stSelectbox"] label, [data-testid="stMultiselect"] label,
     [data-testid="stDateInput"] label, [data-testid="stTextInput"] label,
     [data-testid="stNumberInput"] label, [data-testid="stSlider"] label {{
         color: {MEDIUM_TEXT};
         font-weight: bold;
         font-size: 0.9em;
      }}


</style>
""", unsafe_allow_html=True)

# --------------------------------------------------------
# Title & Introduction
# --------------------------------------------------------

st.markdown("<h1 class='dashboard-title'>Publix District 20 Engagement Dashboard</h1>", unsafe_allow_html=True)
st.markdown("Analyze weekly store engagement data for **Publix Supermarkets – District 20**. "
            "Upload your data using the sidebar to explore performance indicators, trends, and opportunities.")
st.caption("Use the filters on the left to focus on specific time periods or stores.")
st.write("---") # Divider


# --------------------------------------------------------
# Sidebar for Data Upload & Filters
# --------------------------------------------------------

st.sidebar.header("Data Input")
data_file = st.sidebar.file_uploader("Upload Engagement Data (CSV/Excel)", type=['csv', 'xlsx', 'xls'], key="primary_upload")
comp_file = st.sidebar.file_uploader("Optional: Upload Comparison Data", type=['csv', 'xlsx', 'xls'], key="comparison_upload")

# Load data only once after file uploaders are defined
df = load_data(data_file)
df_comp = load_data(comp_file) if comp_file else pd.DataFrame()

if df.empty:
    st.info("Please upload a primary engagement data file using the sidebar to begin analysis.")
    with st.expander("Expected Data Format", expanded=False):
        st.markdown("""
        Your data file should contain columns similar to these:
        - `Store #` or `Store ID`
        - `Week` (numeric week number) or `Date` (parsable date format)
        - `Engaged Transaction %` (as a percentage or number)
        - *Optional*: `Weekly Rank`, `Quarter to Date %`, etc.

        *File types supported: CSV, XLSX, XLS.*
        """)
    st.stop()


# --- Sidebar Filters ---
st.sidebar.header("Filters")

# Quarter Filter (only if 'Quarter' column exists)
quarter_choice = "All"
if 'Quarter' in df.columns and df['Quarter'].notna().any():
    quarters = sorted(df['Quarter'].dropna().unique().astype(int).tolist())
    quarter_options = ["All"] + [f"Q{q}" for q in quarters]
    quarter_choice = st.sidebar.selectbox("Filter by Quarter", quarter_options, index=0)
else:
    st.sidebar.caption("Quarter filter unavailable (missing 'Quarter' data).")


# Week Filter (dependent on Quarter Filter)
week_choice = "All"
if 'Week' in df.columns and df['Week'].notna().any():
     # Determine available weeks based on selected quarter
    if quarter_choice != "All":
        try:
            q_num = int(quarter_choice[1:])
            available_weeks = sorted(df.loc[df['Quarter'] == q_num, 'Week'].dropna().unique().astype(int).tolist())
        except (ValueError, IndexError):
             st.sidebar.warning("Invalid Quarter selection.")
             available_weeks = sorted(df['Week'].dropna().unique().astype(int).tolist()) # Fallback
    else:
        available_weeks = sorted(df['Week'].dropna().unique().astype(int).tolist())

    if available_weeks:
         week_options = ["All"] + [str(w) for w in available_weeks]
         week_choice = st.sidebar.selectbox("Filter by Week", week_options, index=0, help="Select 'All' to see data across all available weeks in the chosen quarter.")
    else:
         st.sidebar.caption("No weeks available for the selected quarter.")
else:
    st.sidebar.caption("Week filter unavailable (missing 'Week' data).")


# Store Filter
if 'Store #' in df.columns:
    store_list = sorted(df['Store #'].dropna().unique().tolist())
    if store_list:
        store_choice = st.sidebar.multiselect("Filter by Store(s)", store_list, default=[], help="Select one or more stores. Leave blank to include all.")
    else:
        store_choice = []
        st.sidebar.caption("No store data available.")
else:
    store_choice = []
    st.sidebar.caption("Store filter unavailable (missing 'Store #' data).")


# --- Advanced Settings ---
st.sidebar.markdown("---")
st.sidebar.subheader("Analysis Settings")
with st.sidebar.expander("Adjust Calculations", expanded=False):
    z_threshold = st.slider("Anomaly Sensitivity (Z-score)", 1.0, 3.0, 2.0, 0.1, help="Lower threshold = more anomalies flagged.")
    trend_analysis_weeks = st.slider("Trend Window (Weeks)", 3, 8, 4, help="Number of recent weeks used for trend calculation.")
    show_ma = st.checkbox("Show 4-week Moving Average on Trend Chart", value=True)
    # Removed highlight top/bottom - use chart interactions instead


# --- Filter DataFrames ---
df_filtered = df.copy()
df_comp_filtered = df_comp.copy() if not df_comp.empty else None

# Apply Quarter Filter
if quarter_choice != "All" and 'Quarter' in df_filtered.columns:
    try:
        q_num_filter = int(quarter_choice[1:])
        df_filtered = df_filtered[df_filtered['Quarter'] == q_num_filter]
        if df_comp_filtered is not None and 'Quarter' in df_comp_filtered.columns:
            df_comp_filtered = df_comp_filtered[df_comp_filtered['Quarter'] == q_num_filter]
    except (ValueError, IndexError):
        pass # Ignore invalid quarter selection

# Apply Week Filter
if week_choice != "All" and 'Week' in df_filtered.columns:
    try:
        week_num_filter = int(week_choice)
        df_filtered = df_filtered[df_filtered['Week'] == week_num_filter]
        if df_comp_filtered is not None and 'Week' in df_comp_filtered.columns:
            df_comp_filtered = df_comp_filtered[df_comp_filtered['Week'] == week_num_filter]
    except ValueError:
         pass # Ignore invalid week selection

# Apply Store Filter
if store_choice and 'Store #' in df_filtered.columns:
    df_filtered = df_filtered[df_filtered['Store #'].isin(store_choice)]
    if df_comp_filtered is not None and 'Store #' in df_comp_filtered.columns:
        df_comp_filtered = df_comp_filtered[df_comp_filtered['Store #'].isin(store_choice)]


if df_filtered.empty:
    st.error("No data available for the selected filters. Please adjust filters or upload new data.")
    st.stop()

# --- Calculate Moving Averages after filtering ---
if 'Week' in df_filtered.columns and 'Store #' in df_filtered.columns and 'Engaged Transaction %' in df_filtered.columns:
    df_filtered = df_filtered.sort_values(['Store #', 'Week'])
    df_filtered['MA_4W'] = df_filtered.groupby('Store #')['Engaged Transaction %']\
        .transform(lambda s: s.rolling(window=4, min_periods=1).mean())


# --------------------------------------------------------
# Executive Summary Calculations & Display
# --------------------------------------------------------
st.subheader("Executive Summary")

# Identify current/previous week based on filtered data
current_week, prev_week = None, None
if 'Week' in df_filtered.columns:
    valid_weeks = df_filtered['Week'].dropna().astype(int)
    if not valid_weeks.empty:
        if week_choice != "All":
            current_week = int(week_choice)
            prev_weeks_options = valid_weeks[valid_weeks < current_week]
            if not prev_weeks_options.empty:
                prev_week = int(prev_weeks_options.max())
        else:
            current_week = int(valid_weeks.max())
            prev_weeks_options = valid_weeks[valid_weeks < current_week]
            if not prev_weeks_options.empty:
                prev_week = int(prev_weeks_options.max())

# Calculate metrics
current_avg, prev_avg, delta_val, delta_pct = None, None, None, None
top_store, bottom_store, top_val, bottom_val = None, None, None, None

if 'Engaged Transaction %' in df_filtered.columns:
    # District/Selection average
    if current_week is not None:
        current_avg = df_filtered.loc[df_filtered['Week'] == current_week, 'Engaged Transaction %'].mean()
    if prev_week is not None:
        prev_avg = df_filtered.loc[df_filtered['Week'] == prev_week, 'Engaged Transaction %'].mean()

    if pd.notna(current_avg) and pd.notna(prev_avg):
        delta_val = current_avg - prev_avg
        if prev_avg != 0:
             delta_pct = (delta_val / prev_avg) * 100
        else:
             delta_pct = 0 # Avoid division by zero

    # Top/Bottom performer (based on current week if selected, otherwise overall average)
    if 'Store #' in df_filtered.columns:
        if current_week is not None:
            current_week_perf = df_filtered.loc[df_filtered['Week'] == current_week].groupby('Store #')['Engaged Transaction %'].mean()
            if not current_week_perf.empty:
                 top_store = current_week_perf.idxmax()
                 bottom_store = current_week_perf.idxmin()
                 top_val = current_week_perf.max()
                 bottom_val = current_week_perf.min()
        # Fallback to overall average if no specific week or no data for current week
        if top_store is None:
             overall_perf = df_filtered.groupby('Store #')['Engaged Transaction %'].mean()
             if not overall_perf.empty:
                 top_store = overall_perf.idxmax()
                 bottom_store = overall_perf.idxmin()
                 top_val = overall_perf.max()
                 bottom_val = overall_perf.min()


# Display Metrics
col1, col2, col3 = st.columns(3)

# Metric 1: Average Engagement
avg_label = "District Avg Engagement"
if store_choice:
    if len(store_choice) == 1:
        avg_label = f"Store {store_choice[0]} Engagement"
    elif len(store_choice) < len(store_list):
        avg_label = f"Selected Stores Avg ({len(store_choice)})"

metric_period = f"(Week {current_week})" if current_week is not None else "(Period Avg)"
avg_display_val = f"{current_avg:.2f}%" if pd.notna(current_avg) else "N/A"
delta_display_str = f"{delta_val:+.2f} pts" if pd.notna(delta_val) else None # More descriptive delta

col1.metric(
    f"{avg_label} {metric_period}",
    avg_display_val,
    delta=delta_display_str,
    delta_color="normal" if pd.notna(delta_val) else "off" # Auto color delta
)

# Metric 2: Top Performer
top_label = f"Top Performer {metric_period}"
top_display_val = f"Store {top_store}" if top_store else "N/A"
top_help = f"{top_val:.2f}%" if pd.notna(top_val) else None
col2.metric(top_label, top_display_val, help=top_help) # Use help for the value

# Metric 3: Bottom Performer
bottom_label = f"Bottom Performer {metric_period}"
bottom_display_val = f"Store {bottom_store}" if bottom_store else "N/A"
bottom_help = f"{bottom_val:.2f}%" if pd.notna(bottom_val) else None
col3.metric(bottom_label, bottom_display_val, help=bottom_help) # Use help for the value


# Trend Indicator Text (simpler)
if pd.notna(delta_val):
    trend_direction = "up" if delta_val > 0 else "down" if delta_val < 0 else "flat"
    trend_class = "highlight-good" if delta_val > 0 else "highlight-bad" if delta_val < 0 else "highlight-neutral"
    st.markdown(
        f"Compared to Week {prev_week}, the average engagement is "
        f"<span class='{trend_class}'>{trend_direction} by {abs(delta_val):.2f} percentage points</span>.",
        unsafe_allow_html=True
    )
elif pd.notna(current_avg):
     st.markdown(f"Current average engagement: <span class='highlight-neutral'>{current_avg:.2f}%</span>", unsafe_allow_html=True)

st.write("---") # Divider


# --------------------------------------------------------
# Key Insights (Simplified)
# --------------------------------------------------------
st.subheader("Key Insights")
insights = []

if 'Store #' in df_filtered.columns and 'Engaged Transaction %' in df_filtered.columns:
    store_stats = df_filtered.groupby('Store #')['Engaged Transaction %'].agg(['mean', 'std']).reset_index()
    store_stats['std'] = store_stats['std'].fillna(0) # Handle single data point stores

    if not store_stats.empty and len(store_stats) > 1:
        # 1) Consistency
        store_stats_sorted_std = store_stats.sort_values('std')
        most_consistent = store_stats_sorted_std.iloc[0]['Store #']
        least_consistent = store_stats_sorted_std.iloc[-1]['Store #']
        insights.append(f"**Most Consistent:** Store **{most_consistent}** (Lowest performance variability).")
        insights.append(f"**Most Variable:** Store **{least_consistent}** (Highest performance variability).")

        # 2) Performance Gap
        engagement_gap = store_stats['mean'].max() - store_stats['mean'].min()
        insights.append(f"**Performance Gap:** **{engagement_gap:.2f}%** between the highest and lowest average performers.")
        if engagement_gap > 10: # Threshold for large gap
            insights.append("Consider knowledge sharing opportunities between top and bottom performers.")

    # 3) Trend analysis (using the calculate_trend function)
    if 'Week' in df_filtered.columns:
        store_trends = df_filtered.groupby('Store #').apply(
            lambda x: calculate_trend(x, trend_analysis_weeks)
        ).reset_index(name='Trend')

        trending_up = store_trends[store_trends['Trend'].isin(["Upward", "Strong Upward"])]['Store #'].tolist()
        trending_down = store_trends[store_trends['Trend'].isin(["Downward", "Strong Downward"])]['Store #'].tolist()

        if trending_up:
            insights.append(f"**Positive Trends:** Stores showing recent upward movement: **{', '.join(trending_up)}**.")
        if trending_down:
            insights.append(f"**Negative Trends:** Stores showing recent downward movement: **{', '.join(trending_down)}**.")

if insights:
    for i, insight in enumerate(insights[:5], start=1): # Limit insights shown initially
        st.markdown(f"▪️ {insight}")
else:
    st.write("Further insights will be generated once more data or filters are applied.")

st.write("---") # Divider

# --------------------------------------------------------
# Main Tabs
# --------------------------------------------------------

tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Engagement Trends",
    "📊 Store Comparison",
    "📋 Performance Categories",
    "🔍 Anomalies & Insights"
])


# --------------------------------------------------------
# TAB 1: Engagement Trends
# --------------------------------------------------------
with tab1:
    st.subheader("Engagement Trends Over Time")

    # Combine current and comparison period if available
    # Add 'Period' column for differentiation in charts
    if df_comp_filtered is not None and not df_comp_filtered.empty and all(col in df_comp_filtered for col in ['Store #', 'Week', 'Engaged Transaction %']):
        df_filtered['Period'] = 'Current Period'
        # Ensure comparison df has the same needed columns before concat
        comp_cols_needed = ['Store #', 'Week', 'Engaged Transaction %', 'MA_4W'] # Add MA_4W if calculated
        df_comp_filtered['Period'] = 'Comparison Period'
        # Calculate MA for comparison period if needed
        df_comp_filtered = df_comp_filtered.sort_values(['Store #', 'Week'])
        df_comp_filtered['MA_4W'] = df_comp_filtered.groupby('Store #')['Engaged Transaction %']\
            .transform(lambda s: s.rolling(window=4, min_periods=1).mean())

        combined_df = pd.concat([df_filtered, df_comp_filtered], ignore_index=True)
        period_color_scale = alt.Scale(domain=['Current Period', 'Comparison Period'], range=[PUBLIX_GREEN, MEDIUM_TEXT])
        period_dash_scale = alt.Scale(domain=['Current Period', 'Comparison Period'], range=[[1,0], [3,3]]) # Solid line for current, dashed for comparison

    else:
        combined_df = df_filtered.copy()
        combined_df['Period'] = 'Current Period' # Still add Period column for consistency
        period_color_scale = alt.Scale(range=[PUBLIX_GREEN]) # Only one color needed
        period_dash_scale = alt.Scale(range=[[1,0]]) # Solid line


    # --- Line Chart: Engagement Over Time ---
    st.markdown("#### Weekly Engagement Percentage")

    if 'Week' not in combined_df.columns or 'Engaged Transaction %' not in combined_df.columns:
         st.warning("Cannot display trend chart. Missing 'Week' or 'Engaged Transaction %' data.")
    else:
        # Base chart
        base = alt.Chart(combined_df).encode(
            x=alt.X('Week:O', title='Week', axis=alt.Axis(labelAngle=0)), # Ensure labels are horizontal
            tooltip=[
                'Store #',
                'Week:O',
                alt.Tooltip('Engaged Transaction %:Q', format='.2f', title='Engagement %'),
                'Period:N'
            ]
        )

        # Main engagement lines
        lines = base.mark_line(point=False).encode( # Point=False for cleaner lines, maybe add later
            y=alt.Y('Engaged Transaction %:Q', title='Engaged Transaction %', scale=alt.Scale(zero=False)), # Don't force y-axis to zero
            color=alt.Color('Store #:N', title='Store'), # Color by store
            strokeDash=alt.StrokeDash('Period:N', scale=period_dash_scale, title='Period') # Dash by period
        )

        # Optional Moving Average lines
        ma_lines = base.mark_line(strokeWidth=1.5, opacity=0.6).encode(
             y=alt.Y('MA_4W:Q'), # Using the precalculated MA column
             color=alt.Color('Store #:N'), # Match store color
             strokeDash=alt.StrokeDash('Period:N', scale=period_dash_scale) # Match period dash
         ).properties(
             title="4-Week Moving Average" # Add title to make it clear
         )

        # Interactive Legend (Selection)
        store_selection = alt.selection_point(fields=['Store #'], bind='legend')

        # Combine lines and add interactivity
        chart = lines.add_params(
            store_selection
        ).encode(
            opacity=alt.condition(store_selection, alt.value(1), alt.value(0.2)) # Dim non-selected stores
        )

        if show_ma and 'MA_4W' in combined_df.columns:
            chart_final = alt.layer(chart, ma_lines).resolve_scale(y='shared')
        else:
            chart_final = chart


        # District Average Line (calculated from df_filtered only)
        if not store_choice: # Only show district avg if all stores are viewed
            district_avg_trend = df_filtered.groupby('Week')['Engaged Transaction %'].mean().reset_index()
            if not district_avg_trend.empty:
                 dist_avg_line = alt.Chart(district_avg_trend).mark_line(
                     color='black', strokeDash=[5,5], size=2.5
                 ).encode(
                     x='Week:O',
                     y=alt.Y('Engaged Transaction %:Q'),
                     tooltip=[alt.Tooltip('Engaged Transaction %:Q', format='.2f', title='District Average')]
                 ).properties(title="District Average") # Helps identify the line
                 chart_final = alt.layer(chart_final, dist_avg_line).resolve_scale(y='shared')


        st.altair_chart(
             chart_final.interactive().properties(height=400), # Enable zoom/pan
             use_container_width=True
         )
        st.caption("Click legend items to highlight specific stores. Use mouse wheel/trackpad to zoom and pan.")
        if show_ma:
             st.caption("Fainter lines represent the 4-week moving average.")
        if df_comp_filtered is not None:
             st.caption("Solid lines: Current Period. Dashed lines: Comparison Period.")


    # --- Heatmap ---
    st.markdown("#### Weekly Engagement Heatmap")
    if 'Week' not in df_filtered.columns or 'Store #' not in df_filtered.columns or 'Engaged Transaction %' not in df_filtered.columns:
         st.warning("Cannot display heatmap. Missing required data.")
    else:
        with st.expander("Heatmap Settings", expanded=False):
            col1, col2 = st.columns([1, 1])
            with col1:
                sort_method = st.selectbox(
                    "Sort stores by:",
                    ["Average Engagement", "Recent Performance"],
                    index=0, help="Order stores vertically in the heatmap."
                )
            with col2:
                normalize_colors = st.checkbox(
                    "Normalize colors by week", value=False,
                    help="Color intensity relative to each week (vs. overall)."
                )

        # Filter data for heatmap (copy to avoid modifying df_filtered)
        heatmap_data = df_filtered[['Store #', 'Week', 'Engaged Transaction %']].copy()
        heatmap_data = heatmap_data.rename(columns={'Store #': 'StoreID', 'Engaged Transaction %': 'EngagedPct'})

        if heatmap_data.empty or heatmap_data['EngagedPct'].isna().all():
            st.info("No data available for the heatmap based on current filters.")
        else:
            # Sort stores
            if sort_method == "Average Engagement":
                store_avg = heatmap_data.groupby('StoreID')['EngagedPct'].mean().reset_index()
                store_order = store_avg.sort_values('EngagedPct', ascending=False)['StoreID'].tolist()
            else: # Recent Performance
                 most_recent_week = heatmap_data['Week'].max()
                 recent_perf = heatmap_data[heatmap_data['Week'] == most_recent_week]
                 store_order = recent_perf.sort_values('EngagedPct', ascending=False)['StoreID'].tolist()

            # Handle normalization
            if normalize_colors:
                 week_stats = heatmap_data.groupby('Week')['EngagedPct'].agg(['min', 'max']).reset_index()
                 heatmap_data = pd.merge(heatmap_data, week_stats, on='Week', how='left')
                 # Avoid division by zero
                 heatmap_data['NormalizedPct'] = heatmap_data.apply(
                     lambda row: 0 if row['min'] == row['max'] or pd.isna(row['min']) else
                     100 * (row['EngagedPct'] - row['min']) / (row['max'] - row['min']),
                     axis=1
                 )
                 color_field = 'NormalizedPct:Q'
                 color_title = 'Normalized %'
                 color_scheme = 'greens' # Use greens for normalized
            else:
                 color_field = 'EngagedPct:Q'
                 color_title = 'Engaged %'
                 color_scheme = 'greens' # Consistent green scheme


            heatmap_chart = alt.Chart(heatmap_data).mark_rect().encode(
                 x=alt.X('Week:O', title='Week'),
                 y=alt.Y('StoreID:O', title='Store', sort=store_order), # Apply sort order
                 color=alt.Color(
                     color_field,
                     title=color_title,
                     scale=alt.Scale(scheme=color_scheme),
                     legend=alt.Legend(orient='right', titleFontSize=11, labelFontSize=10)
                 ),
                 tooltip=['StoreID', 'Week:O', alt.Tooltip('EngagedPct:Q', format='.2f', title='Engagement %')]
             ).properties(
                 height=max(200, len(store_order) * 20) # Dynamic height
             )

            st.altair_chart(heatmap_chart, use_container_width=True)
            st.caption(f"Stores sorted by {sort_method.lower()}. {'Colors normalized within each week.' if normalize_colors else 'Global color scale used.'} Darker colors represent higher engagement.")


    # --- Recent Performance Trends Section ---
    st.markdown("#### Recent Performance Analysis")
    st.write("Highlights short-term direction (improving, stable, declining) over the last few weeks.")

    # Re-use trend_analysis_weeks slider from sidebar
    trend_window = trend_analysis_weeks

    # Sensitivity (simplified)
    sensitivity = st.select_slider(
        "Sensitivity to Change",
        options=["Low", "Medium", "High"], value="Medium",
        help="High sensitivity detects smaller changes.", key="trend_sensitivity"
    )
    momentum_threshold = {"Low": 0.5, "Medium": 0.3, "High": 0.1}[sensitivity]

    store_directions = []
    if 'Store #' in df_filtered.columns and 'Week' in df_filtered.columns and 'Engaged Transaction %' in df_filtered.columns:
         analysis_df = df_filtered.rename(columns={'Store #': 'StoreID', 'Engaged Transaction %': 'EngagedPct'})

         for store_id, store_data in analysis_df.groupby('StoreID'):
             store_data_sorted = store_data.sort_values('Week')
             if len(store_data_sorted) < trend_window: continue # Skip if not enough data

             recent_data = store_data_sorted.tail(trend_window).dropna(subset=['EngagedPct'])
             if len(recent_data) < 2: continue # Need at least 2 points

             # Use slope from polyfit for trend direction (more robust than simple average diff)
             x = recent_data['Week'].values
             y = recent_data['EngagedPct'].values
             try:
                 coeffs = np.polyfit(x, y, 1)
                 slope = coeffs[0]
             except (np.linalg.LinAlgError, ValueError):
                 slope = 0 # Treat calculation errors as stable

             # Determine direction based on slope and threshold
             if abs(slope) < momentum_threshold:
                 direction = "Stable"
                 strength = "Holding Steady"
                 color = ACCENT_BLUE # Blue for Stable
                 indicator = "➡️"
             elif slope > 0:
                 direction = "Improving"
                 strength = "Strong Improvement" if slope > momentum_threshold * 1.5 else "Gradual Improvement"
                 color = PUBLIX_GREEN # Green for Improving
                 indicator = "🔼" if strength == "Strong Improvement" else "↗️"
             else:
                 direction = "Declining"
                 strength = "Significant Decline" if slope < -momentum_threshold * 1.5 else "Gradual Decline"
                 color = WARNING_RED # Red for Declining
                 indicator = "🔽" if strength == "Significant Decline" else "↘️"


             start_value = recent_data.iloc[0]['EngagedPct']
             current_value = recent_data.iloc[-1]['EngagedPct']
             total_change = current_value - start_value

             store_directions.append({
                 'store': store_id, 'direction': direction, 'strength': strength,
                 'indicator': indicator, 'start_value': start_value,
                 'current_value': current_value, 'total_change': total_change,
                 'color': color, 'weeks': len(recent_data), 'slope': slope
             })

    if not store_directions:
        st.info(f"Not enough data points (need at least {trend_window}) within the filtered range to analyze recent trends.")
    else:
        direction_df = pd.DataFrame(store_directions)
        direction_order = {"Improving": 0, "Stable": 1, "Declining": 2}
        direction_df['direction_order'] = direction_df['direction'].map(direction_order)
        sorted_stores = direction_df.sort_values(['direction_order', 'slope'], ascending=[True, False])

        # Display summary metrics
        col1, col2, col3 = st.columns(3)
        improving_count = len(direction_df[direction_df['direction'] == 'Improving'])
        stable_count = len(direction_df[direction_df['direction'] == 'Stable'])
        declining_count = len(direction_df[direction_df['direction'] == 'Declining'])

        col1.metric("Improving", f"{improving_count} stores", delta="↗️", delta_color="off")
        col2.metric("Stable", f"{stable_count} stores", delta="➡️", delta_color="off")
        col3.metric("Declining", f"{declining_count} stores", delta="↘️", delta_color="off")


        # Display store cards grouped by direction
        for direction in ['Improving', 'Stable', 'Declining']:
            direction_data = sorted_stores[sorted_stores['direction'] == direction]
            if direction_data.empty: continue

            color = direction_data.iloc[0]['color'] # Get color for the group
            st.markdown(f"""
            <div style="border-left: 5px solid {color}; padding-left: 10px; margin-top: 20px; margin-bottom: 10px;">
                <h4 style="color: {color}; margin-bottom: 5px;">{direction} ({len(direction_data)} stores)</h4>
            </div>
            """, unsafe_allow_html=True)

            cols_per_row = 3
            num_rows = (len(direction_data) + cols_per_row - 1) // cols_per_row
            store_indices = direction_data.index

            for row_idx in range(num_rows):
                cols = st.columns(cols_per_row)
                start_idx = row_idx * cols_per_row
                end_idx = start_idx + cols_per_row
                current_row_indices = store_indices[start_idx:end_idx]

                for i, store_index in enumerate(current_row_indices):
                     store_info = direction_data.loc[store_index]
                     with cols[i]:
                         change_display = f"{store_info['total_change']:+.2f}%"
                         # Use the styled card CSS class
                         st.markdown(f"""
                         <div class="styled-card" style="border-left: 5px solid {store_info['color']};">
                             <h4 style="text-align: center; color: {store_info['color']};">
                                 {store_info['indicator']} Store {store_info['store']}
                             </h4>
                             <p style="text-align: center;">
                                 <strong>{store_info['strength']}</strong><br>
                                 <span style="font-size: 0.9em;">
                                     <strong>{change_display}</strong> over {store_info['weeks']} weeks
                                 </span><br>
                                 <span class="secondary-info">
                                     ({store_info['start_value']:.2f}% → {store_info['current_value']:.2f}%)
                                 </span>
                             </p>
                         </div>
                         """, unsafe_allow_html=True)

        # --- Trend Change Bar Chart ---
        st.markdown("#### Change in Engagement (Recent Period)")
        change_chart = alt.Chart(direction_df).mark_bar().encode(
            x=alt.X('total_change:Q', title=f'Engagement % Change (Last {trend_window} Weeks)'),
            y=alt.Y('store:N', title='Store', sort=alt.EncodingSortField(field='total_change', order='descending')),
            color=alt.Color('direction:N',
                            scale=alt.Scale(domain=['Improving', 'Stable', 'Declining'],
                                            range=[PUBLIX_GREEN, ACCENT_BLUE, WARNING_RED]), # Themed colors
                            title="Direction"),
            tooltip=[
                alt.Tooltip('store:N', title='Store'),
                alt.Tooltip('direction:N', title='Direction'),
                alt.Tooltip('strength:N', title='Performance Detail'),
                alt.Tooltip('start_value:Q', title=f'Start % (Week {direction_df["weeks"].min() if not direction_df.empty else ""})', format='.2f'), # Approx start week
                alt.Tooltip('current_value:Q', title='End %', format='.2f'),
                alt.Tooltip('total_change:Q', title='Total Change', format='+.2f')
            ]
        ).properties(
            height=max(250, len(direction_df) * 25)
        )

        zero_line = alt.Chart(pd.DataFrame({'x': [0]})).mark_rule(color=DARK_TEXT, strokeDash=[2, 2]).encode(x='x:Q')
        st.altair_chart(change_chart + zero_line, use_container_width=True)
        st.caption("Compares engagement percentage at the start and end of the selected trend analysis window.")



# --------------------------------------------------------
# TAB 2: Store Comparison
# --------------------------------------------------------
with tab2:
    st.subheader("Store Performance Comparison")

    comp_period_label = f"Week {week_choice}" if week_choice != "All" else "Selected Period Average"

    if 'Store #' not in df_filtered.columns or 'Engaged Transaction %' not in df_filtered.columns:
        st.warning("Cannot display comparisons. Missing 'Store #' or 'Engaged Transaction %' data.")
    elif len(df_filtered['Store #'].unique()) < 2:
        st.info("At least two stores must be selected or available in the data to show comparisons.")
    else:
        # --- Comparison Bar Chart ---
        st.markdown(f"#### Engagement Percentage by Store ({comp_period_label})")
        if week_choice != "All":
            # Ensure week_choice is valid before filtering
            try:
                week_num_comp = int(week_choice)
                comp_data = df_filtered[df_filtered['Week'] == week_num_comp].copy()
                # Check if data exists for the specific week
                if comp_data.empty:
                     st.warning(f"No data found for Week {week_num_comp}. Showing period average instead.")
                     comp_data = df_filtered.groupby('Store #', as_index=False)['Engaged Transaction %'].mean()
                else:
                     # Ensure we take the mean if multiple entries exist for a store in a week (shouldn't happen with good data)
                     comp_data = comp_data.groupby('Store #', as_index=False)['Engaged Transaction %'].mean()
            except ValueError:
                 st.warning("Invalid week selected. Showing period average.")
                 comp_data = df_filtered.groupby('Store #', as_index=False)['Engaged Transaction %'].mean()

        else:
            comp_data = df_filtered.groupby('Store #', as_index=False)['Engaged Transaction %'].mean()


        comp_data = comp_data.sort_values('Engaged Transaction %', ascending=False)

        bar_chart = alt.Chart(comp_data).mark_bar().encode(
            y=alt.Y('Store #:N', title='Store', sort='-x'), # Sort bars descending
            x=alt.X('Engaged Transaction %:Q', title='Engaged Transaction %'),
            color=alt.Color('Engaged Transaction %:Q',
                            scale=alt.Scale(scheme='greens'), # Use green scale
                            legend=None), # No legend needed for single color gradient
            tooltip=[
                'Store #:N',
                alt.Tooltip('Engaged Transaction %:Q', format='.2f', title='Engagement %')
            ]
        ).properties(
             height=alt.Step(20) # Adjust height per bar automatically
        )

        # Add District Average Rule Line
        district_avg_comp = comp_data['Engaged Transaction %'].mean()
        rule = alt.Chart(pd.DataFrame({'avg': [district_avg_comp]})).mark_rule(
             color='black', strokeDash=[3,3], size=2
         ).encode(
             x='avg:Q',
             tooltip=[alt.Tooltip('avg:Q', title='District Average', format='.2f')]
         )

        st.altair_chart(bar_chart + rule, use_container_width=True)
        st.caption(f"Average engagement across selected stores/period: {district_avg_comp:.2f}%. Dashed line indicates the average.")


        # --- Performance Relative to Average Chart ---
        st.markdown("#### Performance Relative to Average")
        comp_data['Difference'] = comp_data['Engaged Transaction %'] - district_avg_comp
        comp_data['Percentage'] = (comp_data['Difference'] / district_avg_comp * 100) if district_avg_comp != 0 else 0

        # Use diverging color scale centered at 0
        min_perc = comp_data['Percentage'].min()
        max_perc = comp_data['Percentage'].max()
        # Ensure the domain covers zero, even if all values are pos/neg
        domain_max = max(abs(min_perc), abs(max_perc))
        color_domain = [-domain_max, 0, domain_max] # Centered domain


        diff_chart = alt.Chart(comp_data).mark_bar().encode(
             y=alt.Y('Store #:N', title='Store', sort=alt.EncodingSortField(field='Difference', order='descending')), # Sort by difference
             x=alt.X('Percentage:Q', title='% Difference from Average'),
             color=alt.Color(
                 'Percentage:Q',
                 scale=alt.Scale(domain=color_domain, range=[WARNING_RED, LIGHT_GRAY_BG, PUBLIX_GREEN]), # Red-Gray-Green
                 legend=None # Legend not practical here
             ),
             tooltip=[
                 'Store #:N',
                 alt.Tooltip('Engaged Transaction %:Q', format='.2f', title='Actual %'),
                 alt.Tooltip('Percentage:Q', format='+,.2f', title='% Diff from Avg') # Include + sign
             ]
         ).properties(height=alt.Step(20))

        center_rule = alt.Chart(pd.DataFrame({'center': [0]})).mark_rule(color='black', size=1).encode(x='center:Q')

        st.altair_chart(diff_chart + center_rule, use_container_width=True)
        st.caption(f"Shows percentage difference from the average ({district_avg_comp:.2f}%). Green = above average, Red = below average.")


        # --- Weekly Rank Tracking ---
        if 'Weekly Rank' in df_filtered.columns and 'Week' in df_filtered.columns:
            st.markdown("#### Weekly Rank Tracking")
            rank_data = df_filtered[['Week', 'Store #', 'Weekly Rank']].dropna()
            rank_data['Weekly Rank'] = rank_data['Weekly Rank'].astype(int) # Ensure integer for scale

            if not rank_data.empty:
                # Find min/max rank for dynamic y-axis scaling (inverted)
                min_rank = rank_data['Weekly Rank'].max() # Max rank value is worst (bottom of axis)
                max_rank = rank_data['Weekly Rank'].min() # Min rank value is best (top of axis)

                rank_chart = alt.Chart(rank_data).mark_line(point=True).encode(
                    x=alt.X('Week:O', title='Week'),
                    y=alt.Y('Weekly Rank:Q',
                            title='Rank (1 = Best)',
                            scale=alt.Scale(domain=[min_rank + 0.5, max_rank - 0.5])), # Inverted scale with padding
                    color=alt.Color('Store #:N', title='Store'),
                    tooltip=['Store #', 'Week:O', 'Weekly Rank:Q']
                ).properties(
                    height=350
                )

                # Add interactivity
                rank_selection = alt.selection_point(fields=['Store #'], bind='legend')
                rank_chart_interactive = rank_chart.add_params(rank_selection).encode(
                    opacity=alt.condition(rank_selection, alt.value(1), alt.value(0.2))
                )

                st.altair_chart(rank_chart_interactive.interactive(), use_container_width=True)
                st.caption("Lower rank number indicates better performance. Click legend to highlight.")
            else:
                st.info("Weekly rank data not available or not numeric for the selected filters.")
        else:
             st.caption("Weekly Rank data not found in the uploaded file.")


# --------------------------------------------------------
# TAB 3: Store Performance Categories
# --------------------------------------------------------
with tab3:
    st.subheader("Store Performance Categories")
    st.write("Stores are categorized based on average engagement and recent performance trend.")

    if not all(col in df_filtered.columns for col in ['Store #', 'Week', 'Engaged Transaction %']):
         st.warning("Cannot categorize stores. Missing 'Store #', 'Week', or 'Engaged Transaction %' data.")
    else:
        # --- Category Calculation ---
        store_stats_cat = df_filtered.groupby('Store #')['Engaged Transaction %'].agg(['mean', 'std']).reset_index()
        store_stats_cat.columns = ['Store #', 'Average Engagement', 'Std Dev']
        store_stats_cat['Std Dev'] = store_stats_cat['Std Dev'].fillna(0.0)

        # Calculate trend correlation (Week vs. Engagement) - more robust trend indicator
        trend_corr_data = []
        for store_id, grp in df_filtered.groupby('Store #'):
            if len(grp) >= 3: # Need at least 3 points for meaningful correlation
                 grp = grp.sort_values('Week').dropna(subset=['Engaged Transaction %'])
                 if len(grp) >= 2: # Still need at least 2 points after dropna
                     # Check for variance before calculating correlation
                     if grp['Week'].nunique() > 1 and grp['Engaged Transaction %'].nunique() > 1:
                         corr_val = grp[['Week', 'Engaged Transaction %']].corr().iloc[0, 1]
                         trend_corr_data.append({'Store #': store_id, 'Trend Correlation': corr_val})
                     else:
                          trend_corr_data.append({'Store #': store_id, 'Trend Correlation': 0.0}) # No variance = stable trend
                 else:
                     trend_corr_data.append({'Store #': store_id, 'Trend Correlation': 0.0}) # Not enough points = stable trend

            else:
                trend_corr_data.append({'Store #': store_id, 'Trend Correlation': 0.0}) # Treat stores with <3 points as stable trend

        if trend_corr_data:
             trend_df_cat = pd.DataFrame(trend_corr_data)
             store_stats_cat = store_stats_cat.merge(trend_df_cat, on='Store #', how='left')
             store_stats_cat['Trend Correlation'] = store_stats_cat['Trend Correlation'].fillna(0.0)
        else:
             store_stats_cat['Trend Correlation'] = 0.0


        # Define Categories based on Median Engagement and Trend Correlation
        if not store_stats_cat.empty:
             median_engagement = store_stats_cat['Average Engagement'].median()
             trend_threshold = 0.1 # Correlation threshold for positive/negative trend

             def assign_category(row):
                 avg_eng = row['Average Engagement']
                 trend_corr = row['Trend Correlation']

                 is_above_median = avg_eng >= median_engagement
                 has_positive_trend = trend_corr > trend_threshold
                 has_negative_trend = trend_corr < -trend_threshold

                 if is_above_median:
                     if has_negative_trend:
                         return "Needs Stabilization" # High but declining
                     else: # Stable or Improving trend
                         return "Star Performer" # High and stable/improving
                 else: # Below Median
                     if has_positive_trend:
                         return "Improving" # Low but improving
                     else: # Stable or Declining trend
                         return "Requires Intervention" # Low and stable/declining

             store_stats_cat['Category'] = store_stats_cat.apply(assign_category, axis=1)

             # Merge category back to main filtered df for potential use elsewhere
             if 'Category' not in df_filtered.columns:
                  df_filtered = df_filtered.merge(store_stats_cat[['Store #', 'Category']], on='Store #', how='left')

        else:
             st.warning("Could not calculate store statistics for categorization.")
             store_stats_cat = pd.DataFrame(columns=['Store #', 'Average Engagement', 'Std Dev', 'Trend Correlation', 'Category']) # Empty df


        # --- Display Category Definitions ---
        st.markdown("#### Category Definitions")
        category_definitions = {
            "Star Performer": {
                "accent": PUBLIX_GREEN, "icon": "⭐",
                "description": "High engagement (at or above median) with stable or positive trend.",
                "action": "Maintain performance, share best practices."
            },
            "Needs Stabilization": {
                "accent": ACCENT_ORANGE, "icon": "⚠️",
                "description": "High engagement (at or above median) but showing a negative trend.",
                "action": "Investigate decline, reinforce successful processes."
            },
            "Improving": {
                "accent": ACCENT_BLUE, "icon": "📈",
                "description": "Below median engagement but showing a positive improvement trend.",
                "action": "Support and encourage continued positive momentum."
            },
            "Requires Intervention": {
                "accent": WARNING_RED, "icon": "🚨",
                "description": "Below median engagement with flat or declining trend.",
                "action": "Needs comprehensive review and support plan."
            }
        }

        col_defs1, col_defs2 = st.columns(2)
        defs_list = list(category_definitions.items())

        for i, (cat, info) in enumerate(defs_list):
             target_col = col_defs1 if i < len(defs_list) / 2 else col_defs2
             with target_col:
                 st.markdown(f"""
                 <div class="styled-card" style="border-left: 5px solid {info['accent']};">
                     <h4 style="color: {info['accent']};">{info['icon']} {cat}</h4>
                     <p>{info['description']}</p>
                     <p><strong>Recommended Focus:</strong> {info['action']}</p>
                 </div>
                 """, unsafe_allow_html=True)


        # --- Display Stores by Category ---
        st.markdown("#### Store Category Results")
        if not store_stats_cat.empty:
            # Order categories for display
            category_order = ["Star Performer", "Needs Stabilization", "Improving", "Requires Intervention"]
            all_stores_categorized = store_stats_cat['Store #'].tolist()

            for cat in category_order:
                cat_stores = store_stats_cat[store_stats_cat['Category'] == cat]
                if not cat_stores.empty:
                     info = category_definitions[cat]
                     st.markdown(f"""
                     <div style="border-left: 5px solid {info['accent']}; padding-left: 15px; margin-bottom: 15px; margin-top: 20px;">
                         <h4 style="color: {info['accent']}; margin-bottom: 5px;">{info['icon']} {cat} ({len(cat_stores)} stores)</h4>
                     </div>
                     """, unsafe_allow_html=True)

                     # Display stores in columns
                     cols_per_row = 4 # Adjust number of columns
                     num_rows_cat = (len(cat_stores) + cols_per_row - 1) // cols_per_row
                     store_indices_cat = cat_stores.index

                     for row_idx_cat in range(num_rows_cat):
                         cols_cat = st.columns(cols_per_row)
                         start_idx_cat = row_idx_cat * cols_per_row
                         end_idx_cat = start_idx_cat + cols_per_row
                         current_row_indices_cat = store_indices_cat[start_idx_cat:end_idx_cat]

                         for i_cat, store_index_cat in enumerate(current_row_indices_cat):
                             store_cat_info = cat_stores.loc[store_index_cat]
                             with cols_cat[i_cat]:
                                 trend_corr = store_cat_info['Trend Correlation']
                                 if trend_corr > 0.3: trend_icon = "🔼"
                                 elif trend_corr > trend_threshold: trend_icon = "↗️"
                                 elif trend_corr < -0.3: trend_icon = "🔽"
                                 elif trend_corr < -trend_threshold: trend_icon = "↘️"
                                 else: trend_icon = "➡️"

                                 st.markdown(f"""
                                 <div class="styled-card" style="text-align: center;">
                                     <h4 style="color: {info['accent']};">Store {store_cat_info['Store #']}</h4>
                                     <p>Avg: {store_cat_info['Average Engagement']:.2f}%</p>
                                     <p>Trend: {trend_icon} <span class="secondary-info">(Corr: {trend_corr:.2f})</span></p>
                                 </div>
                                 """, unsafe_allow_html=True)

            # --- Store-Specific Deep Dive ---
            st.markdown("---") # Divider
            st.subheader("Detailed Store View")
            if all_stores_categorized:
                 selected_store_cat = st.selectbox("Select a store for details:", options=sorted(all_stores_categorized))
                 if selected_store_cat:
                     store_detail = store_stats_cat[store_stats_cat['Store #'] == selected_store_cat].iloc[0]
                     store_cat_detail = store_detail['Category']
                     info_detail = category_definitions[store_cat_detail]
                     accent_detail = info_detail['accent']

                     trend_corr_detail = store_detail['Trend Correlation']
                     if trend_corr_detail > 0.3: trend_desc = "Strong Positive Trend"; trend_icon_detail = "🔼"
                     elif trend_corr_detail > trend_threshold: trend_desc = "Mild Positive Trend"; trend_icon_detail = "↗️"
                     elif trend_corr_detail < -0.3: trend_desc = "Strong Negative Trend"; trend_icon_detail = "🔽"
                     elif trend_corr_detail < -trend_threshold: trend_desc = "Mild Negative Trend"; trend_icon_detail = "↘️"
                     else: trend_desc = "Stable Trend"; trend_icon_detail = "➡️"


                     st.markdown(f"""
                     <div class="styled-card" style="border-left: 5px solid {accent_detail};">
                         <h3 style="color: {accent_detail};">
                             {info_detail['icon']} Store {selected_store_cat} - Category: {store_cat_detail}
                         </h3>
                         <p><strong>Average Engagement:</strong> {store_detail['Average Engagement']:.2f}%</p>
                         <p><strong>Trend:</strong> {trend_icon_detail} {trend_desc} (Correlation: {trend_corr_detail:.2f})</p>
                         <p><strong>Category Definition:</strong> {info_detail['description']}</p>
                         <h4 style="color: {accent_detail}; margin-top: 1em;">Recommended Focus:</h4>
                         <p>{info_detail['action']}</p>
                     </div>
                     """, unsafe_allow_html=True)

                     # Trend Chart for Selected Store
                     st.markdown(f"##### Store {selected_store_cat} Engagement Trend")
                     store_trend_data = df_filtered[df_filtered['Store #'] == selected_store_cat].sort_values('Week')
                     if not store_trend_data.empty and 'Week' in store_trend_data.columns and 'Engaged Transaction %' in store_trend_data.columns:
                          trend_chart_detail = alt.Chart(store_trend_data).mark_line(
                              point=True, strokeWidth=3 # Thicker line for emphasis
                          ).encode(
                              x=alt.X('Week:O', title='Week'),
                              y=alt.Y('Engaged Transaction %:Q', title='Engagement %', scale=alt.Scale(zero=False)),
                              color=alt.value(accent_detail), # Use category color
                              tooltip=['Week:O', alt.Tooltip('Engaged Transaction %:Q', format='.2f')]
                          )
                          # Add regression line
                          regression_line = trend_chart_detail.transform_regression(
                              'Week', 'Engaged Transaction %'
                          ).mark_line(strokeDash=[2,2], color='black', opacity=0.6)

                          # Add district average line if available
                          if not store_choice: # Only if viewing all stores
                               district_avg_trend_detail = df_filtered.groupby('Week')['Engaged Transaction %'].mean().reset_index()
                               avg_line_detail = alt.Chart(district_avg_trend_detail).mark_line(
                                   strokeDash=[5,5], color=MEDIUM_TEXT, strokeWidth=2
                               ).encode(
                                   x='Week:O',
                                   y='Engaged Transaction %:Q',
                                   tooltip=[alt.Tooltip('Engaged Transaction %:Q', format='.2f', title='District Avg')]
                               )
                               final_detail_chart = alt.layer(trend_chart_detail, regression_line, avg_line_detail).interactive()
                          else:
                               final_detail_chart = alt.layer(trend_chart_detail, regression_line).interactive()


                          st.altair_chart(final_detail_chart.properties(height=300), use_container_width=True)
                          st.caption("Store trend shown with regression line. District average shown as dashed gray line if viewing all stores.")

                     else:
                          st.info("Trend data not available for this store within the filtered range.")

            else:
                 st.info("No stores available for detailed view based on current filters.")
        else:
             st.info("Perform categorization first by ensuring sufficient data and appropriate filters.")


# --------------------------------------------------------
# TAB 4: Anomalies & Insights
# --------------------------------------------------------
with tab4:
    st.subheader("Anomaly Detection")
    st.write(f"Highlights significant week-over-week changes based on Z-score (Sensitivity: {z_threshold:.1f}).")

    # Calculate anomalies using the filtered data
    anomalies_df = find_anomalies(df_filtered, z_threshold)

    if anomalies_df.empty:
        st.success("✅ No significant anomalies detected for the selected criteria.")
    else:
        st.warning(f"🚨 {len(anomalies_df)} anomalies detected.")
        with st.expander("View Anomaly Details", expanded=True):
             # Select and format columns for display
             display_cols = ['Store #', 'Week', 'Engaged Transaction %', 'Change %pts', 'Z-score', 'Possible Explanation']
             # Format numerical columns
             anomalies_display = anomalies_df[display_cols].copy()
             for col in ['Engaged Transaction %', 'Change %pts', 'Z-score']:
                 anomalies_display[col] = anomalies_display[col].map('{:.2f}'.format)

             st.dataframe(anomalies_display, hide_index=True, use_container_width=True)
             st.caption("Z-score measures how many standard deviations a weekly change is from the store's average weekly change.")


    # --- Additional Insights ---
    st.markdown("---")
    st.subheader("Performance Summary & Opportunities")

    insight_tabs = st.tabs(["📊 Period Summary", "💡 Opportunities"])

    with insight_tabs[0]: # Period Summary
        st.markdown("#### Overall Performance (Filtered Period)")
        if not df_filtered.empty and 'Store #' in df_filtered.columns and 'Engaged Transaction %' in df_filtered.columns:
             summary_data = df_filtered.groupby('Store #')['Engaged Transaction %'].agg(['mean', 'std', 'min', 'max']).reset_index()
             summary_data.columns = ['Store #', 'Average %', 'Std Dev', 'Min %', 'Max %']
             summary_data['Range %'] = summary_data['Max %'] - summary_data['Min %']
             summary_data = summary_data.sort_values('Average %', ascending=False)

             # Format for display
             summary_display = summary_data.copy()
             for col in ['Average %', 'Std Dev', 'Min %', 'Max %', 'Range %']:
                  summary_display[col] = summary_display[col].map('{:.2f}'.format)

             st.dataframe(summary_display, hide_index=True, use_container_width=True)
             st.caption("Summary statistics calculated across the currently filtered data.")

             # Trend Correlation Chart (Re-using calculation from Tab 3 if available)
             st.markdown("#### Overall Trend Direction (Filtered Period)")
             if 'Trend Correlation' in store_stats_cat.columns:
                 trend_corr_summary = store_stats_cat[['Store #', 'Trend Correlation']].copy()
                 trend_corr_summary = trend_corr_summary.sort_values('Trend Correlation', ascending=False)

                 trend_chart_summary = alt.Chart(trend_corr_summary).mark_bar().encode(
                     x=alt.X('Trend Correlation:Q', title='Week vs. Engagement Correlation', axis=alt.Axis(format='.2f')),
                     y=alt.Y('Store #:N', title='Store', sort='-x'),
                     color=alt.Color('Trend Correlation:Q',
                                     scale=alt.Scale(domain=[-1, 0, 1], range=[WARNING_RED, LIGHT_GRAY_BG, PUBLIX_GREEN]), # Red-Gray-Green
                                     legend=None),
                     tooltip=['Store #', alt.Tooltip('Trend Correlation:Q', format='.3f')]
                 ).properties(
                     height=max(150, len(trend_corr_summary)*20) # Dynamic height
                 )
                 st.altair_chart(trend_chart_summary, use_container_width=True)
                 st.caption("Correlation between Week Number and Engagement %. Positive (Green) suggests upward trend, Negative (Red) suggests downward trend within the filtered period.")
             else:
                  st.info("Trend correlation requires sufficient weekly data points.")

        else:
             st.info("No data available for summary.")


    with insight_tabs[1]: # Opportunities
        st.markdown("#### Improvement Opportunities")
        if not df_filtered.empty and 'Store #' in df_filtered.columns and 'Engaged Transaction %' in df_filtered.columns:
             store_perf_opp = df_filtered.groupby('Store #')['Engaged Transaction %'].mean()

             if len(store_perf_opp) > 1:
                 current_district_avg_opp = store_perf_opp.mean()
                 median_value_opp = store_perf_opp.median()
                 top_val_opp = store_perf_opp.max()

                 # Scenario 1: Improve bottom performer to median
                 bottom_store_opp = store_perf_opp.idxmin()
                 bottom_value_opp = store_perf_opp.min()
                 if bottom_value_opp < median_value_opp: # Only if improvement is possible
                      scenario_perf_1 = store_perf_opp.copy()
                      scenario_perf_1[bottom_store_opp] = median_value_opp
                      scenario_avg_1 = scenario_perf_1.mean()
                      improvement_1 = scenario_avg_1 - current_district_avg_opp

                      st.markdown(f"""
                      **Scenario 1: Improve Bottom Performer**
                      - If Store **{bottom_store_opp}** (currently {bottom_value_opp:.2f}%) improved to the median ({median_value_opp:.2f}%):
                      - The overall average could increase by **{improvement_1:+.2f}** points to **{scenario_avg_1:.2f}%**.
                      """)

                 # Gap to Top Performer Chart
                 st.markdown("---")
                 st.markdown("#### Gap to Top Performer")
                 gap_df = pd.DataFrame({
                     'Store #': store_perf_opp.index,
                     'Current %': store_perf_opp.values,
                     'Gap to Top (%)': top_val_opp - store_perf_opp.values
                 }).round(2)
                 gap_df = gap_df[gap_df['Gap to Top (%)'] > 0.01].sort_values('Gap to Top (%)', ascending=False) # Show only gaps > 0.01%

                 if not gap_df.empty:
                      gap_chart = alt.Chart(gap_df).mark_bar().encode(
                          x=alt.X('Gap to Top (%):Q'),
                          y=alt.Y('Store #:N', title='Store', sort='-x'),
                          color=alt.Color('Gap to Top (%):Q',
                                          scale=alt.Scale(scheme='oranges'), # Use orange/yellow for gaps
                                          legend=None),
                          tooltip=[
                              'Store #:N',
                              alt.Tooltip('Current %:Q', format='.2f'),
                              alt.Tooltip('Gap to Top (%):Q', format='.2f')
                          ]
                      ).properties(
                           height=max(150, len(gap_df)*20)
                      )
                      st.altair_chart(gap_chart, use_container_width=True)
                      st.caption(f"Difference between each store's average engagement and the top performer ({top_val_opp:.2f}%).")
                 else:
                      st.info("All stores are performing at the same level, or only one store is selected.")

             else: # Only one store
                  st.info("Opportunity analysis requires comparing at least two stores.")
        else:
             st.info("No data available for opportunity analysis.")


# --------------------------------------------------------
# Footer
# --------------------------------------------------------
st.sidebar.markdown("---")
now = datetime.datetime.now()
st.sidebar.caption(f"© Publix Supermarkets, Inc. {now.year}")
# st.sidebar.caption(f"Data last updated: {now.strftime('%Y-%m-%d %H:%M')}") # Consider using file modification time if possible

with st.sidebar.expander("Help & Information"):
    st.markdown("""
    **Using This Dashboard**
    1.  **Upload Data**: Use 'Data Input' to upload your weekly engagement file (CSV/Excel). Optionally add a comparison file.
    2.  **Apply Filters**: Refine your view by Quarter, Week, or specific Stores.
    3.  **Adjust Settings**: Fine-tune anomaly detection and trend analysis windows under 'Analysis Settings'.
    4.  **Explore Tabs**:
        * 📈 **Engagement Trends**: View performance over time, compare periods, explore heatmaps, and see recent trend directions.
        * 📊 **Store Comparison**: Directly compare store performance and see rankings.
        * 📋 **Performance Categories**: Understand store groupings based on performance and trend, with recommended focus areas.
        * 🔍 **Anomalies & Insights**: Identify significant weekly changes and explore overall performance summaries and improvement opportunities.

    *Need assistance? Contact District Support.*
    """)
