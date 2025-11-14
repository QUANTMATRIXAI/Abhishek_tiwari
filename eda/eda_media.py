"""
EDA Insights - Production-Ready Data Analysis Platform

A professional-grade Streamlit application for comprehensive data exploration,
visualization, and insight management with integrated comment system.

Key Features:
- Robust date parsing with multiple format support
- Mutually exclusive column classification (Identifiers vs Metrics)
- Enterprise-grade error handling and validation
- Integrated comment and insight management system
- Export and reporting capabilities

Author: Production Release
Version: 1.0.0
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import json
import io
import traceback
from pathlib import Path

# ============================================================================
# CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="EDA Insights",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS STYLING
# ============================================================================

def inject_custom_css():
    """Inject custom CSS for professional styling with brand colors"""
    st.markdown("""
        <style>
        /* Import Google Fonts */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
        
        /* Global styles */
        * {
            font-family: 'Inter', sans-serif;
        }
        
        /* Color scheme variables */
        :root {
            --primary-yellow: #FFBD59;
            --yellow-light: #FFCF87;
            --yellow-lighter: #FFE7C2;
            --yellow-lightest: #FFF2DF;
            --secondary-green: #41C185;
            --secondary-blue: #458EE2;
            --text-dark: #333333;
            --text-medium: #666666;
            --text-light: #999999;
            --background-white: #FFFFFF;
            --background-light: #F5F5F5;
            --shadow-sm: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
            --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
            --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
            --shadow-xl: 0 20px 25px -5px rgba(0, 0, 0, 0.1);
        }
        
        /* Main app background */
        .main {
            background-color: var(--background-light);
        }
        
        /* Card component styling */
        .card {
            background: var(--background-white);
            border-radius: 16px;
            padding: 28px;
            margin: 12px 0;
            box-shadow: var(--shadow-lg);
            border: 1px solid var(--yellow-lighter);
            transition: all 0.3s ease;
            position: relative;
            overflow: hidden;
        }
        
        .card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            width: 4px;
            height: 100%;
            background: linear-gradient(180deg, var(--primary-yellow) 0%, var(--secondary-green) 100%);
        }
        
        .card:hover {
            transform: translateY(-4px);
            box-shadow: var(--shadow-xl);
            border-color: var(--primary-yellow);
        }
        
        .card-title {
            color: var(--text-dark);
            font-size: 1.4em;
            font-weight: 700;
            margin-bottom: 18px;
            letter-spacing: -0.02em;
        }
        
        /* Header styling */
        .app-header {
            text-align: center;
            background: linear-gradient(135deg, var(--primary-yellow) 0%, var(--yellow-light) 100%);
            color: var(--text-dark);
            padding: 32px 20px;
            margin: -20px 0 32px 0;
            border-radius: 16px;
            box-shadow: var(--shadow-lg);
        }
        
        .app-header h1 {
            font-size: 2.5em;
            font-weight: 700;
            margin: 0;
            letter-spacing: -0.02em;
        }
        
        .app-header p {
            font-size: 1.1em;
            margin-top: 8px;
            color: var(--text-dark);
            font-weight: 500;
            opacity: 0.9;
        }
        
        /* Button styling */
        .stButton > button {
            background: linear-gradient(135deg, var(--primary-yellow) 0%, var(--yellow-light) 100%);
            color: var(--text-dark);
            border: none;
            border-radius: 10px;
            padding: 12px 24px;
            font-weight: 600;
            font-size: 1em;
            transition: all 0.3s ease;
            box-shadow: var(--shadow-md);
        }
        
        .stButton > button:hover {
            background: linear-gradient(135deg, var(--yellow-light) 0%, var(--primary-yellow) 100%);
            box-shadow: var(--shadow-lg);
            transform: translateY(-2px);
        }
        
        /* Primary button variant */
        .stButton > button[kind="primary"] {
            background: linear-gradient(135deg, var(--secondary-green) 0%, #35a372 100%);
            color: white;
        }
        
        .stButton > button[kind="primary"]:hover {
            background: linear-gradient(135deg, #35a372 0%, var(--secondary-green) 100%);
        }
        
        /* Secondary button variant */
        .stButton > button[kind="secondary"] {
            background: linear-gradient(135deg, var(--secondary-blue) 0%, #3a7bc8 100%);
            color: white;
        }
        
        .stButton > button[kind="secondary"]:hover {
            background: linear-gradient(135deg, #3a7bc8 0%, var(--secondary-blue) 100%);
        }
        
        /* Info styling */
        .stInfo {
            background-color: var(--yellow-lightest);
            border-left-color: var(--secondary-blue);
        }
        
        /* Success styling */
        .stSuccess {
            background-color: #e8f5f0;
            border-left-color: var(--secondary-green);
        }
        
        /* Warning styling */
        .stWarning {
            background-color: #fff4e6;
            border-left-color: var(--primary-yellow);
        }
        
        /* Metric styling */
        .stMetric {
            background: var(--background-white);
            padding: 16px;
            border-radius: 12px;
            box-shadow: var(--shadow-sm);
            border: 1px solid var(--yellow-lightest);
        }
        
        /* Divider styling */
        hr {
            border-color: var(--yellow-lighter);
            margin: 24px 0;
        }
        
        /* Dataframe styling */
        .dataframe {
            border: 1px solid var(--yellow-lighter) !important;
            border-radius: 8px;
        }
        
        /* Tab styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
            background-color: var(--background-white);
            padding: 8px;
            border-radius: 12px;
            box-shadow: var(--shadow-sm);
        }
        
        .stTabs [data-baseweb="tab"] {
            background-color: transparent;
            border-radius: 8px;
            color: var(--text-medium);
            font-weight: 600;
            padding: 12px 24px;
        }
        
        .stTabs [aria-selected="true"] {
            background: linear-gradient(135deg, var(--primary-yellow) 0%, var(--yellow-light) 100%);
            color: var(--text-dark);
        }
        
        /* Sidebar styling */
        section[data-testid="stSidebar"] {
            background-color: var(--background-white);
            border-right: 2px solid var(--yellow-lighter);
        }
        
        /* Expander styling */
        .streamlit-expanderHeader {
            background-color: var(--yellow-lightest);
            border-radius: 8px;
            font-weight: 600;
        }
        
        /* Select box styling */
        .stSelectbox > div > div {
            border-color: var(--yellow-lighter);
            border-radius: 8px;
        }
        
        /* Multiselect styling */
        .stMultiSelect > div > div {
            border-color: var(--yellow-lighter);
            border-radius: 8px;
        }
        
        /* Text input styling */
        .stTextInput > div > div {
            border-color: var(--yellow-lighter);
            border-radius: 8px;
        }
        
        /* Date input styling */
        .stDateInput > div > div {
            border-color: var(--yellow-lighter);
            border-radius: 8px;
        }
        </style>
    """, unsafe_allow_html=True)

def create_card(title, content_func):
    """
    Create a styled card container
    Args:
        title: Card header text
        content_func: Function that renders card content
    """
    st.markdown(f'<div class="card">', unsafe_allow_html=True)
    if title:
        st.markdown(f'<div class="card-title">{title}</div>', unsafe_allow_html=True)
    content_func()
    st.markdown('</div>', unsafe_allow_html=True)

# Inject custom CSS
inject_custom_css()

COMMENTS_FILE = "eda_comments.json"

# ============================================================================
# COMMENT MANAGEMENT SYSTEM
# ============================================================================

class CommentManager:
    """Enterprise-grade comment management with file-based persistence"""
    
    @staticmethod
    def load_comments():
        """Load all comments with error handling"""
        try:
            if not Path(COMMENTS_FILE).exists():
                return pd.DataFrame(), None
            
            with open(COMMENTS_FILE, 'r', encoding='utf-8') as f:
                comments = json.load(f)
            
            if not comments:
                return pd.DataFrame(), None
                
            df = pd.DataFrame(comments)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            return df, None
        except Exception as e:
            return pd.DataFrame(), f"Error loading comments: {str(e)}"
    
    @staticmethod
    def save_comment(comment_text, tab_name, context_data=None):
        """Save new comment with validation"""
        try:
            if not comment_text or not comment_text.strip():
                return False, "Comment text cannot be empty"
            
            # Load existing comments
            df, error = CommentManager.load_comments()
            if error:
                df = pd.DataFrame()
            
            # Create new comment
            new_comment = {
                'id': datetime.now().strftime('%Y%m%d%H%M%S%f'),
                'timestamp': datetime.now().isoformat(),
                'comment_text': comment_text.strip(),
                'tab_name': tab_name,
                'context_data': json.dumps(context_data) if context_data else None
            }
            
            # Append
            if df.empty:
                df = pd.DataFrame([new_comment])
            else:
                df = pd.concat([df, pd.DataFrame([new_comment])], ignore_index=True)
            
            # Convert timestamps for JSON serialization
            comments_list = df.to_dict('records')
            for c in comments_list:
                if isinstance(c['timestamp'], pd.Timestamp):
                    c['timestamp'] = c['timestamp'].isoformat()
            
            # Save to file
            with open(COMMENTS_FILE, 'w', encoding='utf-8') as f:
                json.dump(comments_list, f, indent=2, ensure_ascii=False)
            
            return True, "Comment saved successfully"
        except Exception as e:
            return False, f"Failed to save comment: {str(e)}"
    
    @staticmethod
    def delete_comment(comment_id):
        """Delete comment by ID"""
        try:
            df, error = CommentManager.load_comments()
            if error:
                return False, error
            
            if df.empty:
                return False, "No comments found"
            
            df = df[df['id'] != comment_id]
            
            # Save updated list
            comments_list = df.to_dict('records')
            for c in comments_list:
                if isinstance(c['timestamp'], pd.Timestamp):
                    c['timestamp'] = c['timestamp'].isoformat()
            
            with open(COMMENTS_FILE, 'w', encoding='utf-8') as f:
                json.dump(comments_list, f, indent=2, ensure_ascii=False)
            
            return True, "Comment deleted successfully"
        except Exception as e:
            return False, f"Failed to delete comment: {str(e)}"
    
    @staticmethod
    def update_comment(comment_id, new_text):
        """Update comment text"""
        try:
            if not new_text or not new_text.strip():
                return False, "Comment text cannot be empty"
            
            df, error = CommentManager.load_comments()
            if error:
                return False, error
            
            if df.empty or comment_id not in df['id'].values:
                return False, "Comment not found"
            
            df.loc[df['id'] == comment_id, 'comment_text'] = new_text.strip()
            
            # Save updated list
            comments_list = df.to_dict('records')
            for c in comments_list:
                if isinstance(c['timestamp'], pd.Timestamp):
                    c['timestamp'] = c['timestamp'].isoformat()
            
            with open(COMMENTS_FILE, 'w', encoding='utf-8') as f:
                json.dump(comments_list, f, indent=2, ensure_ascii=False)
            
            return True, "Comment updated successfully"
        except Exception as e:
            return False, f"Failed to update comment: {str(e)}"
    
    @staticmethod
    def export_comments():
        """Export all comments as CSV"""
        try:
            df, error = CommentManager.load_comments()
            if error:
                return None, error
            
            if df.empty:
                return None, "No comments to export"
            
            return df.to_csv(index=False), None
        except Exception as e:
            return None, f"Export failed: {str(e)}"

# ============================================================================
# DATA LOADING AND PROCESSING
# ============================================================================

class DataLoader:
    """Robust data loading with comprehensive error handling"""
    
    @staticmethod
    def parse_dates(df, date_column):
        """Attempt multiple date parsing strategies"""
        parse_strategies = [
            # Strategy 1: Auto-detect with infer_datetime_format
            lambda col: pd.to_datetime(col, infer_datetime_format=True, errors='coerce'),
            # Strategy 2: Common formats (DD-MM-YYYY priority for your data)
            lambda col: pd.to_datetime(col, format='%d-%m-%Y', errors='coerce'),
            lambda col: pd.to_datetime(col, format='%d/%m/%Y', errors='coerce'),
            lambda col: pd.to_datetime(col, format='%m/%d/%Y', errors='coerce'),
            lambda col: pd.to_datetime(col, format='%Y-%m-%d', errors='coerce'),
            lambda col: pd.to_datetime(col, format='%Y/%m/%d', errors='coerce'),
            # Strategy 3: Day first (European format)
            lambda col: pd.to_datetime(col, dayfirst=True, errors='coerce'),
            # Strategy 4: Mixed format
            lambda col: pd.to_datetime(col, format='mixed', errors='coerce'),
        ]
        
        original_values = df[date_column].copy()
        
        for strategy in parse_strategies:
            try:
                parsed = strategy(df[date_column])
                
                # Check if parsing was successful
                if parsed.notna().sum() / len(parsed) > 0.5:
                    # Additional validation: Check if dates are reasonable (not all 1970)
                    valid_dates = parsed[parsed.notna()]
                    if len(valid_dates) > 0:
                        # Check if most dates are NOT 1970-01-01 (Unix epoch default/garbage)
                        year_1970_count = (valid_dates.dt.year == 1970).sum()
                        if year_1970_count / len(valid_dates) < 0.9:  # Less than 90% are 1970
                            df[date_column] = parsed
                            return True, None
            except:
                continue
        
        # If all strategies fail, return error
        return False, f"Could not parse dates in column '{date_column}'"
    
    @staticmethod
    @st.cache_data(show_spinner=False)
    def load_file(_file):
        """Load and validate data file with comprehensive error handling"""
        try:
            _file.seek(0)
            
            # Determine file type and load
            if _file.name.endswith('.csv'):
                try:
                    df = pd.read_csv(_file, encoding='utf-8')
                except UnicodeDecodeError:
                    _file.seek(0)
                    df = pd.read_csv(_file, encoding='latin-1')
            elif _file.name.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(_file)
            else:
                return None, "Unsupported file format. Please upload CSV or Excel files."
            
            # Validate dataframe
            if df.empty:
                return None, "File is empty. Please upload a file with data."
            
            if len(df.columns) == 0:
                return None, "No columns found in file."
            
            # Try to identify and parse date column
            # Priority: Look for "Date" first, then "Day", then other date-related columns
            date_candidates_all = []
            
            # First priority: columns named exactly "Date"
            exact_date_cols = [col for col in df.columns if col.lower() == 'date']
            
            # Second priority: columns with "date" in the name
            date_in_name = [col for col in df.columns 
                          if 'date' in col.lower() and col.lower() != 'date']
            
            # Third priority: "Day" column
            day_cols = [col for col in df.columns if col.lower() == 'day']
            
            # Fourth priority: other date-related keywords
            other_date_cols = [col for col in df.columns 
                             if any(keyword in col.lower() for keyword in ['time', 'period', 'timestamp']) 
                             and col not in exact_date_cols + date_in_name + day_cols]
            
            # Combine in priority order
            date_candidates_all = exact_date_cols + date_in_name + day_cols + other_date_cols
            
            date_column_found = None
            invalid_date_columns = []
            
            if date_candidates_all:
                # Try each candidate until we find one with valid data
                for date_col in date_candidates_all:
                    success, error = DataLoader.parse_dates(df, date_col)
                    
                    if success:
                        date_column_found = date_col
                        # Rename to standard 'Day' column
                        if date_col != 'Day':
                            df = df.rename(columns={date_col: 'Day'})
                        
                        # Remove rows with invalid dates
                        df = df[df['Day'].notna()].copy()
                        
                        # Sort by date
                        df = df.sort_values('Day').reset_index(drop=True)
                        
                        # Don't show success message here - it will be shown after with date range
                        break
                    else:
                        # This date column didn't work, mark it as invalid
                        invalid_date_columns.append(date_col)
                
                if not date_column_found:
                    st.warning("⚠️ Could not find a valid date column. Some features will be limited.")
            
            # Store invalid date columns for later exclusion
            df.attrs['invalid_date_columns'] = invalid_date_columns
            if date_column_found:
                df.attrs['date_column_used'] = date_column_found
            
            return df, None
            
        except Exception as e:
            error_msg = f"Error loading file: {str(e)}\n\n{traceback.format_exc()}"
            return None, error_msg

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def format_value(value, col_name):
    """Smart value formatting based on column characteristics"""
    if pd.isna(value):
        return "N/A"
    
    try:
        # Convert to float for consistent handling
        value = float(value)
        
        # Currency indicators
        if any(x in col_name.lower() for x in ['amount', 'spent', 'price', 'revenue', 'sales', 'cost', 'usd', '$', 'dollar', 'euro', '€']):
            return f"${value:,.2f}"
        
        # Percentage indicators
        elif any(x in col_name.lower() for x in ['rate', 'percent', '%', 'ratio']):
            if value <= 1:
                return f"{value:.2%}"
            else:
                return f"{value:.2f}%"
        
        # Count/Integer indicators
        elif any(x in col_name.lower() for x in ['count', 'quantity', 'qty', 'number', 'total', 'impressions', 'clicks', 'views']):
            return f"{int(value):,}"
        
        # Default formatting
        elif abs(value) >= 1000 or value % 1 == 0:
            return f"{value:,.0f}"
        else:
            return f"{value:,.2f}"
    except:
        return str(value)

def aggregate_data(df, level, method='sum'):
    """Aggregate data by time period"""
    if 'Day' not in df.columns:
        return df
    
    df_agg = df.copy()
    
    # Create period column
    if level == "Weekly":
        df_agg['Period'] = df_agg['Day'].dt.to_period('W').apply(lambda r: r.start_time)
    elif level == "Monthly":
        df_agg['Period'] = df_agg['Day'].dt.to_period('M').apply(lambda r: r.start_time)
    elif level == "Quarterly":
        df_agg['Period'] = df_agg['Day'].dt.to_period('Q').apply(lambda r: r.start_time)
    else:  # Daily
        df_agg['Period'] = df_agg['Day']
    
    # Get numeric columns
    numeric_cols = df_agg.select_dtypes(include=[np.number]).columns
    
    # Aggregate
    agg_dict = {col: method for col in numeric_cols}
    
    return df_agg.groupby('Period').agg(agg_dict).reset_index()

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

def initialize_session_state():
    """Initialize all session state variables with defaults"""
    defaults = {
        'file_loaded': False,
        'columns_classified': False,
        'selected_identifiers': [],
        'selected_metrics': [],
        'start_date': None,
        'end_date': None,
        'editing_comment_id': None,
        'delete_confirm_id': None,
        'dataset_states': {},
        'active_dataset_key': None,
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

initialize_session_state()

ANALYSIS_TABS = ["overview", "explore", "pivot", "correlation", "clustering"]
DEFAULT_SHEET_NAME = "Sheet 1"

def ensure_dataset_state(dataset_key: str, file_name: str):
    """Ensure a dataset entry exists in session state."""
    datasets = st.session_state.dataset_states
    if dataset_key not in datasets:
        datasets[dataset_key] = {
            'name': file_name,
            'columns_classified': False,
            'selected_identifiers': [],
            'selected_metrics': [],
            'global_start_date': None,
            'global_end_date': None,
            'worksheets': {
                tab: {
                    'order': [DEFAULT_SHEET_NAME],
                    'active': DEFAULT_SHEET_NAME
                } for tab in ANALYSIS_TABS
            },
        }
    else:
        datasets[dataset_key]['name'] = file_name
        if 'worksheets' not in datasets[dataset_key]:
            datasets[dataset_key]['worksheets'] = {
                tab: {
                    'order': [DEFAULT_SHEET_NAME],
                    'active': DEFAULT_SHEET_NAME
                } for tab in ANALYSIS_TABS
            }
    return datasets[dataset_key]

def sync_session_from_dataset(dataset_key: str | None):
    """Load dataset-specific selections into global session values."""
    if not dataset_key or dataset_key not in st.session_state.dataset_states:
        st.session_state.columns_classified = False
        st.session_state.selected_identifiers = []
        st.session_state.selected_metrics = []
        st.session_state.start_date = None
        st.session_state.end_date = None
        return
    ds_state = st.session_state.dataset_states[dataset_key]
    st.session_state.columns_classified = ds_state.get('columns_classified', False)
    st.session_state.selected_identifiers = ds_state.get('selected_identifiers', []).copy()
    st.session_state.selected_metrics = ds_state.get('selected_metrics', []).copy()
    st.session_state.start_date = ds_state.get('global_start_date')
    st.session_state.end_date = ds_state.get('global_end_date')

def persist_active_dataset_state():
    """Save current selections back to the active dataset entry."""
    dataset_key = st.session_state.get('active_dataset_key')
    if not dataset_key:
        return
    ds_state = ensure_dataset_state(dataset_key, st.session_state.dataset_states.get(dataset_key, {}).get('name', ''))
    ds_state['columns_classified'] = st.session_state.get('columns_classified', False)
    ds_state['selected_identifiers'] = st.session_state.get('selected_identifiers', []).copy()
    ds_state['selected_metrics'] = st.session_state.get('selected_metrics', []).copy()
    ds_state['global_start_date'] = st.session_state.get('start_date')
    ds_state['global_end_date'] = st.session_state.get('end_date')

def get_workspace_state(dataset_key: str | None, tab_key: str):
    """Return the worksheet registry for a dataset/tab."""
    if not dataset_key:
        return {'order': [DEFAULT_SHEET_NAME], 'active': DEFAULT_SHEET_NAME}
    ds_state = ensure_dataset_state(dataset_key, st.session_state.dataset_states.get(dataset_key, {}).get('name', ''))
    worksheets = ds_state.setdefault('worksheets', {
        tab: {
            'order': [DEFAULT_SHEET_NAME],
            'active': DEFAULT_SHEET_NAME
        } for tab in ANALYSIS_TABS
    })
    if tab_key not in worksheets:
        worksheets[tab_key] = {'order': [DEFAULT_SHEET_NAME], 'active': DEFAULT_SHEET_NAME}
    sheet_state = worksheets[tab_key]
    if not sheet_state.get('order'):
        sheet_state['order'] = [DEFAULT_SHEET_NAME]
        sheet_state['active'] = DEFAULT_SHEET_NAME
    if sheet_state['active'] not in sheet_state['order']:
        sheet_state['order'].append(sheet_state['active'])
    return sheet_state

def render_workspace_selector(tab_key: str, dataset_key: str | None):
    """Render a horizontal selector for worksheets and return the active sheet name."""
    sheet_state = get_workspace_state(dataset_key, tab_key)
    sheet_names = sheet_state['order']
    current_selection = sheet_state.get('active', sheet_names[0])
    radio_key = f"{dataset_key or 'default'}_{tab_key}_workspace_radio"
    col1, col2 = st.columns([5, 1])
    add_key = f"{dataset_key}_{tab_key}_add_workspace"
    with col2:
        if st.button("➕ Add", key=add_key):
            counter = 1
            base = "Sheet"
            new_name = f"{base} {len(sheet_names) + counter}"
            while new_name in sheet_names:
                counter += 1
                new_name = f"{base} {len(sheet_names) + counter}"
            sheet_names.append(new_name)
            sheet_state['order'] = sheet_names
            sheet_state['active'] = new_name
            if dataset_key and dataset_key in st.session_state.dataset_states:
                st.session_state.dataset_states[dataset_key]['worksheets'][tab_key] = sheet_state
            st.rerun()
    with col1:
        selected = st.radio(
            "Workspaces",
            sheet_names,
            index=sheet_names.index(current_selection) if current_selection in sheet_names else 0,
            key=radio_key,
            horizontal=True
        )
    sheet_state['active'] = selected
    if dataset_key and dataset_key in st.session_state.dataset_states:
        st.session_state.dataset_states[dataset_key]['worksheets'][tab_key] = sheet_state
    return selected

def render_dataset_selector_control(control_key: str, label: str = "Select dataset"):
    """Render a selectbox to choose the active dataset."""
    dataset_keys = list(st.session_state.dataset_states.keys())
    if not dataset_keys:
        st.info("Upload at least one dataset to get started.")
        return None
    if st.session_state.active_dataset_key not in dataset_keys:
        st.session_state.active_dataset_key = dataset_keys[0]
        sync_session_from_dataset(st.session_state.active_dataset_key)
    default_index = dataset_keys.index(st.session_state.active_dataset_key)
    selection = st.selectbox(
        label,
        dataset_keys,
        index=default_index,
        key=control_key,
        format_func=lambda key: st.session_state.dataset_states[key]['name']
    )
    if selection != st.session_state.active_dataset_key:
        st.session_state.active_dataset_key = selection
        sync_session_from_dataset(selection)
        st.rerun()
    return selection

# ============================================================================
# MAIN APPLICATION
# ============================================================================

# App header with styling
st.markdown("""
<div class="app-header">
    <h1>📊 EDA Insights</h1>
    <p>Production-Ready Data Analysis Platform</p>
</div>
""", unsafe_allow_html=True)

# ============================================================================
# SIDEBAR: FILE UPLOAD
# ============================================================================

with st.sidebar:
    st.markdown("""
    <div style="background: linear-gradient(135deg, var(--primary-yellow) 0%, var(--yellow-light) 100%); 
         padding: 20px; border-radius: 12px; margin-bottom: 24px; text-align: center;">
        <h2 style="color: var(--text-dark); margin: 0; font-size: 1.5em;">dY"? Data Upload</h2>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_files = st.file_uploader(
        "Upload CSV or Excel files",
        type=['csv', 'xlsx', 'xls'],
        accept_multiple_files=True,
        help="Supported formats: CSV, XLSX, XLS. Use Ctrl/Cmd or the + button to add multiple files."
    )
    uploaded_file = None
    dataset_file_map = {}
    
    if uploaded_files:
        dataset_keys = []
        for idx, file in enumerate(uploaded_files):
            dataset_key = f"{file.name}_{file.size}_{idx}"
            dataset_keys.append(dataset_key)
            dataset_file_map[dataset_key] = file
            ds_state = ensure_dataset_state(dataset_key, file.name)
            size_kb = file.size / 1024 if file.size else 0
            st.success(f"✅ **{file.name}** ({size_kb:.1f} KB)")
            if ds_state.get('columns_classified'):
                st.caption("• Classification saved")
        for existing_key in list(st.session_state.dataset_states.keys()):
            if existing_key not in dataset_keys:
                st.session_state.dataset_states.pop(existing_key, None)
                if st.session_state.active_dataset_key == existing_key:
                    st.session_state.active_dataset_key = None
        if dataset_keys:
            if not st.session_state.active_dataset_key:
                st.session_state.active_dataset_key = dataset_keys[0]
                sync_session_from_dataset(st.session_state.active_dataset_key)
            elif st.session_state.active_dataset_key not in dataset_keys:
                st.session_state.active_dataset_key = dataset_keys[0]
                sync_session_from_dataset(st.session_state.active_dataset_key)
        
        if dataset_keys:
            if not st.session_state.active_dataset_key or st.session_state.active_dataset_key not in dataset_keys:
                st.session_state.active_dataset_key = dataset_keys[0]
                sync_session_from_dataset(st.session_state.active_dataset_key)
            uploaded_file = dataset_file_map.get(st.session_state.active_dataset_key)
            if uploaded_file is not None:
                uploaded_file.seek(0)

    st.divider()
    
    st.markdown("### File Management")
    if st.session_state.dataset_states:
        for key, ds in st.session_state.dataset_states.items():
            status_icon = "✅" if ds.get('columns_classified') else "🟡"
            status_label = "Classified" if ds.get('columns_classified') else "Needs classification"
            active_label = " (active)" if key == st.session_state.get('active_dataset_key') else ""
            st.markdown(
                f"{status_icon} **{ds.get('name', 'Unnamed')}**{active_label}<br>"
                f"<span style='color: var(--text-medium); font-size: 0.9em;'>{status_label}</span>",
                unsafe_allow_html=True
            )
    else:
        st.caption("No files uploaded yet.")
    
    # Sample data
    st.markdown("""
    <div style="background: var(--yellow-lightest); padding: 12px; border-radius: 8px; margin-bottom: 12px;">
        <p style="color: var(--text-dark); font-weight: 600; margin: 0; font-size: 0.95em;">
            dY'� Need sample data?
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    sample_df = pd.DataFrame({
        'Date': pd.date_range('2024-01-01', periods=30),
        'Region': np.random.choice(['North', 'South', 'East', 'West'], 30),
        'Product': np.random.choice(['A', 'B', 'C'], 30),
        'Sales': np.random.randint(1000, 5000, 30),
        'Quantity': np.random.randint(10, 100, 30),
        'Revenue': np.random.uniform(5000, 20000, 30).round(2)
    })
    
    st.download_button(
        'dY"� Download Sample',
        sample_df.to_csv(index=False),
        "sample_data.csv",
        "text/csv",
        use_container_width=True
    )
    
    # Reset app state button
    if uploaded_file:
        st.divider()
        st.markdown("""
        <div style="background: #fff4e6; padding: 12px; border-radius: 8px; margin-bottom: 12px;">
            <p style="color: var(--text-dark); font-weight: 600; margin: 0; font-size: 0.95em;">
                dY", Need to start over?
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("dY-`�,? Clear All State", use_container_width=True, type="secondary", help="Reset all selections and state"):
            # Clear all session state
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.success("✅. App state cleared!")
            st.rerun()

# ============================================================================
# CHECK FILE UPLOAD
# ============================================================================

if not uploaded_file:
    st.markdown("""
    <div class="card" style="max-width: 800px; margin: 60px auto;">
        <div class="card-title" style="text-align: center; font-size: 2em; margin-bottom: 24px;">
            👆 Getting Started
        </div>
        <p style="text-align: center; color: var(--text-medium); font-size: 1.1em; margin-bottom: 32px;">
            Upload a CSV or Excel file using the sidebar to begin your analysis
        </p>
        
        <div style="background: var(--yellow-lightest); padding: 24px; border-radius: 12px; margin-bottom: 24px;">
            <h3 style="color: var(--text-dark); margin-top: 0;">🚀 Platform Features:</h3>
            <ul style="color: var(--text-medium); line-height: 1.8;">
                <li><strong>🏷️ Smart Column Classification</strong> - Separate identifiers from metrics</li>
                <li><strong>📊 Interactive Visualizations</strong> - Line, bar, area charts with aggregation</li>
                <li><strong>🔄 Dynamic Pivot Tables</strong> - Multi-dimensional analysis</li>
                <li><strong>📈 Correlation Analysis</strong> - Discover relationships in your data</li>
                <li><strong>🎯 Clustering Analysis</strong> - Find patterns with K-Means & DBSCAN</li>
                <li><strong>💬 Comment System</strong> - Save insights and observations</li>
                <li><strong>📋 Report Generation</strong> - Export all your findings</li>
            </ul>
        </div>
        
        <div style="background: var(--background-white); padding: 20px; border-radius: 12px; border: 2px solid var(--secondary-green);">
            <h3 style="color: var(--secondary-green); margin-top: 0;">✅ Requirements:</h3>
            <ul style="color: var(--text-medium); line-height: 1.8;">
                <li>File format: <strong>CSV</strong> or <strong>Excel</strong> (.xlsx, .xls)</li>
                <li>At least <strong>one numeric column</strong> for metrics</li>
                <li><em>Optional:</em> A date column for time-series analysis</li>
            </ul>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

render_dataset_selector_control("dataset_selector_main", "Dataset to configure")

# ============================================================================
# LOAD DATA
# ============================================================================

with st.spinner("Loading and validating data..."):
    df, error = DataLoader.load_file(uploaded_file)

if error:
    st.error(f"❌ **Failed to load file**\n\n{error}")
    st.stop()

if df is None:
    st.error("❌ Unknown error occurred while loading file")
    st.stop()

# ============================================================================
# DATA VALIDATION AND INFO
# ============================================================================

has_dates = 'Day' in df.columns
min_date = None
max_date = None

if has_dates:
    try:
        min_date = df['Day'].min()
        max_date = df['Day'].max()
        
        # Validate dates are reasonable
        if pd.isna(min_date) or pd.isna(max_date):
            st.error("❌ Date column contains only null values")
            has_dates = False
        elif min_date.year < 1900 or max_date.year > 2100:
            st.warning(f"⚠️ Date range seems unusual: {min_date.date()} to {max_date.date()}")
        
        # Convert to date objects for consistent use
        min_date_obj = min_date.date()
        max_date_obj = max_date.date()
        
        # Reset session state dates if they're outside the valid range or not set
        if (st.session_state.start_date is None or 
            st.session_state.start_date < min_date_obj or 
            st.session_state.start_date > max_date_obj):
            st.session_state.start_date = min_date_obj
        
        if (st.session_state.end_date is None or 
            st.session_state.end_date < min_date_obj or 
            st.session_state.end_date > max_date_obj):
            st.session_state.end_date = max_date_obj
        
        # Store for use in tabs
        min_date = min_date_obj
        max_date = max_date_obj
        persist_active_dataset_state()
            
    except Exception as e:
        st.error(f"❌ Error processing dates: {str(e)}")
        has_dates = False

# Get column types
# Exclude 'Day' column and any invalid date columns
excluded_cols = ['Day']

# Add invalid date columns to exclusion list
if hasattr(df, 'attrs') and 'invalid_date_columns' in df.attrs:
    excluded_cols.extend(df.attrs['invalid_date_columns'])

all_columns = [col for col in df.columns if col not in excluded_cols]
numeric_columns = [col for col in df.select_dtypes(include=[np.number]).columns.tolist() if col not in excluded_cols]
categorical_columns = [col for col in df.select_dtypes(include=['object', 'category']).columns.tolist() if col not in excluded_cols]

# Show info about excluded columns
if len(excluded_cols) > 1:  # More than just 'Day'
    excluded_info = [col for col in excluded_cols if col != 'Day']
    if excluded_info:
        st.caption(f"ℹ️ Excluded {len(excluded_info)} column(s) with invalid date data: {', '.join(excluded_info)}")

# ============================================================================
# FILE INFORMATION PANEL
# ============================================================================

st.markdown("---")

info_col1, info_col2, info_col3, info_col4, info_col5 = st.columns(5)

with info_col1:
    st.metric("📄 Rows", f"{len(df):,}")

with info_col2:
    st.metric("📋 Columns", len(df.columns))

with info_col3:
    st.metric("🔢 Numeric", len(numeric_columns))

with info_col4:
    st.metric("🏷️ Categorical", len(categorical_columns))

with info_col5:
    if has_dates:
        date_range_days = (max_date - min_date).days
        st.metric("📅 Date Span", f"{date_range_days} days")
    else:
        st.metric("📅 Dates", "Not found")

if has_dates:
    date_column_used = df.attrs.get('date_column_used', 'Day')
    invalid_cols = df.attrs.get('invalid_date_columns', [])
    
    if invalid_cols:
        st.info(f"📅 **Using date column:** {date_column_used} | **Excluded invalid columns:** {', '.join(invalid_cols)}")
    else:
        st.info(f"📅 **Date Column:** {date_column_used} | **Range:** {min_date} to {max_date}")

# ============================================================================
# COLUMN CLASSIFICATION SECTION
# ============================================================================

st.markdown("---")

st.markdown("""
<div style="background: linear-gradient(135deg, var(--yellow-lightest) 0%, var(--background-white) 100%); 
     padding: 24px; border-radius: 12px; border-left: 4px solid var(--primary-yellow); margin-bottom: 24px;">
    <h2 style="color: var(--text-dark); margin: 0 0 8px 0; font-size: 1.8em;">🎯 Step 1: Column Classification</h2>
    <p style="color: var(--text-medium); margin: 0; font-size: 1.05em;">
        Classify columns as <strong>Identifiers</strong> (categorical dimensions) or <strong>Metrics</strong> (numeric values)
    </p>
</div>
""", unsafe_allow_html=True)

# Get current selections and filter to only include available columns
current_identifiers = [col for col in st.session_state.selected_identifiers if col in all_columns]
current_metrics = [col for col in st.session_state.selected_metrics if col in all_columns]
dataset_suffix = st.session_state.active_dataset_key or "default"

col1, col2 = st.columns(2)

with col1:
    st.subheader("🏷️ Identifiers")
    st.caption("Categorical columns for grouping and filtering")
    
    # Filter out columns already selected as metrics
    available_for_identifiers = [col for col in all_columns if col not in current_metrics]
    
    # Auto-suggest identifiers (categorical columns not in metrics)
    suggested_identifiers = [col for col in categorical_columns if col not in current_metrics]
    
    # Ensure defaults are in available options
    valid_defaults = [col for col in current_identifiers if col in available_for_identifiers]
    if not valid_defaults and not current_identifiers:
        valid_defaults = suggested_identifiers
    
    selected_identifiers = st.multiselect(
        "Select identifier columns",
        options=available_for_identifiers,
        default=valid_defaults,
        key=f"identifier_select_{dataset_suffix}",
        help="Columns used for filtering, grouping, and segmentation"
    )

with col2:
    st.subheader("📊 Metrics")
    st.caption("Numeric columns for calculations and analysis")
    
    # Filter out columns already selected as identifiers
    available_for_metrics = [col for col in all_columns if col not in selected_identifiers]
    
    # Auto-suggest metrics (numeric columns not in identifiers)
    suggested_metrics = [col for col in numeric_columns if col not in selected_identifiers]
    
    # Ensure defaults are in available options
    valid_defaults = [col for col in current_metrics if col in available_for_metrics]
    if not valid_defaults and not current_metrics:
        valid_defaults = suggested_metrics[:5]
    
    selected_metrics = st.multiselect(
        "Select metric columns",
        options=available_for_metrics,
        default=valid_defaults,
        key=f"metric_select_{dataset_suffix}",
        help="Columns used for calculations, aggregations, and visualizations"
    )

# Action buttons
action_col1, action_col2, action_col3 = st.columns([1, 1, 3])

with action_col1:
    if st.button("✅ Apply Classification", type="primary", use_container_width=True):
        if not selected_metrics:
            st.error("❌ You must select at least one metric column")
        else:
            st.session_state.selected_identifiers = selected_identifiers
            st.session_state.selected_metrics = selected_metrics
            st.session_state.columns_classified = True
            persist_active_dataset_state()
            st.success("✅ Classification applied!")
            st.rerun()

with action_col2:
    if st.button("🔄 Auto-Classify", use_container_width=True):
        st.session_state.selected_identifiers = categorical_columns
        st.session_state.selected_metrics = numeric_columns[:min(10, len(numeric_columns))]
        st.session_state.columns_classified = True
        persist_active_dataset_state()
        st.rerun()

# Show summary
if st.session_state.columns_classified:
    summary_col1, summary_col2 = st.columns(2)
    with summary_col1:
        st.success(f"✅ **{len(st.session_state.selected_identifiers)} Identifiers:** {', '.join(st.session_state.selected_identifiers[:3])}{'...' if len(st.session_state.selected_identifiers) > 3 else ''}")
    with summary_col2:
        st.success(f"✅ **{len(st.session_state.selected_metrics)} Metrics:** {', '.join(st.session_state.selected_metrics[:3])}{'...' if len(st.session_state.selected_metrics) > 3 else ''}")

st.markdown("---")

# ============================================================================
# VALIDATION: CHECK CLASSIFICATION
# ============================================================================

if not st.session_state.columns_classified or not st.session_state.selected_metrics:
    st.markdown("""
    <div class="card" style="max-width: 700px; margin: 40px auto; background: linear-gradient(135deg, #fff4e6 0%, var(--background-white) 100%); border-color: var(--primary-yellow);">
        <div style="text-align: center;">
            <h2 style="color: var(--primary-yellow); font-size: 2em; margin-bottom: 16px;">⚠️ Classification Required</h2>
            <p style="color: var(--text-medium); font-size: 1.1em; margin-bottom: 24px;">
                Please classify your columns above and click <strong>'Apply Classification'</strong> to continue
            </p>
            <div style="background: var(--background-white); padding: 16px; border-radius: 8px; border: 1px solid var(--yellow-lighter);">
                <p style="color: var(--text-medium); margin: 0;">
                    📝 Select <strong>Identifiers</strong> (categorical) and <strong>Metrics</strong> (numeric)<br>
                    then click the <strong>Apply Classification</strong> button
                </p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# Get finalized selections
identifiers = st.session_state.selected_identifiers
metrics = st.session_state.selected_metrics

def render_filter_controls(tab_key: str, source_df: pd.DataFrame, dataset_key: str | None, sheet_name: str):
    """
    Show date/identifier filters inside an expander and return the filtered dataframe.
    Returns tuple: (filtered_df, active_filters_dict, start_date, end_date)
    """
    filtered_df = source_df.copy()
    active_filters = {}
    selected_start = None
    selected_end = None
    date_filtered_df = filtered_df

    with st.expander("🎯 Data Filters", expanded=False):
        st.caption("Adjust the dataset powering this tab. Filters apply only within the current tab.")

        if has_dates and min_date and max_date:
            suffix_parts = [
                dataset_key or "default",
                tab_key,
                sheet_name.replace(" ", "_")
            ]
            start_suffix = "_".join(filter(None, suffix_parts))
            start_key = f"{start_suffix}_start_date"
            end_key = f"{start_suffix}_end_date"

            default_start = st.session_state.get(start_key, min_date)
            default_end = st.session_state.get(end_key, max_date)

            controls = st.columns([2, 2, 1, 1, 1])

            with controls[0]:
                st.markdown("**Start Date**")
                selected_start = st.date_input(
                    f"{start_suffix}_start_picker",
                    value=default_start,
                    min_value=min_date,
                    max_value=max_date,
                    key=start_key,
                    label_visibility="collapsed"
                )

            with controls[1]:
                st.markdown("**End Date**")
                selected_end = st.date_input(
                    f"{start_suffix}_end_picker",
                    value=default_end,
                    min_value=min_date,
                    max_value=max_date,
                    key=end_key,
                    label_visibility="collapsed"
                )

            def _set_quick_range(days: int | None):
                if days is None:
                    new_start = min_date
                    new_end = max_date
                else:
                    new_end = max_date
                    new_start = max(min_date, max_date - timedelta(days=days))
                st.session_state[start_key] = new_start
                st.session_state[end_key] = new_end
                st.rerun()

            with controls[2]:
                if st.button("30d", key=f"{start_suffix}_quick_30d", use_container_width=True):
                    _set_quick_range(30)
            with controls[3]:
                if st.button("90d", key=f"{start_suffix}_quick_90d", use_container_width=True):
                    _set_quick_range(90)
            with controls[4]:
                if st.button("All", key=f"{start_suffix}_quick_all", use_container_width=True):
                    _set_quick_range(None)

            if selected_start and selected_end:
                if selected_start > selected_end:
                    st.error("❌ Start date must be before end date")
                    return pd.DataFrame(), {}, selected_start, selected_end
                mask = (
                    (date_filtered_df['Day'].dt.date >= selected_start) &
                    (date_filtered_df['Day'].dt.date <= selected_end)
                )
                date_filtered_df = date_filtered_df[mask].copy()
            else:
                selected_start = selected_start or min_date
                selected_end = selected_end or max_date

        filtered_df = date_filtered_df.copy()

        if identifiers:
            st.divider()
            st.markdown("**Identifier Filters**")
            max_filters = min(9, len(identifiers))
            if max_filters > 0:
                grid_cols = st.columns(min(3, max_filters))
                for idx, identifier in enumerate(identifiers[:max_filters]):
                    with grid_cols[idx % len(grid_cols)]:
                        unique_vals = sorted(date_filtered_df[identifier].dropna().unique())
                        if 0 < len(unique_vals) <= 100:
                            selected_vals = st.multiselect(
                                f"**{identifier}**",
                                options=unique_vals,
                                key=f"{start_suffix}_filter_{identifier}"
                            )
                            if selected_vals:
                                active_filters[identifier] = selected_vals

        for col, values in active_filters.items():
            filtered_df = filtered_df[filtered_df[col].isin(values)]

    return filtered_df, active_filters, selected_start, selected_end

# ============================================================================
# ANALYSIS TABS
# ============================================================================

tabs = st.tabs([
    "📊 Overview",
    "📈 Explore",
    "🔄 Pivot Table",
    "🔗 Correlation",
    "🎯 Clustering",
    "📋 Report"
])

# ----------------------------------------------------------------------------
# TAB 1: OVERVIEW
# ----------------------------------------------------------------------------
with tabs[0]:
    st.markdown("""
    <div style="background: linear-gradient(135deg, var(--yellow-lightest) 0%, var(--background-white) 100%); 
         padding: 20px; border-radius: 12px; margin-bottom: 24px;">
        <h2 style="color: var(--text-dark); margin: 0;">📊 Data Overview</h2>
        <p style="color: var(--text-medium); margin: 8px 0 0 0;">View and filter your data with interactive controls</p>
    </div>
    """, unsafe_allow_html=True)
    
    render_dataset_selector_control("dataset_selector_overview", "Dataset")
    overview_sheet = render_workspace_selector("overview", st.session_state.active_dataset_key)
    df_overview, overview_filters, overview_start, overview_end = render_filter_controls(
        "overview",
        df,
        st.session_state.active_dataset_key,
        overview_sheet
    )

    if df_overview.empty:
        range_text = f"{overview_start} to {overview_end}" if has_dates and overview_start and overview_end else "the selected filters"
        st.warning(f"No data available for {range_text}. Adjust your filters and try again.")
        st.stop()

    # Summary metrics
    st.markdown("""
    <div class="card">
        <div class="card-title">📈 Key Metrics Summary</div>
    """, unsafe_allow_html=True)
    
    metric_cols = st.columns(min(len(metrics), 5))
    
    for idx, metric in enumerate(metrics[:5]):
        with metric_cols[idx]:
            value = df_overview[metric].sum()
            st.metric(
                label=metric.replace('_', ' '),
                value=format_value(value, metric)
            )
    
    if len(metrics) > 5:
        st.caption(f"+ {len(metrics) - 5} more metrics")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.divider()
    
    # Data table
    st.markdown("""
    <div class="card">
        <div class="card-title">📋 Data Table</div>
    """, unsafe_allow_html=True)
    
    display_columns = (['Day'] if has_dates else []) + identifiers + metrics
    display_columns = [col for col in display_columns if col in df_overview.columns]
    
    st.dataframe(
        df_overview[display_columns],
        use_container_width=True,
        height=450
    )
    
    # Export button
    csv_data = df_overview[display_columns].to_csv(index=False)
    st.download_button(
        "📥 Download Filtered Data",
        csv_data,
        f"data_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        "text/csv"
    )
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Comment section
    st.divider()
    st.markdown("### 💬 Add Insight")
    
    with st.expander("✍️ Save observations about this data", expanded=False):
        comment = st.text_area(
            "Your insights",
            placeholder="Example: Data shows strong Q4 performance with 25% increase in sales...",
            height=100,
            key="overview_comment_input"
        )
        
        if st.button("💾 Save Comment", key="save_overview_comment"):
            if comment.strip():
                context = {
                    'view': 'Overview',
                    'date_range': f"{overview_start} to {overview_end}" if has_dates and overview_start and overview_end else "All data",
                    'total_rows': len(df_overview),
                    'filters': list(overview_filters.keys()) if overview_filters else None
                }
                success, msg = CommentManager.save_comment(comment, "Overview", context)
                if success:
                    st.success("✅ Comment saved!")
                    st.rerun()
                else:
                    st.error(f"❌ {msg}")
            else:
                st.warning("⚠️ Please enter a comment")

# ----------------------------------------------------------------------------
# TAB 2: EXPLORE
# ----------------------------------------------------------------------------
with tabs[1]:
    st.markdown("""
    <div style="background: linear-gradient(135deg, var(--yellow-lightest) 0%, var(--background-white) 100%); 
         padding: 20px; border-radius: 12px; margin-bottom: 24px;">
        <h2 style="color: var(--text-dark); margin: 0;">📈 Data Exploration</h2>
        <p style="color: var(--text-medium); margin: 8px 0 0 0;">Visualize trends and patterns with interactive charts</p>
    </div>
    """, unsafe_allow_html=True)
    
    if not has_dates:
        st.warning("⚠️ Date column not found. This tab requires a date column for time-series visualization.")
        st.info("Please ensure your data has a date/day column, or use other tabs for analysis.")
        st.stop()
    
    render_dataset_selector_control("dataset_selector_explore", "Dataset")
    explore_sheet = render_workspace_selector("explore", st.session_state.active_dataset_key)
    explore_df, explore_filters, explore_start, explore_end = render_filter_controls(
        "explore",
        df,
        st.session_state.active_dataset_key,
        explore_sheet
    )

    if explore_df.empty:
        range_text = f"{explore_start} to {explore_end}" if explore_start and explore_end else "the selected filters"
        st.warning(f"No data available for {range_text}. Adjust your filters and try again.")
        st.stop()

    config_col, viz_col = st.columns([1, 3])
    
    with config_col:
        st.markdown("""
        <div class="card">
            <div class="card-title">⚙️ Configuration</div>
        """, unsafe_allow_html=True)
        
        if explore_start and explore_end:
            st.caption(f"Date range: {explore_start} to {explore_end}")

        # Aggregation
        st.markdown("**🔧 Aggregation**")
        agg_level = st.selectbox(
            "Level",
            ["Daily", "Weekly", "Monthly", "Quarterly"],
            key="explore_agg_level"
        )
        
        agg_method = st.selectbox(
            "Method",
            ["Sum", "Mean", "Median", "Min", "Max"],
            key="explore_agg_method"
        )
        
        st.divider()
        
        # Chart settings
        st.markdown("**📊 Chart**")
        chart_type = st.selectbox(
            "Type",
            ["Line", "Bar", "Area"],
            key="explore_chart_type"
        )
        
        # Metric selection
        chart_metrics = st.multiselect(
            "Metrics",
            metrics,
            default=metrics[:min(3, len(metrics))],
            key="explore_chart_metrics"
        )

        normalize_choice = st.selectbox(
            "Normalize metrics",
            ["None", "Min-Max (0-1)", "Z-score"],
            index=0,
            key="explore_normalize",
            help="Scale metrics to a common range for easier comparison."
        )
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with viz_col:
        st.markdown("""
        <div class="card">
            <div class="card-title">📊 Visualization</div>
        """, unsafe_allow_html=True)
        if not chart_metrics:
            st.info("👈 Select at least one metric to visualize")
        else:
            # Aggregate data
            df_agg = aggregate_data(explore_df, agg_level, agg_method.lower())

            if explore_start and explore_end:
                df_plot = df_agg[
                    (df_agg['Period'].dt.date >= explore_start) &
                    (df_agg['Period'].dt.date <= explore_end)
                ].copy()
            else:
                df_plot = df_agg.copy()

            range_text = f"{explore_start} to {explore_end}" if explore_start and explore_end else "the selected filters"

            if len(df_plot) == 0:
                st.warning(f"?? No data in selected range: {range_text}")
                st.stop()

            df_viz = df_plot.copy()

            if normalize_choice != "None":
                for metric in chart_metrics:
                    if metric not in df_viz.columns:
                        continue
                    series = pd.to_numeric(df_plot[metric], errors='coerce')
                    if series.notna().sum() == 0:
                        continue
                    if normalize_choice.startswith("Min-Max"):
                        col_min = series.min()
                        col_max = series.max()
                        denom = col_max - col_min
                        if pd.isna(denom) or denom == 0:
                            normalized = pd.Series(0, index=series.index)
                        else:
                            normalized = (series - col_min) / denom
                    else:  # Z-score
                        mean_val = series.mean()
                        std_val = series.std()
                        if pd.isna(std_val) or std_val == 0:
                            normalized = pd.Series(0, index=series.index)
                        else:
                            normalized = (series - mean_val) / std_val
                    df_viz[metric] = normalized

            y_axis_label = "Normalized Value" if normalize_choice != "None" else "Value"

            # Create visualization
            fig = go.Figure()
            
            for metric in chart_metrics:
                if metric in df_plot.columns:
                    if chart_type == "Line":
                        fig.add_trace(go.Scatter(
                            x=df_plot['Period'],
                            y=df_viz[metric] if normalize_choice != "None" else df_plot[metric],
                            name=metric.replace('_', ' '),
                            mode='lines+markers',
                            line=dict(width=2),
                            marker=dict(size=6)
                        ))
                    elif chart_type == "Bar":
                        fig.add_trace(go.Bar(
                            x=df_plot['Period'],
                            y=df_viz[metric] if normalize_choice != "None" else df_plot[metric],
                            name=metric.replace('_', ' ')
                        ))
                    else:  # Area
                        fig.add_trace(go.Scatter(
                            x=df_plot['Period'],
                            y=df_viz[metric] if normalize_choice != "None" else df_plot[metric],
                            name=metric.replace('_', ' '),
                            fill='tonexty' if metric != chart_metrics[0] else 'tozeroy',
                            line=dict(width=1)
                        ))
            
            fig.update_layout(
                title=f"{agg_method} by {agg_level} ({range_text})",
                xaxis_title="Date",
                yaxis_title=y_axis_label,
                height=600,
                hovermode='x unified',
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)

            if normalize_choice != "None":
                st.caption(f"Displayed series normalized via {normalize_choice}. Summary metrics below reflect original values.")
            
            # Summary statistics
            st.divider()
            st.markdown("**📊 Summary Statistics**")
            stats_cols = st.columns(len(chart_metrics))
            
            for idx, metric in enumerate(chart_metrics):
                with stats_cols[idx]:
                    total = df_plot[metric].sum()
                    avg = df_plot[metric].mean()
                    st.metric(
                        metric.replace('_', ' '),
                        format_value(total, metric),
                        delta=f"Avg: {format_value(avg, metric)}"
                    )
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Comment section
            st.divider()
            st.markdown("### 💬 Add Insight")
            
            with st.expander("✍️ Save observations about this visualization", expanded=False):
                comment = st.text_area(
                    "Your insights",
                    placeholder="Example: Notable spike in metrics during March due to seasonal campaign...",
                    height=100,
                    key="explore_comment_input"
                )
                
                if st.button("💾 Save Comment", key="save_explore_comment"):
                    if comment.strip():
                        context = {
                            'view': 'Explore',
                            'chart_type': chart_type,
                            'aggregation': f"{agg_method} by {agg_level}",
                            'date_range': range_text,
                            'metrics': chart_metrics,
                            'filters': list(explore_filters.keys()) if explore_filters else None,
                            'normalization': normalize_choice
                        }
                        success, msg = CommentManager.save_comment(comment, "Explore", context)
                        if success:
                            st.success("✅ Comment saved!")
                            st.rerun()
                        else:
                            st.error(f"❌ {msg}")
                    else:
                        st.warning("⚠️ Please enter a comment")

# ----------------------------------------------------------------------------
# TAB 3: PIVOT TABLE
# ----------------------------------------------------------------------------
with tabs[2]:
    st.markdown("""
    <div style="background: linear-gradient(135deg, var(--yellow-lightest) 0%, var(--background-white) 100%); 
         padding: 20px; border-radius: 12px; margin-bottom: 24px;">
        <h2 style="color: var(--text-dark); margin: 0;">🔄 Pivot Table Analysis</h2>
        <p style="color: var(--text-medium); margin: 8px 0 0 0;">Create dynamic pivot tables for multi-dimensional analysis</p>
    </div>
    """, unsafe_allow_html=True)
    
    render_dataset_selector_control("dataset_selector_pivot", "Dataset")
    pivot_sheet = render_workspace_selector("pivot", st.session_state.active_dataset_key)
    pivot_df, pivot_filters, pivot_start, pivot_end = render_filter_controls(
        "pivot",
        df,
        st.session_state.active_dataset_key,
        pivot_sheet
    )

    if pivot_df.empty:
        range_text = f"{pivot_start} to {pivot_end}" if pivot_start and pivot_end else "the selected filters"
        st.warning(f"No data available for {range_text}. Adjust your filters and try again.")
        st.stop()

    config_col, result_col = st.columns([1, 2.5])
    
    with config_col:
        st.markdown("""
        <div class="card">
            <div class="card-title">⚙️ Configuration</div>
        """, unsafe_allow_html=True)
        
        # Row fields
        st.markdown("**🎯 Rows** (Required)")
        available_rows = identifiers + (['Day'] if has_dates else [])
        pivot_rows = st.multiselect(
            "Select row fields",
            available_rows,
            key="pivot_rows",
            help="Fields that will become rows in the pivot table"
        )
        
        # Column fields
        st.markdown("**📊 Columns** (Optional)")
        pivot_cols = st.multiselect(
            "Select column fields",
            identifiers,
            key="pivot_cols",
            help="Fields that will become columns in the pivot table"
        )
        
        # Value fields
        st.markdown("**💰 Values** (Required)")
        pivot_vals = st.multiselect(
            "Select value fields",
            metrics,
            key="pivot_vals",
            help="Numeric fields to aggregate"
        )
        
        # Aggregation
        pivot_agg = st.selectbox(
            "Aggregation Method",
            ["Sum", "Mean", "Count", "Min", "Max", "Median"],
            key="pivot_agg"
        )

        show_column_pct = st.checkbox(
            "Show Column %",
            value=False,
            help="Display each value as a percentage of its column total (detail rows only)"
        )

        include_grand_totals = st.checkbox(
            "Show Grand Totals",
            value=True,
            help="Include an overall total row/column in the pivot result"
        )
        
        # Validation
        can_generate = len(pivot_rows) > 0 and len(pivot_vals) > 0
        
        if not can_generate:
            if not pivot_rows:
                st.warning("⚠️ Select at least one row field")
            if not pivot_vals:
                st.warning("⚠️ Select at least one value field")
        
        # Generate button
        if st.button(
            "🔄 Generate Pivot Table",
            type="primary",
            disabled=not can_generate,
            use_container_width=True
        ):
            st.session_state.pivot_generated = True
            st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with result_col:
        st.markdown("""
        <div class="card">
            <div class="card-title">📋 Pivot Results</div>
        """, unsafe_allow_html=True)
        if not can_generate:
            st.markdown("""
            <div style="background: var(--yellow-lightest); padding: 24px; border-radius: 12px; text-align: center;">
                <h3 style="color: var(--text-dark); margin-top: 0;">👈 Configure Pivot Table</h3>
                <p style="color: var(--text-medium); line-height: 1.6;">
                    <strong>Required:</strong><br>
                    • At least one <strong>Row field</strong><br>
                    • At least one <strong>Value field</strong><br><br>
                    <strong>Optional:</strong><br>
                    • Column fields for cross-tabulation<br><br>
                    Click <strong>Generate Pivot Table</strong> when ready
                </p>
            </div>
            """, unsafe_allow_html=True)
        elif st.session_state.get('pivot_generated', False):
            try:
                total_label = 'TOTAL'

                # Map aggregation functions
                agg_map = {
                    'Sum': 'sum',
                    'Mean': 'mean',
                    'Count': 'count',
                    'Min': 'min',
                    'Max': 'max',
                    'Median': 'median'
                }
                
                agg_func = {val: agg_map[pivot_agg] for val in pivot_vals}

                def _resolve_column_label(frame: pd.DataFrame, field_name: str):
                    """Match a row field name to the actual column label (handles MultiIndex columns)."""
                    if field_name in frame.columns:
                        return field_name
                    for col in frame.columns:
                        if isinstance(col, tuple) and len(col) > 0 and col[0] == field_name:
                            return col
                    return None

                def _is_total_column_label(col_label):
                    """Identify columns that represent grand totals so we can keep them as absolute values."""
                    if isinstance(col_label, tuple):
                        return any(
                            isinstance(level, str) and level.upper() == total_label
                            for level in col_label
                        )
                    return isinstance(col_label, str) and col_label.upper() == total_label

                def format_column_percentages(frame: pd.DataFrame) -> pd.DataFrame:
                    """Convert detail values to column percentages while preserving totals."""
                    formatted = frame.copy()
                    if not pivot_rows:
                        return formatted

                    row_column_labels = [
                        _resolve_column_label(formatted, row_field)
                        for row_field in pivot_rows
                    ]
                    row_column_labels = [col for col in row_column_labels if col is not None]

                    value_columns = [col for col in formatted.columns if col not in row_column_labels]
                    if not value_columns:
                        return formatted

                    numeric_subset = formatted[value_columns].apply(pd.to_numeric, errors='coerce')

                    if include_grand_totals and row_column_labels:
                        row_series = formatted[row_column_labels[0]].astype(str)
                        total_row_mask = row_series.str.upper() == total_label
                    else:
                        total_row_mask = pd.Series(False, index=formatted.index, dtype=bool)

                    detail_index = formatted.index[~total_row_mask]
                    if detail_index.empty:
                        return formatted

                    detail_value_cols = [col for col in value_columns if not _is_total_column_label(col)]
                    if not detail_value_cols:
                        return formatted

                    column_totals = numeric_subset.loc[detail_index, detail_value_cols].sum(axis=0).replace(0, np.nan)
                    percent_values = (numeric_subset[detail_value_cols].div(column_totals, axis=1) * 100).round(2)

                    formatted.loc[detail_index, detail_value_cols] = percent_values.loc[detail_index, detail_value_cols]
                    return formatted
                
                if not pivot_cols:
                    # Simple groupby
                    pivot_result = pivot_df.groupby(pivot_rows).agg(agg_func).reset_index()
                    
                    if include_grand_totals:
                        # Add totals row
                        totals = {row: total_label if idx == 0 else '' for idx, row in enumerate(pivot_rows)}
                        for val in pivot_vals:
                            totals[val] = pivot_df[val].agg(agg_map[pivot_agg])
                        
                        pivot_result = pd.concat([
                            pivot_result,
                            pd.DataFrame([totals])
                        ], ignore_index=True)
                else:
                    # Full pivot table
                    pivot_result = pd.pivot_table(
                        pivot_df,
                        values=pivot_vals,
                        index=pivot_rows,
                        columns=pivot_cols,
                        aggfunc=agg_func,
                        margins=include_grand_totals,
                        margins_name=total_label,
                        fill_value=0
                    ).reset_index()

                display_result = pivot_result.copy()
                if show_column_pct:
                    display_result = format_column_percentages(display_result)
                
                # Display pivot table
                st.dataframe(
                    display_result,
                    use_container_width=True,
                    height=500
                )
                
                # Export button
                csv_data = display_result.to_csv(index=False)
                st.download_button(
                    "📥 Download Pivot Table",
                    csv_data,
                    f"pivot_table_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    "text/csv"
                )
                
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Comment section
                st.divider()
                st.markdown("""
                <div class="card">
                    <div class="card-title">💬 Add Insight</div>
                """, unsafe_allow_html=True)
                
                with st.expander("✍️ Save observations about this pivot analysis", expanded=False):
                    comment = st.text_area(
                        "Your insights",
                        placeholder="Example: Pivot analysis reveals that Region A consistently outperforms...",
                        height=100,
                        key="pivot_comment_input"
                    )
                    
                    if st.button("💾 Save Comment", key="save_pivot_comment"):
                        if comment.strip():
                            context = {
                                'view': 'Pivot Table',
                                'rows': pivot_rows,
                                'columns': pivot_cols if pivot_cols else None,
                                'values': pivot_vals,
                                'aggregation': pivot_agg,
                                'date_range': f"{pivot_start} to {pivot_end}" if pivot_start and pivot_end else "All data",
                                'filters': list(pivot_filters.keys()) if pivot_filters else None
                            }
                            success, msg = CommentManager.save_comment(comment, "Pivot Table", context)
                            if success:
                                st.success("✅ Comment saved!")
                                st.rerun()
                            else:
                                st.error(f"❌ {msg}")
                        else:
                            st.warning("⚠️ Please enter a comment")
                
                st.markdown('</div>', unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"❌ Error generating pivot table: {str(e)}")
                st.exception(e)
        
        st.markdown('</div>', unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# TAB 4: CORRELATION
# ----------------------------------------------------------------------------
with tabs[3]:
    st.markdown("""
    <div style="background: linear-gradient(135deg, var(--yellow-lightest) 0%, var(--background-white) 100%); 
         padding: 20px; border-radius: 12px; margin-bottom: 24px;">
        <h2 style="color: var(--text-dark); margin: 0;">🔗 Correlation Analysis</h2>
        <p style="color: var(--text-medium); margin: 8px 0 0 0;">Discover relationships between metrics</p>
    </div>
    """, unsafe_allow_html=True)
    
    if len(metrics) < 2:
        st.warning("⚠️ Need at least 2 metrics for correlation analysis")
        st.info("Please select more metrics in the column classification section.")
        st.stop()
    
    render_dataset_selector_control("dataset_selector_correlation", "Dataset")
    corr_sheet = render_workspace_selector("correlation", st.session_state.active_dataset_key)
    corr_df, corr_filters, corr_start, corr_end = render_filter_controls(
        "correlation",
        df,
        st.session_state.active_dataset_key,
        corr_sheet
    )

    if corr_df.empty:
        range_text = f"{corr_start} to {corr_end}" if corr_start and corr_end else "the selected filters"
        st.warning(f"No data available for {range_text}. Adjust your filters and try again.")
        st.stop()

    config_col, viz_col = st.columns([1, 2.5])
    
    with config_col:
        st.markdown("""
        <div class="card">
            <div class="card-title">⚙️ Settings</div>
        """, unsafe_allow_html=True)
        
        corr_metrics = st.multiselect(
            "Select metrics to analyze",
            metrics,
            default=metrics[:min(5, len(metrics))],
            key="corr_metrics",
            help="Choose which metrics to include in correlation analysis"
        )
        
        corr_method = st.selectbox(
            "Correlation Method",
            ["Pearson", "Spearman", "Kendall"],
            key="corr_method",
            help="Pearson: Linear relationships | Spearman: Monotonic relationships | Kendall: Rank correlation"
        )
        
        if len(corr_metrics) >= 2:
            st.divider()
            st.markdown("**Threshold**")
            min_corr = st.slider(
                "Highlight correlations ≥",
                0.0, 1.0, 0.5, 0.05,
                key="min_corr"
            )
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with viz_col:
        st.markdown("""
        <div class="card">
            <div class="card-title">📊 Correlation Matrix</div>
        """, unsafe_allow_html=True)
        if len(corr_metrics) < 2:
            st.info("👈 Select at least 2 metrics for correlation analysis")
        else:
            # Calculate correlation matrix
            corr_matrix = corr_df[corr_metrics].corr(method=corr_method.lower())
            
            # Create heatmap
            fig = go.Figure(data=go.Heatmap(
                z=corr_matrix.values,
                x=[m.replace('_', ' ') for m in corr_matrix.columns],
                y=[m.replace('_', ' ') for m in corr_matrix.columns],
                colorscale='RdBu_r',
                zmid=0,
                zmin=-1,
                zmax=1,
                text=corr_matrix.values,
                texttemplate='%{text:.2f}',
                textfont={"size": 10},
                colorbar=dict(title="Correlation<br>Coefficient")
            ))
            
            fig.update_layout(
                title=f"{corr_method} Correlation Matrix",
                height=600,
                xaxis=dict(tickangle=-45)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Strong correlations table
            st.divider()
            st.markdown(f"**📊 Strong Correlations (|r| ≥ {min_corr})**")
            
            strong_corr = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i + 1, len(corr_matrix.columns)):
                    corr_val = corr_matrix.iloc[i, j]
                    if abs(corr_val) >= min_corr:
                        strong_corr.append({
                            'Metric 1': corr_matrix.columns[i],
                            'Metric 2': corr_matrix.columns[j],
                            'Correlation': corr_val,
                            'Strength': 'Strong Positive' if corr_val > 0 else 'Strong Negative'
                        })
            
            if strong_corr:
                strong_df = pd.DataFrame(strong_corr)
                strong_df = strong_df.sort_values('Correlation', key=abs, ascending=False)
                
                st.dataframe(
                    strong_df.style.format({'Correlation': '{:.3f}'}),
                    use_container_width=True
                )
                
                st.caption(f"Found {len(strong_corr)} strong correlations")
            else:
                st.info(f"No correlations found with |r| ≥ {min_corr}")
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Comment section
            st.divider()
            st.markdown("""
            <div class="card">
                <div class="card-title">💬 Add Insight</div>
            """, unsafe_allow_html=True)
            
            with st.expander("✍️ Save observations about correlations", expanded=False):
                comment = st.text_area(
                    "Your insights",
                    placeholder="Example: Strong positive correlation between X and Y suggests causation...",
                    height=100,
                    key="corr_comment_input"
                )
                
                if st.button("💾 Save Comment", key="save_corr_comment"):
                    if comment.strip():
                        context = {
                            'view': 'Correlation',
                            'method': corr_method,
                            'metrics': corr_metrics,
                            'strong_correlations': len(strong_corr),
                            'threshold': min_corr,
                            'date_range': f"{corr_start} to {corr_end}" if corr_start and corr_end else "All data",
                            'filters': list(corr_filters.keys()) if corr_filters else None
                        }
                        success, msg = CommentManager.save_comment(comment, "Correlation", context)
                        if success:
                            st.success("✅ Comment saved!")
                            st.rerun()
                        else:
                            st.error(f"❌ {msg}")
                    else:
                        st.warning("⚠️ Please enter a comment")
            
            st.markdown('</div>', unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# TAB 5: CLUSTERING
# ----------------------------------------------------------------------------
with tabs[4]:
    st.markdown("""
    <div style="background: linear-gradient(135deg, var(--yellow-lightest) 0%, var(--background-white) 100%); 
         padding: 20px; border-radius: 12px; margin-bottom: 24px;">
        <h2 style="color: var(--text-dark); margin: 0;">🎯 Clustering Analysis</h2>
        <p style="color: var(--text-medium); margin: 8px 0 0 0;">Discover patterns and group similar data points</p>
    </div>
    """, unsafe_allow_html=True)
    
    if len(metrics) < 2:
        st.warning("⚠️ Need at least 2 metrics for clustering analysis")
        st.info("Please select more metrics in the column classification section.")
        st.stop()
    
    render_dataset_selector_control("dataset_selector_clustering", "Dataset")
    cluster_sheet = render_workspace_selector("clustering", st.session_state.active_dataset_key)
    cluster_df, cluster_filters, cluster_start, cluster_end = render_filter_controls(
        "clustering",
        df,
        st.session_state.active_dataset_key,
        cluster_sheet
    )

    if cluster_df.empty:
        range_text = f"{cluster_start} to {cluster_end}" if cluster_start and cluster_end else "the selected filters"
        st.warning(f"No data available for {range_text}. Adjust your filters and try again.")
        st.stop()

    config_col, viz_col = st.columns([1, 2.5])
    
    with config_col:
        st.markdown("""
        <div class="card">
            <div class="card-title">⚙️ Settings</div>
        """, unsafe_allow_html=True)
        
        cluster_features = st.multiselect(
            "Select features for clustering",
            metrics,
            default=metrics[:min(3, len(metrics))],
            key="cluster_features",
            help="Choose numeric features to use for clustering"
        )
        
        if len(cluster_features) >= 2:
            algo = st.selectbox(
                "Algorithm",
                ["K-Means", "DBSCAN"],
                key="cluster_algo"
            )
            
            if algo == "K-Means":
                n_clusters = st.slider(
                    "Number of Clusters",
                    2, 10, 3,
                    key="n_clusters"
                )
            else:  # DBSCAN
                eps = st.slider(
                    "Epsilon (neighborhood size)",
                    0.1, 5.0, 0.5, 0.1,
                    key="dbscan_eps"
                )
                min_samples = st.slider(
                    "Minimum Samples",
                    2, 10, 5,
                    key="dbscan_min_samples"
                )
            
            st.divider()
            
            if len(cluster_features) >= 2:
                st.markdown("**Visualization**")
                viz_x = st.selectbox(
                    "X-axis",
                    cluster_features,
                    index=0,
                    key="cluster_viz_x"
                )
                viz_y = st.selectbox(
                    "Y-axis",
                    cluster_features,
                    index=min(1, len(cluster_features) - 1),
                    key="cluster_viz_y"
                )
            
            run_clustering = st.button(
                "🚀 Run Clustering",
                type="primary",
                use_container_width=True
            )
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with viz_col:
        st.markdown("""
        <div class="card">
            <div class="card-title">📊 Results</div>
        """, unsafe_allow_html=True)
        if len(cluster_features) < 2:
            st.info("👈 Select at least 2 features for clustering")
        elif 'run_clustering' in locals() and run_clustering:
            try:
                from sklearn.cluster import KMeans, DBSCAN
                from sklearn.preprocessing import StandardScaler
                
                # Prepare data
                cluster_data = cluster_df[cluster_features].dropna()
                
                if len(cluster_data) < 3:
                    st.error("❌ Not enough data points (minimum 3 required)")
                    st.stop()
                
                # Standardize features
                scaler = StandardScaler()
                scaled_data = scaler.fit_transform(cluster_data)
                
                # Perform clustering
                if algo == "K-Means":
                    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                    labels = model.fit_predict(scaled_data)
                    st.success(f"✅ K-Means clustering completed: {n_clusters} clusters")
                else:  # DBSCAN
                    model = DBSCAN(eps=eps, min_samples=min_samples)
                    labels = model.fit_predict(scaled_data)
                    n_clusters_found = len(set(labels)) - (1 if -1 in labels else 0)
                    n_noise = list(labels).count(-1)
                    st.success(f"✅ DBSCAN completed: {n_clusters_found} clusters, {n_noise} noise points")
                
                # Add cluster labels
                cluster_data['Cluster'] = labels
                
                # Create visualization
                fig = px.scatter(
                    cluster_data,
                    x=viz_x,
                    y=viz_y,
                    color='Cluster',
                    title=f"{algo} Clustering Results",
                    labels={'Cluster': 'Cluster ID'},
                    color_discrete_sequence=px.colors.qualitative.Set2
                )
                
                fig.update_layout(height=600)
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Cluster statistics
                st.divider()
                st.markdown("### 📊 Cluster Statistics")
                
                cluster_stats = cluster_data.groupby('Cluster')[cluster_features].mean()
                
                st.dataframe(
                    cluster_stats.style.background_gradient(cmap='YlOrRd', axis=1),
                    use_container_width=True
                )
                
                # Cluster sizes
                cluster_sizes = cluster_data['Cluster'].value_counts().sort_index()
                
                st.markdown("### 📈 Cluster Distribution")
                
                size_fig = px.bar(
                    x=cluster_sizes.index,
                    y=cluster_sizes.values,
                    labels={'x': 'Cluster ID', 'y': 'Number of Points'},
                    title='Data Points per Cluster'
                )
                size_fig.update_layout(height=300)
                
                st.plotly_chart(size_fig, use_container_width=True)
                
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Comment section
                st.divider()
                st.markdown("""
                <div class="card">
                    <div class="card-title">💬 Add Insight</div>
                """, unsafe_allow_html=True)
                
                with st.expander("✍️ Save observations about clusters", expanded=False):
                    comment = st.text_area(
                        "Your insights",
                        placeholder="Example: Cluster 1 represents high-value customers characterized by...",
                        height=100,
                        key="cluster_comment_input"
                    )
                    
                    if st.button("💾 Save Comment", key="save_cluster_comment"):
                        if comment.strip():
                            context = {
                                'view': 'Clustering',
                                'algorithm': algo,
                                'features': cluster_features,
                                'n_clusters': n_clusters if algo == "K-Means" else n_clusters_found,
                                'date_range': f"{cluster_start} to {cluster_end}" if cluster_start and cluster_end else "All data",
                                'filters': list(cluster_filters.keys()) if cluster_filters else None
                            }
                            success, msg = CommentManager.save_comment(comment, "Clustering", context)
                            if success:
                                st.success("✅ Comment saved!")
                                st.rerun()
                            else:
                                st.error(f"❌ {msg}")
                        else:
                            st.warning("⚠️ Please enter a comment")
                
                st.markdown('</div>', unsafe_allow_html=True)
                
            except ImportError:
                st.error("❌ scikit-learn is required for clustering")
                st.info("Install with: `pip install scikit-learn`")
            except Exception as e:
                st.error(f"❌ Clustering error: {str(e)}")
                st.exception(e)
        
        st.markdown('</div>', unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# TAB 6: REPORT
# ----------------------------------------------------------------------------
with tabs[5]:
    st.markdown("""
    <div style="background: linear-gradient(135deg, var(--yellow-lightest) 0%, var(--background-white) 100%); 
         padding: 20px; border-radius: 12px; margin-bottom: 24px;">
        <h2 style="color: var(--text-dark); margin: 0;">📋 Analysis Report</h2>
        <p style="color: var(--text-medium); margin: 8px 0 0 0;">View, manage, and export all your saved comments and insights</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load comments
    comments_df, error = CommentManager.load_comments()
    
    if error:
        st.error(f"⚠️ {error}")
    
    # Header
    header_col1, header_col2 = st.columns([3, 1])
    
    with header_col1:
        if not comments_df.empty:
            st.markdown(f"**{len(comments_df)} saved comments** across all analysis tabs")
        else:
            st.markdown("**No comments yet** - Add insights from any analysis tab")
    
    with header_col2:
        if not comments_df.empty:
            csv_export, export_error = CommentManager.export_comments()
            if csv_export:
                st.download_button(
                    "📥 Export CSV",
                    csv_export,
                    f"comments_export_{datetime.now().strftime('%Y%m%d')}.csv",
                    "text/csv",
                    use_container_width=True
                )
    
    st.divider()
    
    # Display comments
    if comments_df.empty:
        st.markdown("""
        <div class="card" style="text-align: center; padding: 48px;">
            <div style="font-size: 4em; margin-bottom: 16px;">📝</div>
            <h3 style="color: var(--text-dark); margin-bottom: 16px;">No Comments Yet</h3>
            <p style="color: var(--text-medium); font-size: 1.1em; margin-bottom: 24px;">
                Start adding insights and observations from any analysis tab
            </p>
            <div style="background: var(--yellow-lightest); padding: 20px; border-radius: 12px; margin: 0 auto; max-width: 500px;">
                <p style="color: var(--text-medium); margin: 0; line-height: 1.8;">
                    📊 <strong>Overview</strong> - Data insights<br>
                    📈 <strong>Explore</strong> - Chart observations<br>
                    🔄 <strong>Pivot Table</strong> - Analysis findings<br>
                    🔗 <strong>Correlation</strong> - Relationship insights<br>
                    🎯 <strong>Clustering</strong> - Pattern discoveries
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Sort by timestamp (newest first)
        comments_df = comments_df.sort_values('timestamp', ascending=False)
        
        # Display each comment
        for idx, row in comments_df.iterrows():
            comment_id = row['id']
            
            st.markdown('<div class="card">', unsafe_allow_html=True)
            
            with st.container():
                # Header
                col1, col2, col3 = st.columns([3, 1, 1])
                
                with col1:
                    timestamp = row['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
                    st.markdown(f"### 💬 {row['tab_name']} · {timestamp}")
                
                with col2:
                    if st.session_state.delete_confirm_id != comment_id:
                        if st.button("✏️ Edit", key=f"edit_btn_{comment_id}", use_container_width=True):
                            st.session_state.editing_comment_id = comment_id
                            st.rerun()
                
                with col3:
                    if st.session_state.delete_confirm_id != comment_id:
                        if st.button("🗑️ Delete", key=f"del_btn_{comment_id}", use_container_width=True):
                            st.session_state.delete_confirm_id = comment_id
                            st.rerun()
                
                # Delete confirmation
                if st.session_state.delete_confirm_id == comment_id:
                    st.warning("⚠️ Are you sure you want to delete this comment?")
                    confirm_col1, confirm_col2 = st.columns(2)
                    
                    with confirm_col1:
                        if st.button("✅ Yes, Delete", key=f"confirm_{comment_id}", use_container_width=True):
                            success, msg = CommentManager.delete_comment(comment_id)
                            if success:
                                st.session_state.delete_confirm_id = None
                                st.success("✅ Comment deleted")
                                st.rerun()
                            else:
                                st.error(f"❌ {msg}")
                    
                    with confirm_col2:
                        if st.button("❌ Cancel", key=f"cancel_{comment_id}", use_container_width=True):
                            st.session_state.delete_confirm_id = None
                            st.rerun()
                    
                    st.markdown("---")
                    continue
                
                # Context (if available)
                if row['context_data'] and row['context_data'] != 'null':
                    try:
                        context = json.loads(row['context_data'])
                        if context:
                            with st.expander("📊 Context", expanded=False):
                                for key, value in context.items():
                                    if value is not None:
                                        st.caption(f"**{key.replace('_', ' ').title()}:** {value}")
                    except:
                        pass
                
                # Edit mode
                if st.session_state.editing_comment_id == comment_id:
                    edit_text = st.text_area(
                        "Edit your comment",
                        value=row['comment_text'],
                        height=100,
                        key=f"edit_area_{comment_id}"
                    )
                    
                    edit_col1, edit_col2 = st.columns(2)
                    
                    with edit_col1:
                        if st.button("💾 Save Changes", key=f"save_{comment_id}", use_container_width=True):
                            if edit_text.strip():
                                success, msg = CommentManager.update_comment(comment_id, edit_text.strip())
                                if success:
                                    st.session_state.editing_comment_id = None
                                    st.success("✅ Comment updated")
                                    st.rerun()
                                else:
                                    st.error(f"❌ {msg}")
                            else:
                                st.warning("⚠️ Comment cannot be empty")
                    
                    with edit_col2:
                        if st.button("❌ Cancel", key=f"cancel_edit_{comment_id}", use_container_width=True):
                            st.session_state.editing_comment_id = None
                            st.rerun()
                else:
                    # Display mode
                    st.markdown(f"_{row['comment_text']}_")
                
                st.markdown('</div>', unsafe_allow_html=True)
                st.markdown('<br>', unsafe_allow_html=True)

# ============================================================================
# FOOTER
# ============================================================================

st.divider()
st.markdown("""
<div style="text-align: center; padding: 20px; color: var(--text-medium);">
    <p style="font-weight: 600; font-size: 1.1em;">📊 EDA Insights</p>
    <p style="font-size: 0.9em;">Production-Ready Data Analysis Platform · Version 1.0.0</p>
</div>
""", unsafe_allow_html=True)
