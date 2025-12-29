"""
Tida Sports Booking Analytics Dashboard
A comprehensive dashboard for venue booking analytics with advanced metrics and visualizations
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pymysql
from datetime import datetime, timedelta
import re
import plotly.io as pio

# ---------------------------
# Plotly: create a light template so charts are visible on a white Streamlit bg
# ---------------------------
pio.templates["tida_light"] = pio.templates["plotly_white"]
pio.templates["tida_light"].layout.update(
    paper_bgcolor="white",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#0a0e27", family="Inter"),
    colorway=["#00a896", "#ffd700", "#3a7bd5", "#ff6b6b", "#4ecdc4"]
)
pio.templates.default = "tida_light"

# -------------------------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------------------------
st.set_page_config(
    page_title="Tida Sports Analytics",
    layout="wide",
    page_icon="🏆",
    initial_sidebar_state="expanded"
)

# -------------------------------------------------------------------
# CUSTOM CSS FOR STYLING (adjusted for white background)
# -------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&family=Poppins:wght@400;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .main {
        background: linear-gradient(to bottom, #ffffff 0%, #ffffff 100%);
        background-attachment: fixed;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f3460 0%, #16213e 100%);
        border-right: 2px solid rgba(0, 169, 150, 0.25);
    }
    
    [data-testid="stSidebar"] * {
        color: #ffffff !important;
    }
    
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #00a896 !important;
    }
    
    /* Metric cards - visibility for light background */
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, rgba(0, 169, 150, 0.06) 0%, rgba(58, 123, 213, 0.04) 100%);
        backdrop-filter: blur(4px);
        border: 1px solid rgba(0, 169, 150, 0.15);
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 6px 18px rgba(10, 14, 39, 0.06);
    }
    
    [data-testid="stMetric"] label {
        color: #0a0e27 !important;
        font-size: 0.9rem !important;
        font-weight: 700 !important;
        letter-spacing: 1px;
        text-transform: uppercase;
    }
    
    [data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #00796b !important;
        font-size: 2.2rem !important;
        font-weight: 800 !important;
        font-family: 'Poppins', sans-serif !important;
        text-shadow: none !important;
    }
    
    [data-testid="stMetric"] [data-testid="stMetricDelta"] {
        color: #b8860b !important;
        font-weight: 600 !important;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #0a0e27 !important;
        font-weight: 700 !important;
        font-family: 'Poppins', sans-serif !important;
    }
    
    h1 {
        font-size: 2.4rem !important;
        color: #00a896 !important;
        margin-bottom: 1rem !important;
    }
    
    h2 {
        color: #00a896 !important;
        font-size: 1.6rem !important;
        margin-top: 1.6rem !important;
    }
    
    h3 {
        color: #0a0e27 !important;
        font-size: 1.1rem !important;
    }
    
    /* Text colors */
    p, span, div, label {
        color: #0a0e27 !important;
    }
    
    .stMarkdown {
        color: #0a0e27 !important;
    }
    
    /* Select boxes and inputs */
    .stSelectbox label, .stMultiSelect label, .stDateInput label, .stCheckbox label {
        color: #0a0e27 !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #00a896 0%, #3a7bd5 100%);
        color: #ffffff !important;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 1.5rem;
        font-weight: 700;
        letter-spacing: 0.5px;
        text-transform: uppercase;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(0, 169, 150, 0.15);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 18px rgba(0, 169, 150, 0.2);
    }
    
    /* Dataframe styling */
    [data-testid="stDataFrame"] {
        background: rgba(250, 250, 250, 1);
        border-radius: 12px;
        border: 1px solid rgba(0, 169, 150, 0.08);
    }
    
    /* Info/Success boxes */
    .stAlert, .stInfo {
        background: rgba(0, 169, 150, 0.06) !important;
        border-left: 4px solid #00a896 !important;
        color: #0a0e27 !important;
    }
    
    .stSuccess {
        background: rgba(0, 169, 150, 0.08) !important;
        border: 1px solid #00a896 !important;
        color: #0a0e27 !important;
    }
    
    .stWarning {
        background: rgba(255, 215, 0, 0.06) !important;
        border-left: 4px solid #ffd700 !important;
        color: #0a0e27 !important;
    }
    
    /* Date range display */
    .date-range-box {
        background: linear-gradient(135deg, rgba(0, 169, 150, 0.06) 0%, rgba(58, 123, 213, 0.04) 100%);
        border: 1px solid #00a896;
        border-radius: 12px;
        padding: 18px;
        text-align: center;
        margin: 20px 0;
    }
    
    .date-range-text {
        color: #0a0e27 !important;
        font-size: 1.05rem !important;
        font-weight: 700 !important;
    }
    
    .highlight-box {
        background: linear-gradient(135deg, rgba(0, 169, 150, 0.04) 0%, rgba(58, 123, 213, 0.03) 100%);
        border-left: 4px solid #00a896;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.6rem 0;
    }
    
    .highlight-box h4 {
        color: #00a896 !important;
        margin-top: 0 !important;
    }
    
    .highlight-box p {
        color: #0a0e27 !important;
        margin: 0.3rem 0 !important;
    }
    
    .element-container {
        color: #0a0e27 !important;
    }
    
    .stCaption {
        color: #4b5563 !important;
        font-size: 0.9rem !important;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------------
# DATABASE CONNECTION CONFIGURATION
# -------------------------------------------------------------------
import os

# Try Streamlit secrets first, then environment variables, then local defaults
try:
    # Check if running on Streamlit Cloud with secrets
    if "database" in st.secrets:
        DB_CONFIG = {
            'host': st.secrets["database"]["DB_HOST"],
            'port': int(st.secrets["database"]["DB_PORT"]),
            'user': st.secrets["database"]["DB_USER"],
            'password': st.secrets["database"]["DB_PASS"],
            'database': st.secrets["database"]["DB_NAME"]
        }
        st.sidebar.success(f"🔗 Connected to: {DB_CONFIG['host']}")
    else:
        raise KeyError("No secrets found")
except (KeyError, FileNotFoundError, AttributeError):
    # Fallback to environment variables or local config
    DB_CONFIG = {
        'host': os.getenv('DB_HOST', 'localhost'),
        'port': int(os.getenv('DB_PORT', '3307')),
        'user': os.getenv('DB_USER', 'root'),
        'password': os.getenv('DB_PASS', 'anshitdassdA2'),
        'database': os.getenv('DB_NAME', 'tida')
    }
    if os.getenv('DB_HOST'):
        st.sidebar.info(f"🔗 Using env vars: {DB_CONFIG['host']}")
    else:
        st.sidebar.warning("🔗 Using localhost (local development)")

@st.cache_data(ttl=300)
def load_and_clean_data():
    """Load data from MySQL and perform comprehensive cleaning"""
    
    try:
        # Create direct pymysql connection (this works!)
        connection = pymysql.connect(**DB_CONFIG)
        
        query = """
        SELECT 
            order_id,
            date_paid,
            product_id,
            variation_id,
            product_net_revenue,
            coupon_amount,
            order_status,
            payment_method_title,
            date_created_gmt,
            post_title,
            post_status,
            slot_start,
            slot_end,
            billing_address_index,
            tida_sports_type
        FROM tida_order_details
        WHERE post_status = 'publish'
            AND tida_sports_type NOT IN ('subscription', 'subscription_variation')
            AND LOWER(post_title) NOT LIKE '%test%'
            AND LOWER(post_title) NOT LIKE '%testing%'
        """
        
        df = pd.read_sql(query, connection)
        connection.close()
        
    except Exception as e:
        st.error(f"Database error: {e}")
        raise
    
    # Parse billing address to extract name, email, and phone
    def parse_billing_address(address):
        if pd.isna(address) or address == '':
            return pd.Series({'customer_name': None, 'email': None, 'phone': None})
        
        try:
            # Pattern: Name IN email phone
            parts = str(address).strip().split()
            
            # Find 'IN' or 'in' as separator
            in_index = -1
            for i, part in enumerate(parts):
                if part.upper() == 'IN':
                    in_index = i
                    break
            
            if in_index > 0:
                # Name is before IN
                name = ' '.join(parts[:in_index]).strip()
                
                # Email and phone are after IN
                remaining = parts[in_index + 1:]
                
                # Find email (contains @)
                email = None
                phone = None
                
                for item in remaining:
                    if '@' in item:
                        email = item
                    elif item.isdigit() and len(item) >= 10:
                        phone = item
                
                return pd.Series({'customer_name': name, 'email': email, 'phone': phone})
            else:
                return pd.Series({'customer_name': None, 'email': None, 'phone': None})
        except:
            return pd.Series({'customer_name': None, 'email': None, 'phone': None})
    
    # Apply parsing
    billing_info = df['billing_address_index'].apply(parse_billing_address)
    df = pd.concat([df, billing_info], axis=1)
    
    # Drop billing_address_index column if present
    if 'billing_address_index' in df.columns:
        df = df.drop(columns=['billing_address_index'])
    
    # Convert datetime columns
    datetime_cols = ['date_paid', 'date_created_gmt', 'slot_start', 'slot_end']
    for col in datetime_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
    
    # Extract time-based features (guard for missing columns)
    if 'date_paid' in df.columns:
        df['booking_date'] = df['date_paid'].dt.date
        df['booking_month'] = df['date_paid'].dt.month_name()
        df['booking_year'] = df['date_paid'].dt.year
        df['booking_week'] = df['date_paid'].dt.isocalendar().week
        df['booking_day_of_week'] = df['date_paid'].dt.day_name()
    else:
        df['booking_date'] = pd.NaT
        df['booking_day_of_week'] = None
    
    if 'slot_start' in df.columns:
        df['booking_hour'] = df['slot_start'].dt.hour
    else:
        df['booking_hour'] = None
    
    # Calculate slot duration in hours
    if 'slot_end' in df.columns and 'slot_start' in df.columns:
        df['slot_duration_hours'] = (df['slot_end'] - df['slot_start']).dt.total_seconds() / 3600
    else:
        df['slot_duration_hours'] = 0.0
    
    # Rename post_title to venue_name for clarity (guard)
    if 'post_title' in df.columns:
        df = df.rename(columns={'post_title': 'venue_name'})
    else:
        df['venue_name'] = None
    
    # Add is_package flag (variation_id > 0 means it's a package) - guard
    if 'variation_id' in df.columns:
        df['is_package'] = df['variation_id'] > 0
    else:
        df['is_package'] = False
    
    # Convert numeric columns safely
    if 'product_net_revenue' in df.columns:
        df['product_net_revenue'] = pd.to_numeric(df['product_net_revenue'], errors='coerce')
    else:
        df['product_net_revenue'] = 0.0
    if 'coupon_amount' in df.columns:
        df['coupon_amount'] = pd.to_numeric(df['coupon_amount'], errors='coerce')
    else:
        df['coupon_amount'] = 0.0
    
    # Calculate actual revenue (net revenue - coupon)
    df['actual_revenue'] = df['product_net_revenue'].fillna(0) - df['coupon_amount'].fillna(0)
    
    # Remove rows with missing critical data (guard)
    drop_subset = [c for c in ['date_paid', 'venue_name', 'product_net_revenue'] if c in df.columns]
    if len(drop_subset) > 0:
        df = df.dropna(subset=drop_subset)
    
    # Ensure email column exists (used downstream)
    if 'email' not in df.columns:
        df['email'] = None
    
    return df

# -------------------------------------------------------------------
# HELPER FUNCTIONS
# -------------------------------------------------------------------
def get_date_range(df, range_type, custom_start=None, custom_end=None):
    """Filter dataframe based on date range selection"""
    if 'date_paid' not in df.columns or df['date_paid'].isna().all():
        return df
    max_date = df['date_paid'].max()
    
    if range_type == "Last 7 Days":
        start_date = max_date - timedelta(days=7)
    elif range_type == "Last 14 Days":
        start_date = max_date - timedelta(days=14)
    elif range_type == "Last 30 Days":
        start_date = max_date - timedelta(days=30)
    elif range_type == "Last 2 Months":
        start_date = max_date - timedelta(days=60)
    elif range_type == "Custom" and custom_start and custom_end:
        start_date = pd.to_datetime(custom_start)
        return df[(df['date_paid'] >= start_date) & (df['date_paid'] <= pd.to_datetime(custom_end))]
    else:
        start_date = df['date_paid'].min()
    
    return df[df['date_paid'] >= start_date]

def create_kpi_cards(col, label, value, delta=None, delta_label="vs prev period"):
    """Create styled KPI metric cards"""
    with col:
        if delta is not None:
            st.metric(label=label, value=value, delta=f"{delta:+.1f}% {delta_label}")
        else:
            st.metric(label=label, value=value)

# -------------------------------------------------------------------
# VISUALIZATION FUNCTIONS (all update_layout/xaxes/yaxes use Plotly-safe syntax)
# -------------------------------------------------------------------
def create_revenue_trend(df):
    daily_revenue = df.groupby('booking_date').agg({
        'actual_revenue': 'sum',
        'order_id': 'count'
    }).reset_index()

    daily_revenue.columns = ['Date', 'Revenue', 'Bookings']
    daily_revenue = daily_revenue.sort_values('Date')

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Scatter(
            x=daily_revenue['Date'],
            y=daily_revenue['Revenue'],
            name="Revenue (₹)",
            fill='tozeroy',
            line=dict(color='#00a896', width=3),
            fillcolor='rgba(0,168,150,0.18)'
        ),
        secondary_y=False
    )

    fig.add_trace(
        go.Scatter(
            x=daily_revenue['Date'],
            y=daily_revenue['Bookings'],
            name="Bookings",
            mode='lines+markers',
            line=dict(color='#ffd700', width=2),
            marker=dict(size=6, color='#ffd700')
        ),
        secondary_y=True
    )

    fig.update_layout(
        title=dict(text="📈 Revenue & Booking Trends Over Time", font=dict(size=18, color="#0a0e27")),
        hovermode="x unified",
        height=450,
        margin=dict(l=60, r=60, t=80, b=60),
        legend=dict(orientation="h", y=1.05, x=1, xanchor="right")
    )

    fig.update_xaxes(title=dict(text="Date", font=dict(color="#0a0e27", size=12)), showgrid=True, gridcolor="rgba(0,0,0,0.05)")
    fig.update_yaxes(title=dict(text="Revenue (₹)", font=dict(color="#0a0e27", size=12)), secondary_y=False)
    fig.update_yaxes(title=dict(text="Bookings", font=dict(color="#0a0e27", size=12)), secondary_y=True, showgrid=False)

    return fig


def create_venue_performance(df):
    venue_stats = df.groupby('venue_name').agg({
        'actual_revenue': 'sum',
        'order_id': 'count',
        'slot_duration_hours': 'sum'
    }).reset_index()
    venue_stats.columns = ['Venue', 'Revenue', 'Bookings', 'Total Hours']
    
    top_10_revenue = venue_stats.nlargest(10, 'Revenue').sort_values('Revenue', ascending=True)
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=top_10_revenue['Venue'],
        x=top_10_revenue['Revenue'],
        orientation='h',
        marker=dict(
            color=top_10_revenue['Revenue'],
            colorscale=[[0, '#3a7bd5'], [0.5, '#00a896'], [1, '#ffd700']],
            showscale=False,
            line=dict(color='rgba(0, 168, 150, 0.25)', width=1)
        ),
        text=[f"₹{x:,.0f}" for x in top_10_revenue['Revenue']],
        textposition='outside',
        textfont=dict(color='#0a0e27', size=11, family='Inter'),
        hovertemplate='<b>%{y}</b><br>Revenue: ₹%{x:,.0f}<br>Bookings: %{customdata}<extra></extra>',
        customdata=top_10_revenue['Bookings']
    ))
    
    fig.update_layout(
        title=dict(text="🏆 Top 10 Venues by Revenue", font=dict(size=18, color="#0a0e27")),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="white",
        font=dict(color="#0a0e27", family="Inter"),
        height=480,
        showlegend=False,
        margin=dict(l=180, r=80, t=70, b=40)
    )

    fig.update_xaxes(title=dict(text="Revenue (₹)", font=dict(color="#0a0e27", size=12)), showgrid=True, gridcolor="rgba(0,0,0,0.05)")
    fig.update_yaxes(title=dict(text=""), showgrid=False)
    
    return fig

def create_venue_bookings_chart(df):
    venue_stats = df.groupby('venue_name').agg({
        'actual_revenue': 'sum',
        'order_id': 'count'
    }).reset_index()
    venue_stats.columns = ['Venue', 'Revenue', 'Bookings']
    
    top_10_bookings = venue_stats.nlargest(10, 'Bookings').sort_values('Bookings', ascending=True)
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=top_10_bookings['Venue'],
        x=top_10_bookings['Bookings'],
        orientation='h',
        marker=dict(
            color=top_10_bookings['Bookings'],
            colorscale=[[0, '#764ba2'], [0.5, '#667eea'], [1, '#00a896']],
            showscale=False,
            line=dict(color='rgba(102, 126, 234, 0.2)', width=1)
        ),
        text=[f"{x:,}" for x in top_10_bookings['Bookings']],
        textposition='outside',
        textfont=dict(color='#0a0e27', size=11, family='Inter'),
        hovertemplate='<b>%{y}</b><br>Bookings: %{x}<br>Revenue: ₹%{customdata:,.0f}<extra></extra>',
        customdata=top_10_bookings['Revenue']
    ))
    
    fig.update_layout(
        title=dict(text="📊 Top 10 Venues by Bookings", font=dict(size=18, color="#0a0e27")),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='white',
        font=dict(color='#0a0e27', family='Inter'),
        height=480,
        showlegend=False,
        margin=dict(l=180, r=80, t=70, b=40)
    )

    fig.update_xaxes(title=dict(text="Number of Bookings", font=dict(color="#0a0e27", size=12)), showgrid=True, gridcolor="rgba(0,0,0,0.05)")
    fig.update_yaxes(title=dict(text=""), showgrid=False)
    
    return fig

def create_time_heatmap(df):
    heatmap_data = df.groupby(['booking_day_of_week', 'booking_hour']).size().reset_index(name='count')
    
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    heatmap_pivot = heatmap_data.pivot(index='booking_day_of_week', columns='booking_hour', values='count').fillna(0)
    heatmap_pivot = heatmap_pivot.reindex(day_order)
    
    fig = go.Figure(data=go.Heatmap(
        z=heatmap_pivot.values,
        x=heatmap_pivot.columns,
        y=heatmap_pivot.index,
        colorscale='Viridis',
        hovertemplate='Day: %{y}<br>Hour: %{x}:00<br>Bookings: %{z}<extra></extra>'
    ))
    
    fig.update_layout(
        title=dict(text="Booking Density by Day & Hour", font=dict(size=16, color="#0a0e27")),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='white',
        font=dict(color='#0a0e27', family='Outfit'),
        xaxis_title="Hour of Day",
        yaxis_title="Day of Week",
        height=400
    )
    
    return fig

def create_payment_distribution(df):
    payment_stats = df.groupby('payment_method_title').agg({
        'order_id': 'count',
        'actual_revenue': 'sum'
    }).reset_index()
    payment_stats.columns = ['Payment Method', 'Count', 'Revenue']
    payment_stats = payment_stats.sort_values('Revenue', ascending=False)
    
    payment_stats['Revenue %'] = (payment_stats['Revenue'] / payment_stats['Revenue'].sum() * 100).round(2)
    payment_stats['Bookings %'] = (payment_stats['Count'] / payment_stats['Count'].sum() * 100).round(2)
    
    fig = go.Figure()
    
    colors = ['#00a896', '#3a7bd5', '#ffd700', '#ff6b6b', '#4ecdc4', '#95e1d3', '#f38181', '#ffd89b']
    
    fig.add_trace(
        go.Pie(
            labels=payment_stats['Payment Method'],
            values=payment_stats['Revenue'],
            marker=dict(colors=colors[:len(payment_stats)], line=dict(color='white', width=1)),
            hole=0.35,
            textposition='auto',
            textinfo='percent',
            textfont=dict(color='#0a0e27', size=12),
            hovertemplate='<b>%{label}</b><br>Revenue: ₹%{value:,.0f}<extra></extra>',
            showlegend=True
        )
    )
    
    fig.update_layout(
        title=dict(text="💳 Revenue by Payment Method", font=dict(size=16, color="#0a0e27")),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='white',
        font=dict(color='#0a0e27', family='Inter'),
        showlegend=True,
        height=460,
        margin=dict(l=40, r=200, t=70, b=40)
    )
    
    config = {'displayModeBar': False}
    
    return fig, payment_stats, config

def create_monthly_comparison(df):
    df_copy = df.copy()
    df_copy['year_month'] = df_copy['date_paid'].dt.to_period('M')
    
    monthly = df_copy.groupby('year_month').agg({
        'actual_revenue': 'sum',
        'order_id': 'count'
    }).reset_index()
    
    monthly['year_month_str'] = monthly['year_month'].astype(str)
    monthly = monthly.sort_values('year_month')
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=monthly['year_month_str'],
        y=monthly['actual_revenue'],
        name='Revenue',
        marker=dict(
            color=monthly['actual_revenue'],
            colorscale=[[0, '#3a7bd5'], [0.5, '#00a896'], [1, '#ffd700']],
            showscale=False,
            line=dict(color='rgba(0, 168, 150, 0.2)', width=1)
        ),
        text=[f"₹{x:,.0f}" for x in monthly['actual_revenue']],
        textposition='outside',
        textfont=dict(color='#0a0e27', size=11),
        hovertemplate='<b>%{x}</b><br>Revenue: ₹%{y:,.0f}<extra></extra>'
    ))
    
    fig.add_trace(go.Scatter(
        x=monthly['year_month_str'],
        y=monthly['order_id'],
        name='Bookings',
        mode='lines+markers',
        marker=dict(size=10, color='#ffd700', line=dict(color='white', width=1)),
        line=dict(width=2, color='#ffd700'),
        yaxis='y2',
        hovertemplate='<b>%{x}</b><br>Bookings: %{y}<extra></extra>'
    ))
    
    fig.update_layout(
        title=dict(text="📅 Monthly Revenue & Booking Trends", font=dict(size=16, color="#0a0e27")),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='white',
        font=dict(color='#0a0e27', family='Inter', size=12),
        hovermode='x unified',
        height=430,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    fig.update_xaxes(title=dict(text='Month', font=dict(color='#0a0e27', size=12)), showgrid=True, gridcolor='rgba(0,0,0,0.05)')
    fig.update_yaxes(title=dict(text='Revenue (₹)', font=dict(color='#0a0e27', size=12)), side='left', showgrid=True, gridcolor='rgba(0,0,0,0.05)', secondary_y=False)
    fig.update_layout(yaxis2=dict(title=dict(text='Bookings', font=dict(color='#0a0e27', size=12)), overlaying='y', side='right', showgrid=False))
    
    return fig

def create_customer_analysis(df):
    customer_stats = df.groupby('email').agg({
        'order_id': 'count',
        'actual_revenue': 'sum'
    }).reset_index()
    customer_stats.columns = ['Email', 'Total Bookings', 'Total Spent']
    customer_stats = customer_stats.sort_values('Total Spent', ascending=False)
    
    segments = pd.cut(customer_stats['Total Bookings'], bins=[0, 1, 3, 10, float('inf')],
                      labels=['One-time', 'Occasional', 'Regular', 'VIP'])
    customer_stats['Segment'] = segments
    
    segment_summary = customer_stats.groupby('Segment').agg({
        'Email': 'count',
        'Total Spent': 'sum'
    }).reset_index()
    segment_summary.columns = ['Segment', 'Customer Count', 'Total Revenue']
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=segment_summary['Segment'],
        y=segment_summary['Customer Count'],
        name='Customers',
        marker=dict(
            color=['#f38181', '#ffd89b', '#a8dadc', '#64ffda'],
            line=dict(color='rgba(0,0,0,0.06)', width=1)
        ),
        text=segment_summary['Customer Count'],
        textposition='outside',
        hovertemplate='%{x}<br>Customers: %{y}<extra></extra>'
    ))
    
    fig.update_layout(
        title=dict(text="Customer Segmentation", font=dict(size=16, color="#0a0e27")),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='white',
        font=dict(color='#0a0e27', family='Outfit'),
        xaxis_title="Customer Segment",
        yaxis_title="Number of Customers",
        height=400,
        showlegend=False
    )
    
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridcolor='rgba(0,0,0,0.05)')
    
    return fig, customer_stats

# -------------------------------------------------------------------
# MAIN DASHBOARD
# -------------------------------------------------------------------
def main():
    # Header
    st.markdown("<h1>🏆 Tida Sports Analytics Dashboard</h1>", unsafe_allow_html=True)
    
    # Load data
    try:
        with st.spinner("🔄 Loading data from database..."):
            df = load_and_clean_data()
        
        st.success(f"✅ Loaded {len(df):,} bookings successfully!")
        
    except Exception as e:
        st.error(f"❌ Error connecting to database: {str(e)}")
        st.info("Please check your database connection settings and ensure the MySQL server is running.")
        return
    
    # Sidebar Filters
    st.sidebar.markdown("## 📊 Dashboard Filters")
    st.sidebar.markdown("---")
    
    # Date Range Filter
    st.sidebar.markdown("### 📅 Date Range")
    date_range_options = ["Last 7 Days", "Last 14 Days", "Last 30 Days", "Last 2 Months", "All Time", "Custom"]
    selected_range = st.sidebar.selectbox("Select Time Period", date_range_options, index=2)
    
    custom_start = None
    custom_end = None
    
    if selected_range == "Custom":
        st.sidebar.markdown("**Custom Date Selection:**")
        col1, col2 = st.sidebar.columns(2)
        with col1:
            custom_start = st.date_input("From", value=df['date_paid'].min().date())
        with col2:
            custom_end = st.date_input("To", value=df['date_paid'].max().date())
    
    # Apply date filter
    if selected_range == "All Time":
        filtered_df = df.copy()
    else:
        filtered_df = get_date_range(df, selected_range, custom_start, custom_end)
    
    st.sidebar.markdown("---")
    
    # Venue Filter
    st.sidebar.markdown("### 🏟️ Venue Selection")
    all_venues = sorted(df['venue_name'].dropna().unique())
    
    venue_select_all = st.sidebar.checkbox("Select All Venues", value=True, key='venue_all')
    if venue_select_all:
        selected_venues = all_venues
    else:
        selected_venues = st.sidebar.multiselect("Choose Venues", all_venues, default=all_venues[:3] if len(all_venues) >= 3 else all_venues)
    
    if selected_venues and len(selected_venues) > 0:
        filtered_df = filtered_df[filtered_df['venue_name'].isin(selected_venues)]
    else:
        st.sidebar.warning("⚠️ Please select at least one venue")
        filtered_df = filtered_df[filtered_df['venue_name'].isin(all_venues)]
        selected_venues = all_venues
    
    st.sidebar.markdown("---")
    
    # Payment Method Filter
    st.sidebar.markdown("### 💳 Payment Method")
    all_payment_methods = sorted(df['payment_method_title'].dropna().unique())
    payment_select_all = st.sidebar.checkbox("All Payment Methods", value=True)
    if payment_select_all:
        selected_payment = all_payment_methods
    else:
        selected_payment = st.sidebar.multiselect("Choose Payment Methods", all_payment_methods, default=[])
    if selected_payment:
        filtered_df = filtered_df[filtered_df['payment_method_title'].isin(selected_payment)]
    
    st.sidebar.markdown("---")
    
    # Order Status Filter
    st.sidebar.markdown("### 📋 Order Status")
    all_statuses = sorted(df['order_status'].dropna().unique())
    status_select_all = st.sidebar.checkbox("All Statuses", value=True)
    if status_select_all:
        selected_statuses = all_statuses
    else:
        selected_statuses = st.sidebar.multiselect("Choose Status", all_statuses, default=[])
    if selected_statuses:
        filtered_df = filtered_df[filtered_df['order_status'].isin(selected_statuses)]
    
    st.sidebar.markdown("---")
    
    # Filter Summary
    st.sidebar.markdown("### 📊 Filter Summary")
    st.sidebar.info(f"""
    **Active Filters:**
    - **Period:** {selected_range}
    - **Venues:** {len(selected_venues)} selected
    - **Payments:** {len(selected_payment)} types
    - **Status:** {len(selected_statuses)} types
    
    **Showing:** {len(filtered_df):,} bookings
    """)
    
    # Quick Stats in Sidebar
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🎯 Quick Stats")
    st.sidebar.metric("Total Revenue", f"₹{filtered_df['actual_revenue'].sum():,.0f}")
    st.sidebar.metric("Avg per Booking", f"₹{filtered_df['actual_revenue'].mean():,.0f}")
    st.sidebar.metric("Total Bookings", f"{len(filtered_df):,}")
    
    # Check if data exists after filtering
    if len(filtered_df) == 0:
        st.warning("⚠️ No data available for the selected filters. Please adjust your filters.")
        return
    
    # Display date range prominently AFTER filtering
    date_min = filtered_df['date_paid'].min().strftime('%d %B %Y')
    date_max = filtered_df['date_paid'].max().strftime('%d %B %Y')
    total_days = (filtered_df['date_paid'].max() - filtered_df['date_paid'].min()).days
    
    st.markdown(f"""
    <div class='date-range-box'>
        <div class='date-range-text'>
            📅 Data Period: <strong>{date_min}</strong> to <strong>{date_max}</strong> 
            ({total_days} days | {len(filtered_df):,} bookings)
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # -------------------------------------------------------------------
    # KEY METRICS
    # -------------------------------------------------------------------
    st.markdown("## 📈 Key Performance Indicators")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    total_revenue = filtered_df['actual_revenue'].sum()
    total_bookings = len(filtered_df)
    avg_booking_value = filtered_df['actual_revenue'].mean()
    total_customers = filtered_df['email'].nunique()
    total_hours_booked = filtered_df['slot_duration_hours'].sum()
    
    create_kpi_cards(col1, "Total Revenue", f"₹{total_revenue:,.0f}")
    create_kpi_cards(col2, "Total Bookings", f"{total_bookings:,}")
    create_kpi_cards(col3, "Avg Booking Value", f"₹{avg_booking_value:,.0f}")
    create_kpi_cards(col4, "Unique Customers", f"{total_customers:,}")
    create_kpi_cards(col5, "Hours Booked", f"{total_hours_booked:,.0f}h")
    
    # Second KPI row
    col1, col2, col3, col4, col5 = st.columns(5)
    total_venues = filtered_df['venue_name'].nunique()
    total_discount = filtered_df['coupon_amount'].sum()
    avg_slot_duration = filtered_df['slot_duration_hours'].mean()
    bookings_per_day = total_bookings / max(1, (filtered_df['date_paid'].max() - filtered_df['date_paid'].min()).days)
    repeat_customers = filtered_df.groupby('email').size()
    repeat_rate = (repeat_customers > 1).sum() / max(1, total_customers) * 100
    
    create_kpi_cards(col1, "Active Venues", f"{total_venues}")
    create_kpi_cards(col2, "Total Discounts", f"₹{total_discount:,.0f}")
    create_kpi_cards(col3, "Avg Slot Duration", f"{avg_slot_duration:.1f}h")
    create_kpi_cards(col4, "Bookings/Day", f"{bookings_per_day:.1f}")
    create_kpi_cards(col5, "Repeat Customer %", f"{repeat_rate:.1f}%")
    
    st.markdown("---")
    
    # -------------------------------------------------------------------
    # REVENUE & BOOKING TRENDS
    # -------------------------------------------------------------------
    st.markdown("### 📊 Revenue & Booking Trends")
    col1, col2 = st.columns([2, 1])
    with col1:
        st.plotly_chart(create_revenue_trend(filtered_df), use_container_width=True)
    with col2:
        st.markdown("#### 💡 Key Insights")
        if len(filtered_df) > 0:
            mid_point = len(filtered_df) // 2
            first_half = filtered_df.iloc[:mid_point]
            second_half = filtered_df.iloc[mid_point:]
            if len(first_half) > 0 and len(second_half) > 0:
                first_half_revenue = first_half['actual_revenue'].sum()
                second_half_revenue = second_half['actual_revenue'].sum()
                if first_half_revenue > 0:
                    growth = ((second_half_revenue - first_half_revenue) / first_half_revenue) * 100
                    st.markdown(f"""
                    <div class='highlight-box'>
                        <h4 style='margin-top: 0;'>📈 Period Growth</h4>
                        <p style='font-size: 1.5rem; font-weight: 700; color: #00796b; margin: 0.5rem 0;'>{growth:+.1f}%</p>
                        <p style='margin: 0; color: #4b5563;'>Revenue change between periods</p>
                    </div>
                    """, unsafe_allow_html=True)
        peak_day = filtered_df['booking_day_of_week'].value_counts().idxmax()
        st.markdown(f"**🔥 Peak Day:** {peak_day}")
        peak_hour = filtered_df['booking_hour'].value_counts().idxmax()
        st.markdown(f"**⏰ Peak Hour:** {peak_hour}:00")
        top_venue = filtered_df.groupby('venue_name')['actual_revenue'].sum().idxmax()
        st.markdown(f"**🏅 Top Venue:** {top_venue}")
    
    st.markdown("---")
    
    # -------------------------------------------------------------------
    # VENUE PERFORMANCE
    # -------------------------------------------------------------------
    st.markdown("## 🏟️ Venue Performance Analysis")
    st.plotly_chart(create_venue_performance(filtered_df), use_container_width=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.plotly_chart(create_venue_bookings_chart(filtered_df), use_container_width=True)
    
    st.markdown("---")
    
    # -------------------------------------------------------------------
    # DETAILED VENUE ANALYSIS (unchanged logic)
    # -------------------------------------------------------------------
    st.markdown("## 🎯 Detailed Venue Analysis")
    venue_detailed = filtered_df.groupby('venue_name').agg({
        'actual_revenue': ['sum', 'mean', 'count'],
        'slot_duration_hours': 'sum',
        'email': 'nunique',
        'coupon_amount': 'sum'
    }).reset_index()
    venue_detailed.columns = ['Venue', 'Total Revenue', 'Avg Revenue', 'Bookings', 'Total Hours', 'Unique Customers', 'Total Discounts']
    venue_detailed['Revenue per Hour'] = venue_detailed.apply(lambda x: x['Total Revenue'] / x['Total Hours'] if x['Total Hours'] > 0 else 0, axis=1)
    venue_detailed['Customer Retention'] = (venue_detailed['Bookings'] / venue_detailed['Unique Customers']).round(2)
    venue_detailed = venue_detailed.sort_values('Total Revenue', ascending=False)
    
    col1, col2, col3, col4 = st.columns(4)
    top_venue = venue_detailed.iloc[0]
    with col1:
        st.markdown(f"""
        <div class='highlight-box'>
            <h4 style='margin-top: 0; color: #00a896; text-align: center;'>💰 Highest Revenue</h4>
            <p style='font-size: 1.1rem; font-weight: 700; color: #0a0e27; margin: 0.5rem 0; text-align: center;'>{top_venue['Venue']}</p>
            <p style='margin: 0; color: #4b5563; text-align: center; font-size: 1rem;'>₹{top_venue['Total Revenue']:,.0f}</p>
        </div>
        """, unsafe_allow_html=True)
    most_booked_venue = venue_detailed.nlargest(1, 'Bookings').iloc[0]
    with col2:
        st.markdown(f"""
        <div class='highlight-box'>
            <h4 style='margin-top: 0; color: #ffd700; text-align: center;'>🎯 Most Bookings</h4>
            <p style='font-size: 1.1rem; font-weight: 700; color: #0a0e27; margin: 0.5rem 0; text-align: center;'>{most_booked_venue['Venue']}</p>
            <p style='margin: 0; color: #4b5563; text-align: center; font-size: 1rem;'>{most_booked_venue['Bookings']:,.0f} bookings</p>
        </div>
        """, unsafe_allow_html=True)
    best_retention_venue = venue_detailed.nlargest(1, 'Customer Retention').iloc[0]
    with col3:
        st.markdown(f"""
        <div class='highlight-box'>
            <h4 style='margin-top: 0; color: #3a7bd5; text-align: center;'>🔄 Best Retention</h4>
            <p style='font-size: 1.1rem; font-weight: 700; color: #0a0e27; margin: 0.5rem 0; text-align: center;'>{best_retention_venue['Venue']}</p>
            <p style='margin: 0; color: #4b5563; text-align: center; font-size: 1rem;'>{best_retention_venue['Customer Retention']:.1f}x bookings/customer</p>
        </div>
        """, unsafe_allow_html=True)
    highest_rpb_venue = venue_detailed.nlargest(1, 'Avg Revenue per Booking').iloc[0] if 'Avg Revenue per Booking' in venue_detailed.columns else venue_detailed.nlargest(1, 'Avg Revenue').iloc[0]
    with col4:
        st.markdown(f"""
        <div class='highlight-box'>
            <h4 style='margin-top: 0; color: #ff6b6b; text-align: center;'>💎 Highest Value</h4>
            <p style='font-size: 1.1rem; font-weight: 700; color: #0a0e27; margin: 0.5rem 0; text-align: center;'>{highest_rpb_venue['Venue']}</p>
            <p style='margin: 0; color: #4b5563; text-align: center; font-size: 1rem;'>₹{(highest_rpb_venue.get('Avg Revenue per Booking') if 'Avg Revenue per Booking' in highest_rpb_venue else highest_rpb_venue['Avg Revenue']):,.0f} per booking</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 📊 Comprehensive Venue Metrics")
    st.info("**Note:** Unique Customers and Customer Retention are tracked using email addresses.")
    st.dataframe(
        venue_detailed.style.format({
            'Total Revenue': '₹{:,.0f}',
            'Avg Revenue': '₹{:,.0f}',
            'Bookings': '{:,}',
            'Total Hours': '{:,.0f}h',
            'Unique Customers': '{:,}',
            'Total Discounts': '₹{:,.0f}',
            'Revenue per Hour': '₹{:,.0f}/h',
            'Customer Retention': '{:.2f}x'
        }).background_gradient(subset=['Total Revenue'], cmap='YlGn'),
        use_container_width=True,
        height=400
    )
    
    st.markdown("---")
    
    # ADDITIONAL VENUE INSIGHTS (unchanged logic but Plotly-safe layout)
    st.markdown("## 📈 Venue Efficiency Metrics")
    st.markdown("### ⏰ Average Slot Duration by Venue (Top 10)")
    slot_duration_venue = filtered_df.groupby('venue_name')['slot_duration_hours'].mean().reset_index()
    slot_duration_venue.columns = ['Venue', 'Avg Duration']
    slot_duration_venue = slot_duration_venue.nlargest(10, 'Avg Duration').sort_values('Avg Duration', ascending=True)
    
    fig_duration = go.Figure()
    fig_duration.add_trace(go.Bar(
        y=slot_duration_venue['Venue'],
        x=slot_duration_venue['Avg Duration'],
        orientation='h',
        marker=dict(color=slot_duration_venue['Avg Duration'], colorscale='Teal', showscale=False, line=dict(color='rgba(0,168,150,0.12)', width=1)),
        text=[f"{x:.1f}h" for x in slot_duration_venue['Avg Duration']],
        textposition='outside',
        textfont=dict(color='#0a0e27', size=11, family='Inter'),
        hovertemplate='<b>%{y}</b><br>Avg Duration: %{x:.2f} hours<extra></extra>'
    ))
    
    fig_duration.update_layout(title=dict(text="Average Booking Duration (Hours)", font=dict(color="#0a0e27", size=14)), plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='white', height=420, margin=dict(l=180, r=120, t=50, b=80))
    fig_duration.update_xaxes(showgrid=True, gridcolor='rgba(0,0,0,0.05)')
    st.plotly_chart(fig_duration, use_container_width=True)
    
    st.markdown("---")
    
    # Revenue per Hour
    st.markdown("### 💵 Revenue per Hour Efficiency (Top 10)")
    venue_rph = venue_detailed[venue_detailed['Revenue per Hour'] > 0].copy()
    venue_rph = venue_rph[venue_rph['Revenue per Hour'] != float('inf')]
    venue_rph = venue_rph.nlargest(10, 'Revenue per Hour')[['Venue', 'Revenue per Hour']].sort_values('Revenue per Hour', ascending=True)
    
    if len(venue_rph) > 0:
        col1, col2, col3 = st.columns([1, 1, 1])
        top_rph_venue = venue_rph.iloc[-1]
        with col2:
            st.markdown(f"""
            <div class='highlight-box'>
                <h4 style='margin-top: 0; color: #ffd700; text-align: center;'>⚡ Most Efficient Venue</h4>
                <p style='font-size: 1.1rem; font-weight: 700; color: #0a0e27; margin: 0.5rem 0; text-align: center;'>{top_rph_venue['Venue']}</p>
                <p style='margin: 0; color: #4b5563; text-align: center; font-size: 1rem;'>₹{top_rph_venue['Revenue per Hour']:,.0f} per hour</p>
            </div>
            """, unsafe_allow_html=True)
        
        fig_rph = go.Figure()
        fig_rph.add_trace(go.Bar(
            y=venue_rph['Venue'],
            x=venue_rph['Revenue per Hour'],
            orientation='h',
            marker=dict(color=venue_rph['Revenue per Hour'], colorscale='YlOrRd', showscale=False, line=dict(color='rgba(255,215,0,0.08)', width=1)),
            text=[f"₹{x:,.0f}" for x in venue_rph['Revenue per Hour']],
            textposition='outside',
            textfont=dict(color='#0a0e27', size=11, family='Inter'),
            hovertemplate='<b>%{y}</b><br>Revenue/Hour: ₹%{x:,.0f}<extra></extra>'
        ))
        fig_rph.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='white', height=420, margin=dict(l=180, r=120, t=50, b=80))
        fig_rph.update_xaxes(title=dict(text='Revenue per Hour (₹)', font=dict(color='#0a0e27', size=12)), showgrid=True, gridcolor='rgba(0,0,0,0.05)')
        st.plotly_chart(fig_rph, use_container_width=True)
    else:
        st.warning("No valid revenue per hour data available for the selected filters.")
    
    st.markdown("---")
    
    # Revenue per Booking
    st.markdown("### 💎 Average Revenue per Booking (Top 10)")
    venue_rpb = venue_detailed.nlargest(10, 'Avg Revenue')[['Venue', 'Avg Revenue']].sort_values('Avg Revenue', ascending=True)
    fig_rpb = go.Figure()
    fig_rpb.add_trace(go.Bar(
        y=venue_rpb['Venue'],
        x=venue_rpb['Avg Revenue'],
        orientation='h',
        marker=dict(color=venue_rpb['Avg Revenue'], colorscale='Reds', showscale=False, line=dict(color='rgba(255,107,107,0.08)', width=1)),
        text=[f"₹{x:,.0f}" for x in venue_rpb['Avg Revenue']],
        textposition='outside',
        textfont=dict(color='#0a0e27', size=11, family='Inter'),
        hovertemplate='<b>%{y}</b><br>Avg per Booking: ₹%{x:,.0f}<extra></extra>'
    ))
    fig_rpb.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='white', height=420, margin=dict(l=180, r=120, t=50, b=80))
    fig_rpb.update_xaxes(title=dict(text='Average Revenue per Booking (₹)', font=dict(color='#0a0e27', size=12)), showgrid=True, gridcolor='rgba(0,0,0,0.05)')
    st.plotly_chart(fig_rpb, use_container_width=True)
    
    st.markdown("---")
    
    # BOOKING PATTERNS & PAYMENT
    st.markdown("## ⏱️ Booking Patterns & Payment Analysis")
    col1, col2 = st.columns([1, 1])
    with col1:
        st.plotly_chart(create_time_heatmap(filtered_df), use_container_width=True)
        st.markdown("#### 📊 Booking Pattern Insights")
        busiest_day = filtered_df['booking_day_of_week'].value_counts().idxmax()
        busiest_day_count = filtered_df['booking_day_of_week'].value_counts().max()
        quietest_day = filtered_df['booking_day_of_week'].value_counts().idxmin()
        quietest_day_count = filtered_df['booking_day_of_week'].value_counts().min()
        peak_hour = filtered_df['booking_hour'].value_counts().idxmax()
        peak_hour_count = filtered_df['booking_hour'].value_counts().max()
        weekend_bookings = filtered_df[filtered_df['booking_day_of_week'].isin(['Saturday', 'Sunday'])].shape[0]
        weekend_pct = (weekend_bookings / len(filtered_df) * 100) if len(filtered_df) > 0 else 0
        morning_bookings = filtered_df[filtered_df['booking_hour'] < 12].shape[0]
        afternoon_bookings = filtered_df[(filtered_df['booking_hour'] >= 12) & (filtered_df['booking_hour'] < 18)].shape[0]
        evening_bookings = filtered_df[filtered_df['booking_hour'] >= 18].shape[0]
        st.markdown(f"""
        <div class='highlight-box'>
            <p style='margin: 0.3rem 0; color: #0a0e27;'><strong>🔥 Busiest Day:</strong> {busiest_day} ({busiest_day_count} bookings)</p>
            <p style='margin: 0.3rem 0; color: #0a0e27;'><strong>😴 Quietest Day:</strong> {quietest_day} ({quietest_day_count} bookings)</p>
            <p style='margin: 0.3rem 0; color: #0a0e27;'><strong>⏰ Peak Hour:</strong> {peak_hour}:00 ({peak_hour_count} bookings)</p>
            <p style='margin: 0.3rem 0; color: #0a0e27;'><strong>📅 Weekend vs Weekday:</strong> {weekend_pct:.1f}% weekend bookings</p>
            <p style='margin: 0.3rem 0; color: #0a0e27;'><strong>🌅 Time Distribution:</strong></p>
            <p style='margin: 0.3rem 0 0.3rem 1.2rem; color: #4b5563;'>Morning: {morning_bookings} bookings</p>
            <p style='margin: 0.3rem 0 0.3rem 1.2rem; color: #4b5563;'>Afternoon: {afternoon_bookings} bookings</p>
            <p style='margin: 0.3rem 0 0.3rem 1.2rem; color: #4b5563;'>Evening: {evening_bookings} bookings</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        payment_fig, payment_data, payment_config = create_payment_distribution(filtered_df)
        st.plotly_chart(payment_fig, use_container_width=True, config=payment_config)
    
    st.markdown("### 💰 Payment Method Breakdown")
    col1, col2 = st.columns([2, 1])
    with col1:
        st.dataframe(
            payment_data.style.format({
                'Revenue': '₹{:,.0f}',
                'Count': '{:,}',
                'Revenue %': '{:.2f}%',
                'Bookings %': '{:.2f}%'
            }).background_gradient(subset=['Revenue'], cmap='Greens'),
            use_container_width=True,
            height=300
        )
    with col2:
        csv = payment_data.to_csv(index=False).encode('utf-8')
        st.download_button(label="📥 Download Payment Data (CSV)", data=csv, file_name=f"payment_methods_{datetime.now().strftime('%Y%m%d')}.csv", mime="text/csv")
        st.markdown(f"""
        <div class='highlight-box'>
            <h4 style='margin-top: 0; color: #00a896;'>📊 Payment Summary</h4>
            <p style='margin: 0.3rem 0; color: #0a0e27;'><strong>Methods:</strong> {len(payment_data)}</p>
            <p style='margin: 0.3rem 0; color: #0a0e27;'><strong>Total Revenue:</strong> ₹{payment_data['Revenue'].sum():,.0f}</p>
            <p style='margin: 0.3rem 0; color: #0a0e27;'><strong>Top Method:</strong> {payment_data.iloc[0]['Payment Method']}</p>
            <p style='margin: 0; color: #4b5563;'>({payment_data.iloc[0]['Revenue %']:.1f}% of revenue)</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### 📅 Monthly Performance Comparison")
    st.plotly_chart(create_monthly_comparison(filtered_df), use_container_width=True)
    
    st.markdown("---")
    st.markdown("### 👥 Customer Behavior & Segmentation")
    col1, col2 = st.columns([2, 1])
    customer_fig, customer_data = create_customer_analysis(filtered_df)
    with col1:
        st.plotly_chart(customer_fig, use_container_width=True)
    with col2:
        st.markdown("#### 🏆 Top Customers")
        top_customers = customer_data.head(10)[['Email', 'Total Bookings', 'Total Spent']]
        st.dataframe(top_customers.style.format({'Total Bookings': '{:,}', 'Total Spent': '₹{:,.0f}'}), use_container_width=True, height=400)
    
    st.markdown("---")
    with st.expander("🔬 Advanced Analytics & Detailed Data", expanded=False):
        st.markdown("#### 📊 Detailed Booking Data")
        display_df = filtered_df[[
            'order_id', 'booking_date', 'venue_name', 'customer_name', 
            'email', 'phone', 'slot_start', 'slot_end', 'slot_duration_hours',
            'actual_revenue', 'payment_method_title', 'order_status'
        ]].copy()
        display_df = display_df.sort_values('booking_date', ascending=False)
        st.dataframe(display_df.style.format({'actual_revenue': '₹{:,.2f}', 'slot_duration_hours': '{:.1f}h'}), use_container_width=True, height=400)
        csv = display_df.to_csv(index=False).encode('utf-8')
        st.download_button(label="📥 Download Data as CSV", data=csv, file_name=f"tida_sports_bookings_{datetime.now().strftime('%Y%m%d')}.csv", mime="text/csv")
    
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #4b5563; padding: 2rem;'>
        <p>Tida Sports Analytics Dashboard | Powered by Streamlit & Plotly</p>
        <p style='font-size: 0.9rem;'>Real-time booking intelligence for data-driven decisions</p>
    </div>
    """, unsafe_allow_html=True)

# -------------------------------------------------------------------
# RUN APPLICATION
# -------------------------------------------------------------------
if __name__ == "__main__":
    main()
