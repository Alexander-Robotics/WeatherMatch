# app.py
# License: AGPL-3.0-or-later

import streamlit as st
import pandas as pd
import numpy as np
import requests
import datetime as dt
import time
import json
from typing import Dict, List, Tuple, Optional
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

import folium
from streamlit_folium import st_folium

# Optional geolocation component
try:
    from streamlit_geolocation import streamlit_geolocation
    GEO_OK = True
except Exception:
    GEO_OK = False

# ---------- Helper Functions (Defined First) ----------
def build_hourly_df(fj: dict) -> pd.DataFrame:
    """Build hourly DataFrame from forecast JSON"""
    if not fj or 'hourly' not in fj:
        return pd.DataFrame()
    
    hourly = fj.get("hourly", {})
    times = pd.to_datetime(hourly.get("time", []))
    df = pd.DataFrame({"time": times})
    for k, v in hourly.items():
        if k != "time":
            df[k] = v
    return df.sort_values("time").reset_index(drop=True)

def wmo_icon(code: int, is_day: bool) -> str:
    """Enhanced weather icons with better mapping"""
    icon_map = {
        0: "☀️" if is_day else "🌙",  # Clear
        1: "🌤️",  # Mainly clear
        2: "⛅",  # Partly cloudy
        3: "☁️",  # Overcast
        45: "🌫️", 48: "🌫️",  # Fog
        51: "🌦️", 53: "🌦️", 55: "🌦️",  # Drizzle
        56: "🌨️", 57: "🌨️",  # Freezing drizzle
        61: "🌧️", 63: "🌧️", 65: "🌧️",  # Rain
        66: "🌨️", 67: "🌨️",  # Freezing rain
        71: "❄️", 73: "❄️", 75: "❄️",  # Snow
        77: "🌨️",  # Snow grains
        80: "🌦️", 81: "🌧️", 82: "🌧️",  # Rain showers
        85: "🌨️", 86: "🌨️",  # Snow showers
        95: "⛈️", 96: "⛈️", 99: "⛈️"  # Thunderstorm
    }
    return icon_map.get(code, "🌥️")

def geocode_city(query: str):
    """Enhanced geocoding with better error handling"""
    if not query or len(query.strip()) < 2:
        return []
    try:
        r = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": query.strip(), "count": 8, "language": "en", "format": "json"},
            timeout=10
        )
        r.raise_for_status()
        results = r.json().get("results", [])
        return sorted(results, key=lambda x: x.get('population', 0), reverse=True)
    except Exception as e:
        st.error(f"Geocoding error: {e}")
        return []

def open_meteo_forecast(lat: float, lon: float, start: dt.date, end: dt.date, tz="auto"):
    """Enhanced forecast with more parameters"""
    try:
        r = requests.get("https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": lat, "longitude": lon, "timezone": tz,
                "hourly": ",".join([
                    "temperature_2m", "relative_humidity_2m", "precipitation", 
                    "cloud_cover", "windspeed_10m", "windgusts_10m", 
                    "uv_index", "visibility", "weather_code", "is_day"
                ]),
                "daily": ["weather_code", "temperature_2m_max", "temperature_2m_min", 
                         "precipitation_sum", "windspeed_10m_max"],
                "start_date": start.strftime("%Y-%m-%d"),
                "end_date": end.strftime("%Y-%m-%d"),
            }, timeout=30)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"Forecast API error: {e}")
        return None

def nasa_power_daily_point(lat: float, lon: float, start_date: str, end_date: str):
    """Fixed NASA POWER API call with proper date formatting and parameters"""
    try:
        # NASA POWER API requires dates in YYYYMMDD format
        start_fmt = start_date.replace("-", "")
        end_fmt = end_date.replace("-", "")
        
        url = "https://power.larc.nasa.gov/api/temporal/daily/point"
        params = {
            "parameters": "T2M,T2M_MAX,T2M_MIN,RH2M,PRECTOTCORR,WS2M",
            "community": "RE", 
            "longitude": lon, 
            "latitude": lat,
            "start": start_fmt,
            "end": end_fmt, 
            "format": "JSON"
        }
        
        # Add timeout and better error handling
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        
        # Check if we got valid data
        if 'properties' in data and 'parameter' in data['properties']:
            return data
        else:
            st.warning("NASA POWER API returned unexpected data format")
            return None
            
    except requests.exceptions.RequestException as e:
        st.error(f"NASA POWER API request failed: {e}")
        return None
    except Exception as e:
        st.error(f"Error processing NASA POWER data: {e}")
        return None

def generate_synthetic_historical_data(lat: float, lon: float, start_date: dt.date, end_date: dt.date, years_back: int = 5):
    """Generate realistic synthetic historical data when NASA POWER is unavailable"""
    # Create date range for historical analysis
    dates = []
    current_year = dt.datetime.now().year
    
    for year_offset in range(1, years_back + 1):
        year = current_year - year_offset
        # Create dates for the same period as requested but in previous years
        current_date = start_date.replace(year=year)
        end_date_historical = end_date.replace(year=year)
        
        while current_date <= end_date_historical:
            dates.append(current_date)
            current_date += dt.timedelta(days=1)
    
    # Generate realistic weather data based on latitude and season
    historical_data = []
    for date in dates:
        # Base temperatures based on latitude and month
        base_temp = 20 - abs(lat) * 0.5 + np.sin((date.month - 6) * np.pi / 6) * 10
        
        # Add some randomness
        temp = base_temp + np.random.normal(0, 5)
        temp_max = temp + np.random.uniform(2, 8)
        temp_min = temp - np.random.uniform(2, 8)
        
        # Humidity varies with temperature
        humidity = max(30, min(95, 80 - (temp - 20) * 2 + np.random.normal(0, 10)))
        
        # Precipitation probability varies by season
        precip_prob = 0.3 + np.sin((date.month - 6) * np.pi / 6) * 0.2
        precipitation = np.random.exponential(0.5) if np.random.random() < precip_prob else 0
        
        # Wind speed
        wind_speed = max(0.5, np.random.exponential(3))
        
        historical_data.append({
            'date': date,
            'Avg Temp (°C)': round(temp, 1),
            'Max Temp (°C)': round(temp_max, 1),
            'Min Temp (°C)': round(temp_min, 1),
            'Humidity (%)': round(humidity, 1),
            'Precip (mm/day)': round(precipitation, 1),
            'Wind Speed (m/s)': round(wind_speed, 1),
            'Conditions Met': True  # Will be calculated later
        })
    
    return pd.DataFrame(historical_data)

def process_historical_data(historical_df: pd.DataFrame, thresholds: Dict) -> Tuple[pd.DataFrame, float]:
    """Process historical data and calculate odds of meeting thresholds"""
    if historical_df.empty:
        return pd.DataFrame(), 0.0
    
    # Calculate if conditions meet thresholds
    conditions_met = pd.Series(True, index=historical_df.index)
    
    # Temperature checks
    if 'Avg Temp (°C)' in historical_df.columns:
        if thresholds.get('temp_c_min') is not None:
            conditions_met &= (historical_df['Avg Temp (°C)'] >= thresholds['temp_c_min'])
        if thresholds.get('temp_c_max') is not None:
            conditions_met &= (historical_df['Avg Temp (°C)'] <= thresholds['temp_c_max'])
    
    # Humidity check
    if 'Humidity (%)' in historical_df.columns and thresholds.get('humidity_pct_max') is not None:
        conditions_met &= (historical_df['Humidity (%)'] <= thresholds['humidity_pct_max'])
    
    # Wind speed check
    if 'Wind Speed (m/s)' in historical_df.columns and thresholds.get('wind_ms_max') is not None:
        conditions_met &= (historical_df['Wind Speed (m/s)'] <= thresholds['wind_ms_max'])
    
    # Precipitation check (convert mm/day to mm/hour equivalent for comparison)
    if 'Precip (mm/day)' in historical_df.columns and thresholds.get('precip_mm_max') is not None:
        # Assume worst-case hourly precipitation is daily total / 6 (for heavy rain periods)
        conditions_met &= (historical_df['Precip (mm/day)'] / 6.0 <= thresholds['precip_mm_max'])
    
    # Update the dataframe
    historical_df = historical_df.copy()
    historical_df['Conditions Met'] = conditions_met
    
    # Calculate historical odds percentage
    historical_odds = (conditions_met.sum() / len(conditions_met)) * 100 if len(conditions_met) > 0 else 0
    
    return historical_df, historical_odds

def calculate_comfort_index(temp: float, humidity: float, wind: float) -> float:
    """Calculate a comfort index from 0-100"""
    # Ideal temperature around 22°C, ideal humidity around 50%
    temp_score = max(0, 100 - abs(temp - 22) * 4)
    humidity_score = max(0, 100 - abs(humidity - 50) * 1.5)
    wind_score = max(0, 100 - wind * 3)  # Lower wind is better
    
    return (temp_score * 0.4 + humidity_score * 0.3 + wind_score * 0.3) / 100

def evaluate_windows(df: pd.DataFrame, thresholds: Dict, min_block_hours: int, preferred_hours: Tuple[int,int]):
    """Enhanced window evaluation with scoring"""
    start_h, end_h = preferred_hours
    hour_ok = df["time"].dt.hour.between(start_h, end_h-1)
    
    # Calculate comfort scores
    df['comfort_score'] = df.apply(
        lambda row: calculate_comfort_index(
            row['temperature_2m'], 
            row['relative_humidity_2m'], 
            row['windspeed_10m']
        ), axis=1
    )
    
    ok = hour_ok.copy()
    t = thresholds
    
    # Apply thresholds
    def in_range(series, vmin, vmax):
        cond = pd.Series(True, index=series.index)
        if vmin is not None: cond &= (series >= vmin)
        if vmax is not None: cond &= (series <= vmax)
        return cond
    
    ok &= in_range(df["temperature_2m"], t.get("temp_c_min"), t.get("temp_c_max"))
    if t.get("humidity_pct_max") is not None: 
        ok &= (df["relative_humidity_2m"] <= t["humidity_pct_max"])
    if t.get("precip_mm_max") is not None: 
        ok &= (df["precipitation"] <= t["precip_mm_max"])
    if t.get("wind_ms_max") is not None: 
        ok &= (df["windspeed_10m"] <= t["wind_ms_max"])
    if t.get("uv_index_max") is not None and "uv_index" in df.columns: 
        ok &= (df["uv_index"] <= t["uv_index_max"])
    if t.get("cloud_cover_max") is not None and "cloud_cover" in df.columns:
        ok &= (df["cloud_cover"] <= t["cloud_cover_max"])

    df = df.copy()
    df["ok"] = ok.values

    windows = []
    grp = (df["ok"] != df["ok"].shift()).cumsum()
    
    for _, seg in df.groupby(grp):
        if not seg["ok"].iloc[0]: 
            continue
        if len(seg) >= min_block_hours:
            avg_comfort = seg['comfort_score'].mean()
            windows.append({
                "start": seg["time"].iloc[0],
                "end": seg["time"].iloc[-1],
                "hours": int(len(seg)),
                "median_temp_c": float(seg["temperature_2m"].median()),
                "max_wind_ms": float(seg["windspeed_10m"].max()),
                "sum_precip_mm": float(seg["precipitation"].sum()),
                "max_uv": float(seg["uv_index"].max()) if "uv_index" in seg.columns else None,
                "avg_comfort": avg_comfort,
                "weather_codes": seg["weather_code"].tolist()
            })
    
    def score(w):
        """Enhanced scoring considering comfort and duration"""
        comfort_weight = 0.4
        duration_weight = 0.4
        precipitation_weight = 0.2
        
        return (
            - (w["avg_comfort"] * comfort_weight + 
               w["hours"] * duration_weight - 
               w["sum_precip_mm"] * precipitation_weight)
        )
    
    windows.sort(key=score)
    return windows[:5]

def create_weather_visualization(df: pd.DataFrame, windows: List[Dict], thresholds: Dict, text_color: str):
    """Create comprehensive weather visualization"""
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=('Temperature & Comfort', 'Precipitation & Wind', 'UV Index & Cloud Cover'),
        vertical_spacing=0.08,
        row_heights=[0.4, 0.3, 0.3]
    )
    
    # Temperature and comfort
    fig.add_trace(
        go.Scatter(x=df['time'], y=df['temperature_2m'], name='Temperature', 
                  line=dict(color='#FF6B6B', width=2)),
        row=1, col=1
    )
    
    # Add comfort score (scaled for visibility)
    comfort_scaled = df.get('comfort_score', 0.5) * 40
    fig.add_trace(
        go.Scatter(x=df['time'], y=comfort_scaled, 
                  name='Comfort Score', line=dict(color='#4ECDC4', width=2, dash='dot'),
                  yaxis='y2'),
        row=1, col=1
    )
    
    # Precipitation and wind
    fig.add_trace(
        go.Bar(x=df['time'], y=df['precipitation'], name='Precipitation', 
               marker_color='#45B7D1', opacity=0.7),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=df['time'], y=df['windspeed_10m'], name='Wind Speed',
                  line=dict(color='#F9C80E', width=2)),
        row=2, col=1
    )
    
    # UV and cloud cover
    if 'uv_index' in df.columns:
        fig.add_trace(
            go.Scatter(x=df['time'], y=df['uv_index'], name='UV Index',
                      line=dict(color='#F96900', width=2)),
            row=3, col=1
        )
    
    if 'cloud_cover' in df.columns:
        fig.add_trace(
            go.Scatter(x=df['time'], y=df['cloud_cover'], name='Cloud Cover',
                      line=dict(color='#96CEB4', width=2, dash='dot')),
            row=3, col=1
        )
    
    # Highlight best windows
    for i, window in enumerate(windows):
        color = ['#00C851', '#007E33', '#FF8800', '#CC0000', '#9933CC'][i % 5]
        fig.add_vrect(
            x0=window['start'], x1=window['end'],
            fillcolor=color, opacity=0.2, line_width=0,
            row="all", col=1
        )
    
    fig.update_layout(
        height=700,
        showlegend=True,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color=text_color),
        margin=dict(l=50, r=50, t=100, b=80)
    )
    
       # Remove y-axis titles to prevent overlapping
    fig.update_yaxes(title_text="", row=1, col=1)
    fig.update_yaxes(title_text="", row=2, col=1)
    fig.update_yaxes(title_text="", row=3, col=1)
    return fig

# ---------- Theme Configuration ----------
def apply_custom_theme(is_dark: bool):
    PRIMARY = "#3B82F6"
    SECONDARY = "#10B981"
    BG = "#0B1220" if is_dark else "#FFFFFF"
    PANEL = "#111827" if is_dark else "#F8FAFC"
    TEXT = "#E5E7EB" if is_dark else "#0F172A"
    MUTED = "#9CA3AF" if is_dark else "#475569"
    ACCENT = "#22C55E"
    WARN = "#F59E0B"
    ERROR = "#EF4444"
    BORDER = "#1F2937" if is_dark else "#E2E8F0"
    
    st.markdown(f"""
    <style>
    .stApp {{ 
        background: {BG}; 
        color: {TEXT};
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }}
    
    /* Header Styling */
    .main-header {{
        background: linear-gradient(135deg, {PRIMARY}, {SECONDARY});
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        text-align: center;
    }}
    
    .tagline {{
        color: {MUTED};
        font-size: 1.1rem;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 400;
    }}
    
    /* Panel Styling */
    .panel {{
        background: {PANEL};
        border: 1px solid {BORDER};
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }}
    
    .card {{
        background: {PANEL};
        border: 1px solid {BORDER};
        border-radius: 12px;
        padding: 1rem;
        margin: 0.5rem 0;
        transition: all 0.3s ease;
    }}
    
    .card:hover {{
        border-color: {PRIMARY};
        transform: translateY(-2px);
        box-shadow: 0 8px 15px -3px rgba(0, 0, 0, 0.1);
    }}
    
    /* Chip Styling */
    .chip {{
        display: inline-flex;
        align-items: center;
        padding: 0.35rem 0.75rem;
        border-radius: 20px;
        border: 1px solid {BORDER};
        margin: 0.25rem 0.35rem 0.25rem 0;
        color: {TEXT};
        font-size: 0.85rem;
        font-weight: 500;
    }}
    
    .good {{ border-color: {ACCENT}; color: {ACCENT}; background: rgba(34, 197, 94, 0.1); }}
    .warn {{ border-color: {WARN}; color: {WARN}; background: rgba(245, 158, 11, 0.1); }}
    .bad {{ border-color: {ERROR}; color: {ERROR}; background: rgba(239, 68, 68, 0.1); }}
    
    /* KPI Styling */
    .kpi-container {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
        gap: 0.75rem;
        margin: 1rem 0;
    }}
    
    .kpi {{
        background: {PANEL};
        border: 1px solid {BORDER};
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
    }}
    
    .kpi .value {{
        font-weight: 700;
        color: {TEXT};
        font-size: 1.4rem;
        margin-bottom: 0.25rem;
    }}
    
    .kpi .label {{
        color: {MUTED};
        font-size: 0.8rem;
        font-weight: 500;
    }}
    
    /* Button enhancements */
    .stButton button {{
        border-radius: 10px;
        font-weight: 600;
        transition: all 0.3s ease;
    }}
    
    .stButton button:hover {{
        transform: translateY(-1px);
    }}
    </style>
    """, unsafe_allow_html=True)
    
    return {
        'PRIMARY': PRIMARY,
        'SECONDARY': SECONDARY,
        'ACCENT': ACCENT,
        'WARN': WARN,
        'ERROR': ERROR,
        'TEXT': TEXT
    }

# ---------- Activity Presets ----------
ACTIVITY_PRESETS = {
    "Beach day": {
        "temp_c_min": 20, "temp_c_max": 32, "wind_ms_max": 7, "precip_mm_max": 0.2, 
        "uv_index_max": 8, "humidity_pct_max": 75, "cloud_cover_max": 30,
        "description": "Warm, sunny, light breeze", "icon": "🏖️"
    },
    "Hiking": {
        "temp_c_min": 10, "temp_c_max": 28, "wind_ms_max": 9, "precip_mm_max": 0.5, 
        "uv_index_max": 9, "humidity_pct_max": 85, "cloud_cover_max": 70,
        "description": "Comfortable, light rain only", "icon": "🥾"
    },
    "Running": {
        "temp_c_min": 5, "temp_c_max": 24, "wind_ms_max": 8, "precip_mm_max": 0.3, 
        "uv_index_max": 7, "humidity_pct_max": 80, "cloud_cover_max": 60,
        "description": "Cool to mild, low rain", "icon": "🏃"
    },
    "Cycling": {
        "temp_c_min": 8, "temp_c_max": 28, "wind_ms_max": 10, "precip_mm_max": 0.3, 
        "uv_index_max": 9, "humidity_pct_max": 80, "cloud_cover_max": 50,
        "description": "Mild, manageable wind", "icon": "🚴"
    },
    "Picnic": {
        "temp_c_min": 15, "temp_c_max": 30, "wind_ms_max": 7, "precip_mm_max": 0.2, 
        "uv_index_max": 8, "humidity_pct_max": 75, "cloud_cover_max": 40,
        "description": "Pleasant, minimal rain", "icon": "🧺"
    },
    "Photography": {
        "temp_c_min": 0, "temp_c_max": 35, "wind_ms_max": 6, "precip_mm_max": 0.1, 
        "uv_index_max": 6, "humidity_pct_max": 70, "cloud_cover_max": 80,
        "description": "Clear visibility, stable conditions", "icon": "📸"
    },
    "Custom": {"icon": "⚙️"}
}

# ---------- Streamlit App Configuration ----------
st.set_page_config(page_title="Weather Probability Dashboard", page_icon="⛅", layout="wide")

# Initialize theme
theme_type = getattr(getattr(st, "context", None), "theme", {}).get("type", "light")
is_dark = st.sidebar.toggle("Dark mode", value=(theme_type == "dark"))
colors = apply_custom_theme(is_dark)
TEXT_COLOR = colors['TEXT']

# Header
st.markdown('<div class="main-header">🌤️ Weather Probability Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="tagline">Plan your outdoor activities with confidence using historical weather data and AI-powered predictions</div>', unsafe_allow_html=True)

# Initialize session state
if 'lat' not in st.session_state:
    st.session_state.lat = 40.7128
if 'lon' not in st.session_state:
    st.session_state.lon = -74.0060
if 'historical_data' not in st.session_state:
    st.session_state.historical_data = None
if 'historical_odds' not in st.session_state:
    st.session_state.historical_odds = 0.0

# ---------- Sidebar Configuration ----------
with st.sidebar:
    st.markdown("### 🎯 Activity & Location")
    
    # Activity selection with icons
    activity_options = list(ACTIVITY_PRESETS.keys())
    activity = st.selectbox(
        "Select Activity", 
        activity_options,
        format_func=lambda x: f"{ACTIVITY_PRESETS[x]['icon']} {x}",
        index=0
    )
    
    if activity != "Custom":
        st.info(f"**{ACTIVITY_PRESETS[activity]['description']}**")
    
    st.markdown("---")
    st.markdown("### 📍 Location Settings")
    
    # Location input methods
    location_method = st.radio("Input Method", ["Search City", "Map Pick", "Coordinates"], horizontal=True)
    
    if location_method == "Search City":
        city_query = st.text_input("City Name", placeholder="e.g., Paris, Tokyo, New York")
        if st.button("Search Location", use_container_width=True):
            with st.spinner("Searching..."):
                results = geocode_city(city_query)
                if results:
                    st.session_state.lat = float(results[0]["latitude"])
                    st.session_state.lon = float(results[0]["longitude"])
                    st.success(f"Found: {results[0]['name']}")
                else:
                    st.error("No results found")
    
    elif location_method == "Coordinates":
        col1, col2 = st.columns(2)
        with col1:
            lat = st.number_input("Latitude", value=st.session_state.lat, format="%.6f")
        with col2:
            lon = st.number_input("Longitude", value=st.session_state.lon, format="%.6f")
        st.session_state.lat = lat
        st.session_state.lon = lon
    
    st.markdown("---")
    st.markdown("### 📅 Date & Time")
    
    today = dt.date.today()
    start_date = st.date_input("Start Date", today)
    end_date = st.date_input("End Date", today + dt.timedelta(days=7))
    
    col1, col2 = st.columns(2)
    with col1:
        pref_start = st.time_input("Preferred Start", dt.time(9, 0))
    with col2:
        pref_end = st.time_input("Preferred End", dt.time(17, 0))
    
    st.markdown("---")
    st.markdown("### ⚙️ Advanced Settings")
    
    with st.expander("Weather Thresholds"):
        if activity == "Custom" or st.checkbox("Customize Thresholds"):
            col1, col2 = st.columns(2)
            with col1:
                temp_min = st.number_input("Min Temp (°C)", value=10)
                humidity_max = st.number_input("Max Humidity (%)", value=80)
                cloud_max = st.number_input("Max Cloud Cover (%)", value=60)
            with col2:
                temp_max = st.number_input("Max Temp (°C)", value=28)
                wind_max = st.number_input("Max Wind (m/s)", value=8)
                precip_max = st.number_input("Max Rain (mm/h)", value=0.5)
            
            thresholds = {
                "temp_c_min": temp_min, "temp_c_max": temp_max,
                "humidity_pct_max": humidity_max, "wind_ms_max": wind_max,
                "precip_mm_max": precip_max, "cloud_cover_max": cloud_max
            }
        else:
            thresholds = ACTIVITY_PRESETS[activity]
    
    min_block_hours = st.slider("Minimum Continuous Hours", 1, 8, 3)
    
    st.markdown("---")
    compute = st.button("🎯 Find Best Weather Windows", type="primary", use_container_width=True)

# ---------- Main Dashboard Layout ----------
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown("### 🗺️ Interactive Map")
    
    # Create interactive map
    m = folium.Map(
        location=[st.session_state.lat, st.session_state.lon], 
        zoom_start=10,
        tiles='OpenStreetMap'
    )
    
    # Add marker
    folium.Marker(
        [st.session_state.lat, st.session_state.lon],
        popup=f"Selected Location\n{st.session_state.lat:.4f}, {st.session_state.lon:.4f}",
        tooltip="Drag me!",
        draggable=True
    ).add_to(m)
    
    # Display map and get interactions
    map_data = st_folium(m, height=400, width=None)
    
    # Update location if marker was dragged
    if map_data and map_data.get('last_clicked'):
        st.session_state.lat = map_data['last_clicked']['lat']
        st.session_state.lon = map_data['last_clicked']['lng']
        st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown("### 📊 Quick Stats")
    
    if compute:
        with st.spinner("Fetching weather data..."):
            try:
                forecast_data = open_meteo_forecast(
                    st.session_state.lat, st.session_state.lon, start_date, end_date
                )
                
                if forecast_data:
                    df = build_hourly_df(forecast_data)
                    
                    if not df.empty:
                        # Calculate some quick stats
                        avg_temp = df['temperature_2m'].mean()
                        max_temp = df['temperature_2m'].max()
                        total_precip = df['precipitation'].sum()
                        avg_wind = df['windspeed_10m'].mean()
                        
                        st.markdown(f"""
                        <div class="kpi-container">
                            <div class="kpi">
                                <div class="value">{avg_temp:.1f}°</div>
                                <div class="label">Avg Temp</div>
                            </div>
                            <div class="kpi">
                                <div class="value">{total_precip:.1f}mm</div>
                                <div class="label">Total Rain</div>
                            </div>
                            <div class="kpi">
                                <div class="value">{avg_wind:.1f}m/s</div>
                                <div class="label">Avg Wind</div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Weather summary
                        st.markdown("**Current Conditions:**")
                        current = df.iloc[0] if len(df) > 0 else None
                        if current is not None:
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("Temperature", f"{current['temperature_2m']:.1f}°C")
                                st.metric("Humidity", f"{current['relative_humidity_2m']:.0f}%")
                            with col2:
                                st.metric("Wind Speed", f"{current['windspeed_10m']:.1f} m/s")
                                st.metric("Precipitation", f"{current['precipitation']:.1f} mm")
                    else:
                        st.warning("No forecast data available")
            except Exception as e:
                st.error(f"Error fetching data: {e}")
    else:
        st.info("Click 'Find Best Weather Windows' to see statistics")
    
    st.markdown('</div>', unsafe_allow_html=True)

# ---------- Main Results Section ----------
if compute:
    with st.spinner("Analyzing weather patterns..."):
        try:
            # Get forecast data
            forecast_data = open_meteo_forecast(
                st.session_state.lat, st.session_state.lon, start_date, end_date
            )
            
            if forecast_data:
                df = build_hourly_df(forecast_data)
                
                if not df.empty:
                    windows = evaluate_windows(
                        df, thresholds, min_block_hours, 
                        (pref_start.hour, pref_end.hour)
                    )
                    
                    # Display results
                    st.markdown('<div class="panel">', unsafe_allow_html=True)
                    st.markdown("### 🎯 Best Weather Windows")
                    
                    if not windows:
                        st.warning("No suitable weather windows found. Try adjusting your criteria.")
                    else:
                        for i, window in enumerate(windows, 1):
                            badge_class = "good" if window['hours'] >= 4 else "warn" if window['hours'] >= 2 else "bad"
                            
                            with st.container():
                                st.markdown(f"""
                                <div class="card">
                                    <div style="display: flex; justify-content: space-between; align-items: center;">
                                        <div>
                                            <span class="chip {badge_class}">#{i} • {window['hours']} hours</span>
                                            <strong>{window['start'].strftime('%a, %b %d %H:%M')} - {window['end'].strftime('%H:%M')}</strong>
                                        </div>
                                        <div style="text-align: right;">
                                            <small>Comfort: {window['avg_comfort']:.0%}</small>
                                        </div>
                                    </div>
                                    <div class="kpi-container">
                                        <div class="kpi">
                                            <div class="value">{window['median_temp_c']:.1f}°</div>
                                            <div class="label">Temperature</div>
                                        </div>
                                        <div class="kpi">
                                            <div class="value">{window['max_wind_ms']:.1f}</div>
                                            <div class="label">Max Wind</div>
                                        </div>
                                        <div class="kpi">
                                            <div class="value">{window['sum_precip_mm']:.1f}mm</div>
                                            <div class="label">Total Rain</div>
                                        </div>
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)
                        
                        # Visualization
                        st.markdown("### 📈 Weather Overview")
                        fig = create_weather_visualization(df, windows, thresholds, TEXT_COLOR)
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.error("No forecast data available for the selected location and dates")
                
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.error("Failed to fetch forecast data")
            
        except Exception as e:
            st.error(f"Error in analysis: {e}")

    # ---------- Historical Analysis Section ----------
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown("### 📊 Historical Probability Analysis")
    
    with st.spinner("Fetching historical data..."):
        try:
             # Try to get real NASA POWER data first - only same dates from previous years
            current_year = dt.datetime.now().year
            years_back = 5
            
            # Create list of specific same dates from previous years
            specific_dates = []
            for year_offset in range(1, years_back + 1):
                year = current_year - year_offset
                # Add each date in the current range for previous years
                current_dt = start_date
                while current_dt <= end_date:
                    historical_date = current_dt.replace(year=year)
                    specific_dates.append(historical_date)
                    current_dt += dt.timedelta(days=1)
            
            # Group dates by year to minimize API calls
            nasa_data = None
            if specific_dates:
                # Get unique years to make separate API calls
                years = sorted(set(date.year for date in specific_dates))
                all_records = []
                
                for year in years:
                    year_dates = [d for d in specific_dates if d.year == year]
                    if year_dates:
                        year_start = min(year_dates).strftime('%Y%m%d')
                        year_end = max(year_dates).strftime('%Y%m%d')
                        
                        year_data = nasa_power_daily_point(
                            st.session_state.lat, st.session_state.lon, 
                            year_start, year_end
                        )
                        
                        if year_data:
                            # Extract data for this year and filter to only our specific dates
                            parameters = year_data.get('properties', {}).get('parameter', {})
                            dates_in_year = list(parameters.get('T2M', {}).keys()) if 'T2M' in parameters else []
                            
                            for date_str in dates_in_year:
                                date_obj = pd.to_datetime(date_str)
                                if date_obj in specific_dates:
                                    record = {'date': date_obj}
                                    for param, values in parameters.items():
                                        if date_str in values:
                                            record[param] = values[date_str]
                                    all_records.append(record)
                
                if all_records:
                    nasa_data = {'properties': {'parameter': {}}}
                    # Reconstruct NASA data format from our filtered records
                    for record in all_records:
                        date_str = record['date'].strftime('%Y%m%d')
                        for param, value in record.items():
                            if param != 'date':
                                if param not in nasa_data['properties']['parameter']:
                                    nasa_data['properties']['parameter'][param] = {}
                                nasa_data['properties']['parameter'][param][date_str] = value
                
            if nasa_data and nasa_data.get('properties', {}).get('parameter'):
                # Process real NASA data
                st.success("✅ Using real NASA POWER historical data")
                historical_df = pd.DataFrame()
                
                # Extract data from NASA response
                parameters = nasa_data.get('properties', {}).get('parameter', {})
                dates = list(parameters.get('T2M', {}).keys()) if 'T2M' in parameters else []
                
                records = []
                for date_str in dates:
                    record = {'date': pd.to_datetime(date_str)}
                    for param, values in parameters.items():
                        if date_str in values:
                            record[param] = values[date_str]
                    records.append(record)
                
                historical_df = pd.DataFrame(records)
                
                # Rename columns for display
                column_mapping = {
                    'T2M': 'Avg Temp (°C)',
                    'T2M_MAX': 'Max Temp (°C)',
                    'T2M_MIN': 'Min Temp (°C)',
                    'RH2M': 'Humidity (%)',
                    'PRECTOTCORR': 'Precip (mm/day)',
                    'WS2M': 'Wind Speed (m/s)'
                }
                historical_df = historical_df.rename(columns=column_mapping)
                
            else:
                # Fall back to synthetic data
                st.info("📊 Using synthetic historical data (NASA POWER unavailable)")
                historical_df = generate_synthetic_historical_data(
                    st.session_state.lat, st.session_state.lon, 
                    start_date, end_date, years_back=5
                )
            
            if not historical_df.empty:
                # Process and calculate odds
                historical_df, historical_odds = process_historical_data(historical_df, thresholds)
                
                # Store in session state
                st.session_state.historical_data = historical_df
                st.session_state.historical_odds = historical_odds
                
                # Display historical odds
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    odds_color = colors['ACCENT'] if historical_odds >= 60 else colors['WARN'] if historical_odds >= 40 else colors['ERROR']
                    muted_color = "#9CA3AF" if is_dark else "#475569"
                    st.markdown(f"""
                    <div style="text-align: center; padding: 2rem;">
                        <h1 style="color: {odds_color}; font-size: 3rem; margin: 0;">{historical_odds:.1f}%</h1>
                        <p style="color: {muted_color}; font-size: 1.1rem;">
                            Historical probability of meeting your criteria
                        </p>
                        <p style="color: {muted_color}; font-size: 0.9rem;">
                            Based on {len(historical_df)} days of historical data
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Display historical data table
                st.markdown("### 📋 Historical Weather Data")
                
                # Show summary statistics
                st.markdown("#### Summary Statistics")
                summary_cols = st.columns(4)
                
                numeric_columns = historical_df.select_dtypes(include=[np.number]).columns
                
                if 'Avg Temp (°C)' in numeric_columns:
                    with summary_cols[0]:
                        avg_temp = historical_df['Avg Temp (°C)'].mean()
                        st.metric("Average Temperature", f"{avg_temp:.1f}°C")
                
                if 'Humidity (%)' in numeric_columns:
                    with summary_cols[1]:
                        avg_humidity = historical_df['Humidity (%)'].mean()
                        st.metric("Average Humidity", f"{avg_humidity:.1f}%")
                
                if 'Wind Speed (m/s)' in numeric_columns:
                    with summary_cols[2]:
                        avg_wind = historical_df['Wind Speed (m/s)'].mean()
                        st.metric("Average Wind", f"{avg_wind:.1f} m/s")
                
                if 'Precip (mm/day)' in numeric_columns:
                    with summary_cols[3]:
                        avg_precip = historical_df['Precip (mm/day)'].mean()
                        st.metric("Avg Precipitation", f"{avg_precip:.1f} mm/day")
                
                # Display the data table
                st.markdown("#### Detailed Historical Data")
                
                # Format the dataframe for display
                display_df = historical_df.copy()
                display_df['date'] = pd.to_datetime(display_df['date']).dt.strftime('%Y-%m-%d')
                
                # Reorder columns to put Conditions Met first
                if 'Conditions Met' in display_df.columns:
                    cols = ['date', 'Conditions Met'] + [col for col in display_df.columns if col not in ['date', 'Conditions Met']]
                    display_df = display_df[cols]
                
                # Display the table
                st.dataframe(
                    display_df,
                    use_container_width=True,
                    height=400
                )
                
                # Add download button for historical data
                csv_data = historical_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Historical Data (CSV)",
                    data=csv_data,
                    file_name=f"historical_weather_{st.session_state.lat:.4f}_{st.session_state.lon:.4f}.csv",
                    mime="text/csv"
                )
                
                # Add insights
                st.markdown("#### 📈 Historical Insights")
                
                if historical_odds >= 70:
                    st.success(f"**High Probability Area:** There's a {historical_odds:.1f}% chance of favorable conditions based on historical data. This location and time period have consistently good weather for your activity.")
                elif historical_odds >= 40:
                    st.warning(f"**Moderate Probability Area:** There's a {historical_odds:.1f}% chance of favorable conditions. Consider having backup plans as weather can be variable.")
                else:
                    st.error(f"**Low Probability Area:** Only {historical_odds:.1f}% chance of favorable conditions historically. You might want to consider alternative locations or dates.")
                    
            else:
                st.warning("No historical data available for analysis.")
                
        except Exception as e:
            st.error(f"Error in historical analysis: {str(e)}")
            st.info("This could be due to NASA POWER API limitations or network issues. The system will use synthetic data as a fallback.")
    
    st.markdown('</div>', unsafe_allow_html=True)

# ---------- Footer ----------
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #6B7280; font-size: 0.9rem;">
        <p>Powered by Open-Meteo, NASA POWER, and OpenStreetMap • Built with Streamlit</p>
        <p>Data sources provide free weather and climate information for research and personal use</p>
    </div>
    """, 
    unsafe_allow_html=True
)
