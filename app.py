"""AHPS Web Demo - Streamlit app for Accidents Hotspot Prediction System"""
import os
import sys
import pickle
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests
import streamlit as st
import folium
from streamlit_folium import st_folium
import plotly.graph_objects as go
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, '.')
from src.predict import load_model, load_thresholds, add_derived_features, predict_hotspot
from src.preprocessing import _group_weather
from src.osm_lookup import get_infra_features, INFRA_FEATURES

# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="AHPS - Accidents Hotspot Prediction System",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ==================== CUSTOM CSS ====================
st.markdown("""
<style>
.main-header {
    background: linear-gradient(135deg, rgba(255,75,75,0.25), rgba(255,75,75,0.08));
    padding: 24px 32px;
    border-radius: 14px;
    border: 1px solid rgba(255,75,75,0.3);
    margin-bottom: 24px;
    box-shadow: 0 4px 16px rgba(0,0,0,0.25);
    text-align: center;
}
.main-header h1 {
    color: #FFFFFF;
    margin: 0;
    font-size: 2rem;
    font-weight: 800;
    letter-spacing: 1.5px;
    text-transform: uppercase;
}
.main-header p {
    color: rgba(255,255,255,0.75);
    margin: 8px 0 0 0;
    font-size: 1rem;
}
.section-title {
    background: linear-gradient(90deg, rgba(255,75,75,0.20), rgba(255,75,75,0.04));
    border-left: 4px solid #FF4B4B;
    padding: 14px 22px;
    border-radius: 8px 8px 0 0;
    margin: 22px 0 0 0;
    font-size: 1.1rem;
    font-weight: 800;
    color: #FFFFFF;
    letter-spacing: 1px;
    text-transform: uppercase;
    box-shadow: 0 2px 8px rgba(0,0,0,0.15);
}
.section-body {
    background: rgba(20,20,25,0.55);
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
    border: 1px solid rgba(255,255,255,0.08);
    border-top: none;
    border-radius: 0 0 10px 10px;
    padding: 18px 24px;
    margin-bottom: 18px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.2);
}
.info-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 10px 0;
    border-bottom: 1px solid rgba(255,255,255,0.06);
}
.info-row:last-child { border-bottom: none; }
.info-label { color: rgba(255,255,255,0.7); font-size: 0.95rem; }
.info-value { color: #FFFFFF; font-weight: 600; font-size: 1rem; text-align: right; }

/* Coord box */
.coord-box {
    background: linear-gradient(135deg, rgba(255,75,75,0.12), rgba(20,20,25,0.65));
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
    border: 1px solid rgba(255,75,75,0.25);
    border-radius: 10px;
    padding: 14px 22px;
    min-height: 52px;
    display: flex;
    align-items: center;
    gap: 18px;
    color: #FFFFFF;
    box-shadow: 0 2px 8px rgba(0,0,0,0.25);
}
.coord-items { display: flex; flex: 1; align-items: center; gap: 18px; }
.coord-item { display: flex; flex-direction: column; line-height: 1.2; }
.coord-key {
    color: rgba(255,255,255,0.55);
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
.coord-val {
    color: #FFFFFF;
    font-weight: 700;
    font-size: 1.05rem;
    font-family: 'Courier New', monospace;
}
.coord-sep { width: 1px; height: 28px; background: rgba(255,255,255,0.15); }
.coord-empty {
    color: rgba(255,255,255,0.6);
    font-style: italic;
    justify-content: center;
}

div[data-testid="column"] { display: flex; flex-direction: column; }
.stButton > button {
    border-radius: 10px;
    border: 1px solid rgba(255,255,255,0.15);
    min-height: 52px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    transition: all 0.2s ease;
}
.stButton > button:hover { transform: translateY(-1px); box-shadow: 0 4px 14px rgba(0,0,0,0.3); }
[data-testid="stAlert"] {
    background: rgba(20,20,25,0.55) !important;
    backdrop-filter: blur(10px);
    border-radius: 10px;
    border: 1px solid rgba(255,255,255,0.1);
}
[data-testid="stExpander"] {
    background: rgba(20,20,25,0.4);
    border-radius: 10px;
    border: 1px solid rgba(255,255,255,0.08);
}
.stCaption, [data-testid="stCaptionContainer"] { color: rgba(255,255,255,0.6) !important; }
</style>
""", unsafe_allow_html=True)

# ==================== CONSTANTS ====================
MODELS_DIR = "models"

STATE_NAME_TO_CODE = {
    "Alabama":"AL","Alaska":"AK","Arizona":"AZ","Arkansas":"AR","California":"CA",
    "Colorado":"CO","Connecticut":"CT","Delaware":"DE","Florida":"FL","Georgia":"GA",
    "Hawaii":"HI","Idaho":"ID","Illinois":"IL","Indiana":"IN","Iowa":"IA",
    "Kansas":"KS","Kentucky":"KY","Louisiana":"LA","Maine":"ME","Maryland":"MD",
    "Massachusetts":"MA","Michigan":"MI","Minnesota":"MN","Mississippi":"MS","Missouri":"MO",
    "Montana":"MT","Nebraska":"NE","Nevada":"NV","New Hampshire":"NH","New Jersey":"NJ",
    "New Mexico":"NM","New York":"NY","North Carolina":"NC","North Dakota":"ND","Ohio":"OH",
    "Oklahoma":"OK","Oregon":"OR","Pennsylvania":"PA","Rhode Island":"RI","South Carolina":"SC",
    "South Dakota":"SD","Tennessee":"TN","Texas":"TX","Utah":"UT","Vermont":"VT",
    "Virginia":"VA","Washington":"WA","West Virginia":"WV","Wisconsin":"WI","Wyoming":"WY",
    "District of Columbia":"DC",
}

WEATHER_GROUPS = ["Clear","Cloudy","Partly Cloudy","Rain","Snow/Ice","Fog/Haze","Storm","Wind/Dust","Other"]
WEATHER_VN = {
    "Clear":"Trời quang","Cloudy":"Nhiều mây","Partly Cloudy":"Mây rải rác",
    "Rain":"Mưa","Snow/Ice":"Tuyết / Băng","Fog/Haze":"Sương mù",
    "Storm":"Bão","Wind/Dust":"Gió / Bụi","Other":"Khác",
}
WIND_DIRS_VN = {
    "N":"Bắc","NE":"Đông Bắc","E":"Đông","SE":"Đông Nam",
    "S":"Nam","SW":"Tây Nam","W":"Tây","NW":"Tây Bắc","CALM":"Lặng gió",
}
RAIN_INTENSITY_LABELS = {0:"Không mưa",1:"Mưa nhẹ",2:"Mưa vừa",3:"Mưa to"}

DEFAULTS = {
    "Temperature(F)":65.0,"Humidity(%)":70.0,"Pressure(in)":29.8,
    "Visibility(mi)":10.0,"Wind_Speed(mph)":8.0,"WindDir":"N",
}

# ==================== HELPERS ====================
def section_title(text):
    st.markdown(f'<div class="section-title">{text}</div>', unsafe_allow_html=True)

def info_block(rows):
    html = '<div class="section-body">'
    for label, value in rows:
        html += f'<div class="info-row"><span class="info-label">{label}</span><span class="info-value">{value}</span></div>'
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)

def render_tabs(labels, active_index, key_prefix="tabbtn"):
    cols = st.columns(len(labels))
    for i, (col, label) in enumerate(zip(cols, labels)):
        is_active = (i == active_index)
        btn_type = "primary" if is_active else "secondary"
        with col:
            if st.button(label.upper(), key=f"{key_prefix}_{i}", type=btn_type, use_container_width=True):
                st.session_state.active_tab = i
                st.rerun()

@st.cache_resource
def load_artifacts():
    tuned_path = os.path.join(MODELS_DIR, "model_xgb_tuned.pkl")
    base_path = os.path.join(MODELS_DIR, "model_xgboost.pkl")
    thresh_path = os.path.join(MODELS_DIR, "best_thresholds.pkl")
    enc_path = os.path.join(MODELS_DIR, "encoders.pkl")

    if os.path.exists(tuned_path):
        model_path = tuned_path
        model_name = "xgb_tuned"
    elif os.path.exists(base_path):
        model_path = base_path
        model_name = "xgboost"
    else:
        raise FileNotFoundError("Không tìm thấy model trong thư mục models/")

    model = load_model(model_path)
    thresholds = load_thresholds(thresh_path) if os.path.exists(thresh_path) else {}
    threshold = thresholds.get(model_name, 0.5)
    with open(enc_path, "rb") as f:
        encoders = pickle.load(f)
    return model, threshold, encoders, model_name, model_path

def encode_safe(encoder, value, default="Other"):
    try:
        if hasattr(encoder, "classes_") and value in encoder.classes_:
            return int(encoder.transform([value])[0])
        if hasattr(encoder, "classes_") and default in encoder.classes_:
            return int(encoder.transform([default])[0])
        return 0
    except Exception:
        return 0

def get_encoder(encoders, *candidates):
    for name in candidates:
        if name in encoders:
            return encoders[name]
    return None

def encode_with_fallback(encoders, candidates, value, default="Other"):
    enc = get_encoder(encoders, *candidates)
    return encode_safe(enc, value, default) if enc is not None else 0

def deg_to_compass(deg):
    if deg is None:
        return "N"
    dirs = ["N","NE","E","SE","S","SW","W","NW"]
    ix = int((deg + 22.5) // 45) % 8
    return dirs[ix]

def rain_mm_to_intensity(mm):
    if mm is None or mm <= 0:
        return 0
    if mm < 2.5:
        return 1
    if mm < 7.6:
        return 2
    return 3

def hour_to_is_night(hour):
    return 1 if (hour < 6 or hour >= 19) else 0

def hour_to_is_rush(hour):
    return 1 if (7 <= hour <= 9) or (16 <= hour <= 19) else 0

def get_state_from_coords(lat, lng):
    try:
        url = "https://nominatim.openstreetmap.org/reverse"
        params = {"lat": lat, "lon": lng, "format": "json", "addressdetails": 1}
        headers = {"User-Agent": "AHPS-Demo/1.0"}
        r = requests.get(url, params=params, headers=headers, timeout=8)
        if r.status_code == 200:
            data = r.json()
            addr = data.get("address", {})
            country_code = addr.get("country_code", "")
            if country_code.lower() != "us":
                return None
            state_name = addr.get("state", "")
            return STATE_NAME_TO_CODE.get(state_name)
    except Exception:
        return None
    return None

def fetch_openweather(lat, lng, api_key):
    if not api_key:
        return None
    try:
        url = "https://api.openweathermap.org/data/2.5/weather"
        params = {"lat": lat, "lon": lng, "appid": api_key, "units": "imperial"}
        r = requests.get(url, params=params, timeout=8)
        if r.status_code == 200:
            return r.json()
    except Exception:
        return None
    return None

def get_local_datetime(lat, lng, utc_offset_sec=0):
    return datetime.utcnow() + timedelta(seconds=utc_offset_sec)

def get_alert_vn(level):
    return {
        "HIGH RISK": "⚠️ NGUY HIỂM CAO",
        "CAUTION": "⚡ CẦN THẬN TRỌNG",
        "SAFE": "✅ AN TOÀN",
    }.get(level, level)

def get_alert_color(level):
    return {"HIGH RISK":"red","CAUTION":"orange","SAFE":"green"}.get(level, "blue")

def make_gauge(probability):
    pct = float(probability) * 100
    if pct >= 70:
        color = "#FF4B4B"
    elif pct >= 40:
        color = "#FFA500"
    else:
        color = "#4CAF50"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=pct,
        number={"suffix": "%", "font": {"size": 36, "color": "white"}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "white"},
            "bar": {"color": color, "thickness": 0.3},
            "bgcolor": "rgba(0,0,0,0)",
            "steps": [
                {"range": [0, 40], "color": "rgba(76,175,80,0.25)"},
                {"range": [40, 70], "color": "rgba(255,165,0,0.25)"},
                {"range": [70, 100], "color": "rgba(255,75,75,0.25)"},
            ],
        },
    ))
    fig.update_layout(
        height=280, margin=dict(l=20, r=20, t=20, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": "white"},
    )
    return fig

def fetch_all_location_info(lat, lng):
    info = {"lat": lat, "lng": lng}

    state = get_state_from_coords(lat, lng)
    if state is None:
        return None, "Vị trí không thuộc Hoa Kỳ. Vui lòng chọn vị trí trong nước Mỹ."
    info["state"] = state

    api_key = ""
    try:
        api_key = st.secrets.get("OPENWEATHER_API_KEY", "")
    except Exception:
        api_key = os.getenv("OPENWEATHER_API_KEY", "")

    weather = fetch_openweather(lat, lng, api_key)
    if weather:
        info["temperature"] = weather["main"].get("temp", DEFAULTS["Temperature(F)"])
        info["humidity"] = weather["main"].get("humidity", DEFAULTS["Humidity(%)"])
        pressure_hpa = weather["main"].get("pressure", 1013.25)
        info["pressure"] = pressure_hpa * 0.02953
        info["visibility"] = weather.get("visibility", 16093) / 1609.34
        info["wind_speed"] = weather.get("wind", {}).get("speed", DEFAULTS["Wind_Speed(mph)"])
        info["wind_deg"] = weather.get("wind", {}).get("deg", 0)
        info["wind_dir"] = deg_to_compass(info["wind_deg"])
        weather_main = weather["weather"][0].get("main", "Clear") if weather.get("weather") else "Clear"
        weather_desc = weather["weather"][0].get("description", "") if weather.get("weather") else ""
        info["weather_main"] = weather_main
        info["weather_desc"] = weather_desc
        info["weather_group"] = _group_weather(weather_main)
        rain_mm = weather.get("rain", {}).get("1h", 0)
        info["rain_intensity"] = rain_mm_to_intensity(rain_mm)
        info["utc_offset"] = weather.get("timezone", 0)
    else:
        info.update({
            "temperature": DEFAULTS["Temperature(F)"],
            "humidity": DEFAULTS["Humidity(%)"],
            "pressure": DEFAULTS["Pressure(in)"],
            "visibility": DEFAULTS["Visibility(mi)"],
            "wind_speed": DEFAULTS["Wind_Speed(mph)"],
            "wind_deg": 0,
            "wind_dir": DEFAULTS["WindDir"],
            "weather_main": "Clear",
            "weather_desc": "Trời quang",
            "weather_group": "Clear",
            "rain_intensity": 0,
            "utc_offset": 0,
        })

    info["local_dt"] = get_local_datetime(lat, lng, info["utc_offset"])

    try:
        infra_lookup_dir = os.path.join(MODELS_DIR, "infra_lookup")
        if os.path.exists(infra_lookup_dir):
            infra = get_infra_features(lat, lng, infra_lookup_dir)
        else:
            infra = {k: 0 for k in INFRA_FEATURES}
    except Exception:
        infra = {k: 0 for k in INFRA_FEATURES}
    info["infra"] = infra

    return info, None

def run_prediction(model, threshold, encoders, info):
    dt = info["local_dt"]
    hour = dt.hour
    month = dt.month
    is_raining = 1 if info["rain_intensity"] > 0 else 0
    is_rush = hour_to_is_rush(hour)
    is_night = hour_to_is_night(hour)

    # Build dict đúng theo FEATURE_COLS
    point = {
        # Thời gian
        "hour": hour,
        "month": month,
        "day_of_week": dt.weekday(),
        "is_rush_hour": is_rush,
        "is_night": is_night,
        # Khí tượng
        "Temperature(F)": info["temperature"],
        "Humidity(%)": info["humidity"],
        "Pressure(in)": info["pressure"],
        "Visibility(mi)": info["visibility"],
        "Wind_Speed(mph)": info["wind_speed"],
        "is_raining": is_raining,
        # Cơ sở hạ tầng
        "Amenity": info["infra"].get("Amenity", 0),
        "Crossing": info["infra"].get("Crossing", 0),
        "Give_Way": info["infra"].get("Give_Way", 0),
        "Junction": info["infra"].get("Junction", 0),
        "No_Exit": info["infra"].get("No_Exit", 0),
        "Railway": info["infra"].get("Railway", 0),
        "Station": info["infra"].get("Station", 0),
        "Stop": info["infra"].get("Stop", 0),
        "Traffic_Signal": info["infra"].get("Traffic_Signal", 0),
        # Encoders
        "Weather_enc": encode_with_fallback(encoders, ["weather"], info["weather_group"]),
        "WindDir_enc": encode_with_fallback(encoders, ["wind"], info["wind_dir"]),
        "State_enc": encode_with_fallback(encoders, ["state"], info["state"]),
        "County_enc": encode_with_fallback(encoders, ["county"], "Other"),
        # Cường độ mưa
        "rain_intensity": info["rain_intensity"],
    }

    # add_derived_features sẽ tự tính hour_sin, hour_cos, month_sin, month_cos,
    # rush_rain, night_vis, rain_wind
    point = add_derived_features(point)
    result_dict = predict_hotspot(model, point, threshold)

    return {
        "probability": float(result_dict["probability"]),
        "prediction": int(result_dict["prediction"]),
        "alert_level": str(result_dict["alert_level"]),
        "point": point,
        "info": info,
    }

# ==================== LOAD MODEL ====================
try:
    model, threshold, encoders, model_name, model_path = load_artifacts()
except Exception as e:
    st.error(f"Lỗi tải model: {e}")
    st.stop()

# ==================== SESSION STATE ====================
defaults_state = {
    "selected_lat": None,
    "selected_lng": None,
    "location_confirmed": False,
    "location_info": None,
    "last_result": None,
    "active_tab": 0,
}
for k, v in defaults_state.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ==================== HEADER ====================
st.markdown("""
<div class="main-header">
    <h1>AHPS - Hệ thống dự đoán điểm nóng tai nạn</h1>
    <p>Accidents Hotspot Prediction System | CS313 Project</p>
</div>
""", unsafe_allow_html=True)

# ==================== TABS ====================
TAB_LABELS = ["1. Chọn vị trí", "2. Thông tin chi tiết", "3. Kết quả dự đoán"]
render_tabs(TAB_LABELS, st.session_state.active_tab)
st.markdown("")

# -------------------- TAB 1: CHỌN VỊ TRÍ --------------------
if st.session_state.active_tab == 0:
    section_title("Chọn vị trí trên bản đồ Hoa Kỳ")
    st.markdown('<div class="section-body">', unsafe_allow_html=True)
    st.write("Nhấp chuột vào bản đồ để chọn vị trí cần dự đoán.")
    m = folium.Map(location=[39.5, -98.35], zoom_start=4, tiles="OpenStreetMap")
    if st.session_state.selected_lat is not None:
        folium.Marker(
            [st.session_state.selected_lat, st.session_state.selected_lng],
            tooltip="Vị trí đã chọn",
            icon=folium.Icon(color="red", icon="map-marker"),
        ).add_to(m)
    map_data = st_folium(m, height=450, width=None, key="select_map")
    st.markdown('</div>', unsafe_allow_html=True)

    if map_data and map_data.get("last_clicked"):
        st.session_state.selected_lat = map_data["last_clicked"]["lat"]
        st.session_state.selected_lng = map_data["last_clicked"]["lng"]

    section_title("Xác nhận vị trí")
    col_coord, col_btn = st.columns(2)
    with col_coord:
        if st.session_state.selected_lat is not None:
            coord_html = (
                '<div class="coord-box">'
                '<div class="coord-items">'
                f'<div class="coord-item"><span class="coord-key">Vĩ độ</span><span class="coord-val">{st.session_state.selected_lat:.4f}</span></div>'
                '<div class="coord-sep"></div>'
                f'<div class="coord-item"><span class="coord-key">Kinh độ</span><span class="coord-val">{st.session_state.selected_lng:.4f}</span></div>'
                '</div>'
                '</div>'
            )
        else:
            coord_html = '<div class="coord-box coord-empty">Chưa chọn vị trí. Hãy nhấp chuột lên bản đồ.</div>'
        st.markdown(coord_html, unsafe_allow_html=True)
    with col_btn:
        if st.button("Xác nhận vị trí", type="primary", use_container_width=True, key="confirm_btn",
                     disabled=(st.session_state.selected_lat is None)):
            with st.spinner("Đang lấy thông tin vị trí..."):
                info, err = fetch_all_location_info(
                    st.session_state.selected_lat, st.session_state.selected_lng
                )
                if err:
                    st.error(err)
                else:
                    st.session_state.location_info = info
                    st.session_state.location_confirmed = True
                    st.session_state.active_tab = 1
                    st.rerun()

# -------------------- TAB 2: THÔNG TIN CHI TIẾT --------------------
elif st.session_state.active_tab == 1:
    if not st.session_state.location_confirmed or st.session_state.location_info is None:
        section_title("Thông tin chi tiết")
        st.warning("Vui lòng chọn và xác nhận vị trí ở tab 1 trước.")
        if st.button("Về tab 1", key="back_to_1"):
            st.session_state.active_tab = 0
            st.rerun()
    else:
        info = st.session_state.location_info

        section_title("Vị trí và thời gian")
        info_block([
            ("Vĩ độ (Lat)", f"{info['lat']:.4f}"),
            ("Kinh độ (Lng)", f"{info['lng']:.4f}"),
            ("Bang", info["state"]),
            ("Thời gian địa phương", info["local_dt"].strftime("%H:%M %d/%m/%Y")),
        ])

        section_title("Tình trạng thời tiết")
        info_block([
            ("Mô tả", info.get("weather_desc", "N/A")),
            ("Nhóm thời tiết", WEATHER_VN.get(info["weather_group"], info["weather_group"])),
            ("Cường độ mưa", RAIN_INTENSITY_LABELS[info["rain_intensity"]]),
        ])

        section_title("Thông số khí tượng chi tiết")
        info_block([
            ("Nhiệt độ", f"{info['temperature']:.1f} °F"),
            ("Độ ẩm", f"{info['humidity']:.0f} %"),
            ("Áp suất", f"{info['pressure']:.2f} inHg"),
            ("Tầm nhìn", f"{info['visibility']:.1f} mi"),
            ("Tốc độ gió", f"{info['wind_speed']:.1f} mph"),
            ("Hướng gió", WIND_DIRS_VN.get(info["wind_dir"], info["wind_dir"])),
        ])

        section_title("Cơ sở hạ tầng xung quanh (OpenStreetMap)")
        infra_vn = {
            "Amenity":"Tiện ích công cộng","Crossing":"Vạch sang đường",
            "Give_Way":"Biển nhường đường","Junction":"Giao lộ",
            "No_Exit":"Đường cụt","Railway":"Đường sắt",
            "Station":"Trạm/Ga","Stop":"Biển Stop","Traffic_Signal":"Đèn giao thông",
        }
        infra_rows = []
        for k, v in info["infra"].items():
            infra_rows.append((infra_vn.get(k, k), "Có" if v == 1 else "—"))
        info_block(infra_rows)

        st.markdown("---")
        col_back, col_predict = st.columns(2)
        with col_back:
            if st.button("Quay lại chọn vị trí", use_container_width=True, key="back_btn"):
                st.session_state.active_tab = 0
                st.rerun()
        with col_predict:
            if st.button("Dự đoán điểm nóng", type="primary", use_container_width=True, key="predict_btn"):
                with st.spinner("Đang dự đoán..."):
                    try:
                        result = run_prediction(model, threshold, encoders, info)
                        st.session_state.last_result = result
                        st.session_state.active_tab = 2
                        st.rerun()
                    except Exception as e:
                        st.error(f"Lỗi khi dự đoán: {e}")
                        import traceback
                        st.code(traceback.format_exc())

# -------------------- TAB 3: KẾT QUẢ DỰ ĐOÁN --------------------
elif st.session_state.active_tab == 2:
    result = st.session_state.last_result
    if result is None:
        section_title("Kết quả dự đoán")
        st.warning("Chưa có kết quả. Vui lòng hoàn tất tab 1 và 2 trước.")
        if st.button("Về tab 1", key="to_tab1"):
            st.session_state.active_tab = 0
            st.rerun()
    else:
        info = result["info"]
        alert_vn = get_alert_vn(result["alert_level"])

        section_title("Kết quả dự đoán điểm nóng")
        col_gauge, col_info = st.columns(2)
        with col_gauge:
            st.plotly_chart(make_gauge(result["probability"]), use_container_width=True)
        with col_info:
            if result["alert_level"] == "HIGH RISK":
                st.error(f"### {alert_vn}")
            elif result["alert_level"] == "CAUTION":
                st.warning(f"### {alert_vn}")
            else:
                st.success(f"### {alert_vn}")
            info_block([
                ("Xác suất điểm nóng", f"{result['probability']:.2%}"),
                ("Phân loại", "ĐIỂM NÓNG" if result["prediction"] == 1 else "AN TOÀN"),
                ("Vị trí", f"({info['lat']:.4f}, {info['lng']:.4f})"),
                ("Bang", info["state"]),
                ("Thời gian", info["local_dt"].strftime('%H:%M %d/%m/%Y')),
                ("Thời tiết", WEATHER_VN.get(info["weather_group"], info["weather_group"])),
            ])

        section_title("Vị trí dự đoán trên bản đồ")
        st.markdown('<div class="section-body">', unsafe_allow_html=True)
        result_map = folium.Map(location=[info["lat"], info["lng"]], zoom_start=12)
        folium.CircleMarker(
            [info["lat"], info["lng"]], radius=20,
            color=get_alert_color(result["alert_level"]),
            fill=True, fillOpacity=0.6,
            popup=f"{alert_vn} - {result['probability']:.1%}",
        ).add_to(result_map)
        folium.Marker(
            [info["lat"], info["lng"]], tooltip="Vị trí dự đoán",
            icon=folium.Icon(color="blue", icon="info-sign"),
        ).add_to(result_map)
        st_folium(result_map, height=400, width=None)
        st.markdown('</div>', unsafe_allow_html=True)

        with st.expander("Xem chi tiết đặc trưng đã dùng để dự đoán"):
            feat_df = pd.DataFrame([result["point"]]).T.reset_index()
            feat_df.columns = ["Feature", "Value"]
            st.dataframe(feat_df, use_container_width=True, height=400)

        st.markdown("---")
        if st.button("Dự đoán vị trí mới", use_container_width=True, key="new_predict"):
            st.session_state.location_confirmed = False
            st.session_state.location_info = None
            st.session_state.last_result = None
            st.session_state.active_tab = 0
            st.rerun()

# ==================== FOOTER ====================
st.markdown("---")
st.caption("AHPS - Accidents Hotspot Prediction System | Đồ án CS313 | Powered by Streamlit + XGBoost + OpenStreetMap + OpenWeatherMap")
