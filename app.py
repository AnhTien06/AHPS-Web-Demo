"""AHPS Web Demo - Streamlit app for Accidents Hotspot Prediction System"""
import os
import pickle
import sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests
import streamlit as st
import streamlit_antd_components as sac
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
)

# ==================== CUSTOM CSS ====================
st.markdown("""
<style>
/* ===== Tiêu đề section: HOA + đậm, nền đỏ gradient + viền trái ===== */
.section-title {
    background: linear-gradient(90deg, rgba(255, 75, 75, 0.18), rgba(255, 75, 75, 0.04));
    border-left: 4px solid #FF4B4B;
    padding: 14px 22px;
    border-radius: 8px;
    margin-bottom: 10px;
    margin-top: 24px;
    font-size: 1.2rem;
    font-weight: 800;
    color: #FFFFFF;
    letter-spacing: 1px;
    text-transform: uppercase;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.15);
}

/* ===== Tab labels: HOA + đậm ===== */
.ant-tabs-tab,
.ant-tabs-tab-btn,
[class*="ant-tabs-tab"] {
    text-transform: uppercase !important;
    font-weight: 700 !important;
    letter-spacing: 0.5px !important;
}

/* ===== Tiêu đề chính trang ===== */
h1 {
    background: linear-gradient(135deg, rgba(255, 75, 75, 0.12), rgba(255, 75, 75, 0.02));
    backdrop-filter: blur(12px);
    padding: 20px 28px !important;
    border-radius: 14px;
    border: 1px solid rgba(255, 75, 75, 0.25);
    box-shadow: 0 6px 20px rgba(0, 0, 0, 0.25);
    margin-bottom: 20px !important;
}

/* ===== Metric card ===== */
[data-testid="stMetric"] {
    background: rgba(255, 255, 255, 0.05);
    padding: 14px 18px;
    border-radius: 10px;
    border: 1px solid rgba(255, 255, 255, 0.08);
    box-shadow: 0 2px 6px rgba(0, 0, 0, 0.12);
}

/* ===== Alert boxes ===== */
[data-testid="stAlert"] {
    background: rgba(255, 255, 255, 0.05) !important;
    backdrop-filter: blur(8px);
    border-radius: 10px;
    border: 1px solid rgba(255, 255, 255, 0.1);
}

/* ===== Expander ===== */
[data-testid="stExpander"] {
    background: rgba(255, 255, 255, 0.04);
    backdrop-filter: blur(6px);
    border-radius: 10px;
    border: 1px solid rgba(255, 255, 255, 0.08);
}

/* ===== Caption ===== */
.stCaption, [data-testid="stCaptionContainer"] {
    color: rgba(255, 255, 255, 0.6) !important;
}

/* ===== Button ===== */
.stButton > button {
    border-radius: 8px;
    border: 1px solid rgba(255, 255, 255, 0.15);
    transition: all 0.2s ease;
    font-weight: 500;
}
.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
}
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
WEATHER_VN = {"Clear":"Trời quang","Cloudy":"Nhiều mây","Partly Cloudy":"Có mây",
              "Rain":"Mưa","Snow/Ice":"Tuyết/Băng","Fog/Haze":"Sương mù",
              "Storm":"Bão","Wind/Dust":"Gió/Bụi","Other":"Khác"}

WIND_DIRS_VN = {"N":"Bắc","NE":"Đông Bắc","E":"Đông","SE":"Đông Nam",
                "S":"Nam","SW":"Tây Nam","W":"Tây","NW":"Tây Bắc"}

RAIN_INTENSITY_LABELS = {0:"Không mưa", 1:"Mưa nhẹ", 2:"Mưa vừa", 3:"Mưa to"}

DEFAULTS = {"Temperature(F)":65.0, "Humidity(%)":70.0, "Pressure(in)":29.8,
            "Visibility(mi)":10.0, "Wind_Speed(mph)":8.0, "WindDir":"N"}

# ==================== HELPER: HTML SECTION ====================
def section_title(text):
    """Tiêu đề section: HOA + đậm, nền gradient đỏ + viền trái."""
    st.markdown(f'<div class="section-title">{text}</div>', unsafe_allow_html=True)

# ==================== MODEL LOADING ====================
@st.cache_resource
def load_artifacts():
    tuned_path = os.path.join(MODELS_DIR, "model_xgb_tuned.pkl")
    base_path = os.path.join(MODELS_DIR, "model_xgboost.pkl")
    thresh_path = os.path.join(MODELS_DIR, "best_thresholds.pkl")
    enc_path = os.path.join(MODELS_DIR, "encoders.pkl")

    if os.path.exists(tuned_path):
        model_path, model_name = tuned_path, "XGBoost (Tuned)"
    elif os.path.exists(base_path):
        model_path, model_name = base_path, "XGBoost (Base)"
    else:
        return None, None, None, None, None

    if not os.path.exists(enc_path):
        return None, None, None, None, None

    model = load_model(model_path)
    thresholds = load_thresholds(thresh_path) if os.path.exists(thresh_path) else {}
    threshold = thresholds.get("XGBoost", 0.5)
    with open(enc_path, "rb") as f:
        encoders = pickle.load(f)
    return model, threshold, encoders, model_name, model_path

# ==================== HELPER FUNCTIONS ====================
def encode_safe(encoder, value, default="Other"):
    try:
        if value in encoder.classes_:
            return int(encoder.transform([value])[0])
        if default in encoder.classes_:
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
    if enc is None:
        return 0
    return encode_safe(enc, value, default)

def deg_to_compass(deg):
    dirs = ["N","NE","E","SE","S","SW","W","NW"]
    return dirs[round(deg/45) % 8]

def rain_mm_to_intensity(rain_1h_mm):
    rain_in = rain_1h_mm / 25.4
    if rain_in == 0: return 0
    if rain_in <= 0.10: return 1
    if rain_in <= 0.30: return 2
    return 3

def hour_to_is_night(hour):
    return hour < 6 or hour >= 18

@st.cache_data(ttl=86400)
def get_state_from_coords(lat, lng):
    try:
        url = "https://nominatim.openstreetmap.org/reverse"
        params = {"lat": lat, "lon": lng, "format": "json", "zoom": 5, "addressdetails": 1}
        headers = {"User-Agent": "AHPS-Demo/1.0"}
        resp = requests.get(url, params=params, headers=headers, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        if data.get("address", {}).get("country_code") != "us":
            return None
        state_name = data.get("address", {}).get("state")
        return STATE_NAME_TO_CODE.get(state_name)
    except Exception:
        return None

def fetch_openweather(lat, lng, api_key):
    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {"lat": lat, "lon": lng, "appid": api_key, "units": "imperial"}
    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    return resp.json()

def get_local_datetime(lat, lng, timezone_offset_sec):
    utc_now = datetime.utcnow()
    return utc_now + timedelta(seconds=timezone_offset_sec)

def get_alert_vn(alert_level):
    return {"HIGH RISK":"NGUY HIỂM CAO","CAUTION":"CẢNH BÁO","SAFE":"AN TOÀN"}.get(alert_level, alert_level)

def get_alert_color(alert_level):
    return {"HIGH RISK":"red","CAUTION":"orange","SAFE":"green"}.get(alert_level, "blue")

def make_gauge(probability):
    pct = float(probability) * 100
    if pct >= 70: bar_color = "#E74C3C"
    elif pct >= 40: bar_color = "#F39C12"
    else: bar_color = "#27AE60"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=pct,
        number={"suffix":"%","font":{"size":56,"color":"white","family":"Arial Black"},"valueformat":".1f"},
        domain={"x":[0,1],"y":[0,1]},
        gauge={"axis":{"range":[0,100],"tickwidth":1,"tickcolor":"lightgray",
                       "tickfont":{"size":14,"color":"lightgray"},
                       "tickmode":"array","tickvals":[0,20,40,60,80,100]},
               "bar":{"color":bar_color,"thickness":0.7},
               "bgcolor":"rgba(0,0,0,0)","borderwidth":2,
               "bordercolor":"rgba(255,255,255,0.2)",
               "steps":[{"range":[0,40],"color":"rgba(39,174,96,0.25)"},
                        {"range":[40,70],"color":"rgba(243,156,18,0.25)"},
                        {"range":[70,100],"color":"rgba(231,76,60,0.25)"}]}))
    fig.update_layout(height=320, margin=dict(l=30,r=30,t=40,b=20),
                      paper_bgcolor="rgba(0,0,0,0)", font={"color":"white","family":"Arial"})
    return fig

# ==================== AUTO-FETCH LOCATION INFO ====================
def fetch_all_location_info(lat, lng):
    api_key = st.secrets.get("OPENWEATHER_API_KEY", os.getenv("OPENWEATHER_API_KEY", "")).strip()
    info = {"lat": lat, "lng": lng}
    info["state"] = get_state_from_coords(lat, lng) or "CA"

    if api_key:
        try:
            data = fetch_openweather(lat, lng, api_key)
            info["temperature"] = data["main"]["temp"]
            info["humidity"] = data["main"]["humidity"]
            info["pressure"] = data["main"]["pressure"] * 0.02953
            info["visibility"] = data.get("visibility", 16093) / 1609.34
            info["wind_speed"] = data["wind"]["speed"]
            info["wind_dir"] = deg_to_compass(data["wind"].get("deg", 0))
            info["weather_desc"] = data["weather"][0]["description"]
            wg = _group_weather(data["weather"][0]["description"])
            info["weather_group"] = wg if wg in WEATHER_GROUPS else "Other"
            rain_mm = data.get("rain", {}).get("1h", 0.0)
            info["rain_intensity"] = rain_mm_to_intensity(rain_mm)
            info["local_dt"] = get_local_datetime(lat, lng, data.get("timezone", 0))
            info["weather_ok"] = True
        except Exception as e:
            info["weather_ok"] = False
            info["weather_error"] = str(e)
    else:
        info["weather_ok"] = False
        info["weather_error"] = "Chưa cấu hình API key OpenWeatherMap"

    if not info.get("weather_ok"):
        info["temperature"] = DEFAULTS["Temperature(F)"]
        info["humidity"] = DEFAULTS["Humidity(%)"]
        info["pressure"] = DEFAULTS["Pressure(in)"]
        info["visibility"] = DEFAULTS["Visibility(mi)"]
        info["wind_speed"] = DEFAULTS["Wind_Speed(mph)"]
        info["wind_dir"] = DEFAULTS["WindDir"]
        info["weather_desc"] = "N/A"
        info["weather_group"] = "Clear"
        info["rain_intensity"] = 0
        info["local_dt"] = datetime.now()

    try:
        infra_db = os.path.join(MODELS_DIR, "infra_lookup")
        info["infra"] = get_infra_features(lat, lng, db_path=infra_db)
    except Exception:
        info["infra"] = {f: 0 for f in INFRA_FEATURES}

    return info

# ==================== PREDICTION ====================
def run_prediction(model, threshold, encoders, info):
    dt = info["local_dt"]
    point = {
        "Start_Lat": info["lat"], "Start_Lng": info["lng"],
        "Temperature(F)": info["temperature"], "Humidity(%)": info["humidity"],
        "Pressure(in)": info["pressure"], "Visibility(mi)": info["visibility"],
        "Wind_Speed(mph)": info["wind_speed"],
        "hour": dt.hour, "month": dt.month, "day_of_week": dt.weekday(),
        "is_rush_hour": int(dt.hour in [7,8,9,16,17,18]),
        "is_night": int(hour_to_is_night(dt.hour)),
        "is_raining": int(info["rain_intensity"] > 0),
        "rain_intensity": info["rain_intensity"],
        "Weather_enc": encode_with_fallback(
            encoders, ["weather", "Weather", "Weather_Condition"],
            info["weather_group"]),
        "WindDir_enc": encode_with_fallback(
            encoders, ["wind", "WindDir", "Wind_Direction"],
            info["wind_dir"]),
        "State_enc": encode_with_fallback(
            encoders, ["state", "State"], info["state"]),
        "County_enc": encode_with_fallback(
            encoders, ["county", "County"], "Other"),
    }
    for f in INFRA_FEATURES:
        point[f] = info["infra"].get(f, 0)
    point = add_derived_features(point)
    result = predict_hotspot(model, point, threshold)

    if isinstance(result, dict):
        prob = result.get("probability", result.get("prob", 0))
        pred = result.get("prediction", result.get("pred", 0))
        alert = result.get("alert_level", result.get("alert", "SAFE"))
    elif isinstance(result, (tuple, list)) and len(result) >= 3:
        prob, pred, alert = result[0], result[1], result[2]
    else:
        prob, pred, alert = 0.0, 0, "SAFE"

    try:
        prob = float(prob)
    except Exception:
        prob = 0.0
    try:
        pred = int(pred)
    except Exception:
        pred = 0
    alert = str(alert)

    return {"probability": prob, "prediction": pred, "alert_level": alert,
            "point": point, "info": info}

# ==================== LOAD MODEL ====================
model, threshold, encoders, model_name, model_path = load_artifacts()
if model is None:
    st.error("Không tìm thấy file model. Vui lòng chạy training trước.")
    st.stop()

# ==================== SESSION STATE ====================
defaults_state = {
    "clicked_lat": 34.0522,
    "clicked_lng": -118.2437,
    "location_confirmed": False,
    "location_info": None,
    "last_result": None,
    "active_tab": 0,
}
for k, v in defaults_state.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ==================== HEADER ====================
st.title("AHPS - Hệ thống dự đoán điểm nóng tai nạn giao thông")
st.caption("Đồ án CS313 | Dataset: US Accidents (Kaggle) | Model: XGBoost")

# ==================== TABS ====================
current_tab = sac.tabs(
    items=[
        sac.TabsItem(label="1. Chọn vị trí"),
        sac.TabsItem(label="2. Thông tin chi tiết"),
        sac.TabsItem(label="3. Kết quả dự đoán"),
    ],
    index=st.session_state.active_tab,
    align="center",
    size="lg",
    color="red",
    use_container_width=True,
    return_index=True,
    key="tab_control",
)

if current_tab != st.session_state.active_tab:
    st.session_state.active_tab = current_tab

# -------------------- TAB 1: CHỌN VỊ TRÍ --------------------
if st.session_state.active_tab == 0:
    section_title("Chọn vị trí cần dự đoán trên bản đồ nước Mỹ")
    st.markdown("Nhấp vào vị trí bất kỳ trên bản đồ, sau đó bấm **Xác nhận vị trí** để tải toàn bộ thông tin.")

    m = folium.Map(location=[39.8283, -98.5795], zoom_start=4)
    folium.Marker(
        [st.session_state.clicked_lat, st.session_state.clicked_lng],
        tooltip="Vị trí đã chọn",
        icon=folium.Icon(color="blue", icon="info-sign"),
    ).add_to(m)
    map_data = st_folium(m, height=500, width=None, returned_objects=["last_clicked"])

    if map_data and map_data.get("last_clicked"):
        new_lat = map_data["last_clicked"]["lat"]
        new_lng = map_data["last_clicked"]["lng"]
        if (new_lat, new_lng) != (st.session_state.clicked_lat, st.session_state.clicked_lng):
            st.session_state.clicked_lat = new_lat
            st.session_state.clicked_lng = new_lng
            st.session_state.location_confirmed = False
            st.session_state.location_info = None
            st.rerun()

    col_info, col_btn = st.columns(2)
    with col_info:
        st.info(f"**Tọa độ đã chọn:** `{st.session_state.clicked_lat:.4f}, {st.session_state.clicked_lng:.4f}`")
    with col_btn:
        if st.button("Xác nhận vị trí", type="primary", use_container_width=True):
            with st.spinner("Đang tải thông tin từ vị trí..."):
                info = fetch_all_location_info(
                    st.session_state.clicked_lat,
                    st.session_state.clicked_lng,
                )
                st.session_state.location_info = info
                st.session_state.location_confirmed = True
                st.session_state.active_tab = 1
            st.rerun()

# -------------------- TAB 2: THÔNG TIN CHI TIẾT --------------------
elif st.session_state.active_tab == 1:
    if not st.session_state.location_confirmed or st.session_state.location_info is None:
        section_title("Thông tin chi tiết")
        st.warning("Vui lòng quay lại tab '1. Chọn vị trí' và bấm **Xác nhận vị trí** trước.")
        if st.button("Quay lại tab 1"):
            st.session_state.active_tab = 0
            st.rerun()
    else:
        info = st.session_state.location_info
        if not info.get("weather_ok"):
            st.warning(f"API thời tiết lỗi: {info.get('weather_error', 'unknown')}. Đang dùng giá trị mặc định.")

        section_title("Vị trí và thời gian")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Vĩ độ (Lat)", f"{info['lat']:.4f}")
        c2.metric("Kinh độ (Lng)", f"{info['lng']:.4f}")
        c3.metric("Bang", info["state"])
        c4.metric("Thời gian địa phương", info["local_dt"].strftime("%H:%M %d/%m/%Y"))

        section_title("Tình trạng thời tiết")
        c1, c2, c3 = st.columns(3)
        c1.metric("Mô tả", info.get("weather_desc", "N/A"))
        c2.metric("Nhóm thời tiết", WEATHER_VN.get(info["weather_group"], info["weather_group"]))
        c3.metric("Cường độ mưa", RAIN_INTENSITY_LABELS[info["rain_intensity"]])

        section_title("Thông số khí tượng chi tiết")
        c1, c2, c3 = st.columns(3)
        c1.metric("Nhiệt độ", f"{info['temperature']:.1f} °F")
        c2.metric("Độ ẩm", f"{info['humidity']:.0f} %")
        c3.metric("Áp suất", f"{info['pressure']:.2f} inHg")
        c1, c2, c3 = st.columns(3)
        c1.metric("Tầm nhìn", f"{info['visibility']:.1f} mi")
        c2.metric("Tốc độ gió", f"{info['wind_speed']:.1f} mph")
        c3.metric("Hướng gió", WIND_DIRS_VN.get(info["wind_dir"], info["wind_dir"]))

        section_title("Cơ sở hạ tầng xung quanh (OpenStreetMap)")
        infra_vn = {
            "Amenity":"Tiện ích công cộng","Crossing":"Vạch sang đường",
            "Give_Way":"Biển nhường đường","Junction":"Giao lộ",
            "No_Exit":"Đường cụt","Railway":"Đường sắt",
            "Station":"Trạm/Ga","Stop":"Biển Stop","Traffic_Signal":"Đèn giao thông",
        }
        infra_present = [infra_vn.get(k, k) for k, v in info["infra"].items() if v == 1]
        if infra_present:
            cols = st.columns(3)
            for i, item in enumerate(infra_present):
                cols[i % 3].success(f"Có {item}")
        else:
            st.info("Không phát hiện cơ sở hạ tầng đặc biệt trong khu vực.")

        st.markdown("---")
        col_back, col_predict = st.columns(2)
        with col_back:
            if st.button("Quay lại chọn vị trí", use_container_width=True):
                st.session_state.active_tab = 0
                st.rerun()
        with col_predict:
            if st.button("DỰ ĐOÁN ĐIỂM NÓNG", type="primary", use_container_width=True):
                with st.spinner("Đang dự đoán..."):
                    result = run_prediction(model, threshold, encoders, info)
                    st.session_state.last_result = result
                    st.session_state.active_tab = 2
                st.rerun()

# -------------------- TAB 3: KẾT QUẢ DỰ ĐOÁN --------------------
elif st.session_state.active_tab == 2:
    result = st.session_state.last_result
    if result is None:
        section_title("Kết quả dự đoán")
        st.warning("Chưa có kết quả. Vui lòng hoàn tất tab 1 và 2 trước.")
        if st.button("Về tab 1"):
            st.session_state.active_tab = 0
            st.rerun()
    else:
        info = result["info"]

        section_title("Kết quả dự đoán điểm nóng")
        col_gauge, col_info = st.columns(2)
        with col_gauge:
            st.plotly_chart(make_gauge(result["probability"]), use_container_width=True)
        with col_info:
            alert_vn = get_alert_vn(result["alert_level"])
            if result["alert_level"] == "HIGH RISK":
                st.error(f"### {alert_vn}")
            elif result["alert_level"] == "CAUTION":
                st.warning(f"### {alert_vn}")
            else:
                st.success(f"### {alert_vn}")
            c1, c2 = st.columns(2)
            c1.metric("Xác suất điểm nóng", f"{result['probability']:.2%}")
            c2.metric("Phân loại", "ĐIỂM NÓNG" if result["prediction"] == 1 else "AN TOÀN")
            st.markdown(f"""
            **Vị trí:** ({info['lat']:.4f}, {info['lng']:.4f})  
            **Bang:** {info['state']}  
            **Thời gian:** {info['local_dt'].strftime('%H:%M %d/%m/%Y')}  
            **Thời tiết:** {WEATHER_VN.get(info['weather_group'], info['weather_group'])}
            """)

        section_title("Vị trí dự đoán trên bản đồ")
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

        with st.expander("Xem chi tiết đặc trưng đã dùng để dự đoán"):
            feat_df = pd.DataFrame([result["point"]]).T.reset_index()
            feat_df.columns = ["Feature", "Value"]
            st.dataframe(feat_df, use_container_width=True, height=400)

        st.markdown("---")
        if st.button("Dự đoán vị trí mới", use_container_width=True):
            st.session_state.location_confirmed = False
            st.session_state.location_info = None
            st.session_state.last_result = None
            st.session_state.active_tab = 0
            st.rerun()

# ==================== FOOTER ====================
st.markdown("---")
st.caption("AHPS - Accidents Hotspot Prediction System | Đồ án CS313 | Powered by Streamlit + XGBoost + OpenStreetMap + OpenWeatherMap")
