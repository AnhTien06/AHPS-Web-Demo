"""
AHPS Web Demo - Streamlit app for Accidents Hotspot Prediction System
Run: streamlit run app.py
"""
import os
import pickle
import sys
from datetime import datetime, time as dtime

from dotenv import load_dotenv
load_dotenv()

import numpy as np
import pandas as pd
import requests
import streamlit as st
import folium
from streamlit_folium import st_folium
import plotly.graph_objects as go

sys.path.insert(0, '.')
from src.predict import load_model, load_thresholds, add_derived_features, predict_hotspot
from src.preprocessing import _group_weather
from src.osm_lookup import get_infra_features, INFRA_FEATURES

# ============================================================================
# CONFIG
# ============================================================================
st.set_page_config(
    page_title="AHPS - Dự đoán điểm nóng tai nạn",
    page_icon="🚗",
    layout="wide",
)

MODELS_DIR = "models"

US_STATES = [
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA",
    "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD",
    "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ",
    "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC",
    "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY", "DC",
]

STATE_NAME_TO_CODE = {
    "Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR",
    "California": "CA", "Colorado": "CO", "Connecticut": "CT", "Delaware": "DE",
    "Florida": "FL", "Georgia": "GA", "Hawaii": "HI", "Idaho": "ID",
    "Illinois": "IL", "Indiana": "IN", "Iowa": "IA", "Kansas": "KS",
    "Kentucky": "KY", "Louisiana": "LA", "Maine": "ME", "Maryland": "MD",
    "Massachusetts": "MA", "Michigan": "MI", "Minnesota": "MN", "Mississippi": "MS",
    "Missouri": "MO", "Montana": "MT", "Nebraska": "NE", "Nevada": "NV",
    "New Hampshire": "NH", "New Jersey": "NJ", "New Mexico": "NM", "New York": "NY",
    "North Carolina": "NC", "North Dakota": "ND", "Ohio": "OH", "Oklahoma": "OK",
    "Oregon": "OR", "Pennsylvania": "PA", "Rhode Island": "RI", "South Carolina": "SC",
    "South Dakota": "SD", "Tennessee": "TN", "Texas": "TX", "Utah": "UT",
    "Vermont": "VT", "Virginia": "VA", "Washington": "WA", "West Virginia": "WV",
    "Wisconsin": "WI", "Wyoming": "WY", "District of Columbia": "DC",
}

WEATHER_GROUPS = [
    "Clear", "Cloudy", "Partly Cloudy", "Rain", "Snow/Ice",
    "Fog/Haze", "Storm", "Wind/Dust", "Other",
]
WEATHER_VN = {
    "Clear": "☀️ Trời quang",
    "Cloudy": "☁️ Nhiều mây",
    "Partly Cloudy": "⛅ Có mây",
    "Rain": "🌧️ Mưa",
    "Snow/Ice": "❄️ Tuyết/Băng",
    "Fog/Haze": "🌫️ Sương mù",
    "Storm": "⛈️ Bão",
    "Wind/Dust": "💨 Gió/Bụi",
    "Other": "🌡️ Khác",
}

WIND_DIRS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
WIND_DIRS_VN = {
    "N": "Bắc (N)", "NE": "Đông Bắc (NE)", "E": "Đông (E)", "SE": "Đông Nam (SE)",
    "S": "Nam (S)", "SW": "Tây Nam (SW)", "W": "Tây (W)", "NW": "Tây Bắc (NW)",
}

RAIN_INTENSITY_LABELS = {
    0: "Không mưa",
    1: "Mưa nhẹ (< 0.1 inch/h)",
    2: "Mưa vừa (0.1 - 0.3 inch/h)",
    3: "Mưa nặng (> 0.3 inch/h)",
}

DEFAULTS = {
    "Temperature(F)": 65.0,
    "Humidity(%)": 70.0,
    "Pressure(in)": 29.8,
    "Visibility(mi)": 10.0,
    "Wind_Speed(mph)": 8.0,
    "WindDir": "N",
}


# ============================================================================
# LOAD MODEL (cached)
# ============================================================================
@st.cache_resource
def load_artifacts():
    """Load model, thresholds, encoders. Prefer tuned model if exists."""
    tuned_path = os.path.join(MODELS_DIR, "model_xgb_tuned.pkl")
    base_path = os.path.join(MODELS_DIR, "model_xgboost.pkl")
    thresh_path = os.path.join(MODELS_DIR, "best_thresholds.pkl")
    enc_path = os.path.join(MODELS_DIR, "encoders.pkl")

    if os.path.exists(tuned_path):
        model_path = tuned_path
        model_name = "XGBoost (Tuned)"
    elif os.path.exists(base_path):
        model_path = base_path
        model_name = "XGBoost (Base)"
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


# ============================================================================
# HELPERS
# ============================================================================
def encode_safe(encoder, value, fallback_idx=0):
    try:
        return int(encoder.transform([value])[0])
    except (ValueError, KeyError):
        return fallback_idx


def deg_to_compass(deg):
    return WIND_DIRS[round(deg / 45) % 8]


def fetch_openweather(lat, lng, api_key):
    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {"lat": lat, "lon": lng, "appid": api_key, "units": "imperial"}
    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    return resp.json()


@st.cache_data(ttl=86400, show_spinner=False)
def get_state_from_coords(lat, lng):
    """Reverse geocoding để lấy bang Mỹ từ tọa độ (dùng Nominatim của OSM)."""
    try:
        url = "https://nominatim.openstreetmap.org/reverse"
        params = {
            "lat": lat,
            "lon": lng,
            "format": "json",
            "zoom": 5,
            "addressdetails": 1,
        }
        headers = {"User-Agent": "AHPS-CS313-Demo/1.0"}
        resp = requests.get(url, params=params, headers=headers, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        address = data.get("address", {})
        country = address.get("country_code", "").lower()

        if country != "us":
            return None  # Không phải Mỹ

        state_full = address.get("state", "")
        return STATE_NAME_TO_CODE.get(state_full)
    except Exception:
        return None


def get_alert_color(alert_level):
    return {"HIGH RISK": "red", "CAUTION": "orange", "SAFE": "green"}.get(alert_level, "blue")


def get_alert_vn(alert_level):
    return {
        "HIGH RISK": "🔴 NGUY HIỂM CAO",
        "CAUTION": "🟠 CẦN THẬN TRỌNG",
        "SAFE": "🟢 AN TOÀN",
    }.get(alert_level, alert_level)


def make_gauge(probability):
    """Gauge chart đẹp hơn, dùng plotly với màu hài hòa."""
    pct = probability * 100

    if pct >= 70:
        bar_color = "#E74C3C"  # đỏ
    elif pct >= 40:
        bar_color = "#F39C12"  # cam
    else:
        bar_color = "#27AE60"  # xanh lá

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=pct,
        number={
            "suffix": "%",
            "font": {"size": 56, "color": "white", "family": "Arial Black"},
            "valueformat": ".1f",
        },
        domain={"x": [0, 1], "y": [0, 1]},
        gauge={
            "axis": {
                "range": [0, 100],
                "tickwidth": 1,
                "tickcolor": "lightgray",
                "tickfont": {"size": 14, "color": "lightgray"},
                "tickmode": "array",
                "tickvals": [0, 20, 40, 60, 80, 100],
            },
            "bar": {"color": bar_color, "thickness": 0.7},
            "bgcolor": "rgba(0,0,0,0)",
            "borderwidth": 2,
            "bordercolor": "rgba(255,255,255,0.2)",
            "steps": [
                {"range": [0, 40], "color": "rgba(39, 174, 96, 0.25)"},
                {"range": [40, 70], "color": "rgba(243, 156, 18, 0.25)"},
                {"range": [70, 100], "color": "rgba(231, 76, 60, 0.25)"},
            ],
        },
    ))

    fig.update_layout(
        height=320,
        margin=dict(l=30, r=30, t=40, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": "white", "family": "Arial"},
    )
    return fig


# ============================================================================
# MAIN UI
# ============================================================================
st.title("🚗 AHPS - Hệ thống dự đoán điểm nóng tai nạn giao thông")
st.markdown("*Click vào bản đồ để chọn vị trí, sau đó điền thông tin và nhấn **Dự đoán***")

model, threshold, encoders, model_name, model_path = load_artifacts()

if model is None:
    st.error(
        "❌ **Không tìm thấy file model!**\n\n"
        "Bạn cần chạy training trước:\n\n"
        "```bash\npython main.py --data data/US_Accidents_March23_sampled_500k.csv\n```"
    )
    st.stop()

st.success(f"✅ Đã load model: **{model_name}** | Threshold: `{threshold:.3f}` | File: `{os.path.basename(model_path)}`")

# Init session state
if "clicked_lat" not in st.session_state:
    st.session_state.clicked_lat = 34.0522
if "clicked_lng" not in st.session_state:
    st.session_state.clicked_lng = -118.2437
if "last_result" not in st.session_state:
    st.session_state.last_result = None
if "weather_autofill" not in st.session_state:
    st.session_state.weather_autofill = {}
if "detected_state" not in st.session_state:
    st.session_state.detected_state = "CA"

col_left, col_right = st.columns([1, 1])

# ----------------------------------------------------------------------------
# LEFT: Map
# ----------------------------------------------------------------------------
with col_left:
    st.subheader("🗺️ Chọn vị trí trên bản đồ (Mỹ)")

    m = folium.Map(location=[39.8283, -98.5795], zoom_start=4, tiles="OpenStreetMap")

    folium.Marker(
        [st.session_state.clicked_lat, st.session_state.clicked_lng],
        tooltip="Vị trí đã chọn",
        icon=folium.Icon(color="blue", icon="info-sign"),
    ).add_to(m)

    if st.session_state.last_result:
        res = st.session_state.last_result
        folium.CircleMarker(
            [res["lat"], res["lng"]],
            radius=15,
            color=get_alert_color(res["alert_level"]),
            fill=True,
            fillOpacity=0.6,
            popup=f"{get_alert_vn(res['alert_level'])}<br>Xác suất: {res['probability']:.1%}",
        ).add_to(m)

    map_data = st_folium(m, height=500, width=None, returned_objects=["last_clicked"])

    if map_data and map_data.get("last_clicked"):
        new_lat = map_data["last_clicked"]["lat"]
        new_lng = map_data["last_clicked"]["lng"]
        if (new_lat != st.session_state.clicked_lat) or (new_lng != st.session_state.clicked_lng):
            st.session_state.clicked_lat = new_lat
            st.session_state.clicked_lng = new_lng
            st.session_state.weather_autofill = {}
            # Tự động phát hiện bang từ tọa độ
            detected = get_state_from_coords(new_lat, new_lng)
            st.session_state.detected_state = detected if detected else None
            st.rerun()

    detected = st.session_state.detected_state
    state_info = f" | Bang: **{detected}**" if detected else " | ⚠️ Ngoài lãnh thổ Mỹ"
    st.info(
        f"📍 **Vị trí:** `{st.session_state.clicked_lat:.4f}, {st.session_state.clicked_lng:.4f}`{state_info}"
    )

# ----------------------------------------------------------------------------
# RIGHT: Form
# ----------------------------------------------------------------------------
with col_right:
    st.subheader("📝 Thông tin dự đoán")

    # --- BẮT BUỘC ---
    st.markdown("#### 🟢 Thông tin bắt buộc")

    detected = st.session_state.detected_state
    if detected and detected in US_STATES:
        default_idx = US_STATES.index(detected)
        state_label = f"Bang (State) — 🎯 Đã tự phát hiện: **{detected}**"
    else:
        default_idx = US_STATES.index("CA")
        state_label = "Bang (State) — ⚠️ Không phát hiện được, vui lòng chọn thủ công"

    state = st.selectbox(state_label, US_STATES, index=default_idx)

    col_d, col_t = st.columns(2)
    with col_d:
        date_input = st.date_input("Ngày", value=datetime.now().date())
    with col_t:
        time_input = st.time_input("Giờ", value=dtime(8, 0))

    is_night = st.radio("Thời điểm", ["☀️ Ban ngày", "🌙 Ban đêm"], horizontal=True) == "🌙 Ban đêm"

    weather_label = st.selectbox(
        "Tình trạng thời tiết",
        WEATHER_GROUPS,
        format_func=lambda x: WEATHER_VN[x],
        index=0,
    )

    rain_intensity = st.selectbox(
        "Cường độ mưa",
        [0, 1, 2, 3],
        format_func=lambda x: RAIN_INTENSITY_LABELS[x],
        index=0,
    )

    st.markdown("---")

    # --- TÙY CHỌN ---
    st.markdown("#### 🔵 Thông tin chi tiết (tùy chọn)")
    st.caption("💡 Bỏ trống = tự động dùng giá trị mặc định, hoặc nhấn nút bên dưới để lấy thời tiết thực tế")

    if st.button("☁️ Lấy thời tiết hiện tại tại vị trí đã chọn", use_container_width=True):
        # Đọc API key từ Streamlit secrets (khi deploy) hoặc .env (khi chạy local)
        api_key = ""
        try:
            api_key = st.secrets["OPENWEATHER_API_KEY"].strip()
        except (KeyError, FileNotFoundError):
            api_key = os.getenv("OPENWEATHER_API_KEY", "").strip()

        if not api_key:
            st.error(
                "❌ Chưa cấu hình API key. Vui lòng tạo file `.env` trong thư mục dự án với nội dung:\n\n"
                "`OPENWEATHER_API_KEY=your_key_here`"
            )
        else:
            try:
                with st.spinner("Đang lấy dữ liệu thời tiết..."):
                    data = fetch_openweather(
                        st.session_state.clicked_lat,
                        st.session_state.clicked_lng,
                        api_key,
                    )
                st.session_state.weather_autofill = {
                    "Temperature(F)": data["main"]["temp"],
                    "Humidity(%)": data["main"]["humidity"],
                    "Pressure(in)": data["main"]["pressure"] * 0.02953,
                    "Visibility(mi)": data.get("visibility", 16093) / 1609.34,
                    "Wind_Speed(mph)": data["wind"]["speed"],
                    "WindDir": deg_to_compass(data["wind"].get("deg", 0)),
                    "weather_desc": data["weather"][0]["description"],
                }
                st.success(
                    f"✅ Đã tự động điền thời tiết: **{data['weather'][0]['description']}**"
                )
                st.rerun()
            except requests.HTTPError as e:
                if e.response.status_code == 401:
                    st.error("❌ API key không hợp lệ hoặc chưa active (chờ ~10 phút sau khi tạo).")
                else:
                    st.error(f"❌ Lỗi API: {e}")
            except Exception as e:
                st.error(f"❌ Lỗi: {e}")

    af = st.session_state.weather_autofill

    col_a, col_b = st.columns(2)
    with col_a:
        temperature = st.number_input(
            "🌡️ Nhiệt độ (°F)",
            value=float(af.get("Temperature(F)", DEFAULTS["Temperature(F)"])),
            min_value=-50.0, max_value=130.0, step=1.0,
        )
        pressure = st.number_input(
            "🔵 Áp suất (inHg)",
            value=float(af.get("Pressure(in)", DEFAULTS["Pressure(in)"])),
            min_value=26.0, max_value=32.0, step=0.1,
        )
        wind_speed = st.number_input(
            "💨 Tốc độ gió (mph)",
            value=float(af.get("Wind_Speed(mph)", DEFAULTS["Wind_Speed(mph)"])),
            min_value=0.0, max_value=200.0, step=1.0,
        )

    with col_b:
        humidity = st.number_input(
            "💧 Độ ẩm (%)",
            value=float(af.get("Humidity(%)", DEFAULTS["Humidity(%)"])),
            min_value=0.0, max_value=100.0, step=1.0,
        )
        visibility = st.number_input(
            "👁️ Tầm nhìn (mi)",
            value=float(af.get("Visibility(mi)", DEFAULTS["Visibility(mi)"])),
            min_value=0.0, max_value=20.0, step=0.5,
        )
        wind_dir = st.selectbox(
            "🧭 Hướng gió",
            WIND_DIRS,
            format_func=lambda x: WIND_DIRS_VN[x],
            index=WIND_DIRS.index(af.get("WindDir", DEFAULTS["WindDir"])),
        )

    st.markdown("---")

    predict_btn = st.button("🔮 DỰ ĐOÁN", type="primary", use_container_width=True)


# ============================================================================
# PREDICTION
# ============================================================================
if predict_btn:
    lat = st.session_state.clicked_lat
    lng = st.session_state.clicked_lng

    dt_combined = datetime.combine(date_input, time_input)
    hour = dt_combined.hour
    month = dt_combined.month
    day_of_week = dt_combined.weekday()
    is_rush_hour = int(hour in [7, 8, 9, 16, 17, 18, 19])
    is_night_int = int(is_night)

    is_raining = int(rain_intensity > 0)

    weather_enc = encode_safe(encoders["weather"], weather_label, fallback_idx=0)
    wind_enc = encode_safe(encoders["wind"], wind_dir, fallback_idx=0)
    state_enc = encode_safe(encoders["state"], state.upper(), fallback_idx=0)
    county_enc = encode_safe(encoders["county"], "Other", fallback_idx=0)

    with st.spinner("🛣️ Đang truy vấn cơ sở hạ tầng đường bộ từ OpenStreetMap..."):
        try:
            infra = get_infra_features(lat, lng, db_path=os.path.join(MODELS_DIR, "infra_lookup.db"))
        except Exception as e:
            st.warning(f"Không truy vấn được OSM ({e}). Dùng giá trị mặc định.")
            infra = {k: 0 for k in INFRA_FEATURES}

    point = {
        "hour": hour, "month": month, "day_of_week": day_of_week,
        "is_rush_hour": is_rush_hour, "is_night": is_night_int,
        "Temperature(F)": temperature, "Humidity(%)": humidity,
        "Pressure(in)": pressure, "Visibility(mi)": visibility,
        "Wind_Speed(mph)": wind_speed,
        "is_raining": is_raining, "rain_intensity": rain_intensity,
        **infra,
        "Weather_enc": weather_enc, "WindDir_enc": wind_enc,
        "State_enc": state_enc, "County_enc": county_enc,
    }
    point = add_derived_features(point)

    result = predict_hotspot(model, point, threshold)
    result["lat"] = lat
    result["lng"] = lng
    result["infra"] = infra
    result["point"] = point
    st.session_state.last_result = result

    st.markdown("---")
    st.markdown("## 📊 Kết quả dự đoán")

    col_g, col_info = st.columns([1, 1])
    with col_g:
        st.plotly_chart(make_gauge(result["probability"]), use_container_width=True)

    with col_info:
        alert_vn = get_alert_vn(result["alert_level"])

        st.markdown(f"### {alert_vn}")
        st.metric("Xác suất là điểm nóng", f"{result['probability']:.2%}")
        st.metric("Phân loại", "ĐIỂM NÓNG" if result["prediction"] == 1 else "KHÔNG PHẢI ĐIỂM NÓNG")

        st.markdown(f"""
        **Vị trí:** ({lat:.4f}, {lng:.4f})  
        **Bang:** {state}  
        **Thời gian:** {dt_combined.strftime('%Y-%m-%d %H:%M')}  
        **Threshold model:** {threshold:.3f}
        """)

    with st.expander("🔍 Xem chi tiết các đặc trưng đã dùng"):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**🛣️ Cơ sở hạ tầng (từ OSM):**")
            infra_detected = [k for k, v in infra.items() if v == 1]
            if infra_detected:
                for it in infra_detected:
                    st.markdown(f"- ✅ {it}")
            else:
                st.markdown("_Không phát hiện cơ sở hạ tầng đặc biệt nào_")

        with col2:
            st.markdown("**⚙️ Đặc trưng thời gian:**")
            st.markdown(f"- Giờ: `{hour}`")
            st.markdown(f"- Tháng: `{month}`")
            st.markdown(f"- Thứ trong tuần: `{day_of_week}` (0=Thứ 2)")
            st.markdown(f"- Giờ cao điểm: `{'Có' if is_rush_hour else 'Không'}`")
            st.markdown(f"- Ban đêm: `{'Có' if is_night_int else 'Không'}`")

        st.markdown("**📋 Full feature vector đưa vào model:**")
        feat_df = pd.DataFrame([point]).T.reset_index()
        feat_df.columns = ["Feature", "Value"]
        st.dataframe(feat_df, use_container_width=True, height=300)


st.markdown("---")
st.caption(
    "🎓 **AHPS** - Accidents Hotspot Prediction System | "
    "Đồ án CS313 | Dataset: US Accidents (Kaggle)"
)
