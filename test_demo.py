"""Quick demo: OSM lookup + batch predict with existing trained model."""
import sys
import os
import pickle
sys.path.insert(0, '.')

# ── 1. OSM Lookup ────────────────────────────────────────────────────────────
print("=" * 60)
print("OSM Infrastructure Lookup")
print("=" * 60)

from src.osm_lookup import get_infra_features

cases_osm = [
    (34.0522, -118.2437, "Los Angeles (downtown)"),
    (34.0195, -118.4912, "Santa Monica"),
    (33.9425, -118.4081, "LAX Airport"),
]

for lat, lng, desc in cases_osm:
    print(f"\n{desc}  ({lat}, {lng})")
    feats = get_infra_features(lat, lng)
    flags = [k for k, v in feats.items() if v == 1]
    print(f"  Detected : {flags if flags else 'none'}")
    print(f"  All feats: {feats}")

# ── 2. Batch Predict demo ────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Batch Predict Demo (XGBoost)")
print("=" * 60)

from src.predict import load_model, load_thresholds, batch_predict, add_derived_features

model = load_model("models/model_xgboost.pkl")
thresholds = load_thresholds("models/best_thresholds.pkl")
threshold = thresholds.get("XGBoost", 0.5)
print(f"Threshold: {threshold:.3f}\n")

base = {
    "month": 3, "Temperature(F)": 65.0, "Humidity(%)": 70.0,
    "Pressure(in)": 29.8, "Visibility(mi)": 10.0, "Wind_Speed(mph)": 8.0,
    "rain_intensity": 0, "Amenity": 0, "Crossing": 1, "Give_Way": 0,
    "No_Exit": 0, "Railway": 0, "Station": 0, "Stop": 0,
    "WindDir_enc": 5, "Weather_enc": 4, "State_enc": 3, "County_enc": 12,
}

test_cases = [
    {"desc": "LA, 2am weekend",
     "hour": 0,  "day_of_week": 5, "is_night": 1, "is_rush_hour": 0,
     "is_raining": 0, "Junction": 1, "Traffic_Signal": 1},
    {"desc": "LA, 8am rush hour weekday",
     "hour": 8,  "day_of_week": 0, "is_night": 0, "is_rush_hour": 1,
     "is_raining": 0, "Junction": 1, "Traffic_Signal": 1},
    {"desc": "LA, 8am + heavy rain",
     "hour": 8,  "day_of_week": 0, "is_night": 0, "is_rush_hour": 1,
     "is_raining": 1, "Junction": 1, "Traffic_Signal": 1},
    {"desc": "LA, 2am + OSM infra from downtown",
     "hour": 0,  "day_of_week": 5, "is_night": 1, "is_rush_hour": 0,
     "is_raining": 0,
     **{k: v for k, v in get_infra_features(34.0522, -118.2437).items()}},
]

print(f"{'Scenario':<45} {'Probability':>11} {'Alert'}")
print("-" * 75)
batch_predict(model, test_cases, base, threshold=threshold)
