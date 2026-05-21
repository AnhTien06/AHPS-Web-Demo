# Accidents Hotspot Prediction System (AHPS)

Binary classification system that predicts whether a road location/time combination is a traffic accident hotspot, using the US Accidents dataset (~500k records).

## Project Structure

```
.
├── src/
│   ├── data_loader.py      # Load raw CSV
│   ├── preprocessing.py    # Feature engineering & encoding
│   ├── clustering.py       # DBSCAN spatio-temporal labeling
│   ├── modeling.py         # Train & evaluate 4 models
│   ├── tuning.py           # Optuna hyperparameter tuning
│   └── predict.py          # Inference utilities
├── main.py                 # Full pipeline entry point
├── requirements.txt
└── README.md
```

## Pipeline

1. **Load** — reads raw `US_Accidents_March23_sampled_500k.csv`
2. **Preprocess** — drops leaky/redundant columns, removes physical outliers, fills missing values, engineers time/rain features, label-encodes categoricals
3. **Label** — DBSCAN (eps=300m, 10 time buckets) assigns `label=1` (hotspot) or `label=0` (noise)
4. **Train** — Logistic Regression, Random Forest, LightGBM, XGBoost (70/15/15 split, stratified)
5. **Tune** *(optional)* — Optuna searches 50 trials each for LightGBM and XGBoost, optimising F1 on the val set, then finds the best probability threshold
6. **Demo** — batch predictions on scenario test cases

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```bash
# Place the dataset under data/
mkdir data
cp US_Accidents_March23_sampled_500k.csv data/

# Run full pipeline
python main.py

# With Optuna tuning (slower, ~1-2h on GPU)
python main.py --tune --n-trials 50

# Custom data path
python main.py --data /path/to/your/data.csv
```

## Features (25 total)

| Group | Features |
|---|---|
| Time | `hour`, `month`, `day_of_week`, `is_rush_hour`, `is_night` |
| Weather | `Temperature(F)`, `Humidity(%)`, `Pressure(in)`, `Visibility(mi)`, `Wind_Speed(mph)`, `is_raining`, `rain_intensity`, `Weather_enc`, `WindDir_enc` |
| Infrastructure | `Amenity`, `Crossing`, `Give_Way`, `Junction`, `No_Exit`, `Railway`, `Station`, `Stop`, `Traffic_Signal` |
| Location | `State_enc`, `County_enc` |

## Outputs

| File | Description |
|---|---|
| `data/US_Accidents_Labeled_Dataset.csv` | Preprocessed + labeled dataset |
| `models/model_*.pkl` | Trained base models |
| `models/model_lgb_tuned.pkl` | Tuned LightGBM |
| `models/model_xgb_tuned.pkl` | Tuned XGBoost |
| `models/best_thresholds.pkl` | Optimal decision thresholds |

## Inference

```python
from src.predict import load_model, load_thresholds, predict_hotspot

model = load_model('models/model_xgboost.pkl')

point = {
    'hour': 8, 'month': 3, 'day_of_week': 0,
    'is_rush_hour': 1, 'is_night': 0,
    'Temperature(F)': 65.0, 'Humidity(%)': 70.0,
    'Pressure(in)': 29.8, 'Visibility(mi)': 10.0,
    'Wind_Speed(mph)': 8.0, 'is_raining': 1, 'rain_intensity': 2,
    'Amenity': 0, 'Crossing': 1, 'Give_Way': 0, 'Junction': 1,
    'No_Exit': 0, 'Railway': 0, 'Station': 0, 'Stop': 0,
    'Traffic_Signal': 1, 'Weather_enc': 4, 'WindDir_enc': 5,
    'State_enc': 3, 'County_enc': 12,
}

result = predict_hotspot(model, point, threshold=0.5)
print(result)
# {'probability': 0.82, 'prediction': 1, 'alert_level': 'HIGH RISK'}
```
# CS313
# CS313
