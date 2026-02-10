import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import warnings
import traceback

warnings.filterwarnings("ignore")

st.set_page_config(page_title="House Price Predictor", layout="wide")

st.markdown(
    """
<style>
.block-container { padding-top: 2.2rem; padding-bottom: 1.2rem; }
div[data-testid="stVerticalBlock"] { gap: 0.6rem; }

.main-header {
  font-size: 2.2rem; font-weight: 900; color: #1f77b4;
  text-align:center; margin: 1rem 0 0.4rem 0;
}
.sub-header {
  text-align:center; color:#666; margin-bottom: 1.0rem; font-size: 1rem;
}

.card {
  border: 1px solid #e0e0e0;
  border-radius: 12px;
  padding: 20px 24px;
  background: #fafafa;
  margin-bottom: 18px;
}

.prediction-box {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  padding: 22px 18px;
  border-radius: 16px;
  color: white;
  text-align: center;
  margin: 12px 0 8px 0;
  box-shadow: 0 8px 25px rgba(0,0,0,0.10);
}
.muted { color:#777; font-size:0.9rem; }

label { font-size: 0.85rem !important; color: #444 !important; }

div[data-testid="stFormSubmitButton"] > button {
  width: 100% !important;
  background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%) !important;
  color: white !important;
  font-weight: 800 !important;
  border: none !important;
  padding: 0.85rem 2rem !important;
  font-size: 1.1rem !important;
  border-radius: 12px !important;
  box-shadow: 0 8px 22px rgba(99, 102, 241, 0.45) !important;
  transition: transform 0.2s ease, box-shadow 0.2s ease !important;
}

/* hover */
div[data-testid="stFormSubmitButton"] > button:hover {
  background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%) !important;
  transform: translateY(-2px) !important;
  box-shadow: 0 12px 28px rgba(99, 102, 241, 0.60) !important;
}

/* active */
div[data-testid="stFormSubmitButton"] > button:active {
  transform: translateY(0px) !important;
  box-shadow: 0 6px 16px rgba(99, 102, 241, 0.40) !important;
}
</style>
""",
    unsafe_allow_html=True,
)

st.markdown('<div class="main-header">House Price Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Aligned preprocessing pipeline (same as your notebook)</div>', unsafe_allow_html=True)


FEATURE_GROUPS = {
    "Quality & Condition": [
        "OverallQual",
        "OverallCond",
        "QualityCond",   # engineered, computed automatically
    ],
    "Year & Age": [
        "YearBuilt",
        "YearRemodAdd",
        "GarageYrBlt",
        "MoSold",
    ],
    "Living Area": [
        "GrLivArea",
        "1stFlrSF",
        "2ndFlrSF",
        "TotRmsAbvGrd",
        "AreaPerRoom",   # engineered, computed automatically
        "TotalSF",       # engineered, computed automatically
        "TotalBath",     # engineered, computed automatically
        "TotalRooms",    # engineered, computed automatically
    ],
    "Basement": [
        "TotalBsmtSF",
        "BsmtFinSF1",
        "BsmtFinSF2",
        "BsmtUnfSF",
        "BsmtFinishedRatio",  # engineered
    ],
    "Garage": [
        "GarageCars",
        "GarageArea",
        "GarageType_Detchd",  # one-hot/binary-like in your top-25
    ],
    "Lot & Outdoor": [
        "LotArea",
        "LotFrontage",
        "OpenPorchSF",
        "WoodDeckSF",
        "EnclosedPorch",
        "3SsnPorch",
        "ScreenPorch",
        "TotalPorchSF",  # engineered
    ],
    "Other": [
        "MasVnrArea",
        "CentralAir_Y",  # one-hot/binary-like
        "IsRemodeled",   # engineered
    ],
}


@st.cache_resource
def load_model_assets():
    model_path = Path("house_price_model.pkl")
    scaler_path = Path("house_scaler.pkl")
    features_path = Path("feature_names.txt")
    skewed_path = Path("skewed_features.pkl")

    if not model_path.exists():
        return None, None, [], []

    model = joblib.load(model_path)

    scaler = None
    if scaler_path.exists():
        try:
            scaler = joblib.load(scaler_path)
        except Exception:
            scaler = None

    feature_names = []
    if features_path.exists():
        with open(features_path, "r", encoding="utf-8") as f:
            feature_names = [line.strip() for line in f if line.strip()]

    skewed_features = []
    if skewed_path.exists():
        try:
            skewed_features = joblib.load(skewed_path)
        except Exception:
            skewed_features = []

    return model, scaler, feature_names, skewed_features


model, scaler, feature_names, skewed_features = load_model_assets()


def infer_spec(feat: str):
    f = feat.lower()

    # binary-like
    if feat.startswith("Has") or feat in {"IsRemodeled"}:
        return {"type": "select", "options": [0, 1], "default": 0}

    # one-hot-ish
    if "_" in feat and feat not in {"1stFlrSF", "2ndFlrSF", "3SsnPorch"}:
        return {"type": "select", "options": [0, 1], "default": 0}

    # YEAR features with minimums
    if feat in {"YearBuilt", "YearRemodAdd", "GarageYrBlt"}:
        return {"type": "int", "min": 1800, "max": 2026, "step": 1, "default": 2000}

    if feat == "MoSold":
        return {"type": "int", "min": 1, "max": 12, "step": 1, "default": 6}

    # quality/condition/cars/rooms are int-ish
    if any(k in f for k in ["qual", "cond", "cars", "rooms", "bedroom", "kitchen"]):
        return {"type": "int", "min": 0, "max": 20, "step": 1, "default": 0}

    # ratio-ish
    if "ratio" in f:
        return {"type": "float", "min": 0.0, "max": 1.0, "step": 0.01, "default": 0.0}

    # area/sf-ish
    if any(k in f for k in ["area", "sf", "frontage"]):
        return {"type": "float", "min": 0.0, "max": 100000.0, "step": 10.0, "default": 0.0}

    return {"type": "float", "min": 0.0, "max": 100000.0, "step": 1.0, "default": 0.0}


def render_feature_input(name: str, spec: dict):
    key = f"feat_{name}"

    if spec["type"] == "select":
        options = spec["options"]
        default = spec.get("default", options[0])
        idx = options.index(default) if default in options else 0
        return st.selectbox(name, options=options, index=idx, key=key)

    is_int = spec["type"] == "int"
    return st.number_input(
        name,
        min_value=spec["min"],
        max_value=spec["max"],
        value=spec.get("default", spec["min"]),
        step=spec["step"],
        format="%d" if is_int else None,
        key=key,
    )


def build_model_input(feature_names_list: list[str], raw: dict | pd.Series) -> pd.DataFrame:
    """
    Build one-row DataFrame aligned to MODEL feature names (top-25 in your case),
    computing engineered features from raw BASE inputs.
    """
    row = raw.to_dict() if isinstance(raw, pd.Series) else dict(raw)

    X = pd.DataFrame([[0.0] * len(feature_names_list)], columns=feature_names_list)

    # Copy whatever raw values match directly
    for k, v in row.items():
        if k in X.columns:
            try:
                X.loc[0, k] = float(v)
            except Exception:
                pass

    # Engineered features (only if model expects them)
    if "QualityCond" in X.columns:
        X.loc[0, "QualityCond"] = float(row.get("OverallQual", 0)) * float(row.get("OverallCond", 0))

    if "TotalSF" in X.columns:
        X.loc[0, "TotalSF"] = float(row.get("1stFlrSF", 0)) + float(row.get("2ndFlrSF", 0)) + float(row.get("TotalBsmtSF", 0))

    if "TotalBath" in X.columns:
        # NOTE: matches your notebook (no *0.5)
        X.loc[0, "TotalBath"] = (
            float(row.get("FullBath", 0)) + float(row.get("HalfBath", 0))
            + float(row.get("BsmtFullBath", 0)) + float(row.get("BsmtHalfBath", 0))
        )

    if "TotalPorchSF" in X.columns:
        X.loc[0, "TotalPorchSF"] = (
            float(row.get("WoodDeckSF", 0)) + float(row.get("OpenPorchSF", 0))
            + float(row.get("EnclosedPorch", 0)) + float(row.get("3SsnPorch", 0)) + float(row.get("ScreenPorch", 0))
        )

    if "BsmtFinishedRatio" in X.columns:
        bsmt_fin = float(row.get("BsmtFinSF1", 0)) + float(row.get("BsmtFinSF2", 0))
        X.loc[0, "BsmtFinishedRatio"] = bsmt_fin / (float(row.get("TotalBsmtSF", 0)) + 1e-8)

    if "AreaPerRoom" in X.columns:
        X.loc[0, "AreaPerRoom"] = float(row.get("GrLivArea", 0)) / (float(row.get("TotRmsAbvGrd", 0)) + 1e-8)

    if "TotalRooms" in X.columns:
        X.loc[0, "TotalRooms"] = float(row.get("TotRmsAbvGrd", 0)) + float(row.get("BedroomAbvGr", 0)) + float(row.get("KitchenAbvGr", 0))

    if "IsRemodeled" in X.columns:
        X.loc[0, "IsRemodeled"] = 1.0 if float(row.get("YearRemodAdd", 0)) != float(row.get("YearBuilt", 0)) else 0.0

    return X


def predict_from_raw_row(_model, _scaler, _feature_names, _skewed_features, raw: dict | pd.Series) -> float | None:
    """
    raw: user inputs (base features), not scaled.
    returns: predicted SalePrice (original scale)
    """
    try:
        if _model is None or _scaler is None or not _feature_names:
            return None

        row = raw.to_dict() if isinstance(raw, pd.Series) else dict(raw)

        # (A) log1p on skewed BASE features FIRST
        row_transformed = {}
        for k, v in row.items():
            try:
                fv = float(v)
            except Exception:
                fv = 0.0
            if k in _skewed_features:
                row_transformed[k] = np.log1p(fv)
            else:
                row_transformed[k] = fv

        # (B) feature engineering (computed using transformed base features)
        X_model = build_model_input(_feature_names, row_transformed)

        # (C) align to scaler expected + scale
        expected_cols = _scaler.feature_names_in_.tolist()
        X_for_scaler = X_model.reindex(columns=expected_cols, fill_value=0.0)
        X_scaled_all = pd.DataFrame(_scaler.transform(X_for_scaler), columns=expected_cols, index=X_model.index)

        # (D) select model features for prediction
        Xs = X_scaled_all.reindex(columns=_feature_names, fill_value=0.0)

        pred_log = float(_model.predict(Xs)[0])
        return float(np.exp(pred_log))  # because target was log(SalePrice)

    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        st.code(traceback.format_exc())
        return None


if model is None or scaler is None or not feature_names:
    st.error("Model/scaler/features not loaded. Make sure these files exist in the same folder:\n"
             "- house_price_model.pkl\n- house_scaler.pkl\n- feature_names.txt\n- skewed_features.pkl")
    st.stop()

st.success(f"Model loaded: {len(feature_names)} features")

NEEDED_BASE = {
    "OverallQual", "OverallCond", "GrLivArea",
    "1stFlrSF", "2ndFlrSF", "TotalBsmtSF",
    "FullBath", "HalfBath", "BsmtFullBath", "BsmtHalfBath",
    "WoodDeckSF", "OpenPorchSF", "EnclosedPorch", "3SsnPorch", "ScreenPorch",
    "BsmtFinSF1", "BsmtFinSF2",
    "TotRmsAbvGrd", "BedroomAbvGr", "KitchenAbvGr",
    "YearBuilt", "YearRemodAdd",
}

DISPLAY_SET = set(feature_names) | NEEDED_BASE

with st.form("all_features_form", clear_on_submit=False):
    user_values = {}

    for group_name, feats in FEATURE_GROUPS.items():
        feats = [f for f in feats if f in DISPLAY_SET]

        if not feats:
            continue

        st.markdown(f"#### {group_name}")
        cols = st.columns(3, gap="large")

        for i, feat in enumerate(feats):
            with cols[i % 3]:
                spec = infer_spec(feat)
                user_values[feat] = render_feature_input(feat, spec)

        st.divider()

    submitted = st.form_submit_button("PREDICT PRICE", use_container_width=True)

def validate_inputs(user_values: dict) -> list[str]:
    """
    Return a list of user-facing error messages.
    If empty => inputs are valid.
    """
    errs = []

    def f(name, default=0.0):
        try:
            return float(user_values.get(name, default))
        except Exception:
            return default

    required = ["OverallQual", "OverallCond", "GrLivArea", "1stFlrSF", "TotalBsmtSF", "YearBuilt"]
    missing = [r for r in required if r not in user_values]
    if missing:
        errs.append(f"Missing required inputs: {', '.join(missing)}")

    nonneg = [
        "GrLivArea","1stFlrSF","2ndFlrSF","TotalBsmtSF",
        "LotArea","LotFrontage","GarageArea","WoodDeckSF","OpenPorchSF",
        "EnclosedPorch","3SsnPorch","ScreenPorch","MasVnrArea",
        "BsmtFinSF1","BsmtFinSF2","BsmtUnfSF"
    ]
    for col in nonneg:
        if col in user_values and f(col) < 0:
            errs.append(f"'{col}' cannot be negative.")

    oq = f("OverallQual")
    oc = f("OverallCond")
    if "OverallQual" in user_values and not (1 <= oq <= 10):
        errs.append("OverallQual must be between 1 and 10.")
    if "OverallCond" in user_values and not (1 <= oc <= 10):
        errs.append("OverallCond must be between 1 and 10.")

    gla = f("GrLivArea")
    if "GrLivArea" in user_values and (gla <= 0 or gla > 10000):
        errs.append("GrLivArea must be between 1 and 10,000 sqft.")

    first = f("1stFlrSF")
    if "1stFlrSF" in user_values and (first <= 0 or first > 8000):
        errs.append("1stFlrSF must be between 1 and 8,000 sqft.")

    bsmt = f("TotalBsmtSF")
    if "TotalBsmtSF" in user_values and (bsmt < 0 or bsmt > 8000):
        errs.append("TotalBsmtSF must be between 0 and 8,000 sqft.")

    rooms = f("TotRmsAbvGrd")
    bed = f("BedroomAbvGr")
    kit = f("KitchenAbvGr")
    if "TotRmsAbvGrd" in user_values and rooms <= 0:
        errs.append("TotRmsAbvGrd must be at least 1.")
    if "BedroomAbvGr" in user_values and bed < 0:
        errs.append("BedroomAbvGr cannot be negative.")
    if "KitchenAbvGr" in user_values and kit < 0:
        errs.append("KitchenAbvGr cannot be negative.")
    if ("BedroomAbvGr" in user_values and "TotRmsAbvGrd" in user_values) and bed > rooms:
        errs.append("BedroomAbvGr cannot be greater than TotRmsAbvGrd.")

    yb = int(f("YearBuilt"))
    yr = int(f("YearRemodAdd"))
    yg = int(f("GarageYrBlt"))

    if "YearBuilt" in user_values and not (1800 <= yb <= 2026):
        errs.append("YearBuilt must be between 1800 and 2026.")

    if "YearRemodAdd" in user_values:
        if not (1800 <= yr <= 2026):
            errs.append("YearRemodAdd must be between 1800 and 2026.")
        if yr < yb:
            errs.append("YearRemodAdd cannot be earlier than YearBuilt.")

    if "GarageYrBlt" in user_values:
        if yg != 0 and not (1800 <= yg <= 2026):
            errs.append("GarageYrBlt must be 0 (no garage) or between 1800 and 2026.")
        if yg != 0 and yg < yb:
            errs.append("GarageYrBlt cannot be earlier than YearBuilt (unless it is 0).")

    ms = int(f("MoSold"))
    if "MoSold" in user_values and not (1 <= ms <= 12):
        errs.append("MoSold must be between 1 and 12.")

    return errs

if submitted:
    errors = validate_inputs(user_values)

    if errors:
        st.error("Please fix the following input issues before predicting:")
        for msg in errors:
            st.write(f"• {msg}")
        st.stop()  
        
    price = predict_from_raw_row(model, scaler, feature_names, skewed_features, raw=user_values)

    if price is None:
        st.error("Prediction failed due to an internal preprocessing mismatch. Please try again.")
    else:
        st.markdown(
            f"""
            <div class="prediction-box">
                <div style="font-size: 3rem; font-weight: 900; margin: 6px 0;">
                    ${price:,.2f}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

st.divider()
st.markdown(
    "<div style='text-align:center;color:#666;padding:8px 0 14px 0;'>House Price Predictor | Streamlit</div>",
    unsafe_allow_html=True,
)
