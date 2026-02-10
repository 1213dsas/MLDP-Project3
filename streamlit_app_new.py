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

.error-message {
  color: #dc2626;
  font-size: 0.85rem;
  margin-top: 4px;
  margin-bottom: 8px;
  font-weight: 500;
}

.input-error label {
  color: #dc2626 !important;
}

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

div[data-testid="stFormSubmitButton"] > button:hover {
  background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%) !important;
  transform: translateY(-2px) !important;
  box-shadow: 0 12px 28px rgba(99, 102, 241, 0.60) !important;
}

div[data-testid="stFormSubmitButton"] > button:active {
  transform: translateY(0px) !important;
  box-shadow: 0 6px 16px rgba(99, 102, 241, 0.40) !important;
}
</style>
""",
    unsafe_allow_html=True,
)

st.markdown('<div class="main-header">🏠 House Price Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Aligned preprocessing pipeline (same as your notebook)</div>', unsafe_allow_html=True)


FEATURE_GROUPS = {
    "Quality & Condition": [
        "OverallQual",
        "OverallCond",
        "QualityCond",
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
        "AreaPerRoom",
        "TotalSF",
        "TotalBath",
        "TotalRooms",
    ],
    "Basement": [
        "TotalBsmtSF",
        "BsmtFinSF1",
        "BsmtFinSF2",
        "BsmtUnfSF",
        "BsmtFinishedRatio",
    ],
    "Garage": [
        "GarageCars",
        "GarageArea",
        "GarageType_Detchd",
    ],
    "Lot & Outdoor": [
        "LotArea",
        "LotFrontage",
        "OpenPorchSF",
        "WoodDeckSF",
        "EnclosedPorch",
        "3SsnPorch",
        "ScreenPorch",
        "TotalPorchSF",
    ],
    "Other": [
        "MasVnrArea",
        "CentralAir_Y",
        "IsRemodeled",
        "FullBath",
        "HalfBath",
        "BsmtFullBath",
        "BsmtHalfBath",
        "BedroomAbvGr",
        "KitchenAbvGr",
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

    if feat.startswith("Has") or feat in {"IsRemodeled"}:
        return {"type": "select", "options": [0, 1], "default": 1}

    if "_" in feat and feat not in {"1stFlrSF", "2ndFlrSF", "3SsnPorch"}:
        return {"type": "select", "options": [0, 1], "default": 1}

    if feat in {"YearBuilt", "YearRemodAdd", "GarageYrBlt"}:
        return {"type": "int", "step": 1, "default": 1800}

    if feat == "MoSold":
        return {"type": "int",  "step": 1, "default": 1}

    if any(k in f for k in ["qual", "cond", "cars", "rooms", "bedroom", "kitchen", "bath"]):
        return {"type": "int",  "step": 1, "default": 1}

    if "ratio" in f:
        return {"type": "float", "step": 0.01, "default": 1.0}

    if any(k in f for k in ["area", "sf", "frontage"]):
        return {"type": "float", "step": 10.0, "default": 1.0}

    return {"type": "float", "step": 1.0, "default": 1.0}


def validate_single_field(feat: str, value, all_values: dict) -> str | None:

    try:
        v = float(value)
    except:
        return None
    
    # OverallQual validation
    if feat == "OverallQual":
        if not (1 <= v <= 10):
            return f"Must be between 1-10"
    
    # OverallCond validation
    if feat == "OverallCond":
        if not (1 <= v <= 10):
            return f"Must be between 1-10"
    
    # Year validations
    if feat == "YearBuilt":
        if v < 1800:
            return f"Too early for {int(v)}. Min: 1800"
        if v > 2026:
            return f"Future year {int(v)}. Max: 2026"
    
    if feat == "YearRemodAdd":
        if v < 1800:
            return f"Too early for {int(v)}. Min: 1800"
        if v > 2026:
            return f"Future year {int(v)}. Max: 2026"
        yb = all_values.get("YearBuilt", 0)
        if yb and v < yb:
            return f"Cannot be before YearBuilt ({int(yb)})"
    
    if feat == "GarageYrBlt":
        if v != 0:
            if v < 1800:
                return f"Too early for {int(v)}. Use 0 for no garage or >= 1800"
            if v > 2026:
                return f"Future year {int(v)}. Max: 2026"
            yb = all_values.get("YearBuilt", 0)
            if yb and v < yb:
                return f"Cannot be before YearBuilt ({int(yb)})"
    
    # MoSold validation
    if feat == "MoSold":
        if not (1 <= v <= 12):
            return f"Must be 1-12"
    
    # Area validations
    if feat == "GrLivArea":
        if v <= 0 or v > 10000:
            return f"Must be 1-10,000 sqft"
    
    if feat == "1stFlrSF":
        if v <= 0 or v > 8000:
            return f"Must be 1-8,000 sqft"
    
    if feat == "TotalBsmtSF":
        if v < 0 or v > 8000:
            return f"Must be 0-8,000 sqft"
    
    # Non-negative fields
    nonneg = [
        "2ndFlrSF", "LotArea", "LotFrontage", "GarageArea", 
        "WoodDeckSF", "OpenPorchSF", "EnclosedPorch", "3SsnPorch", 
        "ScreenPorch", "MasVnrArea", "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF"
    ]
    if feat in nonneg and v < 0:
        return f"Cannot be negative"
    
    # Room validations
    if feat == "TotRmsAbvGrd":
        if v <= 0:
            return f"Must be at least 1"
    
    if feat == "BedroomAbvGr":
        if v < 0:
            return f"Cannot be negative"
        rooms = all_values.get("TotRmsAbvGrd", 999)
        if rooms and v > rooms:
            return f"Cannot exceed TotRmsAbvGrd ({rooms:.0f})"
    
    if feat in ["KitchenAbvGr", "FullBath", "HalfBath", "BsmtFullBath", "BsmtHalfBath"]:
        if v < 0:
            return f"Cannot be negative"
    
    return None


def render_feature_input(name: str, spec: dict, user_values: dict):
    """Render input with inline validation"""
    key = f"feat_{name}"
    
    # Get current value from session state
    if key in st.session_state:
        current_val = st.session_state[key]
    else:
        current_val = spec.get("default", spec.get(1))
    
    # Validate current value
    error_msg = validate_single_field(name, current_val, user_values)
    
    # Render input
    if spec["type"] == "select":
        options = spec["options"]
        default = spec.get("default", options[0])
        idx = options.index(default) if default in options else 0
        val = st.selectbox(name, options=options, index=idx, key=key)
    else:
        is_int = spec["type"] == "int"
        val = st.number_input(
            name,
            value=current_val,
            step=spec["step"],
            format="%d" if is_int else None,
            key=key,
        )
    
    # Show error message if invalid
    if error_msg:
        st.markdown(f'<div class="error-message">{error_msg}</div>', unsafe_allow_html=True)
    
    return val


def build_model_input(feature_names_list: list[str], raw: dict | pd.Series) -> pd.DataFrame:
    """
    Build one-row DataFrame aligned to MODEL feature names (top-25 in your case),
    computing engineered features from raw BASE inputs.
    """
    row = raw.to_dict() if isinstance(raw, pd.Series) else dict(raw)

    X = pd.DataFrame([[1.0] * len(feature_names_list)], columns=feature_names_list)

    for k, v in row.items():
        if k in X.columns:
            try:
                X.loc[1, k] = float(v)
            except Exception:
                pass

    if "QualityCond" in X.columns:
        X.loc[1, "QualityCond"] = float(row.get("OverallQual", 0)) * float(row.get("OverallCond", 0))

    if "TotalSF" in X.columns:
        X.loc[1, "TotalSF"] = float(row.get("1stFlrSF", 0)) + float(row.get("2ndFlrSF", 0)) + float(row.get("TotalBsmtSF", 0))

    if "TotalBath" in X.columns:
        X.loc[1, "TotalBath"] = (
            float(row.get("FullBath", 0)) + float(row.get("HalfBath", 0))
            + float(row.get("BsmtFullBath", 0)) + float(row.get("BsmtHalfBath", 0))
        )

    if "TotalPorchSF" in X.columns:
        X.loc[1, "TotalPorchSF"] = (
            float(row.get("WoodDeckSF", 0)) + float(row.get("OpenPorchSF", 0))
            + float(row.get("EnclosedPorch", 0)) + float(row.get("3SsnPorch", 0)) + float(row.get("ScreenPorch", 0))
        )

    if "BsmtFinishedRatio" in X.columns:
        bsmt_fin = float(row.get("BsmtFinSF1", 0)) + float(row.get("BsmtFinSF2", 0))
        X.loc[1, "BsmtFinishedRatio"] = bsmt_fin / (float(row.get("TotalBsmtSF", 0)) + 1e-8)

    if "AreaPerRoom" in X.columns:
        X.loc[1, "AreaPerRoom"] = float(row.get("GrLivArea", 0)) / (float(row.get("TotRmsAbvGrd", 0)) + 1e-8)

    if "TotalRooms" in X.columns:
        X.loc[1, "TotalRooms"] = float(row.get("TotRmsAbvGrd", 0)) + float(row.get("BedroomAbvGr", 0)) + float(row.get("KitchenAbvGr", 0))

    if "IsRemodeled" in X.columns:
        X.loc[1, "IsRemodeled"] = 1.0 if float(row.get("YearRemodAdd", 0)) != float(row.get("YearBuilt", 0)) else 0.0

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

        X_model = build_model_input(_feature_names, row_transformed)

        expected_cols = _scaler.feature_names_in_.tolist()
        X_for_scaler = X_model.reindex(columns=expected_cols, fill_value=0.0)
        X_scaled_all = pd.DataFrame(_scaler.transform(X_for_scaler), columns=expected_cols, index=X_model.index)

        Xs = X_scaled_all.reindex(columns=_feature_names, fill_value=0.0)

        pred_log = float(_model.predict(Xs)[0])
        return float(np.exp(pred_log))

    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        st.code(traceback.format_exc())
        return None


def validate_all_inputs(user_values: dict) -> list[str]:
    """
    Comprehensive validation returning list of all errors.
    """
    errors = []
    
    # Required fields
    required = ["OverallQual", "OverallCond", "GrLivArea", "1stFlrSF", "TotalBsmtSF", "YearBuilt"]
    missing = [r for r in required if r not in user_values]
    if missing:
        errors.append(f"Missing required inputs: {', '.join(missing)}")
    
    # Validate each field
    for feat, value in user_values.items():
        err = validate_single_field(feat, value, user_values)
        if err:
            errors.append(f"{feat}: {err}")
    
    return errors


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
                user_values[feat] = render_feature_input(feat, spec, user_values)

        st.divider()

    submitted = st.form_submit_button("PREDICT PRICE", use_container_width=True)

if submitted:
    errors = validate_all_inputs(user_values)

    if errors:
        st.error("Please fix the validation errors shown in red above before predicting.")
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
        
        lo, hi = price * 0.92, price * 1.08
        st.caption(f"Confidence Range (±8%): ${lo:,.2f} - ${hi:,.2f}")

st.divider()
st.markdown(
    "<div style='text-align:center;color:#666;padding:8px 0 14px 0;'>House Price Predictor | Streamlit</div>",
    unsafe_allow_html=True,
)