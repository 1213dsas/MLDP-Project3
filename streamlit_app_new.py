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

st.markdown('<div class="main-header">House Price Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Aligned preprocessing pipeline (same as your notebook)</div>', unsafe_allow_html=True)


FEATURE_GROUPS = {
    "Quality & Condition": [
        "OverallQual",
        "OverallCond",
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
    ],
    "Basement": [
        "TotalBsmtSF",
        "BsmtFinSF1",
        "BsmtFinSF2",
        "BsmtUnfSF",
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
    ],
    "Other": [
        "MasVnrArea",
        "CentralAir_Y",
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

    # Text display for specific binary features
    if feat == "GarageType_Detchd":
        return {"type": "select", "options": ["No", "Yes"], "default": "Yes", "map": {"No": 0, "Yes": 1}}
    if feat == "CentralAir_Y":
        return {"type": "select", "options": ["No", "Yes"], "default": "Yes", "map": {"No": 0, "Yes": 1}}

    if "_" in feat and feat not in {"1stFlrSF", "2ndFlrSF", "3SsnPorch"}:
        return {"type": "select", "options": [0, 1], "default": 1}

    if feat in {"YearBuilt", "YearRemodAdd"}:
        return {"type": "int", "min": 1850, "step": 1, "default": 2000}
    
    if feat == "GarageYrBlt":
        return {"type": "int", "min": 1850, "step": 1, "default": 2000}

    if feat == "MoSold":
        return {"type": "int", "min": 1, "step": 1, "default": 1}
    
    # Specific defaults for important features
    if feat == "OverallQual":
        return {"type": "int", "min": 1, "step": 1, "default": 7}
    if feat == "OverallCond":
        return {"type": "int", "min": 1, "step": 1, "default": 7}
    if feat == "GrLivArea":
        return {"type": "float", "min": 1.0, "step": 10.0, "default": 1500.0}
    if feat == "1stFlrSF":
        return {"type": "float", "min": 1.0, "step": 10.0, "default": 1000.0}
    if feat == "2ndFlrSF":
        return {"type": "float", "min": 1.0, "step": 10.0, "default": 500.0}
    if feat == "TotalBsmtSF":
        return {"type": "float", "min": 1.0, "step": 10.0, "default": 1000.0}
    if feat == "BsmtFinSF1":
        return {"type": "float", "min": 1.0, "step": 10.0, "default": 500.0}
    if feat == "BsmtFinSF2":
        return {"type": "float", "min": 1.0, "step": 10.0, "default": 100.0}
    if feat == "BsmtUnfSF":
        return {"type": "float", "min": 1.0, "step": 10.0, "default": 400.0}
    if feat == "GarageArea":
        return {"type": "float", "min": 1.0, "step": 10.0, "default": 400.0}
    if feat == "LotArea":
        return {"type": "float", "min": 1.0, "step": 100.0, "default": 8000.0}
    if feat == "LotFrontage":
        return {"type": "float", "min": 1.0, "step": 10.0, "default": 60.0}
    if feat == "TotRmsAbvGrd":
        return {"type": "int", "min": 1, "step": 1, "default": 6}
    if feat == "GarageCars":
        return {"type": "int", "min": 1, "step": 1, "default": 2}
    if feat == "FullBath":
        return {"type": "int", "min": 1, "step": 1, "default": 2}
    if feat == "HalfBath":
        return {"type": "int", "min": 1, "step": 1, "default": 1}
    if feat in ["BsmtFullBath", "BsmtHalfBath"]:
        return {"type": "int", "min": 1, "step": 1, "default": 1}
    if feat == "BedroomAbvGr":
        return {"type": "int", "min": 1, "step": 1, "default": 3}
    if feat == "KitchenAbvGr":
        return {"type": "int", "min": 1, "step": 1, "default": 1}
    if feat == "OpenPorchSF":
        return {"type": "float", "min": 1.0, "step": 10.0, "default": 40.0}
    if feat == "WoodDeckSF":
        return {"type": "float", "min": 1.0, "step": 10.0, "default": 50.0}
    if feat in ["EnclosedPorch", "3SsnPorch", "ScreenPorch"]:
        return {"type": "float", "min": 1.0, "step": 10.0, "default": 10.0}
    if feat == "MasVnrArea":
        return {"type": "float", "min": 1.0, "step": 10.0, "default": 100.0}

    if any(k in f for k in ["qual", "cond", "cars", "rooms", "bedroom", "kitchen", "bath"]):
        return {"type": "int", "min": 1, "step": 1, "default": 1}

    if "ratio" in f:
        return {"type": "float", "min": 0.01, "step": 0.1, "default": 0.5}

    if any(k in f for k in ["area", "sf", "frontage"]):
        return {"type": "float", "min": 1.0, "step": 10.0, "default": 100.0}

    return {"type": "float", "min": 1.0, "step": 1.0, "default": 1.0}


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
        if v < 1850:
            return f"Too early for {int(v)}. Min: 1850"
        if v > 2026:
            return f"Future year {int(v)}. Max: 2026"
    
    if feat == "YearRemodAdd":
        if v < 1850:
            return f"Too early for {int(v)}. Min: 1850"
        if v > 2026:
            return f"Future year {int(v)}. Max: 2026"
        yb = all_values.get("YearBuilt", 0)
        if yb and v < yb:
            return f"Cannot be before YearBuilt ({int(yb)})"
    
    if feat == "GarageYrBlt":
        if v < 1850:
            return f"Too early for {int(v)}. Min: 1850"
        if v > 2026:
            return f"Future year {int(v)}. Max: 2026"
        yb = all_values.get("YearBuilt", 0)
        if yb and v < yb:
            return f"Cannot be before YearBuilt ({int(yb)})"
    
    # MoSold validation
    if feat == "MoSold":
        if not (1 <= v <= 12):
            return f"Must be 1-12"
    
    # Area range validations
    if feat == "GrLivArea":
        if v > 10000:
            return f"Cannot exceed 10,000 sqft"
    
    if feat == "1stFlrSF":
        if v > 8000:
            return f"Cannot exceed 8,000 sqft"
    
    if feat == "TotalBsmtSF":
        if v > 8000:
            return f"Cannot exceed 8,000 sqft"
        # Check consistency with basement components
        bsmt_fin1 = all_values.get("BsmtFinSF1", 0)
        bsmt_fin2 = all_values.get("BsmtFinSF2", 0)
        bsmt_unf = all_values.get("BsmtUnfSF", 0)
        total_components = bsmt_fin1 + bsmt_fin2 + bsmt_unf
        if total_components > v + 10:  # Allow small tolerance
            return f"Total ({v:.0f}) < components ({total_components:.0f})"
    
    # GarageCars validation
    if feat == "GarageCars":
        if v > 10:
            return f"Too many (max 10)"
    
    if feat == "GarageArea":
        if v > 2000:
            return f"Cannot exceed 2,000 sqft"
    
    # LotArea validation
    if feat == "LotArea":
        if v > 100000:
            return f"Cannot exceed 100,000 sqft"
    
    # LotFrontage validation
    if feat == "LotFrontage":
        if v > 500:
            return f"Cannot exceed 500 feet"
    
    if feat in ["BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF"]:
        if v > 5000:
            return f"Cannot exceed 5,000 sqft"
    
    # Room validations
    if feat == "TotRmsAbvGrd":
        if v > 50:
            return f"Too many (max 50)"
    
    if feat == "BedroomAbvGr":
        rooms = all_values.get("TotRmsAbvGrd", 999)
        if rooms and v > rooms:
            return f"Cannot exceed TotRmsAbvGrd ({rooms:.0f})"
    
    if feat in ["KitchenAbvGr", "FullBath", "HalfBath", "BsmtFullBath", "BsmtHalfBath"]:
        if v > 10:
            return f"Too many (max 10)"
    
    # 2ndFlrSF should not exceed reasonable limits
    if feat == "2ndFlrSF":
        if v > 5000:
            return f"Cannot exceed 5,000 sqft"
    
    # Porch area validations
    if feat == "WoodDeckSF":
        if v > 2000:
            return f"Cannot exceed 2,000 sqft"
    
    if feat == "OpenPorchSF":
        if v > 2000:
            return f"Cannot exceed 2,000 sqft"
    
    porch_fields = ["EnclosedPorch", "3SsnPorch", "ScreenPorch"]
    if feat in porch_fields:
        if v > 2000:
            return f"Cannot exceed 2,000 sqft"
    
    # MasVnrArea validation
    if feat == "MasVnrArea":
        if v > 2000:
            return f"Cannot exceed 2,000 sqft"
    
    return None


def render_feature_input(name: str, spec: dict, user_values: dict):
    """Render input with inline validation"""
    key = f"feat_{name}"
    
    # Get current value from session state
    if key in st.session_state:
        current_val = st.session_state[key]
    else:
        current_val = spec.get("default", spec.get(1))
    
    # For mapped values, validate the numeric value
    if "map" in spec and spec["type"] == "select":
        # Convert display value to numeric for validation
        numeric_val = spec["map"].get(current_val, current_val)
        error_msg = validate_single_field(name, numeric_val, user_values)
    else:
        error_msg = validate_single_field(name, current_val, user_values)
    
    # Render input
    if spec["type"] == "select":
        options = spec["options"]
        default = spec.get("default", options[0])
        idx = options.index(default) if default in options else 0
        val = st.selectbox(name, options=options, index=idx, key=key)
        # Convert text to numeric if mapping exists
        if "map" in spec:
            val = spec["map"][val]
    else:
        is_int = spec["type"] == "int"
        min_val = spec.get("min", None)
        val = st.number_input(
            name,
            min_value=min_val,
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

    X = pd.DataFrame([[0.0] * len(feature_names_list)], columns=feature_names_list)

    # First, copy all direct values
    for k, v in row.items():
        if k in X.columns:
            try:
                X.loc[0, k] = float(v)
            except Exception:
                pass

    # Only calculate engineered features if the necessary raw features exist
    if "QualityCond" in X.columns and "OverallQual" in row and "OverallCond" in row:
        X.loc[0, "QualityCond"] = float(row.get("OverallQual", 0)) * float(row.get("OverallCond", 0))

    if "TotalSF" in X.columns and "1stFlrSF" in row and "2ndFlrSF" in row and "TotalBsmtSF" in row:
        X.loc[0, "TotalSF"] = float(row.get("1stFlrSF", 0)) + float(row.get("2ndFlrSF", 0)) + float(row.get("TotalBsmtSF", 0))

    if "TotalBath" in X.columns and any(k in row for k in ["FullBath", "HalfBath", "BsmtFullBath", "BsmtHalfBath"]):
        X.loc[0, "TotalBath"] = (
            float(row.get("FullBath", 0)) + float(row.get("HalfBath", 0))
            + float(row.get("BsmtFullBath", 0)) + float(row.get("BsmtHalfBath", 0))
        )

    if "TotalPorchSF" in X.columns and any(k in row for k in ["WoodDeckSF", "OpenPorchSF", "EnclosedPorch", "3SsnPorch", "ScreenPorch"]):
        X.loc[0, "TotalPorchSF"] = (
            float(row.get("WoodDeckSF", 0)) + float(row.get("OpenPorchSF", 0))
            + float(row.get("EnclosedPorch", 0)) + float(row.get("3SsnPorch", 0)) + float(row.get("ScreenPorch", 0))
        )

    if "BsmtFinishedRatio" in X.columns and "BsmtFinSF1" in row and "TotalBsmtSF" in row:
        total_bsmt = float(row.get("TotalBsmtSF", 0))
        if total_bsmt > 0:
            bsmt_fin = float(row.get("BsmtFinSF1", 0)) + float(row.get("BsmtFinSF2", 0))
            X.loc[0, "BsmtFinishedRatio"] = bsmt_fin / total_bsmt

    if "AreaPerRoom" in X.columns and "GrLivArea" in row and "TotRmsAbvGrd" in row:
        tot_rooms = float(row.get("TotRmsAbvGrd", 0))
        if tot_rooms > 0:
            X.loc[0, "AreaPerRoom"] = float(row.get("GrLivArea", 0)) / tot_rooms

    if "TotalRooms" in X.columns and any(k in row for k in ["TotRmsAbvGrd", "BedroomAbvGr", "KitchenAbvGr"]):
        X.loc[0, "TotalRooms"] = float(row.get("TotRmsAbvGrd", 0)) + float(row.get("BedroomAbvGr", 0)) + float(row.get("KitchenAbvGr", 0))

    if "IsRemodeled" in X.columns and "YearRemodAdd" in row and "YearBuilt" in row:
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
    
    # Required fields (only those in the 25 features)
    required = ["OverallQual", "GrLivArea", "1stFlrSF", "TotalBsmtSF", "YearBuilt"]
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

st.success(f"Model loaded: {len(feature_names)} features (Top 25)")

# Top 25 features that the model uses
TOP_25_FEATURES = set(feature_names)

# Base features needed to calculate the Top 25 engineered features
# These MUST be collected from the user even if not in Top 25
NEEDED_BASE = {
    # For QualityCond = OverallQual × OverallCond
    "OverallQual", "OverallCond",
    
    # For TotalSF = 1stFlrSF + 2ndFlrSF + TotalBsmtSF
    "1stFlrSF", "2ndFlrSF", "TotalBsmtSF",
    
    # For TotalBath = FullBath + HalfBath + BsmtFullBath + BsmtHalfBath
    "FullBath", "HalfBath", "BsmtFullBath", "BsmtHalfBath",
    
    # For TotalPorchSF = WoodDeckSF + OpenPorchSF + EnclosedPorch + 3SsnPorch + ScreenPorch
    "WoodDeckSF", "OpenPorchSF", "EnclosedPorch", "3SsnPorch", "ScreenPorch",
    
    # For BsmtFinishedRatio = (BsmtFinSF1 + BsmtFinSF2) / TotalBsmtSF
    "BsmtFinSF1", "BsmtFinSF2",
    
    # For AreaPerRoom = GrLivArea / TotRmsAbvGrd
    "GrLivArea", "TotRmsAbvGrd",
    
    # For TotalRooms = TotRmsAbvGrd + BedroomAbvGr + KitchenAbvGr
    "BedroomAbvGr", "KitchenAbvGr",
    
    # For IsRemodeled = (YearRemodAdd != YearBuilt)
    "YearBuilt", "YearRemodAdd",
    
    # Other direct features in Top 25
    "BsmtUnfSF", "LotArea", "LotFrontage", "MasVnrArea",
    "GarageCars", "GarageArea", "GarageYrBlt", "GarageType_Detchd",
    "CentralAir_Y", "MoSold",
}

# Display ALL base features needed for Top 25 calculation
DISPLAY_SET = TOP_25_FEATURES | NEEDED_BASE

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
        st.error("Validation Errors Found:")
        for error in errors:
            st.error(f"• {error}")
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