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
.block-container { padding-top: 2.5rem; padding-bottom: 1.2rem; }
div[data-testid="stVerticalBlock"] { gap: 0.6rem; }

.main-header {
  font-size: 2.2rem; font-weight: 900; color: #1f77b4;
  text-align:center; margin: 1rem 0 0.5rem 0;
}
.sub-header {
  text-align:center; color:#666; margin-bottom: 1.2rem; font-size: 1rem;
}

.card {
  border: 1px solid #e0e0e0;
  border-radius: 12px;
  padding: 20px 24px;
  background: #fafafa;
  margin-bottom: 18px;
}
.card-title {
  font-weight: 700;
  font-size: 1.1rem;
  margin-bottom: 16px;
  color: #333;
  padding-bottom: 8px;
  border-bottom: 2px solid #e0e0e0;
}

.prediction-box {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  padding: 26px 22px;
  border-radius: 16px;
  color: white;
  text-align: center;
  margin: 12px 0 8px 0;
  box-shadow: 0 8px 25px rgba(0,0,0,0.10);
}
.muted { color:#777; font-size:0.9rem; }

label { font-size: 0.85rem !important; color: #444 !important; }

/* Style the predict button */
button[kind="formSubmit"] {
  background: linear-gradient(135deg, #ff8a00 0%, #ff5e00 100%) !important;
  color: white !important;
  font-weight: 700 !important;
  border: none !important;
  padding: 0.75rem 2rem !important;
  font-size: 1.1rem !important;
  border-radius: 8px !important;
  box-shadow: 0 4px 15px rgba(255, 138, 0, 0.4) !important;
  transition: all 0.3s ease !important;
}

button[kind="formSubmit"]:hover {
  background: linear-gradient(135deg, #ff9d1f 0%, #ff6f1f 100%) !important;
  transform: translateY(-2px) !important;
  box-shadow: 0 6px 20px rgba(255, 138, 0, 0.6) !important;
}
</style>
""",
    unsafe_allow_html=True,
)
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
        "AreaPerRoom",
        "TotalSF",
        "TotalBath",
    ],

    "Basement": [
        "TotalBsmtSF",
        "BsmtFinSF1",
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
        "TotalPorchSF",
    ],

    "Other": [
        "MasVnrArea",
        "CentralAir_Y",
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


@st.cache_data
def get_mock_houses():
    return pd.DataFrame(
        {
            "House_ID": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "Address": [
                "123 Oak Street",
                "456 Maple Avenue",
                "789 Pine Road",
                "321 Elm Drive",
                "654 Cedar Lane",
                "987 Birch Court",
                "147 Willow Way",
                "258 Spruce Circle",
                "369 Ash Boulevard",
                "741 Cherry Street",
            ],
            "OverallQual": [7, 5, 8, 6, 9, 4, 7, 6, 8, 5],
            "OverallCond": [5, 6, 7, 5, 8, 4, 6, 5, 7, 6],
            "GrLivArea": [1500, 1200, 2200, 1400, 2800, 900, 1600, 1300, 2400, 1100],
            "LotArea": [10000, 8000, 15000, 9000, 20000, 7000, 11000, 8500, 16000, 7500],
            "TotalBsmtSF": [800, 600, 1200, 700, 1500, 400, 850, 650, 1300, 550],
            "GarageArea": [400, 300, 600, 350, 800, 200, 450, 320, 650, 280],
            "GarageCars": [2, 1, 3, 2, 3, 1, 2, 1, 3, 1],
            "Fireplaces": [1, 0, 2, 1, 2, 0, 1, 0, 2, 0],
            "MasVnrArea": [100, 0, 300, 50, 500, 0, 150, 20, 400, 0],
            "YearBuilt": [2003, 1976, 2001, 1990, 2010, 1965, 1998, 1985, 2005, 1972],
            "YearRemodAdd": [2003, 1976, 2010, 2000, 2010, 1985, 2015, 1995, 2015, 1990],
            "1stFlrSF": [850, 1200, 1100, 1000, 1300, 900, 900, 1100, 1200, 1100],
            "2ndFlrSF": [650, 0, 1100, 400, 1500, 0, 700, 200, 1200, 0],
            "BsmtFinSF1": [700, 500, 1000, 600, 1200, 300, 750, 550, 1100, 450],
            "BsmtUnfSF": [100, 100, 200, 100, 300, 100, 100, 100, 200, 100],
            "LotFrontage": [65, 80, 70, 60, 90, 50, 68, 62, 75, 55],
            "OpenPorchSF": [35, 0, 150, 40, 200, 0, 45, 25, 180, 0],
            "TotRmsAbvGrd": [7, 6, 9, 6, 10, 5, 7, 6, 9, 5],
            "ActualPrice": [185000, 125000, 285000, 155000, 425000, 95000, 195000, 135000, 325000, 110000],
        }
    )


mock_houses = get_mock_houses()


ENGINEERED_SET = {
    "TotalSF", "QualityArea", "QualityCond", "TotalBath", "TotalPorchSF",
    "BsmtFinishedRatio", "AreaPerRoom", "TotalRooms",
    "HasGarage", "HasFireplace", "Has2ndFloor", "IsRemodeled", "HasPool", "HasBsmt"
}


def build_model_input(feature_names: list[str], raw: dict | pd.Series) -> pd.DataFrame:
    """Build one-row df aligned to feature_names; compute engineered features if needed."""
    row = raw.to_dict() if isinstance(raw, pd.Series) else dict(raw)
    X = pd.DataFrame([[0.0] * len(feature_names)], columns=feature_names)

    # copy base raw
    for k, v in row.items():
        if k in X.columns:
            try:
                X.loc[0, k] = float(v)
            except Exception:
                pass

    # engineered only if model expects them
    if "QualityArea" in X.columns:
        X.loc[0, "QualityArea"] = float(row.get("OverallQual", 0)) * float(row.get("GrLivArea", 0))

    if "QualityCond" in X.columns:
        X.loc[0, "QualityCond"] = float(row.get("OverallQual", 0)) * float(row.get("OverallCond", 0))

    if "TotalSF" in X.columns:
        X.loc[0, "TotalSF"] = (
            float(row.get("1stFlrSF", 0))
            + float(row.get("2ndFlrSF", 0))
            + float(row.get("TotalBsmtSF", 0))
        )

    if "TotalBath" in X.columns:
        X.loc[0, "TotalBath"] = (
            float(row.get("FullBath", 0))
            + float(row.get("HalfBath", 0))
            + float(row.get("BsmtFullBath", 0))
            + float(row.get("BsmtHalfBath", 0))
        )

    if "TotalPorchSF" in X.columns:
        X.loc[0, "TotalPorchSF"] = (
            float(row.get("WoodDeckSF", 0))
            + float(row.get("OpenPorchSF", 0))
            + float(row.get("EnclosedPorch", 0))
            + float(row.get("3SsnPorch", 0))
            + float(row.get("ScreenPorch", 0))
        )

    if "BsmtFinishedRatio" in X.columns:
        bsmt_fin = float(row.get("BsmtFinSF1", 0)) + float(row.get("BsmtFinSF2", 0))
        X.loc[0, "BsmtFinishedRatio"] = bsmt_fin / (float(row.get("TotalBsmtSF", 0)) + 1e-8)

    X.loc[0, "AreaPerRoom"] = float(row.get("GrLivArea", 0)) / (float(row.get("TotRmsAbvGrd", 0)) + 1e-8)

    if "HasGarage" in X.columns:
        X.loc[0, "HasGarage"] = 1.0 if float(row.get("GarageArea", 0)) > 0 else 0.0

    if "HasFireplace" in X.columns:
        X.loc[0, "HasFireplace"] = 1.0 if float(row.get("Fireplaces", 0)) > 0 else 0.0

    if "Has2ndFloor" in X.columns:
        X.loc[0, "Has2ndFloor"] = 1.0 if float(row.get("2ndFlrSF", 0)) > 0 else 0.0

    if "IsRemodeled" in X.columns:
        X.loc[0, "IsRemodeled"] = 1.0 if float(row.get("YearRemodAdd", 0)) != float(row.get("YearBuilt", 0)) else 0.0

    if "HasBsmt" in X.columns:
        X.loc[0, "HasBsmt"] = 1.0 if float(row.get("TotalBsmtSF", 0)) > 0 else 0.0

    return X


def predict_from_features_row(feature_row: pd.DataFrame):
    try:
        # 1) log1p skewed
        for col in skewed_features:
            if col in feature_row.columns:
                feature_row[col] = np.log1p(feature_row[col])

        # 2) one-hot if needed
        cat_cols = feature_row.select_dtypes(include=["object"]).columns
        if len(cat_cols) > 0:
            feature_row = pd.get_dummies(feature_row, columns=cat_cols, drop_first=True, dtype=int)

        # 3) align to scaler expected cols
        expected_cols = scaler.feature_names_in_.tolist()
        X_for_scaler = feature_row.reindex(columns=expected_cols, fill_value=0)

        # 4) scale ALL columns at once
        X_scaled_all = pd.DataFrame(
            scaler.transform(X_for_scaler),
            columns=expected_cols,
            index=feature_row.index
        )

        # 5) select model features
        Xs = X_scaled_all.reindex(columns=feature_names, fill_value=0)

        pred_log = model.predict(Xs)
        return float(np.exp(pred_log)[0])

    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        st.error(traceback.format_exc())
        return None

def predict_from_raw_row(model, scaler, feature_names, skewed_features, raw: dict | pd.Series) -> float | None:
    try:
        X = build_model_input(feature_names, raw)

        for f in skewed_features:
            if f in X.columns:
                X.loc[0, f] = np.log1p(X.loc[0, f])

        # Align to scaler's features, then scale, then select model features
        if scaler is not None:
            scaler_features = scaler.feature_names_in_.tolist()
            X_for_scaler = X.reindex(columns=scaler_features, fill_value=0)
            X_scaled_all = pd.DataFrame(scaler.transform(X_for_scaler), 
                                       columns=scaler_features, 
                                       index=X.index)
            Xs = X_scaled_all[feature_names]
        else:
            Xs = X
            
        pred_log = float(model.predict(Xs)[0])
        # Use exp not expm1 since target is log not log1p
        return float(np.exp(pred_log))
    except Exception as e:
        st.warning(f"Prediction error: {str(e)}")
        return None


@st.cache_data
def add_predicted_prices(houses_df: pd.DataFrame, _model, _scaler, _feature_names, _skewed_features):
    df = houses_df.copy()
    df["PredictedPrice"] = df.apply(
        lambda r: predict_from_features_row(r),
        axis=1,
    )
    return df

def infer_spec(feat: str):
    """
    Infer input widget type.
    - if feature looks binary (HasX / one-hot like A_B / contains '_') => selectbox [0,1]
    - else number_input with broad range
    """
    f = feat.lower()

    # binary-ish
    if feat.startswith("Has") or feat in {"IsRemodeled"}:
        return {"type": "select", "options": [0, 1], "default": 0}

    # common one-hot pattern: has underscore and not a known numeric like "1stFlrSF"
    if "_" in feat and feat not in {"1stFlrSF", "2ndFlrSF", "3SsnPorch"}:
        return {"type": "select", "options": [0, 1], "default": 0}

    # integer-ish
    if any(k in f for k in ["year", "qual", "cond", "cars", "fireplaces", "rooms"]):
        return {"type": "int", "min": 0, "max": 9999, "step": 1, "default": 0}

    # ratio-ish
    if "ratio" in f:
        return {"type": "float", "min": 0.0, "max": 1.0, "step": 0.01, "default": 0.0}

    # fallback float (reasonable max for house features)
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


def preprocess_for_model(df_raw, high_skew_features, scaler, X_train_cols, drop_id=True):
    df = df_raw.copy()

    if drop_id and "Id" in df.columns:
        df = df.drop(columns=["Id"])

    for col in ["PoolQC", "MiscFeature", "Alley", "Fence"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    if "LotFrontage" in df.columns and "Neighborhood" in df.columns:
        df["LotFrontage"] = df.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.median()))

    if "GarageYrBlt" in df.columns:
        df["GarageYrBlt"] = df["GarageYrBlt"].fillna(0)

    for col in ["GarageType", "GarageFinish", "GarageQual", "GarageCond"]:
        if col in df.columns:
            df[col] = df[col].fillna("None")

    if "FireplaceQu" in df.columns:
        df["FireplaceQu"] = df["FireplaceQu"].fillna("None")

    for col in ["BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2"]:
        if col in df.columns:
            df[col] = df[col].fillna("None")

    if "MasVnrArea" in df.columns:
        df["MasVnrArea"] = df["MasVnrArea"].fillna(0)
    if "MasVnrType" in df.columns:
        df["MasVnrType"] = df["MasVnrType"].fillna("None")
    if "Electrical" in df.columns:
        df["Electrical"] = df["Electrical"].fillna(df["Electrical"].mode()[0])

    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].fillna("None")
        else:
            df[col] = df[col].fillna(0)

    for col in high_skew_features:
        if col in df.columns:
            df[col] = np.log1p(df[col])

    df["TotalSF"] = df["1stFlrSF"] + df["2ndFlrSF"] + df["TotalBsmtSF"]

    # ipynb: HalfBath 没有 *0.5
    df["TotalBath"] = df["FullBath"] + df["HalfBath"] + df["BsmtFullBath"] + df["BsmtHalfBath"]

    df["TotalPorchSF"] = df["WoodDeckSF"] + df["OpenPorchSF"] + df["EnclosedPorch"] + df["3SsnPorch"] + df["ScreenPorch"]

    df["QualityCond"] = df["OverallQual"] * df["OverallCond"]

    df["BsmtFinishedRatio"] = (df["BsmtFinSF1"] + df["BsmtFinSF2"]) / (df["TotalBsmtSF"] + 1e-8)
    df["AreaPerRoom"] = df["GrLivArea"] / (df["TotRmsAbvGrd"] + 1e-8)

    df["TotalRooms"] = df["TotRmsAbvGrd"] + df["BedroomAbvGr"] + df["KitchenAbvGr"]
    df["IsRemodeled"] = (df["YearRemodAdd"] != df["YearBuilt"]).astype(int)

    # 5) One-hot
    cat_cols = df.select_dtypes(include=["object"]).columns
    df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True, dtype=int)

    # 6) Align to scaler expected columns
    expected = scaler.feature_names_in_.tolist()
    X_for_scaler = df_encoded.reindex(columns=expected, fill_value=0)

    # 7) Scale（scaler 是 fit 在 expected 这些列上）
    X_scaled_all = pd.DataFrame(
        scaler.transform(X_for_scaler),
        columns=expected,
        index=df.index
    )

    # 8) Select top 25 columns
    X_final = X_scaled_all.reindex(columns=X_train_cols, fill_value=0)
    return X_final


st.markdown('<div class="main-header">House Price Predictor</div>', unsafe_allow_html=True)

tab1, tab2 = st.tabs(["Predict (All Features)", "Train Dashboard"])


with tab1:
    st.markdown("### Enter ALL model features")

    if model is None or not feature_names:
        st.error("Model failed to load. Please ensure model files exist.")
    else:
        st.success(f"Model loaded: {len(feature_names)} features")

        # Form for all features
        with st.form("all_features_form", clear_on_submit=False):

            user_values = {}

            # Render feature groups
            for group_name, feats in FEATURE_GROUPS.items():

                feats = [f for f in feats if f in feature_names]
                if not feats:
                    continue

                # Display category title
                st.markdown(f"#### {group_name}")

                cols = st.columns(3, gap="large")

                for i, feat in enumerate(feats):
                    with cols[i % 3]:
                        spec = infer_spec(feat)
                        
                        # Set all defaults to 0
                        if spec["type"] == "select":
                            spec["default"] = 0
                        elif spec["type"] == "int":
                            spec["default"] = 0
                        else:  # float
                            spec["default"] = 0.0

                        user_values[feat] = render_feature_input(feat, spec)

                # Add divider between groups
                st.divider()

            submitted = st.form_submit_button("PREDICT PRICE", use_container_width=True)

        # Process form submission
        if submitted:
            X = pd.DataFrame([[0.0] * len(feature_names)], columns=feature_names)

            for feat in feature_names:
                key = f"feat_{feat}"
                if key in st.session_state:
                    try:
                        X.loc[0, feat] = float(st.session_state[key])
                    except Exception:
                        X.loc[0, feat] = 0.0

            price = predict_from_raw_row(model, scaler, feature_names, skewed_features, raw=user_values)

            if price is None:
                st.error("Prediction failed. Check scaler/model/feature alignment.")
            else:
                st.markdown(
                    f"""
                    <div class="prediction-box">
                        <div style="font-size: 3rem; font-weight: 900; margin: 8px 0;">
                            ${price:,.2f}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                lo, hi = price * 0.92, price * 1.08
                st.caption(f"Confidence Range (±8%): ${lo:,.2f} – ${hi:,.2f}")

                with st.expander("View all feature values used"):
                    st.dataframe(X.T, use_container_width=True)

with tab2:
    st.markdown("### Training Data Analysis")

    if model is None or not feature_names or scaler is None:
        st.error("Model/scaler not loaded. Cannot display training dashboard.")
    else:
        @st.cache_data
        def load_train_data(_model, _scaler, _feature_names, _skewed_features):
            train_df = pd.read_csv("train.csv")

            # Remove outliers (same as in good.ipynb Cell 57)
            outliers_mask = (train_df['GrLivArea'] > 4000) & (train_df['SalePrice'] < 300000)
            outliers = train_df[outliers_mask]
            if len(outliers) > 0:
                st.info(f"Removed {len(outliers)} outliers (GrLivArea > 4000 & SalePrice < 300000) - same as training")
            train_df = train_df[~outliers_mask].copy()

            y_actual = train_df["SalePrice"].values
            train_raw = train_df.drop(columns=["SalePrice"])

            X_train_final = preprocess_for_model(
                train_raw,
                high_skew_features=_skewed_features,
                scaler=_scaler,
                X_train_cols=_feature_names,
                drop_id=True,
            )

            pred_log = _model.predict(X_train_final)
            pred_price = np.exp(pred_log)

            train_df["PredictedPrice"] = pred_price
            train_df["Error"] = pred_price - y_actual
            train_df["ErrorPct"] = (train_df["Error"] / y_actual) * 100
            train_df["AbsErrorPct"] = np.abs(train_df["ErrorPct"])

            train_df["AccuracyScore"] = 100 - np.minimum(train_df["AbsErrorPct"], 100)
            train_df["ValueRatio"] = train_df["PredictedPrice"] / train_df["SalePrice"]
            train_df["ValueScore"] = np.clip(train_df["ValueRatio"] * 50, 0, 100)
            train_df["FitScore"] = 0.7 * train_df["AccuracyScore"] + 0.3 * train_df["ValueScore"]

            noise_threshold = train_df["AbsErrorPct"].quantile(0.95)
            train_df["IsNoise"] = train_df["AbsErrorPct"] > noise_threshold

            return train_df, float(noise_threshold)

        try:
            train_df, noise_threshold = load_train_data(model, scaler, feature_names, skewed_features)

            st.divider()
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.metric("Total Houses", len(train_df))
            with m2:
                st.metric("Avg Abs Error", f"{train_df['AbsErrorPct'].mean():.1f}%")
            with m3:
                noise_count = int(train_df["IsNoise"].sum())
                st.metric("Noise Count", f"{noise_count} ({noise_count/len(train_df)*100:.1f}%)")
            with m4:
                st.metric("Avg FitScore", f"{train_df['FitScore'].mean():.1f}/100")

            st.divider()
            st.info(f"Noise threshold: AbsErrorPct > {noise_threshold:.1f}%")

            display_cols = [
                "Id", "SalePrice", "PredictedPrice", "ErrorPct", "AbsErrorPct",
                "OverallQual", "GrLivArea", "YearBuilt", "Neighborhood"
            ]
            noise_data = train_df[train_df["IsNoise"]].sort_values("AbsErrorPct", ascending=False)
            st.dataframe(noise_data[display_cols].head(30), use_container_width=True, height=450)

        except Exception as e:
            st.error(f"Error loading training data: {e}")
            st.exception(e)

st.divider()
st.markdown(
    "<div style='text-align:center;color:#666;padding:8px 0 14px 0;'>House Price Predictor | Streamlit</div>",
    unsafe_allow_html=True,
)