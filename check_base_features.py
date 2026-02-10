import joblib

# Load scaler to see all features it needs
scaler = joblib.load('house_scaler.pkl')
scaler_features = scaler.feature_names_in_.tolist()

# Load model features (top 25)
with open('feature_names.txt', 'r', encoding='utf-8') as f:
    model_features = [line.strip() for line in f if line.strip()]

print("模型的Top 25特征需要的底层特征：")
print("=" * 80)

# Features that need to be computed
engineered_features = {
    'TotalSF': ['1stFlrSF', '2ndFlrSF', 'TotalBsmtSF'],
    'TotalBath': ['FullBath', 'HalfBath', 'BsmtFullBath', 'BsmtHalfBath'],
    'TotalPorchSF': ['WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch'],
    'QualityCond': ['OverallQual', 'OverallCond'],
    'BsmtFinishedRatio': ['BsmtFinSF1', 'BsmtFinSF2', 'TotalBsmtSF'],
    'AreaPerRoom': ['GrLivArea', 'TotRmsAbvGrd'],
    'TotalRooms': ['TotRmsAbvGrd', 'BedroomAbvGr', 'KitchenAbvGr'],
    'IsRemodeled': ['YearRemodAdd', 'YearBuilt']
}

# Check which features in top 25 are engineered
base_features_needed = set()
for feat in model_features:
    if feat in engineered_features:
        print(f"\n{feat} (engineered):")
        for base in engineered_features[feat]:
            print(f"  - {base}")
            base_features_needed.add(base)
    else:
        base_features_needed.add(feat)

print("\n" + "=" * 80)
print(f"\n总共需要 {len(base_features_needed)} 个底层特征：")
print("-" * 80)
for i, feat in enumerate(sorted(base_features_needed), 1):
    in_scaler = "✅" if feat in scaler_features else "❌"
    print(f"{i:2d}. {feat:25s} {in_scaler}")
