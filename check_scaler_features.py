import joblib
import pandas as pd

# Load scaler
scaler = joblib.load('house_scaler.pkl')

print("Scaler的特征列表：")
print("=" * 80)
feature_names_in_scaler = scaler.feature_names_in_.tolist()
print(f"总共 {len(feature_names_in_scaler)} 个特征")
print()

# Check if engineered features are in scaler
engineered_features = ['TotalSF', 'TotalBath', 'TotalPorchSF', 'QualityCond', 
                       'BsmtFinishedRatio', 'AreaPerRoom', 'TotalRooms', 'IsRemodeled']

print("Engineered Features 是否在 scaler 中：")
print("-" * 80)
for feat in engineered_features:
    status = "✅ 存在" if feat in feature_names_in_scaler else "❌ 不存在"
    print(f"{feat:20s} {status}")
print()

# Load model features (top 25)
with open('feature_names.txt', 'r', encoding='utf-8') as f:
    model_features = [line.strip() for line in f if line.strip()]

print(f"\n模型使用的 Top 25 特征：")
print("=" * 80)
for i, feat in enumerate(model_features, 1):
    in_scaler = "✅" if feat in feature_names_in_scaler else "❌"
    print(f"{i:2d}. {feat:25s} {in_scaler}")

print()
print("=" * 80)
missing_in_scaler = [f for f in model_features if f not in feature_names_in_scaler]
if missing_in_scaler:
    print(f"❌ 警告：{len(missing_in_scaler)} 个模型特征不在 scaler 中：")
    for f in missing_in_scaler:
        print(f"   - {f}")
    print()
    print("需要重新训练模型！运行 Notebook Section 3.1 → 4.3 → 4.4 → 4.5 → 7")
else:
    print("✅ 所有模型特征都在 scaler 中！")
