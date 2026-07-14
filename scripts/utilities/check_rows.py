import pandas as pd
df_15m = pd.read_csv('dataset/processed/marts_features_15m.csv')
print(f"15m: {df_15m.shape}")
df_1h = pd.read_csv('dataset/processed/marts_features.csv')
print(f"1h: {df_1h.shape}")
df_raw = pd.read_csv('dataset/raw/final_dataset.csv')
print(f"Raw: {df_raw.shape}")
