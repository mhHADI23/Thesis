import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib

df = pd.read_csv('Data_Cleaned.csv')
print(f"Loaded: {df.shape[0]} rows, {df.shape[1]} columns")

df['Species'] = df['Species'].str.strip()

df['Dose_log'] = np.log1p(df['Dose'])
df['Core_size_log'] = np.log1p(df['Core_size'])

print(f"\nDose        range: {df['Dose'].min()} – {df['Dose'].max()}")
print(f"Dose_log    range: {df['Dose_log'].min():.2f} – {df['Dose_log'].max():.2f}")
print(f"Core_size   range: {df['Core_size'].min()} – {df['Core_size'].max()}")
print(f"Core_size_log range: {df['Core_size_log'].min():.2f} – {df['Core_size_log'].max():.2f}")

cat_cols = ['NPs', 'Coating', 'Shape', 'Type(Gram( +/-)', 'Class', 'Family', 'Species']
df_enc = pd.get_dummies(df, columns=cat_cols, drop_first=False)
print(f"\nShape after one-hot encoding: {df_enc.shape}")

df_enc.to_csv('encoded_dataset.csv', index=False)

X = df_enc.drop(columns=['ZOI', 'Dose', 'Core_size'])
y = df_enc['ZOI']

num_cols = ['Core_size_log', 'Dose_log', 'Duration']
onehot_cols = [c for c in X.columns if c not in num_cols]
X = X[num_cols + onehot_cols]

print(f"\nFeature matrix X: {X.shape[1]} features × {X.shape[0]} rows")
print(f"Target y (ZOI):   min={y.min()}, max={y.max()}, mean={y.mean():.2f}")
print(f"\nFeatures breakdown:")
print(f"  Numerical (pre-scale) : {num_cols}")
print(f"  One-hot encoded       : {len(onehot_cols)} columns")

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=df['NPs']
)

print(f"\nTrain: {len(X_train)} rows  \nTest: {len(X_test)} rows")

train_idx = X_train.index
test_idx = X_test.index

print("\nNP type distribution (train vs test):")
for np_type in sorted(df['NPs'].unique()):
    n_train = (df.loc[train_idx, 'NPs'] == np_type).sum()
    n_test = (df.loc[test_idx, 'NPs'] == np_type).sum()
    print(f"  {np_type:<6}: train={n_train}, test={n_test}")

X_train.to_csv('X_train.csv', index=False)
X_test.to_csv('X_test.csv', index=False)
y_train.to_csv('y_train.csv', index=False, header=True)
y_test.to_csv('y_test.csv', index=False, header=True)

scaler = StandardScaler()
X_train = X_train.copy()
X_test = X_test.copy()

X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test[num_cols] = scaler.transform(X_test[num_cols])

print(f"\nStandardScaler fitted on training data:")
for col, mean, std in zip(num_cols, scaler.mean_, scaler.scale_):
    print(f"  {col:<18}: mean={mean:.3f}, std={std:.3f}")

assert X_train.isnull().sum().sum() == 0, "NaNs in X_train"
assert X_test.isnull().sum().sum() == 0, "NaNs in X_test"
assert X_train.shape[1] == X_test.shape[1], "Column mismatch"

print(f"\nAll checks passed. Final feature count: {X_train.shape[1]}")

print(f"\nFull feature list ({len(X.columns)} total):")
for c in X.columns:
    print(f"  {c}")

X_train.to_csv('X_train.csv', index=False)
X_test.to_csv('X_test.csv', index=False)
y_train.to_csv('y_train.csv', index=False, header=True)
y_test.to_csv('y_test.csv', index=False, header=True)

joblib.dump(scaler, 'scaler.pkl')

print(f"\nSaved to :")
print("  X_train.csv, X_test.csv, y_train.csv, y_test.csv, scaler.pkl")

