import pandas as pd
import numpy as np
import re

df = pd.read_excel('Raw Dataset.xlsx')
print(f"Original shape: {df.shape}")

df.drop(columns=['paper No', 'DOI'], inplace=True)

df.dropna(subset=['ZOI'], inplace=True)
print(f"After dropping missing ZOI: {df.shape}")

df['Dose'] = df['Dose'].apply(
    lambda val: float(re.search(r'[\d.]+', str(val)).group())
    if pd.notna(val) and re.search(r'[\d.]+', str(val)) else np.nan
)
print(f"Missing Dose after parsing: {df['Dose'].isna().sum()}")

df['Core_size'] = pd.to_numeric(df['Core_size'].replace('-', np.nan), errors='coerce')
print(f"Missing Core_size after coercion: {df['Core_size'].isna().sum()}")

shape_map = {
    'sphreical':       'spherical',
    'spherical':       'spherical',
    'quasi-spherical': 'quasi-spherical',
    'irregular':       'irregular',
    'rod':             'rod',
    'platelet':        'platelet',
    'Tetragonal':      'tetragonal',
    'petal':           'petal',
    'star':            'star',
    'asymmetrical':    'asymmetrical',
    'rhombohedral':    'rhombohedral',
    'cubic':           'cubic',
}
df['Shape'] = df['Shape'].map(shape_map)

shape_mode = df['Shape'].mode()[0]
df['Shape'] = df['Shape'].fillna(shape_mode)
print(f"Shape value counts after cleaning:\n{df['Shape'].value_counts()}")

df['Coating'] = df['Coating'].str.lower().str.strip()

coating_mode = df['Coating'].mode()[0]
df['Coating'] = df['Coating'].fillna(coating_mode)
print(f"Coating value counts:\n{df['Coating'].value_counts()}")

df['Type(Gram( +/-)'] = df['Type(Gram( +/-)'].replace('+', 'p')
print(f"Gram type value counts:\n{df['Type(Gram( +/-)'].value_counts()}")

duration_median = df['Duration'].median()
df['Duration'] = df['Duration'].fillna(duration_median)
print(f"Duration median used for imputation: {duration_median}")

df['Dose'] = df.groupby('NPs')['Dose'].transform(lambda x: x.fillna(x.median()))
df['Dose'] = df['Dose'].fillna(df['Dose'].median())

df['Core_size'] = df.groupby('NPs')['Core_size'].transform(lambda x: x.fillna(x.median()))
df['Core_size'] = df['Core_size'].fillna(df['Core_size'].median())

print(f"\nFinal null counts:\n{df.isnull().sum()}")
print(f"\nFinal shape: {df.shape}")
print(f"\nColumn dtypes:\n{df.dtypes}")
print(f"\nSample cleaned rows:")
print(df.head(5).to_string())

print("\n" + "=" * 55)
print("UNIQUE ATTRIBUTE COUNTS PER COLUMN")
print("=" * 55)
for col in df.columns:
    n_unique = df[col].nunique()
    print(f"\n[{col}]  —  {n_unique} unique value(s):")
    if df[col].dtype == object or df[col].nunique() <= 20:
        print(df[col].value_counts(dropna=False).to_string())
    else:
        print(f"  min={df[col].min():.4g},  max={df[col].max():.4g},  "
              f"mean={df[col].mean():.4g},  std={df[col].std():.4g}")
print("=" * 55)

df.to_csv('Data_Cleaned.csv', index=False)
print("\nSaved: Data_Cleaned.csv")