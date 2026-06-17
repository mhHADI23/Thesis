import pandas as pd
import numpy as np
import joblib

model  = joblib.load('best_model.pkl')
scaler = joblib.load('scaler.pkl')
cols   = pd.read_csv('X_train.csv').columns.tolist()

#  Input Features
NPs       = 'Au'                   # Ag, Au, CuO, NiO, TiO2, ZnO
Coating   = 'uncoated'             # coated or uncoated
Core_size = 50                     # in nm
Shape     = 'rod'            # spherical, rod, irregular, platelet ...
Dose      = 1000                    # in ug/ml
Gram_type = 'n'                    # n = Gram-negative, p = Gram-positive
Class     = 'Gammaproteobacteria'  # Bacilli, Gammaproteobacteria, Betaproteobacteria
Family    = 'Enterobacteriaceae'
Species   = 'Escherichia coli'
Duration  = 20                     # in hours

row = {
    'Core_size_log': np.log1p(Core_size),
    'Dose_log':      np.log1p(Dose),
    'Duration':      float(Duration),
    f'NPs_{NPs}':                    1,
    f'Coating_{Coating}':            1,
    f'Shape_{Shape}':                1,
    f'Type(Gram( +/-)_{Gram_type}':  1,
    f'Class_{Class}':                1,
    f'Family_{Family}':              1,
    f'Species_{Species}':            1,
}

X = pd.DataFrame([row], columns=cols).fillna(0)
X[['Core_size_log','Dose_log','Duration']] = scaler.transform(
    X[['Core_size_log','Dose_log','Duration']]
)

zoi = model.predict(X)[0]
print(f"\nPredicted ZOI: {zoi:.2f} mm")
