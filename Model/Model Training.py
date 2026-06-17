import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor

X_train = pd.read_csv('X_train.csv')
X_test = pd.read_csv('X_test.csv')
y_train = pd.read_csv('y_train.csv').squeeze()
y_test = pd.read_csv('y_test.csv').squeeze()
print(f"Train: {X_train.shape}  |  Test: {X_test.shape}")

models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(alpha=1.0),
    'Random Forest': RandomForestRegressor(
        n_estimators=200,
        max_depth=None,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    ),
    'Gradient Boosting': GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        random_state=42
    ),
    'XGBoost': XGBRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        random_state=42,
        n_jobs=-1
    ),
}

kf = KFold(n_splits=5, shuffle=True, random_state=42)

print("\n── 5-Fold Cross-Validation (on training set) ──")
print(f"{'Model':<25} {'R²':>8} {'RMSE':>8} {'MAE':>8}")
print("-" * 52)

cv_results = {}

for name, model in models.items():
    r2 = cross_val_score(model, X_train, y_train, cv=kf, scoring='r2')
    rmse = cross_val_score(
        model, X_train, y_train, cv=kf,
        scoring='neg_root_mean_squared_error'
    )
    mae = cross_val_score(
        model, X_train, y_train, cv=kf,
        scoring='neg_mean_absolute_error'
    )

    cv_results[name] = {
        'R2_mean': r2.mean(),
        'R2_std': r2.std(),
        'RMSE_mean': -rmse.mean(),
        'RMSE_std': -rmse.std(),
        'MAE_mean': -mae.mean(),
        'MAE_std': -mae.std(),
    }

    print(f"{name:<25} {r2.mean():>7.3f}  {-rmse.mean():>7.3f}  {-mae.mean():>7.3f}")

print("\n── Test Set Evaluation ──")
print(f"{'Model':<25} {'R²':>8} {'RMSE':>8} {'MAE':>8}")
print("-" * 52)

test_results = {}
trained_models = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)

    test_results[name] = {
        'R2': r2,
        'RMSE': rmse,
        'MAE': mae
    }

    trained_models[name] = model

    print(f"{name:<25} {r2:>7.3f}  {rmse:>7.3f}  {mae:>7.3f}")

best_name = max(test_results, key=lambda k: test_results[k]['R2'])
best_model = trained_models[best_name]

print(f"\nBest model: {best_name}  (R²={test_results[best_name]['R2']:.3f})")

if hasattr(best_model, 'feature_importances_'):
    importances = pd.Series(best_model.feature_importances_, index=X_train.columns)
    top20 = importances.sort_values(ascending=False).head(20)

    print(f"\nTop 20 feature importances ({best_name}):")
    for feat, imp in top20.items():
        bar = '█' * int(imp * 200)
        print(f"  {feat:<45} {imp:.4f}  {bar}")

cv_df = pd.DataFrame(cv_results).T
cv_df.index.name = 'Model'
cv_df.to_csv('cv_results.csv')

test_df = pd.DataFrame(test_results).T
test_df.index.name = 'Model'
test_df.to_csv('test_results.csv')

joblib.dump(best_model, 'best_model.pkl')

print(f"\nSaved: cv_results.csv, test_results.csv, best_model.pkl")
