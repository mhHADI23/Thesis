import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.inspection import permutation_importance
import joblib

model = joblib.load('best_model.pkl')
X_train = pd.read_csv('X_train.csv')
X_test = pd.read_csv('X_test.csv')
y_train = pd.read_csv('y_train.csv').squeeze()
y_test = pd.read_csv('y_test.csv').squeeze()

y_pred = model.predict(X_test)
y_pred_train = model.predict(X_train)

r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2_train = r2_score(y_train, y_pred_train)

residuals = y_test.values - y_pred

print("═" * 50)
print("  MODEL EVALUATION — Gradient Boosting")
print("═" * 50)
print(f"  Train R²  : {r2_train:.4f}")
print(f"  Test  R²  : {r2:.4f}")
print(f"  Test  RMSE: {rmse:.4f} mm")
print(f"  Test  MAE : {mae:.4f} mm")
print(f"  Overfit gap (train-test R²): {r2_train - r2:.4f}")
print("═" * 50)

print("\nCalculating permutation importances...")
perm = permutation_importance(
    model, X_test, y_test,
    n_repeats=30,
    random_state=42,
    n_jobs=-1
)

perm_df = pd.DataFrame({
    'feature': X_test.columns,
    'importance': perm.importances_mean,
    'std': perm.importances_std
}).sort_values('importance', ascending=False)

print("\nTop 20 features by permutation importance:")
print(f"  {'Feature':<45} {'Importance':>10}  {'Std':>6}")
print("  " + "-" * 65)

for _, row in perm_df.head(20).iterrows():
    print(f"  {row['feature']:<45} {row['importance']:>10.4f}  {row['std']:>6.4f}")

fig = plt.figure(figsize=(16, 14))
fig.suptitle('ZOI Prediction — Gradient Boosting: Evaluation Report',
             fontsize=14, fontweight='bold', y=0.98)

gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.32)

ACCENT = '#4C6EF5'
CORAL = '#E8593C'
GRAY = '#888780'
BG = '#F8F8F6'

ax1 = fig.add_subplot(gs[0, 0])
ax1.scatter(y_test, y_pred, color=ACCENT, alpha=0.65, s=45,
            edgecolors='white', linewidths=0.5)
lims = [min(y_test.min(), y_pred.min()) - 1,
        max(y_test.max(), y_pred.max()) + 1]
ax1.plot(lims, lims, '--', color=CORAL, linewidth=1.5, label='Perfect fit')
ax1.set_xlabel('Actual ZOI (mm)', fontsize=11, fontweight='bold')
ax1.set_ylabel('Predicted ZOI (mm)', fontsize=11, fontweight='bold')
ax1.set_title(f'Predicted vs Actual (R²={r2:.3f})', fontsize=12, fontweight='bold')
ax1.legend(fontsize=11)
ax1.set_facecolor(BG)
ax1.text(0.05, 0.92, f'RMSE={rmse:.2f}  MAE={mae:.2f}',
         transform=ax1.transAxes, fontsize=11)

ax2 = fig.add_subplot(gs[0, 1])
ax2.scatter(y_pred, residuals, color=ACCENT, alpha=0.65, s=45,
            edgecolors='white', linewidths=0.5)
ax2.axhline(0, color=CORAL, linewidth=1.5, linestyle='--')
ax2.axhline(rmse, color=GRAY, linewidth=0.8, linestyle=':', alpha=0.7)
ax2.axhline(-rmse, color=GRAY, linewidth=0.8, linestyle=':', alpha=0.7)
ax2.set_xlabel('Predicted ZOI (mm)', fontsize=11, fontweight='bold')
ax2.set_ylabel('Residual (Actual − Predicted)', fontsize=11, fontweight='bold')
ax2.set_title('Residuals vs Predicted', fontsize=12, fontweight='bold')
ax2.set_facecolor(BG)
ax2.text(0.05, 0.93, f'±RMSE band shown',
         transform=ax2.transAxes, fontsize=8, color=GRAY)

ax3 = fig.add_subplot(gs[1, 0])
ax3.hist(residuals, bins=20, color=ACCENT, edgecolor='white',
         linewidth=0.5, alpha=0.85)
ax3.axvline(0, color=CORAL, linewidth=1.5, linestyle='--', label='Zero error')
ax3.axvline(residuals.mean(), color=GRAY, linewidth=1.2, linestyle='-',
            label=f'Mean={residuals.mean():.2f}')
ax3.set_xlabel('Residual (mm)', fontsize=11)
ax3.set_ylabel('Count', fontsize=11)
ax3.set_title('Residual Distribution', fontsize=12, fontweight='bold')
ax3.legend(fontsize=9)
ax3.set_facecolor(BG)

ax4 = fig.add_subplot(gs[1, 1])
top15 = perm_df.head(15).iloc[::-1]

def clean_label(name):
    for prefix in ['NPs_', 'Species_', 'Family_', 'Shape_',
                   'Coating_', 'Class_', 'Type(Gram( +/-)_']:
        if name.startswith(prefix):
            return name.replace(prefix, f'[{prefix.rstrip("_")}] ')
    return name

labels = [clean_label(f) for f in top15['feature']]
colors = [ACCENT if top15['importance'].iloc[i] > 0 else CORAL
          for i in range(len(top15))]

ax4.barh(labels, top15['importance'], xerr=top15['std'],
         color=colors, alpha=0.85, edgecolor='white',
         linewidth=0.5, capsize=3, error_kw={'elinewidth': 0.8})

ax4.axvline(0, color=GRAY, linewidth=0.8)
ax4.set_xlabel('Permutation Importance (↑ = more important)', fontsize=10)
ax4.set_title('Top 15 Feature Importances', fontsize=12, fontweight='bold')
ax4.set_facecolor(BG)
ax4.tick_params(axis='y', labelsize=8)

subplot_axes = [ax1, ax2, ax3, ax4]
subplot_names = [
    'subplot_1_actual_vs_predicted.png',
    'subplot_2_residuals_vs_predicted.png',
    'subplot_3_residual_distribution.png',
    'subplot_4_feature_importances.png'
]

fig.canvas.draw()
renderer = fig.canvas.get_renderer()

# for ax, name in zip(subplot_axes, subplot_names):
#     extent = ax.get_tightbbox(renderer).transformed(fig.dpi_scale_trans.inverted())
#     fig.savefig(
#         name,
#         dpi=600,
#         bbox_inches=extent.expanded(1.08, 1.12),
#         facecolor='white',
#         pad_inches=0.3
#     )
#     print(f"Saved subplot: {name}")

plt.savefig('evaluation_report.png', dpi=160,
            bbox_inches='tight', facecolor='white')
plt.close()

print(f"\nPlot saved: evaluation_report.png")

perm_df.to_csv('feature_importance.csv', index=False)
print("Saved: feature_importance.csv")

print("\n── Summary ──")
print(f"  The Gradient Boosting model explains {r2*100:.1f}% of ZOI variance (R²={r2:.3f}).")
print(f"  Predictions deviate by ±{rmse:.2f} mm on average (RMSE).")
print(f"  The train-test R² gap is {r2_train-r2:.3f} — "
      f"{'minimal overfitting' if r2_train-r2 < 0.15 else 'some overfitting, consider regularisation'}.")

top3 = perm_df.head(3)['feature'].tolist()
print(f"  Top predictors: {', '.join(top3)}")
