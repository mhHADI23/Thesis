import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import shap
import joblib
import warnings
warnings.filterwarnings('ignore')


model   = joblib.load('best_model.pkl')
X_train = pd.read_csv('X_train.csv')
X_test  = pd.read_csv('X_test.csv')
y_test  = pd.read_csv('y_test.csv').squeeze()

print(f"Model : {type(model).__name__}")
print(f"X_test: {X_test.shape}")


print("\nComputing SHAP values...")
explainer   = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)
expected_value = float(explainer.expected_value.flat[0]) if hasattr(explainer.expected_value, "__len__") else float(explainer.expected_value)
print(f"SHAP values shape  : {shap_values.shape}")
print(f"Expected value (mean ZOI): {expected_value:.3f} mm")


def group_shap(shap_vals, feature_names):
    """Sum SHAP magnitudes of one-hot dummies back to original feature groups."""
    prefixes = ['NPs', 'Coating', 'Shape', 'Class', 'Family', 'Species',
                'Type(Gram( +/-)']
    numerical = ['Core_size_log', 'Dose_log', 'Duration']

    grouped = {}
    used = set()

    
    for col in numerical:
        if col in feature_names:
            idx = list(feature_names).index(col)
            grouped[col] = shap_vals[:, idx]
            used.add(col)

    
    for prefix in prefixes:
        cols = [c for c in feature_names if c.startswith(prefix + '_')]
        if cols:
            idxs = [list(feature_names).index(c) for c in cols]
            # Sum absolute contributions, preserve sign of dominant column
            grouped[prefix] = shap_vals[:, idxs].sum(axis=1)
            used.update(cols)

    return grouped

grouped_shap = group_shap(shap_values, X_test.columns)


mean_abs_shap = {k: np.abs(v).mean() for k, v in grouped_shap.items()}
mean_abs_shap = dict(sorted(mean_abs_shap.items(),
                             key=lambda x: x[1], reverse=True))

print("\n── Grouped Feature Importance (Mean |SHAP|) ──")
print(f"{'Feature':<25} {'Mean |SHAP|':>12}  Interpretation")
print("-" * 70)
interp = {
    'Dose_log':        'How much NP was applied',
    'Core_size_log':   'Physical size of the NP',
    'Species':         'Which bacterial species was tested',
    'NPs':             'Type of nanoparticle material',
    'Family':          'Bacterial family group',
    'Duration':        'Length of assay (hours)',
    'Shape':           'Morphology of the NP',
    'Class':           'Bacterial class (broad taxonomy)',
    'Coating':         'Surface coating of NP',
    'Type(Gram( +/-)': 'Gram-positive vs negative',
}
for feat, val in mean_abs_shap.items():
    print(f"  {feat:<23} {val:>10.4f}  {interp.get(feat,'')}")


mean_abs_raw = pd.Series(np.abs(shap_values).mean(axis=0),
                          index=X_test.columns).sort_values(ascending=False)
print("\n── Top 20 Individual Feature Importances ──")
for feat, val in mean_abs_raw.head(20).items():
    print(f"  {feat:<45} {val:.4f}")


from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score

perm = permutation_importance(model, X_test, y_test,
                               n_repeats=30, random_state=42, n_jobs=-1)
perm_df = pd.DataFrame({
    'feature': X_test.columns,
    'importance': perm.importances_mean,
    'std': perm.importances_std
}).sort_values('importance', ascending=False)

print("\n── Top 15 Permutation Importances ──")
for _, row in perm_df.head(15).iterrows():
    print(f"  {row['feature']:<45} {row['importance']:.4f} ± {row['std']:.4f}")


ACCENT='#4C6EF5'; CORAL='#E8593C'; TEAL='#1D9E75'; AMBER='#EF9F27'
PURPLE='#9B59B6'; GRAY='#888780'; BG='#F8F8F6'


fig1 = plt.figure(figsize=(18, 14))
fig1.suptitle('Feature Importance Analysis — SHAP Values\n'
              'Which NP properties most influence ZOI prediction?',
              fontsize=14, fontweight='bold', y=0.98)
gs1 = gridspec.GridSpec(2, 2, figure=fig1, hspace=0.42, wspace=0.36)


ax1 = fig1.add_subplot(gs1[0, 0])
feat_labels = list(mean_abs_shap.keys())
feat_vals   = list(mean_abs_shap.values())
bar_colors  = [ACCENT if v == max(feat_vals) else
               CORAL  if v == sorted(feat_vals)[-2] else
               '#AABCE8' for v in feat_vals]
bars = ax1.barh(feat_labels[::-1], feat_vals[::-1],
                color=bar_colors[::-1], alpha=0.85, edgecolor='white')
ax1.set_xlabel('Mean |SHAP value| (mm)', fontsize=11)
ax1.set_title('Grouped Feature Importance\n(original features)', fontsize=11, fontweight='bold')
ax1.set_facecolor(BG)
# Annotate values
for bar, val in zip(bars, feat_vals[::-1]):
    ax1.text(val + 0.01, bar.get_y() + bar.get_height()/2,
             f'{val:.3f}', va='center', fontsize=8, color='#333333')


ax2 = fig1.add_subplot(gs1[0, 1])
top15_raw  = mean_abs_raw.head(15)

def clean_label(name):
    label_map = {
        'Dose_log': 'Dose (log)',
        'Core_size_log': 'Core size (log)',
        'Duration': 'Duration',
    }
    if name in label_map:
        return label_map[name]
    for prefix in ['NPs_','Species_','Family_','Shape_','Coating_',
                   'Class_','Type(Gram( +/-)_']:
        if name.startswith(prefix):
            cat = prefix.rstrip('_').replace('Type(Gram( +/-)','Gram')
            val = name[len(prefix):]
            return f'[{cat}] {val}'
    return name

labels2 = [clean_label(f) for f in top15_raw.index]
colors2 = []
for f in top15_raw.index:
    if 'Dose' in f: colors2.append(ACCENT)
    elif 'Core' in f: colors2.append(CORAL)
    elif 'Species' in f: colors2.append(TEAL)
    elif 'NPs_' in f: colors2.append(AMBER)
    elif 'Duration' in f: colors2.append(PURPLE)
    else: colors2.append('#AABCE8')

ax2.barh(labels2[::-1], top15_raw.values[::-1],
         color=colors2[::-1], alpha=0.85, edgecolor='white')
ax2.set_xlabel('Mean |SHAP value| (mm)', fontsize=11)
ax2.set_title('Top 15 Individual Features\n(one-hot expanded)', fontsize=11, fontweight='bold')
ax2.set_facecolor(BG)
ax2.tick_params(axis='y', labelsize=8)


from matplotlib.patches import Patch
legend_els = [
    Patch(facecolor=ACCENT,  label='Dose'),
    Patch(facecolor=CORAL,   label='Core size'),
    Patch(facecolor=TEAL,    label='Species'),
    Patch(facecolor=AMBER,   label='NP type'),
    Patch(facecolor=PURPLE,  label='Duration'),
    Patch(facecolor='#AABCE8', label='Other'),
]
ax2.legend(handles=legend_els, fontsize=7, loc='lower right')


ax3 = fig1.add_subplot(gs1[1, 0])
top12_cols = mean_abs_raw.head(12).index.tolist()
shap_top12 = shap_values[:, [list(X_test.columns).index(c) for c in top12_cols]]
X_top12    = X_test[top12_cols]
clean_top12 = [clean_label(c) for c in top12_cols]


for i, (col, clean) in enumerate(zip(top12_cols, clean_top12)):
    col_idx = list(X_test.columns).index(col)
    sv = shap_values[:, col_idx]
    fv = X_test[col].values

    # Normalise feature values for color
    fv_norm = (fv.astype(float) - fv.astype(float).min()) / (fv.astype(float).max() - fv.astype(float).min() + 1e-9)

    # Jitter y
    y_jitter = i + np.random.uniform(-0.25, 0.25, len(sv))
    scatter = ax3.scatter(sv, y_jitter, c=fv_norm, cmap='RdYlBu_r',
                          s=12, alpha=0.6, vmin=0, vmax=1)

ax3.set_yticks(range(len(clean_top12)))
ax3.set_yticklabels(clean_top12[::-1][::-1], fontsize=8)
ax3.axvline(0, color=GRAY, linewidth=1, linestyle='--', alpha=0.7)
ax3.set_xlabel('SHAP value (impact on ZOI prediction in mm)', fontsize=10)
ax3.set_title('SHAP Beeswarm — Top 12 Features\n'
              'Red=high value  Blue=low value', fontsize=11, fontweight='bold')
ax3.set_facecolor(BG)
plt.colorbar(scatter, ax=ax3, label='Feature value\n(normalised)',
             fraction=0.035, pad=0.01)


ax4 = fig1.add_subplot(gs1[1, 1])

def group_perm(perm_df, feature_names):
    prefixes = ['NPs', 'Coating', 'Shape', 'Class', 'Family', 'Species',
                'Type(Gram( +/-)']
    numerical = ['Core_size_log', 'Dose_log', 'Duration']
    grouped = {}
    for col in numerical:
        row = perm_df[perm_df['feature'] == col]
        if len(row): grouped[col] = row['importance'].values[0]
    for prefix in prefixes:
        rows = perm_df[perm_df['feature'].str.startswith(prefix + '_')]
        if len(rows): grouped[prefix] = rows['importance'].sum()
    return grouped

grouped_perm = group_perm(perm_df, X_test.columns)


common_keys = [k for k in mean_abs_shap if k in grouped_perm]
shap_vals_plot = np.array([mean_abs_shap[k] for k in common_keys])
perm_vals_plot = np.array([grouped_perm[k] for k in common_keys])


shap_norm = shap_vals_plot / shap_vals_plot.sum()
perm_norm = np.clip(perm_vals_plot, 0, None)
perm_sum = perm_norm.sum()
if perm_sum > 0: perm_norm = perm_norm / perm_sum

x4 = np.arange(len(common_keys))
w4 = 0.35
ax4.bar(x4 - w4/2, shap_norm, w4, label='SHAP (normalised)',
        color=ACCENT, alpha=0.82, edgecolor='white')
ax4.bar(x4 + w4/2, perm_norm, w4, label='Permutation (normalised)',
        color=CORAL, alpha=0.82, edgecolor='white')

short_keys = [k.replace('Type(Gram( +/-)','Gram').replace('Core_size_log','CoreSize')
               .replace('Dose_log','Dose') for k in common_keys]
ax4.set_xticks(x4)
ax4.set_xticklabels(short_keys, rotation=35, ha='right', fontsize=9)
ax4.set_ylabel('Normalised importance', fontsize=11)
ax4.set_title('SHAP vs Permutation Importance\n(both normalised to sum=1)',
              fontsize=11, fontweight='bold')
ax4.legend(fontsize=10)
ax4.set_facecolor(BG)

plt.savefig('shap_main.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("\nFigure 1 saved: shap_main.png")



fig3, axes3 = plt.subplots(1, 3, figsize=(18, 7))
fig3.suptitle('SHAP Waterfall — Individual Prediction Explanations\n'
              'How each feature contributed to 3 specific predictions',
              fontsize=13, fontweight='bold', y=1.01)


y_pred_test = model.predict(X_test)
idx_high   = np.argmax(y_pred_test)
idx_median = np.argmin(np.abs(y_pred_test - np.median(y_pred_test)))
idx_low    = np.argmin(y_pred_test)
cases = [
    (idx_high,   f'Highest predicted ZOI\n({y_pred_test[idx_high]:.1f} mm)'),
    (idx_median, f'Median predicted ZOI\n({y_pred_test[idx_median]:.1f} mm)'),
    (idx_low,    f'Lowest predicted ZOI\n({y_pred_test[idx_low]:.1f} mm)'),
]

for ax, (idx, title) in zip(axes3, cases):
    sv_i  = shap_values[idx]
    top_n = 10

    # Get top contributing features for this prediction
    top_idx  = np.argsort(np.abs(sv_i))[-top_n:][::-1]
    top_feats = [clean_label(X_test.columns[i]) for i in top_idx]
    top_shaps = sv_i[top_idx]

    colors_wf = [ACCENT if s > 0 else CORAL for s in top_shaps]
    y_pos = range(len(top_feats))

    ax.barh(list(y_pos), top_shaps[::-1], color=colors_wf[::-1],
            alpha=0.82, edgecolor='white')
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(top_feats[::-1], fontsize=8)
    ax.axvline(0, color=GRAY, linewidth=1.2)
    ax.set_xlabel('SHAP contribution (mm)', fontsize=10)
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.set_facecolor(BG)

    actual_zoi = y_test.iloc[idx] if hasattr(y_test, 'iloc') else y_test[idx]
    ax.text(0.02, 0.02,
            f'Actual ZOI: {actual_zoi:.1f} mm\nBase: {expected_value:.1f} mm',
            transform=ax.transAxes, fontsize=8, color='#444441',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

from matplotlib.patches import Patch
legend_wf = [Patch(facecolor=ACCENT, label='Increases ZOI ↑'),
             Patch(facecolor=CORAL,  label='Decreases ZOI ↓')]
fig3.legend(handles=legend_wf, loc='upper right', fontsize=10)
plt.tight_layout()
plt.savefig('shap_waterfall.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("Figure 3 saved: shap_waterfall.png")


shap_df = pd.DataFrame(shap_values, columns=X_test.columns)
shap_df.to_csv('shap_values.csv', index=False)

# Summary table
summary = pd.DataFrame({
    'Feature': list(mean_abs_shap.keys()),
    'Mean_Abs_SHAP': list(mean_abs_shap.values()),
    'Rank': range(1, len(mean_abs_shap)+1)
})
summary.to_csv('shap_summary.csv', index=False)
print("Saved: shap_values.csv, shap_summary.csv")


print("\n══ Interpretation Summary ══")
ranked = list(mean_abs_shap.items())
print(f"\n  Rank 1 — {ranked[0][0]}: {ranked[0][1]:.4f} mm average ZOI influence")
print(f"  Rank 2 — {ranked[1][0]}: {ranked[1][1]:.4f} mm average ZOI influence")
print(f"  Rank 3 — {ranked[2][0]}: {ranked[2][1]:.4f} mm average ZOI influence")
print(f"  Rank 4 — {ranked[3][0]}: {ranked[3][1]:.4f} mm average ZOI influence")


np_props = ['NPs','Core_size_log','Shape','Coating']
np_total = sum(mean_abs_shap.get(k,0) for k in np_props)
all_total = sum(mean_abs_shap.values())
print(f"    NP-specific properties (NPs + Core_size + Shape + Coating): ({np_total/all_total*100:.1f}% )")


bio_props = ['Species','Family','Class','Type(Gram( +/-)']
bio_total = sum(mean_abs_shap.get(k,0) for k in bio_props)
print(f"    Biological context (Species + Family + Class + Gram):  ({bio_total/all_total*100:.1f}% )")


exp_props = ['Dose_log','Duration']
exp_total = sum(mean_abs_shap.get(k,0) for k in exp_props)
print(f"     Experimental conditions (Dose + Duration):  ({exp_total/all_total*100:.1f}% )")
