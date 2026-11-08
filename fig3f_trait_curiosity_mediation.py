# Foldername: custom_python
# Filename: fig3f_trait_curiosity_mediation.py
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
import pandas as pd
from scipy.stats import pearsonr
import utility_custom_python as utility
from utility_custom_python import online_CURIOSITY as CURIOSITY, online_DASS_OFFSET as DASS_OFFSET, online_DASS_SCALES as DASS_SCALES, online_DISTRESS as DISTRESS, online_KEYS as KEYS, online_SEED, online_build_regression_design_matrix, online_load_online_questionnaire_scores as load_online_questionnaire_scores, online_load_trials as load_trials


def bootstrap_indirect_effect(df, predictor, n_boot=20000, family_size=10):
    part = df.dropna(subset=[predictor, 'Self_Rating', 'P_Story']).reset_index(drop=True)
    x, m = (utility.standardize_mediation_predictor(part[predictor]), utility.standardize_mediation_predictor(part.Self_Rating))
    y = part.P_Story.to_numpy(float)

    def estimate_mediation_indirect_effect(idx):
        xi, mi, yi = (x[idx], m[idx], y[idx])
        a = np.linalg.lstsq(online_build_regression_design_matrix([xi]), mi, rcond=None)[0][1]
        b = np.linalg.lstsq(online_build_regression_design_matrix([xi, mi]), yi, rcond=None)[0][2]
        return a * b
    estimate = estimate_mediation_indirect_effect(np.arange(len(x)))
    rng = np.random.default_rng(online_SEED)
    boot = np.array([estimate_mediation_indirect_effect(rng.integers(0, len(x), len(x))) for _ in range(n_boot)])
    tail = 100 * 0.05 / (2 * family_size)
    ci = np.percentile(boot, [tail, 100 - tail])
    tail_count = min(np.count_nonzero(boot <= 0), np.count_nonzero(boot >= 0))
    p_raw = min(1.0, 2.0 * (tail_count + 1) / (n_boot + 1))
    p_corrected = min(1.0, p_raw * family_size)
    return dict(indirect=estimate, indirect_se=boot.std(ddof=1), ci_indirect_family=ci, p_raw=p_raw, p_corrected=p_corrected, significant_family=bool(p_corrected < 0.05), n=len(x))


matplotlib.rcParams.update({'font.family': 'Arial', 'font.size': 8, 'figure.dpi': 160})
fig, ax = plt.subplots(figsize=(3.2, 3.6))
df = load_online_questionnaire_scores().copy()
for name in DASS_SCALES:
    df[name] = df[name] - DASS_OFFSET
df['Curiosity'] = np.mean([utility.standardize_mediation_predictor(df[c]) for c in CURIOSITY], axis=0)
df['Distress'] = np.mean([utility.standardize_mediation_predictor(df[c]) for c in DISTRESS], axis=0)
supplementary_trials = load_trials(df.Subject)
med_results = []
for supplementary_label, levels, color, offset, marker in [('Low threat', [1, 2], '#f3a6a0', -0.154, 'o'), ('High threat', [3, 4], '#a50f15', 0.154, 's')]:
    part = supplementary_trials.loc[supplementary_trials.Risk.isin(levels)].copy()
    means = part.groupby('Subject').agg(Self_Rating=('Self', 'mean'), P_Story=('Choice', 'mean'), N_Trials=('Choice', 'count'), N_Rated_Trials=('Self', 'count')).reset_index()
    stratum_data = df[['Subject'] + KEYS].merge(means, on='Subject', validate='one_to_one')
    correlations = []
    for key in KEYS:
        pair = stratum_data[[key, 'Self_Rating']].replace([np.inf, -np.inf], np.nan).dropna()
        r, p = pearsonr(pair[key], pair.Self_Rating)
        ci = np.tanh(np.arctanh(r) + np.array([-1, 1]) * 1.96 / np.sqrt(len(pair) - 3))
        correlations.append(dict(trait=key, n=len(pair), r=r, p=p, ci_low=ci[0], ci_high=ci[1]))
    adjusted = np.minimum(np.array([v['p'] for v in correlations]) * 10, 1.0)
    results = []
    for c, q in zip(correlations, adjusted):
        c['p_bonferroni'] = q
        med = bootstrap_indirect_effect(stratum_data, c['trait'], n_boot=20000, family_size=10)
        results.append(dict(threat=supplementary_label, **c, indirect=med['indirect'], indirect_se=med['indirect_se'], indirect_ci_low=med['ci_indirect_family'][0], indirect_ci_high=med['ci_indirect_family'][1], indirect_p_raw=med['p_raw'], indirect_p_corrected=med['p_corrected'], indirect_n_tests=10, indirect_n_boot=20000, indirect_significant=med['significant_family'], mediation_n=med['n']))
    result = pd.DataFrame(results)
    med_results.append(result)
    for y, row in result.reset_index(drop=True).iterrows():
        ax.errorbar(row.indirect, y + offset, xerr=row.indirect_se, fmt=marker, color=color, ms=2.3, elinewidth=0.9, capsize=2, capthick=0.8, mec='white', mew=0.3, zorder=3, alpha=1.0 if bool(row.indirect_significant) else 0.45, label=supplementary_label if y == 0 else None)
print('\nFig3F | Indirect effect: trait curiosity -> self-rating -> P(story); Bonferroni x10')
for group_index, (result, color, offset) in enumerate(zip(med_results, ['#f3a6a0', '#a50f15'], [-0.154, 0.154])):
    other = med_results[1 - group_index].reset_index(drop=True)
    for y, row in result.reset_index(drop=True).iterrows():
        significant = bool(row.indirect_significant)
        side = (1 if row.indirect >= other.iloc[y].indirect else -1) if significant else 1
        edge = row.indirect + side * row.indirect_se
        padding = side * 5
        if not significant:
            padding = 4
        ax.annotate(utility.format_significance_label(row.indirect_p_corrected), (edge, y + offset), xytext=(padding, 2 if group_index == 0 else -2), textcoords='offset points', ha='left' if side > 0 else 'right', va='center', fontsize=7, color=color)
        print(f'{KEYS[y]} | {row.threat}: n={row.mediation_n}, indirect={row.indirect:+.4f}, adjusted CI=[{row.indirect_ci_low:+.4f}, {row.indirect_ci_high:+.4f}], p_corrected={row.indirect_p_corrected:.3g}')
ax.axvline(0, color='.65', ls='--', lw=0.7)
ax.set(yticks=range(5), yticklabels=['Joy. Exp.', 'Dep. Sens.', 'Soc. Cur.', 'Str. Tol.', 'Thr. Seek.'], ylim=(4.65, -0.65), xlabel='Indirect effect on P(story)', title='Indirect associations\nwith story choice')
med_extent = max(((part.indirect.abs() + part.indirect_se).max() for part in med_results))
med_limit = max(0.045, med_extent * 2.25)
ax.set_xlim(-med_limit, med_limit)
ax.legend(handles=[Line2D([], [], color=c, marker=m, markersize=3, lw=0.9, label=l) for l, c, m in [('Low threat', '#f3a6a0', 'o'), ('High threat', '#a50f15', 's')]], loc='upper center', bbox_to_anchor=(0.5, -0.39), frameon=False, fontsize=7, ncol=2, handlelength=1, columnspacing=0.8)
utility.standardize_figure_labels(fig)
fig.tight_layout()
plt.show()
