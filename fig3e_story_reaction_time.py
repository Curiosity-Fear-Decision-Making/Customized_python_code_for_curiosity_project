# Foldername: custom_python
# Filename: fig3e_story_reaction_time.py
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde, ttest_1samp
import utility_custom_python as utility
from utility_custom_python import online_INPUT_DIR, online_load_story_reaction_time_summaries


matplotlib.rcParams.update({'font.family': 'Arial', 'font.size': 8, 'figure.dpi': 160})
fig, ax = plt.subplots(figsize=(3.2, 3.6))
_, trials, _ = online_load_story_reaction_time_summaries()
behavior_files = list(online_INPUT_DIR.iterdir())
rows = []
for sub, part in trials.groupby('Subject', sort=False):

    def find_reaction_time_trial_file(keyword):
        hits = [p for p in behavior_files if sub in p.name.replace('\u200b', '') and keyword in p.name]
        return hits[0]
    raw = pd.read_csv(find_reaction_time_trial_file('result_trial_present'), sep='\t', header=None).drop_duplicates()
    cue = pd.DataFrame({'trial': pd.to_numeric(raw[0]), 'session': pd.to_numeric(raw[1]), 'cue': pd.to_numeric(raw[7], errors='coerce'), 'social_rating': pd.to_numeric(raw[9], errors='coerce')})
    rs = pd.read_csv(find_reaction_time_trial_file('result_Rating_session'), sep='\t', header=None)
    ratings = pd.DataFrame({'cue': pd.to_numeric(rs[1], errors='coerce'), 'c': pd.to_numeric(rs[4], errors='coerce')})
    ratings = ratings[ratings.c.between(1, 6)].groupby('cue').c.mean()
    merged = part.merge(cue, on=['trial', 'session'], validate='one_to_one')
    merged['c'] = merged.cue.map(ratings)
    rows.append(merged)
all_trials = pd.concat(rows, ignore_index=True)
valid = all_trials[['rt_s', 'c', 'threat_level', 'social_rating']].apply(np.isfinite).all(axis=1)
all_trials['included_regression'] = valid
data = all_trials.loc[valid].copy()
terms = ['c', 'social_rating', 'threat_level']
colors = [utility.CURIOSITY_COLOR, plt.cm.Purples(0.65), utility.THREAT_COLOR]
rows = []
for subject, part in data.groupby('Subject'):
    Xraw = part[terms].to_numpy(float)
    X = np.column_stack([np.ones(len(part))] + [utility.standardize_values(Xraw[:, j]) for j in range(3)])
    coef = np.linalg.lstsq(X, part.rt_s.to_numpy(float), rcond=None)[0][1:]
    rows.append(dict(Subject=subject, n_trials=len(part), **dict(zip(terms, coef))))
coefficients = pd.DataFrame(rows)
rng = np.random.default_rng(0)
print('\nFig3E | Story RT model')
for i, (term, color, label) in enumerate(zip(terms, colors, ['Self-rating', 'Social-rating', 'Threat'])):
    values = coefficients[term].to_numpy(float)
    mean, sem = (values.mean(), values.std(ddof=1) / np.sqrt(len(values)))
    test = ttest_1samp(values, 0)
    print(f'{label}: n={len(values)}, beta={mean:+.4f}, t({len(values) - 1})={test.statistic:.3f}, p={test.pvalue:.3g}')
    if len(np.unique(values)) > 1:
        grid = np.linspace(values.min(), values.max(), 300)
        density = gaussian_kde(values, bw_method=0.25)(grid)
        density = density / density.max() * 0.38
        ax.fill_betweenx(grid, i + 0.05, i + 0.05 + density, facecolor=color, alpha=0.2, lw=0, zorder=1)
        ax.plot(i + 0.05 + density, grid, color=color, lw=0.5, alpha=0.5, zorder=2)
    ax.scatter(i - 0.15 + rng.uniform(-0.05, 0.05, len(values)), values, s=3, facecolors=color, edgecolors='none', alpha=utility.POINT_ALPHA, zorder=3)
    ax.vlines(i, mean - sem, mean + sem, colors=color, lw=utility.MEAN_LINE_WIDTH, zorder=5)
    ax.hlines([mean - sem, mean + sem], i - 0.1, i + 0.1, colors=color, lw=utility.MEAN_LINE_WIDTH, zorder=5)
    ax.scatter(i, mean, s=utility.LARGE_MEAN_POINT_SIZE, facecolors=color, edgecolors='white', linewidths=0.5, zorder=7)
    ax.annotate(utility.format_significance_label(test.pvalue), (i, mean + sem), xytext=(0, 5), textcoords='offset points', ha='center', va='bottom', fontsize=8, fontweight='bold', color=color)
ax.axhline(0, color='gray', lw=0.6, ls='--', zorder=0)
ax.set(xticks=range(3), xlim=(-0.6, 2.98), ylabel='RT coefficient (s/SD)', title='Story-choice RT effect\ncontrolling for social interest')
ax.margins(y=0.12)
ax.set_xticklabels(['β Self-rating', 'β Social-rating', 'β Threat'], fontsize=8, rotation=35, ha='right', rotation_mode='anchor')
utility.style_plot_axes(ax)
utility.standardize_figure_labels(fig)
fig.tight_layout()
plt.show()
