# Foldername: custom_python
# Filename: fig3h_sticsa_cognitive_curiosity.py
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import linregress, pearsonr
import utility_custom_python as utility
from utility_custom_python import online_CURIOSITY as CURIOSITY, online_load_online_questionnaire_scores as load_online_questionnaire_scores


matplotlib.rcParams.update({'font.family': 'Arial', 'font.size': 8, 'figure.dpi': 160})
fig, ax = plt.subplots(figsize=(3.2, 3.6))
df = load_online_questionnaire_scores().copy()
df['Curiosity'] = np.mean([utility.standardize_mediation_predictor(df[c]) for c in CURIOSITY], axis=0)
compact = df[['Subject', 'Curiosity', 'Self_Rating', 'STICSA_Cognitive']].copy()
for key in ['STICSA_Cognitive']:
    compact[key + '_z'] = (compact[key] - compact[key].mean()) / compact[key].std(ddof=1)
purple, green = ('#6a51a3', '#238b45')
specs = [('Self_Rating', 'STICSA_Cognitive_z', 'Self-rating', purple, '-'), ('Curiosity', 'STICSA_Cognitive_z', '5DC composite', green, '-')]
ylabel = 'STICSA cognitive (z)'
title = 'Anxiety–curiosity\nassociation'
family_size = 4

ax.clear()
rows = []
for xkey, ykey, label, color, linestyle in specs:
    pair = compact[[xkey, ykey]].replace([np.inf, -np.inf], np.nan).dropna()
    xraw = pair[xkey].to_numpy(float)
    y = pair[ykey].to_numpy(float)
    x = (xraw - xraw.mean()) / xraw.std(ddof=1)
    r, pvalue = pearsonr(x, y)
    fit = linregress(x, y)
    grid = np.linspace(x.min(), x.max(), 120)
    prediction = fit.intercept + fit.slope * grid
    residual = np.sqrt(np.sum((y - fit.intercept - fit.slope * x) ** 2) / (len(x) - 2))
    se = residual * np.sqrt(1 / len(x) + (grid - x.mean()) ** 2 / np.sum((x - x.mean()) ** 2))
    ax.scatter(x, y, s=2, color=color, alpha=0.5, linewidths=0, zorder=1)
    ax.plot(grid, prediction, color=color, ls=linestyle, lw=1, label=label, zorder=3)
    ax.fill_between(grid, prediction - se, prediction + se, color=color, alpha=0.08, lw=0)
    q = min(1.0, pvalue * family_size)
    ax.lines[-1].set_label(f'{label} {utility.format_significance_label(q)}')
    rows.append(dict(comparison=label, n=len(pair), r=r, p_raw=pvalue, p_corrected=q, n_tests=family_size))
ax.set(xlabel='Standardized score (z)', ylabel=ylabel)
ax.set_title(title, fontsize=8, pad=10)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.4), frameon=False, fontsize=8, handlelength=1.2, handletextpad=0.3, labelspacing=0.25)
utility.style_plot_axes(ax)
print(f'\nFig3H | {title.replace(chr(10), ' ')}; Bonferroni x{family_size}')
print(pd.DataFrame(rows).to_string(index=False, float_format=lambda value: f'{value:.4g}'))
utility.standardize_figure_labels(fig)
fig.tight_layout()
plt.show()
