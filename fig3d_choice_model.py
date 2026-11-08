# Foldername: custom_python
# Filename: fig3d_choice_model.py
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import gaussian_kde, ttest_1samp
from scipy.special import expit
import utility_custom_python as utility
from utility_custom_python import online_find_file as find_file, online_load_online_v6_subjects as load_online_v6_subjects


def calculate_logistic_negative_log_likelihood(params, X, y):
    bias = params[-1]
    betas = np.asarray(params[:-1])
    eta = np.clip(X @ betas + bias, -500, 500)
    pr = np.clip(expit(eta), 1e-12, 1 - 1e-12)
    return -np.sum(y * np.log(pr) + (1 - y) * np.log(1 - pr))


matplotlib.rcParams.update({'font.family': 'Arial', 'font.size': 8, 'figure.dpi': 160})
fig, ax = plt.subplots(figsize=(3.2, 3.6))
subject_list = load_online_v6_subjects()['subject_list']
print('\nFig3D | Story-choice model')
coefs = []
for s in subject_list:
    f1 = find_file(s, 'result_trial_present')
    f2 = find_file(s, 'result_Rating_session')
    if not (f1 and f2):
        continue
    tp = utility.read_tab_delimited_table(f1)
    rs = utility.read_tab_delimited_table(f2)
    yr = pd.to_numeric(tp[:, 12], errors='coerce')
    mv = ~np.isnan(yr)
    tpv = tp[mv]
    y = (yr[mv].astype(int) == 1).astype(int)
    model_coefficients_risk = pd.to_numeric(tpv[:, 10], errors='coerce').astype(float)
    ap = pd.to_numeric(rs[:, 1], errors='coerce')
    ar = pd.to_numeric(rs[:, 4], errors='coerce')
    cb = np.array([np.nanmean(ar[np.where(ap == float(tpv[i, 7]))[0]]) if np.where(ap == float(tpv[i, 7]))[0].size else np.nan for i in range(len(tpv))])
    soc = pd.to_numeric(tpv[:, 9], errors='coerce').astype(float)
    X = np.column_stack([cb, soc, model_coefficients_risk])
    stk = np.column_stack([y, X])
    mask = ~np.any(np.isnan(stk), axis=1)
    ym = y[mask]
    Xm = X[mask]
    if len(ym) < 5 or len(np.unique(ym)) == 1:
        continue
    Xz = np.column_stack([utility.standardize_values(Xm[:, j]) for j in range(3)])
    r = minimize(calculate_logistic_negative_log_likelihood, [0.5, 0.5, 0.5, 0.0], args=(Xz, ym.astype(float)), method='L-BFGS-B', bounds=[(-10, 10)] * 3 + [(-5, 5)])
    coefs.append(r.x[:3])
coefs = np.array(coefs)
model_coefficients_labels = ['β\nSelf-rating', 'β\nSocial-rating', 'β\nThreat']
colors = [utility.CURIOSITY_COLOR, plt.cm.Purples(0.65), utility.THREAT_COLOR]
n_pred = coefs.shape[1]
x_pos = np.arange(n_pred)
rng = np.random.default_rng(0)
pdf_w = 0.38
for i in range(n_pred):
    ys = coefs[:, i]
    ys = ys[~np.isnan(ys)]
    if len(ys) < 2 or np.unique(ys).size < 2:
        continue
    kde = gaussian_kde(ys, bw_method=0.25)
    yk = np.linspace(ys.min(), ys.max(), 300)
    dens = kde(yk)
    if dens.max() > 0:
        dens = dens / dens.max() * pdf_w
    bx = x_pos[i] + 0.05
    ax.fill_betweenx(yk, bx, bx + dens, facecolor=colors[i], alpha=0.2, lw=0, zorder=1)
    ax.plot(bx + dens, yk, color=colors[i], lw=0.5, alpha=0.5, zorder=2)
    xs = x_pos[i] - 0.15 + rng.uniform(-0.05, 0.05, len(ys))
    ax.scatter(xs, ys, s=3, facecolors=colors[i], edgecolors='none', alpha=utility.POINT_ALPHA, zorder=3)
ax.axhline(0, color='gray', lw=0.6, ls='--', zorder=0)
means = np.nanmean(coefs, axis=0)
stds = np.nanstd(coefs, axis=0, ddof=1) / np.sqrt(np.sum(np.isfinite(coefs), axis=0))
for i in range(n_pred):
    ax.vlines(x_pos[i], means[i] - stds[i], means[i] + stds[i], colors=colors[i], lw=utility.MEAN_LINE_WIDTH, zorder=5)
    ax.hlines([means[i] - stds[i], means[i] + stds[i]], x_pos[i] - 0.1, x_pos[i] + 0.1, colors=colors[i], lw=utility.MEAN_LINE_WIDTH, zorder=5)
    ax.scatter(x_pos[i], means[i], s=utility.LARGE_MEAN_POINT_SIZE, facecolors=colors[i], edgecolors='white', linewidths=0.5, zorder=7)
y_hi = np.nanmax(coefs)
y_lo = np.nanmin(coefs)
pad = 0.06 * (y_hi - y_lo if y_hi > y_lo else 1.0)
for i in range(n_pred):
    vals = coefs[:, i]
    vals = vals[~np.isnan(vals)]
    if len(vals) < 2 or np.std(vals, ddof=1) == 0:
        continue
    t_c, model_coefficients_p = ttest_1samp(vals, 0.0)
    _n = len(vals)
    _sd = np.std(vals, ddof=1)
    _d = vals.mean() / _sd if _sd > 0 else np.nan
    _lab = model_coefficients_labels[i].replace('\n', ' ')
    print(f'Fig3D | Coefficient [{_lab}]: n={_n}, mean={vals.mean():+.4f}, SD={_sd:.4f}, t({_n - 1})={t_c:.3f}, p={model_coefficients_p:.3g}, d={_d:.3f}')
    star = utility.format_significance_label(model_coefficients_p)
    is_sig = star not in ('', 'n.s.')
    ax.text(x_pos[i], means[i] + stds[i] + pad, star, ha='center', va='bottom', fontsize=utility.ANNOTATION_FONT_SIZE, fontweight='bold' if is_sig else 'normal', color=colors[i], zorder=9)
ax.set_ylabel('Coefficient', fontsize=utility.FONT_SIZE)
ax.set_xticks(x_pos)
ax.set_xticklabels(model_coefficients_labels, fontsize=utility.TICK_FONT_SIZE)
ax.set_xlim(-0.6, n_pred - 0.4 + pdf_w)
ax.set_ylim(-12, 12)
utility.style_plot_axes(ax)
ax.set_title('Main effect model', fontsize=utility.TITLE_FONT_SIZE)
ax.set_xticklabels(['β Self-rating', 'β Social-rating', 'β Threat'], rotation=35, ha='right')
utility.standardize_figure_labels(fig)
fig.tight_layout()
plt.show()
