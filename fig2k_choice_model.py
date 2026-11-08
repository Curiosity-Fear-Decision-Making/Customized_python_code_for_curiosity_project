# Foldername: custom_python
# Filename: fig2k_choice_model.py
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy.optimize import minimize
from scipy.special import expit
import utility_custom_python as utility
from utility_custom_python import spider_filter_excluded_participants as filter_excluded_participants, spider_list_regret_participant_ids as list_regret_participant_ids, spider_load_spider_subject


def choice_negative_log_likelihood(params, X, y):
    bias = params[-1]
    betas = np.asarray(params[:-1])
    eta = np.clip(X @ betas + bias, -500, 500)
    pr = np.clip(expit(eta), 1e-12, 1 - 1e-12)
    return -np.sum(y * np.log(pr) + (1 - y) * np.log(1 - pr))


matplotlib.rcParams.update({'font.family': 'Arial', 'font.size': 8, 'figure.dpi': 160})
fig, ax = plt.subplots(figsize=(3.2, 3.6))
subs = filter_excluded_participants(list_regret_participant_ids())
fits = []
for sub_id in subs:
    rows = spider_load_spider_subject(sub_id)
    y = np.array([r['y'] for r in rows], dtype=float)
    cb = np.array([r['clickbait'] for r in rows], dtype=float)
    rk = np.array([r['risk'] for r in rows], dtype=float)
    X_raw = np.column_stack([cb, rk])
    mask = ~np.any(np.isnan(np.column_stack([y, X_raw])), axis=1)
    y_m, X_m = (y[mask], X_raw[mask])
    X_z = np.column_stack([utility.standardize_values(X_m[:, j]) for j in range(X_m.shape[1])])
    res = minimize(choice_negative_log_likelihood, [0.5, 0.5, 0.0], args=(X_z, y_m), method='L-BFGS-B', bounds=[(-10, 10)] * 2 + [(-5, 5)])
    fits.append(res.x[:2])
coef_mat = np.array(fits, float)
labels = ['β\nclickbait', 'γ\nrisk']
colors = [utility.CURIOSITY_COLOR, utility.THREAT_COLOR]
n_pred = coef_mat.shape[1]
x_pos = np.arange(n_pred)
rng = np.random.default_rng(0)
pdf_w = 0.38
for i in range(n_pred):
    ys = coef_mat[:, i]
    ys = ys[~np.isnan(ys)]
    if len(ys) >= 2 and np.unique(ys).size >= 2:
        kde = stats.gaussian_kde(ys, bw_method=0.25)
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
means = np.nanmean(coef_mat, axis=0)
stds = np.nanstd(coef_mat, axis=0, ddof=1)
for i in range(n_pred):
    ax.vlines(x_pos[i], means[i] - stds[i], means[i] + stds[i], colors=colors[i], lw=utility.MEAN_LINE_WIDTH, zorder=5)
    ax.hlines([means[i] - stds[i], means[i] + stds[i]], x_pos[i] - 0.1, x_pos[i] + 0.1, colors=colors[i], lw=utility.MEAN_LINE_WIDTH, zorder=5)
    ax.scatter(x_pos[i], means[i], s=utility.LARGE_MEAN_POINT_SIZE, facecolors=colors[i], edgecolors='white', linewidths=0.5, zorder=7)
y_hi = np.nanmax(coef_mat)
y_lo = np.nanmin(coef_mat)
pad = 0.06 * (y_hi - y_lo if y_hi > y_lo else 1.0)
for i in range(n_pred):
    vals = coef_mat[:, i]
    vals = vals[~np.isnan(vals)]
    if len(vals) < 2 or np.std(vals, ddof=1) == 0:
        continue
    t, p = stats.ttest_1samp(vals, 0.0)
    star = utility.format_significance_label(p)
    is_sig = star not in ('', 'n.s.')
    ax.text(x_pos[i], means[i] + stds[i] + pad, star, ha='center', va='bottom', fontsize=utility.ANNOTATION_FONT_SIZE, fontweight='bold' if is_sig else 'normal', color=colors[i], zorder=9)
    print('\nFig2K | %-12s mean=%+.3f sd=%.3f t(%d)=%+.2f p=%s %s' % (labels[i].replace('\n', ' '), means[i], stds[i], len(vals) - 1, t, utility.format_p_value(p), star))
ax.set_ylabel('Coefficient', fontsize=utility.FONT_SIZE)
ax.set_xticks(x_pos)
ax.set_xticklabels(labels, fontsize=utility.TICK_FONT_SIZE)
ax.set_xlim(-0.6, n_pred - 0.4 + pdf_w)
utility.style_plot_axes(ax)
ax.set_title('Predicting story choice', fontsize=utility.TITLE_FONT_SIZE)
utility.standardize_figure_labels(fig)
fig.tight_layout()
plt.show()
