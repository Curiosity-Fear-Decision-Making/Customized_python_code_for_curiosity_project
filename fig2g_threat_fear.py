# Foldername: custom_python
# Filename: fig2g_threat_fear.py
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import utility_custom_python as utility
from utility_custom_python import spider_RISK_LEVELS as RISK_LEVELS, spider_SCARED_MAX as SCARED_MAX, spider_SCARED_MIN as SCARED_MIN, spider_filter_excluded_participants as filter_excluded_participants, spider_list_regret_participant_ids as list_regret_participant_ids, spider_load_spider_subject


matplotlib.rcParams.update({'font.family': 'Arial', 'font.size': 8, 'figure.dpi': 160})
fig, ax = plt.subplots(figsize=(3.2, 3.6))
subs = filter_excluded_participants(list_regret_participant_ids())
mat = []
for s in subs:
    rows = spider_load_spider_subject(s)
    sc = np.array([r.get('scared', np.nan) for r in rows if r.get('y', np.nan) == 1], float)
    rk = np.array([r.get('risk_lv', np.nan) for r in rows if r.get('y', np.nan) == 1], float)
    ok = np.isfinite(sc) & np.isfinite(rk)
    if not ok.any():
        continue
    sc, rk = (sc[ok], rk[ok])
    mat.append([sc[rk == r].mean() if (rk == r).any() else np.nan for r in RISK_LEVELS])
scared = np.array(mat, float)
color = utility.THREAT_COLOR
xlabel = 'Risk level'
ylabel = 'Fear rating'
ylim = (SCARED_MIN - 0.4, SCARED_MAX + 0.6)
yticks = np.arange(SCARED_MIN, SCARED_MAX + 1)
tag = '2G'
significance_y = 0.96

x_arr = np.array(RISK_LEVELS, float)
slopes, intercepts = ([], [])
for i in range(scared.shape[0]):
    y_i = scared[i]
    m = ~np.isnan(y_i)
    if m.sum() < 2 or np.unique(x_arr[m]).size < 2:
        continue
    s, b = np.polyfit(x_arr[m], y_i[m], 1)
    slopes.append(s)
    intercepts.append(b)
slopes = np.asarray(slopes)
intercepts = np.asarray(intercepts)
xg = np.linspace(x_arr.min() - 0.2, x_arr.max() + 0.2, 200)
for i in range(len(slopes)):
    ax.plot(xg, slopes[i] * xg + intercepts[i], color=color, lw=utility.PARTICIPANT_LINE_WIDTH, alpha=0.3, zorder=2)
_rng = np.random.default_rng(0)
for j, xv in enumerate(x_arr):
    col = scared[:, j]
    col = col[~np.isnan(col)]
    if col.size:
        jittered = xv - utility.POINT_OFFSET + _rng.uniform(-0.07, 0.07, col.size)
        ax.scatter(jittered, col, s=utility.POINT_SIZE, color=color, alpha=utility.POINT_ALPHA, edgecolors='none', zorder=3)
line = None
if slopes.size:
    line = float(intercepts.mean()) + float(slopes.mean()) * xg
    ax.plot(xg, line, color=color, lw=utility.MEAN_LINE_WIDTH, zorder=6)
means, stds, xs_keep = ([], [], [])
for j, xv in enumerate(x_arr):
    col = scared[:, j]
    col = col[~np.isnan(col)]
    if col.size == 0:
        continue
    means.append(col.mean())
    stds.append(col.std(ddof=1) / np.sqrt(col.size) if col.size >= 2 else 0.0)
    xs_keep.append(xv)
means = np.asarray(means)
stds = np.asarray(stds)
xs_keep = np.asarray(xs_keep)
if slopes.size and xs_keep.size >= 2:
    std_c = np.interp(xg, xs_keep, stds, left=stds[0], right=stds[-1])
    ax.fill_between(xg, line - std_c, line + std_c, facecolor=color, alpha=0.2, edgecolor='none', zorder=4)
if xs_keep.size:
    ax.errorbar(xs_keep, means, yerr=stds, fmt='none', ecolor=color, elinewidth=utility.ERROR_BAR_LINE_WIDTH, capsize=utility.ERROR_BAR_CAP_SIZE, capthick=utility.ERROR_BAR_LINE_WIDTH, zorder=6.5)
ax.scatter(xs_keep, means, s=utility.MEAN_POINT_SIZE, facecolors=color, edgecolors='white', linewidths=0.5, zorder=7)
if slopes.size > 1:
    t_slope, p_slope = stats.ttest_1samp(slopes, 0.0)
    st = utility.format_significance_label(p_slope)
    print('\nFig%s | slope [%s -> %s]: mean=%+.3f t(%d)=%+.2f p=%s %s (n=%d)' % (tag, xlabel, ylabel, slopes.mean(), len(slopes) - 1, t_slope, utility.format_p_value(p_slope), st, len(slopes)))
    if st:
        ax.text(0.05, significance_y, st, transform=ax.transAxes, ha='left', va='top', fontsize=utility.ANNOTATION_FONT_SIZE, fontweight='bold', color=color)
ax.set_xlabel(xlabel, fontsize=utility.FONT_SIZE)
if ylabel:
    ax.set_ylabel(ylabel, fontsize=utility.FONT_SIZE)
ax.set_xticks(x_arr)
ax.set_yticks(yticks)
ax.set_xlim(x_arr.min() - 0.3, x_arr.max() + 0.3)
ax.set_ylim(*ylim)
utility.style_plot_axes(ax)
ax.set_title('Threat level vs. fear rating', fontsize=utility.TITLE_FONT_SIZE)
utility.standardize_figure_labels(fig)
fig.tight_layout()
plt.show()
