# Foldername: custom_python
# Filename: fig2i_choice_probability.py
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from scipy import stats
import utility_custom_python as utility
from utility_custom_python import spider_RATING_LEVELS as RATING_LEVELS, spider_RISK_LEVELS as RISK_LEVELS, spider_filter_excluded_participants as filter_excluded_participants, spider_list_spider_participant_ids as list_spider_participant_ids, spider_load_spider_subject, spider_spider_RATING_LEVELS


matplotlib.rcParams.update({'font.family': 'Arial', 'font.size': 8, 'figure.dpi': 160})
fig, ax = plt.subplots(figsize=(3.2, 3.6))
subs = filter_excluded_participants(list_spider_participant_ids())
mat = []
for s in subs:
    rows = spider_load_spider_subject(s)
    yy = np.array([r['y'] for r in rows], float)
    cb = np.array([r.get('clickbait', np.nan) for r in rows], float)
    ok = np.isfinite(yy) & np.isfinite(cb)
    if not ok.any():
        continue
    yy = yy[ok]
    cb_r = np.clip(np.round(cb[ok]), 1, 6).astype(int)
    mat.append([yy[cb_r == lv].mean() if (cb_r == lv).any() else np.nan for lv in spider_spider_RATING_LEVELS])
mat = np.array(mat, float)
plot_risk = []
for s in subs:
    rows = spider_load_spider_subject(s)
    choices = np.array([row['y'] for row in rows], float)
    risk = np.array([row['risk_lv'] for row in rows], float)
    if choices.size == 0:
        continue
    plot_risk.append([choices[risk == level].mean() if (risk == level).any() else np.nan
                      for level in RISK_LEVELS])
plot_risk = np.array(plot_risk, float)

ylabel = 'P(choose story)'
ylim = (-0.08, 1.15)
yticks = np.arange(0, 1.01, 0.2)
tag = '2I'
for axis_index, (data, x_levels, color, xlabel, significance_y) in enumerate([
    (mat, RATING_LEVELS, utility.CURIOSITY_COLOR, 'Self-rating', 0.96),
    (plot_risk, RISK_LEVELS, utility.THREAT_COLOR, 'Threat rating', 0.8),
]):
    if axis_index == 0:
        plot_ax = ax
    else:
        threat_ax = ax.twinx()
        plot_ax = threat_ax

    x_arr = np.array(x_levels, float)
    slopes, intercepts = ([], [])
    for i in range(data.shape[0]):
        y_i = data[i]
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
        plot_ax.plot(slopes[i] * xg + intercepts[i], xg, color=color, lw=utility.PARTICIPANT_LINE_WIDTH, alpha=0.3, zorder=2)
    _rng = np.random.default_rng(0)
    for j, xv in enumerate(x_arr):
        col = data[:, j]
        col = col[~np.isnan(col)]
        if col.size:
            jittered = xv - utility.POINT_OFFSET + _rng.uniform(-0.07, 0.07, col.size)
            plot_ax.scatter(col, jittered, s=utility.POINT_SIZE, color=color, alpha=utility.POINT_ALPHA, edgecolors='none', zorder=3)
    line = None
    if slopes.size:
        line = float(intercepts.mean()) + float(slopes.mean()) * xg
        plot_ax.plot(line, xg, color=color, lw=utility.MEAN_LINE_WIDTH, zorder=6)
    means, stds, xs_keep = ([], [], [])
    for j, xv in enumerate(x_arr):
        col = data[:, j]
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
        plot_ax.fill_betweenx(xg, line - std_c, line + std_c, facecolor=color, alpha=0.2, edgecolor='none', zorder=4)
    if xs_keep.size:
        plot_ax.errorbar(means, xs_keep, xerr=stds, fmt='none', ecolor=color, elinewidth=utility.ERROR_BAR_LINE_WIDTH, capsize=utility.ERROR_BAR_CAP_SIZE, capthick=utility.ERROR_BAR_LINE_WIDTH, zorder=6.5)
    plot_ax.scatter(means, xs_keep, s=utility.MEAN_POINT_SIZE, facecolors=color, edgecolors='white', linewidths=0.5, zorder=7)
    if slopes.size > 1:
        t_slope, p_slope = stats.ttest_1samp(slopes, 0.0)
        st = utility.format_significance_label(p_slope)
        print('\nFig%s | slope [%s -> %s]: mean=%+.3f t(%d)=%+.2f p=%s %s (n=%d)' % (tag, xlabel, ylabel, slopes.mean(), len(slopes) - 1, t_slope, utility.format_p_value(p_slope), st, len(slopes)))
        if st:
            plot_ax.text(0.05, significance_y, st, transform=plot_ax.transAxes, ha='left', va='top', fontsize=utility.ANNOTATION_FONT_SIZE, fontweight='bold', color=color)
    plot_ax.set_xlabel(xlabel, fontsize=utility.FONT_SIZE)
    if ylabel:
        plot_ax.set_ylabel(ylabel, fontsize=utility.FONT_SIZE)
    plot_ax.set_xticks(x_arr)
    plot_ax.set_yticks(yticks)
    plot_ax.set_xlim(x_arr.min() - 0.3, x_arr.max() + 0.3)
    plot_ax.set_ylim(*ylim)
    plot_ax.set(xlabel=ylabel, ylabel=xlabel, xticks=yticks, yticks=x_arr, xlim=ylim, ylim=(x_arr.min() - 0.3, x_arr.max() + 0.3))
    utility.style_plot_axes(plot_ax)

    if axis_index == 0:
        ax.set_title('Curiosity (subjective) on P(story)', fontsize=utility.TITLE_FONT_SIZE)

threat_ax.patch.set_visible(False)
threat_ax.spines['right'].set_linewidth(utility.AXIS_LINE_WIDTH)
threat_ax.spines['right'].set_color(utility.THREAT_COLOR)
threat_ax.spines['top'].set_visible(False)
threat_ax.spines['left'].set_visible(False)
threat_ax.tick_params(axis='y', colors=utility.THREAT_COLOR)
threat_ax.yaxis.label.set_color(utility.THREAT_COLOR)
ax.set_xlabel('P(story)')
ax.set_xticks([0, 0.5, 1], ['0', '0.5', '1'])
ax.tick_params(axis='y', colors=utility.CURIOSITY_COLOR)
ax.yaxis.label.set_color(utility.CURIOSITY_COLOR)
ax.legend(handles=[Line2D([], [], color=utility.CURIOSITY_COLOR, marker='o', ms=3, label='Curiosity'), Line2D([], [], color=utility.THREAT_COLOR, marker='o', ms=3, label='Threat')], loc='upper center', bbox_to_anchor=(0.5, -0.32), borderaxespad=0, borderpad=0, ncol=2, fontsize=8, frameon=False, handlelength=0.8, handletextpad=0.3, columnspacing=0.6)
utility.standardize_figure_labels(fig)
fig.tight_layout()
plt.show()
