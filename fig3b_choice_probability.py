# Foldername: custom_python
# Filename: fig3b_choice_probability.py
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection, PathCollection
from matplotlib.lines import Line2D
from scipy.stats import ttest_1samp
import utility_custom_python as utility
from utility_custom_python import online_find_file as find_file, online_load_online_v6_subjects as load_online_v6_subjects


# if sys.platform == 'darwin' and matplotlib.get_backend() == 'module://backend_interagg':
#     matplotlib.use('MacOSX')


matplotlib.rcParams.update({'font.family': 'Arial', 'font.size': 8, 'figure.dpi': 160})
fig, ax = plt.subplots(figsize=(3.2, 3.6))
subject_list = load_online_v6_subjects()['subject_list']
risk_levels = [1, 2, 3, 4]
plot_risk = []
for s in subject_list:
    f1 = find_file(s, 'result_trial_present')
    if not f1:
        plot_risk.append([np.nan] * 4)
        continue
    tp = utility.read_tab_delimited_table(f1)
    choices = pd.to_numeric(tp[:, 12], errors='coerce')
    valid = ~np.isnan(choices)
    chose_story = (choices[valid].astype(int) == 1).astype(int)
    trial_risk = pd.to_numeric(tp[valid, 10], errors='coerce')
    plot_risk.append([chose_story[trial_risk == level].mean() if (trial_risk == level).any() else np.nan
                      for level in risk_levels])
risk = np.array(plot_risk, float)
xs, ys, sid = ([], [], [])
for ith, s in enumerate(subject_list):
    f1 = find_file(s, 'result_trial_present')
    f2 = find_file(s, 'result_Rating_session')
    if not (f1 and f2):
        continue
    tp = utility.read_tab_delimited_table(f1)
    rs = utility.read_tab_delimited_table(f2)
    rsp = pd.to_numeric(rs[:, 1], errors='coerce')
    rsv = pd.to_numeric(rs[:, 4], errors='coerce')
    yr = pd.to_numeric(tp[:, 12], errors='coerce')
    mv = ~np.isnan(yr)
    tpv = tp[mv]
    chose = (yr[mv].astype(int) == 1).astype(int)
    for i in range(len(tpv)):
        cb = pd.to_numeric(tpv[i, 7], errors='coerce')
        if np.isnan(cb):
            continue
        mm = np.where(rsp == cb)[0]
        if mm.size == 0:
            continue
        self_v = float(np.nanmean(rsv[mm]))
        if np.isnan(self_v):
            continue
        xs.append(self_v)
        ys.append(int(chose[i]))
        sid.append(ith)
xs, ys, sid = (np.asarray(xs), np.asarray(ys), np.asarray(sid))
rounded = np.round(xs).astype(int)
self_levels = np.unique(rounded)
self_matrix = np.full((len(np.unique(sid)), len(self_levels)), np.nan)
for i, subject in enumerate(np.unique(sid)):
    for j, level in enumerate(self_levels):
        cell = ys[(sid == subject) & (rounded == level)]
        if cell.size:
            self_matrix[i, j] = cell.mean()

ylabel = 'P(choose story)'
ylim = (-0.08, 1.15)
yticks = np.arange(0, 1.01, 0.2)
tag = '3B'
for axis_index, (data, x_levels, color, xlabel, significance_y) in enumerate([
    (self_matrix, self_levels, utility.CURIOSITY_COLOR, 'Self Rating', 0.96),
    (risk, [1, 2, 3, 4], utility.THREAT_COLOR, 'Threat Level', 0.8),
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
        plot_ax.plot(slopes[i] * xg + intercepts[i], xg, color=utility.lighten_plot_color(color), lw=utility.PARTICIPANT_LINE_WIDTH, alpha=0.33, zorder=2)
    _rng = np.random.default_rng(0)
    for j, xv in enumerate(x_arr):
        col = data[:, j]
        col = col[~np.isnan(col)]
        if col.size:
            jittered = xv - utility.POINT_OFFSET + _rng.uniform(-0.07, 0.07, col.size)
            plot_ax.scatter(col, jittered, s=utility.POINT_SIZE, color=utility.lighten_plot_color(color), alpha=0.33, edgecolors='none', zorder=3)
    line = None
    if slopes.size:
        line = float(intercepts.mean()) + float(slopes.mean()) * xg
        plot_ax.plot(line, xg, color=color, lw=utility.MEAN_LINE_WIDTH, alpha=1, zorder=6)
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
        plot_ax.errorbar(means, xs_keep, xerr=stds, fmt='none', ecolor=color, elinewidth=utility.ERROR_BAR_LINE_WIDTH, capsize=utility.ERROR_BAR_CAP_SIZE, capthick=utility.ERROR_BAR_LINE_WIDTH, alpha=1, zorder=6.5)
    plot_ax.scatter(means, xs_keep, s=utility.MEAN_POINT_SIZE, facecolors=color, edgecolors='white', linewidths=0.5, alpha=1, zorder=7)
    if slopes.size > 1:
        t_slope, p_slope = ttest_1samp(slopes, 0.0)
        st = utility.format_significance_label(p_slope)
        print('\nFig%s | slope [%s -> %s]: mean=%+.3f t(%d)=%+.2f p=%s %s (n=%d)' % (tag, xlabel, ylabel, slopes.mean(), len(slopes) - 1, t_slope, format(p_slope, '.3g'), st, len(slopes)))
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
        for line in ax.lines:
            if line.get_zorder() == 2:
                line.set_alpha(0.33 * 1.1 * 1.15 * 1.2 * 1.2)
                line.set_color(utility.lighten_plot_color(utility.CURIOSITY_COLOR, 0.65))
        for collection in ax.collections:
            if isinstance(collection, PathCollection) and collection.get_zorder() == 3:
                collection.set_alpha(0.33 * 1.1 * 1.15 * 1.2 * 1.2)
                collection.set_facecolor(utility.lighten_plot_color(utility.CURIOSITY_COLOR, 0.65))

for artist in list(ax.lines) + list(ax.collections):
    if artist.get_zorder() not in (2, 3):
        continue
    transform = artist.get_transform()
    offsets = artist.get_offset_transform() if isinstance(artist, PathCollection) else None
    artist.remove()
    if isinstance(artist, PathCollection):
        threat_ax.add_collection(artist, autolim=False)
        artist.set_offset_transform(offsets)
    else:
        threat_ax.add_line(artist)
    artist.set_transform(transform)
for line in list(ax.lines):
    if line.get_zorder() >= 6:
        transform = line.get_transform()
        line.remove()
        threat_ax.add_line(line)
        line.set_transform(transform)
        line.set_zorder(10)
for collection in list(ax.collections):
    if collection.get_zorder() >= 6:
        transform = collection.get_transform()
        offsets = collection.get_offset_transform()
        collection.remove()
        threat_ax.add_collection(collection, autolim=False)
        collection.set_transform(transform)
        collection.set_offset_transform(offsets)
        collection.set_zorder(11)
for choice_ax in (ax, threat_ax):
    for line in choice_ax.lines:
        line.set_linewidth(line.get_linewidth() / 2)
    for collection in choice_ax.collections:
        if isinstance(collection, LineCollection):
            collection.set_linewidth(collection.get_linewidth() / 2)
ax.set_xticks([0, 0.5, 1], ['0', '0.5', '1'])
threat_ax.patch.set_visible(False)
threat_ax.spines['right'].set_linewidth(utility.AXIS_LINE_WIDTH)
threat_ax.spines['right'].set_color(utility.THREAT_COLOR)
threat_ax.spines['left'].set_visible(False)
threat_ax.tick_params(axis='y', colors=utility.THREAT_COLOR)
threat_ax.yaxis.label.set_color(utility.THREAT_COLOR)
ax.tick_params(axis='y', colors=utility.CURIOSITY_COLOR)
ax.yaxis.label.set_color(utility.CURIOSITY_COLOR)
ax.legend(handles=[Line2D([], [], color=utility.CURIOSITY_COLOR, marker='o', ms=3, label='Curiosity'), Line2D([], [], color=utility.THREAT_COLOR, marker='o', ms=3, label='Threat')], loc='upper center', bbox_to_anchor=(0.5, -0.32), ncol=2, fontsize=8, frameon=False, handlelength=0.8, handletextpad=0.3, columnspacing=0.6)
utility.standardize_figure_labels(fig)
fig.tight_layout()
plt.show(block=True)
