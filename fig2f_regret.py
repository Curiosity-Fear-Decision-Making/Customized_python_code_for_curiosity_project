# Foldername: custom_python
# Filename: fig2f_regret.py
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy import stats
import utility_custom_python as utility
from utility_custom_python import spider_DATA_DIR, spider_filter_excluded_participants as filter_excluded_participants, spider_list_regret_participant_ids as list_regret_participant_ids, spider_parse_regret_rating


matplotlib.rcParams.update({'font.family': 'Arial', 'font.size': 8, 'figure.dpi': 160})
fig, ax = plt.subplots(figsize=(3.2, 3.6))
subs = filter_excluded_participants(list_regret_participant_ids())

# Pair the ratings and fit each participant.
fits = []
for sub_id in subs:
    f_tr = os.path.join(spider_DATA_DIR, 'output_spider_trials_%s.txt' % sub_id)
    xs, ys = ([], [])
    with open(f_tr) as fh:
        for line in fh:
            parts = line.rstrip('\n').split('\t')
            if len(parts) < 12:
                continue
            if parts[7] != 'skip':
                continue
            want = spider_parse_regret_rating(parts[8], 'r_wanttoknow')
            regret = spider_parse_regret_rating(parts[11], 'r_regret')
            if not (np.isfinite(want) and np.isfinite(regret)):
                continue
            xs.append(want)
            ys.append(regret)
    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)
    fitted = dict(x=x, y=y)
    fits.append((sub_id, fitted))

# Plot the participant fits and test the group slope.
xlabel = 'Want-to-know rating'
ylabel = 'Regret rating'
title = 'Regret vs want-to-know'
scatter_color = '#9c8cc0'
mean_line_color = '#4b2e83'
persub_color = '#cfc7df'
err_color = '#2d1a55'

if fits:
    x_all = np.concatenate([fit['x'] for _, fit in fits])
    y_all = np.concatenate([fit['y'] for _, fit in fits])
    sid_all = np.concatenate([np.full(fit['x'].size, sub) for sub, fit in fits])
    x_all = np.asarray(x_all, float)
    y_all = np.asarray(y_all, float)
    sid_all = np.asarray(sid_all)
    ok = np.isfinite(x_all) & np.isfinite(y_all)
    x_all, y_all, sid_all = (x_all[ok], y_all[ok], sid_all[ok])
    if x_all.size:
        xr = np.round(x_all).astype(int)
        x_levels = np.unique(xr)
        sids = np.unique(sid_all)
        persub_meany = np.full((len(x_levels), len(sids)), np.nan)
        slopes, intercepts = ([], [])
        for ci, sid in enumerate(sids):
            m = sid_all == sid
            sx = xr[m].astype(float)
            sy = y_all[m].astype(float)
            if sx.size == 0:
                continue
            for xi, xv in enumerate(x_levels):
                cell = sy[sx == xv]
                if cell.size > 0:
                    persub_meany[xi, ci] = cell.mean()
            if np.unique(sx).size >= 2:
                b, a = np.polyfit(sx, sy, 1)
                slopes.append(b)
                intercepts.append(a)
        slopes = np.array(slopes)
        intercepts = np.array(intercepts)
        xmin, xmax = (x_levels.min(), x_levels.max())
        xx = np.linspace(xmin - 0.5, xmax + 0.5, 50)
        for b, a in zip(slopes, intercepts):
            ax.plot(xx, a + b * xx, color=persub_color, lw=utility.PARTICIPANT_LINE_WIDTH, alpha=0.35, zorder=2)
        rng = np.random.default_rng(0)
        sc_x_raw = np.tile(x_levels.astype(float).reshape(-1, 1), (1, persub_meany.shape[1]))
        keep = ~np.isnan(persub_meany)
        sc_x = sc_x_raw[keep] + rng.uniform(-0.1, 0.1, keep.sum())
        sc_y = persub_meany[keep]
        ax.scatter(sc_x, sc_y, s=utility.POINT_SIZE, color=scatter_color, alpha=0.28, edgecolors='none', zorder=3)
        ex, ey, es = ([], [], [])
        for xi, xv in enumerate(x_levels):
            col = persub_meany[xi]
            col = col[~np.isnan(col)]
            if col.size == 0:
                continue
            ex.append(xv)
            ey.append(col.mean())
            es.append(col.std(ddof=1) / (np.sqrt(col.size) if utility.PLOT_ERROR_SUMMARY == 'sem' else 1.0) if col.size >= 2 else 0.0)
        ex = np.asarray(ex, float)
        ey = np.asarray(ey, float)
        es = np.asarray(es, float)
        if slopes.size:
            mean_line = intercepts.mean() + slopes.mean() * xx
            if ex.size >= 2:
                std_c = np.interp(xx, ex, es, left=es[0], right=es[-1])
                ax.fill_between(xx, mean_line - std_c, mean_line + std_c, facecolor=mean_line_color, alpha=0.18, edgecolor='none', zorder=4)
            ax.plot(xx, mean_line, color=mean_line_color, lw=utility.MEAN_LINE_WIDTH, zorder=6)
        ax.scatter(ex, ey, s=utility.MEAN_POINT_SIZE, facecolors=err_color, edgecolors='white', linewidths=0.5, zorder=7)
        if slopes.size > 1:
            _t_slope, p_slope = stats.ttest_1samp(slopes, 0.0)
            _sd = slopes.std(ddof=1)
            _d = slopes.mean() / _sd if _sd > 0 else np.nan
            print(f'\nFig2F | slope [{xlabel} -> {ylabel}]: mean={slopes.mean():+.4f} t({len(slopes) - 1})={_t_slope:.2f} p={p_slope:.3g} d={_d:.2f} (n={len(slopes)})')
            star = utility.format_significance_label(p_slope)
            if star:
                ax.text(0.05, 0.96, star, transform=ax.transAxes, ha='left', va='top', fontsize=utility.TITLE_FONT_SIZE, fontweight='bold', color=mean_line_color)
        ax.set_xlabel(xlabel, fontsize=utility.FONT_SIZE)
        ax.set_ylabel(ylabel, fontsize=utility.FONT_SIZE)
        ax.set_xticks(x_levels)
        ax.set_xlim(xmin - 0.7, xmax + 0.7)
        ax.set_ylim(0.5, 6.5)
        utility.style_plot_axes(ax)
    else:
        utility.style_plot_axes(ax)
else:
    ax.text(0.5, 0.5, 'no data', ha='center', va='center', transform=ax.transAxes, fontsize=utility.FONT_SIZE)
    utility.style_plot_axes(ax)
ax.set_title(title, fontsize=utility.TITLE_FONT_SIZE)
utility.standardize_figure_labels(fig)
fig.tight_layout()
plt.show()
