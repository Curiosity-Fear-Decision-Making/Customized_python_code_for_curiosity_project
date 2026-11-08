# Foldername: custom_python
# Filename: fig2j_curiosity_threat_choice.py
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import utility_custom_python as utility
from utility_custom_python import spider_filter_excluded_participants as filter_excluded_participants, spider_list_regret_participant_ids as list_regret_participant_ids, spider_load_spider_subject


matplotlib.rcParams.update({'font.family': 'Arial', 'font.size': 8, 'figure.dpi': 160})
fig, ax = plt.subplots(figsize=(3.2, 3.6))
subs = filter_excluded_participants(list_regret_participant_ids())
mats, used = ([], [])
for s in subs:
    rows = spider_load_spider_subject(s)
    yy = np.array([r['y'] for r in rows], float)
    cur = np.array([r['cur_lv'] for r in rows], float)
    rk = np.array([r['risk_lv'] for r in rows], float)
    if yy.size == 0:
        continue
    bm = np.full((3, 3), np.nan)
    for ci, cl in enumerate([1, 2, 3]):
        for ri, rl in enumerate([1, 2, 3]):
            idx = np.where((cur == cl) & (rk == rl))[0]
            if idx.size > 0:
                bm[ci, ri] = yy[idx].mean()
    mats.append(bm)
    used.append(s)
bar3 = np.array(mats, float)
cur_colors = [plt.cm.Purples(v) for v in (0.4, 0.62, 0.85)]
risk_labels = ['Low\nrisk\n(18%)', 'Mid\nrisk\n(36%)', 'High\nrisk\n(54%)']
n_cur, n_risk = (3, 3)
group_x = np.arange(n_risk) * 2.6
bar_w = 0.62
offs = (np.arange(n_cur) - (n_cur - 1) / 2.0) * bar_w
for ci in range(n_cur):
    for ri in range(n_risk):
        xpos = group_x[ri] + offs[ci]
        col = bar3[:, ci, ri]
        col = col[~np.isnan(col)].astype(float)
        if col.size == 0:
            continue
        m = col.mean()
        sem = col.std(ddof=1) / np.sqrt(col.size) if col.size >= 2 else 0.0
        ax.bar(xpos, m, width=bar_w, color=cur_colors[ci], alpha=0.85, edgecolor='black', linewidth=0.4, zorder=2, label='Curiosity %d' % (ci + 1) if ri == 0 else None)
        ax.errorbar(xpos, m, yerr=sem, fmt='none', ecolor='black', elinewidth=utility.ERROR_BAR_LINE_WIDTH, capsize=utility.ERROR_BAR_CAP_SIZE, capthick=utility.ERROR_BAR_LINE_WIDTH, zorder=6)
ax.axhline(0.5, color='gray', lw=0.6, ls='--', zorder=0)
X = bar3[~np.isnan(bar3).any(axis=(1, 2))]
n_c, a, b = X.shape
if n_c >= 3:
    grand = X.mean()
    mA = X.mean(axis=(0, 2))
    mB = X.mean(axis=(0, 1))
    mS = X.mean(axis=(1, 2))
    mAB = X.mean(axis=0)
    mAS = X.mean(axis=2)
    mBS = X.mean(axis=1)
    ssA = n_c * b * ((mA - grand) ** 2).sum()
    ssB = n_c * a * ((mB - grand) ** 2).sum()
    ssS = a * b * ((mS - grand) ** 2).sum()
    ssAB = n_c * ((mAB - mA[:, None] - mB[None, :] + grand) ** 2).sum()
    ssAS = b * ((mAS - mA[None, :] - mS[:, None] + grand) ** 2).sum()
    ssBS = a * ((mBS - mB[None, :] - mS[:, None] + grand) ** 2).sum()
    ssABS = ((X - grand) ** 2).sum() - ssA - ssB - ssS - ssAB - ssAS - ssBS
    for nm, ss, df, sse, dfe in [('curiosity', ssA, a - 1, ssAS, (a - 1) * (n_c - 1)), ('risk', ssB, b - 1, ssBS, (b - 1) * (n_c - 1)), ('cur x risk', ssAB, (a - 1) * (b - 1), ssABS, (a - 1) * (b - 1) * (n_c - 1))]:
        F = ss / df / (sse / dfe)
        p = stats.f.sf(F, df, dfe)
        print('\nFig2J | RM-ANOVA %-10s: F(%d,%d)=%.2f p=%s %s partial eta2=%.2f (n=%d)' % (nm, df, dfe, F, utility.format_p_value(p), utility.format_significance_label(p), ss / (ss + sse), n_c))
for ri in range(n_risk):
    Xr = bar3[:, :, ri]
    Xr = Xr[~np.isnan(Xr).any(axis=1)]
    nr, k = Xr.shape
    if nr >= 3:
        grand = Xr.mean()
        ssc = nr * ((Xr.mean(0) - grand) ** 2).sum()
        sss = k * ((Xr.mean(1) - grand) ** 2).sum()
        sse = ((Xr - grand) ** 2).sum() - ssc - sss
        df1, df2 = (k - 1, (k - 1) * (nr - 1))
        F = ssc / df1 / (sse / df2)
        p = stats.f.sf(F, df1, df2)
        eta = ssc / (ssc + sse)
    else:
        F, df1, df2, p, eta = (np.nan, 0, 0, np.nan, np.nan)
    print('Fig2J | curiosity 1-3 (risk %d): one-way RM-ANOVA F(%d,%d)=%.2f p=%s %s partial eta2=%.2f (n=%d)' % (ri + 1, df1, df2, F, utility.format_p_value(p), utility.format_significance_label(p), eta, nr))
    c3 = bar3[:, 2, ri]
    c3 = c3[~np.isnan(c3)]
    if c3.size >= 3:
        t3, p3 = stats.ttest_1samp(c3, 0.5)
        d3 = (c3.mean() - 0.5) / c3.std(ddof=1)
        print('Fig2J | curiosity 3 vs chance (risk %d): mean=%.3f t(%d)=%.2f p=%s %s d=%.2f (n=%d)' % (ri + 1, c3.mean(), c3.size - 1, t3, utility.format_p_value(p3), utility.format_significance_label(p3), d3, c3.size))
    x1 = group_x[ri] + offs[0]
    x2 = group_x[ri] + offs[2]
    y, h = (1.06, 0.03)
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], color='black', lw=0.6, zorder=8)
    ax.text((x1 + x2) / 2, y + h, utility.format_significance_label(p), ha='center', va='bottom', fontsize=utility.ANNOTATION_FONT_SIZE, color='black', zorder=8)
ax.set_xticks(group_x)
ax.set_xticklabels(risk_labels, fontsize=utility.TICK_FONT_SIZE)
ax.set_ylabel('P(choose story)', fontsize=utility.FONT_SIZE)
ax.set_yticks(np.arange(0, 1.01, 0.2))
ax.set_ylim(0, 1.3)
ax.set_xlim(group_x[0] - 1.2, group_x[-1] + 1.2)
utility.style_plot_axes(ax)
ax.set_title('Curiosity & Risk effect', fontsize=utility.TITLE_FONT_SIZE)
utility.standardize_figure_labels(fig)
fig.tight_layout()
plt.show()
