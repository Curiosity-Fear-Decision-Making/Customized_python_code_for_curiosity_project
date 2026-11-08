# Foldername: custom_python
# Filename: fig3c_curiosity_threat_choice.py
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from scipy.stats import t
from scipy.stats import f as f_dist, ttest_1samp, ttest_rel
import utility_custom_python as utility
from utility_custom_python import online_find_file as find_file, online_load_online_v6_subjects as load_online_v6_subjects


matplotlib.rcParams.update({'font.family': 'Arial', 'font.size': 8, 'figure.dpi': 160})
fig, ax = plt.subplots(figsize=(3.2, 3.6))
subject_list = load_online_v6_subjects()['subject_list']
plot_bar = []
for s in subject_list:
    f1 = find_file(s, 'result_trial_present')
    if not f1:
        plot_bar.append(np.full((2, 2), np.nan))
        continue
    tp = utility.read_tab_delimited_table(f1)
    yr = pd.to_numeric(tp[:, 12], errors='coerce')
    mv = ~np.isnan(yr)
    tpv = tp[mv]
    yy = (yr[mv].astype(int) == 1).astype(int)
    choice_data_risk = pd.to_numeric(tpv[:, 10], errors='coerce')
    cur = pd.to_numeric(tpv[:, 9], errors='coerce')
    risk_i = np.where(np.isnan(choice_data_risk), -1, choice_data_risk).astype(int)
    cur_i = np.where(np.isnan(cur), -1, cur).astype(int)
    cg = np.where(np.isin(cur_i, [1, 2]), 0, np.where(np.isin(cur_i, [3, 4]), 1, -1))
    rg = np.where(np.isin(risk_i, [1, 2]), 0, np.where(np.isin(risk_i, [3, 4]), 1, -1))
    bm = np.full((2, 2), np.nan)
    for choice_data_a in [0, 1]:
        for b in [0, 1]:
            idx = np.where((cg == choice_data_a) & (rg == b))[0]
            if idx.size > 0:
                bm[choice_data_a, b] = yy[idx].mean()
    plot_bar.append(bm)
plot_bar = np.array(plot_bar, float)

low_c = utility.LIGHT_CURIOSITY_COLOR
high_c = utility.CURIOSITY_COLOR
x_pos = np.array([0.0, 0.75, 2.0, 2.75])
data = [plot_bar[:, 0, 0], plot_bar[:, 1, 0], plot_bar[:, 0, 1], plot_bar[:, 1, 1]]
colors = [low_c, high_c, low_c, high_c]
pdf_w = 0.34
bar_w = 0.6
_rng = np.random.default_rng(0)
means = np.array([np.nanmean(d) for d in data])
stds = np.array([np.nanstd(d[~np.isnan(d)], ddof=1) / np.sqrt(np.sum(~np.isnan(d))) if np.sum(~np.isnan(d)) >= 2 else 0.0 for d in data])
for i in range(4):
    ax.bar(x_pos[i], means[i], width=bar_w, color=colors[i], alpha=0.85, edgecolor='black', linewidth=0.4, zorder=2)
    ys = data[i][~np.isnan(data[i])].astype(float)
    if ys.size:
        jx = x_pos[i] - utility.POINT_OFFSET + _rng.uniform(-bar_w * 0.16, bar_w * 0.16, ys.size)
        ax.scatter(jx, ys, s=3, color=utility.darken_plot_color(colors[i], 0.75), alpha=utility.POINT_ALPHA, edgecolors='none', zorder=3)
    ax.errorbar(x_pos[i], means[i], yerr=stds[i], fmt='none', ecolor='black', elinewidth=utility.ERROR_BAR_LINE_WIDTH, capsize=utility.ERROR_BAR_CAP_SIZE, capthick=utility.ERROR_BAR_LINE_WIDTH, zorder=6)
ax.axhline(0.5, color='gray', lw=0.6, ls='--', zorder=0)

def draw_significance_bracket(x1, x2, y, h, text):
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], color='black', lw=0.6, zorder=8)
    ax.text((x1 + x2) / 2, y + h, text, ha='center', va='bottom', fontsize=utility.ANNOTATION_FONT_SIZE, color='black', zorder=8)
print('\nFig3C | Social-interest cue (1-2 vs 3-4) x threat (1-2 vs 3-4)')
m_all = ~np.isnan(plot_bar).any(axis=(1, 2))
if m_all.sum() >= 2:
    Y = plot_bar[m_all]

    N, a, b = Y.shape
    grand = Y.mean()
    mean_s = Y.mean(axis=(1, 2))
    mean_a = Y.mean(axis=(0, 2))
    mean_b = Y.mean(axis=(0, 1))
    mean_sa = Y.mean(axis=2)
    mean_sb = Y.mean(axis=1)
    mean_ab = Y.mean(axis=0)
    ss_a = N * b * ((mean_a - grand) ** 2).sum()
    ss_sa = b * ((mean_sa - mean_s[:, None] - mean_a[None, :] + grand) ** 2).sum()
    ss_b = N * a * ((mean_b - grand) ** 2).sum()
    ss_sb = a * ((mean_sb - mean_s[:, None] - mean_b[None, :] + grand) ** 2).sum()
    ss_ab = N * ((mean_ab - mean_a[:, None] - mean_b[None, :] + grand) ** 2).sum()
    resid = Y - mean_sa[:, :, None] - mean_sb[:, None, :] - mean_ab[None, :, :] + mean_s[:, None, None] + mean_a[None, :, None] + mean_b[None, None, :] - grand
    ss_sab = (resid ** 2).sum()

    def calculate_anova_effect_statistics(ss, df1, ss_err, df2):
        f = ss / df1 / (ss_err / df2)
        return dict(F=f, df1=df1, df2=df2, p=float(f_dist.sf(f, df1, df2)), peta2=ss / (ss + ss_err))
    aov = {'curiosity': calculate_anova_effect_statistics(ss_a, a - 1, ss_sa, (a - 1) * (N - 1)), 'risk': calculate_anova_effect_statistics(ss_b, b - 1, ss_sb, (b - 1) * (N - 1)), 'interaction': calculate_anova_effect_statistics(ss_ab, (a - 1) * (b - 1), ss_sab, (a - 1) * (b - 1) * (N - 1))}
    for _eff_name, _lab in [('curiosity', 'social-interest cue'), ('risk', 'threat'), ('interaction', 'social-interest cue x threat')]:
        e = aov[_eff_name]
        print(f'Fig3C | RM-ANOVA {_lab}: F({e['df1']},{e['df2']})={e['F']:.2f} p={e['p']:.3g} peta2={e['peta2']:.2f} (n={int(m_all.sum())})')
pair_h = 0.04
pair_top = []
for plo, phi, rg in [(0, 1, 0), (2, 3, 1)]:
    m = ~np.isnan(plot_bar[:, 0, rg]) & ~np.isnan(plot_bar[:, 1, rg])
    t_pair, p = ttest_rel(plot_bar[m, 1, rg], plot_bar[m, 0, rg])
    _hi = plot_bar[m, 1, rg]
    _lo = plot_bar[m, 0, rg]
    _dif = _hi - _lo
    _sd = _dif.std(ddof=1)
    _dz = _dif.mean() / _sd if _sd > 0 else np.nan
    _lab = 'low risk' if rg == 0 else 'high risk'
    print(f'Fig3C | Paired t [{_lab}, high vs low social-interest cue]: n={m.sum()}, mean_high={_hi.mean():.4f}, mean_low={_lo.mean():.4f}, diff={_dif.mean():+.4f}, t({m.sum() - 1})={t_pair:.3f}, p={p:.3g}, dz={_dz:.3f}')
    top = 1.1
    draw_significance_bracket(x_pos[plo], x_pos[phi], top, pair_h, utility.format_significance_label(p))
    pair_top.append(top + pair_h)
m_int = ~np.isnan(plot_bar).any(axis=(1, 2))
d_int = plot_bar[m_int, 1, 0] - plot_bar[m_int, 0, 0] - (plot_bar[m_int, 1, 1] - plot_bar[m_int, 0, 1])
t_int, p_int = ttest_1samp(d_int, 0.0)
_sd_int = d_int.std(ddof=1)
_d_int = d_int.mean() / _sd_int if _sd_int > 0 else np.nan
print(f'Fig3C | Interaction [(high-low social cue | low risk) - (high-low social cue | high risk)]: n={m_int.sum()}, mean={d_int.mean():+.4f}, t({m_int.sum() - 1})={t_int:.3f}, p={p_int:.3g}, d={_d_int:.3f}')
xc_low = (x_pos[0] + x_pos[1]) / 2
xc_high = (x_pos[2] + x_pos[3]) / 2
draw_significance_bracket(xc_low, xc_high, max(pair_top) + 0.23, pair_h, utility.format_significance_label(p_int))
ax.set_xticks([xc_low, xc_high])
ax.set_xticklabels(['Low risk', 'High risk'], fontsize=utility.TICK_FONT_SIZE)
ax.set_ylabel('P(choose story)', fontsize=utility.FONT_SIZE)
ax.set_yticks(np.arange(0, 1.01, 0.2))
ax.set_xlim(x_pos[0] - pdf_w - 0.18, x_pos[3] + pdf_w + 0.18)
ax.set_ylim(0, 1.62)
utility.style_plot_axes(ax)
ax.legend(handles=[Patch(facecolor=low_c, label='Low curiosity'), Patch(facecolor=high_c, label='High curiosity')], loc='upper center', bbox_to_anchor=(0.5, -0.17), ncol=2, borderaxespad=0.0, fontsize=utility.TICK_FONT_SIZE, frameon=False, handlelength=1.0, columnspacing=0.8, labelspacing=0.3)
ax.set_title('Curiosity & Risk effect', fontsize=utility.TITLE_FONT_SIZE)
old = ax.get_legend()
handles = [Patch(facecolor=p.get_facecolor(), label=t.get_text().title()) for p, t in zip(old.legend_handles, old.get_texts())]
old.remove()
ax.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, -0.26), ncol=2, frameon=False, fontsize=8, handlelength=0.8, handletextpad=0.3, borderaxespad=0, labelspacing=0.1)
utility.standardize_figure_labels(fig)
fig.tight_layout()
plt.show()
