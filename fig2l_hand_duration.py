# Foldername: custom_python
# Filename: fig2l_hand_duration.py
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats as sps
from statsmodels.regression.mixed_linear_model import MixedLM
import utility_custom_python as utility
from utility_custom_python import spider_align_video_events_with_trial_log, spider_load_hand_video_events, spider_parse_participant_duration_log, spider_spider_analysis_duration_CURS as curs, spider_spider_analysis_duration_GROUPS as groups


matplotlib.rcParams.update({'font.family': 'Arial', 'font.size': 8, 'figure.dpi': 160})
fig, ax = plt.subplots(figsize=(3.2, 3.6))
data = spider_load_hand_video_events()
if not data:
    raise ValueError('No hand-duration video events were found.')
cell = {(g, c): {} for g in groups for c in curs}
grp_all = {g: {} for g in groups}
for sub in sorted(data):
    box2grp, trials = spider_parse_participant_duration_log(sub)
    if box2grp is None or len(box2grp) < 4:
        continue
    expected = [t['box'] for t in trials]
    assign, missing, bad = spider_align_video_events_with_trial_log(data[sub], expected)
    for t, box, dur in data[sub]:
        j = assign.get(t)
        if j is None:
            continue
        grp = box2grp[box]
        cur = trials[j - 1]['cur']
        cell[grp, cur].setdefault(sub, []).append(dur)
        grp_all[grp].setdefault(sub, []).append(dur)
cell_m = {k: {s: float(np.mean(v)) for s, v in d.items()} for k, d in cell.items()}
grp_m = {g: {s: float(np.mean(v)) for s, v in d.items()} for g, d in grp_all.items()}
thr_avg = {}
for g in groups[1:]:
    for s, d in grp_all[g].items():
        thr_avg.setdefault(s, []).extend(d)
thr_avg = {s: float(np.mean(v)) for s, v in thr_avg.items()}

cur_colors = [plt.cm.Purples(v) for v in (0.4, 0.62, 0.85)]
grp_labels = ['Control\n(0%)', 'Low\nrisk\n(18%)', 'Mid\nrisk\n(36%)', 'High\nrisk\n(54%)']
group_x = np.arange(len(groups)) * 2.6
bar_w = 0.62
offs = (np.arange(3) - 1.0) * bar_w
tops = {}
for gi, g in enumerate(groups):
    for ci, c in enumerate(curs):
        col = np.array(list(cell_m[g, c].values()), float)
        if col.size == 0:
            continue
        m = col.mean()
        sem = col.std(ddof=1) / np.sqrt(col.size) if col.size >= 2 else 0.0
        xpos = group_x[gi] + offs[ci]
        ax.bar(xpos, m, width=bar_w, color=cur_colors[ci], alpha=0.85, edgecolor='black', linewidth=0.4, zorder=2, label='Curiosity %d' % c if gi == 0 else None)
        ax.errorbar(xpos, m, yerr=sem, fmt='none', ecolor='black', elinewidth=utility.ERROR_BAR_LINE_WIDTH, capsize=utility.ERROR_BAR_CAP_SIZE, capthick=utility.ERROR_BAR_LINE_WIDTH, zorder=6)
        tops[gi, ci] = m + sem
ymax = max(tops.values())
control = grp_m['control']
paired_subjects = sorted(set(control) & set(thr_avg))
n = len(paired_subjects)
res = None
if n >= 3 and sps is not None:
    res = sps.ttest_rel([control[s] for s in paired_subjects],
                        [thr_avg[s] for s in paired_subjects])
p_ctrl = res[1] if res else np.nan
if res:
    print('\nFig2L | RT control vs risk-avg: t(%d)=%+.2f p=%s %s (n=%d)' % (n - 1, res[0], utility.format_p_value(p_ctrl), utility.format_significance_label(p_ctrl), n))
condition_means = [grp_m[g] for g in groups[1:]]

observations = [(subject, condition, value) for condition, values in enumerate(condition_means) for subject, value in values.items() if np.isfinite(value)]
subjects, conditions, durations = np.array(observations).T
predictors = np.column_stack([np.ones(len(durations)), conditions == 1, conditions == 2])
full_model = MixedLM(durations, predictors, groups=subjects).fit(reml=False, method='powell', disp=False)
null_model = MixedLM(durations, np.ones((len(durations), 1)), groups=subjects).fit(reml=False, method='powell', disp=False)
likelihood_ratio = max(0.0, 2 * (full_model.llf - null_model.llf))
p_threat = sps.chi2.sf(likelihood_ratio, 2)
print(f'Fig2L | Threat RT mixed model: participants={len(np.unique(subjects))}, observations={len(durations)}, likelihood-ratio chi2(2)={likelihood_ratio:.4f}, p={p_threat:.6g}; converged={full_model.converged and null_model.converged}')
h = ymax * 0.022
for x1, x2, p, y in [(group_x[0], group_x[2], p_ctrl, ymax * 0.96), (group_x[1], group_x[3], p_threat, ymax * 1.21)]:
    if not np.isfinite(p):
        continue
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], color='black', lw=0.6, zorder=8)
    ax.text((x1 + x2) / 2, y + h, utility.format_significance_label(p), ha='center', va='bottom', fontsize=utility.ANNOTATION_FONT_SIZE, zorder=8)
ax.set_xticks(group_x)
ax.set_xticklabels(grp_labels, fontsize=utility.TICK_FONT_SIZE)
ax.set_ylabel('Reaction time (s)', fontsize=utility.FONT_SIZE)
ax.set_ylim(0, ymax * 1.34)
ax.set_xlim(group_x[0] - 1.2, group_x[-1] + 1.2)
utility.style_plot_axes(ax)
ax.set_title('Curiosity & Risk on reaction time', fontsize=utility.TITLE_FONT_SIZE)
utility.standardize_figure_labels(fig)
fig.tight_layout()
plt.show()
