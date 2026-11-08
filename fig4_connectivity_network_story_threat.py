# Foldername: custom_python
# Filename: fig4_connectivity_network_story_threat.py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy import stats
from tqdm import tqdm
import utility_custom_python as utility
from utility_custom_python import PROJECT_DIR as PACKAGE_DIR


def fit_subject_rsa(frame, patterns):
    coefficients = np.full((3, len(utility.conn_ROIS)), np.nan)
    c = frame.curiosity.to_numpy(float)
    t = frame.threat.to_numpy(float)
    social = frame.social.to_numpy(float)
    story = frame.story.to_numpy(float)
    valid = np.isfinite(np.column_stack([c, t, social, story])).all(axis=1)
    if utility.conn_TRIAL_SAMPLE != 'all':
        valid &= story == (1 if utility.conn_TRIAL_SAMPLE == 'story' else 0)
    ii, jj = np.triu_indices(len(frame), 1)
    keep = valid[ii] & valid[jj] & (c[ii] == c[jj]) & (t[ii] == t[jj])
    ii, jj = (ii[keep], jj[keep])
    for index, pattern in enumerate(patterns):
        p = np.asarray(pattern, dtype=np.float64)
        good = np.isfinite(p[valid]).all(axis=0) & np.any(p[valid] != 0, axis=0)
        p = p[:, good]
        if p.shape[1]:
            p = p - p.mean(axis=1, keepdims=True)
        norms = np.linalg.norm(p, axis=1)
        usable = (norms[ii] > 0) & (norms[jj] > 0)
        i, j = (ii[usable], jj[usable])
        if not len(i):
            continue
        normalized = np.divide(p, norms[:, None], out=np.zeros_like(p), where=norms[:, None] > 0)
        similarity = np.clip((normalized @ normalized.T)[i, j], -1, 1)
        cc = c[i] - c[i].mean()
        tt = t[i] - t[i].mean()
        columns = [np.ones(len(i)), cc, tt, cc * tt]
        controls = [('social_same', (social[i] == social[j]).astype(float)), ('both_story', (story[i] * story[j]).astype(float))]
        if utility.conn_TRIAL_SAMPLE == 'all':
            controls.append(('both_skip', ((1 - story[i]) * (1 - story[j])).astype(float)))
        for _, values in controls:
            if np.ptp(values) > 0:
                columns.append(values - values.mean())
        design = np.column_stack(columns)
        if not utility.conn_ALLOW_RANK_DEFICIENT:
            rank = np.linalg.matrix_rank(design)
            if rank != design.shape[1] or len(i) <= rank:
                continue
        if similarity.std() <= 1e-12:
            continue
        y = (similarity - similarity.mean()) / similarity.std() if utility.conn_ZSCORE_PAIR else similarity
        coefficients[:, index] = np.linalg.lstsq(design, y, rcond=None)[0][1:4]
    return coefficients


def across_subject_correlations(coefficients, sample=None):
    utility.conn_correction_label()
    complete = np.isfinite(coefficients).all(axis=(1, 2))
    x = coefficients[complete]
    n = len(x)
    sample = utility.conn_TRIAL_SAMPLE if sample is None else sample
    tested = np.array([utility.conn_select_seed_connected_edges(e, sample=sample) for e in range(len(utility.conn_EFFECTS))])
    family_sizes = tested.sum(axis=1)
    observed = utility.conn_network_correlations(x)
    valid = np.isfinite(observed)
    maximum = np.array([])
    threshold = np.nan
    if utility.conn_CORRECTION_METHOD == 'max_r':
        rng = np.random.default_rng()
        maximum = np.empty(utility.conn_N_PERM)
        for iteration in tqdm(range(utility.conn_N_PERM), desc=f'Permutations: {sample}', unit='perm', leave=False):
            shuffled = np.empty_like(x)
            for j in range(len(utility.conn_ROIS)):
                shuffled[:, :, j] = x[rng.permutation(n), :, j]
            maximum[iteration] = np.max(np.abs(utility.conn_network_correlations(shuffled)[valid]))
        threshold = float(np.quantile(maximum, 0.95, method='higher'))
    utility.PERMUTATION_MAX_ABS_R = maximum
    utility.PERMUTATION_THRESHOLD_95 = threshold
    rows = []
    for e, effect in enumerate(utility.conn_EFFECTS):
        for k, (i, j) in enumerate(utility.conn_EDGES):
            eligible = bool(tested[e, k])
            raw_r = observed[e, k]
            r = raw_r
            if np.isfinite(r):
                p = float(stats.pearsonr(x[:, e, i], x[:, e, j]).pvalue)
                max_p = (1 + np.sum(maximum >= abs(r))) / (utility.conn_N_PERM + 1) if len(maximum) else np.nan
                bonferroni_p = min(1.0, p * utility.conn_FIXED_BONFERRONI_TESTS)
                corrected = {'bonferroni': bonferroni_p, 'max_r': max_p, 'uncorrected': p}[utility.conn_CORRECTION_METHOD]
            else:
                p = corrected = max_p = bonferroni_p = np.nan
            rows.append(dict(effect=effect, roi1=list(utility.conn_ROIS)[i], roi2=list(utility.conn_ROIS)[j], n=n, r=r, p_pearson=p, df=n - 2, network_method='Across-participant Pearson correlation', p_fwer_max_r=max_p, p_bonferroni=bonferroni_p, p_corrected=corrected, correction_method=utility.conn_CORRECTION_METHOD, threshold_95=threshold, tested=eligible, seeds=','.join(sorted(utility.conn_select_effect_seed_regions(e, sample=sample))), tested_seed_pairs=int(family_sizes[e]), correlation_family_size=utility.conn_FIXED_BONFERRONI_TESTS, total_bonferroni_tests=utility.conn_FIXED_BONFERRONI_TESTS, significant_display=bool(eligible and corrected < utility.conn_SIGNIFICANCE_ALPHA)))
    return (pd.DataFrame(rows), complete)


plt.rcParams.update({'font.family': 'Arial', 'font.size': 8})
names = list(utility.conn_ROIS)
edges = utility.conn_EDGES
results = {}
coefficients = {}
for period in ('decision', 'shock'):
    values = {'story': [], 'skip': []}
    for subject in tqdm(utility.conn_SUBJECTS, desc=f'First level: {period}', unit='subject'):
        frame = utility.conn_load_participant_trial_conditions(subject)
        with np.load(PACKAGE_DIR / 'connectivity_inputs' / f'{period}_{subject}.npz') as inputs:
            patterns = [inputs[name] for name in names]
        for sample in ('story', 'skip'):
            utility.conn_TRIAL_SAMPLE = sample
            keep = frame.story.to_numpy() == (1 if sample == 'story' else 0)
            beta = fit_subject_rsa(frame.loc[keep].reset_index(drop=True), [p[keep] for p in patterns])
            values[sample].append(beta)
    results[period] = {}
    for sample in tqdm(('story', 'skip'), desc=f'Second level: {period}', unit='choice'):
        coefficients[period, sample] = np.asarray(values[sample])
        results[period][sample], complete = across_subject_correlations(coefficients[period, sample], sample=sample)
pairs = set()
for period_results in results.values():
    for trial_sample, table in period_results.items():
        for index, effect in enumerate(utility.conn_EFFECTS):
            selected = utility.conn_comparison_display_mask(table.loc[table.effect == effect], index, trial_sample)
            pairs.update(((utility.conn_display_roi(names[i]), utility.conn_display_roi(names[j])) for k, (i, j) in enumerate(edges) if selected[k]))
utility.conn__COMPARISON_LAYOUT_EDGES = pd.DataFrame(sorted(pairs), columns=['source_display', 'target_display'])
sample, effect_index = ('story', 1)
selections = {}
for period in ('decision', 'shock'):
    table = results[period][sample]
    rows = table.loc[table.effect == utility.conn_EFFECTS[effect_index]]
    selections[period] = utility.conn_comparison_display_mask(rows, effect_index, sample)
fig, ax = plt.subplots(figsize=(5, 5))
utility.conn_draw_perimeter_panel(fig, ax, selections, effect_index, utility.conn_perimeter_brain_geometry())
ax.set_title(f'{sample.capitalize()} · ' + ('Curiosity', 'Threat', 'Curiosity × threat')[effect_index])
ax.legend(handles=[Line2D([], [], color='#555555', linestyle='-', label='Decision'), Line2D([], [], color='#555555', linestyle='--', label='Shock')], loc='lower center', frameon=False)
fig.tight_layout()
utility.conn_update_period_edge_spacing(fig)
plt.show()
