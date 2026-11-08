# Foldername: custom_python
# Filename: utility_custom_python.py
import glob
import matplotlib.colors as mcolors
import nibabel as nib
import numpy as np
import os
import pandas as pd
import re
import statsmodels.api as sm
import sys
import textwrap
import warnings
import xml.etree.ElementTree as ET
from collections import defaultdict
from contextlib import redirect_stdout
from functools import cache
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.text import Text
from matplotlib.ticker import FuncFormatter
from nibabel.processing import resample_from_to
from pathlib import Path
from scipy.ndimage import gaussian_filter
from scipy.stats import t
from statsmodels.genmod.cov_struct import Exchangeable
PROJECT_DIR = Path(__file__).resolve().parent
FONT_SIZE = 8
TICK_FONT_SIZE = 8
AXIS_LINE_WIDTH = 0.6
TITLE_FONT_SIZE = 8
ANNOTATION_FONT_SIZE = 8
MEAN_LINE_WIDTH = 1.2
LARGE_MEAN_POINT_SIZE = 16
ERROR_BAR_LINE_WIDTH = 0.8
ERROR_BAR_CAP_SIZE = 1.6
POINT_ALPHA = 0.22
POINT_OFFSET = 0.13
CURIOSITY_COLOR = '#6a51a3'
LIGHT_CURIOSITY_COLOR = '#bcbddc'
PANEL_LETTER_FONT_SIZE = 8
PARTICIPANT_LINE_WIDTH = 0.35
POINT_SIZE = 4
MEAN_POINT_SIZE = 12
THREAT_COLOR = '#cb181d'
PLOT_ERROR_SUMMARY = 'sd'
LIGHT_THREAT_COLOR = '#fb6a4a'
DARK_THREAT_COLOR = '#a50f15'
INTERACTION_POINT_COLOR = '#9c8cc0'
INTERACTION_LINE_COLOR = '#4b2e83'
INTERACTION_FAINT_COLOR = '#cfc7df'
INTERACTION_MARKER_COLOR = '#2d1a55'
OUTLIER_SD_THRESHOLD = 3
POSITIVE_GAP_COLOR = '#e08214'
NEGATIVE_GAP_COLOR = '#2171b5'
CURIOSITY_COLOR_FIGS1_3_BEHAVIORAL_VALIDATION = '#2171b5'
LIGHT_CURIOSITY_COLOR_FIGS1_3_BEHAVIORAL_VALIDATION = '#9ecae1'

def format_significance_label(p, missing_label=''):
    if not np.isfinite(p):
        return missing_label
    if p < 0.001:
        return '***'
    if p < 0.01:
        return '**'
    if p < 0.05:
        return '*'
    return 'n.s.'

def style_plot_axes(ax):
    for a in ['top', 'right']:
        ax.spines[a].set_linewidth(0)
    for a in ['bottom', 'left']:
        ax.spines[a].set_linewidth(AXIS_LINE_WIDTH)
    ax.tick_params(axis='both', labelsize=TICK_FONT_SIZE, width=AXIS_LINE_WIDTH, length=2.2)

def darken_plot_color(c, f=0.6):
    r, g, b = mcolors.to_rgb(c)
    return (r * f, g * f, b * f)

def standardize_values(x):
    x = np.asarray(x, dtype=float).ravel()
    s = x.std()
    return (x - x.mean()) / s if s > 0 else x - x.mean()

def format_p_value(p_value):
    if not np.isfinite(p_value):
        return 'NA'
    if p_value >= 0.0001:
        return f'{p_value:.4f}'
    return f'{p_value:.1e}'

def normalize_label(text):
    text = re.sub('\\bclickbait\\s*\\(self(?:[- ]interest)?\\)', 'Self-rating', text, flags=re.I)
    text = re.sub('\\bown interest rating\\b', 'Self-rating', text, flags=re.I)
    text = re.sub('\\bself(?:[- ]interest)?(?:[- ]rating)?\\b', 'Self-rating', text, flags=re.I)
    text = re.sub('\\bsocial(?!\\s+curios)(?:[- ]interest)?(?:[- ]rating)?\\b', 'social-rating', text, flags=re.I)
    text = re.sub('\\b(?:risk|threat)\\b', 'Threat', text, flags=re.I)
    text = re.sub('(?<=\\n)clickbait\\b', 'Self-rating', text, flags=re.I)
    text = re.sub('\\bclickbait(?=\\s*\\+)', 'Self-rating', text, flags=re.I)
    return text

def remove_invisible_characters(s):
    for c in ('\u200b', '\u200c', '\u200d', '\ufeff', ' '):
        s = s.replace(c, '')
    return s

def lighten_plot_color(c, white_fraction=0.85):
    return tuple(((1 - white_fraction) * v + white_fraction for v in mcolors.to_rgb(c)))

def read_tab_delimited_table(path):
    return pd.read_csv(path, delimiter='\t', header=None, engine='python').dropna(axis=1, how='all').to_numpy()

def bonferroni_adjust_p_values(p, family_size):
    return np.minimum(np.asarray(p) * family_size, 1.0)

def calculate_mean_of_finite_values(values):
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    return float(values.mean()) if values.size else np.nan

def standardize_mediation_predictor(x):
    x = np.asarray(x, float)
    s = np.nanstd(x, ddof=1)
    return (x - np.nanmean(x)) / s if s > 1e-12 else np.zeros_like(x)

def standardize_figure_labels(fig):
    fig.canvas.draw()
    for artist in fig.findobj(match=Text):
        old = artist.get_text()
        new = normalize_label(old)
        if new != old:
            artist.set_text(new)
    for ax in fig.axes:
        title = ax.get_title()
        width = max(24, int(ax.get_position().width * fig.get_figwidth() * 15))
        ax.set_title('\n'.join((textwrap.fill(line, width=width, break_long_words=False, break_on_hyphens=False) for line in title.split('\n'))))
        for axis in (ax.xaxis, ax.yaxis):
            formatter = axis.get_major_formatter()

            def format_title_case_tick_label(value, pos, original=formatter):
                label = normalize_label(original(value, pos))
                if '\n' in label and '×' in label:
                    label = label.replace('×', '×\n')
                return label
            axis.set_major_formatter(FuncFormatter(format_title_case_tick_label))
        if any(('β' in t.get_text() or 'γ' in t.get_text() for t in ax.get_xticklabels())):
            for tick in ax.get_xticklabels():
                tick.set_rotation(60 if len(ax.get_xticklabels()) >= 5 else 30)
                tick.set_ha('right')
        legend = ax.get_legend()
        if legend:
            for label in legend.get_texts():
                value = label.get_text()
                if 'Self-rating' in value and 'social-rating' in value:
                    value = re.sub('\\s*([<>=])\\s*', ' \\1\\n', value)
                    label.set_text(value)
utility = sys.modules[__name__]
spider_DATA_DIR = str(PROJECT_DIR / 'beh_spider')
spider_EXCLUDE_FILE = os.path.join(spider_DATA_DIR, 'spider_excluded_subjects.txt')
spider_duration_MANUAL_DIR = os.path.join(spider_DATA_DIR, 'hand_manual')
spider_RISK_LEVELS = [1, 2, 3]
spider_CUR_LO_LV, spider_CUR_HI_LV = ([1], [3])
spider_RISK_LO_LV, spider_RISK_HI_LV = ([1], [3])
spider_spider_RATING_LEVELS = [1, 2, 3, 4, 5, 6]
spider_scare_RATING_LEVELS = [1, 2, 3, 4, 5, 6]
spider_duration_E_TO_BOX = {'E1': 'box1', 'E2': 'box2', 'E3': 'box3'}
spider_duration_R_TO_COND = {'r1': 'risk 1', 'r2': 'risk 2', 'r3': 'risk 3'}
spider_spider_analysis_duration_GROUPS = ['control', 'risk 1', 'risk 2', 'risk 3']
spider_spider_analysis_duration_CURS = [1, 2, 3]


def spider_list_spider_participant_ids():
    subs = []
    for f in os.listdir(spider_DATA_DIR):
        m = re.match('output_spider_trials_(\\w+)\\.txt$', f)
        if m:
            subs.append(m.group(1))
    return sorted(subs)

def spider_load_spider_subject(sub_id):
    f_tr = os.path.join(spider_DATA_DIR, 'output_spider_trials_%s.txt' % sub_id)
    f_rt = os.path.join(spider_DATA_DIR, 'output_rating_%s.txt' % sub_id)
    own_rating = defaultdict(list)
    with open(f_rt) as fh:
        for line in fh:
            parts = line.rstrip('\n').split('\t')
            if len(parts) < 4:
                continue
            own_rating[int(parts[1])].append(float(parts[3]))
    own_rating = {k: float(np.mean(v)) for k, v in own_rating.items()}
    risk_pct = {'r1': 18.0, 'r2': 36.0, 'r3': 54.0}
    rows = []
    with open(f_tr) as fh:
        for line in fh:
            parts = line.rstrip('\n').split('\t')
            if len(parts) < 10:
                continue
            choice = parts[7]
            if choice not in ('story', 'skip'):
                continue
            cb_idx = int(parts[1])
            m_c = re.match('c(\\d+)$', parts[5])
            m_r = re.match('r(\\d+)$', parts[6])
            m_s = re.match('r_scared(\\d+)$', parts[9]) if len(parts) > 9 else None
            rows.append(dict(cb=cb_idx, y=1 if choice == 'story' else 0, risk=risk_pct.get(parts[6], np.nan), cur_lv=int(m_c.group(1)) if m_c else np.nan, risk_lv=int(m_r.group(1)) if m_r else np.nan, scared=int(m_s.group(1)) if m_s else np.nan, clickbait=own_rating.get(cb_idx, np.nan)))
    return rows


def spider_read_excluded_subjects(as_int=False):
    ids = set()
    with open(spider_EXCLUDE_FILE) as fh:
        for line in fh:
            s = line.strip()
            if not s or s.startswith('#'):
                continue
            ids.add(s)
    if as_int:
        nums = set()
        for s in ids:
            m = re.search('(\\d+)', s)
            if m:
                nums.add(int(m.group(1)))
        return nums
    return ids

def spider_filter_excluded_participants(subs, label=''):
    excl = spider_read_excluded_subjects()
    kept = [s for s in subs if s not in excl]
    return kept


def spider_parse_consistency_rating(token, prefix):
    m = re.match('%s(\\d+|nan)$' % prefix, token)
    if m is None or m.group(1) == 'nan':
        return float('nan')
    return float(m.group(1))


def spider_list_regret_participant_ids():
    subs = []
    for f in os.listdir(spider_DATA_DIR):
        m = re.match('output_spider_trials_(\\w+)\\.txt$', f)
        if m:
            subs.append(m.group(1))
    return sorted(subs)

def spider_parse_regret_rating(token, prefix):
    m = re.match('%s(\\d+|nan)$' % prefix, token)
    if m is None or m.group(1) == 'nan':
        return float('nan')
    return float(m.group(1))


def spider_parse_participant_duration_log(sub):
    p = os.path.join(spider_DATA_DIR, f'output_spider_trials_sub{sub:02d}.txt')
    if not os.path.exists(p):
        return (None, None)
    box2grp, trials = ({}, [])
    with open(p, encoding='utf-8') as fh:
        for ln in fh:
            c = ln.rstrip('\n').split('\t')
            if len(c) < 10:
                continue
            box = spider_duration_E_TO_BOX.get(c[3])
            grp = spider_duration_R_TO_COND.get(c[6])
            mc = re.match('c([123])$', c[5])
            if not (box and grp and mc):
                continue
            box2grp[box] = grp
            trials.append({'box': box if c[7] == 'story' else 'box4', 'cur': int(mc[1]), 'choice': c[7]})
    box2grp['box4'] = 'control'
    return (box2grp, trials)

def spider_load_hand_video_events():
    data = {}
    for f in sorted(glob.glob(os.path.join(spider_duration_MANUAL_DIR, 'par*_events.tsv'))):
        m = re.match('par(\\d+)_events\\.tsv$', os.path.basename(f))
        if not m:
            continue
        sub = int(m[1])
        rows = []
        with open(f, encoding='utf-8') as fh:
            for ln in fh:
                c = ln.rstrip('\n').split('\t')
                if len(c) < 5 or c[0] in ('sub', ''):
                    continue
                trial = int(c[1])
                box = c[2]
                dur = float(c[4]) - float(c[3])
                if re.match('box[1-4]$', box) and dur > 0:
                    rows.append((trial, box, dur))
        if rows:
            data[sub] = rows
    return data

def spider_align_video_events_with_trial_log(rows, expected):
    rows = sorted(rows)
    L, E = (len(rows), len(expected))
    if L == 0 or E == 0 or L > E:
        return ({}, [], [(t, b, None, None) for t, b, _ in rows])
    INF = 10 ** 9
    dp = [[INF] * (E + 1) for _ in range(L + 1)]
    dp[0][0] = 0
    for j in range(1, E + 1):
        dp[0][j] = 0
    for i in range(1, L + 1):
        for j in range(i, E + 1):
            skip = dp[i][j - 1]
            take = dp[i - 1][j - 1] + (rows[i - 1][1] != expected[j - 1])
            dp[i][j] = min(skip, take)
    assign, i, j = ({}, L, E)
    while i > 0:
        take = dp[i - 1][j - 1] + (rows[i - 1][1] != expected[j - 1])
        if dp[i][j] == take and j >= i:
            assign[rows[i - 1][0]] = j
            i, j = (i - 1, j - 1)
        else:
            j -= 1
    used = set(assign.values())
    missing = [j for j in range(1, E + 1) if j not in used]
    bad = [(t, b, assign[t], expected[assign[t] - 1]) for t, b, _ in rows if b != expected[assign[t] - 1]]
    return (assign, missing, bad)


warnings.filterwarnings('ignore')
spider_RATING_LEVELS = [1, 2, 3, 4, 5, 6]
spider_SCARED_MIN, spider_SCARED_MAX = (1, 6)


online_INPUT_DIR = PROJECT_DIR / 'beh_online_seeking_exp'
online_DIR_FILES = str(online_INPUT_DIR) + '/'

def online_load_online_v6_subjects(dir_files=online_DIR_FILES, min_trials=5, require_rating_session=True):
    if not dir_files.endswith('/'):
        dir_files = dir_files + '/'
    all_files = os.listdir(dir_files)
    _clean_map = {f: utility.remove_invisible_characters(f) for f in all_files}
    _tp_files = [f for f in all_files if 'result_trial_present' in _clean_map[f] and _clean_map[f].endswith('.txt') and ('sub' in _clean_map[f])]
    subject_all = sorted({_clean_map[f].split('sub', 1)[1].rsplit('.txt', 1)[0] for f in _tp_files})
    excluded = []
    subject_valid = []
    for sub_id in subject_all:
        trial_paths = [f for f in all_files if 'result_trial_present' in _clean_map[f] and sub_id in _clean_map[f]]
        rating_paths = [f for f in all_files if 'result_Rating_session' in _clean_map[f] and sub_id in _clean_map[f]]
        if require_rating_session and len(rating_paths) == 0:
            excluded.append((sub_id, 'missing result_Rating_session file'))
            continue
        trial_arr = pd.read_csv(dir_files + trial_paths[0], delimiter='\t', header=None, engine='python').dropna(axis=1, how='all').to_numpy()
        if trial_arr.shape[1] <= 12:
            excluded.append((sub_id, 'trial_present has no column 12 (response)'))
            continue
        yy_raw = pd.to_numeric(trial_arr[:, 12], errors='coerce')
        yy_valid = yy_raw[~np.isnan(yy_raw)].astype(int)
        n_valid = int(len(yy_valid))
        if n_valid < min_trials:
            excluded.append((sub_id, f'only {n_valid} valid trials (min={min_trials})'))
            continue
        uniq = np.unique(yy_valid)
        if len(uniq) == 1:
            only = int(uniq[0])
            label_map = {1: 'story', 2: 'skip'}
            label = label_map.get(only, f'value={only}')
            excluded.append((sub_id, f'degenerate: all responses = {only} ({label})'))
            continue
        subject_valid.append(sub_id)
    return dict(subject_list=subject_valid, n_total=len(subject_all), n_kept=len(subject_valid), excluded=excluded, dir_files=dir_files)


@cache
def online_list_files():
    return os.listdir(online_INPUT_DIR)


online_SCALE_PANELS = [('5DC_JoyousExploration', '5DC\nJoyous Exploration', '#2980b9'), ('5DC_DeprivationSensitivity', '5DC\nDeprivation Sens.', '#27ae60'), ('5DC_StressTolerance', '5DC\nStress Tolerance', '#c0392b'), ('5DC_SocialCuriosity', '5DC\nSocial Curiosity', '#7f8c8d'), ('5DC_ThrillSeeking', '5DC\nThrill Seeking', '#f39c12'), ('DASS_Anxiety', 'DASS\nAnxiety', '#16a085'), ('DASS_Stress', 'DASS\nStress', '#d35400'), ('DASS_Depression', 'DASS\nDepression', '#8e44ad'), ('STICSA_Somatic', 'STICSA\nSomatic', '#e67e22'), ('STICSA_Cognitive', 'STICSA\nCognitive', '#1abc9c')]
online_SEED = 20260926
online_CURIOSITY = ['5DC_JoyousExploration', '5DC_DeprivationSensitivity', '5DC_SocialCuriosity', '5DC_ThrillSeeking']
online_DISTRESS = ['DASS_Anxiety', 'DASS_Stress', 'DASS_Depression', 'STICSA_Somatic', 'STICSA_Cognitive']
online_DASS_SCALES = ['DASS_Anxiety', 'DASS_Stress', 'DASS_Depression']
online_DASS_OFFSET = 14
online_N_BOOT = 5000
online_KEYS = ['5DC_JoyousExploration', '5DC_DeprivationSensitivity', '5DC_SocialCuriosity', '5DC_StressTolerance', '5DC_ThrillSeeking']
warnings.filterwarnings('ignore')


def online_find_file(sub_id, keyword, exclude=None):
    sid = sub_id
    for f in online_list_files():
        fn = f.replace('\u200b', '')
        if sid in fn and keyword in f and (exclude is None or exclude not in f):
            return online_DIR_FILES + f
    return None


def online_read_questionnaire_responses(cur_sub_id, keyword):
    path = [online_DIR_FILES + f for f in online_list_files() if cur_sub_id in f.replace('\u200b', '') and keyword in f][0]
    return pd.read_csv(path, sep='\t', header=None, engine='c', names=[0, 1, 2, 3], usecols=[0, 1, 2]).dropna(axis=0, how='all').to_numpy()

def online_score_five_dimensional_curiosity_scale(values):
    values = np.asarray(values, float)
    values = values.copy()
    values[(values < 1) | (values > 7) | ~np.isfinite(values)] = np.nan
    values[10:15] = 8 - values[10:15]
    return {name: float(values[5 * i:5 * i + 5].mean()) for i, (name, _, _) in enumerate(online_SCALE_PANELS[:5])}

def online_score_dass_and_sticsa_scales(sub):
    groups = [('result_dass21', [('DASS_Anxiety', [1, 3, 6, 8, 14, 18, 19]), ('DASS_Stress', [0, 5, 7, 10, 11, 13, 17]), ('DASS_Depression', [2, 4, 9, 12, 15, 16, 20])], 2), ('result_STICSA', [('STICSA_Somatic', [0, 1, 5, 6, 7, 10, 11, 13, 14, 17, 19, 20]), ('STICSA_Cognitive', [2, 3, 4, 8, 9, 12, 15, 16, 18])], 1)]
    result = {}
    for keyword, dimensions, factor in groups:
        values = pd.to_numeric(online_read_questionnaire_responses(sub, keyword)[:, 2], errors='coerce')
        result.update({name: float(values[idx].sum() * factor) for name, idx in dimensions})
    return result

def online_load_online_questionnaire_scores():
    subjects = online_load_online_v6_subjects(dir_files=online_DIR_FILES)['subject_list']
    rows = []
    for sub in subjects:
        path = next((online_DIR_FILES + f for f in online_list_files() if sub in f.replace('\u200b', '') and 'result_Rating_session' in f))
        rs = pd.read_csv(path, sep='\t', header=None, engine='python').dropna(axis=1, how='all').to_numpy()
        qn = online_read_questionnaire_responses(sub, 'result_5dc')
        scores = online_score_five_dimensional_curiosity_scale(pd.to_numeric(qn[:, 2], errors='coerce'))
        scores.update(online_score_dass_and_sticsa_scales(sub))
        scores['Self_Rating'] = utility.calculate_mean_of_finite_values(pd.to_numeric(rs[:, 4], errors='coerce'))
        trial_path = next((online_DIR_FILES + f for f in online_list_files() if sub in f.replace('\u200b', '') and 'result_trial_present' in f))
        tp = pd.read_csv(trial_path, sep='\t', header=None, engine='python').dropna(axis=1, how='all').to_numpy()
        responses = pd.to_numeric(tp[:, 12], errors='coerce')
        valid_choices = responses[np.isin(responses, [1, 2])]
        scores['Story_Count'] = int(np.count_nonzero(valid_choices == 1))
        scores['P_Story'] = scores['Story_Count'] / len(valid_choices)
        rows.append({'Subject': sub, **scores, 'N_Valid_Choices': len(valid_choices)})
    return pd.DataFrame(rows)

def online_find_evidence_trial_file(sub, keyword, exclude=None):
    for f in online_list_files():
        clean = f.replace('\u200b', '')
        if sub in clean and keyword in f and (exclude is None or exclude not in f):
            return online_DIR_FILES + f
    raise FileNotFoundError(keyword)

def online_read_evidence_trial_table(path):
    return pd.read_csv(path, sep='\t', header=None, engine='python').dropna(axis=1, how='all').to_numpy()

def online_load_trials(subjects):
    rows = []
    for sub in subjects:
        tp = online_read_evidence_trial_table(online_find_evidence_trial_file(sub, 'result_trial_present'))
        rs = online_read_evidence_trial_table(online_find_evidence_trial_file(sub, 'result_Rating_session'))
        rr = online_read_evidence_trial_table(online_find_evidence_trial_file(sub, 'result_rating', exclude='Rating_session')).astype(str)
        phrase = pd.to_numeric(rs[:, 1], errors='coerce')
        self_value = pd.to_numeric(rs[:, 4], errors='coerce')
        response = pd.to_numeric(tp[:, 12], errors='coerce')
        keep = np.isin(response, [1, 2])
        trial_id = pd.to_numeric(tp[keep, 0], errors='coerce')
        clickbait = pd.to_numeric(tp[keep, 7], errors='coerce')
        social = pd.to_numeric(tp[keep, 9], errors='coerce')
        risk = pd.to_numeric(tp[keep, 10], errors='coerce')
        choice = (response[keep] == 1).astype(float)
        regret_rows = rr[rr[:, 2] == 'regret']
        regret_id = pd.to_numeric(regret_rows[:, 0], errors='coerce')
        regret_value = pd.to_numeric(regret_rows[:, 6], errors='coerce')
        interest_rows = rr[rr[:, 2] == 'interest']
        interest_value = pd.to_numeric(interest_rows[:, 6], errors='coerce')
        post_interest = float(np.nanmean(interest_value)) if interest_value.size else np.nan
        for i in range(len(trial_id)):
            match = np.where(phrase == clickbait[i])[0]
            self_i = float(np.nanmean(self_value[match])) if match.size else np.nan
            hit = np.where(regret_id == trial_id[i])[0]
            regret_i = float(regret_value[hit[0]]) if hit.size else np.nan
            rows.append(dict(Subject=sub, Self=self_i, Social=social[i], Risk=risk[i], Choice=choice[i], Regret=regret_i, Post_Interest=post_interest))
    return pd.DataFrame(rows)

def online_fit_participant_clustered_gee(y, X, groups, family):
    return sm.GEE(np.asarray(y, float), np.asarray(X, float), groups=np.asarray(groups), family=family, cov_struct=Exchangeable()).fit()

def online_build_regression_design_matrix(columns):
    return np.column_stack([np.ones(len(columns[0]))] + list(columns))

def online_estimate_bootstrap_mediation_effects(df, predictor, n_boot=online_N_BOOT, family_size=1):
    part = df.dropna(subset=[predictor, 'Self_Rating', 'P_Story']).reset_index(drop=True)
    x, m, y = (utility.standardize_mediation_predictor(part[predictor]), utility.standardize_mediation_predictor(part.Self_Rating), part.P_Story.to_numpy(float))
    rng = np.random.default_rng(online_SEED)

    def estimate_mediation_path_coefficients(idx):
        xi, mi, yi = (x[idx], m[idx], y[idx])
        a = np.linalg.lstsq(online_build_regression_design_matrix([xi]), mi, rcond=None)[0][1]
        coefficients = np.linalg.lstsq(online_build_regression_design_matrix([xi, mi]), yi, rcond=None)[0]
        total = np.linalg.lstsq(online_build_regression_design_matrix([xi]), yi, rcond=None)[0][1]
        return (a, coefficients[2], coefficients[1], total)
    a, b, direct, total = estimate_mediation_path_coefficients(np.arange(len(x)))
    boot = np.array([estimate_mediation_path_coefficients(rng.integers(0, len(x), len(x))) for _ in range(n_boot)])
    indirect = boot[:, 0] * boot[:, 1]
    ci_indirect = np.percentile(indirect, [2.5, 97.5])
    ci_direct = np.percentile(boot[:, 2], [2.5, 97.5])
    tail = 100 * 0.05 / (2 * family_size)
    ci_indirect_family = np.percentile(indirect, [tail, 100 - tail])
    significant_family = bool(ci_indirect_family[0] > 0 or ci_indirect_family[1] < 0)
    if family_size > 1:
        print(f'Bonferroni {predictor}: indirect={a * b:+.6f}, adjusted CI [{ci_indirect_family[0]:+.6f}, {ci_indirect_family[1]:+.6f}], significant={significant_family}; family={family_size}, bootstrap={n_boot}')
    print(f'Mediation {predictor} -> self-rating -> P(Story), N={len(x)}')
    print(f'A={a:+.3f} b={b:+.3f} indirect={a * b:+.3f} [{ci_indirect[0]:+.3f},{ci_indirect[1]:+.3f}]')
    print(f'Direct={direct:+.3f} [{ci_direct[0]:+.3f},{ci_direct[1]:+.3f}] total={total:+.3f} proportion mediated={a * b / total:.2f}')
    return dict(a=a, b=b, indirect=a * b, ci_indirect=ci_indirect, direct=direct, ci_direct=ci_direct, total=total, n=len(x), ci_indirect_family=ci_indirect_family, significant_family=significant_family)


def online_load_story_reaction_time_summaries():
    with open(os.devnull, 'w') as sink, redirect_stdout(sink):
        scores = online_load_online_questionnaire_scores().set_index('Subject')
    scores[['DASS_Anxiety', 'DASS_Stress']] -= 14
    trial_presentation_files = list(online_INPUT_DIR.glob('*result_trial_present*'))
    trials, audit = ([], [])
    for sub in scores.index:
        matches = [p for p in trial_presentation_files if sub in p.name.replace('\u200b', '')]
        raw = pd.read_csv(matches[0], sep='\t', header=None)
        duplicate_rows = int(raw.duplicated().sum())
        raw = raw.drop_duplicates()
        n = lambda col: pd.to_numeric(raw[col], errors='coerce')
        frame = pd.DataFrame({'Subject': sub, 'trial': n(0), 'session': n(1), 'onset_ms': n(3), 'threat_level': n(10), 'choice': n(12), 'response_ms': n(13)})
        frame['rt_s'] = (frame.response_ms - frame.onset_ms) / 1000
        valid = frame.choice.eq(1) & np.isfinite(frame.rt_s) & frame.rt_s.gt(0) & frame.threat_level.isin([1, 2, 3, 4])
        frame['threat'] = np.where(frame.threat_level.le(2), 'Low threat', 'High threat')
        audit.append(dict(Subject=sub, file=str(matches[0]), rows=len(frame) + duplicate_rows, exact_duplicates_removed=duplicate_rows, valid_rt=int(valid.sum()), excluded=int((~valid).sum())))
        trials.append(frame.loc[valid].copy())
    return (scores, pd.concat(trials, ignore_index=True), pd.DataFrame(audit))


conn_BRAINMASK_DIR = PROJECT_DIR / 'brainmask'
conn_REMOVE_WHITE_MATTER = False
conn_WHITE_MATTER_PATH = conn_BRAINMASK_DIR / 'mask_white_matter.nii'
conn_TRIAL_SAMPLE = 'story'
conn_ALLOW_RANK_DEFICIENT = True
conn_BRAIN_FIGURE_SCALE = 1.05 * 1.1 * 1.1 * 0.95 ** 2
conn_FONT_SIZE_PT = 8.0
conn_N_PERM = 2000
conn_CORRECTION_METHOD = 'bonferroni'
conn_ATLAS_PATH = conn_BRAINMASK_DIR / 'aal/AAL3/ROI_MNI_V7.nii'
conn_LABEL_PATH = conn_BRAINMASK_DIR / 'aal/AAL3/ROI_MNI_V7.xml'
conn_SIGNIFICANCE_ALPHA = 0.05
conn_ZSCORE_PAIR = True
conn_GROUPS = {'vACC': ('ACC_pre', 'ACC_sub'), 'pmOFC': ('OFCmed', 'OFCpost')}
conn_ROI_ORDER = ('vACC', 'ACC_sup', 'Amygdala', 'Caudate', 'Frontal_Med_Orb', 'Frontal_Sup_Medial', 'Hippocampus', 'Insula', 'OFCant', 'pmOFC', 'Olfactory', 'Putamen', 'Rectus', 'SN_pc', 'VTA', 'Vent_Str')
conn_ROI_LABELS = dict(zip(conn_ROI_ORDER, ('vACC', 'dACC', 'AMY', 'CAU', 'ORBsupmed', 'dmPFC', 'HPC', 'INS', 'aOFC', 'pmOFC', 'OLF', 'PUT', 'vmPFC', 'SNc', 'VTA', 'NAc')))
conn_ROI_NETWORK_LABELS = dict(conn_ROI_LABELS, vACC='vACC', pmOFC='pmOFC')
conn_DISPLAY_ROI_ORDER = ('vACC', 'ACC_sup', 'Amygdala', 'Caudate', 'Frontal_Sup_Medial', 'Hippocampus', 'Insula', 'pmOFC', 'Olfactory', 'Putamen', 'Rectus', 'SN_pc', 'VTA', 'Vent_Str')
conn_ROI_TO_DISPLAY = dict.fromkeys(('ACC_pre', 'ACC_sub'), 'vACC') | dict.fromkeys(('OFCmed', 'OFCpost'), 'pmOFC')

def conn_display_roi(name):
    return conn_ROI_TO_DISPLAY.get(name, name)
conn_SCHEMA_POSITIONS = {'Frontal_Sup_Medial': (54, 50), 'ACC_sup': (30, 33), 'vACC': (43, 11), 'Rectus': (72, -12), 'pmOFC': (22, -35), 'Olfactory': (37, -5), 'Vent_Str': (7, -12), 'Caudate': (3, 36), 'Putamen': (-18, 4), 'Insula': (-29, 21), 'Amygdala': (-8, -38), 'Hippocampus': (-55, -8), 'SN_pc': (-33, -29), 'VTA': (-47, -20)}
conn_SCHEMA_POSITIONS.update(ACC_pre=(47, 17), ACC_sub=(28, 4), Frontal_Med_Orb=(57, 1), OFCant=(55, -33), OFCmed=(35, -28), OFCpost=(17, -33))

def conn_load_roi_table():
    atlas = nib.squeeze_image(nib.load(conn_ATLAS_PATH))
    labels = {entry.findtext('name'): int(entry.findtext('index')) for entry in ET.parse(conn_LABEL_PATH).findall('.//label')}
    a = np.asarray(atlas.dataobj).astype(np.int32)
    names = conn_ROI_ORDER
    voxels = []
    for name in names:
        sources = conn_GROUPS.get(name, ('N_Acc' if name == 'Vent_Str' else name,))
        ids = [labels[source + side] for source in sources for side in ('_L', '_R')]
        voxels.append(np.flatnonzero(np.isin(a, ids).ravel()))
    if conn_REMOVE_WHITE_MATTER:
        wm_image = nib.squeeze_image(nib.load(conn_WHITE_MATTER_PATH))
        if wm_image.shape != atlas.shape or not np.allclose(wm_image.affine, atlas.affine):
            wm_image = resample_from_to(wm_image, (atlas.shape, atlas.affine), order=0)
        white_matter = (wm_image.get_fdata() > 0).ravel()
        voxels = [v[~white_matter[v]] for v in voxels]
    union = np.concatenate(voxels)
    return dict(names=np.asarray(names), roi_voxels=voxels, atlas_shape=atlas.shape, affine=atlas.affine, n_roi=len(names), n_voxels=len(union), source_rois=tuple((source for name in names for source in conn_GROUPS.get(name, (name,)))))
conn_ROI_NAMES = tuple(map(str, conn_load_roi_table()['names']))
conn_N_ROI_PAIRS = len(conn_ROI_NAMES) * (len(conn_ROI_NAMES) - 1) // 2
conn_FIXED_BONFERRONI_TESTS = conn_N_ROI_PAIRS
conn_table = conn_load_roi_table()
conn_ROIS = dict(zip(conn_table['names'], conn_table['roi_voxels']))
conn_EXCLUDED_SUBJECTS = set()
conn_SUBJECTS = [f'par{i:02}' for i in range(1, 31) if f'par{i:02}' not in conn_EXCLUDED_SUBJECTS]
conn_EFFECTS = ['Curiosity RSA coefficient (1–6)', 'Threat RSA coefficient (1–4)', 'C × T RSA coefficient']
conn_ROI_DISPLAY_LABELS = dict(conn_ROI_LABELS)
conn_ROI_NETWORK_LABELS = dict(conn_ROI_NETWORK_LABELS)
conn_EDGES = list(zip(*np.triu_indices(len(conn_ROIS), 1)))
conn_RSA_EFFECT_SEEDS = (frozenset({'Rectus'}), frozenset({'Putamen', 'Frontal_Sup_Medial', 'ACC_sup', 'Insula'}), frozenset({'pmOFC'}))
conn_EFFECT_SEEDS_BY_SAMPLE = {sample: conn_RSA_EFFECT_SEEDS for sample in ('story', 'skip', 'all')}
conn_PERIOD_COMPARISON_COLORS = (('#5E3C99', '#5E3C99'), ('#B2182B', '#B2182B'), ('#454545', '#454545'))
conn_PERIOD_EDGE_LINEWIDTH = 1.05
conn__COMPARISON_LAYOUT_EDGES = None
conn_PERIMETER_NODE_POSITIONS = dict(conn_SCHEMA_POSITIONS)
conn_NETWORK_HIDDEN_NODES = {'vACC', 'Hippocampus', 'Frontal_Med_Orb', 'OFCant'}
conn_NETWORK_REQUIRED_NODES = frozenset({'Caudate', 'VTA', 'SN_pc'})
conn_TRANSFORMER_LABEL_ANCHORS = {'Frontal_Sup_Medial': (66, 64), 'ACC_sup': (48, 78), 'Caudate': (9, 85), 'Insula': (-90, 57), 'Putamen': (-116, 30), 'Hippocampus': (-90, -46), 'Olfactory': (84, -5), 'Rectus': (86, -23), 'pmOFC': (30, -56), 'Vent_Str': (-6, -57), 'Amygdala': (-28, -67), 'SN_pc': (-79, -65), 'VTA': (-100, -40)}
conn_COMPARISON_LABEL_ANCHORS = dict(conn_TRANSFORMER_LABEL_ANCHORS, Insula=(-35, 85), Putamen=(-85, 65), VTA=(-85, -55), SN_pc=(-55, -65), Amygdala=(-25, -70), Vent_Str=(5, -70), pmOFC=(35, -60))

def conn_correction_label():
    if conn_CORRECTION_METHOD == 'uncorrected':
        return 'Uncorrected Pearson p'
    if conn_CORRECTION_METHOD == 'bonferroni':
        return f'Pearson p + Bonferroni ({conn_FIXED_BONFERRONI_TESTS} tests: {conn_N_ROI_PAIRS} ROI pairs; periods/effects/choices corrected separately)'
    if conn_CORRECTION_METHOD == 'max_r':
        counts = [int(conn_select_seed_connected_edges(e).sum()) for e in range(len(conn_EFFECTS))]
        return f'Permutation maximum |r| ({sum(counts)} seed-edge tests)'
    raise ValueError('Invalid CORRECTION_METHOD')

def conn_load_participant_trial_conditions(sub):
    labels = utility.read_participant_behavior_table(sub, 'output_formal_clickbaitLabel_', PROJECT_DIR / 'beh_fmri')[:, 1]
    ratings = utility.read_participant_behavior_table(sub, 'output_self_rating_', PROJECT_DIR / 'beh_fmri')
    lookup = dict(zip(ratings[:, 1].astype(str), ratings[:, 2].astype(float)))
    curiosity = np.array([lookup.get(str(x), np.nan) for x in labels], dtype=float)
    choices = utility.read_participant_behavior_table(sub, 'output_formal_clickbaitRes_', PROJECT_DIR / 'beh_fmri')[:, 1].astype(float)
    return pd.DataFrame(dict(subject=sub, trial=np.arange(1, len(labels) + 1), curiosity=curiosity, threat=utility.read_participant_behavior_table(sub, 'output_formal_FLevel_', PROJECT_DIR / 'beh_fmri')[:, 1].astype(float), social=utility.read_participant_behavior_table(sub, 'output_formal_CLevel_', PROJECT_DIR / 'beh_fmri')[:, 1].astype(float), story=np.where(np.isin(choices, [1, 2]), (choices == 1).astype(float), np.nan)))


def conn_network_correlations(x):
    centered = x - x.mean(axis=0, keepdims=True)
    norms = np.sqrt((centered ** 2).sum(axis=0))
    result = np.full((len(conn_EFFECTS), len(conn_EDGES)), np.nan)
    for effect in range(len(conn_EFFECTS)):
        for k in range(len(conn_EDGES)):
            i, j = conn_EDGES[k]
            denominator = norms[effect, i] * norms[effect, j]
            if denominator > 0:
                result[effect, k] = np.clip(np.dot(centered[:, effect, i], centered[:, effect, j]) / denominator, -1, 1)
    return result

def conn_select_effect_seed_regions(effect_index, seed=None, sample=None):
    sample = conn_TRIAL_SAMPLE if sample is None else sample
    seeds = conn_EFFECT_SEEDS_BY_SAMPLE[sample][effect_index] if seed is None else {seed} if isinstance(seed, str) else set(seed)
    return frozenset(seeds)

def conn_select_seed_connected_edges(effect_index, seed=None, sample=None):
    names = list(conn_ROIS)
    seeds = conn_select_effect_seed_regions(effect_index, seed, sample=sample)
    return np.array([names[i] in seeds or names[j] in seeds for i, j in conn_EDGES])

def conn_format_significance_label(p):
    if not np.isfinite(p) or p >= conn_SIGNIFICANCE_ALPHA:
        return ''
    return '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''

def conn_draw_figure_renderer(fig):
    canvas = fig.canvas
    if callable(getattr(canvas, 'get_renderer', None)):
        canvas.draw()
        return canvas.get_renderer()
    fallback = FigureCanvasAgg(fig)
    fallback.draw()
    renderer = fallback.get_renderer()
    fig.set_canvas(canvas)
    return renderer

def conn_route_connections(edges, positions):
    edges = list(dict.fromkeys((tuple(e) for e in edges)))
    degree = {n: sum((n in edge for edge in edges)) for n in positions}
    order = sorted(edges, key=lambda e: (-(degree[e[0]] + degree[e[1]]), e))
    routes = {}
    chosen = []
    for source, target in order:
        a = np.asarray(positions[source], float)
        b = np.asarray(positions[target], float)
        chord = b - a
        length = np.linalg.norm(chord)
        normal = np.array([-chord[1], chord[0]]) / length
        other = np.array([p for n, p in positions.items() if n not in (source, target)], float)
        t = np.linspace(0, 1, 49)[:, None]
        limit = min(18.0, 0.34 * length)
        bends = [0.0, 0.3 * limit, -0.3 * limit, 0.65 * limit, -0.65 * limit, limit, -limit]
        if (target, source) in routes:
            bends = [v for v in bends if v != 0]
        candidates = []
        for bend in bends:
            control = (a + b) / 2 + bend * normal
            path = (1 - t) ** 2 * a + 2 * (1 - t) * t * control + t * t * b
            score = 0.12 * (bend / max(limit, 1)) ** 2
            if len(other):
                clearance = utility.calculate_point_to_polyline_distances(other, path)
                score += 25 * np.maximum(0, (5.5 - clearance) / 5.5).sum() ** 2
            for _, _, previous in chosen:
                core = path[4:-4]
                distance = utility.calculate_point_to_polyline_distances(core, previous)
                shared = np.maximum(0, 1 - distance / 2.8)
                score += 7 * shared.mean() ** 2 + 0.15 * shared.mean()
            candidates.append((score, abs(bend), bend, path))
        _, _, bend, path = min(candidates, key=lambda row: (row[0], row[1], row[2]))
        routes[source, target] = path
        chosen.append((source, target, path))
    return routes

def conn_comparison_display_mask(rows, effect_index, sample):
    selected = conn_select_seed_connected_edges(effect_index, sample=sample) & rows.significant_display.to_numpy(bool) & np.isfinite(rows.r.to_numpy()) & (rows.r.to_numpy() > 0)
    names = list(conn_ROIS)
    selected &= np.array([conn_display_roi(names[i]) not in conn_NETWORK_HIDDEN_NODES and conn_display_roi(names[j]) not in conn_NETWORK_HIDDEN_NODES and (conn_display_roi(names[i]) != conn_display_roi(names[j])) for i, j in conn_EDGES])
    return selected

def conn_perimeter_brain_geometry():
    image = nib.load(conn_ATLAS_PATH)
    silhouette = gaussian_filter((image.get_fdata() > 0).any(axis=0).astype(float), 1.0)
    yy, zz = np.meshgrid(np.arange(silhouette.shape[0]), np.arange(silhouette.shape[1]), indexing='ij')
    xyz = nib.affines.apply_affine(image.affine, np.column_stack([np.zeros(yy.size), yy.ravel(), zz.ravel()]))
    ygrid = xyz[:, 1].reshape(yy.shape)
    zgrid = xyz[:, 2].reshape(zz.shape)
    return (ygrid, zgrid, silhouette)

def conn_update_period_edge_spacing(fig):
    conn_draw_figure_renderer(fig)
    for ax in fig.axes:
        axis_orientation = np.sign(np.linalg.det(ax.transData.get_affine().get_matrix()[:2, :2]))
        for line in ax.lines:
            centerline = getattr(line, '_period_centerline', None)
            if centerline is None:
                continue
            side = line._period_offset_side
            if not side:
                line.set_data(centerline[:, 0], centerline[:, 1])
                continue
            screen = ax.transData.transform(centerline)
            tangent = np.gradient(screen, axis=0)
            normal = np.column_stack([-tangent[:, 1], tangent[:, 0]])
            lengths = np.linalg.norm(normal, axis=1, keepdims=True)
            normal /= lengths
            offset = side * axis_orientation * line.get_linewidth() * fig.dpi / 72
            points = ax.transData.inverted().transform(screen + offset * normal)
            line.set_data(points[:, 0], points[:, 1])

def conn_draw_perimeter_panel(fig, ax, selections, effect_index, geometry):
    ygrid, zgrid, silhouette = geometry
    ax.contourf(ygrid, zgrid, silhouette, levels=[0.5, 1.5], colors=['#F8F8F8'], zorder=0)
    outline = ax.contour(ygrid, zgrid, silhouette, levels=[0.5], colors=['#C9C9C9'], linewidths=0.65, zorder=1)
    ax._schema_outside_paths = outline.get_paths()
    ax.set(xlim=(115, -145), ylim=(-100, 110), aspect='equal')
    ax.axis('off')
    names = list(conn_ROIS)
    active = {conn_display_roi(names[node]) for k, edge in enumerate(conn_EDGES) if any((mask[k] for mask in selections.values())) for node in edge}
    visible_nodes = [name for name in conn_DISPLAY_ROI_ORDER if name in active | conn_NETWORK_REQUIRED_NODES]
    ax._schema_positions = {name: np.array(conn_PERIMETER_NODE_POSITIONS[name], float) for name in conn_DISPLAY_ROI_ORDER if name not in conn_NETWORK_HIDDEN_NODES}
    positions = ax._schema_positions
    fig._schema_nodes = visible_nodes
    fig._schema_positions = positions
    fig.canvas.draw()
    edges = list(conn_common_schema_edges()[['source_display', 'target_display']].itertuples(index=False, name=None))
    routes = conn_route_connections(edges, conn_PERIMETER_NODE_POSITIONS)
    common = selections['decision'] & selections['shock']
    for period, color, side in zip(('decision', 'shock'), conn_PERIOD_COMPARISON_COLORS[effect_index], (1, -1)):
        for k, (i, j) in enumerate(conn_EDGES):
            if not selections[period][k]:
                continue
            pair = (conn_display_roi(names[i]), conn_display_roi(names[j]))
            points = routes[pair].copy() if pair in routes else routes[pair[::-1]][::-1].copy()
            line, = ax.plot(points[:, 0], points[:, 1], color=color, lw=conn_PERIOD_EDGE_LINEWIDTH, alpha=1.0, linestyle='-' if period == 'decision' else (0, (3.0, 1.8)), solid_capstyle='round', zorder=2, gid=f'edge-{period}-{names[i]}-{names[j]}')
            line._period_centerline = points.copy()
            line._period_offset_side = side if common[k] else 0
    conn_update_period_edge_spacing(fig)
    active = {conn_display_roi(names[node]) for k, edge in enumerate(conn_EDGES) if any((mask[k] for mask in selections.values())) for node in edge}
    for name, point in positions.items():
        active_color = '#777777' if name in active else '#C9C9C9'
        ax.scatter(*point, s=18, facecolor=active_color if name in active else 'white', edgecolor=active_color, lw=0.65, zorder=4, gid='node-' + name)
        annotation = ax.annotate(conn_ROI_NETWORK_LABELS[name], xy=point, xytext=(0, 16), textcoords='offset points', ha='center', va='center', fontsize=conn_FONT_SIZE_PT, fontfamily='Arial', color='#555555', zorder=5, bbox=dict(boxstyle='square,pad=.04', facecolor='white', edgecolor='none'), arrowprops=dict(arrowstyle='-', color='#B0B0B0', lw=0.45, shrinkA=1, shrinkB=3), annotation_clip=False)
        annotation.set_gid('label-' + name)
    for annotation in ax.texts:
        name = annotation.get_gid().removeprefix('label-')
        annotation.set_position((ax.transData.transform(conn_COMPARISON_LABEL_ANCHORS[name]) - ax.transData.transform(positions[name])) * 72 / fig.dpi)
    for a in ax.texts:
        if a.get_gid().removeprefix('label-') not in active | conn_NETWORK_REQUIRED_NODES:
            a.set_visible(False)
    for collection in ax.collections:
        if (collection.get_gid() or '').startswith('node-'):
            collection.set_visible(collection.get_gid().removeprefix('node-') in active | conn_NETWORK_REQUIRED_NODES)
    return ax

def conn_common_schema_edges():
    return conn__COMPARISON_LAYOUT_EDGES

def calculate_point_to_polyline_distances(points, polyline):
    a = polyline[:-1]
    d = np.diff(polyline, axis=0)
    t = np.clip(np.einsum('pki,ki->pk', points[:, None] - a, d) / np.maximum(np.sum(d * d, axis=1), 1e-12), 0, 1)
    closest = a[None] + t[..., None] * d[None]
    return np.sqrt(((points[:, None] - closest) ** 2).sum(-1)).min(axis=1)

def read_participant_behavior_table(cur_sub, key, behavior_directory, behavior_files=None):
    if behavior_files is None:
        paths = list(Path(behavior_directory).glob(f'{key}{cur_sub}.*'))
        return pd.read_csv(paths[0], sep='\t', header=None).to_numpy()
    path = [Path(behavior_directory) / name for name in behavior_files if cur_sub in name and key + cur_sub in name][0]
    return pd.read_csv(path, delimiter='\t', header=None, engine='python').to_numpy()
