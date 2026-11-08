# Foldername: custom_python
# Filename: fig4_analysis_rsa_postchoice_selfrating_binary_firstlevel.py
import os
import pandas as pd
import numpy as np
from joblib import Parallel, delayed, parallel_config
from pathlib import Path
from scipy.io import loadmat
import nibabel as nib
from nilearn.image import smooth_img
from threadpoolctl import threadpool_limits
from tqdm import tqdm
from utility_custom_python import PROJECT_DIR as PACKAGE_DIR

SEARCHLIGHT_DIR = PACKAGE_DIR / 'file_generated' / 'searchlight_r_rsa_s3_shock'
if not SEARCHLIGHT_DIR.is_dir():
    raise FileNotFoundError(f'First-level GLM requires searchlight arrays in {SEARCHLIGHT_DIR}.')


def matlab_triu_indices(n, k=1):
    rows, cols = np.triu_indices(n, k=k)
    order = np.lexsort((rows, cols))
    return (rows[order], cols[order])


def _shock__zscore_gm(brain):
    vals = brain[GM_BOOL]
    m = vals.mean()
    s = vals.std()
    z = np.zeros_like(brain)
    if s > 0:
        z[GM_BOOL] = (vals - m) / s
    return z.astype(np.float32)


def _shock__prep_brain(brain):
    if ZSCORE_GM:
        return _shock__zscore_gm(brain)
    out = np.zeros_like(brain)
    out[GM_BOOL] = brain[GM_BOOL]
    return out.astype(np.float32)


def _shock__zscore_pair(Y):
    m = Y.mean(axis=0, keepdims=True)
    s = Y.std(axis=0, keepdims=True)
    valid = s[0] > 0
    Z = np.zeros_like(Y)
    Z[:, valid] = (Y[:, valid] - m[:, valid]) / s[:, valid]
    return Z.astype(np.float32)


def _shock__beh(cur_sub, key):
    return pd.read_csv([ROOT_PATH_BEH + f for f in beh_dir_list if cur_sub in f and key + cur_sub in f][0], delimiter='\t', header=None, engine='python').to_numpy()


def _shock__align_rows(table, trial_ids, label):
    ids = pd.to_numeric(table[:, 0], errors='raise').astype(int)
    lookup = dict(zip(ids, range(len(ids))))
    return table[[lookup[int(i)] for i in trial_ids]]


def _shock__shock_binary(values):
    strings = np.array([str(v).strip().lower() for v in values])
    return strings == 'yes'


def build_pooled_design(behavior, subject_id):
    ratings = np.asarray(behavior['cur_lvl'], dtype=float)
    valid_trials = np.asarray(behavior['valid_trial'], dtype=bool)
    story_trials = np.asarray(behavior['story_mask'], dtype=bool)
    shock_trials = np.asarray(behavior['shock_mask'], dtype=bool)
    threat_ratings = np.asarray(behavior['risk_lvl'], dtype=float)
    social_ratings = np.asarray(behavior['social_lvl'], dtype=float)
    high_trials = ratings >= 4
    first_trial, second_trial = matlab_triu_indices(behavior['n_trials'], k=1)
    valid_pairs = valid_trials[first_trial] & valid_trials[second_trial]
    same_rating_group = high_trials[first_trial] == high_trials[second_trial]
    same_choice = story_trials[first_trial] == story_trials[second_trial]
    pair_rows = np.flatnonzero(valid_pairs & same_rating_group & same_choice)
    if not len(pair_rows):
        return None
    first_trial = first_trial[pair_rows]
    second_trial = second_trial[pair_rows]
    both_story = story_trials[first_trial].astype(float)
    both_skip = 1.0 - both_story
    high = high_trials[first_trial].astype(float)
    predictor_names = ['intercept', 'BothStory', 'HighStory', 'HighSkip']
    design_columns = [np.ones(len(pair_rows))]
    for values in (both_story, both_story * high, both_skip * high):
        design_columns.append(values - values.mean())
    for name, values in [('SameThreat', (threat_ratings[first_trial] == threat_ratings[second_trial]).astype(float)), ('SameSocial', (social_ratings[first_trial] == social_ratings[second_trial]).astype(float)), ('BothShock', (shock_trials[first_trial] & shock_trials[second_trial]).astype(float))]:
        if np.ptp(values) > 0:
            predictor_names.append(name)
            design_columns.append(values - values.mean())
        else:
            pass
    design_matrix = np.column_stack(design_columns)
    design_pseudoinverse = np.linalg.pinv(design_matrix)
    design_projection = design_pseudoinverse @ design_matrix
    regression_weights = {}
    for choice, indicator, target_name in [('skip', both_skip, 'HighSkip'), ('story', both_story, 'HighStory')]:
        low_count = int(np.count_nonzero((indicator == 1) & (high == 0)))
        high_count = int(np.count_nonzero((indicator == 1) & (high == 1)))
        contrast = np.zeros(design_matrix.shape[1])
        contrast[predictor_names.index(target_name)] = 1.0
        if not np.allclose(contrast @ design_projection, contrast, rtol=1e-07, atol=1e-08):
            continue
        regression_weights[choice] = contrast @ design_pseudoinverse
    return {'pair_rows': pair_rows, 'design_matrix': design_matrix, 'predictor_names': predictor_names, 'regression_weights': regression_weights}


def fit_pooled_glm(similarity_matrix, pooled_design):
    voxel_count = similarity_matrix.shape[1]
    weights = pooled_design['regression_weights']
    beta_by_choice = {choice: np.empty(voxel_count, dtype=np.float32) for choice in weights}
    with threadpool_limits(limits=1):
        for voxel_start in range(0, voxel_count, VOXELS_PER_CHUNK):
            voxel_stop = min(voxel_start + VOXELS_PER_CHUNK, voxel_count)
            similarity_values = np.asarray(similarity_matrix[pooled_design['pair_rows'], voxel_start:voxel_stop], dtype=np.float32) / SCALE
            if ZSCORE_SIMILARITY_ACROSS_PAIRS:
                similarity_values = _shock__zscore_pair(similarity_values)
            for choice, regression_weights in weights.items():
                beta_by_choice[choice][voxel_start:voxel_stop] = regression_weights @ similarity_values
    return beta_by_choice


def process_subject(subject_id, behavior):
    pooled_design = build_pooled_design(behavior, subject_id)
    available_choices = () if pooled_design is None else tuple(pooled_design['regression_weights'])
    for choice in CHOICES:
        if choice not in available_choices:
            (SUBJECT_RESULT_DIR / f'{subject_id}_{choice}_binary.nii.gz').unlink(missing_ok=True)
    if not available_choices:
        return None
    similarity_path = SEARCHLIGHT_DIR / f'{subject_id}_run3.npy'
    similarity_matrix = np.load(similarity_path, mmap_mode='r')
    beta_by_choice = fit_pooled_glm(similarity_matrix, pooled_design)
    for choice, beta_values in beta_by_choice.items():
        beta_volume = np.zeros(TEMPLATE_IMA.shape, dtype=np.float32)
        beta_volume[VOXEL_ARRAY[:, 0], VOXEL_ARRAY[:, 1], VOXEL_ARRAY[:, 2]] = beta_values
        beta_image = nib.Nifti1Image(beta_volume, TEMPLATE_AFFINE)
        if SMOOTH_AT_FIRST_LEVEL:
            beta_image = smooth_img(beta_image, FIRST_LEVEL_SMOOTH_FWHM)
        beta_volume = _shock__prep_brain(beta_image.get_fdata(dtype=np.float32))
        beta_image = nib.Nifti1Image(beta_volume, TEMPLATE_AFFINE)
        applied_fwhm = FIRST_LEVEL_SMOOTH_FWHM if SMOOTH_AT_FIRST_LEVEL else 0.0
        beta_image.header['descrip'] = f'binary123456;pooled;threat_social_shock;level1_fwhm={applied_fwhm:g}'.encode('ascii')
        nib.save(beta_image, SUBJECT_RESULT_DIR / f'{subject_id}_{choice}_binary.nii.gz')
    return tuple(beta_by_choice)


ROOT_PATH_BEH = f'{PACKAGE_DIR}/beh_fmri/'
N_JOBS = 5
SCALE = 10000.0
SUBJECT_LIST = [f'par{i:02d}' for i in range(1, 31)]
voxel_coordinates = loadmat(f'{PACKAGE_DIR}/toolbox_3rdParty/matlab_voxel_coor.mat')
VOXEL_ARRAY = np.column_stack([voxel_coordinates[key].ravel() for key in ('d2', 'd1', 'd3')]) - 1
template_image = nib.load(f'{PACKAGE_DIR}/brainmask/mask_wholebrain.nii')
TEMPLATE_IMA = np.squeeze(np.asarray(template_image.get_fdata(dtype=np.float32)))
IDX_MASK = np.where(TEMPLATE_IMA == 1)
TEMPLATE_AFFINE = template_image.affine
white_matter_image = nib.load(f'{PACKAGE_DIR}/brainmask/mask_white_matter.nii')
white_matter = np.squeeze(np.asarray(white_matter_image.get_fdata(dtype=np.float32)))
IDX_EXCLUDE = np.where(white_matter > 0)
ZSCORE_GM = False
GM_BOOL = np.zeros(TEMPLATE_IMA.shape, dtype=bool)
GM_BOOL[IDX_MASK] = True
GM_BOOL[IDX_EXCLUDE[0], IDX_EXCLUDE[1], IDX_EXCLUDE[2]] = False
beh_dir_list = os.listdir(ROOT_PATH_BEH)
RESULT_TAG = 's3_rsa_shock_selfrating_binary'
SUBJECT_RESULT_DIR = PACKAGE_DIR / f'result_{RESULT_TAG}'
GROUP_RESULT_DIR = PACKAGE_DIR / f'stat_{RESULT_TAG}'
CHOICES = ('skip', 'story')
ZSCORE_SIMILARITY_ACROSS_PAIRS = False
VOXELS_PER_CHUNK = 512
SMOOTH_AT_FIRST_LEVEL = False
FIRST_LEVEL_SMOOTH_FWHM = 5.0
SUBJECT_RESULT_DIR.mkdir(exist_ok=True)
GROUP_RESULT_DIR.mkdir(exist_ok=True)
behavior_by_subject = {}
for sub in tqdm(SUBJECT_LIST, desc='Preparing behavior', unit='subject'):
    cl = _shock__beh(sub, 'output_formal_CLevel_')
    ids = pd.to_numeric(cl[:, 0], errors='raise').astype(int)
    n = len(ids)
    fl = _shock__align_rows(_shock__beh(sub, 'output_formal_FLevel_'), ids, sub + ' FLevel')
    labels = _shock__align_rows(_shock__beh(sub, 'output_formal_clickbaitLabel_'), ids, sub + ' clickbait')
    choice = _shock__align_rows(_shock__beh(sub, 'output_formal_clickbaitRes_'), ids, sub + ' choice')
    sr = _shock__beh(sub, 'output_self_rating_')
    item_ratings = {str(nm): float(r) for nm, r in zip(sr[:, 1], sr[:, 2])}
    curiosity = np.array([item_ratings.get(str(item), np.nan) for item in labels[:, 1]])
    missing = ~np.isfinite(curiosity)
    if missing.any():
        curiosity[missing] = np.nanmean(curiosity)
    valid_cur = np.isfinite(curiosity) & (curiosity >= 1) & (curiosity <= 6)
    cur_lvl = np.rint(curiosity)
    objective = fl[:, 1].astype(float)
    responses = choice[:, 1].astype(int)
    story = responses == 1
    threat = objective.copy()
    behavior_by_subject[sub] = dict(n_trials=n, trial_ids=ids, cur_lvl=cur_lvl, social_lvl=cl[:, 1].astype(float), risk_lvl=objective, threat_val=threat, story_mask=story, valid_trial=valid_cur & np.isin(responses, [1, 2]) & np.isfinite(threat))
for subject_id, behavior in behavior_by_subject.items():
    trial_ids = behavior['trial_ids']
    social_rows = _shock__align_rows(_shock__beh(subject_id, 'output_formal_CLevel_'), trial_ids, subject_id + ' social')
    shock_rows = _shock__align_rows(_shock__beh(subject_id, 'output_formal_shock_'), trial_ids, subject_id + ' shock')
    behavior['social_lvl'] = social_rows[:, 1].astype(float)
    behavior['shock_mask'] = _shock__shock_binary(shock_rows[:, 1])
    behavior['valid_trial'] &= np.isfinite(behavior['social_lvl'])
with parallel_config(backend='loky', inner_max_num_threads=1):
    jobs = (delayed(process_subject)(subject_id, behavior_by_subject[subject_id]) for subject_id in SUBJECT_LIST)
    results = Parallel(n_jobs=N_JOBS, return_as='generator_unordered')(jobs)
    completed_subjects = list(tqdm(results, total=len(SUBJECT_LIST), desc='First-level GLM', unit='subject'))
usable_subject_count = sum((subject_id is not None for subject_id in completed_subjects))
print(f'First level: {usable_subject_count}/{len(completed_subjects)} participants fitted')
for choice in CHOICES:
    count = sum((choices is not None and choice in choices for choices in completed_subjects))
    print(f'First level: {choice}: {count}/{len(completed_subjects)} identifiable beta maps')
