# Foldername: custom_python
# Filename: fig4_analysis_rsa_effect_clickbait_task_firstlevel.py
import numpy as np
import os
from joblib import Parallel, delayed
from pathlib import Path
from scipy.io import loadmat
import nibabel as nib
import pandas as pd
from nilearn.image import smooth_img
from tqdm import tqdm
from utility_custom_python import PROJECT_DIR as PACKAGE_DIR

SEARCHLIGHT_DIR = PACKAGE_DIR / 'file_generated' / 'searchlight_r_rsa_s3_clickbaitshockstick'
if not SEARCHLIGHT_DIR.is_dir():
    raise FileNotFoundError(f'First-level GLM requires searchlight arrays in {SEARCHLIGHT_DIR}.')


def matlab_triu_indices(n, k=1):
    rows, cols = np.triu_indices(n, k=k)
    order = np.lexsort((rows, cols))
    return (rows[order], cols[order])


def _beh(cur_sub, key):
    return pd.read_csv([ROOT_PATH_BEH + f for f in beh_dir_list if cur_sub in f and key + cur_sub in f][0], delimiter='\t', header=None, engine='python').to_numpy()


def _prep_brain(brain):
    out = np.zeros_like(brain)
    out[GM_BOOL] = brain[GM_BOOL]
    return out.astype(np.float32)


def _zscore_pair(Y):
    m = Y.mean(axis=0, keepdims=True)
    s = Y.std(axis=0, keepdims=True)
    valid = s[0] > 0
    Z = np.zeros_like(Y)
    Z[:, valid] = (Y[:, valid] - m[:, valid]) / s[:, valid]
    return Z.astype(np.float32)


def _glm_betas(Y, cur_val, risk_val, both_story_val, social_same_val):
    cur_c = cur_val - cur_val.mean()
    risk_c = risk_val - risk_val.mean()
    story_c = both_story_val - both_story_val.mean()
    social_same_c = social_same_val - social_same_val.mean()
    X = np.column_stack([np.ones(Y.shape[0]), cur_c, risk_c, cur_c * risk_c, story_c, social_same_c]).astype(np.float64)
    coef = np.linalg.pinv(X) @ Y
    return {'rsa_curiosity_parametric': coef[1, :].astype(np.float32), 'rsa_risk_parametric': coef[2, :].astype(np.float32), 'rsa_interaction_parametric': coef[3, :].astype(np.float32)}


def process_subject(cur_sub):
    npy_path = SEARCHLIGHT_DIR / f'{cur_sub}_run3.npy'
    if not npy_path.is_file():
        return None
    mm = np.load(npy_path, mmap_mode='r')
    n_pairs, n_voxels = mm.shape
    matrix_size = int((1 + np.sqrt(1 + 8 * n_pairs)) / 2)
    ix, iy = matlab_triu_indices(matrix_size, k=1)
    beh = beh_store.get(cur_sub)
    if beh is None:
        return None
    cur_lvl = beh['cur_lvl']
    risk_lvl = beh['risk_lvl']
    social_lvl = beh['social_lvl']
    story_mask = beh['story_mask']
    trial_ok = np.ones(beh['n_trials'], dtype=bool) if INCLUDE_SKIP_TRIALS else story_mask
    cur_same = cur_lvl[ix] == cur_lvl[iy]
    risk_same = risk_lvl[ix] == risk_lvl[iy]
    pair_ok = trial_ok[ix] & trial_ok[iy]
    sel = np.where(cur_same & risk_same & pair_ok)[0]
    if not sel.size:
        return None
    cur_val = cur_lvl[ix][sel].astype(np.float64)
    risk_val = risk_lvl[ix][sel].astype(np.float64)
    both_story_val = (story_mask[ix][sel] & story_mask[iy][sel]).astype(np.float64)
    social_same_val = (social_lvl[ix[sel]] == social_lvl[iy[sel]]).astype(np.float64)
    Y = np.asarray(mm[sel, :], dtype=np.float32) / SCALE
    if ZSCORE_PAIR:
        Y = _zscore_pair(Y)
    betas = _glm_betas(Y, cur_val, risk_val, both_story_val, social_same_val)
    for measure, beta_vals in betas.items():
        raw_brain = np.zeros((91, 109, 91), dtype=np.float32)
        raw_brain[VOXEL_ARRAY[:, 0], VOXEL_ARRAY[:, 1], VOXEL_ARRAY[:, 2]] = beta_vals
        raw_img = nib.Nifti1Image(raw_brain, TEMPLATE_AFFINE)
        if SMOOTH_AT_FIRST_LEVEL:
            beta_brain = smooth_img(raw_img, SMOOTH_FWHM).get_fdata(dtype=np.float32)
        else:
            beta_brain = raw_brain
        masked_beta_brain = _prep_brain(beta_brain)
        nib.save(nib.Nifti1Image(masked_beta_brain, TEMPLATE_AFFINE), SAVE_DIR + f'{cur_sub}_{CUR_PERIOD}_{measure}.nii.gz')
        del raw_brain, raw_img, beta_brain, masked_beta_brain
    del mm, Y, betas
    return cur_sub


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
CUR_PERIOD = 'clickbaitshockstick'
RESULT_TAG = 's3_rsa_clickbaitshock_c_attenuate_effect_self_socialcontrol'
SMOOTH_AT_FIRST_LEVEL = False
SMOOTH_FWHM = 0
ZSCORE_PAIR = True
INCLUDE_SKIP_TRIALS = True
SAVE_DIR = f'{PACKAGE_DIR}/result_{RESULT_TAG}/'
STAT_DIR = f'{PACKAGE_DIR}/stat_{RESULT_TAG}/'
GM_BOOL = np.zeros(TEMPLATE_IMA.shape, dtype=bool)
GM_BOOL[IDX_MASK] = True
GM_BOOL[IDX_EXCLUDE[0], IDX_EXCLUDE[1], IDX_EXCLUDE[2]] = False
Path(SAVE_DIR).mkdir(parents=True, exist_ok=True)
Path(STAT_DIR).mkdir(parents=True, exist_ok=True)
beh_dir_list = os.listdir(ROOT_PATH_BEH)
beh_store = {}
for cur_sub in tqdm(SUBJECT_LIST, desc='Preparing behavior', unit='subject'):
    CL = _beh(cur_sub, 'output_formal_CLevel_')
    FL = _beh(cur_sub, 'output_formal_FLevel_')
    clab = _beh(cur_sub, 'output_formal_clickbaitLabel_')
    sr = _beh(cur_sub, 'output_self_rating_')
    CR = _beh(cur_sub, 'output_formal_clickbaitRes_')
    n_trials = CL.shape[0]
    name2rat = {str(nm): float(r) for nm, r in zip(sr[:, 1], sr[:, 2])}
    cur_subj = np.array([name2rat.get(str(nm), np.nan) for nm in clab[:n_trials, 1]], dtype=np.float32)
    if int(np.isnan(cur_subj).sum()):
        cur_subj[np.isnan(cur_subj)] = np.nanmean(cur_subj)
    cur_lvl = np.rint(cur_subj).astype(np.float32)
    risk_lvl = FL[:n_trials, 1].astype(np.float32)
    story_mask = CR[:n_trials, 1].astype(int) == 1
    beh_store[cur_sub] = {'n_trials': n_trials, 'cur_lvl': cur_lvl, 'social_lvl': CL[:n_trials, 1].astype(np.float32), 'risk_lvl': risk_lvl, 'story_mask': story_mask}
jobs = (delayed(process_subject)(sub) for sub in SUBJECT_LIST)
results = Parallel(n_jobs=N_JOBS, return_as='generator_unordered')(jobs)
results = list(tqdm(results, total=len(SUBJECT_LIST), desc='First-level GLM', unit='subject'))
ok_subs = [s for s in results if s is not None]
print(f'Subject-level: {len(ok_subs)}/{len(SUBJECT_LIST)} participants with a usable rsaglm map')
