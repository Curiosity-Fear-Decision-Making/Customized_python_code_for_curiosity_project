# Foldername: custom_python
# Filename: fig4_analysis_rsa_effect_localizer_threat_firstlevel.py
import nibabel as nib
import numpy as np
import os
from joblib import Parallel, delayed
from pathlib import Path
from scipy.io import loadmat
import pandas as pd
import torch as tor
from tqdm import tqdm
from utility_custom_python import PROJECT_DIR as PACKAGE_DIR

SEARCHLIGHT_DIR = PACKAGE_DIR / 'file_generated' / 'searchlight_r_rsa_s2'
if not SEARCHLIGHT_DIR.is_dir():
    raise FileNotFoundError(f'First-level GLM requires searchlight arrays in {SEARCHLIGHT_DIR}.')


def matlab_triu_indices(n, k=1):
    rows, cols = np.triu_indices(n, k=k)
    order = np.lexsort((rows, cols))
    return (rows[order], cols[order])


def _rsa_zscore_t(x):
    x = np.asarray(x, dtype=np.float32)
    s = x.std()
    if s == 0:
        return tor.zeros(len(x), dtype=tor.float32)
    return tor.tensor((x - x.mean()) / s, dtype=tor.float32)


def glm_multi_regressor_all(Y, regressors, reg_names, n_voxels, cur_sub):
    reg_z_list = []
    for vec, name in zip(regressors, reg_names):
        z = _rsa_zscore_t(vec)
        if np.isnan(z.numpy()).any() or np.isinf(z.numpy()).any():
            return {n: np.zeros(n_voxels) for n in reg_names}
        reg_z_list.append(z)
    X = tor.stack(reg_z_list, dim=1)
    X = tor.cat([tor.ones((X.shape[0], 1)), X], dim=1)
    beta = tor.linalg.solve(X.T @ X, X.T @ Y).numpy()
    out = {}
    for i, name in enumerate(reg_names, start=1):
        b = beta[i, :]
        if np.isnan(b).any() or np.isinf(b).any():
            out[name] = np.zeros(n_voxels)
        else:
            out[name] = b
    return out


def save_beta_nii(target_beta, save_path):
    brain_array = np.zeros((91, 109, 91))
    brain_array[VOXEL_ARRAY[:, 0], VOXEL_ARRAY[:, 1], VOXEL_ARRAY[:, 2]] = target_beta
    brain_struct = nib.Nifti1Image(brain_array, affine=TEMPLATE_AFFINE, header=TEMPLATE_HEADER)
    nib.save(brain_struct, save_path)


def _beh(cur_sub, key):
    return pd.read_csv([ROOT_PATH_BEH + f for f in beh_dir_list if cur_sub in f and key + cur_sub in f][0], delimiter='\t', header=None, engine='python').to_numpy()


def process_subject(cur_sub):
    npy_path = SEARCHLIGHT_DIR / f'{cur_sub}_run2.npy'
    if not npy_path.is_file():
        return "missing input"
    corr = np.load(npy_path).astype(np.float32) / SCALE
    n_pairs = corr.shape[0]
    n_voxels = corr.shape[1]
    matrix_size = int((1 + np.sqrt(1 + 8 * n_pairs)) / 2)
    if matrix_size != CUR_MATRIX_SIZE:
        return "unexpected shape"
    ix, iy = matlab_triu_indices(matrix_size, k=1)
    bin_lh = beh_store[cur_sub]['bin_lowhigh']
    same_bin = (bin_lh[ix] == bin_lh[iy]).astype(np.float32)
    n_same = int(same_bin.sum())
    n_diff = int((1 - same_bin).sum())
    reg_name = REG_NAMES[0]
    if n_same == 0 or n_diff == 0:
        save_beta_nii(np.zeros(n_voxels, dtype=np.float32), SAVE_DIR + f'{cur_sub}_{CUR_PERIOD}_{reg_name}.nii.gz')
        del corr
        return "zero map"
    Y_sub = tor.from_numpy(corr)
    beta_dict = glm_multi_regressor_all(Y_sub, [same_bin], [reg_name], n_voxels, cur_sub)
    save_beta_nii(beta_dict[reg_name].astype(np.float32), SAVE_DIR + f'{cur_sub}_{CUR_PERIOD}_{reg_name}.nii.gz')
    del Y_sub, corr
    return "fitted"


ROOT_PATH_BEH = f'{PACKAGE_DIR}/beh_fmri/'
N_JOBS = 5
SCALE = 10000.0
SUBJECT_LIST = [f'par{i:02d}' for i in range(1, 31)]
voxel_coordinates = loadmat(f'{PACKAGE_DIR}/toolbox_3rdParty/matlab_voxel_coor.mat')
VOXEL_ARRAY = np.column_stack([voxel_coordinates[key].ravel() for key in ('d2', 'd1', 'd3')]) - 1
template_image = nib.load(f'{PACKAGE_DIR}/brainmask/mask_wholebrain.nii')
TEMPLATE_AFFINE = template_image.affine
TEMPLATE_HEADER = template_image.header
RESULT_TAG = 's2_rsa_risk_trial_lowhigh_subj'
SAVE_DIR = f'{PACKAGE_DIR}/result_{RESULT_TAG}/'
STAT_DIR = f'{PACKAGE_DIR}/stat_{RESULT_TAG}/'
CUR_PERIOD = 'risk'
CUR_MATRIX_SIZE = 80
REG_NAMES = ['risk_lowhigh']
Path(SAVE_DIR).mkdir(parents=True, exist_ok=True)
Path(STAT_DIR).mkdir(parents=True, exist_ok=True)
beh_dir_list = os.listdir(ROOT_PATH_BEH)
beh_store = {}
for cur_sub in tqdm(SUBJECT_LIST, desc='Preparing behavior', unit='subject'):
    rr_arr = _beh(cur_sub, 'output_risk_ratingResponse_')
    rating_subj = rr_arr[:CUR_MATRIX_SIZE, 1].astype(np.float32)
    med = float(np.median(rating_subj))
    bin_lowhigh = (rating_subj > med).astype(np.int8)
    beh_store[cur_sub] = {'bin_lowhigh': bin_lowhigh}
jobs = (delayed(process_subject)(sub) for sub in SUBJECT_LIST)
results = Parallel(n_jobs=N_JOBS, return_as='generator_unordered')(jobs)
results = list(tqdm(results, total=len(SUBJECT_LIST), desc='First-level GLM', unit='subject'))
print(f"S2 threat first-level GLM: fitted {results.count('fitted')}/{len(SUBJECT_LIST)}; "
      f"zero maps {results.count('zero map')}; missing inputs {results.count('missing input')}; "
      f"unexpected shapes {results.count('unexpected shape')}.")
