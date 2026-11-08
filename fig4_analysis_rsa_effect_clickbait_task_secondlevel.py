# Foldername: custom_python
# Filename: fig4_analysis_rsa_effect_clickbait_task_secondlevel.py
import numpy as np
from joblib import Parallel, delayed
from pathlib import Path
import nibabel as nib
from nilearn.image import smooth_img
from scipy import ndimage
import subprocess
from tqdm import tqdm
from utility_custom_python import PROJECT_DIR as PACKAGE_DIR



def permutation_stack_name(measure):
    return f'rsa4d_{SESSION}_{RESULT_TAG}_{measure}_up'


def run_permutation(measure, subjects):
    paths = [SAVE_DIR / f'{sub}_{CUR_PERIOD}_{measure}.nii.gz' for sub in subjects]
    reference = nib.load(paths[0])
    volumes = []
    for path in tqdm(paths, desc=f'Load beta maps: {measure}', unit='subject'):
        img = nib.load(path)
        volume = np.nan_to_num(img.get_fdata(dtype=np.float32), nan=0.0)
        if SMOOTH_AT_SECOND_LEVEL:
            clean_image = nib.Nifti1Image(volume, img.affine)
            volume = smooth_img(clean_image, SECOND_LEVEL_SMOOTH_FWHM).get_fdata(dtype=np.float32)
        volumes.append(volume)
    stack = np.stack(volumes, axis=3)
    stack_name = permutation_stack_name(measure)
    stack_path = STAT_DIR / (stack_name + '.nii.gz')
    nib.save(nib.Nifti1Image(stack, reference.affine), stack_path)
    subprocess.run(
        ['randomise', '-i', str(stack_path), '-o', str(STAT_DIR / ('corr_' + stack_name)), '-1', '-c', str(PERM_CTHR), '-n', str(PERM_NPERM), '-m', PERM_MASK],
        stdout=subprocess.DEVNULL,
        check=True,
    )


def export_permutation(measure):
    prefix = STAT_DIR / ('corr_' + permutation_stack_name(measure))
    corrp_path = Path(str(prefix) + '_clustere_corrp_tstat1.nii.gz')
    corrp_img = nib.load(corrp_path)
    t_img = nib.load(str(prefix) + '_tstat1.nii.gz')
    mask_img = nib.load(PERM_MASK)
    mask = mask_img.get_fdata() > 0
    corrp = corrp_img.get_fdata(dtype=np.float32)
    min_p = float(1.0 - corrp[mask].max())
    sig_mask = mask & (corrp >= np.float32(CORRP_THR))
    if MIN_CLUSTER_VOX > 1:
        labels, _ = ndimage.label(sig_mask, ndimage.generate_binary_structure(3, 3))
        sizes = np.bincount(labels.ravel())
        sig_mask &= sizes[labels] >= MIN_CLUSTER_VOX
    sig_t = np.where(sig_mask, t_img.get_fdata(dtype=np.float32), 0).astype(np.float32)
    sig_name = f'sigtmap_{SESSION}_{RESULT_TAG}_{measure}_up.nii'
    sig_path = STAT_DIR / sig_name
    if sig_mask.any():
        nib.save(nib.Nifti1Image(sig_t, t_img.affine), sig_path)
    else:
        sig_path.unlink(missing_ok=True)
    print(f'{measure}: min p_FWE={min_p:.6f}; {int(sig_mask.sum())} significant voxels')


CUR_PERIOD = 'clickbaitshockstick'
SESSION = 's3'
RESULT_TAG = 's3_rsa_clickbaitshock_c_attenuate_effect_self_socialcontrol'
SAVE_DIR = PACKAGE_DIR / f'result_{RESULT_TAG}'
STAT_DIR = PACKAGE_DIR / f'stat_{RESULT_TAG}'
STAT_DIR.mkdir(exist_ok=True)
PERM_MASK = f'{PACKAGE_DIR}/assets/svc_mask.nii'
SUBJECT_LIST = [f'par{i:02d}' for i in range(1, 31)]
MEASURES = ('rsa_curiosity_parametric', 'rsa_risk_parametric', 'rsa_interaction_parametric')
RUN_PERMUTATION = True
SMOOTH_AT_SECOND_LEVEL = False
SECOND_LEVEL_SMOOTH_FWHM = 5.0
PERM_CTHR = 2.3
PERM_NPERM = 1000
PERM_CORES = 12
CORRP_THR = 0.95
MIN_CLUSTER_VOX = 10
if RUN_PERMUTATION:
    print('Starting permutation with FSL randomise.', flush=True)
    subjects = [sub for sub in SUBJECT_LIST if all(((SAVE_DIR / f'{sub}_{CUR_PERIOD}_{m}.nii.gz').exists() for m in MEASURES))]
    Parallel(n_jobs=PERM_CORES, prefer='threads')((delayed(run_permutation)(measure, subjects) for measure in MEASURES))
for measure in tqdm(MEASURES, desc='Export significant maps', unit='contrast'):
    export_permutation(measure)
print(f'Permutation outputs: {STAT_DIR}')
