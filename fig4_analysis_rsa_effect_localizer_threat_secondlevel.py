# Foldername: custom_python
# Filename: fig4_analysis_rsa_effect_localizer_threat_secondlevel.py
import nibabel as nib
import numpy as np
import os
from nilearn.image import smooth_img
from pathlib import Path
from scipy import stats, ndimage
from joblib import Parallel, delayed
import subprocess
from tqdm import tqdm
from utility_custom_python import PROJECT_DIR as PACKAGE_DIR


RUN_PERMUTATION = True
template_image = nib.load(f'{PACKAGE_DIR}/brainmask/mask_wholebrain.nii')
TEMPLATE_IMA = np.squeeze(np.asarray(template_image.get_fdata(dtype=np.float32)))
IDX_MASK = np.where(TEMPLATE_IMA == 1)
TEMPLATE_AFFINE = template_image.affine
white_matter_image = nib.load(f'{PACKAGE_DIR}/brainmask/mask_white_matter.nii')
white_matter = np.squeeze(np.asarray(white_matter_image.get_fdata(dtype=np.float32)))
IDX_EXCLUDE = np.where(white_matter > 0)
RESULT_TAG = 's2_rsa_risk_trial_lowhigh_subj'
SAVE_DIR = f'{PACKAGE_DIR}/result_{RESULT_TAG}/'
STAT_DIR = f'{PACKAGE_DIR}/stat_{RESULT_TAG}/'
CUR_PERIOD = 'risk'
REG_NAMES = ['risk_lowhigh']
ALTERNATIVE = {'risk_lowhigh': 'greater'}
APPLY_2ND_LEVEL_ZSCORE = False
Path(SAVE_DIR).mkdir(parents=True, exist_ok=True)
Path(STAT_DIR).mkdir(parents=True, exist_ok=True)
SESSION = 's2'
SMOOTH_FWHM = 5
beta4d_stacks = []
for reg_name in REG_NAMES:
    beta_files = sorted([SAVE_DIR + f for f in os.listdir(SAVE_DIR) if f.endswith(f'_{CUR_PERIOD}_{reg_name}.nii.gz')])
    if len(beta_files) < 2:
        continue
    n_sub = len(beta_files)
    plot_table = np.zeros((91, 109, 91, n_sub), dtype=np.float32)
    for i, p in enumerate(tqdm(beta_files, desc=f'Load beta maps: {reg_name}', unit='subject')):
        img = smooth_img(nib.load(p), SMOOTH_FWHM)
        plot_table[:, :, :, i] = np.asarray(img.get_fdata(dtype=np.float32))
    plot_table[np.isnan(plot_table)] = 0
    if APPLY_2ND_LEVEL_ZSCORE:
        mask_bool = np.zeros(TEMPLATE_IMA.shape, dtype=bool)
        mask_bool[IDX_MASK] = True
        mask_bool[IDX_EXCLUDE[0], IDX_EXCLUDE[1], IDX_EXCLUDE[2]] = False
        for i in tqdm(range(n_sub), desc=f'Z-score: {reg_name}', unit='subject'):
            vol = plot_table[:, :, :, i]
            vals = vol[mask_bool]
            mu = float(np.mean(vals))
            sd = float(np.std(vals))
            vol_z = (vol - mu) / sd if sd > 0 else vol - mu
            vol_z[~mask_bool] = 0
            plot_table[:, :, :, i] = vol_z.astype(np.float32)
    z_tag = '_z' if APPLY_2ND_LEVEL_ZSCORE else ''
    stack_name = f'beta4d_{SESSION}_{CUR_PERIOD}_{reg_name}_lowhigh{z_tag}.nii.gz'
    nib.save(nib.Nifti1Image(plot_table.astype(np.float32), TEMPLATE_AFFINE), STAT_DIR + stack_name)
    beta4d_stacks.append(stack_name[:-7])
    alt = ALTERNATIVE.get(reg_name, 'greater')
    t_stat, _ = stats.ttest_1samp(plot_table, axis=3, popmean=0, alternative=alt)
    t_stat = t_stat.astype(np.float32)
    new_brain = np.zeros(TEMPLATE_IMA.shape, dtype=np.float32)
    new_brain[IDX_MASK] = t_stat[IDX_MASK]
    new_brain[IDX_EXCLUDE[0], IDX_EXCLUDE[1], IDX_EXCLUDE[2]] = 0
    new_brain = np.asarray(smooth_img(nib.Nifti1Image(new_brain, TEMPLATE_AFFINE), SMOOTH_FWHM).get_fdata(dtype=np.float32))
    out_name = f'tval_{SESSION}_{CUR_PERIOD}_{reg_name}_lowhigh{z_tag}_tmap.nii'
    nib.save(nib.Nifti1Image(new_brain, TEMPLATE_AFFINE), STAT_DIR + out_name)
    print(f'Second level {alt}: {reg_name}: n={n_sub} -> {out_name} (output smoothed {SMOOTH_FWHM}mm)')
PERM_MASK = f'{PACKAGE_DIR}/assets/svc_mask.nii'
PERM_NPERM = 1000
PERM_OPTS = ['-1', '-C', '2.1', '-n', str(PERM_NPERM)]
PERM_CORES = 12
if RUN_PERMUTATION and beta4d_stacks:
    print('Starting permutation with FSL randomise.', flush=True)
    Parallel(n_jobs=PERM_CORES, prefer='threads')(
        delayed(subprocess.run)(
            ['randomise', '-i', f'{STAT_DIR}{name}.nii.gz', '-o', f'{STAT_DIR}corr_{name}', *PERM_OPTS, '-m', PERM_MASK],
            stdout=subprocess.DEVNULL,
            check=True,
        )
        for name in beta4d_stacks
    )
CORRP_THR = 0.95
MIN_CLUSTER_VOX = 50
for name in tqdm(beta4d_stacks, desc='Export significant maps', unit='stack'):
    corrp_path = STAT_DIR + f'corr_{name}_clusterm_corrp_tstat1.nii.gz'
    tstat_path = STAT_DIR + f'corr_{name}_tstat1.nii.gz'
    if not (os.path.exists(corrp_path) and os.path.exists(tstat_path)):
        continue
    corrp = np.asarray(nib.load(corrp_path).get_fdata(), dtype=np.float32)
    tstat = np.asarray(nib.load(tstat_path).get_fdata(), dtype=np.float32)
    sig_mask = corrp > CORRP_THR
    if sig_mask.any() and MIN_CLUSTER_VOX > 1:
        lab, _ = ndimage.label(sig_mask, structure=ndimage.generate_binary_structure(3, 3))
        sizes = np.bincount(lab.ravel())
        small = np.flatnonzero(sizes < MIN_CLUSTER_VOX)
        small = small[small > 0]
        if small.size:
            sig_mask &= ~np.isin(lab, small)
    n_sig = int(sig_mask.sum())
    sig_tmap = np.zeros(corrp.shape, dtype=np.float32)
    sig_tmap[sig_mask] = tstat[sig_mask]
    out_name = 'sigtmap_' + name.removeprefix('beta4d_') + '.nii'
    if n_sig == 0:
        if os.path.exists(STAT_DIR + out_name):
            os.remove(STAT_DIR + out_name)
        else:
            print(f'Significant t map: {name}: no significant cluster, skip')
        continue
    nib.save(nib.Nifti1Image(sig_tmap, nib.load(tstat_path).affine), STAT_DIR + out_name)
    print(f'Significant t map: {name}: {n_sig} voxels -> {out_name}')
