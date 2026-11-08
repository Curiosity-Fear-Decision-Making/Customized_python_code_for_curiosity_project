# Foldername: custom_python
# Filename: fig4_analysis_rsa_postchoice_selfrating_binary_secondlevel.py
import nibabel as nib
import numpy as np
from pathlib import Path
import hashlib
import tempfile
import xml.etree.ElementTree as ET
from nibabel.processing import resample_from_to
from nilearn.image import smooth_img
import subprocess
from tqdm import tqdm
from utility_custom_python import PROJECT_DIR as PACKAGE_DIR



def load_permutation_mask():
    atlas_image = nib.squeeze_image(nib.load(OFC_ATLAS_PATH))
    white_matter_image = nib.squeeze_image(nib.load(WHITE_MATTER_MASK_PATH))
    atlas_values = atlas_image.get_fdata(dtype=np.float32)
    white_matter_values = white_matter_image.get_fdata(dtype=np.float32)
    labels = {label.findtext('name'): int(label.findtext('index')) for label in ET.parse(OFC_LABELS_PATH).getroot().iter('label')}
    ofc_voxels = np.isin(atlas_values, [labels[name] for name in OFC_LABEL_NAMES])
    white_matter_image = nib.Nifti1Image((white_matter_values > 0).astype(np.uint8), white_matter_image.affine)
    if white_matter_image.shape != atlas_image.shape or not np.allclose(white_matter_image.affine, atlas_image.affine):
        white_matter_image = resample_from_to(white_matter_image, (atlas_image.shape, atlas_image.affine), order=0)
    excluded_voxels = ofc_voxels & (white_matter_image.get_fdata() > 0)
    correction_voxels = ofc_voxels & ~excluded_voxels
    return nib.Nifti1Image(correction_voxels.astype(np.uint8), atlas_image.affine)


def permutation_mask_signature(mask_image):
    data = np.ascontiguousarray(mask_image.dataobj, dtype=np.uint8)
    affine = np.asarray(mask_image.affine, dtype='<f8')
    model = b'binary_fixed:low123;high456;choice_pooled;choice_main;threat_social_shock'
    smoothing = f'level2_smooth={SMOOTH_AT_SECOND_LEVEL};fwhm_mm={SECOND_LEVEL_SMOOTH_FWHM:g}'.encode('ascii')
    digest = hashlib.sha256(model + smoothing + data.tobytes() + affine.tobytes()).hexdigest()
    return ('OFC-WM:' + digest).encode('ascii')


def prepare_second_level_volume(volume, mask=None):
    volume = np.asarray(volume, dtype=np.float32)
    if not CENTER_BETA_2ND_LEVEL:
        return np.nan_to_num(volume, nan=0.0)
    mask = GM_BOOL if mask is None else np.asarray(mask, dtype=bool)
    centered = np.nan_to_num(volume.copy(), nan=0.0)
    mean_beta = float(volume[mask].mean(dtype=np.float64))
    centered[mask] -= mean_beta
    return centered


def permutation_stack_name(choice):
    suffix = '_centered' if CENTER_BETA_2ND_LEVEL else ''
    return f'rsa4d_{choice}_binary_up{suffix}'


def run_permutation(choice, subject_ids):
    beta_paths = [SUBJECT_RESULT_DIR / f'{subject_id}_{choice}_binary.nii.gz' for subject_id in subject_ids]
    reference_beta_image = nib.load(beta_paths[0])
    mask_image = load_permutation_mask()
    beta_volumes = []
    for beta_path in tqdm(beta_paths, desc=f'Load beta maps: {choice}', unit='subject'):
        beta_image = nib.load(beta_path)
        beta_values = beta_image.get_fdata(dtype=np.float32)
        if SMOOTH_AT_SECOND_LEVEL:
            smoothed_beta_image = smooth_img(beta_image, SECOND_LEVEL_SMOOTH_FWHM)
            beta_values = smoothed_beta_image.get_fdata(dtype=np.float32)
        beta_volumes.append(prepare_second_level_volume(beta_values))
    stack = np.stack(beta_volumes, axis=3)
    stack_name = permutation_stack_name(choice)
    stack_path = GROUP_RESULT_DIR / (stack_name + '.nii.gz')
    nib.save(nib.Nifti1Image(stack, reference_beta_image.affine), stack_path)
    prefix = GROUP_RESULT_DIR / ('corr_' + stack_name)
    with tempfile.TemporaryDirectory(prefix='rsa_ofc_no_wm_') as temporary:
        correction_mask_path = Path(temporary) / 'ofc_no_wm.nii.gz'
        nib.save(mask_image, correction_mask_path)
        subprocess.run(
            ['randomise', '-i', str(stack_path), '-o', str(prefix), '-1', '-c', str(PERM_CTHR), '-n', str(PERM_NPERM), '-m', str(correction_mask_path)],
            stdout=subprocess.DEVNULL,
            check=True,
        )
    signature = permutation_mask_signature(mask_image)
    for suffix in ('_tstat1.nii.gz', '_clustere_corrp_tstat1.nii.gz'):
        output_path = Path(str(prefix) + suffix)
        image = nib.load(output_path)
        values = image.get_fdata(dtype=np.float32)
        header = image.header.copy()
        header['descrip'] = signature
        nib.save(nib.Nifti1Image(values, image.affine, header), output_path)


def export_permutation(choice):
    prefix = GROUP_RESULT_DIR / ('corr_' + permutation_stack_name(choice))
    corrp_img = nib.load(str(prefix) + '_clustere_corrp_tstat1.nii.gz')
    t_img = nib.load(str(prefix) + '_tstat1.nii.gz')
    mask_img = load_permutation_mask()
    mask = mask_img.get_fdata() > 0
    corrp = corrp_img.get_fdata(dtype=np.float32)
    min_p = float(1.0 - corrp[mask].max())
    sig_mask = mask & (corrp > np.float32(CORRP_THR))
    sig_t = np.where(sig_mask, t_img.get_fdata(dtype=np.float32), 0).astype(np.float32)
    suffix = '_centered' if CENTER_BETA_2ND_LEVEL else ''
    sig_name = f'sigtmap_{choice}_binary_up{suffix}.nii'
    sig_path = GROUP_RESULT_DIR / sig_name
    if sig_mask.any():
        nib.save(nib.Nifti1Image(sig_t, t_img.affine), sig_path)
    else:
        sig_path.unlink(missing_ok=True)
    print(f'{choice}: min p_FWE={min_p:.6f}; {int(sig_mask.sum())} significant voxels')


RESULT_TAG = 's3_rsa_shock_selfrating_binary'
SUBJECT_RESULT_DIR = PACKAGE_DIR / f'result_{RESULT_TAG}'
GROUP_RESULT_DIR = PACKAGE_DIR / f'stat_{RESULT_TAG}'
OFC_ATLAS_PATH = f'{PACKAGE_DIR}/brainmask/aal/AAL3/ROI_MNI_V7.nii'
OFC_LABELS_PATH = f'{PACKAGE_DIR}/brainmask/aal/AAL3/ROI_MNI_V7.xml'
OFC_LABEL_NAMES = ('OFCmed_L', 'OFCmed_R', 'OFCant_L', 'OFCant_R', 'OFCpost_L', 'OFCpost_R')
WHITE_MATTER_MASK_PATH = f'{PACKAGE_DIR}/brainmask/mask_white_matter.nii'
SUBJECT_LIST = [f'par{i:02d}' for i in range(1, 31)]
CHOICES = ('skip', 'story')
CENTER_BETA_2ND_LEVEL = globals().get('CENTER_BETA_2ND_LEVEL', False)
RUN_PERMUTATION = True
SMOOTH_AT_SECOND_LEVEL = True
SECOND_LEVEL_SMOOTH_FWHM = 5
PERM_CTHR = 2
PERM_NPERM = 1000
CORRP_THR = 0.95
GM_BOOL = (np.squeeze(nib.load(PACKAGE_DIR / 'brainmask/mask_wholebrain.nii').get_fdata()) == 1) & (np.squeeze(nib.load(WHITE_MATTER_MASK_PATH).get_fdata()) <= 0)
GROUP_RESULT_DIR.mkdir(exist_ok=True)
if RUN_PERMUTATION:
    print('Starting permutation with FSL randomise.', flush=True)
    for choice in CHOICES:
        subject_ids = [subject_id for subject_id in SUBJECT_LIST if (SUBJECT_RESULT_DIR / f'{subject_id}_{choice}_binary.nii.gz').exists()]
        run_permutation(choice, subject_ids)
for choice in tqdm(CHOICES, desc='Export significant maps', unit='choice'):
    export_permutation(choice)
print(f'Permutation outputs: {GROUP_RESULT_DIR}')
