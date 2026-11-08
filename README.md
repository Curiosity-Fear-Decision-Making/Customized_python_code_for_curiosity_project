# Custom Python analyses for the curiosity–threat study

Each plotting script displays one panel with `plt.show()` and does not save images. Fig. 2E–L and Fig. 3B–I retain their source panel labels. The six Fig. 4 connectivity scripts use descriptive filenames for the Story and Skip brain networks for curiosity, threat, and their interaction. Each panel script contains the main analysis and the code that selects what to plot. Shared input readers, statistical utilities, and drawing helpers are in `utility_custom_python.py`.

## Install the Python environment

Install Python 3.12. Open a terminal in the `custom_python/` folder and run these commands on macOS or Linux.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip check
```

In a new terminal, run `source .venv/bin/activate` from `custom_python/` before running the scripts. In PyCharm, select `.venv/bin/python` as the project's interpreter.

## Run the scripts

```bash
python fig2e_rating_consistency.py
python fig3g_regret_curiosity.py
python fig4_connectivity_network_story_threat.py
```

Set up the connectivity inputs described below before running a network script. In PyCharm, use **Run** on the script file.

## Fig. 4 connectivity inputs

The `connectivity_inputs` folder for the `fig4_connectivity_network_*.py` scripts will be distributed separately through [Science Data Bank](https://doi.org/10.57760/sciencedb.0175h). Download and extract it directly inside `custom_python/` so the files are at the paths shown below.

```text
custom_python/connectivity_inputs/decision_par01.npz
custom_python/connectivity_inputs/shock_par01.npz
```

The folder contains 60 files: decision and shock inputs for participants `par01`–`par30` (approximately 667 MB). These contain unchanged trial-wise voxel values extracted from the original beta images in the 16 merged analysis ROIs. The network scripts refit the participant RSA GLMs, calculate across-participant correlations, and apply the multiple-comparison correction before plotting.

## fMRI GLMs

### First level

The first-level GLM scripts are distributed on GitHub, and the searchlight inputs will be distributed separately through Science Data Bank. To reproduce a first-level GLM, download the matching searchlight folder from Science Data Bank and place it directly inside `custom_python/file_generated/`. Create `file_generated/` if needed. Keep the downloaded folder name and its 30 participant arrays together.

| Analysis | Example path under `file_generated/` |
| --- | --- |
| S1 localizer curiosity | `searchlight_r_rsa_s1/par01_run1.npy` |
| S2 localizer threat | `searchlight_r_rsa_s2/par01_run2.npy` |
| S3 clickbait task | `searchlight_r_rsa_s3_clickbaitshockstick/par01_run3.npy` |
| S3 postchoice self-rating | `searchlight_r_rsa_s3_shock/par01_run3.npy` |

Once the inputs are in place, run the corresponding `*_firstlevel.py` script from `custom_python/`. The scripts show participant progress and write participant GLM maps to their matching `result_*` folders.

### Second level

The four `*_secondlevel.py` scripts use participant GLM maps from the matching `result_*` folders. If these folders are missing, run the corresponding first-level scripts or supply the participant GLM maps before running the second level.

```bash
python fig4_analysis_rsa_effect_clickbait_task_secondlevel.py
python fig4_analysis_rsa_effect_localizer_curiosity_secondlevel.py
python fig4_analysis_rsa_effect_localizer_threat_secondlevel.py
python fig4_analysis_rsa_postchoice_selfrating_binary_secondlevel.py
```

For a new permutation run, install FSL separately and check that `command -v randomise` returns a path in the terminal where you run Python. On Windows, use WSL for the FSL workflow and create the Python environment inside WSL. [FSL installation guide](https://fsl.fmrib.ox.ac.uk/fsl/docs/install/index.html)

With `RUN_PERMUTATION = True`, the scripts run FSL `randomise` from `PATH`. The console shows a single message when the permutation stage starts. If the matching `stat_*` folders already contain corrected p-value and t-statistic maps, set `RUN_PERMUTATION = False` to reuse those maps for thresholding. New permutation runs can differ because no seed is set.

Second-level NIfTI outputs go to the matching `stat_*` folders. For S3 postchoice self-rating, a contrast with significant voxels produces a `sigtmap_<choice>_binary_up.nii` file containing the original t values at voxels with FWE-corrected `p < 0.05` within the OFC correction mask. Other voxels are set to zero.

## Behavioral inputs

Download the relevant behavioral data from Science Data Bank and extract any archives. Place each folder directly inside `custom_python/`, with the behavioral files directly inside the corresponding folder.

| Analysis | Science Data Bank download | Folder under `custom_python/` |
| --- | --- | --- |
| Fig. 2 | `beh_spider.zip` | `beh_spider/` |
| Fig. 3 | `beh_online_seeking_exp.zip` | `beh_online_seeking_exp/` |
| Fig. 4 first-level GLMs and brain networks | `beh_fmri` | `beh_fmri/` |

Fig. 3 uses the 486 participants in the shared online information-seeking cohort. The `beh_fmri/` folder contains behavioral records for the 30 fMRI participants, including `output_formal_clickbaitRes_par01.txt`.

## Reference inputs and code
- `brainmask`, `assets`, and `toolbox_3rdParty`: required anatomical inputs and the MATLAB voxel-coordinate mapping.
- `searchlight*.m`: existing MATLAB reference code. The S3 files are trial-based. The existing S1/S2 files are **level-based**, and are not the upstream generators of the trial-wise S1/S2 arrays expected by the Python first-level scripts. These reference files require MATLAB, `MRIread`, `writeNPY`, and their separate beta/design inputs; they are not needed for second-level reproduction.
