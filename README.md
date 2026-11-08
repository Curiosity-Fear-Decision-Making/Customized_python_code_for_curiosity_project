# Curiosity and threat analysis scripts

- **Spider experiment (Fig. 2E–L).** The `fig2*.py` scripts analyze ratings, choices, and hand movement during the in-person spider experiment.
- **Online experiment (Fig. 3B–I).** The `fig3*.py` scripts analyze choices, ratings, reaction times, and questionnaire measures from the online information-seeking experiment.
- **fMRI behavior (Fig. 4).** The first-level RSA scripts and the connectivity scripts read the fMRI participants' ratings, choices, and trial conditions to build their analysis inputs.
- **fMRI RSA (Fig. 4).** The `fig4_analysis_rsa_*_firstlevel.py` scripts fit participant GLMs. The corresponding `*_secondlevel.py` scripts perform group analyses using participant maps from `result_*/`.
- **fMRI connectivity (Fig. 4).** The `fig4_connectivity_network_*.py` scripts analyze and plot the Story and Skip brain networks for curiosity, threat, and their interaction.

## Software

Use Python 3.12. Create the environment in `custom_python/` on macOS or Linux.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Group permutation analyses require FSL `randomise` on `PATH`. [FSL installation guide](https://fsl.fmrib.ox.ac.uk/fsl/docs/install/index.html)

## Data downloads

Download the inputs for your analysis from [Science Data Bank](https://doi.org/10.57760/sciencedb.0175h). Extract the archives and place the folders directly under `custom_python/` as shown below.

| Analysis | Folder under `custom_python/` |
| --- | --- |
| Fig. 2 spider behavior | `beh_spider/` |
| Fig. 3 online information-seeking behavior | `beh_online_seeking_exp/` |
| Fig. 4 fMRI behavior | `beh_fmri/` |
| Fig. 4 connectivity | `connectivity_inputs/` |
| S1 curiosity localizer RSA | `file_generated/searchlight_r_rsa_s1/` |
| S2 threat localizer RSA | `file_generated/searchlight_r_rsa_s2/` |
| S3 clickbait task RSA | `file_generated/searchlight_r_rsa_s3_clickbaitshockstick/` |
| S3 postchoice self-rating RSA | `file_generated/searchlight_r_rsa_s3_shock/` |

## MATLAB searchlight scripts

`searchlight_r_matlab_voxel_s1_level.m` computes searchlight similarities from condition-level beta maps for the **S1 curiosity localizer** and writes them to `file_generated/searchlight_r_rsa_s1_level/`. `searchlight_r_matlab_voxel_s2_level.m` does the same for the **S2 threat localizer** and writes to `file_generated/searchlight_r_rsa_s2_level_subj/`. The Python first-level GLMs for S1 and S2 use trial-level searchlight arrays from Science Data Bank, in the two localizer folders listed above.

The two S3 MATLAB scripts compute trial-level searchlight similarities for the clickbait task and shock period, respectively.
