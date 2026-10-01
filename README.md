# ipv-neuroimaging-representation-mcfarland

This repository includes scripts to reproduce the analyses reported in:

McFarland, C. W., Kucyi, A., & Valera, E. M. (2026). Neuroimaging research on intimate partner violence can underrepresent the most affected survivors. bioRxiv. https://doi.org/10.64898/2026.09.23.753381

It contains the complete pipeline for analyzing screening, interview, questionnaire, and resting-state fMRI head-motion data from a community cohort of women with lifetime exposure to physical intimate partner violence (IPV), collected as part of a study of IPV-related brain injury in the Valera Lab at Massachusetts General Hospital and Harvard Medical School (PI: Eve M. Valera). The workflow spans questionnaire scoring and the analysis of three stages of neuroimaging sample construction: eligibility, participation, and head-motion quality control.

Workflow Overview:

1. Scoring and Injury Exposure

These scripts score the questionnaires and summarize the brain injury exposure used across all analyses.

01_score_ctq_casr.py
Scores the Childhood Trauma Questionnaire–Short Form (25-item total; minimization–denial items scored separately) and the Composite Abuse Scale (Revised)–Short Form (past-12-month frequency, range 0–75) from item-level REDCap data.

02_injury_distribution.py
Summarizes the distribution of cumulative partner-inflicted brain injury count and applies winsorizing at 25.

2. Sample Construction Analyses (Python 3.11)

These scripts reproduce all results and supplementary tables in the manuscript.

03_stage1_eligibility.py

Stage: Eligibility
Analyses: Neuroimaging eligibility among behaviorally eligible women, reasons for ineligibility, criteria retained throughout versus initial but dropped (Supplementary Table 2).

04_stage2_participation.py

Stage: Participation
Analyses: Comparison of enrolled women with and without resting-state fMRI data across socioeconomic, psychiatric, neurobehavioral, injury, abuse, and demographic measures (Supplementary Table 1).

05_stage3_motion_qc.py

Stage: Quality control
Analyses: Head-motion exclusion by partner-inflicted brain injury count at mean framewise displacement thresholds of 0.15, 0.20, and 0.30 mm; injury threshold and leave-one-out sensitivity analyses (Supplementary Tables 3–5).
