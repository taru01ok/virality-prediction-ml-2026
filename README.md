# Predicting Social Media Content Virality

**Course:** CSCI-4930 Machine Learning, Spring 2026 — University of Colorado Denver
**Instructor:** Dr. Ashis Biswas
**Team:** Taruni Sabhavat, Jathin Reddy Pati

## Project Overview

A machine learning classifier that predicts whether a social media post will go viral BEFORE it is published, across Instagram, YouTube, and TikTok. Includes a full fairness audit across platforms and content categories.

**Virality Definition:** Top 20% engagement rate per platform.

## Data
Download datasets from Google Drive: https://drive.google.com/drive/folders/1aWALtvnJTxLUQHyaCJNKfxW6wwZprxgo

Place in data/raw/ with these exact filenames:
- Instagram_Analytics.csv
- USvideos.csv
- US_category_id.json
- tiktok_dataset.csv

## Setup

Python version: 3.10+

pip install -r requirements.txt

Run notebooks in order: 01_EDA → 02_preprocessing → 03_modeling → 04_evaluation → 05_fairness_audit

## Results

Best model: Random Forest (F1=0.53, ROC-AUC=0.80)

Fairness findings:
- YouTube F1=0.84 vs Instagram F1=0.08
- Entertainment F1=0.88 vs Fitness F1=0.11

## Notebooks
- 01_EDA.ipynb - Exploratory data analysis
- 02_preprocessing.ipynb - Feature engineering and virality labeling
- 03_modeling.ipynb - Train all 6 classifiers
- 04_evaluation.ipynb - Model comparison and evaluation
- 05_fairness_audit.ipynb - Platform and content category fairness audit
