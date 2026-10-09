# AI-Driven Ensemble Machine Learning Framework for Multi-Class Autoimmune Disease Classification

Gut microbiome–based classification of 11 phenotypic classes (10 autoimmune conditions + healthy controls) using gradient-boosting models and a soft-voting ensemble.

*Internship project, Jamia Hamdard (MCAI).*


**Authors:**
- Aman Zehra, M.Sc. Biotechnology, Department of Biotechnology, Jamia Hamdard
- Mohammad Younus, B.Tech Computer Science Engineering, Jamia Hamdard

**Supervisor:** Dr. Hira Javed, School of Engineering Sciences and Technology, Jamia Hamdard
**Programme:** Internship at the Multidisciplinary Centre for Artificial Intelligence (MCAI), Jamia Hamdard, New Delhi
## Project Overview

Dysbiosis of the gut microbiota is closely linked to immune dysregulation and autoimmune disease. Instead of single-disease comparisons, this project builds one unified model that distinguishes many autoimmune disorders at once, using genus-level microbial abundance profiles.

- **Samples:** 1,955 across 11 classes (CD, CeD, HC, MG, MS, PS, RA, SLE, SpA, T1D, UC)
- **Features:** 423 gut microbial genera (abundance profiles)
- **Models compared:** XGBoost, LightGBM, CatBoost, Soft Voting Ensemble
- **Tuning:** Optuna hyperparameter optimisation with stratified 5-fold CV
- **Output:** Evaluation plots and feature-importance analysis to identify key discriminating taxa

- ## Data source
The genus-level abundance matrix (1,955 samples, 423 genera, 11 classes) comes from a published, publicly available dataset:
An, T., Zhang, S., Li, J., et al. (2025). *Gut microbiota analysis reveals microbial signature for multi-autoimmune diseases based on machine learning model.* Frontiers in Microbiology, 16, 1660775. https://doi.org/10.3389/fmicb.2025.1660775

This repository reuses that data for an independent modelling exercise; all credit for the data belongs to the original authors.

## Performance (hold-out test set)

| Model | Accuracy | F1-score | ROC-AUC |
|---|---|---|---|
| LightGBM | 0.612 | 0.573 | 0.867 |
| CatBoost | 0.588 | 0.537 | 0.858 |
| XGBoost | **0.629** | **0.600** | 0.871 |
| Soft Voting Ensemble | 0.605 | 0.565 | **0.873** |

Eleven-class classification is a hard problem: accuracy is moderate, while ROC-AUC above 0.85 for all models shows good class discrimination. *Clostridium* had the highest feature importance in XGBoost.

## Citation
If you use this code, please cite this repository and the original data paper above.


## Repository Structure

```
├── main.py                  # full pipeline: load, tune, train, evaluate, plot
├── Dataset.xlsx             # genus-level abundance data
├── evaluation_results.png   # output figure
├── requirements.txt
└── README.md
```

## How to Run

```bash
git clone https://github.com/amanzehra18-oss/MCAI_INTERN.git
cd MCAI_INTERN
pip install -r requirements.txt
python main.py
```

## Results

![Evaluation results](evaluation_results.png)


## My Contribution

- Studied the biological background of gut dysbiosis and autoimmune diseases and wrote the Introduction and Discussion of the project report
- Worked on [data preparation / running and comparing XGBoost, LightGBM, CatBoost and the Soft Voting Ensemble]
- Interpreted the feature-importance results (e.g. Clostridium, Fusobacterium) in the context of immune regulation and gut microbiota
- Co-authored the final project report with my teammate Mohammad Younus under the supervision of Dr. Hira Javed, Jamia Hamdard

## Dataset

Data from An et al. (2025), "Gut microbiota analysis reveals microbial signature for multi-autoimmune diseases based on machine learning model", Frontiers in Microbiology, 16. https://doi.org/10.3389/fmicb.2025.1660775

## Team

Developed as a group internship project with [Mohammad Younus Anis](https://github.com/MOHDYOUNUSANIS).
