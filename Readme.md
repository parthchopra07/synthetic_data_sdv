#  Synthetic Data Generation for Healthcare Analysis using GANs

##  Overview

This project explores the use of advanced deep learning models — specifically **Conditional GANs (CTGAN)**, **TVAE**, and **Gaussian Copula** — to generate synthetic healthcare datasets based on real COVID-19 patient records.

The synthetic data generated preserves the statistical properties of real patient data while ensuring privacy, making it highly valuable for research and development in sensitive domains like healthcare.

---

##  Objectives

- Preprocess real-world healthcare data, addressing missing and inconsistent entries.
- Train synthetic data generators using:
  - CTGAN (Conditional Tabular GAN)
  - TVAE (Tabular Variational AutoEncoder)
  - Gaussian Copula
- Evaluate data quality using `SDV`'s diagnostic reports.
- Visualize distribution comparisons between real and synthetic data.
- Train a Random Forest classifier on the synthetic data.
- Apply SHAP (SHapley Additive exPlanations) for interpretability and feature importance analysis.

---

##  Dataset

The dataset is derived from a publicly available **COVID-19 Mexican Patient Health Record**, including:

- Demographics: Age, Sex
- Health conditions: Hypertension, Diabetes, Asthma, etc.
- Patient classification: ICU admission, recovery, or death

**Note:** Values such as `97` and `99` were used to represent missing data and were cleaned during preprocessing.

---

##  Tech Stack

- **Python** (Pandas, NumPy, Matplotlib, Seaborn)
- **Scikit-learn** (Random Forest Classifier)
- **SDV (Synthetic Data Vault)** - `CTGAN`, `TVAE`, `GaussianCopula`
- **SHAP** - for feature importance visualization

---

##  Results & Evaluation

- All three models successfully generated statistically sound synthetic datasets.
- CTGAN showed superior performance in mimicking the original data distribution.
- SHAP visualizations helped interpret the model's predictions, revealing age and ICU status as major predictors.

---

## Key Takeaways

- Synthetic data can effectively replace sensitive patient records for initial model development.
- CTGANs are particularly effective for capturing complex dependencies in tabular health data.
- Model interpretability using SHAP adds critical trust to AI-driven healthcare systems.

---

## Collaboration

I'm always open to feedback and potential collaboration in the intersection of **AI, healthcare, and data privacy**. Feel free to fork this repo, raise an issue, or connect with me on [LinkedIn](https://www.linkedin.com/in/parth-chopra07/).

---

##  License

This project is open-source and available under the MIT License.
