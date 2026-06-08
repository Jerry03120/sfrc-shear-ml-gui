# SFRC Shear ML GUI

## SFRC Transfer Learning Model

This repository provides a Python-based graphical user interface (GUI) program for predicting the shear strength of reinforced concrete (RC) and steel fiber reinforced concrete (SFRC) deep beams using machine learning and heterogeneous transfer learning.

The main purpose of this program is to support RC-to-SFRC transfer learning analysis, model comparison, and result visualization for research applications.

---

## Main File

```bash
Model_0607_2026_GUI.py
```

---

## Key Features

* Python GUI for RC/SFRC shear prediction analysis
* Heterogeneous transfer learning from RC source data to SFRC target data
* Target-only machine learning baseline models
* Literature-based shear strength equation comparison
* Automatic metric calculation:

  * R²
  * RMSE
  * MAE
  * A/P ratio
  * CoV of A/P
  * Pearson correlation coefficient
* Automatic export of result tables and figures
* Support for manuscript-style comparison figures

---

## Algorithms Used

The GUI is configured to evaluate the main algorithms used in the study, including:

* HTL-XGB
* HTL-ET
* HTL-RF
* XGB Target-Only
* ET Target-Only
* RF Target-Only

Additional RC source model search functions may be included internally for transfer learning source-model selection.

---

## Required Python Packages

Install the required packages before running the program.

```bash
pip install numpy pandas matplotlib scipy scikit-learn xgboost lightgbm catboost shap statsmodels
```

Some packages such as `xgboost`, `lightgbm`, `catboost`, `shap`, and `statsmodels` may be optional depending on the selected analysis options.

---

## How to Run

Run the GUI program with Python:

```bash
python Model_0607_2026_GUI.py
```

After launching the GUI, select the required input CSV files through the interface.

---

## Input Data

The program is designed to use CSV datasets for:

* RC beam database
* SFRC Group 1 database
* SFRC Group 2 database

Typical input variables include:

* Beam width, `b`
* Effective depth, `d`
* Shear span-to-depth ratio, `a/d`
* Concrete compressive strength, `f'c`
* Longitudinal reinforcement ratio, `rho`
* Fiber volume fraction, `Vf`
* Reinforcing index, `RI`
* Fiber aspect ratio, `Lf/Df`
* Residual tensile strength parameters, when available

The target variable is typically the experimental shear strength, such as `Vu` or shear stress `tau_u`.

---

## Notes

This code may contain default local file paths for the original research environment.
If the program is used on another computer, input CSV files should be selected manually through the GUI.

Please make sure that private datasets, unpublished experimental data, or confidential research files are not uploaded to a public repository unless they are intended to be shared.

---

## Repository Structure

```text
sfrc-shear-ml-gui/
│
├── README.md
└── Model_0607_2026_GUI.py
```

---

## Research Purpose

This repository was prepared for research on machine learning-based shear strength prediction of RC and SFRC deep beams, with a focus on heterogeneous transfer learning from RC data to limited SFRC data.

---

## Author

Jerry03120

---

## License

This repository is currently provided for research and academic use.
Please contact the author before using the code for commercial purposes.
