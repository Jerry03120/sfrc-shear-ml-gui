# SFRC Shear TL GUI

## SFRC Transfer Learning Model

This repository contains a Python GUI program for shear strength prediction of reinforced concrete (RC) and steel fiber reinforced concrete (SFRC) beams using machine learning and heterogeneous transfer learning.

The program is designed for research on RC-to-SFRC transfer learning and compares the final selected machine learning models used in the study.

---

## Main File

```bash
Model_0607_2026_GUI.py
```

---

## Algorithms Used

The program evaluates only the final algorithms used in the model comparison:

* HTL-XGB
* HTL-ET
* HTL-RF
* XGB Target-Only
* ET Target-Only
* RF Target-Only

Where:

* HTL = Heterogeneous Transfer Learning
* XGB = Extreme Gradient Boosting
* ET = Extra Trees
* RF = Random Forest
* Target-Only = Model trained using only the target SFRC dataset

---

## Key Features

* Graphical user interface for model execution
* RC-to-SFRC heterogeneous transfer learning
* Target-only baseline model comparison
* Automatic result table export


---

## Performance Metrics

The program calculates model performance using:

* R²
* RMSE
* MAE
* Mean A/P ratio
* Coefficient of variation of A/P
* Pearson correlation coefficient

---

## Required Python Packages

Install the required packages before running the program.

```bash
pip install numpy pandas matplotlib scipy scikit-learn xgboost
```

---

## How to Run

Run the GUI program using Python:

```bash
python Model_0607_2026_GUI.py
```

After launching the GUI, select the required CSV input files through the interface.

---

## Input Data

The program uses CSV files for:

* RC source dataset
* SFRC Group 1 target dataset
* SFRC Group 2 target dataset

Typical input variables include:

* Beam width, `b`
* Effective depth, `d`
* Shear span-to-depth ratio, `a/d`
* Concrete compressive strength, `f'c`
* Longitudinal reinforcement ratio, `rho`
* Fiber volume fraction, `Vf`
* Reinforcing index, `RI`
* Fiber aspect ratio, `Lf/Df`
* Residual strength parameters, when available
* Experimental shear strength, `Vu`

---

## Notes

The source code may include default local paths from the original research environment.
When running the program on another computer, select the input CSV files manually through the GUI.

Do not upload private datasets, unpublished experimental data, or confidential research files to a public repository unless they are intended to be shared.

---

## Repository Structure

```text
sfrc-shear-ml-gui/
│
├── README.md
└── Model_0607_2026_GUI.py
```

---

## Author

Jerry03120

---

## License

This repository is provided for academic and research use.
