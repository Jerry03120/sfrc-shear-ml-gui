# SFRC Shear TL GUI

> A GUI for predicting the shear strength of RC and SFRC beams using Heterogeneous Transfer Learning (HTL)

RC beam data is used as the source domain and SFRC beam data as the target domain to compare HTL models against target-only baseline models. Supported features include model training, result export, prediction with saved models, SHAP analysis, and an input guide for RC and SFRC datasets.

---

## Quick Start

### 1. Install dependencies

```bash
pip install numpy pandas matplotlib scipy scikit-learn xgboost shap
```

### 2. Run the GUI

```bash
python Model_GUI_HTL_XGB_patched.py
```

> **Note** — Default file paths in the source code are set to the original research environment.  
> When running on a different machine, select the CSV files manually in the **Run** tab.

---

## GUI Tabs

| Tab | Description |
|---|---|
| **Run** | Load CSV files → run transfer learning analysis → export results |
| **Input Guide** | View required input columns for RC source data and SFRC Group 1 & 2 |
| **Predict** | Load saved models → predict shear strength from user-entered parameters |

---

## Models

Six machine learning models are evaluated using two training strategies.

### HTL Models — trained on source + target domain data

| Model | Description |
|---|---|
| **HTL-XGB** | Heterogeneous Transfer Learning + XGBoost |
| **HTL-ET** | Heterogeneous Transfer Learning + Extra Trees |
| **HTL-RF** | Heterogeneous Transfer Learning + Random Forest |

### Target-Only Models — baseline, trained on SFRC data only

| Model | Description |
|---|---|
| **XGB Target-Only** | XGBoost trained on SFRC data only |
| **ET Target-Only** | Extra Trees trained on SFRC data only |
| **RF Target-Only** | Random Forest trained on SFRC data only |

---

## Input Data

Three CSV files are required:

| File | Description |
|---|---|
| RC source dataset | Conventional RC beam test data used as the source domain |
| SFRC Group 1 | SFRC beam test data with basic fiber-index variables |
| SFRC Group 2 | SFRC beam test data with additional tensile variables |

---

## Input Variables

### RC Source Domain

| Variable | Description | Unit |
|---|---|---|
| `b_mm` | Beam width | mm |
| `d_mm` | Effective depth | mm |
| `a_d` | Shear span-to-depth ratio | — |
| `fc_MPa` | Concrete compressive strength | MPa |
| `rho` | Longitudinal reinforcement ratio | % |
| `V_u_KN` | Experimental shear strength | kN |
| `tau_u_MPa` | Shear stress — auto-calculated from `V_u_KN`, `b_mm`, `d_mm` if not provided | MPa |

### SFRC Group 1

Includes all RC source domain variables, plus the following fiber-related variables:

| Variable | Description | Unit |
|---|---|---|
| `V_f_pct` | Fiber volume fraction | % |
| `RI` | Reinforcing index | — |
| `Lf_per_Df` | Fiber aspect ratio | — |

### SFRC Group 2

Includes all Group 1 variables, plus the following tensile variables:

| Variable | Description | Unit |
|---|---|---|
| `fsp_MPa` | Splitting tensile strength | MPa |
| `ft_direct_MPa` | Direct tensile strength | MPa |
| `fr_MPa` | Modulus of rupture | MPa |

---

## Prediction Mode (Predict Tab)

Load a saved model after training to predict shear strength from user-entered values.

**Outputs**
- `V_u` (kN)
- `τ_u` (MPa)

**Input requirements**

- `a/d` must be ≥ **2.0**
- **Group 1** — enter RC common variables + fiber-index variables
- **Group 2** — enter Group 1 variables + numeric values for `f_sp`, `f_t,dir`, and `f_r`

---

## Performance Metrics

| Metric | Description |
|---|---|
| **R²** | Coefficient of determination |
| **RMSE** | Root mean squared error |
| **MAE** | Mean absolute error |
| **A/P mean** | Mean of actual-to-predicted ratio |
| **A/P CoV** | Coefficient of variation of actual-to-predicted ratio |
| **Pearson r** | Pearson correlation coefficient |
| **Spearman r** | Spearman rank correlation coefficient |

---

## Exported Outputs

The following files are generated automatically after each run:

- Cross-validation metric summary tables
- Predicted vs. observed scatter plots
- Model comparison figures
- Literature equation comparison results
- SHAP analysis outputs
- Saved model files for use in the Predict tab

---

## Repository Structure

```
sfrc-shear-ml-gui/
├── README.md
└── Model_GUI_HTL_XGB_patched.py
```

---

## Notes

- This GUI was designed for RC-to-SFRC heterogeneous transfer learning research.
- Default paths in the Python file may need to be updated to match your local directory structure.
- For Group 2 prediction, tensile parameters must be entered as numeric values.
- The GUI layout is optimized for readable screenshots suitable for academic papers.

---

## Author

Jerry03120

## License

For academic and research use only.
