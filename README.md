# SFRC Shear TL GUI

A graphical user interface (GUI) for predicting the shear strength of RC and SFRC beams using heterogeneous transfer learning (HTL). The program uses RC beam data as the source domain and SFRC beam data as the target domain to compare HTL models with target-only baseline models.

The GUI supports model training, result export, prediction using saved models, SHAP analysis, and an input guide for RC, SFRC Group 1, and SFRC Group 2 datasets.

---

## Getting started

Install the required Python packages, then run the GUI:

```bash
pip install numpy pandas matplotlib scipy scikit-learn xgboost
python Model_0608_2026_GUI_auto_patched_v9.py
```

Optional packages can be installed if additional model functions are needed:

```bash
pip install lightgbm catboost shap
```

When the GUI opens, use the **Run** tab to select the required CSV input files. The default file paths in the source code are from the original research environment, so users on a different machine should select the CSV files manually.

---

## GUI tabs

The program includes four main tabs:

| Tab         | Description                                                                         |
| ----------- | ----------------------------------------------------------------------------------- |
| Run         | Load input CSV files, run transfer learning analysis, and export results            |
| Input Guide | Shows the required input columns for RC source data, SFRC Group 1, and SFRC Group 2 |
| Predict     | Load saved models and predict shear strength using user-entered parameters          |
| Preview     | Preview exported results, figures, and saved outputs                                |

---

## Models

Six machine learning models are evaluated using two training strategies.

### HTL models

HTL models use RC source-domain data together with SFRC target-domain data.

* HTL-XGB: Heterogeneous Transfer Learning with XGBoost
* HTL-ET: Heterogeneous Transfer Learning with Extra Trees
* HTL-RF: Heterogeneous Transfer Learning with Random Forest

### Target-only models

Target-only models are trained only on SFRC target-domain data and are used as baseline models.

* XGB Target-Only
* ET Target-Only
* RF Target-Only

---

## Input data

Three CSV files are required:

| File              | Description                                                                 |
| ----------------- | --------------------------------------------------------------------------- |
| RC source dataset | Conventional RC beam test data used as the source domain                    |
| SFRC Group 1      | SFRC beam test data with basic fiber-index variables                        |
| SFRC Group 2      | SFRC beam test data with additional tensile and residual strength variables |

---

## Input variables

### RC source domain

| Variable    | Description                                                                                |
| ----------- | ------------------------------------------------------------------------------------------ |
| `b_mm`      | Beam width, mm                                                                             |
| `d_mm`      | Effective depth, mm                                                                        |
| `a_d`       | Shear span-to-depth ratio                                                                  |
| `fc_MPa`    | Concrete compressive strength, MPa                                                         |
| `rho`       | Longitudinal reinforcement ratio, %                                                        |
| `V_u_KN`    | Experimental shear strength, kN                                                            |
| `tau_u_MPa` | Shear stress, MPa. If not provided, it can be calculated from `V_u_KN`, `b_mm`, and `d_mm` |

### SFRC Group 1

| Variable    | Description                                                                                |
| ----------- | ------------------------------------------------------------------------------------------ |
| `b_mm`      | Beam width, mm                                                                             |
| `d_mm`      | Effective depth, mm                                                                        |
| `a_d`       | Shear span-to-depth ratio                                                                  |
| `fc_MPa`    | Concrete compressive strength, MPa                                                         |
| `rho`       | Longitudinal reinforcement ratio, %                                                        |
| `V_f_pct`   | Fiber volume fraction, %                                                                   |
| `RI`        | Reinforcing index                                                                          |
| `Lf_per_Df` | Fiber aspect ratio                                                                         |
| `V_u_KN`    | Experimental shear strength, kN                                                            |
| `tau_u_MPa` | Shear stress, MPa. If not provided, it can be calculated from `V_u_KN`, `b_mm`, and `d_mm` |

### SFRC Group 2

| Variable        | Description                                                                                |
| --------------- | ------------------------------------------------------------------------------------------ |
| `b_mm`          | Beam width, mm                                                                             |
| `d_mm`          | Effective depth, mm                                                                        |
| `a_d`           | Shear span-to-depth ratio                                                                  |
| `fc_MPa`        | Concrete compressive strength, MPa                                                         |
| `rho`           | Longitudinal reinforcement ratio, %                                                        |
| `V_f_pct`       | Fiber volume fraction, %                                                                   |
| `RI`            | Reinforcing index                                                                          |
| `Lf_per_Df`     | Fiber aspect ratio                                                                         |
| `fsp_MPa`       | Splitting tensile strength, MPa                                                            |
| `ft_direct_MPa` | Direct tensile strength, MPa                                                               |
| `fr_MPa`        | Modulus of rupture, MPa                                                                    |
| `V_u_KN`        | Experimental shear strength, kN                                                            |
| `tau_u_MPa`     | Shear stress, MPa. If not provided, it can be calculated from `V_u_KN`, `b_mm`, and `d_mm` |

---

## Prediction mode

The **Predict** tab allows the user to load saved models and predict:

* `V_u` in kN
* `τ_u` in MPa

Saved models are loaded from the model output folder generated after training.

For prediction input:

* `a/d` must be greater than or equal to 2.0.
* Group 1 uses RC common variables and fiber-index variables.
* Group 2 additionally uses `f_sp`, `f_t,dir`, and `f_r` as input features.
* Group 2 tensile and residual strength values must be entered as numeric values for prediction.

---

## Performance metrics

The program evaluates model performance using the following metrics:

| Metric     | Description                                               |
| ---------- | --------------------------------------------------------- |
| R²         | Coefficient of determination                              |
| RMSE       | Root mean squared error                                   |
| MAE        | Mean absolute error                                       |
| A/P mean   | Mean actual-to-predicted ratio                            |
| A/P CoV    | Coefficient of variation of the actual-to-predicted ratio |
| Pearson r  | Pearson correlation coefficient                           |
| Spearman r | Spearman rank correlation coefficient                     |

---

## Exported results

After each run, the program automatically exports result tables and figures. Typical outputs include:

* Cross-validation metric summaries
* Prediction-versus-observation plots
* Model comparison figures
* Literature-equation comparison results
* SHAP analysis outputs, if SHAP is installed
* Saved prediction models for later use in the Predict tab

---

## Repository structure

```text
sfrc-shear-ml-gui/
├── README.md
└── Model_0608_2026_GUI_auto_patched_v9.py
```

---

## Notes

* The GUI was designed for RC-to-SFRC heterogeneous transfer learning research.
* Default paths in the Python file may need to be changed depending on the user’s local directory structure.
* For Group 2 prediction, tensile and residual strength parameters should be provided as numerical input values.
* The GUI layout is optimized for readable screenshots suitable for academic papers.

---

## Author

Jerry03120

---

## License

For academic and research use only.
