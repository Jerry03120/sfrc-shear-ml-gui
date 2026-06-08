# SFRC Shear TL GUI

A machine learning GUI for predicting the shear strength of RC and SFRC beams using heterogeneous transfer learning (HTL). The program compares HTL models — which leverage RC source data — against target-only baselines trained exclusively on SFRC data.

---

## Getting started

Install dependencies, then run:

```bash
pip install numpy pandas matplotlib scipy scikit-learn xgboost
python Model_0607_2026_GUI.py
```

Once the GUI opens, select the three CSV input files through the interface. Default file paths in the source code are from the original research environment, so on a different machine you'll need to select them manually.

---

## Models

Six algorithms are evaluated across two training strategies:

**HTL (Heterogeneous Transfer Learning)** — leverages RC source data for SFRC prediction
- HTL-XGB (Extreme Gradient Boosting)
- HTL-ET (Extra Trees)
- HTL-RF (Random Forest)

**Target-Only** — trained on SFRC data alone, used as baselines
- XGB Target-Only
- ET Target-Only
- RF Target-Only

---

## Input data

Three CSV files are required:

| File | Description |
|------|-------------|
| RC source dataset | Conventional RC beam experiments (source domain) |
| SFRC Group 1 | SFRC beam experiments — target domain, subset 1 |
| SFRC Group 2 | SFRC beam experiments — target domain, subset 2 |

**Input variables:** `b` (beam width), `d` (effective depth), `a/d` (shear span ratio), `f'c` (compressive strength), `ρ` (reinforcement ratio), `Vf` (fiber volume fraction), `RI` (reinforcing index), `Lf/Df` (fiber aspect ratio), residual strength parameters (where available), `Vu` (experimental shear strength)

---

## Performance metrics

R², RMSE, MAE, mean A/P ratio, CV of A/P, Pearson correlation coefficient

---

## Repository structure

```
sfrc-shear-ml-gui/
├── README.md
└── Model_0607_2026_GUI.py
```

---

## A note on data

Results are exported automatically as a table after each run.

Please do not upload unpublished experimental data, private datasets, or confidential research files to a public repository unless you intend for them to be publicly accessible.

---

## Author

Jerry03120

## License

For academic and research use only.
