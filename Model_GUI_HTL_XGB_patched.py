"""
================================================================================
Model_GUI_HTL_XGB_patched.py
RC → SFRC Heterogeneous Transfer Learning — Extended models

================================================================================
"""

# ============================================================
# 0.  Imports
# ============================================================
import os, sys, math, warnings, time, threading, traceback, platform
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.impute import SimpleImputer
from sklearn.model_selection import RepeatedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (r2_score, mean_absolute_error,
                              mean_squared_error, median_absolute_error)
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, BaggingRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel, ConstantKernel as CK
from sklearn.neural_network import MLPRegressor
try:
    from xgboost import XGBRegressor as _XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
try:
    from lightgbm import LGBMRegressor as _LGBMRegressor
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False
try:
    from catboost import CatBoostRegressor as _CatBoostRegressor
    HAS_CAT = True
except ImportError:
    HAS_CAT = False

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

try:
    from statsmodels.nonparametric.smoothers_lowess import lowess as sm_lowess
    HAS_LOWESS = True
except ImportError:
    HAS_LOWESS = False

warnings.filterwarnings("ignore")

# ============================================================
# 1. Default paths
# ============================================================
IS_WIN = platform.system() == "Windows"
SCRIPT_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()

_DEFAULT_RC_FILE = r"Z:\KNUT\★★★★Journal★★★★\2029.04 - CCC_SFRC Shear ML\1. Machine Learning Code\01_01_DB\02_01_RC_DB\RC_wo_shear_DB_0528_2026.csv"
_DEFAULT_G1_FILE = r"Z:\KNUT\★★★★Journal★★★★\2029.04 - CCC_SFRC Shear ML\1. Machine Learning Code\03_01_G1_2_DB\Group1.csv"
_DEFAULT_G2_FILE = r"Z:\KNUT\★★★★Journal★★★★\2029.04 - CCC_SFRC Shear ML\1. Machine Learning Code\03_01_G1_2_DB\Group2.csv"

# ============================================================
# 2. Global hyperparameters
# ============================================================
RANDOM_STATE          = 42
N_SPLITS              = 5
N_REPEATS             = 5
N_JOBS                = -1
MIN_FINE_TUNE_SAMPLES = 12
AVAILABILITY_RATIOS   = [0.10, 0.20, 0.30, 0.40, 0.50,
                          0.60, 0.70, 0.80, 0.90]
USE_KWAK              = True

BASE_COMMON  = ['b_mm', 'd_mm', 'a_d', 'fc_MPa', 'rho']
GROUP1_EXTRA = ['V_f_pct', 'RI', 'Lf_per_Df']
GROUP2_EXTRA = ['V_f_pct', 'RI', 'Lf_per_Df', 'fsp_MPa', 'ft_direct_MPa', 'fr_MPa']
FEATURE_GROUPS = {
    "RC_common":    ['b_mm', 'd_mm', 'a_d', 'fc_MPa', 'rho'],
    "Fiber_basic":  ['V_f_pct', 'RI', 'Lf_per_Df'],
    "Residual_str": ['fsp_MPa', 'ft_direct_MPa', 'fr_MPa'],
    "Source_pred":  ['source_pred'],
}

FEATURE_LABELS = {
    # Source / meta
    'source_pred':       "Source model prediction",
    # Geometry
    'b_mm':              r"$b$ (mm)",
    'h_mm':              r"$h$ (mm)",
    'd_mm':              r"$d$ (mm)",
    # Shear span ratio
    'a_d':               r"$a/d$",
    'a_per_d':           r"$a/d$",
    # Concrete compressive strength  ← IMPROVED: show f'c everywhere
    'fc_MPa':            r"$f'_c$ (MPa)",
    'fc_Mpa':            r"$f'_c$ (MPa)",
    'f_c_MPa':           r"$f'_c$ (MPa)",
    'fck_MPa':           r"$f'_c$ (MPa)",
    'fc__Mpa':           r"$f'_c$ (MPa)",   # after apostrophe strip in safe-rename
    # Reinforcement ratio
    'rho':               r"$\rho$ (%)",
    'rho_pct':           r"$\rho$ (%)",
    # Fiber parameters
    'V_f_pct':           r"$V_f$ (%)",
    'RI':                r"$RI$",
    'RI_factor':         r"$RI$",
    'Lf_per_Df':         r"$L_f/D_f$",
    'L_f_per_D_f':       r"$L_f/D_f$",
    'L_f_per_D':         r"$L_f/D_f$",
    # Tensile / residual strengths
    'fsp_MPa':           r"$f_{sp}$ (MPa)" + "\n(est.)",
    'ft_direct_MPa':     r"$f_{t,\mathrm{dir}}$ (MPa)" + "\n(est.)",
    'fr_MPa':            r"$f_r$ (MPa)" + "\n(est.)",
    # Derived / interaction features
    # Targets
    'V_u_KN':            r"$V_u$ (kN)",
    'Vu_KN':             r"$V_u$ (kN)",
    'Vu_kN':             r"$V_u$ (kN)",
    'V_u_kN':            r"$V_u$ (kN)",
    'V_u_MPa':           r"$V_u$ (kN)",
    'tau_u_MPa':         r"$\tau_u$ (MPa)",
    'v_u_MPa':           r"$v_u$ (MPa)",
    # Imputed column aliases (Group1/Group2 CSV)
    'ft_direct_Mpa_imputed':  r"$f_{t,\mathrm{dir}}$ (MPa)" + "\n(imputed)",
    'fsp_Mpa_imputed':        r"$f_{sp}$ (MPa)" + "\n(imputed)",
    'fr_Mpa_imputed':         r"$f_r$ (MPa)" + "\n(imputed)",
}

TARGET_LABELS = {
    'V_u_KN': r'$V_{u}$ (kN)',
    'V_u_MPa': r'$V_{u}$ (kN)',
    'tau_u_MPa': r'$\tau_{u} = V_{u}/(bd)$ (MPa)',
}

def target_unit(target_name):
    tn = str(target_name); low = tn.lower()
    if low.startswith('v_u') or low.startswith('vu'): return 'kN'
    if tn in ('tau_u_MPa', 'v_u_MPa') or 'tau' in low or low.endswith('mpa'): return 'MPa'
    return 'kN'

def metric_ylabel(metric, target_name):
    unit = target_unit(target_name)
    labels = {
        'R2': r'$R^2$', 'RMSE': f'RMSE ({unit})', 'MAE': f'MAE ({unit})',
        'MedAE': f'MedAE ({unit})', 'MAPE_pct': 'MAPE (%)',
        'nRMSE_mean': 'nRMSE', 'Bias_ratio': 'Bias ratio',
        'AP_mean': 'Mean(A/P)', 'AP_CoV': 'CoV(A/P)',
        'Pearson_r': 'Pearson r', 'Spearman_r': 'Spearman r',
    }
    return labels.get(str(metric), str(metric))

def workflow_metric_list(target_name, include_full=False):
    metrics = ['R2', 'RMSE', 'MAE', 'AP_mean', 'AP_CoV', 'Pearson_r']
    if include_full:
        metrics = ['R2','RMSE','MAE','MedAE','MAPE_pct','nRMSE_mean',
                   'Bias_ratio','AP_mean','AP_CoV','Pearson_r','Spearman_r']
    return [(m, metric_ylabel(m, target_name)) for m in metrics]

def pretty_label(name):
    return FEATURE_LABELS.get(str(name), str(name))

def pretty_labels(names):
    return [pretty_label(n) for n in names]

FS_TAGS = {
    'V_u_KN': 'Vu', 'tau_u_MPa': 'tau',
    'Hetero_TL_ET': 'HTL-ET', 'TargetOnly_ET': 'TO-ET',
    'DirectSourceOnly_ET': 'DSET', 'ANN_TargetOnly': 'ANN',
    'ELNN_TargetOnly': 'ELNN', 'TLNN_Hetero': 'TLNN', 'TENN_Hetero': 'TENN',
    'Hetero_TL_XGB':     'HTL-XGB',    'TargetOnly_XGB':     'TO-XGB',
    'Hetero_TL_XGB_MAE': 'HTL-XGB-MAE','TargetOnly_XGB_MAE': 'TO-XGB-MAE',
    'Hetero_TL_LGBM':    'HTL-LGBM',   'TargetOnly_LGBM':    'TO-LGBM',
    'Hetero_TL_CAT':     'HTL-CAT',    'TargetOnly_CAT':     'TO-CAT',
    'Hetero_TL_RF':      'HTL-RF',     'TargetOnly_RF':      'TO-RF',
    'Hetero_TL_GBT':     'HTL-GBT',    'TargetOnly_GBT':     'TO-GBT',
    'Hetero_TL_GPR':     'HTL-GPR',    'TargetOnly_GPR':     'TO-GPR',
    'ACI318_25': 'ACI318', 'Eurocode2': 'EC2', 'JSCE': 'JSCE',
    'ACI544_4R_18': 'ACI544', 'Kuntia_1999': 'Kuntia',
    'Sharma_1986': 'Sharma', 'Ashour_1992': 'Ashour', 'Kwak_2002': 'Kwak',
    'a_d': 'ad', 'a_per_d': 'ad', 'd_mm': 'd', 'b_mm': 'b', 'h_mm': 'h',
    'rho': 'rho', 'rho_pct': 'rho', 'fc_MPa': 'fc', 'fc_Mpa': 'fc',
    'V_f_pct': 'Vf', 'Lf_per_Df': 'LfDf', 'L_f_per_D_f': 'LfDf',
    'L_f_per_D': 'LfDf',
    'RI': 'RI', 'RI_factor': 'RI', 'fsp_MPa': 'fsp',
    'ft_direct_MPa': 'ftdir', 'fr_MPa': 'fr',
}

def fs_tag(name, max_len=36):
    txt = FS_TAGS.get(str(name), str(name))
    txt = ''.join(ch if ch.isalnum() else '_' for ch in txt)
    txt = '_'.join(part for part in txt.split('_') if part)
    return (txt[:max_len] or 'x')

def ratio_tag(ratio):
    return f"r{int(float(ratio) * 100):02d}" if ratio is not None else 'all'


# Optional labels for modified ACI544/MC2010 residual tensile-strength columns.
FEATURE_LABELS.update({
    'fut_FRC_MPa': r'$f_{ut,\mathrm{FRC}}$ (MPa)',
    'f_Ftu_FRC_MPa': r'$f_{Ftu,\mathrm{FRC}}$ (MPa)',
    'fut_FRC': r'$f_{ut,\mathrm{FRC}}$ (MPa)',
    'f150D_MPa': r'$f_{150}^{D}$ (MPa)',
    'f_150D_MPa': r'$f_{150}^{D}$ (MPa)',
    'fD150_MPa': r'$f_{150}^{D}$ (MPa)',
})

FS_TAGS.update({
    'fut_FRC_MPa': 'futFRC',
    'f_Ftu_FRC_MPa': 'fFtuFRC',
    'fut_FRC': 'futFRC',
    'f150D_MPa': 'f150D',
    'f_150D_MPa': 'f150D',
    'fD150_MPa': 'f150D',
})


# ============================================================
# Models to calculate
# ============================================================
# Only the six algorithms used in the manuscript figure are evaluated:
#   (a) HTL-XGB (Proposed), (b) HTL-ET, (c) HTL-RF,
#   (d) XGB (TO),          (e) ET (TO),  (f) RF (TO).
# Heavy/unused models such as GBT, XGB_MAE, LGBM, CatBoost, GPR,
# ANN/ELNN/TLNN/TENN are intentionally excluded from calculation.
ML_MODELS_WITH_RC = [
    'Hetero_TL_XGB', 'Hetero_TL_ET', 'Hetero_TL_RF',
    'TargetOnly_XGB', 'TargetOnly_ET', 'TargetOnly_RF',
]
ML_MODELS_NO_RC = [
    'TargetOnly_XGB', 'TargetOnly_ET', 'TargetOnly_RF',
]

# ============================================================
# Manuscript/result-display model policy
# ============================================================
# The proposed model in this manuscript version is fixed as Hetero_TL_XGB.
# Other models can still be trained and stored in raw fold/prediction CSVs,
# but manuscript/workflow exports are filtered to the models needed for the
# figure/table comparison shown in the paper draft.
PROPOSED_MODEL = 'Hetero_TL_XGB'

RESULT_DISPLAY_MODELS_WITH_RC = [
    # Same order as the 2×3 prediction-observation figure:
    # (a) HTL-XGB, (b) HTL-ET, (c) HTL-RF,
    # (d) XGB (TO), (e) ET (TO), (f) RF (TO).
    PROPOSED_MODEL, 'Hetero_TL_ET', 'Hetero_TL_RF',
    'TargetOnly_XGB', 'TargetOnly_ET', 'TargetOnly_RF',
]

RESULT_DISPLAY_MODELS_NO_RC = [
    'TargetOnly_XGB', 'TargetOnly_ET', 'TargetOnly_RF',
]

# Proposed-model-only comparison for Group 1 vs Group 2 panels.
G1G2_PROPOSED_MODELS = [PROPOSED_MODEL]


def result_display_models(use_rc=True, available=None):
    """Return only manuscript-display models, preserving the predefined order."""
    base = RESULT_DISPLAY_MODELS_WITH_RC if use_rc else RESULT_DISPLAY_MODELS_NO_RC
    if available is None:
        return list(base)
    avail = set(available)
    return [m for m in base if m in avail]
LIT_BASE = ['ACI318_25', 'Eurocode2', 'JSCE', 'ACI544_4R_18',
            'Kuntia_1999', 'Sharma_1986', 'Ashour_1992', 'Kwak_2002',
            'Arslan_2014', 'Imam_1997', 'Mansur_1986',
            'Greenough_Nehdi_2008', 'Saber_2022', 'Sarveghadi_2019']

CODE_PLUS_EMPIRICAL_MODELS = ['ACI544_4R_18', 'Kuntia_1999', 'Sharma_1986',
                              'Ashour_1992', 'Kwak_2002',
                              'Arslan_2014', 'Imam_1997', 'Mansur_1986',
                              'Greenough_Nehdi_2008', 'Saber_2022', 'Sarveghadi_2019']
GROUP1_EXISTING_LITERATURE  = CODE_PLUS_EMPIRICAL_MODELS.copy()
GROUP2_REFERENCE_LITERATURE = CODE_PLUS_EMPIRICAL_MODELS.copy()
PREDOBS_EMPIRICAL_EXCLUDE   = {'ACI318_25', 'Eurocode2', 'JSCE'}
IMPACT_REFERENCE_EQUATIONS  = CODE_PLUS_EMPIRICAL_MODELS.copy()

SHAP_MAX_DISPLAY = 15

# ============================================================
# ASCE plotting style
# ============================================================
ASCE_FIG_DPI     = 600
ASCE_FONT_FAMILY = "Times New Roman"

mpl.rcParams.update({
    "figure.dpi": 180, "savefig.dpi": ASCE_FIG_DPI,
    "savefig.transparent": False,
    "font.size": 11, "axes.labelsize": 14, "axes.titlesize": 15,
    "legend.fontsize": 11, "xtick.labelsize": 11, "ytick.labelsize": 11,
    "font.family": "serif",
    "font.serif": [ASCE_FONT_FAMILY, "Times", "Nimbus Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "axes.linewidth": 0.85, "axes.edgecolor": "black", "axes.grid": False,
    "xtick.major.width": 0.8, "ytick.major.width": 0.8,
    "xtick.minor.width": 0.6, "ytick.minor.width": 0.6,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.top": True, "ytick.right": True,
    "legend.frameon": False, "pdf.fonttype": 42, "ps.fonttype": 42,
})

METRIC_COLS = ["R2", "RMSE", "MAE", "MedAE", "MAPE_pct", "nRMSE_mean",
               "Bias_ratio", "AP_mean", "AP_std", "AP_CoV",
               "Pearson_r", "Spearman_r"]

# ============================================================
# 3. GUI color palette  —  Light theme
# ============================================================
C = {
    # Modern light theme for publication screenshots
    "bg":        "#F5F7FB",   # app background
    "panel":     "#FFFFFF",   # header / section panels
    "card":      "#FFFFFF",   # card backgrounds
    "border":    "#D8E2EF",   # card borders / separators
    # Accent
    "accent":    "#1D4ED8",   # deep blue — titles / active elements
    "accent2":   "#059669",   # green — Group 1 / success highlights
    "accent_lt": "#EEF5FF",   # soft blue fill
    # Status
    "warn":      "#D97706",
    "danger":    "#DC2626",
    "ok":        "#16A34A",
    "running":   "#D97706",
    # Text
    "text":      "#0B1220",
    "text_dim":  "#64748B",
    "text_inv":  "#FFFFFF",
    # Buttons
    "btn_bg":    "#2563EB",
    "btn_fg":    "#FFFFFF",
    "btn_hover": "#1E40AF",
    # Log panel
    "log_bg":    "#0F172A",
    "log_fg":    "#94A3B8",
    # Input fields
    "entry_bg":  "#F8FBFF",
    "entry_fg":  "#0B1220",
}

FONT_TITLE = ("Arial", 22, "bold")
FONT_LABEL = ("Arial", 18)
FONT_SMALL = ("Arial", 16)
FONT_MONO  = ("Consolas", 15)
FONT_BADGE = ("Arial", 16, "bold")


# ============================================================
# 4. Utility functions
# ============================================================
def beautify(ax):
    ax.spines['top'].set_visible(True); ax.spines['right'].set_visible(True)
    for sp in ax.spines.values():
        sp.set_linewidth(0.85); sp.set_color('black')
    ax.tick_params(axis='both', which='major', direction='in', top=True, right=True,
                   width=0.8, length=4.0, pad=3)
    ax.tick_params(axis='both', which='minor', direction='in', top=True, right=True,
                   width=0.6, length=2.3, pad=3)
    ax.minorticks_on(); ax.grid(False)
    return ax


def savefig_all(fig, path_no_ext):
    """Save PNG + PDF; local-temp → copy strategy for NAS stability."""
    import tempfile, shutil, uuid
    def _win_long_path(p):
        s = os.path.abspath(str(p))
        if os.name != "nt": return s
        if s.startswith("\\\\?\\"): return s
        if s.startswith("\\\\"): return "\\\\?\\UNC\\" + s.lstrip("\\")
        return "\\\\?\\" + s

    p = Path(path_no_ext)
    if len(p.name) > 70:
        p = p.with_name(p.name[:70])
    p.parent.mkdir(parents=True, exist_ok=True)

    local_dir = Path(tempfile.gettempdir()) / "sfrc_tl_fig_cache"
    local_dir.mkdir(parents=True, exist_ok=True)
    local_stem = local_dir / f"{p.name}_{uuid.uuid4().hex[:8]}"

    try:
        for ext in [".png", ".pdf"]:
            local_file = Path(str(local_stem) + ext)
            final_file = Path(str(p) + ext)
            fig.savefig(str(local_file), bbox_inches="tight")
            final_file.parent.mkdir(parents=True, exist_ok=True)
            src, dst = _win_long_path(local_file), _win_long_path(final_file)
            last_err = None
            for attempt in range(5):
                try: shutil.copyfile(src, dst); break
                except Exception as e: last_err = e; time.sleep(0.5*(attempt+1))
            else: raise last_err
            try: local_file.unlink(missing_ok=True)
            except: pass
    finally:
        plt.close(fig)


def safe_csv(df, path):
    """DataFrame → CSV with retry (for network drives)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    import tempfile, shutil, uuid
    def _win_long_path(p):
        s = os.path.abspath(str(p))
        if os.name != "nt": return s
        if s.startswith("\\\\?\\"): return s
        if s.startswith("\\\\"): return "\\\\?\\UNC\\" + s.lstrip("\\")
        return "\\\\?\\" + s

    local_dir = Path(tempfile.gettempdir()) / "sfrc_tl_csv_cache"
    local_dir.mkdir(parents=True, exist_ok=True)
    local_file = local_dir / f"{path.stem}_{uuid.uuid4().hex[:8]}.csv"
    df.to_csv(str(local_file), index=False)
    src, dst = _win_long_path(local_file), _win_long_path(path)
    last_err = None
    for attempt in range(5):
        try: shutil.copyfile(src, dst); break
        except Exception as e: last_err = e; time.sleep(0.5*(attempt+1))
    else: raise last_err
    try: local_file.unlink(missing_ok=True)
    except: pass


def _clean(yt, yp):
    yt = np.asarray(yt, float); yp = np.asarray(yp, float)
    m = np.isfinite(yt) & np.isfinite(yp)
    return yt[m], yp[m]

def rmse_f(yt, yp):   return float(np.sqrt(mean_squared_error(yt, yp)))
def mape_f(yt, yp):
    m = np.abs(yt) > 1e-12
    return float(np.mean(np.abs((yt[m]-yp[m])/yt[m]))*100) if m.sum() else np.nan
def nrmse_f(yt, yp):
    d = np.mean(np.abs(yt))
    return np.nan if d < 1e-12 else rmse_f(yt, yp)/d
def bias_f(yt, yp):
    d = np.mean(np.abs(yt))
    return np.nan if d < 1e-12 else float(np.mean(yp-yt)/d)
def ap_stats(yt, yp):
    m = np.isfinite(yt) & np.isfinite(yp) & (np.abs(yp) > 1e-12)
    if m.sum() == 0: return np.nan, np.nan, np.nan
    ap = yt[m]/yp[m]
    mn = float(np.mean(ap)); sd = float(np.std(ap, ddof=1)) if len(ap)>1 else 0.
    return mn, sd, (sd/mn if abs(mn)>1e-12 else np.nan)

def metrics_dict(yt_raw, yp_raw):
    yt, yp = _clean(yt_raw, yp_raw)
    if len(yt) < 2: return {k: np.nan for k in METRIC_COLS}
    ap_mn, ap_sd, ap_cv = ap_stats(yt, yp)
    try:   pr = float(stats.pearsonr(yt, yp)[0])
    except: pr = np.nan
    try:   sr = float(stats.spearmanr(yt, yp)[0])
    except: sr = np.nan
    return {"R2": float(r2_score(yt,yp)), "RMSE": rmse_f(yt,yp),
            "MAE": float(mean_absolute_error(yt,yp)),
            "MedAE": float(median_absolute_error(yt,yp)),
            "MAPE_pct": mape_f(yt,yp), "nRMSE_mean": nrmse_f(yt,yp),
            "Bias_ratio": bias_f(yt,yp), "AP_mean": ap_mn,
            "AP_std": ap_sd, "AP_CoV": ap_cv,
            "Pearson_r": pr, "Spearman_r": sr}

def summarize_metrics(df, group_cols):
    rows = []
    for keys, sub in df.groupby(group_cols):
        if not isinstance(keys, tuple): keys = (keys,)
        row = dict(zip(group_cols, keys))
        for m in METRIC_COLS:
            if m not in sub.columns: continue
            v = sub[m].dropna().astype(float).values
            row[f"{m}_mean"]   = np.nanmean(v)           if len(v) else np.nan
            row[f"{m}_std"]    = float(np.std(v, ddof=1)) if len(v)>1 else 0.
            row[f"{m}_median"] = np.nanmedian(v)          if len(v) else np.nan
            row[f"{m}_q25"]    = np.nanpercentile(v, 25) if len(v) else np.nan
            row[f"{m}_q75"]    = np.nanpercentile(v, 75) if len(v) else np.nan
        row["n_runs"] = len(sub); rows.append(row)
    return pd.DataFrame(rows)

def safe_div(x, y):
    x=np.asarray(x,float); y=np.asarray(y,float)
    out=np.full_like(x,np.nan,dtype=float); m=np.abs(y)>1e-12; out[m]=x[m]/y[m]
    return out

def _strip_column_names(df):
    df = df.copy(); df.columns = [str(c).strip() for c in df.columns]; return df

def _rename_first_existing(df, aliases):
    df = df.copy(); ren = {}
    for canonical, candidates in aliases.items():
        if canonical in df.columns: continue
        for c in candidates:
            if c in df.columns: ren[c] = canonical; break
    return df.rename(columns=ren)

def _num_series(s):
    if pd.api.types.is_numeric_dtype(s): return pd.to_numeric(s, errors='coerce')
    return pd.to_numeric(
        s.astype(str).str.replace(',','',regex=False).str.replace(' ','',regex=False),
        errors='coerce')

def add_tau(df):
    df = df.copy(); need = ['V_u_KN','b_mm','d_mm']
    if not all(c in df.columns for c in need): return df
    if 'tau_u_MPa' not in df.columns: df['tau_u_MPa'] = np.nan
    b = _num_series(df['b_mm']); d = _num_series(df['d_mm']); vu = _num_series(df['V_u_KN'])
    denom = b*d; valid = (denom>0) & np.isfinite(denom) & np.isfinite(vu)
    df.loc[valid, 'tau_u_MPa'] = vu[valid]*1000.0/denom[valid]; return df

def to_num(df, keep=()):
    out = df.copy()
    for c in out.columns:
        if c not in keep: out[c] = _num_series(out[c])
    return out

def _ensure_rho_percent(df):
    """
    Standardize rho to percent units (%).

    Some RC datasets store rho_pct as fraction (e.g., 0.0157 = 1.57%),
    while Group1/Group2 store rho_pct as percent (e.g., 4.87 = 4.87%).
    The modeling features and literature-equation comments in this script use
    rho as percent, so fraction-like values are automatically converted.
    """
    df = df.copy()
    if 'rho' not in df.columns:
        return df
    rho = _num_series(df['rho'])
    finite = rho[np.isfinite(rho)]
    if len(finite) and float(np.nanmedian(np.abs(finite))) < 0.2:
        rho = rho * 100.0
    df['rho'] = rho
    return df

def harmonize_rc(df):
    df = _strip_column_names(df)
    aliases = {
        'Specimen': ['ID','specimen','Specimen_ID'],
        'a_d':      ['a_per_d','a/d','a_d','a_over_d'],
        # RC CSV uses fc'_Mpa (with apostrophe) — must be listed explicitly.
        # _rename_first_existing checks df.columns after _strip_column_names,
        # so the apostrophe is preserved as-is in the raw column string.
        'fc_MPa':   ["fc'_Mpa", "fc'_MPa", "f'c_Mpa", "f'c_MPa",
                     'fc_MPa', 'fc_Mpa', 'f_c_MPa', 'fck_MPa',
                     'fc__Mpa', 'fc__MPa'],
        # RC files may use rho_pct even when values are stored as fraction.
        'rho':      ['rho_pct','rho','ρ','rho_percent'],
        'V_u_KN':   ['Vu_kN','V_u_KN','V_u_kN','Vu_KN','Vu'],
    }
    df = _rename_first_existing(df, aliases)
    if 'Specimen' not in df.columns: df['Specimen'] = np.arange(len(df)).astype(str)
    keep_nc = tuple(c for c in ['Specimen','Author'] if c in df.columns)
    df = to_num(df, keep=keep_nc)
    df = _ensure_rho_percent(df)
    df = add_tau(df)
    return df

def harmonize_sfrc(df):
    df = _strip_column_names(df)
    aliases = {
        'a_d':           ['a_per_d','a/d','a_d','a_over_d'],
        'fc_MPa':        ["fc'_Mpa", "fc'_MPa", "f'c_Mpa", "f'c_MPa",
                          'fc_Mpa', 'fc_MPa', 'f_c_MPa', 'fck_MPa',
                          'fc__Mpa', 'fc__MPa'],
        'rho':           ['rho_pct','rho','ρ','rho_percent'],
        'V_f_pct':       ['Vf_pct', 'V_f_pct', 'vf_pct', 'Vf_percent',
                          'V_f_percent', 'fiber_volume_pct', 'vf'],
        'Lf_per_Df':     ['L_f_per_D','L_f_per_D_f','Lf_per_Df','L_f/D_f','Lf/Df','lf_per_df'],
        'RI':            ['RI_factor','RI','Reinforcing_index','reinforcing_index'],
        'V_u_KN':        ['Vu_KN','Vu_kN','V_u_KN','V_u_kN','Vu'],
        'fsp_MPa':       ['f_sp_Mpa_imputed','f_sp_MPa_imputed','fsp_Mpa_imputed',
                          'fsp_MPa','f_sp_MPa','f_sp_Mpa','f_sp'],
        'ft_direct_MPa': ['f_t_direct_Mpa_imputed','f_t_direct_MPa_imputed',
                          'ft_direct_Mpa_imputed',
                          'ft_direct_MPa','f_t_direct_MPa','f_t_direct','ft_direct'],
        'fr_MPa':        ['f_r_Mpa_imputed','f_r_MPa_imputed','fr_Mpa_imputed',
                          'fr_MPa','f_r_MPa','f_r_Mpa','f_r_Mpa_in','f_r_MPa_in','f_r'],
    }
    df = _rename_first_existing(df, aliases)
    keep_nc = tuple(c for c in ['Specimen','Author','Journal'] if c in df.columns)
    df = to_num(df, keep=keep_nc)
    df = _ensure_rho_percent(df)
    df = add_tau(df)
    return df

def choose_ft(n, ratio):
    return min(max(int(np.ceil(n*ratio)), MIN_FINE_TUNE_SAMPLES), n)

def build_feature_spec(df_rc, df_tgt, target_name, extra_cands, drop_feats, must_keep):
    common=[c for c in BASE_COMMON if c in df_rc.columns and c in df_tgt.columns
            and c not in drop_feats and c!=target_name]
    extra=[c for c in extra_cands if c in df_tgt.columns and c not in drop_feats and c!=target_name]
    all_t=list(dict.fromkeys(common+extra))
    for c in must_keep:
        if c in df_tgt.columns and c not in all_t and c!=target_name: all_t.append(c)
    return common, extra, all_t


# ============================================================
# 5. Literature equations
# ============================================================
def _ff(row):
    if pd.notna(row.get('V_f_pct',np.nan)) and pd.notna(row.get('Lf_per_Df',np.nan)):
        return (row['V_f_pct']/100.)*row['Lf_per_Df']
    if pd.notna(row.get('RI',np.nan)): return row['RI']/100.
    return np.nan

def _pred_aci318(df):
    out=[]
    for _,r in df.iterrows():
        fc,rho,b,d=r.get('fc_MPa',np.nan),r.get('rho',np.nan),r.get('b_mm',np.nan),r.get('d_mm',np.nan)
        if any(pd.isna([fc,rho,b,d])): out.append(np.nan); continue
        ls=min(1.,np.sqrt(2./(1.+0.004*d)))
        out.append(0.66*ls*(rho**(1/3))*np.sqrt(fc)*b*d/1000.)
    return np.asarray(out,float)

def _pred_ec2(df):
    out=[]
    for _,r in df.iterrows():
        fc,rho,b,d=r.get('fc_MPa',np.nan),r.get('rho',np.nan),r.get('b_mm',np.nan),r.get('d_mm',np.nan)
        if any(pd.isna([fc,rho,b,d])): out.append(np.nan); continue
        k=min(2.,1.+np.sqrt(200./d))
        out.append(0.18*k*((100.*rho*fc)**(1/3))*b*d/1000.)
    return np.asarray(out,float)

def _pred_jsce(df):
    out=[]
    for _,r in df.iterrows():
        fc,rho,b,d=r.get('fc_MPa',np.nan),r.get('rho',np.nan),r.get('b_mm',np.nan),r.get('d_mm',np.nan)
        if any(pd.isna([fc,rho,b,d])): out.append(np.nan); continue
        bp=min(1.5,(100.*rho)**(1/3)); bd=min(1.5,(1000./d)**0.25)
        fv=min(0.72,0.20*(fc**(1/3)))
        out.append(bp*bd*fv*b*d/1000.)
    return np.asarray(out,float)

def _pred_aci544(df):
    """
    Modified ACI544_4R_18 / MC2010-style SFRC shear prediction.

    This version is Group1/Group2-aware.

    Group1 typically has only fiber index variables:
        V_f_pct, RI, Lf_per_Df
    Therefore, f_t and f_ut-FRC are estimated by fallback equations.

    Group2 contains tensile / residual strength information:
        fsp_MPa, ft_direct_MPa, fr_MPa
    Therefore, measured tensile or residual-strength-based values are
    prioritized over fiber-factor empirical estimates.

    Formula:
        V_FRC = 0.18*k_s*[100*rho*(1 + 7.5*f_ut_FRC/f_t)*f'_c]^(1/3)*b*d

    Units:
        fc_MPa, ft, f_ut_FRC : MPa
        b, d                 : mm
        rho                  : percent (%) in this script
        output               : kN

    Important:
        In this codebase, rho is standardized as percent (%), so rho_pct=1.5
        means 1.5%. For EC2/MC2010-type equations, use rho_frac=rho_pct/100.
    """
    out=[]

    for _,r in df.iterrows():
        # ------------------------------------------------------------
        # 1. Basic variables
        # ------------------------------------------------------------
        fc      = r.get('fc_MPa', np.nan)
        rho_pct = r.get('rho', np.nan)       # stored as percent (%) in this script
        b       = r.get('b_mm', np.nan)
        d       = r.get('d_mm', np.nan)

        if any(pd.isna([fc, rho_pct, b, d])):
            out.append(np.nan)
            continue
        if fc <= 0 or rho_pct <= 0 or b <= 0 or d <= 0:
            out.append(np.nan)
            continue

        # rho: percent (%) -> fraction
        rho_frac = rho_pct / 100.0

        # Size effect factor
        ks = min(2.0, 1.0 + np.sqrt(200.0 / d))

        # ------------------------------------------------------------
        # 2. Plain concrete tensile strength, f_t
        # ------------------------------------------------------------
        # Priority:
        #   1) direct tensile strength, if Group2 has it
        #   2) splitting tensile strength
        #   3) empirical estimate from compressive strength
        # ------------------------------------------------------------
        if pd.notna(r.get('ft_direct_MPa', np.nan)) and float(r.get('ft_direct_MPa')) > 0:
            ft = float(r['ft_direct_MPa'])
        elif pd.notna(r.get('fsp_MPa', np.nan)) and float(r.get('fsp_MPa')) > 0:
            ft = float(r['fsp_MPa'])
        else:
            ft = 0.33 * np.sqrt(fc)

        if not np.isfinite(ft) or ft <= 0:
            out.append(np.nan)
            continue

        # ------------------------------------------------------------
        # 3. Ultimate tensile residual strength of FRC, f_ut_FRC
        # ------------------------------------------------------------
        # Priority:
        #   1) Direct f_ut_FRC column
        #   2) Flexural strength fallback: f_ut-FRC ≈ 0.75*f_r
        #   3) Fiber-factor empirical fallback for Group1
        # ------------------------------------------------------------
        fut = np.nan

        if pd.notna(r.get('fut_FRC_MPa', np.nan)) and float(r.get('fut_FRC_MPa')) > 0:
            fut = float(r['fut_FRC_MPa'])
        elif pd.notna(r.get('f_Ftu_FRC_MPa', np.nan)) and float(r.get('f_Ftu_FRC_MPa')) > 0:
            fut = float(r['f_Ftu_FRC_MPa'])
        elif pd.notna(r.get('fut_FRC', np.nan)) and float(r.get('fut_FRC')) > 0:
            fut = float(r['fut_FRC'])

        # Flexural strength fallback
        elif pd.notna(r.get('fr_MPa', np.nan)) and float(r.get('fr_MPa')) > 0:
            fut = 0.75 * float(r['fr_MPa'])

        # Group1 fallback using fiber factor
        else:
            F = _ff(r)
            if pd.isna(F):
                out.append(np.nan)
                continue
            fut = max(0.1, 0.3*np.sqrt(fc) + 7.629*(100.0*F))

        if not np.isfinite(fut) or fut < 0:
            out.append(np.nan)
            continue

        # ------------------------------------------------------------
        # 4. MC2010-style SFRC shear resistance
        # ------------------------------------------------------------
        vc = (
            0.18
            * ks
            * (
                100.0
                * rho_frac
                * (1.0 + 7.5*fut/max(ft, 1e-6))
                * fc
            ) ** (1.0/3.0)
        )

        Vu = vc*b*d/1000.0  # N -> kN
        out.append(Vu)

    return np.asarray(out, float)

def _pred_kuntia(df):
    out=[]
    for _,r in df.iterrows():
        fc,b,d=r.get('fc_MPa',np.nan),r.get('b_mm',np.nan),r.get('d_mm',np.nan); F=_ff(r)
        if any(pd.isna([fc,b,d,F])): out.append(np.nan); continue
        out.append((0.167+0.25*F)*np.sqrt(fc)*b*d/1000.)
    return np.asarray(out,float)

def _pred_sharma(df):
    out=[]
    for _,r in df.iterrows():
        fc,b,d,ad=r.get('fc_MPa',np.nan),r.get('b_mm',np.nan),r.get('d_mm',np.nan),r.get('a_d',np.nan)
        if any(pd.isna([fc,b,d,ad])) or ad<=0: out.append(np.nan); continue
        out.append((2./3.)*0.8*np.sqrt(fc)*(1./ad)**0.25*b*d/1000.)
    return np.asarray(out,float)

def _pred_ashour(df):
    """
    Ashour et al. (1992). Lantsoght (2019) Table 1, Eq.(18)/(19).
    Eq.(18) a/d >= 2.5:  Vu = [2.11*fc^(1/3) + 7F]*(rho/ad)^(1/3)*bw*d
    Eq.(19) a/d < 2.5:   with arching-action modification + vb*(2.5-a/d)*bw*d
    rho stored as % → convert to fraction.
    """
    out=[]
    for _,r in df.iterrows():
        fc,rho_pct,b,d,ad=(r.get('fc_MPa',np.nan),r.get('rho',np.nan),r.get('b_mm',np.nan),
                           r.get('d_mm',np.nan),r.get('a_d',np.nan)); F=_ff(r)
        if any(pd.isna([fc,rho_pct,b,d,ad,F])) or ad<=0: out.append(np.nan); continue
        rho = rho_pct / 100.0   # % → fraction
        if ad >= 2.5:
            v = (2.11*(fc**(1./3.)) + 7.*F) * ((rho/ad)**(1./3.))
        else:
            vb = 0.41 * 4.15 * F
            v = (2.11*(fc**(1./3.)) + 7.*F) * ((rho/ad)**(1./3.)) * (2.5/ad) + vb*(2.5-ad)
        out.append(max(0., v)*b*d/1000.)
    return np.asarray(out, float)

def _pred_kwak(df):
    """
    Kwak et al. (2002). Lantsoght (2019) Table 1, Eq.(5)-(7).
    Vu = [3.7*e*fspfc^(2/3)*(rho*d/a)^(1/3) + 0.8*vb] * bw*d
    fspfc = fcu_f/(20 - sqrt(F)) + 0.7 + 1/sqrt(F)  [Eq.6, MPa]
    fcu_f ≈ fc (cylinder) / 0.85  (cube ≒ cylinder/0.85 per ACI convention)
    Actually Kwak uses fcu = cube strength; we approximate fcu = fc_cyl / 0.85.
    vb = 0.41 * 4.15 * F   [Eq.4]
    rho stored as % → fraction.
    e = 1 if a/d > 3.4, else 3.4/(a/d)  [Eq.7]
    """
    out=[]
    for _,r in df.iterrows():
        fc,rho_pct,b,d,ad=(r.get('fc_MPa',np.nan),r.get('rho',np.nan),r.get('b_mm',np.nan),
                           r.get('d_mm',np.nan),r.get('a_d',np.nan))
        vf,lfdf=r.get('V_f_pct',np.nan),r.get('Lf_per_Df',np.nan)
        if any(pd.isna([fc,rho_pct,b,d,ad,vf,lfdf])) or ad<=0 or lfdf<=0:
            out.append(np.nan); continue
        rho = rho_pct / 100.0   # % → fraction
        F   = (vf/100.)*lfdf    # fiber factor
        fcu = fc / 0.85          # approx cube strength
        denom = max(20. - np.sqrt(max(F, 0.0)), 1e-6)
        sqrtF = np.sqrt(max(F, 1e-12))
        fspfc = fcu/denom + 0.7 + 1./sqrtF   # Eq.(6)
        e     = 1. if ad > 3.4 else 3.4/ad   # Eq.(7)
        vb    = 0.41 * 4.15 * F               # Eq.(4)
        Vu = (3.7*e*(fspfc**(2./3.))*((rho/ad)**(1./3.)) + 0.8*vb) * b*d/1000.
        out.append(max(0., Vu))
    return np.asarray(out, float)

def _pred_arslan_2014(df):
    out=[]
    for _,r in df.iterrows():
        fc,rho,b,d,ad=r.get('fc_MPa',np.nan),r.get('rho',np.nan),r.get('b_mm',np.nan),r.get('d_mm',np.nan),r.get('a_d',np.nan)
        if any(pd.isna([fc,rho,b,d,ad])) or ad<=0 or fc<=0: out.append(np.nan); continue
        F=_ff(r); F = 0.0 if pd.isna(F) else F
        rho_f=rho/100.0; B_coef=600.0*rho_f/fc; disc=B_coef**2+4.0*B_coef
        cd=max(0.,min((-B_coef+np.sqrt(disc))/2.0, 1.0))
        vu=(0.2*fc**(2./3.)*cd+np.sqrt(max(0.,rho_f*(1.+4.*F)*fc)))*(3.0/ad)**(1./3.)
        out.append(max(0.,vu)*b*d/1000.)
    return np.asarray(out,float)

def _pred_imam_1997(df):
    """
    Imam et al. (1997). Lantsoght (2019) Table 1, Eq.(22)-(24).
    Vu = 0.6*psi*(omega^1/3)*[fc^0.44*(a/d) + 275*sqrt(omega/(a/d)^5)] * bw*d
    psi = (1+sqrt(5.08/da)) / sqrt(1+d/(25*da))
    omega = rho*(1+4F),  rho as fraction.
    da default = 10 mm (standard lab mix; Lantsoght uses 10 mm when not reported).
    """
    da_default = 10.0; out=[]
    for _,r in df.iterrows():
        fc,rho_pct,b,d,ad=r.get('fc_MPa',np.nan),r.get('rho',np.nan),r.get('b_mm',np.nan),r.get('d_mm',np.nan),r.get('a_d',np.nan)
        if any(pd.isna([fc,rho_pct,b,d,ad])) or ad<=0 or d<=0: out.append(np.nan); continue
        F=_ff(r); F=0.0 if pd.isna(F) else F
        da = da_default
        psi=(1.+np.sqrt(5.08/da))/np.sqrt(1.+d/(25.*da))
        rho_f=rho_pct/100.; omega=rho_f*(1.+4.*F)
        if omega<=0: out.append(np.nan); continue
        vu=0.6*psi*(omega**(1./3.))*(fc**0.44*ad+275.*np.sqrt(omega/max(ad**5,1e-12)))
        out.append(max(0.,vu)*b*d/1000.)
    return np.asarray(out,float)

def _pred_mansur_1986(df):
    tau=4.15; out=[]
    for _,r in df.iterrows():
        fc,b,d=r.get('fc_MPa',np.nan),r.get('b_mm',np.nan),r.get('d_mm',np.nan)
        if any(pd.isna([fc,b,d])): out.append(np.nan); continue
        Vc=0.167*np.sqrt(fc)*b*d/1000.; F=_ff(r)
        sigma_tu=0.41*tau*F if (not pd.isna(F) and F>0) else 0.
        out.append(max(0.,Vc+sigma_tu*b*d/1000.))
    return np.asarray(out,float)

def _pred_greenough_nehdi_2008(df):
    """
    Greenough & Nehdi (2008). Lantsoght (2019) Table 1, Eq.(8).
    Vu = [0.351*(1+sqrt(400/d))*fc^0.18*((1+F)*rho*(d/a))^0.4
          + 0.9*eta_o*tau*F] * bw*d
    IMPORTANT: rho is in % per Lantsoght footnote (differs from other equations).
    eta_o=0.41 (3D random), tau=4.15 MPa.
    """
    tau=4.15; eta_o=0.41; out=[]
    for _,r in df.iterrows():
        fc,rho_pct,b,d,ad=r.get('fc_MPa',np.nan),r.get('rho',np.nan),r.get('b_mm',np.nan),r.get('d_mm',np.nan),r.get('a_d',np.nan)
        if any(pd.isna([fc,rho_pct,b,d,ad])) or ad<=0 or d<=0: out.append(np.nan); continue
        F=_ff(r); F=0.0 if pd.isna(F) else F
        size_term = 1.0 + np.sqrt(400.0/d)
        rho_da = max(0.0, (1.0+F)*rho_pct*(1.0/ad))   # rho in % as per Lantsoght
        term1 = 0.351 * size_term * (fc**0.18) * (rho_da**0.4)
        term2 = 0.9 * eta_o * tau * F
        vu = term1 + term2
        out.append(max(0.,vu)*b*d/1000.)
    return np.asarray(out,float)

def _pred_saber_2022(df):
    out=[]
    for _,r in df.iterrows():
        fc,rho,b,d,ad=r.get('fc_MPa',np.nan),r.get('rho',np.nan),r.get('b_mm',np.nan),r.get('d_mm',np.nan),r.get('a_d',np.nan)
        if any(pd.isna([fc,rho,b,d,ad])) or ad<=0 or rho<=0 or fc<=0: out.append(np.nan); continue
        out.append(max(0.,4.2*(fc**0.133)*(rho**0.274)*((1./ad)**0.618))*b*d/1000.)
    return np.asarray(out,float)

def _pred_sarveghadi_2019(df):
    tau=4.15; out=[]
    for _,r in df.iterrows():
        fc,rho,b,d,ad=r.get('fc_MPa',np.nan),r.get('rho',np.nan),r.get('b_mm',np.nan),r.get('d_mm',np.nan),r.get('a_d',np.nan)
        if any(pd.isna([fc,rho,b,d,ad])) or ad<=0 or fc<=0: out.append(np.nan); continue
        ft=0.79*np.sqrt(fc); F=_ff(r)
        vb=0.41*tau*F if (not pd.isna(F) and F>0) else 1e-6
        rho_f=rho/100.
        try:
            inner=rho_f*ft*(rho_f+2.)*(ft*ad-3./vb)/ad
            vu=rho_f+rho_f/vb+(1./ad)*inner/(ad+ft)+vb
        except: out.append(np.nan); continue
        if not np.isfinite(vu) or vu<0: out.append(np.nan); continue
        out.append(vu*b*d/1000.)
    return np.asarray(out,float)

def get_lit_preds(df, use_kwak=True):
    out=pd.DataFrame(index=df.index)
    out['ACI318_25']           = _pred_aci318(df)
    out['Eurocode2']           = _pred_ec2(df)
    out['JSCE']                = _pred_jsce(df)
    out['ACI544_4R_18']        = _pred_aci544(df)
    out['Kuntia_1999']         = _pred_kuntia(df)
    out['Sharma_1986']         = _pred_sharma(df)
    out['Ashour_1992']         = _pred_ashour(df)
    if use_kwak: out['Kwak_2002'] = _pred_kwak(df)
    out['Arslan_2014']           = _pred_arslan_2014(df)
    out['Imam_1997']             = _pred_imam_1997(df)
    out['Mansur_1986']           = _pred_mansur_1986(df)
    out['Greenough_Nehdi_2008']  = _pred_greenough_nehdi_2008(df)
    out['Saber_2022']            = _pred_saber_2022(df)
    out['Sarveghadi_2019']       = _pred_sarveghadi_2019(df)
    denom=df['b_mm'].astype(float).values*df['d_mm'].astype(float).values
    for c in list(out.columns):
        out[c+'_tau'] = out[c].values*1000.0/denom
    return out


# ============================================================
# 6. Model builders
# ============================================================
class HeteroTL(BaseEstimator, RegressorMixin):
    def __init__(self, source_model=None, target_model=None):
        self.source_model=source_model; self.target_model=target_model
    def fit(self, Xs_c, ys, Xt_c, Xt_f, yt):
        self.src_=clone(self.source_model).fit(Xs_c,ys)
        Z=np.hstack([self.src_.predict(Xt_c).reshape(-1,1),Xt_f])
        self.tgt_=clone(self.target_model).fit(Z,yt); return self
    def predict(self, Xt_c, Xt_f):
        Z=np.hstack([self.src_.predict(Xt_c).reshape(-1,1),Xt_f])
        return self.tgt_.predict(Z)

def build_src_et():
    # Source model: ExtraTrees (baseline)
    # Ref: Wakjira (2022) SFRC — ET consistently outperforms RF
    return ExtraTreesRegressor(n_estimators=600,min_samples_split=2,
        min_samples_leaf=1,max_features='sqrt',random_state=RANDOM_STATE,n_jobs=N_JOBS)

def build_src_rf():
    # Source model: RandomForest — tuned for large-scale RC data (higher n_estimators)
    return RandomForestRegressor(n_estimators=600, max_features='sqrt',
        min_samples_split=3, min_samples_leaf=1,
        random_state=RANDOM_STATE, n_jobs=N_JOBS)

def build_src_xgb():
    # Source model: XGBoost MSE
    # Ref: Feng et al. (2021) RC deep beam shear — XGB R²=0.96 (top performer)
    #        Barkhordari et al. (2022) RC shear — XGB surpasses RF/ET
    if not HAS_XGB: return None
    return _XGBRegressor(
        n_estimators=800, learning_rate=0.03, max_depth=6,
        subsample=0.8, colsample_bytree=0.8,
        min_child_weight=2, reg_alpha=0.05, reg_lambda=1.0,
        objective='reg:squarederror',
        random_state=RANDOM_STATE, n_jobs=N_JOBS,
        tree_method='hist', verbosity=0,
    )

def build_src_xgb_mae():
    # Source model: XGBoost MAE — robust to outliers
    if not HAS_XGB: return None
    return _XGBRegressor(
        n_estimators=800, learning_rate=0.03, max_depth=6,
        subsample=0.8, colsample_bytree=0.8,
        min_child_weight=2, reg_alpha=0.05, reg_lambda=1.0,
        objective='reg:absoluteerror',
        random_state=RANDOM_STATE, n_jobs=N_JOBS,
        tree_method='hist', verbosity=0,
    )

def build_src_lgbm():
    # Source model: LightGBM MAE
    # Ref: Wakjira et al. (2022) SFRC — LGBM top performance
    #        Maabreh & Almasabha (2024) SFRC deep beam — LGBM R²=97.8%
    if not HAS_LGBM: return None
    return _LGBMRegressor(
        n_estimators=800, learning_rate=0.03, max_depth=6,
        num_leaves=63, subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.05, reg_lambda=1.0,
        objective='mae',
        random_state=RANDOM_STATE, n_jobs=N_JOBS, verbose=-1,
    )

def build_src_cat():
    # Source model: CatBoost MAE — tuned for large-scale RC data
    if not HAS_CAT: return None
    return _CatBoostRegressor(
        iterations=800, learning_rate=0.03, depth=6,
        l2_leaf_reg=3.0, loss_function='MAE', eval_metric='MAE',
        train_dir='', allow_writing_files=False,
        random_seed=RANDOM_STATE, thread_count=N_JOBS, verbose=False,
    )

def build_src_gbt():
    # Source model: GBT Huber — balanced MSE/MAE, outlier-robust
    return GradientBoostingRegressor(
        n_estimators=600, learning_rate=0.03, max_depth=5,
        subsample=0.8, min_samples_split=4, min_samples_leaf=2,
        loss='huber', alpha=0.9,
        random_state=RANDOM_STATE,
    )

def build_src_svr():
    # Source model: SVR RBF — for large-scale RC data (with StandardScaler)
    return Pipeline([
        ('sc', StandardScaler()),
        ('svr', SVR(kernel='rbf', C=100, gamma='scale', epsilon=0.1))
    ])

def build_src_gpr():
    # Source model: GPR — uncertainty quantification, Matérn 5/2
    # O(n³) issue for RC n>1000 → subsample max 500 before fitting
    # (sklearn GPR + subsampling strategy instead of SparseGP)
    kernel = CK(1.0) * Matern(length_scale=1.0, nu=2.5) + WhiteKernel(noise_level=1e-2)
    class _SubsampledGPR(BaseEstimator, RegressorMixin):
        """GPR approximation for large RC dataset: fit on random 500 subsamples."""
        def __init__(self, kernel, n_sub=500, random_state=42):
            self.kernel = kernel
            self.n_sub = n_sub
            self.random_state = random_state
        def fit(self, X, y):
            from sklearn.gaussian_process import GaussianProcessRegressor as _GPR
            rng = np.random.RandomState(self.random_state)
            n = len(y)
            if n > self.n_sub:
                idx = rng.choice(n, self.n_sub, replace=False)
                X, y = X[idx], y[idx]
            self.gpr_ = _GPR(kernel=clone(self.kernel), alpha=1e-3,
                             normalize_y=True, n_restarts_optimizer=3,
                             random_state=self.random_state).fit(X, y)
            return self
        def predict(self, X):
            return self.gpr_.predict(X)
    return Pipeline([
        ('sc', StandardScaler()),
        ('gpr', _SubsampledGPR(kernel=kernel, n_sub=500,
                               random_state=RANDOM_STATE)),
    ])

def build_tgt_et():
    return ExtraTreesRegressor(n_estimators=500,min_samples_split=4,
        min_samples_leaf=2,max_features='sqrt',random_state=RANDOM_STATE,n_jobs=N_JOBS)

def build_xgb():
    # loss: reg:squarederror (MSE) — default, stronger regularization for small data
    if not HAS_XGB: return None
    return _XGBRegressor(
        n_estimators=500, learning_rate=0.05, max_depth=6,
        subsample=0.8, colsample_bytree=0.8,
        min_child_weight=3, reg_alpha=0.1, reg_lambda=1.0,
        objective='reg:squarederror',
        random_state=RANDOM_STATE, n_jobs=N_JOBS,
        tree_method='hist', verbosity=0,
    )

def build_xgb_mae():
    # loss: reg:absoluteerror (MAE) — outlier-robust, uniform A/P ratio
    if not HAS_XGB: return None
    return _XGBRegressor(
        n_estimators=500, learning_rate=0.05, max_depth=6,
        subsample=0.8, colsample_bytree=0.8,
        min_child_weight=3, reg_alpha=0.1, reg_lambda=1.0,
        objective='reg:absoluteerror',
        random_state=RANDOM_STATE, n_jobs=N_JOBS,
        tree_method='hist', verbosity=0,
    )

def build_lgbm():
    # loss: MAE — more stable than MSE on small SFRC datasets
    # Ref: SFRC deep beam R²=97.8% (Maabreh & Almasabha, 2024)
    if not HAS_LGBM: return None
    return _LGBMRegressor(
        n_estimators=500, learning_rate=0.05, max_depth=6,
        num_leaves=31, subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0,
        objective='mae',                 # MAE loss — outlier-robust
        random_state=RANDOM_STATE, n_jobs=N_JOBS, verbose=-1,
    )

def build_cat():
    # loss: MAE — CatBoost strength on small tabular data (Megahed 2024, R²=0.947)
    # train_dir='' : prevent catboost_info directory creation (permission workaround)
    if not HAS_CAT: return None
    return _CatBoostRegressor(
        iterations=500, learning_rate=0.05, depth=6,
        l2_leaf_reg=3.0,
        loss_function='MAE',
        eval_metric='MAE',
        train_dir='',            # suppress catboost_info folder creation
        allow_writing_files=False,
        random_seed=RANDOM_STATE, thread_count=N_JOBS,
        verbose=False,
    )

def build_gpr():
    # GPR: uncertainty quantification on small datasets (Nguyen 2023, Wakjira 2021)
    # Matérn 5/2 + WhiteKernel — suited for nonlinear SFRC shear behavior
    # Note: CK = ConstantKernel (C avoided to prevent clash with GUI color dict)
    kernel = CK(1.0) * Matern(length_scale=1.0, nu=2.5) + WhiteKernel(noise_level=1e-2)
    return Pipeline([
        ('sc', StandardScaler()),
        ('gpr', GaussianProcessRegressor(
            kernel=kernel, alpha=1e-3,
            normalize_y=True, n_restarts_optimizer=5,
            random_state=RANDOM_STATE,
        ))
    ])

def build_svr():
    # SVR: epsilon-insensitive loss (Huber-like) — outlier-robust
    # Ref: RSM-SVR hybrid outperforms standalone SVR (Rahman, 2022)
    return Pipeline([
        ('sc', StandardScaler()),
        ('svr', SVR(kernel='rbf', C=100, gamma='scale', epsilon=0.1))
    ])

def build_ada():
    # AdaBoost + DT base: MAE-based loss — minimizes relative error
    from sklearn.tree import DecisionTreeRegressor
    return AdaBoostRegressor(
        estimator=DecisionTreeRegressor(max_depth=5),
        n_estimators=200, learning_rate=0.1,
        loss='square',          # 'square' or 'linear' (MAE-like)
        random_state=RANDOM_STATE,
    )

def build_rf():
    return RandomForestRegressor(
        n_estimators=500, max_features='sqrt',
        min_samples_split=4, min_samples_leaf=2,
        random_state=RANDOM_STATE, n_jobs=N_JOBS,
    )

def build_gbt():
    # Huber loss — balanced MSE/MAE, robust to SFRC outliers
    return GradientBoostingRegressor(
        n_estimators=400, learning_rate=0.05, max_depth=5,
        subsample=0.8, min_samples_split=4, min_samples_leaf=2,
        loss='huber', alpha=0.9,         # Huber loss
        random_state=RANDOM_STATE,
    )

def build_ann(n_samples=None):
    if n_samples is not None and n_samples < 200:
        mlp=MLPRegressor(hidden_layer_sizes=(64,32,16),activation='relu',solver='lbfgs',
                         alpha=1e-3,max_iter=5000,random_state=RANDOM_STATE)
    else:
        mlp=MLPRegressor(hidden_layer_sizes=(64,32,16),activation='relu',solver='adam',
                         alpha=1e-3,learning_rate_init=3e-4,max_iter=5000,
                         early_stopping=True,validation_fraction=0.15,n_iter_no_change=50,
                         random_state=RANDOM_STATE)
    return Pipeline([('sc',StandardScaler()),('mlp',mlp)])

def build_elnn(n_samples=None):
    base=build_ann(n_samples)
    try: return BaggingRegressor(estimator=base,n_estimators=10,max_samples=0.8,
                                 bootstrap=True,random_state=RANDOM_STATE,n_jobs=N_JOBS)
    except: return BaggingRegressor(base_estimator=base,n_estimators=10,max_samples=0.8,
                                    bootstrap=True,random_state=RANDOM_STATE,n_jobs=N_JOBS)


# ============================================================
# 7. SHAP plots
# ============================================================
def run_shap(model, X, feat_names, out_dir, tag,
             max_display=SHAP_MAX_DISPLAY, log_fn=None):
    """
    Robust SHAP calculation + saving.

    Key behavior:
      1) Always write TXT log on shap missing / compute fail / plot fail
      2) Drop NaN/Inf rows before SHAP computation
      3) CSV and plot saving are independent — one plot failure does not abort others
      4) Manual beeswarm fallback when shap.summary_plot has compatibility issues
    """
    import traceback as _tb_mod
    import matplotlib.colors as mcolors

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def _log(msg):
        if log_fn is not None:
            log_fn(msg)

    def _write_status(filename, text):
        try:
            (out_dir / filename).write_text(str(text), encoding='utf-8')
        except Exception:
            pass

    def _savefig_safe(fig, stem, label):
        try:
            savefig_all(fig, out_dir / stem)
            return True
        except Exception as e:
            _write_status(f"{stem}_SAVE_ERROR.txt", _tb_mod.format_exc())
            _log(f"      [SHAP PLOT SAVE FAIL] {label}: {e}")
            try:
                plt.close(fig)
            except Exception:
                pass
            return False

    if not HAS_SHAP:
        msg = ("shap package is not installed; SHAP analysis was skipped.\n"
               "Fix: run `pip install shap` in the current Python environment.\n")
        _write_status(f"SHAP_NOT_RUN_{tag}.txt", msg)
        _log("    [SHAP SKIP] shap not installed -> pip install shap")
        return

    # ── Input validation / cleanup ─────────────────────────────────
    try:
        X = np.asarray(X, dtype=float)
    except Exception as e:
        _write_status(f"SHAP_ERROR_{tag}.txt", _tb_mod.format_exc())
        _log(f"    [SHAP SKIP] Failed to convert X to float array: {e}")
        return

    if X.ndim != 2:
        msg = f"X.ndim={X.ndim}; SHAP input X must be 2-dimensional."
        _write_status(f"SHAP_ERROR_{tag}.txt", msg)
        _log(f"    [SHAP SKIP] {msg}")
        return

    feat_names = list(feat_names)
    if len(feat_names) != X.shape[1]:
        msg = f"feat_names length ({len(feat_names)}) != X columns ({X.shape[1]})"
        _write_status(f"SHAP_ERROR_{tag}.txt", msg)
        _log(f"    [SHAP SKIP] {msg}")
        return

    finite_rows = np.isfinite(X).all(axis=1)
    if finite_rows.sum() != X.shape[0]:
        pass  # dropped NaN rows
        X = X[finite_rows]

    if X.shape[0] < 3:
        msg = f"Valid samples={X.shape[0]} (minimum 3 required). SHAP aborted after NaN/Inf removal."
        _write_status(f"SHAP_ERROR_{tag}.txt", msg)
        _log(f"    [SHAP SKIP] {msg}")
        return

    n_feats     = X.shape[1]
    max_display = min(int(max_display), n_feats)

    # ── Pipeline unwrapping for SHAP ─────────────────────────────────
    # SVR / GPR are wrapped in Pipeline(StandardScaler + model).
    # TreeExplainer / shap.Explainer cannot handle Pipeline directly.
    # Solution: extract the final estimator and pre-transform X through
    # all preceding steps, so SHAP sees the estimator's native input space.
    from sklearn.pipeline import Pipeline as _SKPipeline
    shap_model = model
    X_shap     = X.copy()
    if isinstance(model, _SKPipeline):
        # Apply all steps except the last (transform steps)
        for step_name, step_obj in model.steps[:-1]:
            try:
                X_shap = step_obj.transform(X_shap)
            except Exception as _pe:
                _log(f"      [SHAP WARN] Pipeline pre-transform '{step_name}' failed: {_pe}")
                X_shap = X.copy()
                break
        shap_model = model.steps[-1][1]  # final estimator
        pass  # Pipeline unwrapped
    else:
        X_shap = X

    # ── SHAP value computation ────────────────────────────────────
    sv = None
    explainer_errors = []

    # 1) TreeExplainer — most stable for sklearn tree ensembles
    try:
        exp = shap.TreeExplainer(shap_model)
        try:
            sv = exp.shap_values(X_shap, check_additivity=False)
        except TypeError:
            sv = exp.shap_values(X_shap)
        pass  # TreeExplainer OK
    except Exception as e:
        explainer_errors.append("TreeExplainer:\n" + _tb_mod.format_exc())
        _log(f"      [SHAP] TreeExplainer failed: {e}")

    # 2) New API fallback
    if sv is None:
        try:
            exp = shap.Explainer(shap_model, X_shap)
            sv_obj = exp(X_shap)
            sv = sv_obj.values if hasattr(sv_obj, 'values') else np.asarray(sv_obj)
            pass  # shap.Explainer OK
        except Exception as e:
            explainer_errors.append("shap.Explainer:\n" + _tb_mod.format_exc())
            _log(f"      [SHAP] shap.Explainer failed: {e}")

    # 3) KernelExplainer fallback
    if sv is None:
        try:
            bg   = shap.sample(X_shap, min(50, X_shap.shape[0]),
                               random_state=RANDOM_STATE)
            kexp = shap.KernelExplainer(shap_model.predict, bg)
            X_ke = (X_shap if X_shap.shape[0] <= 100
                    else shap.sample(X_shap, 100, random_state=RANDOM_STATE))
            if X_ke.shape[0] != X_shap.shape[0]:
                pass  # KernelExplainer subsampled
            X_shap = np.asarray(X_ke, dtype=float)
            sv = kexp.shap_values(X_shap, nsamples=50)
            pass  # KernelExplainer OK
        except Exception as e:
            explainer_errors.append("KernelExplainer:\n" + _tb_mod.format_exc())
            _write_status(f"SHAP_ERROR_{tag}.txt", "\n\n".join(explainer_errors))
            _log(f"      [SHAP FAIL] All explainers failed -> SHAP_ERROR_{tag}.txt")
            return

    # ── SHAP array normalization ──────────────────────────────────
    if isinstance(sv, list):
        sv = sv[0]
    sv = np.asarray(sv, dtype=float)

    # For regression, some shap versions return (n,p,1) or (1,n,p)
    if sv.ndim == 3:
        if sv.shape[0] == X_shap.shape[0] and sv.shape[1] == n_feats:
            sv = sv[:, :, 0]
        elif sv.shape[1] == X_shap.shape[0] and sv.shape[2] == n_feats:
            sv = sv[0, :, :]
        else:
            sv = np.squeeze(sv)

    if sv.shape != (X_shap.shape[0], n_feats):
        msg = f"Shape mismatch: sv={sv.shape}, expected=({X_shap.shape[0]}, {n_feats})"
        _write_status(f"SHAP_ERROR_{tag}.txt", msg)
        _log(f"      [SHAP FAIL] {msg}")
        return

    _log(f"      SHAP done  sv={sv.shape}  NaN={np.isnan(sv).sum()}")

    display_feat_names = pretty_labels(feat_names)
    X_df    = pd.DataFrame(X_shap, columns=display_feat_names)
    shap_df = pd.DataFrame(sv,     columns=display_feat_names)

    # CSV는 그림 실패와 무관하게 먼저 저장
    try:
        # ① SHAP 값 (샘플 × 피처)
        safe_csv(shap_df, out_dir / f'shap_values_{tag}.csv')
        # ② 피처 원본 X 값 (beeswarm 색상 재현용)
        safe_csv(X_df,    out_dir / f'shap_X_{tag}.csv')
        # ③ mean(|SHAP|) 요약 — bar 그래프 재현용
        mean_abs_all = np.nanmean(np.abs(sv), axis=0)
        safe_csv(pd.DataFrame({
            'raw_feature':  feat_names,
            'plot_label':   display_feat_names,
            'mean_abs_shap': mean_abs_all,
            'mean_abs_norm': (mean_abs_all / mean_abs_all.max()
                              if mean_abs_all.max() > 1e-12
                              else mean_abs_all),
        }).sort_values('mean_abs_shap', ascending=False).reset_index(drop=True),
        out_dir / f'shap_mean_abs_{tag}.csv')
        # ④ 피처 레이블 매핑
        safe_csv(pd.DataFrame({'raw_feature': feat_names,
                               'plot_label': display_feat_names}),
                 out_dir / f'shap_feature_labels_{tag}.csv')
        pass  # SHAP CSV saved
    except Exception:
        _write_status(f"SHAP_CSV_SAVE_ERROR_{tag}.txt", _tb_mod.format_exc())
        _log(f"      [SHAP CSV SAVE FAIL] shap_values_{tag}.csv")

    # ══════════════════════════════════════════════════════════
    # [PLOT 1] mean(|SHAP|) bar
    # ══════════════════════════════════════════════════════════
    try:
        mean_abs = np.nanmean(np.abs(sv), axis=0)
        if not np.isfinite(mean_abs).any():
            raise ValueError("mean_abs SHAP values are all NaN/Inf.")

        order         = np.argsort(np.nan_to_num(mean_abs, nan=-np.inf))[::-1][:max_display]
        ordered_names = [display_feat_names[i] for i in order]
        ordered_raw   = mean_abs[order]
        v_min, v_max  = float(np.nanmin(ordered_raw)), float(np.nanmax(ordered_raw))

        # Top-journal friendly normalization for SHAP bar plots:
        # use max-normalization instead of min-max normalization.
        # This preserves nonzero importance for the least-important displayed
        # feature, whereas min-max normalization forces it to exactly zero.
        if v_max < 1e-12:
            ordered_norm = np.zeros_like(ordered_raw, dtype=float)
        else:
            ordered_norm = ordered_raw / v_max

        n_bars = len(ordered_names)
        _nat_colors = ["#D4E6F1", "#85C1E9", "#2E86C1", "#1A5276", "#0B2545"]
        _nat_cmap = mcolors.LinearSegmentedColormap.from_list("nature_bar", _nat_colors, N=256)
        bar_norm_rev  = ordered_norm[::-1]
        bar_colors    = [_nat_cmap(float(v)) for v in bar_norm_rev]
        bar_names_rev = ordered_names[::-1]

        def _draw_bar(figsize, tag_suffix):
            plt.close('all')
            # ── Top-tier journal style (Nature/CACAIE) ──────────────────
            with plt.rc_context({
                'font.family': 'serif',
                'font.serif': ['Times New Roman', 'DejaVu Serif', 'serif'],
                'axes.linewidth': 0.8,
                'xtick.major.width': 0.8,
                'ytick.major.width': 0.8,
            }):
                fig, ax = plt.subplots(figsize=figsize)
                bars = ax.barh(bar_names_rev, bar_norm_rev,
                               color=bar_colors, edgecolor='white',
                               linewidth=0.5, height=0.65)
                for bar, nv in zip(bars, bar_norm_rev):
                    ax.text(float(nv) + 0.013,
                            bar.get_y() + bar.get_height() / 2,
                            f'{float(nv):.2f}',
                            va='center', ha='left',
                            fontsize=11.0, color='#1A1A2E',
                            fontfamily='serif')
                ax.axvline(0, color='#444444', linewidth=0.7, zorder=0)
                ax.set_xlim(0.0, 1.18)
                ax.set_xlabel(r'Relative mean($|\mathrm{SHAP}|$), normalized to max = 1',
                              fontsize=13, labelpad=6, color='#1A1A2E')
                ax.set_ylabel('')
                # y-tick labels: allow multi-line (already encoded as \n)
                ax.tick_params(axis='y', labelsize=12.0, colors='#1A1A2E',
                               pad=5, length=0)
                ax.tick_params(axis='x', labelsize=11.0, colors='#444444',
                               direction='out', length=3, width=0.8)
                ax.set_facecolor('#FFFFFF')
                fig.patch.set_facecolor('#FFFFFF')
                for sp in ['top', 'right']:
                    ax.spines[sp].set_visible(False)
                for sp in ['bottom', 'left']:
                    ax.spines[sp].set_color('#444444')
                    ax.spines[sp].set_linewidth(0.8)
                ax.xaxis.grid(True, color='#E0E0E0', linewidth=0.5,
                              linestyle=':', zorder=0)
                ax.set_axisbelow(True)
                fig.tight_layout(pad=1.2)
                _savefig_safe(fig, f'SHAP_Bar_{tag}{tag_suffix}',
                              f'SHAP_Bar_{tag}{tag_suffix}')

        _draw_bar(figsize=(7.2, max(3.8, 0.40 * n_bars + 1.0)), tag_suffix='_wide')
        _sq = max(4.5, 0.40 * n_bars + 1.0)
        _draw_bar(figsize=(_sq, _sq), tag_suffix='_square')
    except Exception:
        _write_status(f"SHAP_BAR_ERROR_{tag}.txt", _tb_mod.format_exc())
        _log(f"      [SHAP BAR FAIL] SHAP_BAR_ERROR_{tag}.txt")

    # ══════════════════════════════════════════════════════════
    # [PLOT 2] beeswarm
    # ══════════════════════════════════════════════════════════
    _bee_colors = ["#053061", "#2166AC", "#92C5DE", "#F7F7F7", "#F4A582", "#D6604D", "#67001F"]
    _bee_cmap = mcolors.LinearSegmentedColormap.from_list("nature_bee", _bee_colors, N=256)

    def _manual_beeswarm(figsize, tag_suffix):
        mean_abs = np.nanmean(np.abs(sv), axis=0)
        order = np.argsort(np.nan_to_num(mean_abs, nan=-np.inf))[::-1][:max_display]
        plt.close('all')
        with plt.rc_context({
            'font.family': 'serif',
            'font.serif': ['Times New Roman', 'DejaVu Serif', 'serif'],
            'axes.linewidth': 0.8,
        }):
            fig, ax = plt.subplots(figsize=figsize)
            rng = np.random.RandomState(RANDOM_STATE)
            for y_pos, j in enumerate(order[::-1]):
                vals  = sv[:, j]
                xvals = X_shap[:, j]
                if np.nanmax(xvals) - np.nanmin(xvals) > 1e-12:
                    cvals = (xvals - np.nanmin(xvals)) / (np.nanmax(xvals) - np.nanmin(xvals))
                else:
                    cvals = np.full_like(xvals, 0.5, dtype=float)
                jitter = rng.normal(0, 0.055, size=len(vals))
                ax.scatter(vals, np.full(len(vals), y_pos) + jitter,
                           c=cvals, cmap=_bee_cmap, s=14, alpha=0.78,
                           edgecolors='none', vmin=0, vmax=1, rasterized=True)
            ax.set_yticks(range(len(order)))
            ax.set_yticklabels([display_feat_names[i] for i in order[::-1]],
                               fontsize=12.0)
            ax.axvline(0, color='#555555', linewidth=0.8, linestyle='--', zorder=1)
            ax.set_xlabel('SHAP value (impact on model output)',
                          fontsize=13, labelpad=6)
            ax.tick_params(axis='x', labelsize=11.0, direction='out',
                           length=3, width=0.8, colors='#444444')
            ax.tick_params(axis='y', labelsize=12.0, colors='#1A1A2E',
                           length=0, pad=5)
            ax.set_facecolor('#FFFFFF')
            fig.patch.set_facecolor('#FFFFFF')
            for sp in ['top', 'right']:
                ax.spines[sp].set_visible(False)
            for sp in ['bottom', 'left']:
                ax.spines[sp].set_color('#444444')
                ax.spines[sp].set_linewidth(0.8)
            ax.xaxis.grid(True, color='#E0E0E0', linewidth=0.5,
                          linestyle=':', zorder=0)
            ax.set_axisbelow(True)
            # ── Colorbar (journal style) ─────────────────────────────
            sm = plt.cm.ScalarMappable(cmap=_bee_cmap,
                                       norm=mcolors.Normalize(vmin=0, vmax=1))
            sm.set_array([])
            cbar = fig.colorbar(sm, ax=ax, orientation='vertical',
                                fraction=0.025, pad=0.015, aspect=28,
                                ticks=[0.0, 0.5, 1.0])
            cbar.set_label('Feature value\n(low → high)',
                           fontsize=11.0, labelpad=6, color='#1A1A2E')
            cbar.ax.tick_params(labelsize=10.0, colors='#444444', length=2)
            cbar.ax.set_yticklabels(['Low', 'Med', 'High'], fontsize=10.0)
            cbar.outline.set_linewidth(0.5)
            cbar.outline.set_edgecolor('#888888')
            fig.tight_layout(pad=1.2)
            _savefig_safe(fig, f'SHAP_Beeswarm_{tag}{tag_suffix}',
                          f'SHAP_Beeswarm_{tag}{tag_suffix} (manual)')

    def _draw_beeswarm(figsize, tag_suffix):
        try:
            plt.close('all')
            with plt.rc_context({
                'font.family': 'serif',
                'font.serif': ['Times New Roman', 'DejaVu Serif', 'serif'],
                'axes.linewidth': 0.8,
            }):
                fig_bee, ax_bee = plt.subplots(figsize=figsize)
                plt.sca(ax_bee)
                try:
                    shap.summary_plot(
                        sv, X_df,
                        feature_names=display_feat_names,
                        plot_type='dot', show=False,
                        max_display=max_display,
                        color_bar=False, plot_size=None,
                        cmap=_bee_cmap,
                    )
                except TypeError:
                    plt.close(fig_bee)
                    plt.close('all')
                    with plt.rc_context({
                        'font.family': 'serif',
                        'font.serif': ['Times New Roman', 'DejaVu Serif', 'serif'],
                        'axes.linewidth': 0.8,
                    }):
                        fig_bee, ax_bee = plt.subplots(figsize=figsize)
                        plt.sca(ax_bee)
                        shap.summary_plot(
                            sv, X_df,
                            feature_names=display_feat_names,
                            plot_type='dot', show=False,
                            max_display=max_display,
                            color_bar=False, plot_size=None,
                        )
                fig_out = plt.gcf()
                ax_out  = plt.gca()
                ax_out.set_title('')
                ax_out.set_xlabel('SHAP value (impact on model output)',
                                  fontsize=13, labelpad=6, color='#1A1A2E')
                ax_out.tick_params(axis='y', labelsize=12.0, colors='#1A1A2E',
                                   length=0, pad=5)
                ax_out.tick_params(axis='x', labelsize=11.0, colors='#444444',
                                   direction='out', length=3, width=0.8)
                ax_out.set_facecolor('#FFFFFF')
                fig_out.patch.set_facecolor('#FFFFFF')
                for sp in ['top', 'right']:
                    ax_out.spines[sp].set_visible(False)
                for sp in ['bottom', 'left']:
                    ax_out.spines[sp].set_color('#444444')
                    ax_out.spines[sp].set_linewidth(0.8)
                ax_out.xaxis.grid(True, color='#E0E0E0', linewidth=0.5,
                                  linestyle=':', zorder=0)
                ax_out.axvline(0, color='#555555', linewidth=0.8,
                               linestyle='--', zorder=1)
                ax_out.set_axisbelow(True)
                # ── Remove any existing colorbars added by shap ──────
                for ax_cb in fig_out.axes:
                    if ax_cb is ax_out:
                        continue
                    ax_cb.remove()
                # ── Add clean journal-style colorbar ─────────────────
                sm = plt.cm.ScalarMappable(cmap=_bee_cmap,
                                           norm=mcolors.Normalize(vmin=0, vmax=1))
                sm.set_array([])
                cbar = fig_out.colorbar(sm, ax=ax_out, orientation='vertical',
                                        fraction=0.025, pad=0.015, aspect=28,
                                        ticks=[0.0, 0.5, 1.0])
                cbar.set_label('Feature value\n(low → high)',
                               fontsize=11.0, labelpad=6, color='#1A1A2E')
                cbar.ax.tick_params(labelsize=10.0, colors='#444444', length=2)
                cbar.ax.set_yticklabels(['Low', 'Med', 'High'], fontsize=10.0)
                cbar.outline.set_linewidth(0.5)
                cbar.outline.set_edgecolor('#888888')
                fig_out.tight_layout(pad=1.2)
                _savefig_safe(fig_out, f'SHAP_Beeswarm_{tag}{tag_suffix}',
                              f'SHAP_Beeswarm_{tag}{tag_suffix}')
        except Exception:
            _write_status(f"SHAP_BEESWARM_SUMMARYPLOT_ERROR_{tag}{tag_suffix}.txt",
                          _tb_mod.format_exc())
            _log(f"      [SHAP Beeswarm summary_plot FAIL] using manual fallback: {tag_suffix}")
            _manual_beeswarm(figsize, tag_suffix)

    _bee_h = max(4.2, 0.42 * max_display + 1.2)
    _draw_beeswarm(figsize=(7.4, _bee_h),      tag_suffix='_wide')
    _sq_bee = max(5.0, 0.42 * max_display + 1.2)
    _draw_beeswarm(figsize=(_sq_bee, _sq_bee), tag_suffix='_square')


# ============================================================
# 8. CSV-only export helpers (그림 없음)
# ============================================================

def export_summary_csv(summ, out_dir, tag):
    """Export summary DataFrame to CSV."""
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    safe_csv(summ, out_dir / f'summary_{tag}.csv')


def export_pred_csv(pred_df, out_dir, tag):
    """Export predictions DataFrame to CSV."""
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    safe_csv(pred_df, out_dir / f'pred_{tag}.csv')


def export_predobs_csv(pred_df, model_list, ratio, target_name, out_dir, exp_name):
    """
    Export Predicted vs. Observed data to CSV.
    Columns: model, y_true, y_pred, availability_ratio, (feature values)
    """
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    rtag = ratio_tag(ratio)
    sub = pred_df[pred_df['availability_ratio'] == ratio].copy()
    sub = sub[sub['model'].isin(model_list)]
    if len(sub) == 0:
        return
    safe_csv(sub, out_dir / f'PredObs_{fs_tag(exp_name)}_{fs_tag(target_name)}_{rtag}.csv')


def export_four_predobs_csv(pred_vu_df, pred_tau_df, out_dir, exp_name, ratio,
                            model_name='Hetero_TL_ET'):
    """
    Export 4-panel PredObs data to CSV.
    Panels: (a) Vu→Vu  (b) Vu→tau  (c) tau→Vu  (d) tau→tau
    """
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    rtag = ratio_tag(ratio)

    vdf = pred_vu_df[(pred_vu_df['availability_ratio']==ratio) &
                     (pred_vu_df['model']==model_name)].copy()
    tdf = pred_tau_df[(pred_tau_df['availability_ratio']==ratio) &
                      (pred_tau_df['model']==model_name)].copy()
    for df in (vdf, tdf):
        if 'b_mm' in df.columns and 'd_mm' in df.columns:
            df['bd'] = (pd.to_numeric(df['b_mm'],errors='coerce') *
                        pd.to_numeric(df['d_mm'],errors='coerce'))

    all_rows = []
    def _add(panel, xlab, ylab, xvals, yvals):
        for xv,yv in zip(xvals,yvals):
            if np.isfinite(xv) and np.isfinite(yv):
                all_rows.append({'panel':panel,'x_label':xlab,'y_label':ylab,
                                 'x_value':xv,'y_value':yv})

    if len(vdf) >= 5:
        _add('(a) Vu_pred vs Vu_test','Vu_test (kN)','Vu_pred (kN)',
             vdf['y_true'].values, vdf['y_pred'].values)
        vv = vdf.dropna(subset=['bd']); vv = vv[vv['bd']>0]
        if len(vv) >= 5:
            _add('(b) Tau_pred_per_bd vs Vu_test',
                 'Vu_test (kN)','Tau_u,pred/bd (MPa)',
                 vv['y_true'].values,
                 vv['y_pred'].astype(float).values*1000./vv['bd'].astype(float).values)

    if len(tdf) >= 5:
        tt = tdf.dropna(subset=['bd']); tt = tt[tt['bd']>0]
        if len(tt) >= 5:
            _add('(c) Vu_pred_per_bd vs Tau_test',
                 'Tau_u,test (MPa)','Vu,pred/bd (kN)',
                 tt['y_true'].values,
                 tt['y_pred'].astype(float).values*tt['bd'].astype(float).values/1000.)
        _add('(d) Tau_pred vs Tau_test',
             'Tau_u,test (MPa)','Tau_u,pred (MPa)',
             tdf['y_true'].values, tdf['y_pred'].values)

    if all_rows:
        safe_csv(pd.DataFrame(all_rows),
                 out_dir / f'PredObs_4panel_{fs_tag(exp_name)}_{rtag}.csv')


def export_perm_imp_csv(feat_names, imp_mean, imp_std, out_dir, tag):
    """Export permutation importance to CSV."""
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({'feature': feat_names,
                       'feature_label': pretty_labels(feat_names),
                       'imp_mean': imp_mean,
                       'imp_std': imp_std})
    df = df.sort_values('imp_mean', ascending=False)
    safe_csv(df, out_dir / f'permImp_{tag}.csv')


def export_ap_csv(pred_df, model_name, target_name, ratio, out_dir, exp_name):
    """
    Export Actual/Prediction ratio vs variable data to CSV.
    Columns: model, variable, variable_label, variable_value, actual_over_pred
    """
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    rtag = ratio_tag(ratio)
    sub = pred_df.copy()
    if ratio is not None:
        sub = sub[np.isclose(sub['availability_ratio'].astype(float), float(ratio))]
    sub = sub[sub['model'] == model_name].copy()
    if len(sub) == 0:
        return
    sub['actual_over_pred'] = safe_div(sub['y_true'], sub['y_pred'])

    skip = {target_name,'y_true','y_pred','availability_ratio',
            'split_id','n_finetune','actual_over_pred','model'}
    num_cols = [c for c in sub.select_dtypes(include=[np.number]).columns if c not in skip]
    rows = []
    for xcol in num_cols:
        ss = sub[[xcol,'actual_over_pred']].replace([np.inf,-np.inf],np.nan).dropna()
        for _,r in ss.iterrows():
            rows.append({'model':model_name,'variable':xcol,
                         'variable_label':pretty_label(xcol),
                         'variable_value':r[xcol],
                         'actual_over_pred':r['actual_over_pred']})
    if rows:
        safe_csv(pd.DataFrame(rows),
                 out_dir / f'AP_{fs_tag(model_name)}_{fs_tag(target_name)}_{rtag}.csv')


def export_metric_table_csv(summary_df, model_list, ratio, out_dir, tag):
    """
    Export per-model R2/Mean(A/P)/CoV(A/P)/RMSE/MAE table to CSV.
    Intended for manuscript table generation.
    """
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    sub = summary_df.copy()
    if ratio is not None:
        sub = sub[np.isclose(sub['availability_ratio'].astype(float), float(ratio))]
    rows = []
    for model_name in model_list:
        sm = sub[sub['model']==model_name].copy()
        if len(sm) == 0: continue
        sm = sm.sort_values('availability_ratio'); r = sm.iloc[-1]
        rows.append({
            'Model':   model_name,
            'R2':      r.get('R2_median',    np.nan),
            'AP_mean': r.get('AP_mean_median', np.nan),
            'AP_CoV':  r.get('AP_CoV_median', np.nan),
            'RMSE':    r.get('RMSE_median',   np.nan),
            'MAE':     r.get('MAE_median',    np.nan),
        })
    if rows:
        safe_csv(pd.DataFrame(rows),
                 out_dir / f'MetricTable_{tag}.csv')


def export_g1vsg2_csv(g1_sum, g2_sum, target_name, ml_list, out_dir):
    """
    Export Group1 vs Group2 comparison metrics for all ML models to CSV.
    Columns: group, model, availability_ratio, R2_median, RMSE_median, MAE_median,
              R2_q25, R2_q75, RMSE_q25, RMSE_q75, MAE_q25, MAE_q75
    """
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    metric_cols_needed = []
    for m in ['R2','RMSE','MAE']:
        for suf in ['_median','_q25','_q75']:
            metric_cols_needed.append(f'{m}{suf}')

    rows = []
    def _extract(gsdf, group_tag, model_filter=None):
        # Only export the models needed for the manuscript comparison figure.
        models = model_filter if model_filter else result_display_models(use_rc=True, available=ml_list)
        for model_name in models:
            for _,r in gsdf[gsdf['model']==model_name].sort_values('availability_ratio').iterrows():
                row = {'group':group_tag,'model':model_name,
                       'availability_ratio':r['availability_ratio']}
                for c in metric_cols_needed:
                    row[c] = r.get(c, np.nan)
                rows.append(row)

    # Left/middle panels: selected TL and target-only comparison models.
    _extract(g1_sum, 'Group1_ML')
    _extract(g2_sum, 'Group2_ML')
    # Right panels: proposed Hetero TL-XGB only, Group 1 vs Group 2.
    _extract(g1_sum, 'Hetero_G1', model_filter=G1G2_PROPOSED_MODELS)
    _extract(g2_sum, 'Hetero_G2', model_filter=G1G2_PROPOSED_MODELS)

    safe_csv(pd.DataFrame(rows),
             out_dir / f'G1vsG2_metric_{fs_tag(target_name)}.csv')


def export_htl_vs_lit_csv(summ, target_name, ml_list, lit_list, ratio, out_dir, exp_name):
    """
    Export HeteroTL vs empirical equation comparison metrics to CSV.
    Corresponds to Steps 03/04.
    Columns: model, type(ML/Lit), availability_ratio, R2_median, RMSE_median, ...
    """
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    all_models = list(dict.fromkeys(ml_list + lit_list))
    metric_cols_needed = []
    for m in ['R2','RMSE','MAE','AP_mean','AP_CoV','Pearson_r']:
        for suf in ['_median','_q25','_q75']:
            metric_cols_needed.append(f'{m}{suf}')

    rows = []
    for model_name in all_models:
        mtype = 'ML' if model_name in ml_list else 'Lit'
        for _,r in summ[summ['model']==model_name].sort_values('availability_ratio').iterrows():
            row = {'model':model_name,'type':mtype,
                   'availability_ratio':r['availability_ratio']}
            for c in metric_cols_needed:
                row[c] = r.get(c, np.nan)
            rows.append(row)
    if rows:
        safe_csv(pd.DataFrame(rows),
                 out_dir / f'HTL_vs_Lit_{fs_tag(exp_name)}_{fs_tag(target_name)}.csv')


# ============================================================
# 9. Main experiment runner
# ============================================================
def run_experiment(exp_name, cfg, df_rc, use_rc, ml_models, lit_models, log_fn,
                   results_root=None):
    exp_dir = cfg['results_dir']
    for sub in ['02_tables', '03_predictions',
                '04_shap', '05_csv_exports', 'logs']:
        (exp_dir/sub).mkdir(parents=True, exist_ok=True)

    df_target = cfg['df_target'].copy()
    # lit_models가 비어 있으면 경험식 계산 자체를 건너뜀 (속도 절약)
    lit_pred_df = get_lit_preds(df_target, use_kwak=USE_KWAK) if lit_models else pd.DataFrame(index=df_target.index)
    artifacts = {}
    _all_src_info = []   # accumulate source model selection info per target → for txt report

    for target_name in ['V_u_KN','tau_u_MPa']:
        if target_name not in df_target.columns:
            log_fn(f"  [SKIP] {exp_name}/{target_name}: column not found"); continue

        if target_name == 'tau_u_MPa':
            # Ensure BOTH domains have tau available before source-model search.
            # This makes the RC source-model selection target-specific:
            #   V_u_KN    -> best RC model for shear force
            #   tau_u_MPa -> best RC model for shear stress
            if df_target['tau_u_MPa'].notna().sum() == 0:
                log_fn(f"  [INFO] target tau_u_MPa all-NaN -> recomputing from V_u_KN")
                df_target = add_tau(df_target)
            if use_rc and (('tau_u_MPa' not in df_rc.columns) or
                           (df_rc['tau_u_MPa'].notna().sum() == 0)):
                log_fn(f"  [INFO] RC tau_u_MPa missing/all-NaN -> recomputing from V_u_KN")
                df_rc = add_tau(df_rc)

        source_model_used = 'None'

        target_label = ('Shear Force  V_u (kN)'
                        if target_name=='V_u_KN'
                        else 'Shear Stress  τ_u = V_u/(bd)  (MPa)')
        common_feats,_,all_tft = build_feature_spec(
            df_rc, df_target, target_name, cfg['extra_cands'],
            cfg['drop_feats'], cfg['must_keep'])
        if len(common_feats)<2 or len(all_tft)<3:
            log_fn(f"  [SKIP] {exp_name}/{target_name}: insufficient features"); continue

        log_fn(f"\n{'─'*60}")
        log_fn(f"  {exp_name}  |  TARGET: {target_label}")
        log_fn(f"  (V_u_KN and tau_u_MPa are trained as two independent target models)")
        log_fn(f"  common     : {common_feats}")
        log_fn(f"  all_target : {all_tft}")

        rc_sub = (df_rc.dropna(subset=[target_name]).copy().reset_index(drop=True)
                  if use_rc else pd.DataFrame(columns=df_rc.columns))
        tg_sub = df_target.dropna(subset=[target_name]).copy().reset_index(drop=True)
        log_fn(f"  n(SFRC)={len(tg_sub)}   n(RC)={len(rc_sub) if use_rc else 'N/A'}")

        # ── FIX #1/#2: imp_src (RC용) + imp_tgt (SFRC 전체 기준, 루프 밖에서 한 번만 fit) ──
        # BUG #1 수정: 기존 코드는 imp_t를 매 fold/ratio마다 소규모 ft_df로 fit해서
        #   Group2의 잔류강도 피처(fsp, ft_direct, fr)에 서로 다른 median이
        #   적용되어 실제 물성값 정보가 손상되었음.
        # BUG #2 수정: 기존 코드는 sp_ft 생성 시 imp_src(RC 기반)를, Xft 생성 시
        #   imp_t(SFRC 서브셋 기반)를 각각 사용해 common_feats가 두 번 다른 값으로
        #   변환된 채 hstack되어 Hetero_TL 입력에 scale 불일치가 발생했음.
        # 수정 후: imp_tgt를 tg_sub 전체로 한 번 fit → CV 전 과정에서 동일한
        #   변환 기준 적용. source_pred 생성도 imp_tgt로 통일.
        imp_src = SimpleImputer(strategy='median')
        if use_rc:
            Xs_all = imp_src.fit_transform(rc_sub[common_feats])
            ys_all = rc_sub[target_name].astype(float).values

            # ══════════════════════════════════════════════════════════
            # STEP01 : Target-specific source model search (RC pretrained)
            #   This block runs independently for V_u_KN and tau_u_MPa.
            #   The selected best RC model for the current target is then used
            #   to generate source_pred in the HeteroTL stage below.
            # ── Train/Validation/Test 분리 ─────────────────────────
            #   Test  (20%, hold-out) : 모델 최종 성능 평가, 학습에 절대 미사용
            #   Train+Val (80%)       : 5-fold CV로 Val 성능 측정 → 최적 모델 선택
            #   Final fit             : Train+Val 전체(80%)로 재학습 → TL source 사용
            # ══════════════════════════════════════════════════════════
            from sklearn.model_selection import train_test_split, cross_val_predict
            from sklearn.metrics import r2_score as _r2s, mean_squared_error as _mse_fn

            _step01_dir = (results_root / '00_SourceModel'
                           if results_root is not None
                           else exp_dir.parent.parent / '00_SourceModel')
            _step01_dir.mkdir(parents=True, exist_ok=True)
            _s1_tgt_dir = _step01_dir / fs_tag(target_name)
            _s1_tgt_dir.mkdir(parents=True, exist_ok=True)

            # ── 80/20 train+val / test 분리 ────────────────────────
            Xs_tv, Xs_test, ys_tv, ys_test = train_test_split(
                Xs_all, ys_all, test_size=0.20,
                random_state=RANDOM_STATE)

            log_fn(f"  [00_SrcModel] {target_name}  "
                   f"RC total={len(ys_all)}  "
                   f"train+val={len(ys_tv)}  test={len(ys_test)}")

            _src_candidates = [
                # ── Tree / Bagging ────────────────────────────────────
                ('SrcET',       build_src_et()),        # ExtraTrees
                ('SrcRF',       build_src_rf()),        # RandomForest
                # ── Gradient Boosting ─────────────────────────────────
                ('SrcXGB',      build_src_xgb()),       # XGBoost MSE
                ('SrcXGB_MAE',  build_src_xgb_mae()),   # XGBoost MAE
                ('SrcLGBM',     build_src_lgbm()),      # LightGBM MAE
                ('SrcCAT',      build_src_cat()),       # CatBoost MAE
                ('SrcGBT',      build_src_gbt()),       # GBT Huber
                # ── Kernel ────────────────────────────────────────────
                # SrcSVR excluded: O(n²) kernel matrix → poor CV R² on RC n=794
                ('SrcGPR',      build_src_gpr()),       # GPR Matérn5/2 (500-sample subset)
            ]
            _src_candidates = [(n, m) for n, m in _src_candidates if m is not None]

            _src_scores = []
            _src_fitted = {}   # {name: model fit on full train+val}

            for _sname, _smdl in _src_candidates:
                _sdir = _s1_tgt_dir / _sname
                _sdir.mkdir(parents=True, exist_ok=True)
                try:
                    # ── 5-fold CV on train+val → validation 성능 ──────
                    _yp_cv = cross_val_predict(_smdl, Xs_tv, ys_tv, cv=5)
                    _r2_cv   = float(_r2s(ys_tv, _yp_cv))
                    _rmse_cv = float(np.sqrt(_mse_fn(ys_tv, _yp_cv)))

                    # ── Final fit on full train+val ────────────────────
                    _smdl_fit = clone(_smdl).fit(Xs_tv, ys_tv)
                    _src_fitted[_sname] = _smdl_fit

                    # ── Test set evaluation (hold-out) ─────────────────
                    _yp_test   = _smdl_fit.predict(Xs_test)
                    _r2_test   = float(_r2s(ys_test, _yp_test))
                    _rmse_test = float(np.sqrt(_mse_fn(ys_test, _yp_test)))
                    _ap_test   = _yp_test / np.where(
                                     np.abs(ys_test) < 1e-9, np.nan, ys_test)
                    _ap_mean   = float(np.nanmean(_ap_test))
                    _ap_cov    = float(np.nanstd(_ap_test) /
                                       (np.nanmean(_ap_test) + 1e-12))

                    _src_scores.append({
                        'source_model':    _sname,
                        'target':          target_name,
                        'n_RC_total':      len(ys_all),
                        'n_train_val':     len(ys_tv),
                        'n_test':          len(ys_test),
                        'n_common_feats':  len(common_feats),
                        'common_feats':    str(common_feats),
                        'R2_val_5fold':    round(_r2_cv,   4),
                        'RMSE_val_5fold':  round(_rmse_cv, 4),
                        'R2_test':         round(_r2_test,   4),
                        'RMSE_test':       round(_rmse_test, 4),
                        'AP_mean_test':    round(_ap_mean,   4),
                        'AP_CoV_test':     round(_ap_cov,    4),
                    })
                    log_fn(f"  [SrcModel] {_sname:<12} "
                           f"Val R²={_r2_cv:.4f}  RMSE={_rmse_cv:.1f}  |  "
                           f"Test R²={_r2_test:.4f}  RMSE={_rmse_test:.1f}")

                    # ── CSV: CV predicted (train+val) ──────────────────
                    safe_csv(pd.DataFrame({
                        'split': 'train_val_CV',
                        'y_true':   ys_tv,
                        'y_pred':   _yp_cv,
                        'residual': ys_tv - _yp_cv,
                        'AP_ratio': _yp_cv / np.where(
                                        np.abs(ys_tv) < 1e-9, np.nan, ys_tv),
                    }), _sdir / f'pred_trainval_CV_{target_name}.csv')

                    # ── CSV: test predicted (hold-out) ─────────────────
                    safe_csv(pd.DataFrame({
                        'split': 'test',
                        'y_true':   ys_test,
                        'y_pred':   _yp_test,
                        'residual': ys_test - _yp_test,
                        'AP_ratio': _ap_test,
                    }), _sdir / f'pred_test_{target_name}.csv')

                    # ── SHAP on test set (hold-out) ────────────────────
                    if HAS_SHAP:
                        try:
                            run_shap(
                                _smdl_fit, Xs_test,
                                feat_names=common_feats,
                                out_dir=_sdir / 'shap_test',
                                tag=f'{target_name}_{_sname}_test',
                                max_display=min(SHAP_MAX_DISPLAY,
                                                len(common_feats)),
                                log_fn=log_fn,
                            )
                            # SHAP on full train+val (global importance)
                            run_shap(
                                _smdl_fit, Xs_tv,
                                feat_names=common_feats,
                                out_dir=_sdir / 'shap_trainval',
                                tag=f'{target_name}_{_sname}_trainval',
                                max_display=min(SHAP_MAX_DISPLAY,
                                                len(common_feats)),
                                log_fn=log_fn,
                            )
                            pass  # SrcModel SHAP saved
                        except Exception as _se:
                            log_fn(f"  [00_SrcModel SHAP FAIL] {_sname}: {_se}")

                except Exception as _e:
                    import traceback as _tb2
                    log_fn(f"  [00_SrcModel FAIL] {_sname}: {_e}\n{_tb2.format_exc()}")

            # ── Source model 비교 CSV ──────────────────────────────
            if _src_scores:
                safe_csv(pd.DataFrame(_src_scores),
                         _s1_tgt_dir / f'source_model_comparison_{target_name}.csv')

            # ── 최고 Val CV R² → best source model 선택 ───────────
            # validation CV R²로 선택 (test는 최종 보고용, 선택 기준 아님)
            if _src_fitted:
                _valid_scores = [s for s in _src_scores
                                 if s['source_model'] in _src_fitted]
                _best_row = max(_valid_scores, key=lambda x: x['R2_val_5fold'])
                _best_src_name = _best_row['source_model']

                # ── GUI + 콘솔에 비교표 출력 ──────────────────────
                log_fn(f"\n  {'─'*55}")
                log_fn(f"  [00_SrcModel] Source model comparison ({target_name})")
                log_fn(f"  {'─'*55}")
                log_fn(f"  {'Model':<10}  {'Val CV R²':>10}  {'Val RMSE':>10}"
                       f"  {'Test R²':>9}  {'Test RMSE':>10}  {'AP mean':>8}  {'AP CoV':>7}")
                log_fn(f"  {'─'*55}")
                for _s in sorted(_valid_scores, key=lambda x: x['R2_val_5fold'],
                                 reverse=True):
                    _marker = ' ◀ BEST' if _s['source_model'] == _best_src_name else ''
                    log_fn(
                        f"  {_s['source_model']:<10}"
                        f"  {_s['R2_val_5fold']:>10.4f}"
                        f"  {_s['RMSE_val_5fold']:>10.2f}"
                        f"  {_s['R2_test']:>9.4f}"
                        f"  {_s['RMSE_test']:>10.2f}"
                        f"  {_s['AP_mean_test']:>8.4f}"
                        f"  {_s['AP_CoV_test']:>7.4f}"
                        f"{_marker}"
                    )
                log_fn(f"  {'─'*55}")
                log_fn(f"  → Selected: {_best_src_name}  "
                       f"(Val R²={_best_row['R2_val_5fold']:.4f}  "
                       f"Test R²={_best_row['R2_test']:.4f})\n")

                # best model을 전체 RC 데이터로 최종 재학습 (TL source용)
                src_best = clone(_src_candidates[
                    [n for n, _ in _src_candidates].index(_best_src_name)
                ][1]).fit(Xs_all, ys_all)
                source_model_used = _best_src_name
                log_fn(f"  [00_SrcModel] ✓ {_best_src_name} retrained on "
                       f"full RC ({len(ys_all)} samples) → used as HeteroTL source "
                       f"for {target_name}")

                # selected_source_model CSV (모든 점수 포함)
                safe_csv(pd.DataFrame([{
                    'target':               target_name,
                    'selected_source_model': _best_src_name,
                    'R2_val_5fold':         _best_row['R2_val_5fold'],
                    'RMSE_val_5fold':       _best_row['RMSE_val_5fold'],
                    'R2_test':              _best_row['R2_test'],
                    'RMSE_test':            _best_row['RMSE_test'],
                    'AP_mean_test':         _best_row['AP_mean_test'],
                    'AP_CoV_test':          _best_row['AP_CoV_test'],
                    'n_RC_final_fit':       len(ys_all),
                    'selection_criterion':  'max R2_val_5fold',
                    'note': f'Final fit on 100% RC for HeteroTL source_pred of {target_name}',
                }]), _s1_tgt_dir / f'selected_source_model_{target_name}.csv')
                # 누적 (txt 출력용)
                _all_src_info.append({
                    'target':        target_name,
                    'best':          _best_src_name,
                    'all_scores':    _valid_scores,
                    'best_row':      _best_row,
                    'n_RC_total':    len(ys_all),
                    'n_train_val':   len(ys_tv),
                    'n_test':        len(ys_test),
                    'common_feats':  common_feats,
                })
            else:
                src_best = build_src_et().fit(Xs_all, ys_all)
                source_model_used = 'SrcET_fallback'
                log_fn(f"  [00_SrcModel] All candidates failed → fallback ET for {target_name}")

        else:
            imp_src.fit(tg_sub[common_feats]); src_best = None

        # SFRC 전체 기준으로 imputer를 한 번만 fit (BUG #1/#2 핵심 수정)
        imp_tgt = SimpleImputer(strategy='median')
        imp_tgt.fit(tg_sub[all_tft])
        # common_feats의 all_tft 내 인덱스 — sp 생성 시 imp_tgt 변환값에서 추출
        _common_idx = [list(all_tft).index(c) for c in common_feats]

        # ── Cross-validation loop ─────────────────────────────────
        rkf = RepeatedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS,
                            random_state=RANDOM_STATE)
        split_id = 0; fold_rows_tn = []; pred_rows_tn = []

        log_fn(f"  [CV] {target_name}: starting TL/TargetOnly evaluation "
               f"({N_SPLITS}×{N_REPEATS} splits × {len(AVAILABILITY_RATIOS)} ratios); "
               f"source={source_model_used}")

        for train_idx, test_idx in rkf.split(tg_sub):
            split_id += 1
            train_df = tg_sub.iloc[train_idx].reset_index(drop=True)
            test_df  = tg_sub.iloc[test_idx].reset_index(drop=True)
            y_test   = test_df[target_name].astype(float).values
            xcols    = list(dict.fromkeys([c for c in all_tft if c!=target_name]))
            extra    = {c: (test_df[c].values if c in test_df.columns
                            else np.full(len(test_df),np.nan)) for c in xcols}
            base     = dict(experiment=exp_name, target=target_name, split_id=split_id)

            # Literature predictions (ratio-independent)
            orig_idx = tg_sub.iloc[test_idx].index
            for lm in lit_models:
                col = lm if target_name=='V_u_KN' else lm+'_tau'
                if col not in lit_pred_df.columns: continue
                yp_l = lit_pred_df.loc[orig_idx, col].values.astype(float)
                m_l  = np.isfinite(yp_l) & np.isfinite(y_test)
                if m_l.sum() < 5: continue
                md = metrics_dict(y_test[m_l], yp_l[m_l])
                for ratio in AVAILABILITY_RATIOS:
                    fold_rows_tn.append({**base,'availability_ratio':ratio,
                        'model':lm,'n_finetune':0,**md})
                    pred_rows_tn.append(pd.DataFrame({**base,'availability_ratio':ratio,
                        'model':lm,'y_true':y_test,'y_pred':yp_l,**extra}))

            # ML predictions per ratio
            for ratio in AVAILABILITY_RATIOS:
                n_ft = choose_ft(len(train_df), ratio)
                rng  = np.random.RandomState(RANDOM_STATE+split_id+int(ratio*1000))
                fi   = rng.choice(len(train_df), size=n_ft, replace=False)
                ft_df = train_df.iloc[fi].reset_index(drop=True)

                # BUG #1/#2 수정: imp_tgt(전체 기준)로 transform만 수행
                Xft   = imp_tgt.transform(ft_df[all_tft])
                Xtest = imp_tgt.transform(test_df[all_tft])
                yft   = ft_df[target_name].astype(float).values
                br    = {**base,'availability_ratio':ratio,'n_finetune':n_ft,
                         'source_model_used': source_model_used}
                pr_b  = {**base,'availability_ratio':ratio,'y_true':y_test,
                         'source_model_used': source_model_used, **extra}

                # ── TargetOnly models ────────────────────────────────────
                def _to(name, mdl):
                    if name not in ml_models:
                        return
                    if mdl is None:
                        return
                    try:
                        yp = mdl.fit(Xft,yft).predict(Xtest)
                        fold_rows_tn.append({**br,'model':name,**metrics_dict(y_test,yp)})
                        pred_rows_tn.append(pd.DataFrame({**pr_b,'model':name,'y_pred':yp}))
                    except Exception as e:
                        log_fn(f"    [SKIP] {name}: {e}")

                # Target-only models used in the 2×3 figure only
                _to('TargetOnly_XGB',     build_xgb())          # XGB (TO)
                _to('TargetOnly_ET',      build_tgt_et())       # ET (TO)
                _to('TargetOnly_RF',      build_rf())           # RF (TO)

                if use_rc:
                    # BUG #2 수정: common_feats를 imp_tgt 변환값에서 추출 → scale 통일
                    Xft_c  = Xft[:, _common_idx]
                    Xtst_c = Xtest[:, _common_idx]
                    sp_ft  = src_best.predict(Xft_c).reshape(-1,1)
                    sp_tst = src_best.predict(Xtst_c).reshape(-1,1)
                    Zft    = np.hstack([sp_ft, Xft])
                    Ztst   = np.hstack([sp_tst, Xtest])

                    def _htl(name, mdl):
                        if name not in ml_models:
                            return
                        if mdl is None:
                            return
                        try:
                            yp = mdl.fit(Zft,yft).predict(Ztst)
                            fold_rows_tn.append({**br,'model':name,**metrics_dict(y_test,yp)})
                            pred_rows_tn.append(pd.DataFrame({**pr_b,'model':name,'y_pred':yp}))
                        except Exception as e:
                            log_fn(f"    [SKIP] {name}: {e}")

                    # Heterogeneous-transfer models used in the 2×3 figure only
                    _htl('Hetero_TL_XGB',     build_xgb())       # HTL-XGB (Proposed)
                    _htl('Hetero_TL_ET',      build_tgt_et())    # HTL-ET
                    _htl('Hetero_TL_RF',      build_rf())        # HTL-RF

            if split_id % 5 == 0 or split_id == N_SPLITS*N_REPEATS:
                log_fn(f"    [CV] {target_name}: split {split_id}/{N_SPLITS*N_REPEATS} done")

        # ── Aggregate results ─────────────────────────────────
        fold_df = pd.DataFrame(fold_rows_tn)
        pred_df = pd.concat(pred_rows_tn, ignore_index=True)

        tbl_dir = exp_dir / '02_tables'
        tbl_dir.mkdir(parents=True, exist_ok=True)
        safe_csv(fold_df,  tbl_dir / f'fold_{target_name}.csv')
        safe_csv(pred_df,  exp_dir / '03_predictions' / f'pred_{target_name}.csv')

        summ = summarize_metrics(fold_df, ['experiment','target','availability_ratio','model'])
        safe_csv(summ, tbl_dir / f'summary_{target_name}.csv')

        best_model = PROPOSED_MODEL if use_rc else 'TargetOnly_XGB'

        # SHAP 전용 모델 선택
        # NOTE: Hetero_TL_ET는 학습/성능표/AP 분석에는 그대로 포함하지만,
        #       SHAP 계산에서는 제외한다.
        #       본 연구의 제안 모델은 Hetero_TL_XGB로 고정한다.
        shap_model = best_model
        shap_model_builder = build_tgt_et
        if use_rc:
            shap_model = PROPOSED_MODEL
            shap_model_builder = build_xgb
            if shap_model_builder() is None:
                shap_model = None
                shap_model_builder = None

        hsum = summ[summ['model']==best_model].sort_values('R2_median', ascending=False)
        best_ratio = float(hsum.iloc[0]['availability_ratio']) if len(hsum) else 0.9

        log_fn(f"  → Saving CSVs for {target_name}...")

        # ── CSV exports (no plots) ────────────────────────────
        final_ratio = max(AVAILABILITY_RATIOS)
        csv_dir = exp_dir / '05_csv_exports'
        csv_dir.mkdir(parents=True, exist_ok=True)

        # PredObs CSV (manuscript-display models only @ final ratio)
        all_models_avail = list(dict.fromkeys(pred_df['model'].astype(str)))
        display_models_avail = result_display_models(use_rc=use_rc, available=all_models_avail)
        export_predobs_csv(pred_df, display_models_avail, final_ratio, target_name,
                           csv_dir, exp_name)

        # A/P vs variables CSV (Hetero only, all ratios)
        for ratio in AVAILABILITY_RATIOS:
            export_ap_csv(pred_df, best_model, target_name, ratio,
                          csv_dir / 'AP_csv', exp_name)

        # Metric table CSV (manuscript-display models only)
        export_metric_table_csv(summ, display_models_avail, final_ratio,
                                csv_dir, f'{exp_name}_{fs_tag(target_name)}')

        # ── SHAP : 제안 모델 Hetero_TL_XGB로 all & test-only ─────────
        # train/test 구조 재확인:
        #   CV loop : RepeatedKFold(5×5=25 splits), 매 split마다 train/test 분리
        #             → fold_df, pred_df 에 평균 성능 저장
        #   SHAP용  : availability_ratio별로 고정 random seed로 train/test 분리
        #             → (A) 전체 SFRC: 모델 전반적 해석
        #             → (B) test-only: 실제 일반화 성능 해석 (hold-out 기준)
        if HAS_SHAP and use_rc and shap_model is not None and shap_model_builder is not None:
            sdir      = exp_dir / '04_shap' / 'all_data'
            sdir_test = exp_dir / '04_shap' / 'test_only'
            sdir.mkdir(parents=True, exist_ok=True)
            sdir_test.mkdir(parents=True, exist_ok=True)

            feat_names_full = ['source_pred'] + list(all_tft)

            log_fn(f"  [SHAP] {shap_model} | {exp_name} | "
                   f"{target_name} | source={source_model_used} | "
                   f"ratios={[int(r*100) for r in AVAILABILITY_RATIOS]}% "
                   f"(proposed model; Hetero_TL_ET SHAP skipped)")

            for ratio in AVAILABILITY_RATIOS:
                rtag = f"avail{int(ratio*100):03d}"
                try:
                    tg_reset = tg_sub.reset_index(drop=True)
                    n_total  = len(tg_reset)
                    n_sh     = choose_ft(n_total, ratio)
                    rng_sh   = np.random.RandomState(RANDOM_STATE + int(ratio*1000) + 13)

                    # ── train / test 분리 (SHAP 전용, CV와 동일 seed 계열) ──
                    all_idx   = np.arange(n_total)
                    train_idx_sh = rng_sh.choice(n_total, n_sh, replace=False)
                    test_idx_sh  = np.setdiff1d(all_idx, train_idx_sh)

                    ft_sh   = tg_reset.iloc[train_idx_sh].reset_index(drop=True)
                    tst_sh  = tg_reset.iloc[test_idx_sh].reset_index(drop=True)

                    Xft_sh  = imp_tgt.transform(ft_sh[all_tft])
                    Xtst_sh = imp_tgt.transform(tst_sh[all_tft])
                    yft_sh  = ft_sh[target_name].astype(float).values
                    ytst_sh = tst_sh[target_name].astype(float).values

                    sp_sh   = src_best.predict(Xft_sh[:,  _common_idx]).reshape(-1, 1)
                    sp_tst  = src_best.predict(Xtst_sh[:, _common_idx]).reshape(-1, 1)
                    Z_sh    = np.hstack([sp_sh,  Xft_sh])
                    Z_tst   = np.hstack([sp_tst, Xtst_sh])

                    # ── 전체 데이터셋 (for global interpretation) ──────────
                    Xa    = imp_tgt.transform(tg_reset[all_tft])
                    spa   = src_best.predict(Xa[:, _common_idx]).reshape(-1, 1)
                    Z_all = np.hstack([spa, Xa])

                    if len(feat_names_full) != Z_all.shape[1]:
                        log_fn(f"    [SHAP SKIP] ratio={ratio:.2f}: shape mismatch")
                        continue

                    # ── 모델 학습 ───────────────────────────────────────────
                    mdl_sh = shap_model_builder().fit(Z_sh, yft_sh)

                    # ── ratio=0.90 모델 저장 (예측용) ───────────────────────
                    if abs(ratio - 0.90) < 1e-6:
                        try:
                            import joblib as _jl
                            _model_dir = exp_dir / '06_saved_model'
                            _model_dir.mkdir(parents=True, exist_ok=True)
                            # HTL model (target + source_pred feature)
                            _jl.dump(mdl_sh,   _model_dir / f'htl_model_{target_name}.pkl')
                            # Source model (RC → source_pred)
                            _jl.dump(src_best, _model_dir / f'source_model_{target_name}.pkl')
                            # Imputer (SFRC feature imputation)
                            _jl.dump(imp_tgt,  _model_dir / f'imputer_{target_name}.pkl')
                            # Feature metadata
                            import json as _json
                            _meta = {
                                'all_tft':       list(all_tft),
                                'common_feats':  list(common_feats),
                                'feat_names_full': list(feat_names_full),
                                'target_name':   target_name,
                                'exp_name':      exp_name,
                                'model':         shap_model,
                                'ratio':         ratio,
                            }
                            (_model_dir / f'meta_{target_name}.json').write_text(
                                _json.dumps(_meta, indent=2), encoding='utf-8')
                            log_fn(f"  [Model Saved] {target_name} → {_model_dir}")
                        except Exception as _me:
                            log_fn(f"  [Model Save WARN] {_me}")

                    # ── test 성능 로그 ──────────────────────────────────────
                    yp_tst = mdl_sh.predict(Z_tst)
                    from sklearn.metrics import r2_score as _r2s, mean_squared_error as _mse
                    _r2_tst   = _r2s(ytst_sh, yp_tst)
                    _rmse_tst = float(np.sqrt(_mse(ytst_sh, yp_tst)))
                    log_fn(f"    [SHAP] {target_name} | {rtag} | "
                           f"train={n_sh}  test={len(test_idx_sh)}  "
                           f"test R²={_r2_tst:.3f}  RMSE={_rmse_tst:.1f}")

                    # ── test set 예측 결과 CSV ─────────────────────────────
                    safe_csv(pd.DataFrame({
                        'y_true':   ytst_sh,
                        'y_pred':   yp_tst,
                        'residual': ytst_sh - yp_tst,
                        'AP_ratio': yp_tst / np.where(np.abs(ytst_sh) < 1e-9,
                                                       np.nan, ytst_sh),
                    }), sdir_test / f'test_pred_{target_name}_{rtag}.csv')

                    # ── (A) SHAP on full dataset ────────────────────────────
                    run_shap(
                        mdl_sh, Z_all, feat_names_full,
                        out_dir=sdir,
                        tag=f'{target_name}_{shap_model}_{rtag}_all',
                        max_display=min(SHAP_MAX_DISPLAY, len(feat_names_full)),
                        log_fn=log_fn,
                    )

                    # ── (B) SHAP on test set only ───────────────────────────
                    run_shap(
                        mdl_sh, Z_tst, feat_names_full,
                        out_dir=sdir_test,
                        tag=f'{target_name}_{shap_model}_{rtag}_test',
                        max_display=min(SHAP_MAX_DISPLAY, len(feat_names_full)),
                        log_fn=log_fn,
                    )

                except Exception as e:
                    import traceback as _tb
                    log_fn(f"    [SHAP ERROR] ratio={ratio:.2f}: {e}")
                    log_fn(_tb.format_exc())

        elif HAS_SHAP and use_rc:
            # 제안 모델 Hetero_TL_XGB를 사용할 수 없으면 SHAP만 건너뜀.
            sdir = exp_dir / '04_shap'
            sdir.mkdir(parents=True, exist_ok=True)
            msg = (f"{PROPOSED_MODEL} was selected as the proposed SHAP model, "
                   "but XGBoost is not available in this Python environment.\n"
                   "Fix: pip install xgboost\n")
            (sdir / f'SHAP_SKIPPED_NO_PROPOSED_MODEL_{target_name}.txt').write_text(msg, encoding='utf-8')
            log_fn(f"  [SHAP SKIP] {exp_name}/{target_name}: proposed SHAP model {PROPOSED_MODEL} unavailable -> pip install xgboost")

        elif HAS_SHAP and not use_rc:
            # RC 없는 경우: TargetOnly_ET SHAP
            sdir = exp_dir / '04_shap'
            sdir.mkdir(parents=True, exist_ok=True)
            feat_names_full = list(all_tft)

            log_fn(f"  [SHAP] TargetOnly_ET | {exp_name} | {target_name}")

            for ratio in AVAILABILITY_RATIOS:
                rtag = f"avail{int(ratio*100):03d}"
                try:
                    n_sh   = choose_ft(len(tg_sub), ratio)
                    rng_sh = np.random.RandomState(RANDOM_STATE + int(ratio*1000) + 13)
                    fi_sh  = rng_sh.choice(len(tg_sub), n_sh, replace=False)
                    ft_sh  = tg_sub.iloc[fi_sh].reset_index(drop=True)

                    # BUG #5 수정: imp_tgt(전체 기준)로 통일
                    Xft_sh = imp_tgt.transform(ft_sh[all_tft])
                    yft_sh = ft_sh[target_name].astype(float).values
                    mdl_sh = build_tgt_et().fit(Xft_sh, yft_sh)

                    tg_reset = tg_sub.reset_index(drop=True)
                    Z_all    = imp_tgt.transform(tg_reset[all_tft])

                    if len(feat_names_full) != Z_all.shape[1]:
                        continue

                    run_shap(
                        mdl_sh, Z_all, feat_names_full,
                        out_dir=sdir,
                        tag=f'{target_name}_{best_model}_{rtag}',
                        max_display=min(SHAP_MAX_DISPLAY, len(feat_names_full)),
                        log_fn=log_fn,
                    )
                except Exception as e:
                    import traceback as _tb
                    log_fn(f"    [SHAP ERROR] ratio={ratio:.2f}: {e}")
                    log_fn(_tb.format_exc())

        elif not HAS_SHAP:
            sdir = exp_dir / '04_shap'
            sdir.mkdir(parents=True, exist_ok=True)
            msg = ("shap package is not installed in the Python environment, "
                   "so SHAP files were not generated. Install with: pip install shap\n")
            (sdir / f'SHAP_NOT_RUN_{target_name}.txt').write_text(msg, encoding='utf-8')
            log_fn(f"  [SHAP SKIP] {exp_name}/{target_name}: shap not installed -> pip install shap")

        # ── Save final model (ratio=0.90, full SFRC) for inference ──
        try:
            import joblib as _jl
            mdl_dir = exp_dir / '06_saved_model'
            mdl_dir.mkdir(parents=True, exist_ok=True)

            # Re-train on full SFRC at ratio=0.90
            _ratio_save = 0.90
            _n_save  = choose_ft(len(tg_sub), _ratio_save)
            _rng_sv  = np.random.RandomState(RANDOM_STATE + 999 + int(_ratio_save*1000))
            _fi_sv   = _rng_sv.choice(len(tg_sub), _n_save, replace=False)
            _ft_sv   = tg_sub.iloc[_fi_sv].reset_index(drop=True)
            _Xft_sv  = imp_tgt.transform(_ft_sv[all_tft])
            _yft_sv  = _ft_sv[target_name].astype(float).values

            if use_rc and src_best is not None:
                _sp_sv   = src_best.predict(_Xft_sv[:, _common_idx]).reshape(-1, 1)
                _Zft_sv  = np.hstack([_sp_sv, _Xft_sv])
                _fn_save = ['source_pred'] + list(all_tft)
                _mdl_sv  = build_xgb() or build_tgt_et()
                _mdl_sv.fit(_Zft_sv, _yft_sv)
                _jl.dump({
                    'model':        _mdl_sv,
                    'src_model':    src_best,
                    'imp_tgt':      imp_tgt,
                    'all_tft':      list(all_tft),
                    'common_feats': common_feats,
                    'common_idx':   _common_idx,
                    'feat_names':   _fn_save,
                    'target_name':  target_name,
                    'exp_name':     exp_name,
                    'use_rc':       use_rc,
                    'model_name':   best_model,
                }, mdl_dir / f'model_{target_name}.pkl')
            else:
                _mdl_sv = build_xgb() or build_tgt_et()
                _mdl_sv.fit(_Xft_sv, _yft_sv)
                _jl.dump({
                    'model':        _mdl_sv,
                    'src_model':    None,
                    'imp_tgt':      imp_tgt,
                    'all_tft':      list(all_tft),
                    'common_feats': [],
                    'common_idx':   [],
                    'feat_names':   list(all_tft),
                    'target_name':  target_name,
                    'exp_name':     exp_name,
                    'use_rc':       use_rc,
                    'model_name':   best_model,
                }, mdl_dir / f'model_{target_name}.pkl')
            log_fn(f"  [Model Saved] {mdl_dir / f'model_{target_name}.pkl'}")
        except Exception as _me:
            log_fn(f"  [Model Save WARN] {_me}")

        artifacts[target_name] = dict(summary=summ, pred_df=pred_df,
                                      best_ratio=best_ratio, best_model=best_model,
                                      source_model_used=source_model_used,
                                      all_tft=all_tft)

    # ── 01_HeteroTL 폴더에 source model 정보 txt 저장 ─────────────
    if _all_src_info:
        try:
            lines = []
            lines.append("=" * 62)
            lines.append("  SOURCE MODEL SUMMARY  —  used for HeteroTL")
            lines.append(f"  Experiment : {exp_name}")
            lines.append(f"  Selection criterion : max Val CV R² (5-fold on 80% RC)")
            lines.append("=" * 62)
            for info in _all_src_info:
                tn    = info['target']
                best  = info['best']
                br    = info['best_row']
                lines.append("")
                lines.append(f"  TARGET : {tn}")
                lines.append(f"  RC data  total={info['n_RC_total']}  "
                             f"train+val={info['n_train_val']}  "
                             f"test(hold-out)={info['n_test']}")
                lines.append(f"  Common features ({len(info['common_feats'])}) : "
                             f"{', '.join(info['common_feats'])}")
                lines.append("")
                lines.append(f"  {'Model':<10}  {'Val CV R²':>10}  {'Val RMSE':>10}"
                             f"  {'Test R²':>9}  {'Test RMSE':>10}"
                             f"  {'AP mean':>8}  {'AP CoV':>7}")
                lines.append(f"  {'-'*58}")
                for s in sorted(info['all_scores'],
                                key=lambda x: x['R2_val_5fold'], reverse=True):
                    marker = ' ◀ BEST (used for HeteroTL)' \
                             if s['source_model'] == best else ''
                    lines.append(
                        f"  {s['source_model']:<10}"
                        f"  {s['R2_val_5fold']:>10.4f}"
                        f"  {s['RMSE_val_5fold']:>10.2f}"
                        f"  {s['R2_test']:>9.4f}"
                        f"  {s['RMSE_test']:>10.2f}"
                        f"  {s['AP_mean_test']:>8.4f}"
                        f"  {s['AP_CoV_test']:>7.4f}"
                        f"{marker}"
                    )
                lines.append("")
                lines.append(f"  ✓ Selected : {best}")
                lines.append(f"    Val CV R²  = {br['R2_val_5fold']:.4f}")
                lines.append(f"    Val RMSE   = {br['RMSE_val_5fold']:.4f}")
                lines.append(f"    Test R²    = {br['R2_test']:.4f}")
                lines.append(f"    Test RMSE  = {br['RMSE_test']:.4f}")
                lines.append(f"    AP mean    = {br['AP_mean_test']:.4f}")
                lines.append(f"    AP CoV     = {br['AP_CoV_test']:.4f}")
                lines.append(f"    Final fit  : retrained on full RC "
                             f"({info['n_RC_total']} samples) for source_pred")
            lines.append("")
            lines.append("=" * 62)
            txt_path = exp_dir / 'source_model_info.txt'
            txt_path.write_text('\n'.join(lines), encoding='utf-8')
            log_fn(f"  [Info] source_model_info.txt → {txt_path}")
        except Exception as _e:
            log_fn(f"  [WARN] source_model_info.txt save failed: {_e}")

    with open(exp_dir/'logs'/'completed.txt','w') as f:
        f.write(f"Experiment: {exp_name}\nRC used: {use_rc}\n")
    return artifacts


# ============================================================
# 10. 3×4 AP/Bias Plot  ← 핵심 새 그림
#     행: [PROPOSED_MODEL, top-2 empirical]
#     열: [a/d, rho, d_mm, RI]
# ============================================================

AP_PLOT_VARS = [
    ('a_d',   r'$a/d$'),
    ('rho',   r'$\rho$ (%)'),
    ('d_mm',  r'$d$ (mm)'),
    ('RI',    r'$RI$'),
]

# 경험식 후보 (Sharma, Kwak, Ashour 중 상위 2개 자동 선택)
AP_EMPIRICAL_CANDIDATES = ['Kwak_2002', 'Ashour_1992', 'Sharma_1986',
                            'Kuntia_1999', 'ACI544_4R_18']

def _pick_top2_for_applot(summary_df, candidates, ratio=0.90):
    """
    Select top-2 empirical equations by R2_median at given ratio.
    Falls back to first 2 candidates if none found.
    """
    sub = summary_df[
        (np.isclose(summary_df['availability_ratio'].astype(float), float(ratio))) &
        (summary_df['model'].isin(candidates))
    ].copy()
    if len(sub) == 0:
        sub = summary_df[summary_df['model'].isin(candidates)].copy()
    if len(sub) == 0:
        return candidates[:2]
    ranked = (sub.groupby('model')['R2_median']
                .max().reset_index()
                .sort_values('R2_median', ascending=False))
    return ranked['model'].head(2).tolist()


def plot_ap_3x4(pred_df, summary_df, target_name, out_dir, exp_name,
                ratio=0.90, our_model='Hetero_TL_ET',
                empirical_candidates=None):
    """
    3-row × 4-column AP/Bias panel figure.

    Rows: [our_model, empirical_1, empirical_2]
    Cols: [a/d, rho, d_mm, RI]

    x-axis: variable value
    y-axis: Actual/Prediction (= y_true / y_pred)
    Reference line: A/P = 1.0 (dashed)
    Includes LOWESS trend line

    Saved as: PNG + PDF (ASCE 600 dpi)
    CSV:  AP_3x4_{exp_name}_{target}.csv
    """
    if empirical_candidates is None:
        empirical_candidates = AP_EMPIRICAL_CANDIDATES

    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    rtag = ratio_tag(ratio)

    # ── 데이터 준비 ────────────────────────────────────────
    sub = pred_df[np.isclose(pred_df['availability_ratio'].astype(float), float(ratio))].copy()
    if len(sub) == 0:
        return

    sub['actual_over_pred'] = safe_div(sub['y_true'], sub['y_pred'])

    # Top-2 경험식 선택
    top2 = _pick_top2_for_applot(summary_df, empirical_candidates, ratio)
    row_models = [our_model] + top2
    # 없는 모델 제거
    avail = set(sub['model'])
    row_models = [m for m in row_models if m in avail]
    if len(row_models) == 0:
        return

    # 모델 레이블
    model_labels = {
        'Hetero_TL_ET': 'Hetero_TL_ET (proposed)',
        'Kwak_2002': 'Kwak (2002)',
        'Ashour_1992': 'Ashour (1992)',
        'Sharma_1986': 'Sharma (1986)',
        'Kuntia_1999': 'Kuntia (1999)',
        'ACI544_4R_18': 'ACI 544.4R-18',
    }

    # 변수 컬럼 존재 확인
    plot_vars = [(col, lab) for col, lab in AP_PLOT_VARS if col in sub.columns]
    if not plot_vars:
        return

    nrows = len(row_models)
    ncols = len(plot_vars)

    fig_w = 2.55 * ncols + 0.4
    fig_h = 2.35 * nrows + 0.4
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h),
                              squeeze=False)

    # 패널 레이블 (a), (b), ... 순서
    panel_labels = [chr(ord('a') + i) for i in range(nrows * ncols)]

    for i, model_name in enumerate(row_models):
        sm = sub[sub['model'] == model_name].copy()
        sm = sm.replace([np.inf, -np.inf], np.nan)
        color = '#000000' if model_name == our_model else ('#3A3A3A' if i==1 else '#707070')
        marker = 'o' if model_name == our_model else ('s' if i==1 else '^')

        for j, (xcol, xlab) in enumerate(plot_vars):
            ax = axes[i][j]
            panel_idx = i * ncols + j

            ss = sm[[xcol, 'actual_over_pred']].dropna()
            if len(ss) < 5:
                ax.axis('off'); continue

            xv = ss[xcol].values
            ap = ss['actual_over_pred'].values

            # scatter
            ax.scatter(xv, ap, s=16, facecolors='white', edgecolors=color,
                       marker=marker, linewidths=0.75, alpha=0.82, zorder=3)

            # 기준선
            ax.axhline(1.0, color='black', linestyle='--', linewidth=0.85, zorder=2)

            # LOWESS
            if HAS_LOWESS and len(ss) >= 12:
                try:
                    srt = ss.sort_values(xcol)
                    yy = sm_lowess(srt['actual_over_pred'].values, srt[xcol].values,
                                   frac=0.45, return_sorted=True)
                    ax.plot(yy[:,0], yy[:,1], color=color, lw=1.15, zorder=4)
                except Exception:
                    pass

            # 패널 레이블 (a), (b), ...
            ax.text(0.04, 0.97, f'({panel_labels[panel_idx]})',
                    transform=ax.transAxes, ha='left', va='top',
                    fontsize=10.5, fontweight='bold')

            # 열 제목 (첫 번째 행만)
            if i == 0:
                ax.set_title(xlab, fontsize=12.5, pad=5)
            else:
                ax.set_title('')

            # y축 레이블 (첫 번째 열만)
            if j == 0:
                row_label = model_labels.get(model_name, model_name)
                ax.set_ylabel(f'{row_label}\nA/P', fontsize=10.5)
            else:
                ax.set_ylabel('')

            # x축 레이블 (마지막 행만)
            if i == nrows - 1:
                ax.set_xlabel(xlab, fontsize=11)
            else:
                ax.set_xlabel('')

            ax.tick_params(labelsize=8.5)
            beautify(ax)

    fig.tight_layout(pad=0.55, h_pad=0.8, w_pad=0.6)
    savefig_all(fig, out_dir / f'AP_3x{ncols}_{fs_tag(exp_name)}_{fs_tag(target_name)}_{rtag}')

    # ── CSV 저장 ──────────────────────────────────────────
    rows_csv = []
    for model_name in row_models:
        sm = sub[sub['model']==model_name].copy()
        sm = sm.replace([np.inf,-np.inf],np.nan)
        for xcol, xlab in plot_vars:
            if xcol not in sm.columns: continue
            ss = sm[[xcol,'actual_over_pred']].dropna()
            for _, r in ss.iterrows():
                rows_csv.append({'model': model_name,
                                 'model_label': model_labels.get(model_name, model_name),
                                 'variable': xcol,
                                 'variable_label': xlab,
                                 'variable_value': r[xcol],
                                 'actual_over_pred': r['actual_over_pred']})
    if rows_csv:
        safe_csv(pd.DataFrame(rows_csv),
                 out_dir / f'AP_3x{ncols}_{fs_tag(exp_name)}_{fs_tag(target_name)}_{rtag}.csv')


# ============================================================
# 11. Workflow orchestrator (_worker가 호출)
# ============================================================

def run_workflow(all_arts, results_root, ml_models, use_rc, log_fn):
    """
    Called after run_experiment() completes.
    - Steps 02–04: CSV outputs only
    - Step 05: metric table CSV
    - Step 06: 4-panel PredObs CSV
    - AP 3×4 plot: PNG/PDF (Group1, Group2 × Vu + tau)
    """
    wf_dir = Path(results_root) / '06_Workflow'
    wf_dir.mkdir(parents=True, exist_ok=True)

    final_ratio = max(AVAILABILITY_RATIOS)

    # ── Step02: G1 vs G2 CSV ─────────────────────────────
    for tn in ['V_u_KN','tau_u_MPa']:
        g1 = all_arts.get('Group1',{}).get(tn)
        g2 = all_arts.get('Group2',{}).get(tn)
        if g1 and g2:
            export_g1vsg2_csv(g1['summary'], g2['summary'], tn, ml_models,
                              wf_dir / '07_G1vsG2_CSV')

    # ── Step03/04: Hetero vs Lit CSV ─────────────────────
    # (경험식 제외 설정이므로 스킵)
    # for exp_name, arts in all_arts.items():
    #     ...export_htl_vs_lit_csv(...)

    # ── Step05: metric table CSV ─────────────────────────
    for exp_name, arts in all_arts.items():
        for tn in ['V_u_KN','tau_u_MPa']:
            art = arts.get(tn)
            if art is None: continue
            all_m = result_display_models(use_rc=use_rc,
                                          available=list(dict.fromkeys(art['pred_df']['model'].astype(str))))
            export_metric_table_csv(
                art['summary'], all_m, final_ratio,
                wf_dir / '09_MetricTables', f'{exp_name}_{fs_tag(tn)}')

    # ── Step06: 4-panel PredObs CSV ─────────────────────
    for exp_name, arts in all_arts.items():
        vu_art  = arts.get('V_u_KN')
        tau_art = arts.get('tau_u_MPa')
        if vu_art and tau_art:
            export_four_predobs_csv(
                vu_art['pred_df'], tau_art['pred_df'],
                wf_dir / '10_PredObs4panel_CSV',
                exp_name, final_ratio,
                model_name=(vu_art['best_model']))

    # ── AP 3×4 Plot  ← 논문용 핵심 그림 ─────────────────
    log_fn("\n[AP Plot] Generating 3×4 A/P ratio plot...")
    ap_plot_dir = wf_dir / '11_AP_3x4_Plots'
    ap_plot_dir.mkdir(parents=True, exist_ok=True)

    for exp_name, arts in all_arts.items():
        for tn in ['V_u_KN','tau_u_MPa']:
            art = arts.get(tn)
            if art is None: continue
            our_m = art['best_model']
            try:
                plot_ap_3x4(
                    pred_df=art['pred_df'],
                    summary_df=art['summary'],
                    target_name=tn,
                    out_dir=ap_plot_dir / exp_name,
                    exp_name=exp_name,
                    ratio=final_ratio,
                    our_model=our_m,
                    empirical_candidates=[],   # 경험식 행 제외, 우리 모델 1행만
                )
                log_fn(f"  → AP 3×4: {exp_name} / {tn} done")
            except Exception as e:
                log_fn(f"  [AP Plot ERROR] {exp_name}/{tn}: {e}")

    log_fn(f"\n[Workflow] Complete → {wf_dir}")
    return wf_dir


# ============================================================
# 12. GUI  —  Light theme, commercial-grade
# ============================================================

class _FlatBtn(tk.Label):
    """Reliable cross-platform button (Label-based, no Canvas issues on Windows)."""
    def __init__(self, parent, text, command=None,
                 bg=None, fg=None, hover=None,
                 font=None, padx=18, pady=8, **kw):
        self._bg  = bg    or C["btn_bg"]
        self._fg  = fg    or C["btn_fg"]
        self._hov = hover or C["btn_hover"]
        self._cmd = command
        self._on  = True
        super().__init__(parent, text=text, bg=self._bg, fg=self._fg,
                         font=font or FONT_LABEL, cursor="hand2",
                         padx=padx, pady=pady, relief="flat", bd=0, **kw)
        self.bind("<Enter>",           self._e)
        self.bind("<Leave>",           self._l)
        self.bind("<ButtonRelease-1>", self._r)

    def _e(self, _):
        if self._on: self.configure(bg=self._hov)
    def _l(self, _):
        if self._on: self.configure(bg=self._bg)
    def _r(self, _):
        if self._on and self._cmd: self._cmd()

    def enable(self, val=True):
        self._on = val
        self.configure(bg=self._bg if val else C["border"],
                       fg=self._fg if val else C["text_dim"],
                       cursor="hand2" if val else "arrow")


def _sep(parent, orient="h", color=None, **kw):
    color = color or C["border"]
    if orient == "h":
        return tk.Frame(parent, bg=color, height=1, **kw)
    return tk.Frame(parent, bg=color, width=1, **kw)


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("RC → SFRC  Transfer Learning Studio")
        self.resizable(True, True)
        self.configure(bg=C["bg"])
        # Publication-screenshot-friendly default size: wide enough for all columns,
        # not excessively tall, so the GUI looks visually filled instead of sparse.
        sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
        w = min(1380, max(1280, int(sw * 0.92)))
        h = min(860,  max(760,  int(sh * 0.82)))
        self.geometry(f"{w}x{h}+{(sw-w)//2}+{(sh-h)//2}")
        self.minsize(1180, 720)
        self._running = False
        self._thread  = None
        self._build_ui()
        self._check_files()

    # ── Layout ─────────────────────────────────────────────────────
    def _build_ui(self):
        FA = "Arial"
        # ── Header ─────────────────────────────────────────────────
        hdr = tk.Frame(self, bg=C["panel"],
                       highlightbackground=C["border"], highlightthickness=1)
        hdr.pack(fill="x")
        _sep(hdr).pack(side="bottom", fill="x")

        tk.Label(hdr, text="RC → SFRC  Heterogeneous Transfer Learning",
                 font=("Arial", 22, "bold"), bg=C["panel"], fg=C["text"]).pack(pady=(5, 0))
        tk.Label(hdr,
                 text="RC → SFRC Heterogeneous Transfer Learning  •  SHAP analysis  •  CSV/plot export",
                 font=("Arial", 14), bg=C["panel"], fg=C["text_dim"]).pack(pady=(0, 4))

        # ── Slim accent bar ────────────────────────────────────────
        tk.Frame(self, bg=C["accent"], height=3).pack(fill="x")

        # ── Notebook (tabs) ────────────────────────────────────────
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TNotebook",         background=C["bg"],    borderwidth=0)
        style.configure("TNotebook.Tab",     background=C["border"], foreground=C["text"],
                        font=("Arial", 16, "bold"), padding=[24, 6])
        style.map("TNotebook.Tab",
                  background=[("selected", C["accent"])],
                  foreground=[("selected", "white")])
        style.configure("Light.Horizontal.TProgressbar",
                        troughcolor=C["border"], background=C["accent"],
                        bordercolor=C["border"], lightcolor=C["accent"],
                        darkcolor=C["accent"])

        nb = ttk.Notebook(self)
        nb.pack(fill="both", expand=True, padx=0, pady=0)

        # ── Tab frames ─────────────────────────────────────────────
        tab_run   = tk.Frame(nb, bg=C["bg"])
        tab_guide = tk.Frame(nb, bg=C["bg"])
        tab_pred  = tk.Frame(nb, bg=C["bg"])
        nb.add(tab_run,   text="  ▶  Run  ")
        nb.add(tab_guide, text="  📋  Input Guide  ")
        nb.add(tab_pred,  text="  🔮  Predict  ")

        # ════════════════════════════════════════════════════════════
        # TAB 1 — Run  (Data Paths + Options + Run + Log)
        # ════════════════════════════════════════════════════════════
        # scrollable inside tab
        run_canvas = tk.Canvas(tab_run, bg=C["bg"], highlightthickness=0)
        run_vsb    = ttk.Scrollbar(tab_run, orient="vertical", command=run_canvas.yview)
        run_canvas.configure(yscrollcommand=run_vsb.set)
        run_vsb.pack(side="right", fill="y")
        run_canvas.pack(side="left", fill="both", expand=True)
        self._sf = tk.Frame(run_canvas, bg=C["bg"])
        self._cw = run_canvas.create_window((0, 0), window=self._sf, anchor="nw")
        self._sf.bind("<Configure>",
            lambda e: run_canvas.configure(scrollregion=run_canvas.bbox("all")))
        run_canvas.bind("<Configure>",
            lambda e: run_canvas.itemconfig(self._cw, width=e.width))
        def _wheel(event):
            if IS_WIN: run_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
            else:      run_canvas.yview_scroll(-1 if event.delta > 0 else 1, "units")
        run_canvas.bind_all("<MouseWheel>", _wheel)
        run_canvas.bind_all("<Button-4>", lambda e: run_canvas.yview_scroll(-1, "units"))
        run_canvas.bind_all("<Button-5>", lambda e: run_canvas.yview_scroll( 1, "units"))

        # ── Section: Data Paths ─────────────────────────────────────
        self._section("  Data File Paths", self._sf)
        pc = self._card(self._sf)

        self._rc_var  = tk.StringVar(value=_DEFAULT_RC_FILE)
        self._g1_var  = tk.StringVar(value=_DEFAULT_G1_FILE)
        self._g2_var  = tk.StringVar(value=_DEFAULT_G2_FILE)
        self._out_var = tk.StringVar(value=str(SCRIPT_DIR / "results_RC_SFRC_TL"))

        for i, (label, var, badge_attr, browse_fn, is_folder) in enumerate([
            ("RC beam data (.csv)",    self._rc_var,  "_rc_badge", self._browse_rc,  False),
            ("Group 1 SFRC (.csv)",    self._g1_var,  "_g1_badge", self._browse_g1,  False),
            ("Group 2 SFRC (.csv)",    self._g2_var,  "_g2_badge", self._browse_g2,  False),
            ("Output results folder",  self._out_var, None,        self._browse_out, True),
        ]):
            row = tk.Frame(pc, bg=C["card"]); row.pack(fill="x", pady=4, padx=8)
            tk.Label(row, text=label, width=22, anchor="w",
                     font=FONT_LABEL, bg=C["card"], fg=C["text_dim"]).pack(side="left")
            entry = tk.Entry(row, textvariable=var, font=FONT_MONO,
                             bg=C["entry_bg"], fg=C["entry_fg"],
                             insertbackground=C["accent"],
                             relief="flat", bd=0, highlightthickness=1,
                             highlightbackground=C["border"],
                             highlightcolor=C["accent"])
            entry.pack(side="left", fill="x", expand=True, ipady=4, padx=(0, 6))
            if badge_attr:
                badge = tk.Label(row, text="●", font=FONT_BADGE,
                                 bg=C["card"], fg=C["text_dim"], width=2)
                badge.pack(side="left", padx=(0, 4))
                setattr(self, badge_attr, badge)
            _FlatBtn(row, text="Browse…", command=browse_fn,
                     bg=C["border"], fg=C["text"], hover=C["accent_lt"],
                     font=(FA,11), padx=8, pady=2).pack(side="left")
            if is_folder:
                tk.Label(row, text="  Folder name:", font=FONT_SMALL,
                         bg=C["card"], fg=C["text_dim"]).pack(side="left", padx=(10, 2))
                self._folder_name_var = tk.StringVar(value="results_RC_SFRC_TL")
                fn_entry = tk.Entry(row, textvariable=self._folder_name_var,
                                    width=22, font=FONT_MONO,
                                    bg=C["entry_bg"], fg=C["entry_fg"],
                                    insertbackground=C["accent"],
                                    relief="flat", bd=0, highlightthickness=1,
                                    highlightbackground=C["border"],
                                    highlightcolor=C["accent"])
                fn_entry.pack(side="left", ipady=4)
                self._folder_name_var.trace_add("write", lambda *a: self._sync_out_path())
            var.trace_add("write", lambda *a: self._check_files())

        # ── Section: Analysis Options ───────────────────────────────
        self._section("  Analysis Options", self._sf)
        oc = self._card(self._sf)

        sf_row = tk.Frame(oc, bg=C["card"]); sf_row.pack(fill="x", padx=10, pady=(8, 4))
        tk.Label(sf_row, text="SFRC target domain:", font=FONT_LABEL,
                 bg=C["card"], fg=C["text"], width=22, anchor="w").pack(side="left")
        self._use_g1 = tk.BooleanVar(value=True)
        self._use_g2 = tk.BooleanVar(value=True)
        self._chk(sf_row, "Group 1", self._use_g1, C["accent"])
        self._chk(sf_row, "Group 2", self._use_g2, C["accent2"])

        _sep(oc, color=C["border"]).pack(fill="x", padx=10, pady=6)

        rc_row = tk.Frame(oc, bg=C["card"]); rc_row.pack(fill="x", padx=10, pady=(4, 4))
        tk.Label(rc_row, text="RC source domain:", font=FONT_LABEL,
                 bg=C["card"], fg=C["text"], width=22, anchor="w").pack(side="left")
        self._use_rc = tk.BooleanVar(value=True)
        self._chk(rc_row, "Use RC data for Heterogeneous Transfer Learning (recommended)", self._use_rc, C["warn"])

        _sep(oc, color=C["border"]).pack(fill="x", padx=10, pady=6)

        ex_row = tk.Frame(oc, bg=C["card"]); ex_row.pack(fill="x", padx=10, pady=(4, 10))
        tk.Label(ex_row, text="Outputs:", font=FONT_LABEL,
                 bg=C["card"], fg=C["text"], width=22, anchor="w").pack(side="left")
        self._do_g1vsg2 = tk.BooleanVar(value=True)
        self._chk(ex_row, "Group 1 vs. Group 2 comparison CSV", self._do_g1vsg2, C["accent2"])

        # ── Section: Run ────────────────────────────────────────────
        self._section("  Run", self._sf)
        rc = self._card(self._sf)

        btn_row = tk.Frame(rc, bg=C["card"]); btn_row.pack(pady=(10, 6))
        self._run_btn = _FlatBtn(btn_row, text="▶   Start Analysis",
                                 command=self._start_run,
                                 font=("Arial", 16, "bold"), padx=30, pady=11)
        self._run_btn.pack(side="left", padx=(0, 8))
        self._stop_btn = _FlatBtn(btn_row, text="■   Stop",
                                  command=self._stop_run,
                                  bg=C["danger"], fg=C["btn_fg"], hover="#B91C1C",
                                  font=("Arial", 16, "bold"), padx=20, pady=11)
        self._stop_btn.pack(side="left", padx=(0, 8))
        self._stop_btn.enable(False)
        self._open_btn = _FlatBtn(btn_row, text="📂  Open Results Folder",
                                  command=self._open_results,
                                  bg=C["border"], fg=C["text"], hover=C["accent_lt"],
                                  font=FONT_SMALL, padx=14, pady=11)
        self._open_btn.pack(side="left")
        self._open_btn.enable(False)

        prog_row = tk.Frame(rc, bg=C["card"]); prog_row.pack(fill="x", padx=16, pady=6)
        self._prog_var = tk.DoubleVar(value=0)
        ttk.Progressbar(prog_row, variable=self._prog_var, maximum=100,
                        length=400, style="Light.Horizontal.TProgressbar"
                        ).pack(side="left", fill="x", expand=True)
        self._prog_lbl = tk.Label(prog_row, text="0 %", font=FONT_SMALL,
                                  bg=C["card"], fg=C["text_dim"], width=7)
        self._prog_lbl.pack(side="left", padx=6)

        self._status_lbl = tk.Label(rc, text="Ready", font=FONT_LABEL,
                                    bg=C["card"], fg=C["text_dim"])
        self._status_lbl.pack(pady=(0, 4))

        _sep(rc, color=C["border"]).pack(fill="x", padx=12, pady=4)

        log_hdr = tk.Frame(rc, bg=C["card"]); log_hdr.pack(fill="x", padx=12, pady=(4, 2))
        tk.Label(log_hdr, text="Execution Log", font=("Arial", 16),
                 bg=C["card"], fg=C["text_dim"]).pack(side="left")
        for col, lbl in [(C["ok"],"Done"),(C["warn"],"Running"),(C["danger"],"Error")]:
            tk.Label(log_hdr, text="●", fg=col, bg=C["card"],
                     font=FONT_SMALL).pack(side="right", padx=(0,2))
            tk.Label(log_hdr, text=lbl, fg=C["text_dim"], bg=C["card"],
                     font=FONT_SMALL).pack(side="right")

        self._log_box = tk.scrolledtext.ScrolledText(
            rc, height=14, font=FONT_MONO,
            bg=C["log_bg"], fg=C["log_fg"],
            insertbackground=C["log_fg"],
            relief="flat", bd=0, wrap="word", state="disabled")
        self._log_box.pack(fill="both", expand=True, padx=8, pady=(2, 10))

        # ════════════════════════════════════════════════════════════
        # TAB 2 — Input Guide  (RC + Group 1 + Group 2 in one screen)
        # ════════════════════════════════════════════════════════════
        # Compact but publication-readable layout: three equal-width cards.
        FG  = (FA, 17, "bold")   # group header
        FH  = (FA, 12, "bold")   # column header
        FB  = (FA, 12)           # body
        FI  = (FA, 12, "italic") # optional tau

        def _col_block(parent, title, color, rows, col):
            """One compact guide card; fixed text widths keep all 3 cards visible."""
            frm = tk.Frame(parent, bg=C["card"], highlightthickness=1,
                           highlightbackground=C["border"])
            frm.grid(row=0, column=col, sticky="nsew", padx=5, pady=4)
            parent.grid_columnconfigure(col, weight=1, uniform="guide_cols")

            tk.Label(frm, text=title, font=FG, bg=color, fg="white",
                     anchor="w", padx=10, pady=8).pack(fill="x")

            # Column widths are intentionally small so RC, Group 1, and Group 2
            # fit within a 1366-px screenshot without horizontal scrolling.
            colspec = [
                ("Column", 13),
                ("Symbol", 7),
                ("Unit", 5),
                ("Description", 18),
            ]

            hdr = tk.Frame(frm, bg="#E8EEF7")
            hdr.pack(fill="x")
            for txt, w in colspec:
                tk.Label(hdr, text=txt, width=w, font=FH, bg="#E8EEF7",
                         fg=C["text"], anchor="w", padx=4, pady=6).pack(side="left")

            for i, (feat, sym, unit, desc) in enumerate(rows):
                is_opt = feat == "tau_u_MPa"
                bg  = "#FFF7E6" if is_opt else ("#FFFFFF" if i % 2 == 0 else "#F3F6FB")
                fg  = C["warn"] if is_opt else C["text"]
                fnt = FI if is_opt else FB
                row = tk.Frame(frm, bg=bg)
                row.pack(fill="x")
                vals = [(feat, 13), (sym, 7), (unit, 5), (desc, 18)]
                for txt, w in vals:
                    tk.Label(row, text=txt, width=w, font=fnt, bg=bg, fg=fg,
                             anchor="w", padx=4, pady=5).pack(side="left")
            tk.Frame(frm, bg=C["card"]).pack(fill="both", expand=True)

        ig_outer = tk.Frame(tab_guide, bg=C["bg"])
        ig_outer.pack(fill="both", expand=True, padx=8, pady=7)
        ig_outer.grid_rowconfigure(0, weight=1)
        ig_outer.grid_columnconfigure(0, weight=1)

        cols_frame = tk.Frame(ig_outer, bg=C["bg"])
        cols_frame.pack(fill="both", expand=True)
        cols_frame.grid_rowconfigure(0, weight=1)

        _col_block(cols_frame,
            "RC Source Domain",
            C["accent"],
            [
                ("b_mm",      "b",    "mm",  "Beam width"),
                ("d_mm",      "d",    "mm",  "Effective depth"),
                ("a_d",       "a/d",  "—",   "Shear span ratio"),
                ("fc_MPa",    "f'c",  "MPa", "Comp. strength"),
                ("rho",       "ρ",    "%",   "Reinf. ratio"),
                ("V_u_KN",    "Vu",   "kN",  "Shear force"),
                ("tau_u_MPa", "τ_u",  "MPa", "Auto-computable"),
            ], 0)

        _col_block(cols_frame,
            "Group 1 — SFRC",
            C["accent2"],
            [
                ("b_mm",      "b",     "mm",  "Beam width"),
                ("d_mm",      "d",     "mm",  "Effective depth"),
                ("a_d",       "a/d",   "—",   "Shear span ratio"),
                ("fc_MPa",    "f'c",   "MPa", "Comp. strength"),
                ("rho",       "ρ",     "%",   "Reinf. ratio"),
                ("V_f_pct",   "Vf",    "%",   "Fiber volume"),
                ("RI",        "RI",    "—",   "Reinf. index"),
                ("Lf_per_Df", "Lf/Df", "—",   "Aspect ratio"),
                ("V_u_KN",    "Vu",    "kN",  "Shear force"),
                ("tau_u_MPa", "τ_u",   "MPa", "Auto-computable"),
            ], 1)

        _col_block(cols_frame,
            "Group 2 — SFRC",
            "#7C3AED",
            [
                ("b_mm",          "b",       "mm",  "Beam width"),
                ("d_mm",          "d",       "mm",  "Effective depth"),
                ("a_d",           "a/d",     "—",   "Shear span ratio"),
                ("fc_MPa",        "f'c",     "MPa", "Comp. strength"),
                ("rho",           "ρ",       "%",   "Reinf. ratio"),
                ("V_f_pct",       "Vf",      "%",   "Fiber volume"),
                ("RI",            "RI",      "—",   "Reinf. index"),
                ("Lf_per_Df",     "Lf/Df",   "—",   "Aspect ratio"),
                ("fsp_MPa",       "f_sp",    "MPa", "Splitting tensile"),
                ("ft_direct_MPa", "f_t,dir", "MPa", "Direct tensile"),
                ("fr_MPa",        "f_r",     "MPa", "Rupture modulus"),
                ("V_u_KN",        "Vu",      "kN",  "Shear force"),
                ("tau_u_MPa",     "τ_u",     "MPa", "Auto-computable"),
            ], 2)

        tk.Label(ig_outer,
                 text="  Note: Amber italic τ_u is optional and is auto-computed as V_u × 1000 / (b × d) when omitted.",
                 font=(FA, 13, "italic"), bg="#FFF7E6", fg=C["warn"],
                 anchor="w", padx=8, pady=6).pack(fill="x", pady=(7, 0))

        # ════════════════════════════════════════════════════════════
        # TAB 3 — Predict  (Group1 or Group2 → load model → predict)
        # ════════════════════════════════════════════════════════════

        # Compact scrollable Predict tab: designed to fit Step 1–4 in one screenshot.
        pr_cv  = tk.Canvas(tab_pred, bg=C["bg"], highlightthickness=0)
        pr_vsb = ttk.Scrollbar(tab_pred, orient="vertical", command=pr_cv.yview)
        pr_cv.configure(yscrollcommand=pr_vsb.set)
        pr_vsb.pack(side="right", fill="y")
        pr_cv.pack(side="left", fill="both", expand=True)
        pr_sf = tk.Frame(pr_cv, bg=C["bg"])
        pr_cw = pr_cv.create_window((0,0), window=pr_sf, anchor="nw")
        pr_sf.bind("<Configure>", lambda e: pr_cv.configure(scrollregion=pr_cv.bbox("all")))
        pr_cv.bind("<Configure>", lambda e: pr_cv.itemconfig(pr_cw, width=e.width, height=e.height))

        # ── Step 1 + Step 2 in one row ───────────────────────────
        top_row = tk.Frame(pr_sf, bg=C["bg"])
        top_row.pack(fill="x", padx=10, pady=(5, 3))
        top_row.columnconfigure(0, weight=0, minsize=330)
        top_row.columnconfigure(1, weight=1)

        # Step 1: Group selector
        gc = tk.Frame(top_row, bg=C["card"], padx=8, pady=5,
                      highlightbackground=C["border"], highlightthickness=1)
        gc.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        tk.Label(gc, text="Step 1 — Select Group", font=(FA,17,"bold"),
                 bg=C["card"], fg=C["accent"], anchor="w").pack(fill="x")
        gr_row = tk.Frame(gc, bg=C["card"]); gr_row.pack(fill="x", pady=(2, 1))
        tk.Label(gr_row, text="Group:", font=(FA,15,"bold"),
                 bg=C["card"], fg=C["text"]).pack(side="left")
        self._pred_grp_var = tk.StringVar(value="Group 1")
        for gname, gcol in [("Group 1", C["accent2"]), ("Group 2", "#7C3AED")]:
            tk.Radiobutton(gr_row, text=gname,
                           variable=self._pred_grp_var, value=gname,
                           font=(FA,15), bg=C["card"], fg=C["text"],
                           activebackground=C["card"], selectcolor=C["card"],
                           highlightthickness=0, cursor="hand2",
                           command=self._pred_on_group_change
                           ).pack(side="left", padx=(12,0))
        self._pred_grp_desc = tk.Label(gc,
            text="Group 1: RC common + fiber index",
            font=(FA,12), bg=C["accent_lt"], fg=C["accent"],
            anchor="w", padx=7, pady=2)
        self._pred_grp_desc.pack(fill="x", pady=(1,0))

        # Step 2: Model file loader
        mc2 = tk.Frame(top_row, bg=C["card"], padx=8, pady=5,
                       highlightbackground=C["border"], highlightthickness=1)
        mc2.grid(row=0, column=1, sticky="nsew", padx=(6, 0))
        tk.Label(mc2, text="Step 2 — Load Saved Model", font=(FA,17,"bold"),
                 bg=C["card"], fg=C["accent"], anchor="w").pack(fill="x")
        tk.Label(mc2,
            text="Path: results_root / 01_HeteroTL / {Group} / 06_saved_model",
            font=(FA,11), bg=C["card"], fg=C["text_dim"],
            anchor="w").pack(fill="x", pady=(1,0))

        self._pred_vu_var  = tk.StringVar()
        self._pred_tau_var = tk.StringVar()
        model_grid = tk.Frame(mc2, bg=C["card"])
        model_grid.pack(fill="x", pady=(1,0))
        model_grid.columnconfigure(1, weight=1)
        for r, (lbl_txt, var) in enumerate([
            ("V_u model:", self._pred_vu_var),
            ("τ_u model:", self._pred_tau_var),
        ]):
            tk.Label(model_grid, text=lbl_txt, font=(FA,12), bg=C["card"],
                     fg=C["text_dim"], width=10, anchor="w").grid(row=r, column=0, sticky="w", pady=1)
            tk.Entry(model_grid, textvariable=var, font=("Consolas",11),
                     bg=C["entry_bg"], fg=C["entry_fg"],
                     insertbackground=C["accent"], relief="flat", bd=0,
                     highlightthickness=1, highlightbackground=C["border"],
                     highlightcolor=C["accent"]
                     ).grid(row=r, column=1, sticky="ew", ipady=1, padx=(0,4), pady=1)
            _FlatBtn(model_grid, text="Browse…",
                     command=lambda v=var: self._browse_model(v),
                     bg=C["border"], fg=C["text"], hover=C["accent_lt"],
                     font=(FA,11), padx=8, pady=1).grid(row=r, column=2, sticky="e", pady=1)

        load_row2 = tk.Frame(mc2, bg=C["card"]); load_row2.pack(fill="x", pady=(1,0))
        _FlatBtn(load_row2, text="✓  Load Models",
                 command=self._pred_load_models,
                 bg=C["accent2"], fg=C["btn_fg"], hover="#047857",
                 font=(FA,13,"bold"), padx=10, pady=2).pack(side="left")
        self._pred_load_status = tk.Label(load_row2, text="  No model loaded.",
                                          font=(FA,11), bg=C["card"], fg=C["text_dim"])
        self._pred_load_status.pack(side="left", padx=8)

        # ── Step 3: Input fields ─────────────────────────────────
        self._pred_inp_card = tk.Frame(pr_sf, bg=C["card"], padx=6, pady=5,
                                       highlightbackground=C["border"], highlightthickness=1)
        self._pred_inp_card.pack(fill="x", padx=10, pady=3)
        self._pred_field_vars = {}
        self._pred_build_fields()

        # ── Step 4: Predict + Result ─────────────────────────────
        res_card = tk.Frame(pr_sf, bg=C["card"], padx=6, pady=5,
                            highlightbackground=C["border"], highlightthickness=1)
        res_card.pack(fill="both", expand=True, padx=10, pady=(3, 8))
        tk.Label(res_card, text="Step 4 — Predict", font=(FA,17,"bold"),
                 bg=C["card"], fg=C["accent"], anchor="w").pack(fill="x")
        _FlatBtn(res_card, text="🔮  Predict  Vu  &  τu",
                 command=self._pred_run,
                 font=(FA,14,"bold"), padx=18, pady=2).pack(pady=(1,2))

        res_boxes = tk.Frame(res_card, bg=C["card"])
        res_boxes.pack(fill="both", expand=True, padx=32, pady=(2,8))
        self._pred_res_lbl = {}
        for rkey, rlbl, rcol in [
            ("Vu",  "V_u  (kN)",  C["accent"]),
            ("tau", "τ_u  (MPa)", C["accent2"]),
        ]:
            box = tk.Frame(res_boxes, bg=C["accent_lt"],
                           highlightbackground=rcol, highlightthickness=2)
            box.pack(side="left", fill="both", expand=True, padx=14, pady=4)
            tk.Label(box, text=rlbl, font=(FA,15,"bold"),
                     bg=C["accent_lt"], fg=rcol).pack(pady=(8,0))
            val = tk.Label(box, text="—", font=(FA,22,"bold"),
                           bg=C["accent_lt"], fg=C["text"])
            val.pack(pady=(4,10))
            self._pred_res_lbl[rkey] = val

        self._pred_note_lbl = tk.Label(res_card, text="",
                                       font=(FA,11), bg=C["card"], fg=C["text_dim"],
                                       wraplength=900, justify="left")
        self._pred_note_lbl.pack(fill="x", padx=12, pady=(0,1))


    # ── Helper widgets ──────────────────────────────────────────────
    def _section(self, title, parent):
        f = tk.Frame(parent, bg=C["bg"]); f.pack(fill="x", padx=14, pady=(6, 2))
        inner = tk.Frame(f, bg=C["bg"]); inner.pack(fill="x")
        tk.Frame(inner, bg=C["accent"], width=5, height=22).pack(side="left", padx=(0, 8), pady=(2, 0))
        tk.Label(inner, text=title.strip(), font=("Arial", 18, "bold"),
                 bg=C["bg"], fg=C["accent"]).pack(side="left")
        _sep(f, color=C["border"]).pack(fill="x", pady=(3, 0))

    def _card(self, parent):
        f = tk.Frame(parent, bg=C["card"], padx=6, pady=5,
                     highlightbackground=C["border"], highlightthickness=1)
        f.pack(fill="x", padx=14, pady=2)
        return f

    def _chk(self, parent, text, var, color):
        tk.Checkbutton(parent, text=text, variable=var,
                       font=("Arial", 18),
                       bg=parent.cget("bg"), fg=C["text"],
                       activebackground=parent.cget("bg"), activeforeground=color,
                       selectcolor=C["card"], cursor="hand2",
                       highlightthickness=0
                       ).pack(side="left", padx=(0, 20))

    # ── Path sync ───────────────────────────────────────────────────
    def _sync_out_path(self):
        """Update out_var when folder name changes (keeps parent dir)."""
        parent = str(Path(self._out_var.get()).parent)
        name   = self._folder_name_var.get().strip() or "results"
        self._out_var.set(str(Path(parent) / name))

    # ── Browse handlers ─────────────────────────────────────────────
    def _browse_rc(self):
        p = filedialog.askopenfilename(title="Select RC beam CSV",
                                       filetypes=[("CSV","*.csv"),("All","*.*")])
        if p: self._rc_var.set(p)

    def _browse_g1(self):
        p = filedialog.askopenfilename(title="Select Group 1 SFRC CSV",
                                       filetypes=[("CSV","*.csv"),("All","*.*")])
        if p: self._g1_var.set(p)

    def _browse_g2(self):
        p = filedialog.askopenfilename(title="Select Group 2 SFRC CSV",
                                       filetypes=[("CSV","*.csv"),("All","*.*")])
        if p: self._g2_var.set(p)

    def _browse_out(self):
        """Pick a parent directory; folder name is appended from the name field."""
        p = filedialog.askdirectory(title="Select parent directory for results")
        if p:
            name = self._folder_name_var.get().strip() or "results_RC_SFRC_TL"
            self._out_var.set(str(Path(p) / name))

    # ── File status badges ──────────────────────────────────────────
    def _check_files(self):
        for var, attr in [(self._rc_var,"_rc_badge"),
                          (self._g1_var,"_g1_badge"),
                          (self._g2_var,"_g2_badge")]:
            badge = getattr(self, attr, None)
            if badge is None: continue
            exists = Path(var.get()).exists()
            badge.config(fg=C["ok"] if exists else C["danger"],
                         text="✓" if exists else "✗")

    # ── Log helpers ─────────────────────────────────────────────────
    def _log(self, msg):
        def _do():
            self._log_box.config(state="normal")
            self._log_box.insert("end", msg + "\n")
            self._log_box.see("end")
            self._log_box.config(state="disabled")
        self.after(0, _do)

    def _set_status(self, msg, color=None):
        self.after(0, lambda:
            self._status_lbl.config(text=msg, fg=(color or C["text_dim"])))

    def _set_progress(self, pct):
        def _do():
            self._prog_var.set(pct)
            self._prog_lbl.config(text=f"{pct:.0f} %")
        self.after(0, _do)

    # ── Validation ──────────────────────────────────────────────────
    def _validate(self):
        use_g1 = self._use_g1.get(); use_g2 = self._use_g2.get()
        use_rc = self._use_rc.get()
        if not (use_g1 or use_g2):
            messagebox.showerror("Input Error", "Select at least one SFRC dataset.")
            return False
        if use_g1 and not Path(self._g1_var.get()).exists():
            messagebox.showerror("File Not Found", f"Group 1: {self._g1_var.get()}")
            return False
        if use_g2 and not Path(self._g2_var.get()).exists():
            messagebox.showerror("File Not Found", f"Group 2: {self._g2_var.get()}")
            return False
        if use_rc and not Path(self._rc_var.get()).exists():
            ans = messagebox.askyesno("RC File Not Found",
                "RC file not found.\nContinue without RC data (Target-Only mode)?")
            if not ans: return False
            self._use_rc.set(False)
        return True

    # ── Run / Stop ──────────────────────────────────────────────────
    def _start_run(self):
        if not self._validate(): return
        self._running = True
        self._run_btn.enable(False)
        self._stop_btn.enable(True)
        self._open_btn.enable(False)
        self._log_box.config(state="normal"); self._log_box.delete("1.0", "end")
        self._log_box.config(state="disabled")
        self._set_progress(0)
        self._set_status("Running…", C["running"])
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def _stop_run(self):
        self._running = False
        self._log("⚠  Stop requested — finishing current step...")
        self._set_status("Stop requested…", C["warn"])

    def _open_results(self):
        p = self._out_var.get()
        if IS_WIN: os.startfile(p)
        else:      os.system(f'open "{p}"')

    # ── Worker ──────────────────────────────────────────────────────
    def _worker(self):
        t0 = time.time()
        try:
            use_rc = self._use_rc.get()
            use_g1 = self._use_g1.get()
            use_g2 = self._use_g2.get()
            tag    = "withRC" if use_rc else "noRC"

            run_stamp    = datetime.now().strftime("%Y%m%d_%H%M%S")
            folder_name  = self._folder_name_var.get().strip() or "results_RC_SFRC_TL"
            out_base     = Path(self._out_var.get()).parent
            results_root = out_base / f"{folder_name}_{tag}_{run_stamp}"
            results_root.mkdir(parents=True, exist_ok=True)

            # Update out_var to reflect actual path used
            self.after(0, lambda: self._out_var.set(str(results_root)))

            # Algorithm docs copy
            _doc_src = SCRIPT_DIR / 'algorithm_docs'
            if _doc_src.exists():
                import shutil as _shutil
                try:
                    _shutil.copytree(str(_doc_src), str(results_root / 'algorithm_docs'),
                                     dirs_exist_ok=True)
                    self._log(f"  [Docs] algorithm_docs copied")
                except Exception as _de:
                    self._log(f"  [Docs WARN] {_de}")

            ml_models  = ML_MODELS_WITH_RC if use_rc else ML_MODELS_NO_RC
            lit_models = []   # 경험식은 계산 코드 유지, 결과 CSV/fold에서 제외

            self._log("=" * 62)
            self._log("  RC → SFRC  Heterogeneous Transfer Learning")
            self._log(f"  SFRC groups : {'Group1 ' if use_g1 else ''}{'Group2' if use_g2 else ''}")
            self._log(f"  RC domain   : {'used (HeteroTL)' if use_rc else 'not used (Target-Only)'}")
            self._log(f"  Output      : {results_root}")
            self._log("=" * 62)

            # Load data
            self._log("\n[1] Loading data...")
            if use_rc:
                df_rc = harmonize_rc(pd.read_csv(self._rc_var.get()))
                self._log(f"  RC     : {df_rc.shape[0]} rows × {df_rc.shape[1]} cols")
            else:
                df_rc = pd.DataFrame(columns=BASE_COMMON + ['V_u_KN', 'tau_u_MPa'])

            sfrc_data = {}
            if use_g1:
                sfrc_data['Group1'] = harmonize_sfrc(pd.read_csv(self._g1_var.get()))
                self._log(f"  Group1 : {sfrc_data['Group1'].shape[0]} rows × {sfrc_data['Group1'].shape[1]} cols")
            if use_g2:
                sfrc_data['Group2'] = harmonize_sfrc(pd.read_csv(self._g2_var.get()))
                self._log(f"  Group2 : {sfrc_data['Group2'].shape[0]} rows × {sfrc_data['Group2'].shape[1]} cols")

            if not self._running:
                self._finish(t0, cancelled=True); return

            EXPERIMENTS = {}
            if 'Group1' in sfrc_data:
                EXPERIMENTS['Group1'] = dict(
                    df_target=sfrc_data['Group1'],
                    extra_cands=GROUP1_EXTRA, drop_feats=['L_f','D_f'],
                    must_keep=['Lf_per_Df'],
                    results_dir=results_root / '01_HeteroTL' / 'Group1')
            if 'Group2' in sfrc_data:
                EXPERIMENTS['Group2'] = dict(
                    df_target=sfrc_data['Group2'],
                    extra_cands=GROUP2_EXTRA, drop_feats=['L_f','D_f'],
                    must_keep=['Lf_per_Df'],
                    results_dir=results_root / '01_HeteroTL' / 'Group2')

            self._log(f"\n[2] Starting analysis "
                      f"({N_SPLITS}×{N_REPEATS}={N_SPLITS*N_REPEATS} splits "
                      f"× {len(AVAILABILITY_RATIOS)} ratios)\n")

            n_exp = len(EXPERIMENTS); all_arts = {}
            for ei, (exp_name, cfg) in enumerate(EXPERIMENTS.items()):
                if not self._running:
                    self._finish(t0, cancelled=True); return
                self._log(f"\n{'#'*52}\n#  {exp_name}\n{'#'*52}")
                self._set_status(f"Running: {exp_name}", C["running"])
                arts = run_experiment(
                    exp_name, cfg, df_rc, use_rc, ml_models, lit_models,
                    self._log, results_root=results_root)
                all_arts[exp_name] = arts
                self._set_progress((ei + 1) / n_exp * 85)

            self._log("\n[3] Generating workflow outputs (CSV + AP plots)...")
            self._set_status("Creating outputs…", C["running"])
            try:
                wf_dir = run_workflow(all_arts, results_root, ml_models, use_rc, self._log)
                self._log(f"  → {wf_dir}")
            except Exception:
                self._log("\n  [Workflow ERROR]\n" + traceback.format_exc())

            # Global summary CSV
            self._log("\n[4] Saving global summary CSV...")
            rows = []
            for exp_name, arts in all_arts.items():
                for tn, art in arts.items():
                    s = art['summary'].copy(); s['experiment'] = exp_name
                    rows.append(s)
            if rows:
                gdf = pd.concat(rows, ignore_index=True)
                p   = results_root / f'global_summary_{tag}.csv'
                safe_csv(gdf, p); self._log(f"  → {p}")

            self._set_progress(100)
            self._finish(t0, cancelled=False, results_root=results_root)

        except Exception:
            tb = traceback.format_exc()
            self._log("\n[FATAL ERROR]\n" + tb)
            self._set_status("Error occurred", C["danger"])
            self.after(0, lambda: self._run_btn.enable(True))
            self.after(0, lambda: self._stop_btn.enable(False))

    # ── Predict tab helpers ─────────────────────────────────────────
    _PRED_FIELDS_COMMON = [
        ("b_mm",          "b  —  Beam width",          "mm",  "200",  True),
        ("d_mm",          "d  —  Effective depth",      "mm",  "350",  True),
        ("a_d",           "a/d  —  Shear span ratio",   "—",   "3.0",  True),
        ("fc_MPa",        "f'c  —  Comp. strength",    "MPa", "40.0", True),
        ("rho",           "rho  —  Reinf. ratio",       "%",   "2.0",  True),
    ]
    _PRED_FIELDS_FIBER = [
        ("V_f_pct",    "Vf  —  Fiber vol. fraction",   "%",  "1.0",  True),
        ("RI",         "RI  —  Reinforcing index",      "—",  "50.0", True),
        ("Lf_per_Df",  "Lf/Df  —  Fiber aspect ratio", "—",  "65.0", True),
    ]
    _PRED_FIELDS_TENSILE = [
        ("fsp_MPa",       "f_sp  —  Splitting tensile",  "MPa", "", True),
        ("ft_direct_MPa", "f_t,dir  —  Direct tensile",  "MPa", "", True),
        ("fr_MPa",        "f_r  —  Modulus of rupture",  "MPa", "", True),
    ]

    def _pred_fields_for_group(self):
        grp = self._pred_grp_var.get()
        base = self._PRED_FIELDS_COMMON + self._PRED_FIELDS_FIBER
        if grp == "Group 2":
            return base + self._PRED_FIELDS_TENSILE
        return base

    def _pred_build_fields(self):
        """Rebuild input grid based on selected group."""
        for w in self._pred_inp_card.winfo_children():
            w.destroy()
        self._pred_field_vars.clear()

        grp    = self._pred_grp_var.get()
        fields = self._pred_fields_for_group()
        color  = C["accent2"] if grp == "Group 1" else "#7C3AED"

        tk.Label(self._pred_inp_card,
                 text=f"Step 3 — Enter Parameters    |    {grp} input features",
                 font=("Arial",16,"bold"), bg=color, fg="white",
                 anchor="w", padx=10, pady=3).pack(fill="x", padx=4, pady=(0,2))

        grid_f = tk.Frame(self._pred_inp_card, bg=C["card"])
        grid_f.pack(fill="x", padx=6, pady=0)

        ncols = 3
        for c in range(ncols):
            grid_f.columnconfigure(c, weight=1)
        for idx, (key, lbl, unit, default, required) in enumerate(fields):
            var = tk.StringVar(value=default)
            self._pred_field_vars[key] = var
            col  = idx % ncols
            cell = tk.Frame(grid_f, bg=C["card"])
            cell.grid(row=idx//ncols, column=col, sticky="ew", padx=7, pady=1)
            # Do not show required/optional markers in Predict input labels.
            # Validation below still enforces all fields needed by the selected group.
            fg_l = C["text"] if required else C["text_dim"]
            lbl_row = tk.Frame(cell, bg=C["card"])
            lbl_row.pack(fill="x")
            tk.Label(lbl_row, text=f"{lbl}  ({unit})",
                     font=("Arial",13), bg=C["card"], fg=fg_l,
                     anchor="w").pack(side="left", fill="x", expand=True)
            if key == "a_d":
                tk.Label(lbl_row, text="Enter ≥ 2",
                         font=("Arial",11,"bold"), bg=C["card"], fg=C["danger"],
                         anchor="e").pack(side="right")
            ent = tk.Entry(cell, textvariable=var, font=("Arial",14),
                           bg=C["entry_bg"], fg=C["entry_fg"],
                           insertbackground=C["accent"],
                           relief="flat", bd=0, highlightthickness=1,
                           highlightbackground=C["border"],
                           highlightcolor=C["accent"])
            ent.pack(fill="x", ipady=2)
            if key == "a_d":
                def _ad_focusout(_event=None, _v=var):
                    txt = _v.get().strip()
                    if not txt:
                        return
                    try:
                        val = float(txt)
                    except Exception:
                        return
                    if val < 2.0:
                        _v.set("2.0")
                        messagebox.showwarning("Input Check", "a/d must be 2.0 or higher.")
                ent.bind("<FocusOut>", _ad_focusout)

        note_txt = (
            "  Group 2 uses f_sp, f_t,dir, and f_r as input features; enter numeric values for prediction."
            if grp == "Group 2"
            else "  Enter numeric values for the selected group features."
        )
        tk.Label(self._pred_inp_card,
                 text=note_txt,
                 font=("Arial",11,"italic"), bg=C["card"], fg=C["text_dim"],
                 anchor="w").pack(fill="x", padx=8, pady=(2,0))

    def _pred_on_group_change(self):
        grp = self._pred_grp_var.get()
        if grp == "Group 1":
            desc = "Group 1: RC common + fiber index"
        else:
            desc = "Group 2: RC common + fiber index + tensile strength"
        self._pred_grp_desc.config(text=desc)
        self._pred_build_fields()
        for lbl in self._pred_res_lbl.values():
            lbl.config(text="—", fg=C["text"])
        self._pred_note_lbl.config(text="")

    def _browse_model(self, var):
        p = filedialog.askopenfilename(
            title="Select saved model (.pkl)",
            filetypes=[("Pickle","*.pkl"),("All","*.*")])
        if p: var.set(p)

    def _pred_load_models(self):
        """
        Load saved model bundles for prediction.

        Supports TWO save formats produced by run_experiment():

        Format A — dict bundle (model_V_u_KN.pkl / model_tau_u_MPa.pkl):
            Saved by the final "Save final model" block at the end of
            run_experiment().  Contains keys: 'model', 'src_model',
            'imp_tgt', 'all_tft', 'common_feats', 'common_idx',
            'feat_names', 'target_name', 'exp_name', 'use_rc', 'model_name'.

        Format B — individual files (htl_model_*.pkl + source_model_*.pkl
                   + imputer_*.pkl), saved by the SHAP loop at ratio=0.90.
            When the user browses to htl_model_V_u_KN.pkl, we auto-discover
            source_model_V_u_KN.pkl and imputer_V_u_KN.pkl from the same
            folder and also read meta_V_u_KN.json for feature metadata.

        Both formats are normalised into the same dict-bundle structure so
        _pred_run() needs no format-awareness.
        """
        try:
            import joblib as _jl
        except ImportError:
            messagebox.showerror("Error", "pip install joblib"); return

        import json as _json

        def _load_bundle(p, name):
            """Return a normalised bundle dict regardless of save format."""
            p = Path(p)
            obj = _jl.load(str(p))

            # ── Format A: already a dict bundle ──────────────────────
            if isinstance(obj, dict) and "model" in obj:
                return obj

            # ── Format B: individual model file ──────────────────────
            # obj is the HTL model itself (XGBRegressor / ExtraTrees …)
            folder = p.parent
            tgt    = "V_u_KN" if name == "V_u" else "tau_u_MPa"

            # Try to load companion files
            src_path  = folder / f"source_model_{tgt}.pkl"
            imp_path  = folder / f"imputer_{tgt}.pkl"
            meta_path = folder / f"meta_{tgt}.json"

            src_mdl = _jl.load(str(src_path)) if src_path.exists() else None
            imp_tgt = _jl.load(str(imp_path)) if imp_path.exists() else None

            # Feature metadata from JSON
            all_tft = []; common_feats = []; feat_names = []; use_rc_b = False
            if meta_path.exists():
                try:
                    meta = _json.loads(meta_path.read_text(encoding="utf-8"))
                    all_tft      = meta.get("all_tft", [])
                    common_feats = meta.get("common_feats", [])
                    feat_names   = meta.get("feat_names_full",
                                   meta.get("feat_names", []))
                    use_rc_b     = (src_mdl is not None and
                                    len(common_feats) > 0)
                except Exception:
                    pass

            # Fallback: derive feature list from imputer if JSON is missing
            if not all_tft and imp_tgt is not None:
                try:
                    n_feats = imp_tgt.statistics_.shape[0]
                    # Build from GROUP2_EXTRA (superset) as best-effort
                    cands = BASE_COMMON + GROUP2_EXTRA
                    # Filter to the right count — use what the imputer was fit on
                    all_tft = cands[:n_feats]
                except Exception:
                    pass

            # Derive common_idx
            common_idx = []
            if common_feats and all_tft:
                common_idx = [all_tft.index(c) for c in common_feats
                              if c in all_tft]

            if feat_names and all_tft:
                # feat_names_full = ['source_pred'] + all_tft  (with RC)
                # or just all_tft  (without RC)
                pass  # already set from JSON

            bundle = {
                "model":        obj,
                "src_model":    src_mdl,
                "imp_tgt":      imp_tgt,
                "all_tft":      all_tft,
                "common_feats": common_feats,
                "common_idx":   common_idx,
                "feat_names":   feat_names,
                "target_name":  tgt,
                "use_rc":       use_rc_b,
                "model_name":   "Hetero_TL_XGB",
                "_fmt":         "B",
            }
            return bundle

        loaded = []
        for name, var in [("V_u", self._pred_vu_var),
                          ("tau", self._pred_tau_var)]:
            p = var.get().strip()
            tgt = "V_u_KN" if name == "V_u" else "tau_u_MPa"

            if not p:
                # Auto-discover: try both save formats
                results_root = Path(self._out_var.get())
                grp_key = "Group1" if self._pred_grp_var.get() == "Group 1" else "Group2"
                mdl_dir  = results_root / "01_HeteroTL" / grp_key / "06_saved_model"
                # Prefer dict-bundle (Format A), then individual (Format B)
                for cname in [f"model_{tgt}.pkl", f"htl_model_{tgt}.pkl"]:
                    candidate = mdl_dir / cname
                    if candidate.exists():
                        p = str(candidate)
                        var.set(p)
                        break

            if p and Path(p).exists():
                try:
                    bundle = _load_bundle(p, name)
                    if bundle.get("imp_tgt") is None:
                        messagebox.showerror(
                            "Load Error",
                            f"{name}: imputer not found.\n"
                            f"Expected '{tgt}' companion files "
                            f"(imputer_{tgt}.pkl / meta_{tgt}.json) "
                            f"in the same folder as the selected pkl.\n\n"
                            f"Tip: select model_{tgt}.pkl (Format A) "
                            f"or make sure htl_model / source_model / "
                            f"imputer / meta files are in the same folder.")
                        continue
                    if name == "V_u":
                        self._pred_bundle_vu  = bundle
                    else:
                        self._pred_bundle_tau = bundle
                    loaded.append(name)
                except Exception as e:
                    messagebox.showerror("Load Error", f"{name}: {e}")

        if loaded:
            self._pred_load_status.config(
                text=f"  ✓  Loaded: {', '.join(loaded)}",
                fg=C["ok"])
        else:
            self._pred_load_status.config(
                text="  No model found. Run analysis first or browse manually.",
                fg=C["danger"])

    def _pred_run(self):
        vu_b  = getattr(self, "_pred_bundle_vu",  None)
        tau_b = getattr(self, "_pred_bundle_tau", None)
        if vu_b is None and tau_b is None:
            messagebox.showwarning("No Model", "Load models first (Step 2).")
            return

        def _flt(key):
            v = self._pred_field_vars.get(key)
            if v is None: return np.nan
            try: return float(v.get())
            except: return np.nan

        all_keys = ([k for k,*_ in self._PRED_FIELDS_COMMON] +
                    [k for k,*_ in self._PRED_FIELDS_FIBER] +
                    [k for k,*_ in self._PRED_FIELDS_TENSILE])
        row = {k: _flt(k) for k in all_keys}

        req = [k for k, *_ in (self._PRED_FIELDS_COMMON + self._PRED_FIELDS_FIBER)]
        if self._pred_grp_var.get() == "Group 2":
            req += [k for k, *_ in self._PRED_FIELDS_TENSILE]
        missing = [k for k in req if np.isnan(row[k])]
        if missing:
            messagebox.showerror("Missing Input",
                f"Required fields empty: {', '.join(missing)}"); return

        if row.get("a_d", np.nan) < 2.0:
            messagebox.showerror("Input Error",
                "a/d must be 2.0 or higher. Please enter a value of 2.0 or higher.")
            return

        notes = []; results = {}

        for bundle, rkey, tgt in [
            (vu_b,  "Vu",  "V_u_KN"),
            (tau_b, "tau", "tau_u_MPa"),
        ]:
            if bundle is None: continue
            try:
                # ── Unpack bundle (always a dict after _pred_load_models) ──
                if not isinstance(bundle, dict):
                    # Safety fallback: raw model object was stored directly
                    raise ValueError(
                        "'XGBRegressor' object is not subscriptable — "
                        "the loaded file is a raw model, not a bundle dict. "
                        "Please select model_V_u_KN.pkl or model_tau_u_MPa.pkl "
                        "(Format A) instead of htl_model_*.pkl, "
                        "or use the Load Models button to auto-discover.")

                mdl        = bundle["model"]
                src_mdl    = bundle.get("src_model")
                imp_tgt    = bundle.get("imp_tgt")
                all_tft    = bundle.get("all_tft", [])
                common_idx = bundle.get("common_idx", [])
                use_rc_b   = bundle.get("use_rc", False)

                if imp_tgt is None:
                    raise ValueError("Imputer not found in bundle. Reload model.")
                if not all_tft:
                    raise ValueError("Feature list (all_tft) missing in bundle. Reload model.")

                X_row = np.array([[row.get(f, np.nan) for f in all_tft]], dtype=float)
                X_imp = imp_tgt.transform(X_row)

                if use_rc_b and src_mdl is not None and len(common_idx) > 0:
                    sp = src_mdl.predict(X_imp[:, common_idx]).reshape(-1, 1)
                    Z  = np.hstack([sp, X_imp])
                else:
                    Z = X_imp

                pred = float(mdl.predict(Z)[0])
                results[tgt] = pred
                txt = f"{pred:.2f}" if tgt == "V_u_KN" else f"{pred:.4f}"
                self._pred_res_lbl[rkey].config(text=txt, fg=C["text"])
            except Exception as e:
                self._pred_res_lbl[rkey].config(text="Error", fg=C["danger"])
                notes.append(f"{tgt}: {e}")

        if "V_u_KN" in results and "tau_u_MPa" not in results:
            b = row["b_mm"]; d = row["d_mm"]
            if b > 0 and d > 0:
                tau = results["V_u_KN"] * 1000.0 / (b * d)
                self._pred_res_lbl["tau"].config(text=f"{tau:.4f}", fg=C["warn"])
                notes.append("τu = Vu×1000/(b×d)  [no τu model loaded].")

        self._pred_note_lbl.config(
            text="  ".join(notes) if notes else "✓  Prediction complete.")

    def _finish(self, t0, cancelled, results_root=None):
        elapsed = time.time() - t0
        if cancelled:
            self._log(f"\n⚠  Analysis stopped  ({elapsed/60:.1f} min elapsed)")
            self._set_status("Stopped", C["warn"])
        else:
            self._log("\n" + "=" * 62)
            self._log(f"  Done!   Elapsed: {elapsed/60:.1f} min")
            if results_root:
                self._log(f"  Output folder: {results_root}")
            self._log("=" * 62)
            self._set_status(f"Done!  ({elapsed/60:.1f} min)", C["ok"])
        self._running = False
        self.after(0, lambda: self._run_btn.enable(True))
        self.after(0, lambda: self._stop_btn.enable(False))
        if results_root and not cancelled:
            self.after(0, lambda: self._open_btn.enable(True))

# ============================================================
# 13. Entry point
# ============================================================
if __name__ == "__main__":
    import tkinter.scrolledtext
    tk.scrolledtext = tkinter.scrolledtext
    app = App()
    app.mainloop()
