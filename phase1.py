# -*- coding: utf-8 -*-
# FORCE LOCAL MODULE IMPORT
import os as _os
import sys as _sys
_THIS_DIR = _os.path.dirname(_os.path.abspath(__file__))
if _THIS_DIR not in _sys.path:
    _sys.path.insert(0, _THIS_DIR)

"""
UNIFIED PHASE 1 PIPELINE
========================
Consolidated pipeline merging Modules 1-10 into a single script.
Implements the entire workflow from raw data to publication artifacts.

Validation Protocol:
- Scenario B (Full Data Training)
- RepeatedKFold CV (5x5) + Phase Holdout Validation
- Statistical Significance Testing (Permutation Tests)
- Physics-Informed Feature Engineering

Usage:
    python unified_phase1.py --data Mekanik_Test_Cleaned.csv --outdir P1_Output --run-optuna --run-ensembles
"""

VERSION_STAMP = "2026-02-03 16:11 REV#HARDENING"

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') # Prevent Tcl/Tk threading issues
import os
import sys
import argparse
import joblib
import json
import warnings
import re
import uuid
import inspect
import ast
from datetime import datetime
from pathlib import Path
from scipy import stats
from joblib import Parallel, delayed, cpu_count

# ML Libraries
from sklearn.model_selection import train_test_split, RandomizedSearchCV, RepeatedKFold, KFold
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor, StackingRegressor, VotingRegressor
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.calibration import calibration_curve
from sklearn.inspection import PartialDependenceDisplay, permutation_importance
from xgboost import XGBRegressor
import shap
import optuna
from optuna.samplers import TPESampler

# Import Plotting Module
from Figure_Table import ArtifactGenerator
import Figure_Table

warnings.filterwarnings('ignore')

print(f"[VERSION] phase1 VERSION_STAMP={VERSION_STAMP}")
print(f"[VERSION] Figure_Table path={Figure_Table.__file__}")
if hasattr(Figure_Table, 'VERSION_STAMP'):
    print(f"[VERSION] Figure_Table VERSION_STAMP={Figure_Table.VERSION_STAMP}")
else:
    print(f"[VERSION] Figure_Table VERSION_STAMP=UNKNOWN")
# ============================================================================
# Aşağıdaki ayarları True veya False yaparak hattın davranışını değiştirebilirsiniz.
# You can change the behavior of the pipeline by setting these to True or False.

CONFIG = {
    # =========================================================================
    # RECOMMENDATIONS & CONSTRAINTS (ÖNERİLER VE KISITLAMALAR)
    # =========================================================================
    # 1. 'VALIDATION_MODE' = 'RIGOROUS' ise 'RUN_RIGOROUS' mutlaka True olmalıdır.
    #    If VALIDATION_MODE is 'RIGOROUS', RUN_RIGOROUS must be True.
    # 2. 'RUN_OPTUNA' = True ise 'N_TRIALS' en az 50 (tercihen 100) olmalıdır.
    #    If RUN_OPTUNA is True, N_TRIALS should be at least 50 (preferably 100).
    # 3. 'RUN_POLYNOMIAL_FEATURES' = True çalışma süresini uzatır ancak R2 değerini artırır.
    #    Polynomial features increase runtime but typically improve R2.
    # 4. 'RUN_SHAP' = True büyük veride yavaş çalışabilir, 43 örnek için uygundur.
    #    SHAP is fine for 43 samples but can be slow on large data.

    # =========================================================================
    # 1. FEATURE ENGINEERING (ÖZELLİK MÜHENDİSLİĞİ)
    # =========================================================================
    # Polinom özellikler (kareler, çarpımlar) üretilsin mi? (Örn: Temp^2, Pressure*Temp)
    'RUN_POLYNOMIAL_FEATURES': True, 

    # =========================================================================
    # 2. MODEL OPTIMIZATION (MODEL OPTİMİZASYONU)
    # =========================================================================
    'RUN_OPTUNA': True,         # Optuna hiperparametre optimizasyonu (Academic Requirement)
    'N_TRIALS': 100,            # Optuna deneme sayısı (Rigorous)
    'FAST_MODE': False,         # If True, reduce N_TRIALS for quicker runs
    
    'RUN_ENSEMBLES': True,      # Ensemble (Stacking/Voting) modelleri eğitilsin mi?
    
    # Voting Regressor için Ağırlıklar (Weights)
    # Modellerin sırası: [RandomForest, HistGradientBoosting, XGBoost]
    # Order of models:   [RandomForest, HistGradientBoosting, XGBoost]
    #
    # Örnekler / Examples:
    # - [0.4, 0.3, 0.3] : Varsayılan (RF ağırlıklı) / Default (RF biased)
    # - [0.33, 0.33, 0.33] : Eşit ağırlık / Equal weights
    # - [0.2, 0.4, 0.4] : Boosting ağırlıklı (Data büyükse iyidir) / Boosting biased
    # - None : Voting kullanma, sadece Stacking kullan (Otomatik optimizasyon)
    #
    # Eğer bu seçenek None yapılırsa, sadece Stacking (Meta-Learner) sonucu kullanılır.
    # Stacking, ağırlıkları kendisi öğrendiği için en "serbest" ve optimize yöntemdir.
    'VOTING_WEIGHTS': None, 
 

    # =========================================================================
    # 3. VALIDATION STRATEGY (DOĞRULAMA STRATEJİSİ)
    # =========================================================================
    'VALIDATION_MODE': 'RIGOROUS', # 'RIGOROUS' (Dürüst/OOF) veya 'STANDARD' (Eğitim Skoru)
    'RUN_DUAL_MODE': True,      # Hem RIGOROUS hem STANDARD modlarını sırayla çalıştır
    'RUN_RIGOROUS': True,       # Legacy flag (True kalmalı)
    'RUN_STATS': True,          # Permütasyon testleri (p-değeri)
    'PERMUTATION_COUNT': 1000,  # Test sayısı (1000 for p < 0.001)
    'CV_SPLITS': 5,             # 5-Fold CV
    'CV_REPEATS': 5,            # 5 Tekrar (Robust Uncertainty)

    # =========================================================================
    # 4. PLOTTING & REPORTING (ÇİZİM VE RAPORLAMA)
    # =========================================================================
    'GENERATE_LATEX': True,     
    'RUN_VECTOR_PLOTS': True,   
    'RUN_SHAP': True,           
    'RUN_PDP': True,            
    'RUN_IMPORTANCE': True,     
    'RUN_CALIBRATION': True,    
    'RUN_ABLATION': True,
    'ENABLE_LIT_TABLE': True,
    'REQUIRE_PDFLATEX_FOR_FIG1': False,
    'SHAP_STRICT': True,
    'ENABLE_RAW_SCATTER_PLOTS': True,
    'ENABLE_SAMPLEWISE_TABLE': True,
    'RAW_SCATTER_FEATURES_BY_TARGET': {},
    'SAMPLEWISE_TABLE_TOP_N': 10,
    
    # =========================================================================
    # 5. SYSTEM SETTINGS (SİSTEM AYARLARI)
    # =========================================================================
    'SEED': 42,                 
    'RANDOM_SEARCH_ITER': 30    
}

# ============================================================================
# CONSTANTS & MAPPINGS (From generate_literature_figures.py)
# ============================================================================
TARGETS = [
    'Max_Compressive_Strength_MPa',
    'Hardness_HB',
    'Sintered_Density_g_cm3',
    'Toughness_MJ_m3'
]

TARGET_SHORT_NAMES = {
    'Max_Compressive_Strength_MPa': 'strength',
    'Hardness_HB': 'hardness',
    'Sintered_Density_g_cm3': 'density',
    'Toughness_MJ_m3': 'toughness'
}

# Mapping for Literature Figures (Which features to plot for PDP/SHAP)
# Format: Target -> (ShortName, [Feature List])
TARGET_MAP = {
    'Max_Compressive_Strength_MPa': ('Strength', ['Green_Density_g_cm3', 'B4C_Mean_wt']),
    'Hardness_HB': ('Hardness', ['B4C_Mean_wt', 'Green_Density_g_cm3']),
    'Sintered_Density_g_cm3': ('Density', ['Sinter_Temp_C', 'Green_Density_g_cm3']),
    'Toughness_MJ_m3': ('Toughness', ['Mxene_Rate_from_code', 'Green_Density_g_cm3'])
}

TARGET_UNITS = {
    'Strength': 'MPa',
    'Hardness': 'HB',
    'Density': 'g/cm^3',
    'Toughness': 'MJ/m^3'
}

# Teknik Ayarlar (Advanced Settings) -> Moved to CONFIG
# ----------------------------------------------------------------------------
# CV_SPLITS, CV_REPEATS, ETC now accessed via CONFIG or self in class




# Feature Config
PROCESS_FEATURES = ['Pressure_MPa', 'Sinter_Temp_C', 'Sinter_Time_min', 'MA_Time_h']
STRUCTURE_FEATURES = ['Layer_Count_from_code', 'Mxene_Rate_from_code']
FGM_FEATURE = ['Is_FGM']
B4C_SUMMARY_FEATURES = ['B4C_Outer_wt', 'B4C_Center_wt', 'B4C_Mean_wt', 'B4C_Gradient_wt']
LAYER_VECTOR_FEATURES = [f'Layer{i}_B4C_Rate' for i in range(1, 8)]

class Phase1Pipeline:
    def __init__(self, input_file, outdir, override_config=None):
        self.input_file = input_file
        self.outdir = outdir
        
        # Merge overrides
        cfg = CONFIG.copy()
        if override_config:
            cfg.update(override_config)

        # --- Configuration Loading ---
        # 1. Optimization
        self.run_polynomial_features = cfg['RUN_POLYNOMIAL_FEATURES']
        self.run_optuna = cfg['RUN_OPTUNA']
        self.run_ensembles = cfg['RUN_ENSEMBLES']
        self.voting_weights = cfg['VOTING_WEIGHTS']
        self.fast_mode = cfg.get('FAST_MODE', False)
        self.n_trials = cfg.get('N_TRIALS', 100)
        if self.fast_mode:
            self.n_trials = min(self.n_trials, 20)
        
        # 2. Validation
        self.validation_mode = cfg['VALIDATION_MODE']
        self.run_calibration = cfg['RUN_CALIBRATION']
        self.run_ablation = cfg['RUN_ABLATION']
        self.run_stats = cfg['RUN_STATS']
        self.enable_lit_table = cfg.get('ENABLE_LIT_TABLE', True)
        self.require_pdflatex_fig1 = cfg.get('REQUIRE_PDFLATEX_FOR_FIG1', False)
        self.shap_strict = cfg.get('SHAP_STRICT', True)
        self.enable_raw_scatter_plots = cfg.get('ENABLE_RAW_SCATTER_PLOTS', True)
        self.enable_samplewise_table = cfg.get('ENABLE_SAMPLEWISE_TABLE', True)
        self.raw_scatter_features_by_target = cfg.get('RAW_SCATTER_FEATURES_BY_TARGET', {}) or {}
        self.samplewise_table_top_n = cfg.get('SAMPLEWISE_TABLE_TOP_N', 10)
        
        # 3. Plotting
        self.generate_latex = cfg['GENERATE_LATEX']
        self.run_shap = cfg['RUN_SHAP']
        self.run_pdp = cfg['RUN_PDP']
        self.run_importance = cfg['RUN_IMPORTANCE']
        self.run_vector_plots = cfg['RUN_VECTOR_PLOTS']
        
        # 4. System/Technical
        self.seed = cfg['SEED']
        self.cv_splits = cfg['CV_SPLITS']
        self.cv_repeats = cfg['CV_REPEATS']
        self.perm_count = cfg['PERMUTATION_COUNT']
        self.n_iter = cfg['RANDOM_SEARCH_ITER'] # Random Search backup
        self.n_inner = 3
        self.artifact_suffix = cfg.get('ARTIFACT_SUFFIX')
        if not self.artifact_suffix:
            self.artifact_suffix = 'rigorous' if self.validation_mode == 'RIGOROUS' else 'standard'
        self.artifact_suffix = str(self.artifact_suffix).lower()
        
        self.dirs = self._create_dirs()
        self._imputer_kwargs_cache = None
        self._imputer_kwargs_logged = False
        
        # Initialize Plotting Module
        self.artifact_gen = ArtifactGenerator(
            output_dirs=self.dirs,
            log_func=self.log,
            validation_mode=self.validation_mode,
            inputs_specs=None
        )
        self.studies = {} # Store Optuna studies for plotting
        
        # Initialize Log File
        self.log_path = os.path.join(self.outdir, "pipeline_log.txt")
        # Clear old log
        with open(self.log_path, 'w', encoding='utf-8') as f:
            f.write(f"=== Phase 1 Pipeline Log - Started at {datetime.now()} ===\n")
        self._imputer_kwargs()
        self.log(f"[CONFIG] ENABLE_LIT_TABLE={self.enable_lit_table}")
        self.log(f"[CONFIG] REQUIRE_PDFLATEX_FOR_FIG1={self.require_pdflatex_fig1}")
        self.log(f"[CONFIG] SHAP_STRICT={self.shap_strict}")
            
        self.df_raw = None
        self.df_qc = None
        self.final_df = None
        self.feature_sets = {}
        self.models = {}
        self.results = {}
        self.stats_results = []
        self.ablation_results = {} 
        self.run_config = None

    def _create_dirs(self):
        """Create output directory structure"""
        subdirs = ['01_data', '02_features', '02_models', '03_tables', '03_metrics', '04_figures',
                   '08_optimization', '09_publication', '10_stats', 'latex']
        paths = {}
        for d in subdirs:
            path = os.path.join(self.outdir, d)
            os.makedirs(path, exist_ok=True)
            paths[d] = path
        return paths

    def log(self, msg):
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted_msg = f"[{timestamp}] {msg}"
        print(formatted_msg)
        
        # Write to file
        try:
            with open(self.log_path, 'a', encoding='utf-8') as f:
                f.write(formatted_msg + "\n")
        except:
            pass # fallback if file lock issues

    def _ordered_unique(self, items):
        seen = set()
        out = []
        for item in items:
            if item not in seen:
                out.append(item)
                seen.add(item)
        return out

    def _imputer_kwargs(self):
        if self._imputer_kwargs_cache is not None:
            return self._imputer_kwargs_cache
        sig = inspect.signature(SimpleImputer)
        if "keep_empty_features" in sig.parameters:
            kwargs = {"strategy": "median", "keep_empty_features": True}
            if not self._imputer_kwargs_logged:
                self.log("SimpleImputer keep_empty_features enabled")
                self._imputer_kwargs_logged = True
        else:
            kwargs = {"strategy": "median"}
            if not self._imputer_kwargs_logged:
                self.log("SimpleImputer keep_empty_features not supported; continuing without it")
                self._imputer_kwargs_logged = True
        self._imputer_kwargs_cache = kwargs
        return kwargs

    def _build_pipeline(self, estimator):
        return Pipeline([
            ('imputer', SimpleImputer(**self._imputer_kwargs())),
            ('scaler', StandardScaler()),
            ('model', estimator)
        ])

    def _fit_model(self, model, X, y, sample_weight=None):
        if sample_weight is None:
            model.fit(X, y)
            return
        if isinstance(model, Pipeline):
            step_name = model.steps[-1][0]
            try:
                model.fit(X, y, **{f"{step_name}__sample_weight": sample_weight})
                return
            except Exception as e:
                self.log(f"Sample_weight failed for pipeline step {step_name}: {e}. Falling back to unweighted fit.")
                model.fit(X, y)
                return
        try:
            fit_sig = inspect.signature(model.fit)
            if 'sample_weight' in fit_sig.parameters:
                model.fit(X, y, sample_weight=sample_weight)
            else:
                model.fit(X, y)
        except Exception as e:
            self.log(f"Sample_weight failed for estimator {type(model).__name__}: {e}. Falling back to unweighted fit.")
            model.fit(X, y)

    def _build_run_config(self):
        mode = self.validation_mode
        split_mode = 'nested_cv' if mode == 'RIGOROUS' else 'train'
        return {
            'mode': mode,
            'split_mode': split_mode,
            'n_outer': self.cv_splits,
            'n_inner': self.n_inner,
            'n_trials': self.n_trials,
            'n_perm': self.perm_count if self.run_stats and mode == 'RIGOROUS' else 0,
            'seed': self.seed,
            'timestamp': datetime.now().isoformat(),
            'ENABLE_ABLATION': bool(self.run_ablation),
            'ENABLE_LIT_TABLE': bool(self.enable_lit_table),
            'REQUIRE_PDFLATEX_FOR_FIG1': bool(self.require_pdflatex_fig1),
            'SHAP_STRICT': bool(self.shap_strict),
            'enable_raw_scatter_plots': bool(self.enable_raw_scatter_plots),
            'enable_samplewise_table': bool(self.enable_samplewise_table),
            'raw_scatter_features_by_target': self.raw_scatter_features_by_target,
            'samplewise_table_top_n': self.samplewise_table_top_n,
            'ARTIFACT_SUFFIX': self.artifact_suffix
        }

    def _write_run_config(self):
        self.run_config = self._build_run_config()
        out_path = os.path.join(self.outdir, 'run_config.json')
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(self.run_config, f, indent=2)

    # =========================================================================
    # STEP 1: LOAD DATA (Module 1 Logic)
    # =========================================================================
    def step_1_load_data(self):
        self.log(f"Loading data from {self.input_file}...")
        encodings = ['utf-8', 'ISO-8859-9', 'cp1254']
        for enc in encodings:
            try:
                self.df_raw = pd.read_csv(self.input_file, sep=';', encoding=enc)
                self.log(f"Loaded successfully with {enc}. Shape: {self.df_raw.shape}")
                break
            except Exception:
                continue
        
        if self.df_raw is None:
            raise ValueError("Failed to load CSV file with any encoding.")

        # Basic cleanup
        # Percentage string cleanup
        for col in self.df_raw.columns:
            if self.df_raw[col].dtype == object:
                if self.df_raw[col].str.contains('%').any():
                    self.df_raw[col] = self.df_raw[col].astype(str).str.replace('%', '').str.replace(',', '.')
                    self.df_raw[col] = pd.to_numeric(self.df_raw[col], errors='coerce')
        
        # KATMAN YOK cleanup
        self.df_raw.replace(['KATMAN YOK', 'nan', 'NaN'], np.nan, inplace=True)
        
        # Numeric conversion
        cols_to_numeric = PROCESS_FEATURES + TARGETS + LAYER_VECTOR_FEATURES
        for col in cols_to_numeric:
            if col in self.df_raw.columns:
                self.df_raw[col] = pd.to_numeric(self.df_raw[col], errors='coerce')

    # =========================================================================
    # STEP 2: PARSE IDENTITY (Module 2 Logic)
    # =========================================================================
    def step_2_parse_identity(self):
        self.log("Parsing Sample_Code and Identity...")
        df = self.df_raw.copy()
        
        # Regex Patterns
        pat_fgm = re.compile(r'^(3|5|7)K[-\s]?(SAF|\d+X)$', re.IGNORECASE)
        pat_taguchi = re.compile(r'^\d{2}[a-c][-\s]?\d[a-c]$', re.IGNORECASE)
        pat_single = re.compile(r'^1K\s+(SAF|\d+B)$', re.IGNORECASE)

        df['Sample_Code_Normalized'] = df['Sample_Code']
        df['Process_Type'] = ''
        df['Layer_Count_from_code'] = np.nan
        df['Mxene_Rate_from_code'] = np.nan
        df['B4C_Outer_wt'] = np.nan
        df['Is_FGM'] = False

        for idx, row in df.iterrows():
            code = str(row['Sample_Code']).strip()
            
            # FGM Logic
            m_fgm = pat_fgm.match(code)
            if m_fgm:
                lc = int(m_fgm.group(1))
                mx = 0 if 'SAF' in m_fgm.group(2).upper() else int(m_fgm.group(2).upper().replace('X',''))
                b4c_outer = {3:5, 5:10, 7:15}.get(lc, np.nan)
                
                df.at[idx, 'Process_Type'] = 'FGM'
                df.at[idx, 'Layer_Count_from_code'] = lc
                df.at[idx, 'Mxene_Rate_from_code'] = mx
                df.at[idx, 'B4C_Outer_wt'] = b4c_outer
                df.at[idx, 'Is_FGM'] = True
                continue

            # Taguchi Logic
            m_tag = pat_taguchi.match(code)
            if m_tag:
                norm = code.upper()
                if '-' not in norm: norm = norm[:3] + '-' + norm[3:]
                
                df.at[idx, 'Sample_Code_Normalized'] = norm
                df.at[idx, 'Process_Type'] = 'Taguchi'
                df.at[idx, 'Layer_Count_from_code'] = row['Layer_Count']
                df.at[idx, 'Mxene_Rate_from_code'] = 0
                df.at[idx, 'B4C_Outer_wt'] = row['Layer1_B4C_Rate']
                continue

            # Single Layer Logic
            m_single = pat_single.match(code)
            if m_single:
                b4c = 0 if 'SAF' in m_single.group(1).upper() else int(m_single.group(1).upper().replace('B',''))
                
                df.at[idx, 'Process_Type'] = 'Single_Layer'
                df.at[idx, 'Layer_Count_from_code'] = 1
                df.at[idx, 'Mxene_Rate_from_code'] = 0
                df.at[idx, 'B4C_Outer_wt'] = b4c
                continue

        # Derived B4C Features
        df['B4C_Center_wt'] = np.nan
        df['B4C_Mean_wt'] = np.nan
        df['B4C_Gradient_wt'] = np.nan
        
        layer_cols = LAYER_VECTOR_FEATURES
        for idx, row in df.iterrows():
            lc = row['Layer_Count_from_code']
            if pd.notna(lc):
                vals = [row[c] for c in layer_cols[:int(lc)] if pd.notna(row[c])]
                if vals:
                    df.at[idx, 'B4C_Mean_wt'] = np.mean(vals)
                    if row['Is_FGM'] and lc in [3,5,7]:
                        center_idx = int(lc)//2
                        if center_idx < len(vals):
                            df.at[idx, 'B4C_Center_wt'] = vals[center_idx]
                            if pd.notna(row['B4C_Outer_wt']):
                                df.at[idx, 'B4C_Gradient_wt'] = row['B4C_Outer_wt'] - vals[center_idx]

        # Phase Labeling
        df['Phase'] = 0
        df.loc[df['Process_Type'] == 'Taguchi', 'Phase'] = 1
        df.loc[df['Process_Type'].isin(['FGM', 'Single_Layer']), 'Phase'] = 2

        self.df_raw = df
        df.to_csv(os.path.join(self.dirs['01_data'], 'dataset_typed.csv'), index=False)

    # =========================================================================
    # STEP 3: QC & WEIGHTING (Module 3 Logic)
    # =========================================================================
    def step_3_qc_checks(self):
        self.log("Applying QC Checks and calculating Sample Weights...")
        df = self.df_raw.copy()
        
        # Thresholds
        DENSITY_MIN, DENSITY_MAX = 2.20, 2.85
        SWEAT_TEMP, SWEAT_PRESS = 590, 300
        STRENGTH_MIN, STRENGTH_MAX = 50, 600
        
        # Flags
        df['Potential_Sweating_Defect'] = 0
        mask_sweat = (df['Sinter_Temp_C'] > SWEAT_TEMP) & (df['Pressure_MPa'] < SWEAT_PRESS)
        df.loc[mask_sweat, 'Potential_Sweating_Defect'] = 1
        
        df['Low_Density_Error'] = (df['Sintered_Density_g_cm3'] < DENSITY_MIN).astype(int)
        df['High_Density_Error'] = (df['Sintered_Density_g_cm3'] > DENSITY_MAX).astype(int)
        df['Structural_Failure'] = (df['Max_Compressive_Strength_MPa'] < STRENGTH_MIN).astype(int)
        
        # Sample Weighting
        df['sample_weight'] = 1.0
        multipliers = {
            'Potential_Sweating_Defect': 0.25,
            'Structural_Failure': 0.50,
            'Low_Density_Error': 0.10,
            'High_Density_Error': 0.10
        }
        
        for flag, mult in multipliers.items():
            if flag in df.columns:
                df.loc[df[flag] == 1, 'sample_weight'] *= mult
        
        df.loc[df['sample_weight'] < 0.25, 'sample_weight'] = 0.25
        self.df_qc = df
        df.to_csv(os.path.join(self.dirs['01_data'], 'dataset_with_qc.csv'), index=False)

    # =========================================================================
    # STEP 4: FEATURE ENGINEERING (Module 4 Logic)
    # =========================================================================
    def step_4_feature_engineering(self):
        self.log("Engineering Features (Physics-Informed)...")
        df = self.df_qc.copy()
        
        # 1. Structural Nan Filling
        df['Is_FGM'] = df['Is_FGM'].astype(int)
        non_fgm = df['Is_FGM'] == 0
        df.loc[non_fgm, 'B4C_Center_wt'] = df.loc[non_fgm, 'B4C_Center_wt'].fillna(0)
        df.loc[non_fgm, 'B4C_Gradient_wt'] = df.loc[non_fgm, 'B4C_Gradient_wt'].fillna(0)
        
        # 2. Interactions
        df['Pressure_x_Temp'] = df['Pressure_MPa'] * df['Sinter_Temp_C']
        df['Temp_x_Time'] = df['Sinter_Temp_C'] * df['Sinter_Time_min']
        df['Sinter_Energy'] = df['Sinter_Temp_C'] * df['Sinter_Time_min']  # Proxy
        df['Pressure_Temp_Ratio'] = df['Pressure_MPa'] / (df['Sinter_Temp_C'] + 1e-6)
        
        if 'B4C_Gradient_wt' in df.columns:
            df['Layer_Complexity'] = df['Layer_Count_from_code'] * df['B4C_Gradient_wt']
            
        # 3. Missing Indicators
        if 'Green_Density_g_cm3' in df.columns:
            df['Green_Density_missing'] = df['Green_Density_g_cm3'].isna().astype(int)
            
        # 4. Polynomial Features (Squares) if Enabled
        poly_feats = []
        if self.run_polynomial_features:
            self.log("  Generating Polynomial Features (Squares & Interactions)...")
            # Key features to square
            keys = ['Sinter_Temp_C', 'Pressure_MPa', 'B4C_Mean_wt', 'Green_Density_g_cm3']
            for k in keys:
                if k in df.columns:
                    fname = f"{k}_sq"
                    df[fname] = df[k] ** 2
                    poly_feats.append(fname)
                    
            # Additional interaction
            if 'Sinter_Time_min' in df.columns and 'Sinter_Temp_C' in df.columns:
                 df['Sinter_Time_Temp_Sq'] = df['Sinter_Time_min'] * (df['Sinter_Temp_C']**2)
                 poly_feats.append('Sinter_Time_Temp_Sq')

        # 4. Create Modes
        # Core
        feats_core = PROCESS_FEATURES + STRUCTURE_FEATURES + FGM_FEATURE + \
                     ['Pressure_x_Temp', 'Temp_x_Time', 'Sinter_Energy'] + B4C_SUMMARY_FEATURES + poly_feats
        
        # Extended (Includes Green_Density)
        feats_ext = feats_core + ['Green_Density_g_cm3', 'Green_Density_missing']
        
        # Layer Vector
        feats_vec = PROCESS_FEATURES + STRUCTURE_FEATURES + FGM_FEATURE + LAYER_VECTOR_FEATURES

        # Store final dataframe for downstream alignment
        self.final_df = df.copy()
        
        # Saves
        self._save_feature_set('core', df, feats_core)
        self._save_feature_set('core_extended', df, feats_ext)
        self._save_feature_set('layer_vector', df, feats_vec)
        self._save_feature_set('layer_vector_extended', df, feats_vec + ['Green_Density_g_cm3'])

    def _save_feature_set(self, name, df, vocab):
        # Filter columns present (deterministic order)
        valid_vocab = [c for c in vocab if c in df.columns]
        x_cols = self._ordered_unique(valid_vocab)
        X_df = df.loc[:, x_cols].copy()
        self.feature_sets[name] = X_df

        # Save full feature set with targets + meta (machine-readable)
        meta = ['Sample_Code', 'Phase', 'sample_weight']
        final_cols = self._ordered_unique(x_cols + TARGETS + meta)
        out_df = df.loc[:, final_cols].copy()
        out_path = os.path.join(self.dirs['02_features'], f'features_{name}.csv')
        out_df.to_csv(out_path, index=False)

    # =========================================================================
    # OPTUNA HELPERS (Module 8a Logic)
    # =========================================================================
    def _get_param_space(self, trial, algorithm):
        """Define hyperparameter search space for each algorithm"""
        if algorithm == 'RandomForest':
            return {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300, step=50),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None])
            }
        elif algorithm == 'XGBoost':
            return {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300, step=50),
                'max_depth': trial.suggest_int('max_depth', 2, 6),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'gamma': trial.suggest_float('gamma', 0, 5)
            }
        elif algorithm == 'HistGradientBoosting':
            return {
                'max_iter': trial.suggest_int('max_iter', 50, 300, step=50),
                'max_depth': trial.suggest_int('max_depth', 2, 8),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'l2_regularization': trial.suggest_float('l2_regularization', 1e-4, 1.0, log=True)
            }
        elif algorithm == 'Ridge':
            return {
                'alpha': trial.suggest_float('alpha', 0.01, 100.0, log=True)
            }
        return {}

    def _create_model(self, algorithm, params):
        """Create model instance with given parameters"""
        if algorithm == 'RandomForest':
            # Handle None for max_features if coming from JSON or Optuna
            if params.get('max_features') == 'None': params['max_features'] = None
            return RandomForestRegressor(**params, random_state=self.seed, n_jobs=-1)
        elif algorithm == 'XGBoost':
            # Enable GPU acceleration -> Reverting to CPU 'hist' for stability
            gpu_params = {
                'tree_method': 'hist', 
                'n_jobs': -1 
            }
            return XGBRegressor(**params, **gpu_params, random_state=self.seed, verbosity=0)
        elif algorithm == 'HistGradientBoosting':
            return HistGradientBoostingRegressor(**params, random_state=self.seed)
        elif algorithm == 'Ridge':
            return Ridge(**params, random_state=self.seed)
        elif algorithm in ['Stacking', 'Voting']:
            # Fallback for Ablation if ensemble was passed. 
            # Use XGBoost as proxy for "Strong Learner" performance
            base_params = {'tree_method': 'hist', 'n_jobs': -1}
            return XGBRegressor(**base_params, random_state=self.seed, verbosity=0)
        return None

    # =========================================================================
    # STEP 5a: OPTUNA OPTIMIZATION
    # =========================================================================
    def step_5a_optuna_optimization(self):
        if not self.run_optuna:
            self.log("Optuna Optimization skipped (RUN_OPTUNA=False).")
            # Try load existing
            json_path = os.path.join(self.dirs['08_optimization'], 'best_params.json')
            if os.path.exists(json_path):
                try:
                    with open(json_path, 'r') as f:
                        self.best_params_ = json.load(f)
                    self.log("  Loaded existing best_params.json")
                except: pass
            return

        if self.final_df is None:
            raise RuntimeError("final_df not available. Run feature engineering first.")

        self.log(f"Starting Optuna Hyperparameter Optimization (Trials={self.n_trials})...")
        
        best_params_store = {}
        
        # Optimize for each target
        for target in TARGETS:
            self.log(f"  Optimizing for {target}...")
            
            # Use 'layer_vector_extended' or 'core_extended' as generally best represenative
            # Or iterate all? Module 8a iterated all. For consolidated phase1, let's optimize the BEST candidate from quick search or just default to 'layer_vector_extended'.
            # Paper suggests 'layer_vector' is good for density, 'core' for others?
            # Let's iterate ALGORITHMS on ONE robust feature set to save time, or use the mapping?
            # Let's use 'layer_vector_extended' (Scenario B) as it's the most complete set.
            mode = 'layer_vector_extended'
            if mode not in self.feature_sets: continue
            
            X_all = self.feature_sets[mode]
            valid_mask = self.final_df[target].notna()
            X = X_all.loc[valid_mask]
            y = self.final_df.loc[valid_mask, target]
            w = self.final_df.loc[valid_mask, 'sample_weight']
            
            best_score = -np.inf
            best_algo_name = None
            best_algo_params = None
            
            # Algos to optimize
            algos = ['RandomForest', 'XGBoost', 'HistGradientBoosting', 'Ridge']
            
            for algo in algos:
                def objective(trial):
                    params = self._get_param_space(trial, algo)
                    # Manually handle categorical string restrictions if needed
                    estimator = self._create_model(algo, params)
                    if estimator is None:
                        return -999
                    
                    # 5-Fold CV
                    cv_scores = []
                    kf = RepeatedKFold(n_splits=3, n_repeats=1, random_state=self.seed) # 3-fold for speed inside optuna
                    for tr_idx, te_idx in kf.split(X):
                        X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
                        y_tr, y_te = y.iloc[tr_idx], y.iloc[te_idx]
                        w_tr = w.iloc[tr_idx] if w is not None else None
                        
                        try:
                           m = self._build_pipeline(self._create_model(algo, params))
                           self._fit_model(m, X_tr, y_tr, sample_weight=w_tr)
                           preds = m.predict(X_te)
                           cv_scores.append(r2_score(y_te, preds))
                        except:
                           return -999
                    return np.mean(cv_scores)

                sampler = TPESampler(seed=self.seed)
                study = optuna.create_study(direction='maximize', sampler=sampler)
                optuna.logging.set_verbosity(optuna.logging.WARNING)
                try:
                    study.optimize(objective, n_trials=self.n_trials)
                except Exception as e:
                    self.log(f"    Optuna failed for {algo}: {e}")
                    continue
                
                if study.best_value > best_score:
                    best_score = study.best_value
                    best_algo_name = algo
                    best_algo_params = study.best_params
                    best_study_obj = study # Capture study object
            
            # Store best for this target
            if best_algo_name:
                best_params_store[target] = {
                    'algorithm': best_algo_name,
                    'params': best_algo_params,
                    'cv_r2': best_score
                }
                # Save study for plotting traces
                if 'best_study_obj' in locals():
                    self.studies[target] = best_study_obj
                    
                self.log(f"    Best for {target}: {best_algo_name} (R2={best_score:.3f})")

        # Save to JSON
        json_path = os.path.join(self.dirs['08_optimization'], 'best_params.json')
        with open(json_path, 'w') as f:
            json.dump(best_params_store, f, indent=4)
        self.best_params_ = best_params_store

    # =========================================================================
    # STEP 5: TRAINING SCENARIO B (Module 6 Logic)
    # =========================================================================
    def step_5_train_scenario_B(self):
        self.log("Starting Training Scenario B (Full Data)...")
        if self.final_df is None:
            raise RuntimeError("final_df not available. Run feature engineering first.")

        results = []

        for target in TARGETS:
            self.log(f"Training for target: {target}")

            best_rmse = np.inf
            best_model_info = None

            for mode_name, X_all in self.feature_sets.items():
                if X_all is None or X_all.empty:
                    continue

                feature_cols = list(X_all.columns)
                valid_mask = self.final_df[target].notna()

                X = X_all.loc[valid_mask]
                y = self.final_df.loc[valid_mask, target]
                weights = self.final_df.loc[valid_mask, 'sample_weight']
                phase = self.final_df.loc[valid_mask, 'Phase']

                # Algorithms
                algos = {
                    'XGBoost': XGBRegressor(random_state=self.seed, verbosity=0),
                    'RandomForest': RandomForestRegressor(random_state=self.seed),
                    'Ridge': Ridge(random_state=self.seed)
                }

                for algo_name, model in algos.items():
                    pipe = self._build_pipeline(model)

                    mask_p1 = phase == 1
                    mask_p2 = phase == 2

                    if mask_p1.sum() == 0 or mask_p2.sum() == 0:
                        continue

                    self._fit_model(pipe, X[mask_p1], y[mask_p1], sample_weight=weights[mask_p1])
                    pred_p2 = pipe.predict(X[mask_p2])
                    rmse_p2 = np.sqrt(mean_squared_error(y[mask_p2], pred_p2))
                    r2_p2 = r2_score(y[mask_p2], pred_p2)

                    if rmse_p2 < best_rmse:
                        best_rmse = rmse_p2
                        self._fit_model(pipe, X, y, sample_weight=weights)
                        best_model_info = {
                            'best_model': pipe,
                            'feature_set': mode_name,
                            'feature_names': feature_cols,
                            'algo': algo_name,
                            'holdout_rmse': rmse_p2,
                            'holdout_r2': r2_p2
                        }

            if best_model_info:
                self.models[target] = best_model_info
                joblib.dump(best_model_info['best_model'],
                           os.path.join(self.dirs['02_models'], f'model_{TARGET_SHORT_NAMES[target]}.joblib'))
                results.append(best_model_info)
                self.log(f"Best model for {target}: {best_model_info['algo']} (Mode: {best_model_info['feature_set']}) RMSE: {best_model_info['holdout_rmse']:.4f}")

    # =========================================================================
    # STEP 5b: ENSEMBLE TRAINING (Module 7 Logic)
    # =========================================================================
    def _train_ensembles(self):
        self.log("Training Ensemble Models (Stacking & Voting)...")
        
        for target in TARGETS:
            if target not in self.models: continue
            
            info = self.models[target]
            mode = info['feature_set']
            feature_cols = info['feature_names']

            X_all = self.feature_sets.get(mode)
            if X_all is None or X_all.empty:
                continue

            valid_mask = self.final_df[target].notna()
            X = X_all.loc[valid_mask, feature_cols]
            y = self.final_df.loc[valid_mask, target]
            weights = self.final_df.loc[valid_mask, 'sample_weight']
            phase = self.final_df.loc[valid_mask, 'Phase']
            
            # Base Estimators (Diverse set)
            estimators = [
                ('rf', RandomForestRegressor(n_estimators=100, max_depth=5, random_state=self.seed)),
                ('gb', HistGradientBoostingRegressor(max_iter=100, max_depth=3, random_state=self.seed)),
                ('xgb', XGBRegressor(n_estimators=100, max_depth=3, verbosity=0, random_state=self.seed))
            ]
            
            # 1. Stacking Ensemble
            stack = StackingRegressor(
                estimators=estimators,
                final_estimator=Ridge(random_state=self.seed),
                cv=self.cv_splits
            )
            
            # 2. Voting Ensemble
            vote = None
            if self.voting_weights is not None:
                vote = VotingRegressor(
                    estimators=estimators,
                    weights=self.voting_weights 
                )
            
            # Train and Evaluate Stacking
            pipe_stack = Pipeline([('imputer', SimpleImputer(**self._imputer_kwargs())),('scaler', StandardScaler()),('stack', stack)])
            mask_p2 = phase == 2
            mask_p1 = phase == 1
            if mask_p2.sum() > 0:
                self._fit_model(pipe_stack, X[mask_p1], y[mask_p1], sample_weight=weights[mask_p1])
                pred_stack = pipe_stack.predict(X[mask_p2])
                rmse_stack = np.sqrt(mean_squared_error(y[mask_p2], pred_stack))
                self.log(f"  {target} Stacking RMSE: {rmse_stack:.4f}")
                
                
                # Check if better than single model
                if rmse_stack < info.get('holdout_rmse', np.inf):
                    self.log(f"    -> Stacking Improved RMSE! ({info.get('holdout_rmse', np.inf):.4f} -> {rmse_stack:.4f})")
                    # Update best model
                    self._fit_model(pipe_stack, X, y, sample_weight=weights)
                    self.models[target]['best_model'] = pipe_stack
                    self.models[target]['algo'] = 'Stacking'
                    self.models[target]['holdout_rmse'] = rmse_stack
                    info = self.models[target]
            
            # Train and Evaluate Voting (if enabled)
            if vote is not None:
                pipe_vote = Pipeline([('imputer', SimpleImputer(**self._imputer_kwargs())),('scaler', StandardScaler()),('vote', vote)])
                if mask_p2.sum() > 0:
                    self._fit_model(pipe_vote, X[mask_p1], y[mask_p1], sample_weight=weights[mask_p1])
                    pred_vote = pipe_vote.predict(X[mask_p2])
                    rmse_vote = np.sqrt(mean_squared_error(y[mask_p2], pred_vote))
                    self.log(f"  {target} Voting RMSE: {rmse_vote:.4f}")
                    
                    if rmse_vote < info.get('holdout_rmse', np.inf):
                        self.log(f"    -> Voting Improved RMSE! ({info.get('holdout_rmse', np.inf):.4f} -> {rmse_vote:.4f})")
                        self._fit_model(pipe_vote, X, y, sample_weight=weights)
                        self.models[target]['best_model'] = pipe_vote
                        self.models[target]['algo'] = 'Voting'
                        self.models[target]['holdout_rmse'] = rmse_vote

    # =========================================================================
    # STEP 6: STATISTICAL VALIDATION (Module 10 Logic)
    # =========================================================================
    def step_6_statistical_validation(self):
        stats_path = os.path.join(self.dirs['10_stats'], 'permutation_results.csv')
        
        # Check if exists to skip
        if os.path.exists(stats_path):
            self.log(f"Found existing statistics at {stats_path}. Loading...")
            try:
                self.stats_results = pd.read_csv(stats_path).to_dict('records')
                # Parse perm_scores back to list if string? CSV saves lists as strings.
                # It's okay, we only use 'r2' and 'p_value' generally in artifacts. 
                # If we need perm_scores for plotting, we might need to eval().
                for res in self.stats_results:
                     if isinstance(res.get('perm_scores'), str):
                         try:
                             res['perm_scores'] = json.loads(res['perm_scores'])
                         except Exception:
                             try:
                                 res['perm_scores'] = ast.literal_eval(res['perm_scores'])
                             except Exception:
                                 res['perm_scores'] = []
                
                self.log("Loaded stats successfully.")
                self.run_stats = False # Prevent re-run logic if called again? Or just return.
                return
            except Exception as e:
                self.log(f"Failed to load stats: {e}. Re-running...")

        if not self.run_stats:
            self.log("Statistical Validation skipped (RUN_STATS=False).")
            return

    # =========================================================================
    # STEP 6 & 7: RIGOROUS VALIDATION (Nested CV + Permutation)
    # =========================================================================
    def _run_optuna_on_fold(self, X_tr, y_tr, algo, n_inner, n_trials, seed, sample_weight=None):
        """Helper to run Optuna on a specific fold's training data"""
        if not self.run_optuna:
            return {} 

        def objective(trial):
            params = self._get_param_space(trial, algo)
            # Create model
            estimator = self._create_model(algo, params)
            if estimator is None: return -999
            
            # Inner CV
            inner_cv = KFold(n_splits=n_inner, shuffle=True, random_state=seed)
            scores = []
            for tr, val in inner_cv.split(X_tr, y_tr):
                m = self._build_pipeline(self._create_model(algo, params))
                w_tr = sample_weight.iloc[tr] if sample_weight is not None else None
                self._fit_model(m, X_tr.iloc[tr], y_tr.iloc[tr], sample_weight=w_tr)
                preds = m.predict(X_tr.iloc[val])
                scores.append(r2_score(y_tr.iloc[val], preds))
            return np.mean(scores)

        sampler = TPESampler(seed=seed)
        study = optuna.create_study(direction='maximize', sampler=sampler)
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study.optimize(objective, n_trials=n_trials)
        return study.best_params

    def _nested_cv_optuna_oof(self, X, y, algo, n_outer=5, n_inner=3, n_trials=100, seed=42, sample_weight=None, conformal=True, splits=None):
        """
        Strict Nested CV Loop with Inner Optuna Optimization.
        Returns OOF predictions, metrics, and split info.
        """
        # Outer CV
        if splits is None:
            outer_cv = KFold(n_splits=n_outer, shuffle=True, random_state=seed)
            splits = list(outer_cv.split(X, y))
        
        # Storage
        oof_preds = np.zeros(len(y))
        fold_rmses = []
        outer_splits_info = [] 
        
        self.log(f"    Starting Nested CV ({n_outer}x{n_inner}, Trials={n_trials})...")

        best_params_last = {}

        for fold_id, (tr_idx, te_idx) in enumerate(splits):
            X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
            y_tr, y_te = y.iloc[tr_idx], y.iloc[te_idx]
            w_tr = sample_weight.iloc[tr_idx] if sample_weight is not None else None
            
            # 1. Inner Loop (Hyperparameter Optimization)
            best_params = self._run_optuna_on_fold(X_tr, y_tr, algo, n_inner, n_trials, seed + fold_id, sample_weight=w_tr)
            best_params_last = best_params

            # 2. Train Final Model for this Fold
            estimator = self._create_model(algo, best_params)
            model = self._build_pipeline(estimator)
            self._fit_model(model, X_tr, y_tr, sample_weight=w_tr)
            
            # 3. Predict OOF
            preds = model.predict(X_te)
            oof_preds[te_idx] = preds
            
            # Metrics
            rmse = np.sqrt(mean_squared_error(y_te, preds))
            fold_rmses.append(rmse)
            outer_splits_info.append({'fold': fold_id, 'test_idx': X.iloc[te_idx].index.tolist()})

        # Calculate Metrics
        oof_r2 = r2_score(y, oof_preds)
        oof_rmse = np.sqrt(mean_squared_error(y, oof_preds))
        oof_residuals = np.abs(y - oof_preds)
        
        # Conformal Prediction (Cross-Conformal)
        q = np.quantile(oof_residuals, 0.95) if conformal else 0
        oof_lower = oof_preds - q
        oof_upper = oof_preds + q
        pi_coverage = np.mean((y >= oof_lower) & (y <= oof_upper)) if conformal else 0
        
        return {
            'oof_preds': oof_preds,
            'r2': oof_r2,
            'rmse': oof_rmse,
            'fold_rmses': fold_rmses,
            'outer_splits_info': outer_splits_info,
            'q': q,
            'pi_coverage': pi_coverage,
            'oof_lower': oof_lower,
            'oof_upper': oof_upper,
            'best_params_last': best_params_last,
            'splits': splits
        }

    def step_6_rigorous_validation(self):
        """
        Unified Validation Step:
        - If VALIDATION_MODE='RIGOROUS': Runs Nested CV, Cohen's d, and Permutation Tests.
        - If VALIDATION_MODE='STANDARD': Runs simple training variance check.
        """
        self.log(f"Starting Validation (Mode: {self.validation_mode})...")
        
        is_rigorous = (self.validation_mode == 'RIGOROUS')
        should_run_stats = (self.run_stats and is_rigorous)
        stats_data_list = []
        outer_splits_data = {}

        for target in TARGETS:
            if target not in self.models:
                continue

            info = self.models[target]
            mode = info['feature_set']
            algo = info.get('algo', 'XGBoost')

            X_all = self.feature_sets.get(mode)
            if X_all is None or X_all.empty:
                continue

            valid_mask = self.final_df[target].notna()
            X = X_all.loc[valid_mask, info['feature_names']]
            y = self.final_df.loc[valid_mask, target]
            w = self.final_df.loc[valid_mask, 'sample_weight']

            target_algo = algo if algo in ['XGBoost', 'RandomForest', 'HistGradientBoosting', 'Ridge'] else 'XGBoost'
            if algo in ['Stacking', 'Voting']:
                target_algo = 'XGBoost'

            outer_cv = KFold(n_splits=self.cv_splits, shuffle=True, random_state=self.seed)
            splits = list(outer_cv.split(X, y))

            if is_rigorous:
                self.log(f"  {target}: Running Nested CV (Optuna OOF)...")

                nested_res = self._nested_cv_optuna_oof(
                    X, y, target_algo,
                    n_outer=self.cv_splits,
                    n_inner=self.n_inner,
                    n_trials=self.n_trials if self.run_optuna else 0,
                    seed=self.seed,
                    sample_weight=w,
                    conformal=True,
                    splits=splits
                )

                oof_indices = y.index.to_numpy()
                info['oof'] = {
                    'y_true': y.to_numpy(),
                    'y_pred': nested_res['oof_preds'],
                    'indices': oof_indices,
                    'r2': float(nested_res['r2']),
                    'rmse': float(nested_res['rmse']),
                    'fold_indices': [X.index[te].tolist() for _, te in splits]
                }

                info['pi'] = {
                    'coverage': float(nested_res['pi_coverage']),
                    'width': float(nested_res['q'] * 2),
                    'q_low': 0.05,
                    'q_high': 0.95
                }

                outer_splits_data[target] = nested_res['outer_splits_info']

                # 2. Cohen's d (Effect Size) with aligned splits
                if 'core' in self.feature_sets:
                    X_core_all = self.feature_sets['core']
                    if X_core_all is not None and not X_core_all.empty:
                        sample_codes = self.final_df.loc[valid_mask, 'Sample_Code'].astype(str).tolist()
                        X_core = X_core_all.loc[valid_mask].copy()
                        X_core.index = sample_codes
                        X_main = X.copy()
                        X_main.index = sample_codes
                        y_core = y.copy()
                        y_core.index = sample_codes
                        w_core = w.copy()
                        w_core.index = sample_codes

                        X_core = X_core.reindex(sample_codes)
                        X_main = X_main.reindex(sample_codes)
                        y_core = y_core.reindex(sample_codes)
                        w_core = w_core.reindex(sample_codes)

                        if len(X_core) == len(X_main):
                            core_res = self._nested_cv_optuna_oof(
                                X_core, y_core, target_algo,
                                n_outer=self.cv_splits,
                                n_inner=self.n_inner,
                                n_trials=self.n_trials if self.run_optuna else 0,
                                seed=self.seed,
                                sample_weight=w_core,
                                conformal=False,
                                splits=splits
                            )

                            if len(core_res['fold_rmses']) == len(nested_res['fold_rmses']):
                                diffs = np.array(core_res['fold_rmses']) - np.array(nested_res['fold_rmses'])
                                mean_diff = np.mean(diffs)
                                sd_diff = np.std(diffs, ddof=1) + 1e-9
                                d_val = mean_diff / sd_diff
                                info['cohens_d'] = float(d_val)
                                info['sd_diff'] = float(sd_diff)
                                self.log(f"    Cohen's d: {d_val:.2f} (SD_diff={sd_diff:.4f})")
                            else:
                                info['cohens_d'] = np.nan
                                info['sd_diff'] = np.nan
                        else:
                            info['cohens_d'] = np.nan
                            info['sd_diff'] = np.nan
                    else:
                        info['cohens_d'] = np.nan
                        info['sd_diff'] = np.nan
                else:
                    info['cohens_d'] = np.nan
                    info['sd_diff'] = np.nan

                # 3. Permutation Test
                perm_scores = []
                p_value = 1.0
                if should_run_stats:
                    self.log(f"    Running Permutation Test ({self.perm_count} runs)...")

                    base_model = self._build_pipeline(self._create_model(target_algo, nested_res['best_params_last']))
                    y_array = y.to_numpy()
                    for i in range(self.perm_count):
                        rng = np.random.RandomState(self.seed + i)
                        y_perm = rng.permutation(y_array)
                        preds_perm = np.zeros(len(y_perm))
                        for tr_idx, te_idx in splits:
                            m = clone(base_model)
                            w_tr = w.iloc[tr_idx] if w is not None else None
                            self._fit_model(m, X.iloc[tr_idx], y_perm[tr_idx], sample_weight=w_tr)
                            preds_perm[te_idx] = m.predict(X.iloc[te_idx])
                        perm_scores.append(r2_score(y_perm, preds_perm))

                    obs_r2 = nested_res['r2']
                    p_value = (np.sum(np.array(perm_scores) >= obs_r2) + 1) / (len(perm_scores) + 1)

                    stats_data_list.append({
                        'target': target,
                        'original_score': obs_r2,
                        'p_value': float(p_value),
                        'perm_scores': json.dumps(perm_scores),
                        'n_perm': self.perm_count,
                        'cv_splits': self.cv_splits,
                        'cv_repeats': self.cv_repeats
                    })

                info['perm_test'] = {
                    'target': target,
                    'original_score': float(info['oof']['r2']),
                    'p_value': float(p_value),
                    'perm_scores': json.dumps(perm_scores),
                    'n_perm': int(self.perm_count if should_run_stats else 0),
                    'cv_splits': int(self.cv_splits),
                    'cv_repeats': int(self.cv_repeats)
                }
                info['flag_not_significant'] = bool(p_value >= 0.05) if should_run_stats else False

            else:
                model_template = info['best_model']
                preds = model_template.predict(X)
                train_r2 = r2_score(y, preds)
                train_rmse = np.sqrt(mean_squared_error(y, preds))
                resid = np.abs(y - preds)
                q = np.quantile(resid, 0.95)

                info['oof'] = {
                    'y_true': y.to_numpy(),
                    'y_pred': preds,
                    'indices': y.index.to_numpy(),
                    'r2': float(train_r2),
                    'rmse': float(train_rmse),
                    'fold_indices': []
                }
                info['pi'] = {
                    'coverage': float(np.mean((y >= (preds - q)) & (y <= (preds + q)))),
                    'width': float(q * 2),
                    'q_low': 0.05,
                    'q_high': 0.95
                }
                info['perm_test'] = {
                    'target': target,
                    'original_score': float(train_r2),
                    'p_value': 1.0,
                    'perm_scores': json.dumps([]),
                    'n_perm': 0,
                    'cv_splits': int(self.cv_splits),
                    'cv_repeats': int(self.cv_repeats)
                }
                info['cohens_d'] = np.nan
                info['sd_diff'] = np.nan
                info['flag_not_significant'] = False

                self.log(f"  {target} STANDARD: R2(train)={train_r2:.3f}")
                if 'holdout_r2' in info:
                    self.log(f"  {target} STANDARD: R2(holdout)={info['holdout_r2']:.3f}")

        # Save Rigorous Artifacts
        if is_rigorous:
            if outer_splits_data:
                with open(os.path.join(self.dirs['10_stats'], 'outer_splits.json'), 'w') as f:
                    json.dump(outer_splits_data, f, indent=4)

            if stats_data_list:
                self.stats_results = stats_data_list
                pd.DataFrame(stats_data_list).to_csv(os.path.join(self.dirs['10_stats'], 'stats_results.csv'), index=False)

    # =========================================================================
    # Step 7 stub removed.
    # =========================================================================

    # =========================================================================
    # HELPER: RIGOROUS CV EVALUATION (Wrapper for Ablation)
    # =========================================================================
    def _evaluate_cv(self, X, y, model_tmpl, weights=None, conformal=False, splits=None):
        """
        Wrapper to call _nested_cv_optuna_oof compatible with Ablation Study interface.
        Note: Ablation calls this. We must decide if Ablation runs FULL Optuna (Time consuming).
        Prompt says: "Ablation table... phase1.py... _evaluate_cv RepeatedKFold kullanmaktadır... Bunları kaldır".
        So yes, Ablation should use the rigorous logic (Nested CV).
        """
        # Determine Algorithm string from model_tmpl
        # Check if model_tmpl is a Pipeline or Estimator
        algo = 'XGBoost' # Default
        if hasattr(model_tmpl, 'steps'):
            est = model_tmpl.steps[-1][1]
        else:
            est = model_tmpl
            
        if isinstance(est, RandomForestRegressor): algo = 'RandomForest'
        elif isinstance(est, HistGradientBoostingRegressor): algo = 'HistGradientBoosting'
        elif isinstance(est, Ridge): algo = 'Ridge'
        elif isinstance(est, XGBRegressor): algo = 'XGBoost'
        if hasattr(est, 'estimators'):
            algo = 'Stacking'

        n_trials = self.n_trials
        
        res = self._nested_cv_optuna_oof(
            X, y, algo, 
            n_outer=self.cv_splits, 
            n_inner=self.n_inner, 
            n_trials=n_trials if self.run_optuna else 0,
            seed=self.seed,
            sample_weight=weights,
            conformal=conformal,
            splits=splits
        )
        return res


    # =========================================================================
    # STEP 7: RIGOROUS VALIDATION (Conformal Prediction & Cohen's d)
    # =========================================================================
    # Step 7 merged into Step 6_rigorous_validation

    # =========================================================================
    # PLOTTING HELPERS (Generate_* Logic)
    # =========================================================================
    # =========================================================================
    # STEP 8: ADVANCED PLOTTING (Consolidated)
    # =========================================================================
    def step_8_advanced_plots(self):
        # Delegate to Figure_Table module
        bp = getattr(self, 'best_params_', {})
        
        # Ensure final_df and run_config are passed
        self.artifact_gen.run_advanced_plots(
            final_df=self.final_df,
            models=self.models,
            feature_sets=self.feature_sets,
            df_qc=self.df_qc,
            run_vector_plots=self.run_vector_plots,
            run_pdp=self.run_pdp,
            run_shap=self.run_shap,
            best_params_=bp,
            run_config=self.run_config
        )
        
        # Explicit call for Sample-wise Table if enabled
        if self.enable_samplewise_table:
            self.log("Exporting Sample-wise Prediction Table...")
            try:
                self.artifact_gen.export_samplewise_prediction_table(
                    models=self.models,
                    final_df=self.final_df,
                    top_n=self.samplewise_table_top_n,
                    validation_mode=self.validation_mode
                )
            except Exception as e:
                self.log(f"Sample-wise table export failed: {e}")

    # _set_plot_style, _plot_boxplots, etc. removed as they are now in Figure_Table.py

    def step_8b_calibration_analysis(self):
        if not self.run_calibration: return
        self.artifact_gen.run_calibration_analysis(self.models, self.feature_sets, self.final_df, self.run_config)

    # =========================================================================
    # STEP 10: ABLATION STUDY
    # =========================================================================
    def step_10_ablation_study(self):
        if not self.run_ablation:
            self.log("Ablation Study skipped (RUN_ABLATION=False).")
            return
        self.artifact_gen.run_ablation_study(self.models, self.feature_sets, self.final_df, self._evaluate_cv, self.run_config)

    # =========================================================================
    # STEP 11: ARTIFACTS (Tables & Variables)
    # =========================================================================
    def step_11_artifacts(self):
        self.log("Generating Complete Manuscript Artifacts (Figures & Tables)...")
        self.artifact_gen.generate_all_manuscript_artifacts(
            final_df=self.final_df,
            models=self.models,
            feature_sets=self.feature_sets,
            df_qc=self.df_qc,
            stats_results=self.stats_results,
            best_params_=getattr(self, 'best_params_', {}),
            studies=getattr(self, 'studies', {}),
            run_config=self.run_config
        )

        self.artifact_gen.generate_final_artifacts(
            final_df=self.final_df,
            models=self.models,
            feature_sets=self.feature_sets,
            run_optuna=self.run_optuna,
            run_config=self.run_config
        )

    # =========================================================================
    # MAIN RUNNER
    # =========================================================================
    def run(self):
        self.log("Starting Phase 1 Pipeline...")
        if self.fast_mode:
            self.log(f"FAST_MODE enabled: N_TRIALS reduced to {self.n_trials}")
        self._write_run_config()
        
        try:
            self.step_1_load_data()
            self.step_2_parse_identity()
            self.step_3_qc_checks()
            self.step_4_feature_engineering()
            # Training
            self.step_5a_optuna_optimization()
            self.step_5_train_scenario_B()
            if self.run_ensembles:
                self._train_ensembles()
                pass
            
            self.step_6_rigorous_validation()
            self.step_8_advanced_plots()
            self.step_8b_calibration_analysis()
            
            self.step_10_ablation_study()
            self.step_11_artifacts()
            
            self.log("Pipeline Completed Successfully.")
        except Exception as e:
            self.log(f"CRITICAL FAILURE: {e}")
            import traceback
            traceback.print_exc()
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified Phase 1 Pipeline")
    
    # Determine default paths relative to the script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_data = os.path.join(script_dir, "Mekanik_Test_Cleaned.csv")
    default_outdir = os.path.join(script_dir, "outputs")

    parser.add_argument("--data", default=default_data, help="Input CSV path")
    parser.add_argument("--outdir", default=default_outdir, help="Output directory")
    parser.add_argument("--mode", choices=["STANDARD", "RIGOROUS", "SINGLE"], default=None, help="Run mode override")
    parser.add_argument("--enable_lit_table", action="store_true", default=False, help="Enable literature comparison table")
    parser.add_argument("--disable_lit_table", action="store_true", default=False, help="Disable literature comparison table")
    parser.add_argument("--require_pdflatex_fig1", action="store_true", default=False, help="Require pdflatex for Figure 1")
    parser.add_argument("--no_require_pdflatex_fig1", action="store_true", default=False, help="Do not require pdflatex for Figure 1")
    parser.add_argument("--shap_strict", action="store_true", default=False, help="Fail-fast on SHAP errors")
    parser.add_argument("--shap_soft", action="store_true", default=False, help="Allow SHAP fallback on error")
    
    args = parser.parse_args()
    
    # Determine execution strategy
    base_cli_config = {}
    if args.enable_lit_table:
        base_cli_config['ENABLE_LIT_TABLE'] = True
    if args.disable_lit_table:
        base_cli_config['ENABLE_LIT_TABLE'] = False
    if args.require_pdflatex_fig1:
        base_cli_config['REQUIRE_PDFLATEX_FOR_FIG1'] = True
    if args.no_require_pdflatex_fig1:
        base_cli_config['REQUIRE_PDFLATEX_FOR_FIG1'] = False
    if args.shap_strict:
        base_cli_config['SHAP_STRICT'] = True
    if args.shap_soft:
        base_cli_config['SHAP_STRICT'] = False

    if args.mode:
        if args.mode == "RIGOROUS":
            runs = [{
                'mode': 'RIGOROUS',
                'suffix': '_RIGOROUS',
                'config': {
                    'VALIDATION_MODE': 'RIGOROUS',
                    'VOTING_WEIGHTS': None
                }
            }]
        elif args.mode == "STANDARD":
            runs = [{
                'mode': 'STANDARD',
                'suffix': '_STANDARD',
                'config': {
                    'VALIDATION_MODE': 'STANDARD',
                    'VOTING_WEIGHTS': None,
                    'RUN_STATS': False,
                    'PERMUTATION_COUNT': 10
                }
            }]
        else:
            runs = [{
                'mode': 'SINGLE',
                'suffix': '',
                'config': {}
            }]
    elif CONFIG.get('RUN_DUAL_MODE', False):
        # Dual Run Configuration
        runs = [
            {
                'mode': 'RIGOROUS', 
                'suffix': '_RIGOROUS', 
                'config': {
                    'VALIDATION_MODE': 'RIGOROUS',
                    'VOTING_WEIGHTS': None # Auto-weights (Stacking)
                }
            },
            {
                'mode': 'STANDARD', 
                'suffix': '_STANDARD', 
                'config': {
                    'VALIDATION_MODE': 'STANDARD',
                    'VOTING_WEIGHTS': None,
                    'RUN_STATS': False, 
                    'PERMUTATION_COUNT': 10 
                }
            }
        ]
    else:
        # Standard Single Run (uses defaults from CONFIG)
        runs = [{
            'mode': 'SINGLE',
            'suffix': '',
            'config': {} 
        }]

    if base_cli_config:
        for r in runs:
            r['config'].update(base_cli_config)
    
    for r in runs:
        current_outdir = args.outdir.rstrip('/\\') + r['suffix']
        print(f"\n{'='*60}")
        print(f"STARTING RUN: {r['mode']}")
        print(f"Output Directory: {current_outdir}")
        print(f"{'='*60}\n")
        
        pipeline = Phase1Pipeline(
            input_file=args.data,
            outdir=current_outdir,
            override_config=r['config']
        )
        pipeline.run()
        print(f"run {r['mode']} finished.")
