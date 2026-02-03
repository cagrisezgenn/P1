
import os
import shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg') # Safe backend
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.model_selection import KFold
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.inspection import PartialDependenceDisplay, permutation_importance
import shap
import json
import re
import subprocess
import traceback
from contextlib import contextmanager

VERSION_STAMP = "2026-02-03 16:11 REV#HARDENING"

# Constants needed for plotting (can be imported or redefined if simple)
TARGETS = ['Max_Compressive_Strength_MPa', 'Hardness_HB', 'Sintered_Density_g_cm3', 'Toughness_MJ_m3']

TARGET_SHORT_NAMES = {
    'Max_Compressive_Strength_MPa': 'Strength',
    'Hardness_HB': 'Hardness',
    'Sintered_Density_g_cm3': 'Density',
    'Toughness_MJ_m3': 'Toughness'
}

TARGET_UNITS = {
    'Strength': 'MPa',
    'Hardness': 'HB',
    'Density': 'g/cm^3',
    'Toughness': 'MJ/m^3'
}

TARGET_MAP = {
    'Max_Compressive_Strength_MPa': ('Strength', ['B4C_Mean_wt', 'Green_Density_g_cm3']),
    'Hardness_HB': ('Hardness', ['B4C_Mean_wt', 'Green_Density_g_cm3']),
    'Sintered_Density_g_cm3': ('Density', ['Sinter_Temp_C', 'Green_Density_g_cm3']),
    'Toughness_MJ_m3': ('Toughness', ['Mxene_Rate_from_code', 'Green_Density_g_cm3'])
}

# Task 4 Feature Dictionary
FEATURE_DICTIONARY = [
    # Core Process
    {'name': 'Pressure_MPa', 'unit': 'MPa', 'desc': 'Sintering Pressure'},
    {'name': 'Sinter_Temp_C', 'unit': 'C', 'desc': 'Sintering Temperature'},
    {'name': 'Sinter_Time_min', 'unit': 'min', 'desc': 'Sintering Duration'},
    {'name': 'MA_Time_h', 'unit': 'h', 'desc': 'Mechanical Alloying Time'},
    # Structural
    {'name': 'Layer_Count_from_code', 'unit': '-', 'desc': 'Number of discrete layers (1,3,5,7)'},
    {'name': 'Mxene_Rate_from_code', 'unit': 'wt%', 'desc': 'Ti3C2Tx additive content'},
    {'name': 'Is_FGM', 'unit': 'Binary', 'desc': '1 if Functionally Graded, 0 if Homogeneous'},
    # B4C Derived
    {'name': 'B4C_Outer_wt', 'unit': 'wt%', 'desc': 'Boron Carbide content at outer surface'},
    {'name': 'B4C_Center_wt', 'unit': 'wt%', 'desc': 'Boron Carbide content at center core'},
    {'name': 'B4C_Mean_wt', 'unit': 'wt%', 'desc': 'Volume-averaged B4C content across layers'},
    {'name': 'B4C_Gradient_wt', 'unit': 'wt%', 'desc': 'Gradient magnitude (Outer - Center)'},
    # Layer Vectors
    {'name': 'Layer1_B4C_Rate', 'unit': 'wt%', 'desc': 'B4C content in Layer 1'},
    {'name': 'Layer2_B4C_Rate', 'unit': 'wt%', 'desc': 'B4C content in Layer 2'},
    {'name': 'Layer3_B4C_Rate', 'unit': 'wt%', 'desc': 'B4C content in Layer 3'},
    {'name': 'Layer4_B4C_Rate', 'unit': 'wt%', 'desc': 'B4C content in Layer 4'},
    {'name': 'Layer5_B4C_Rate', 'unit': 'wt%', 'desc': 'B4C content in Layer 5'},
    {'name': 'Layer6_B4C_Rate', 'unit': 'wt%', 'desc': 'B4C content in Layer 6'},
    {'name': 'Layer7_B4C_Rate', 'unit': 'wt%', 'desc': 'B4C content in Layer 7'},
    # Interactions
    {'name': 'Pressure_x_Temp', 'unit': 'Index', 'desc': 'Interaction: Pressure * Temperature'},
    {'name': 'Temp_x_Time', 'unit': 'Index', 'desc': 'Interaction: Temperature * Time'},
    {'name': 'Sinter_Energy', 'unit': 'Index', 'desc': 'Proxy for total energy input'},
    {'name': 'Layer_Complexity', 'unit': 'Index', 'desc': 'Interaction: Layer Count * Gradient'},
    # Extended
    {'name': 'Green_Density_g_cm3', 'unit': 'g/cm3', 'desc': 'Density before sintering (In-Process)'},
]

class ArtifactGenerator:
    def __init__(self, output_dirs, log_func, validation_mode='RIGOROUS', inputs_specs=None):
        """
        :param output_dirs: Dictionary of output paths (e.g. self.dirs from Phase1)
        :param log_func: Function to log messages (e.g. self.log)
        :param validation_mode: 'RIGOROUS' or 'STANDARD'
        """
        self.dirs = output_dirs
        self.log = log_func
        self.validation_mode = validation_mode
        self.INPUT_SPECS = inputs_specs if inputs_specs else {}
        self.run_config = None
        self.artifact_suffix = validation_mode.lower()
        self.expected_files = []
        self.skipped_files = []
        self._output_root = os.path.dirname(self.dirs['04_figures'])

    @property
    def output_root(self):
        return self._output_root

    def _set_run_config(self, run_config):
        if run_config:
            self.run_config = run_config
            self.artifact_suffix = str(run_config.get('ARTIFACT_SUFFIX', self.artifact_suffix)).lower()

    def _target_slug(self, target):
        return TARGET_SHORT_NAMES.get(target, target).lower()

    def _with_suffix(self, name):
        base, ext = os.path.splitext(name)
        return f"{base}_{self.artifact_suffix}{ext}"

    def _register_expected(self, rel_path):
        if rel_path not in self.expected_files:
            self.expected_files.append(rel_path)

    def _register_skipped(self, rel_path):
        if rel_path not in self.skipped_files:
            self.skipped_files.append(rel_path)

    def _log_artifact_ok(self, path):
        try:
            size = os.path.getsize(path)
        except Exception:
            size = -1
        self.log(f"[ARTIFACT_OK] path={path} bytes={size}")

    def _log_artifact_fail(self, name, err):
        tb = traceback.format_exc()
        self.log(f"[ARTIFACT_FAIL] name={name} error={err} traceback={tb}")

    def _resolve_transformed_feature_names(self, preproc, feature_names, n_out):
        col_names = None
        if hasattr(preproc, "get_feature_names_out"):
            try:
                col_names = list(preproc.get_feature_names_out(feature_names))
            except TypeError:
                col_names = list(preproc.get_feature_names_out())
            except Exception:
                col_names = None

        if (col_names is None) or (len(col_names) != n_out):
            # Fallback 1: Use feature_names if length matches
            if len(feature_names) == n_out:
                col_names = list(feature_names)
            # Fallback 2: Generate generic names
            else:
                col_names = [f"f{i}" for i in range(n_out)]
        return col_names

    def _resolve_raw_scatter_features(self, df_qc, target):
        """Determine x-axis features for raw scatter plots"""
        # 1. Config override
        if self.run_config:
            custom_map = self.run_config.get('raw_scatter_features_by_target', {})
            if target in custom_map:
                return custom_map[target]
        
        # 2. Hardcoded logic (TARGET_MAP)
        if target in TARGET_MAP:
            return TARGET_MAP[target][1]
            
        # 3. Fallback: Numeric columns excluding known exclusions
        excludes = set(TARGETS + ['Sample_Code', 'Phase', 'sample_weight'])
        candidates = [c for c in df_qc.columns if c not in excludes and pd.api.types.is_numeric_dtype(df_qc[c])]
        return candidates[:4] # Return first 4

    def plot_raw_physical_scatters(self, df_qc, final_df=None, run_config=None):
        """Generates raw data scatter plots (Physical Trend Graphs)"""
        self.log("  Plotting Raw Physical Trend Graphs...")
        
        for target in TARGETS:
            if target not in df_qc.columns:
                continue
                
            x_feats = self._resolve_raw_scatter_features(df_qc, target)
            if not x_feats:
                continue
                
            slug = self._target_slug(target)
            fname = self._with_suffix(f"raw_scatter_{slug}.pdf")
            
            # Create figure
            n = len(x_feats)
            rows = (n + 1) // 2
            fig, axes = plt.subplots(rows, 2, figsize=(10, 4*rows))
            axes = axes.flatten()
            
            for i, feat in enumerate(x_feats):
                if feat not in df_qc.columns:
                    continue
                ax = axes[i]
                # Plot All Data (from df_qc which has all phases)
                sns.scatterplot(data=df_qc, x=feat, y=target, hue='Phase', style='Phase', ax=ax, alpha=0.7)
                
                # Trend line if enough points
                df_clean = df_qc[[feat, target]].dropna()
                if len(df_clean) > 5:
                    try:
                        z = np.polyfit(df_clean[feat], df_clean[target], 1)
                        p = np.poly1d(z)
                        x_range = np.linspace(df_clean[feat].min(), df_clean[feat].max(), 100)
                        ax.plot(x_range, p(x_range), "r--", alpha=0.5, label='Trend')
                    except:
                        pass
                        
                ax.set_title(f"{TARGET_SHORT_NAMES.get(target, target)} vs {feat}")
                
            for i in range(n, len(axes)):
                axes[i].axis('off')
                
            plt.tight_layout()
            self._save_plot(fname)

    def export_samplewise_prediction_table(self, models, final_df, top_n=10, validation_mode="STANDARD"):
        """Export sample-wise predictions and errors"""
        self.log(f"  Exporting Sample-wise Prediction Table (Top {top_n})...")
        
        frames = []
        for target in models:
            info = models[target]
            
            # Support both OOF key styles
            oof_data = info.get('oof', {})
            if not oof_data:
                # Try fallback keys
                if 'oof_indices' in info:
                    oof_data = {
                        'indices': info['oof_indices'],
                        'y_true': info['oof_y_true'],
                        'y_pred': info['oof_y_pred']
                    }
            
            if not oof_data or 'y_pred' not in oof_data:
                self.log(f"    Warning: No OOF data for {target}, skipping table export.")
                continue

            indices = oof_data['indices']
            y_true = oof_data['y_true']
            y_pred = oof_data['y_pred']
            
            # Create DataFrame
            df_res = pd.DataFrame({
                'Target': target,
                'Ref_Index': indices,
                'Actual': y_true,
                'Predicted': y_pred
            })
            
            # Merge with Sample Code if available
            if 'Sample_Code' in final_df.columns:
                codes = final_df.loc[indices, 'Sample_Code'].values
                df_res.insert(0, 'Sample_Code', codes)
                
            # Calc Errors
            df_res['Abs_Error'] = (df_res['Actual'] - df_res['Predicted']).abs()
            df_res['Pct_Error'] = (df_res['Abs_Error'] / df_res['Actual'].replace(0, np.nan)).abs() * 100
            
            frames.append(df_res)
            
        if not frames:
            raise RuntimeError("No OOF data available for any target to generate Sample-wise table.")
            
        full_df = pd.concat(frames, ignore_index=True)
        
        # Sort by Abs_Error descending
        full_df.sort_values('Abs_Error', ascending=False, inplace=True)
        
        # Filter Top N
        if top_n:
            top_df = full_df.groupby('Target').apply(lambda x: x.nlargest(top_n, 'Abs_Error')).reset_index(drop=True)
        else:
            top_df = full_df
            
        # Save CSV
        out_dir = self.dirs.get('03_tables')
        if out_dir is None:
            out_dir = os.path.join(self._output_root, '03_tables')
            os.makedirs(out_dir, exist_ok=True)
            self.dirs['03_tables'] = out_dir

        csv_name = f"samplewise_top_errors_{self.artifact_suffix}.csv"
        csv_path = os.path.join(out_dir, csv_name)
        top_df.to_csv(csv_path, index=False)
        self._log_artifact_ok(csv_path)
        self._register_expected(os.path.join('03_tables', csv_name))
        
        # Save TeX
        tex_name = f"samplewise_top_errors_{self.artifact_suffix}.tex"
        tex_path = os.path.join(self.dirs['latex'], tex_name)
        
        # Format for latex
        latex_df = top_df.copy()
        float_format = "%.2f"
        latex_content = latex_df.to_latex(index=False, float_format=float_format, escape=False)
        
        with open(tex_path, 'w') as f:
            f.write(latex_content)
        self._log_artifact_ok(tex_path)
        self._register_expected(os.path.join('latex', tex_name))

    def _write_shap_placeholder(self, slug, target):
        msg = f"SHAP failed for target={target}. See logs for traceback."
        plt.figure(figsize=(10, 4))
        plt.axis('off')
        plt.text(0.5, 0.5, msg, ha='center', va='center', wrap=True)
        self._save_plot(self._with_suffix(f"shap_{slug}.pdf"))

    def _write_text(self, filename: str, content: str, subdir_key: str = "latex") -> str:
        """Write text content to file and register as expected artifact."""
        base = self.dirs.get(subdir_key)
        if base is None:
            # Fallback: join outputs root
            base = os.path.join(self._output_root, subdir_key)
            os.makedirs(base, exist_ok=True)
        path = os.path.join(base, filename)
        with open(path, 'w', encoding='utf-8', newline='\n') as f:
            f.write(content)
        rel_path = os.path.join(subdir_key, filename)
        self._register_expected(rel_path)
        self._log_artifact_ok(path)
        return path

    def _write_csv(self, filename, df):
        path = os.path.join(self.dirs['latex'], filename)
        df.to_csv(path, index=False)
        self._register_expected(os.path.join('latex', filename))
        self._log_artifact_ok(path)
        return path

    def _write_manifest(self):
        produced_files = []
        for key in ['04_figures', 'latex']:
            root = self.dirs.get(key)
            if not root or not os.path.exists(root):
                continue
            for fname in os.listdir(root):
                rel = os.path.join(key, fname)
                produced_files.append(rel)

        missing_files = [
            f for f in self.expected_files
            if f not in produced_files and f not in self.skipped_files
        ]

        manifest = {
            'expected_files': sorted(self.expected_files),
            'produced_files': sorted(produced_files),
            'missing_files': sorted(missing_files),
            'skipped_files': sorted(self.skipped_files)
        }
        manifest_path = os.path.join(self._output_root, f"artifact_manifest_{self.artifact_suffix}.json")
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        self._log_artifact_ok(manifest_path)
        return missing_files

    def _is_tree_estimator(self, estimator):
        return isinstance(estimator, (RandomForestRegressor, XGBRegressor, HistGradientBoostingRegressor))

    def _extract_estimator(self, model):
        if isinstance(model, Pipeline):
            estimator = model.steps[-1][1]
            preproc = None
            if len(model.steps) > 1:
                preproc = Pipeline(model.steps[:-1])
            return estimator, preproc
        return model, None
        
    def _save_plot(self, filename):
        fig_path = os.path.join(self.dirs['04_figures'], filename)
        latex_path = os.path.join(self.dirs['latex'], filename)
        self._register_expected(os.path.join('04_figures', filename))
        self._register_expected(os.path.join('latex', filename))
        plt.savefig(fig_path, format='pdf', bbox_inches='tight')
        try:
            shutil.copy2(fig_path, latex_path)
        except Exception as e:
            self._log_artifact_fail(filename, e)
            raise
        finally:
            plt.close()
        self._log_artifact_ok(fig_path)

    def _set_plot_style(self):
        plt.style.use('seaborn-v0_8-paper')
        plt.rcParams.update({
            'font.family': 'serif',
            'axes.grid': True,
            'grid.alpha': 0.3,
            'axes.labelsize': 10,
            'font.size': 10,
            'legend.fontsize': 8,
            'xtick.labelsize': 8,
            'ytick.labelsize': 8,
            'figure.constrained_layout.use': False
        })
        # Explicitly force matplotlib global params to avoid layout conflicts
        plt.rcParams['figure.constrained_layout.use'] = False
        plt.rcParams['figure.autolayout'] = False

    def plot_boxplots(self, df, mode='input'):
        """Generates distribution boxplots"""
        self.log(f"  Plotting {mode} distributions...")
        
        if mode == 'input':
            cols = [c for c in df.columns if c not in TARGETS + ['Sample_Code', 'Phase', 'sample_weight']]
            cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
            fname = self._with_suffix("input_distributions.pdf")
            title = "Input Feature Distributions"
        else:
            cols = [c for c in TARGETS if c in df.columns]
            fname = self._with_suffix("target_distributions.pdf")
            title = "Target Variable Distributions"
            
        num_cols = len(cols)
        if num_cols == 0: return
        
        rows = (num_cols + 2) // 3
        fig, axes = plt.subplots(rows, 3, figsize=(12, 3.5*rows))
        axes = axes.flatten()
        
        for i, col in enumerate(cols):
             sns.boxplot(y=df[col].dropna(), ax=axes[i], color='lightblue', width=0.5)
             sns.stripplot(y=df[col].dropna(), ax=axes[i], color='black', alpha=0.3, size=3)
             axes[i].set_title(col, fontweight='bold', fontsize=10)
             axes[i].set_ylabel(col)
        
        for i in range(num_cols, len(axes)): axes[i].axis('off')
        
        plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
        self._save_plot(fname)

    def plot_correlation_matrix(self, df):
        """Generates correlation matrix PDF"""
        self.log("  Plotting correlation matrix...")
        numeric_df = df.select_dtypes(include=[np.number])
        if numeric_df.empty: return

        plt.figure(figsize=(14, 12))
        corr = numeric_df.corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, cmap='coolwarm', center=0, square=True, annot=False, linewidths=.5)
        plt.title('Correlation Matrix', fontsize=16)
        self._save_plot(self._with_suffix("correlation_matrix.pdf"))

    def plot_pdp(self, model, X, target):
        """Generates PDP for a single target"""
        short_name, features_to_plot = TARGET_MAP.get(target, (target, []))
        
        valid_feats = [f for f in features_to_plot if f in X.columns]
        if not valid_feats: 
            return

        fig, ax = plt.subplots(figsize=(10, 4))
        PartialDependenceDisplay.from_estimator(model, X, valid_feats, grid_resolution=20, ax=ax)
        plt.suptitle(f'Partial Dependence: {target}', fontsize=12, fontweight='bold', y=1.05)
        plt.tight_layout()
        self._save_plot(self._with_suffix(f"pdp_{short_name.lower()}.pdf"))
    def plot_pdp_2d(self, model, X, target):
        """Generates 2D Partial Dependence Plot (Interaction)"""
        short_name, features_to_plot = TARGET_MAP.get(target, (target, []))
        
        # Pick top 2 features for interaction if available, or specific physics pair
        # Physics pair: Pressure vs Temp is classic sintering interaction
        interaction_pair = ['Pressure_MPa', 'Sinter_Temp_C']
        
        # Check if they exist
        valid = [f for f in interaction_pair if f in X.columns]
        if len(valid) < 2:
            # Fallback to top 2 mapped features
            valid = [f for f in features_to_plot if f in X.columns][:2]
            
        if len(valid) < 2: return

        fig, ax = plt.subplots(figsize=(7, 6))
        # Pass list of tuples for interactions
        PartialDependenceDisplay.from_estimator(model, X, [tuple(valid)], ax=ax)
        plt.suptitle(f'Interaction: {valid[0]} vs {valid[1]}', fontsize=12, fontweight='bold', y=1.02)
        plt.tight_layout()
        self._save_plot(self._with_suffix(f"pdp_2d_{short_name.lower()}.pdf"))

    def plot_parity_single(self, y_true, y_pred, target, r2, rmse):
        """Plot single parity plot (legacy method, but useful)"""
        plt.figure(figsize=(6, 6))
        plt.scatter(y_true, y_pred, alpha=0.6, color='blue', edgecolors='k')
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        margin = (max_val - min_val) * 0.1
        plt.plot([min_val-margin, max_val+margin], [min_val-margin, max_val+margin], 'r--', lw=2)
        plt.xlabel(f'Experimental {TARGET_SHORT_NAMES.get(target,target)}')
        plt.ylabel(f'Predicted {TARGET_SHORT_NAMES.get(target,target)}')
        plt.title(f'{target}\n$R^2={r2:.3f}$, RMSE={rmse:.3f}')
        plt.axis('square')
        plt.grid(True, which='both', linestyle='--', alpha=0.7)
        self._save_plot(self._with_suffix(f"parity_{TARGET_SHORT_NAMES.get(target,target)}.pdf"))

    # =========================================================================
    # HIGH-LEVEL ORCHESTRATION METHODS (Moved from Phase1)
    # =========================================================================

    @contextmanager
    def _disable_tight_layout(self):
        original_tight_layout = plt.tight_layout
        original_fig_tight_layout = matplotlib.figure.Figure.tight_layout
        try:
            plt.tight_layout = lambda *args, **kwargs: None
            matplotlib.figure.Figure.tight_layout = lambda *args, **kwargs: None
            yield
        finally:
            plt.tight_layout = original_tight_layout
            matplotlib.figure.Figure.tight_layout = original_fig_tight_layout

    def _render_shap_summary_pdf(self, shap_values, X_shap, slug, target, estimator):
        """Helper to safely render SHAP summary plot with fallback"""
        self.log(f"  Rendering SHAP summary for {target} (estimator={type(estimator).__name__})...")
        
        # Log current layout setting
        layout_setting = plt.rcParams.get('figure.constrained_layout.use')
        self.log(f"    Current figure.constrained_layout.use: {layout_setting}")

        try:
            # Attempt 1: Beeswarm with context managers
            with plt.rc_context({'figure.constrained_layout.use': False, 'figure.autolayout': False}):
                with self._disable_tight_layout():
                    fig = plt.figure(figsize=(10, 8), constrained_layout=False)
                    try:
                        # Ensure internal layout engine is tight (or none) but protected by monkeypatch
                        fig.set_layout_engine('tight')
                    except Exception:
                        pass
                    
                    shap.summary_plot(shap_values, X_shap, show=False)
                    self._save_plot(self._with_suffix(f"shap_{slug}.pdf"))
                    
        except RuntimeError as e:
            if "Colorbar layout" in str(e) or "layout engine" in str(e):
                self.log(f"    SHAP beeswarm failed due to layout engine; fallback to bar plot. target={target}, reason={e}")
                plt.close('all')

                # Attempt 2: Manual Bar Plot Fallback
                try:
                    with plt.rc_context({'figure.constrained_layout.use': False, 'figure.autolayout': False}):
                        vals = np.abs(shap_values)
                        if vals.ndim > 2: vals = vals.mean(2)
                        if vals.ndim == 3: vals = vals.mean(2) # Safety

                        importance_vals = np.mean(vals, axis=0)
                        feature_importance = pd.DataFrame(
                            list(zip(X_shap.columns, importance_vals)), 
                            columns=['col_name','feature_importance_vals']
                        )
                        feature_importance.sort_values(by=['feature_importance_vals'], ascending=False, inplace=True)
                        feature_importance = feature_importance.head(20)

                        fig2 = plt.figure(figsize=(10, 8), constrained_layout=False)
                        try:
                            fig2.set_layout_engine('tight')
                        except Exception:
                            pass

                        plt.barh(feature_importance['col_name'], feature_importance['feature_importance_vals'], color='teal')
                        plt.xlabel("mean(|SHAP value|) (average impact on model output magnitude)")
                        plt.gca().invert_yaxis()
                        plt.title(f"SHAP Feature Importance: {target}")
                        
                        self._save_plot(self._with_suffix(f"shap_{slug}.pdf"))

                except Exception as e2:
                    self.log(f"    SHAP manual fallback also failed: {e2}")
                    self._handle_shap_failure(slug, target, e2)
            else:
                 self._handle_shap_failure(slug, target, e)

        except Exception as e:
             self._handle_shap_failure(slug, target, e)

    def _handle_shap_failure(self, slug, target, e):
        self.log(f"SHAP failed: target={target} error={e}")
        self._log_artifact_fail(f"shap_{slug}", e)
        
        shap_strict = True
        if self.run_config is not None:
             shap_strict = bool(self.run_config.get('SHAP_STRICT', True))
        
        if shap_strict:
            raise RuntimeError(f"SHAP failed for target {target}") from e
        
        self.log(f"    SHAP_STRICT=False, generating placeholder for {target}")
        self._write_shap_placeholder(slug, target)
        self._register_skipped(os.path.join('04_figures', self._with_suffix(f"shap_{slug}.pdf")))
        self._register_skipped(os.path.join('latex', self._with_suffix(f"shap_{slug}.pdf")))

    def run_advanced_plots(self, final_df, models, feature_sets, df_qc, run_vector_plots, run_pdp, run_shap, best_params_, run_config):
        self._set_run_config(run_config)
        self._set_plot_style()
        self.log("Generating Advanced Plots (Vector/Literature Style) via Figure_Table.py...")

        # 1. Distributions & Correlation
        if run_vector_plots and df_qc is not None:
            self.plot_boxplots(df_qc, mode='input')
            self.plot_boxplots(df_qc, mode='target')
            self.plot_correlation_matrix(df_qc)
            
            # Raw Physical Scatter Plots
            if self.run_config.get("enable_raw_scatter_plots", True):
                self.plot_raw_physical_scatters(df_qc, final_df=final_df, run_config=self.run_config)

        # 2. Model-Based Plots
        for target in TARGETS:
            if target not in models:
                continue

            info = models[target]
            model = info['best_model']
            feature_set = info['feature_set']
            feature_names = info['feature_names']

            X_all = feature_sets.get(feature_set)
            if X_all is None or X_all.empty:
                continue

            if info.get('oof', {}).get('indices') is not None:
                idx = info['oof']['indices']
            else:
                idx = final_df.index[final_df[target].notna()]

            X = X_all.loc[idx, feature_names]
            y_true = info.get('oof', {}).get('y_true', final_df.loc[idx, target].to_numpy())

            slug = self._target_slug(target)

            if run_pdp:
                self.plot_pdp(model, X, target)
                self.plot_pdp_2d(model, X, target)

            if run_shap:
                estimator, preproc = self._extract_estimator(model)
                if not self._is_tree_estimator(estimator):
                    self.log(f"SHAP skipped: reason=non_tree_estimator, estimator={type(estimator).__name__}, target={target}")
                    shap_name = self._with_suffix(f"shap_{slug}.pdf")
                    self._register_expected(os.path.join('04_figures', shap_name))
                    self._register_expected(os.path.join('latex', shap_name))
                    self._register_skipped(os.path.join('04_figures', shap_name))
                    self._register_skipped(os.path.join('latex', shap_name))
                    self.plot_feature_importance_single(model, X, y_true, target)
                    if not run_pdp:
                        self.plot_pdp(model, X, target)
                        self.plot_pdp_2d(model, X, target)
                    continue

                try:
                    X_shap = X
                    if preproc is not None:
                        X_trans = preproc.transform(X)
                        col_names = self._resolve_transformed_feature_names(preproc, feature_names, X_trans.shape[1])
                        X_shap = pd.DataFrame(X_trans, columns=col_names, index=X.index)
                    
                    self.log(
                        f"SHAP input aligned: target={target}, "
                        f"X_in={X.shape}, X_trans={X_shap.shape}, names={len(X_shap.columns)}"
                    )
                    
                    explainer = shap.TreeExplainer(estimator)
                    shap_values = explainer.shap_values(X_shap)
                    
                    # Call new helper
                    self._render_shap_summary_pdf(shap_values, X_shap, slug, target, estimator)

                except Exception as e:
                    self._handle_shap_failure(slug, target, e)

    def plot_permutation_hist(self, models):
        """Generates Permutation Test Null Distribution Plots"""
        self.log("  Plotting Permutation Test Histograms...")

        for target in TARGETS:
            if target not in models:
                continue
            perm = models[target].get('perm_test', {})
            scores_raw = perm.get('perm_scores', '[]')
            if not scores_raw:
                continue
            if isinstance(scores_raw, str):
                try:
                    scores = json.loads(scores_raw)
                except Exception:
                    scores = []
            else:
                scores = scores_raw

            if not scores:
                continue

            obs_score = perm.get('original_score', 0)
            p_val = perm.get('p_value', 1.0)

            short = TARGET_SHORT_NAMES.get(target, target)

            plt.figure(figsize=(8, 6))
            plt.hist(scores, bins=30, alpha=0.7, color='gray', edgecolor='black', label='Null Distribution')
            plt.axvline(obs_score, color='red', linestyle='--', linewidth=2, label=f'Observed (R2={obs_score:.2f})')

            plt.title(f"Permutation Test: {short}\np-value = {p_val:.4f} (N={len(scores)})")
            plt.xlabel("R2 Score")
            plt.ylabel("Frequency")
            plt.legend()
            self._save_plot(self._with_suffix(f"perm_test_{short.lower()}.pdf"))

    def plot_feature_importance_single(self, model, X, y, target):
        """Generates Feature Importance Plot for a single target"""
        slug = self._target_slug(target)
        fname = self._with_suffix(f"importance_{slug}.pdf")
        fig_path = os.path.join(self.dirs['04_figures'], fname)
        if os.path.exists(fig_path):
            return

        try:
            res = permutation_importance(model, X, y, n_repeats=10, random_state=42, n_jobs=-1)
            start_indices = np.argsort(res.importances_mean)[::-1]
            top_indices = start_indices[:15]

            top_feats = [X.columns[i] for i in top_indices]
            top_scores = res.importances_mean[top_indices]
            top_stds = res.importances_std[top_indices]

            plt.figure(figsize=(10, 8))
            plt.barh(range(len(top_feats)), top_scores, xerr=top_stds, align='center', color='teal', alpha=0.8)
            plt.yticks(range(len(top_feats)), top_feats)
            plt.gca().invert_yaxis()
            plt.xlabel("Mean Decrease in R2")
            plt.title(f"Feature Importance: {TARGET_SHORT_NAMES.get(target, target)}")
            plt.tight_layout()
            self._save_plot(fname)
        except Exception as e:
            self.log(f"  Importance calculation failed for {target}: {e}")

    def plot_feature_importance(self, models, feature_sets, final_df):
        """Generates Feature Importance Plots (Permutation Importance on OOF)"""
        self.log("  Plotting Feature Importance...")

        for t in TARGETS:
            if t not in models:
                continue
            info = models[t]
            model = info['best_model']
            feature_set = info['feature_set']
            feature_names = info['feature_names']
            X_all = feature_sets.get(feature_set)
            if X_all is None or X_all.empty:
                continue

            if info.get('oof', {}).get('indices') is not None:
                idx = info['oof']['indices']
                y = info['oof']['y_true']
            else:
                idx = final_df.index[final_df[t].notna()]
                y = final_df.loc[idx, t].to_numpy()

            X = X_all.loc[idx, feature_names]
            self.plot_feature_importance_single(model, X, y, t)

    def generate_table_1_stats(self, df_qc):
        """Generates Table 1: Dataset Summary Statistics"""
        self.log("  Generating Table 1 (Dataset Summary)...")
        
        # Columns to summarize
        input_cols = ['Layer_Count_from_code', 'Mxene_Rate_from_code', 'Pressure_MPa', 
                      'Sinter_Temp_C', 'Sinter_Time_min', 'MA_Time_h', 'Green_Density_g_cm3']
        
        # Friendly Names
        name_map = {
            'Layer_Count_from_code': 'Layer Count',
            'Mxene_Rate_from_code': 'Mxene Rate (wt%)',
            'Pressure_MPa': 'Pressure (MPa)',
            'Sinter_Temp_C': 'Sinter Temp (C)',
            'Sinter_Time_min': 'Sinter Time (min)',
            'MA_Time_h': 'MA Time (h)',
            'Green_Density_g_cm3': 'Green Density (g/cm3)',
            'Max_Compressive_Strength_MPa': 'Comp. Strength (MPa)',
            'Hardness_HB': 'Hardness (HB)',
            'Sintered_Density_g_cm3': 'Sintered Density (g/cm3)',
            'Toughness_MJ_m3': 'Toughness (MJ/m3)'
        }
        
        cols = input_cols + TARGETS
        valid_cols = [c for c in cols if c in df_qc.columns]
        
        stats = []
        for c in valid_cols:
            s_ = df_qc[c].describe()
            row = {
                'Variable': name_map.get(c, c),
                'N': int(s_['count']),
                'Min': f"{s_['min']:.2f}",
                'Max': f"{s_['max']:.2f}",
                'Mean': f"{s_['mean']:.2f}",
                'Std': f"{s_['std']:.2f}"
            }
            stats.append(row)
            
        df_stats = pd.DataFrame(stats)
        tex = df_stats.to_latex(index=False)
        if self.validation_mode == 'STANDARD':
            tex = "% DEBUG/DIAGNOSTIC (STANDARD MODE)\n" + tex
        self._write_text(self._with_suffix('table_1_stats.tex'), tex)

    # =========================================================================
    # TASK 4: Feature Dictionary -> Table 2
    # =========================================================================
    def generate_table_2_features(self):
        """Generates Table 2: Feature Definitions from Constant"""
        self.log("  Generating Table 2 (Features)...")
        
        # FEATURE_DICTIONARY is defined globally or class level
        # Assuming global
        
        df_feats = pd.DataFrame(FEATURE_DICTIONARY)
        # Rename for Latex
        df_feats = df_feats.rename(columns={'name': 'Feature Name', 'unit': 'Unit', 'desc': 'Description'})
        
        tex = df_feats.to_latex(index=False, column_format='l l p{8cm}')
        if self.validation_mode == 'STANDARD':
            tex = "% DEBUG/DIAGNOSTIC (STANDARD MODE)\n" + tex
        self._write_text(self._with_suffix('table_2_features.tex'), tex)

    # =========================================================================
    # TASK 7: Schematic PDF Generation
    # =========================================================================
    def generate_fig1_schematic_script(self):
        """Generates Fig 1 TikZ script and attempts to compile it"""
        self.log("  Generating Figure 1 (Schematic)...")
        
        tikz_content = r"""
\documentclass[tikz,border=2mm]{standalone}
\usepackage{tikz}
\usetikzlibrary{shapes,arrows,positioning,automata,shadows}
\begin{document}
\begin{tikzpicture}[
    node distance=1.5cm,
    auto,
    block/.style={rectangle, draw, fill=blue!10, text width=6em, text centered, rounded corners, minimum height=3em, drop shadow},
    cloud/.style={draw, ellipse, fill=red!10, node distance=2cm, minimum height=2em, drop shadow},
    line/.style={draw, -latex', thick}
]
    % Nodes
    \node [cloud] (start) {Raw Inputs};
    \node [block, below of=start] (prep) {Data Cleaning \\ \& QC};
    \node [block, below of=prep] (feat) {Physics-Informed \\ Feature Eng.};
    \node [block, below of=feat] (model) {Nested CV \\ Optimization};
    \node [block, left of=model, node distance=3.5cm] (perm) {Permutation \\ Tests};
    \node [block, right of=model, node distance=3.5cm] (expl) {SHAP \& PDP \\ Explainability};
    \node [cloud, below of=model] (end) {Final Artifacts};

    % Paths
    \path [line] (start) -- (prep);
    \path [line] (prep) -- (feat);
    \path [line] (feat) -- (model);
    \path [line] (model) -- (end);
    \path [line] (model) -- (perm);
    \path [line] (model) -- (expl);
    \path [line] (perm) |- (end);
    \path [line] (expl) |- (end);

\end{tikzpicture}
\end{document}
        """

        base_name = f"pipeline_schematic_fig1_{self.artifact_suffix}"
        tex_name = f"{base_name}.tex"
        pdf_name = f"{base_name}.pdf"
        tex_path = os.path.join(self.dirs['latex'], tex_name)
        with open(tex_path, 'w') as f:
            f.write(tikz_content)

        self.log(f"    Saved {tex_name}")

        require_pdflatex = False
        if self.run_config is not None:
            require_pdflatex = bool(self.run_config.get('REQUIRE_PDFLATEX_FOR_FIG1', False))

        if shutil.which("pdflatex") is None:
            if require_pdflatex:
                msg = "pdflatex not found. Install TeX Live/MiKTeX or set REQUIRE_PDFLATEX_FOR_FIG1=false."
                self.log(f"[ARTIFACT_FAIL] name=pipeline_schematic_fig1 error={msg} traceback=")
                raise RuntimeError(msg)
            self.log("[ARTIFACT_SKIP] pipeline_schematic_fig1: pdflatex not found; skipping PDF build (REQUIRE_PDFLATEX_FOR_FIG1=false).")
            self._register_skipped(os.path.join('04_figures', pdf_name))
            self._register_skipped(os.path.join('latex', pdf_name))
            return

        try:
            cmd = ['pdflatex', '-interaction=nonstopmode', '-output-directory', self.dirs['latex'], tex_path]
            subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        except FileNotFoundError as e:
            msg = "pdflatex not found. Install a TeX distribution to compile Figure 1."
            self.log(msg)
            raise RuntimeError(msg) from e
        except subprocess.CalledProcessError as e:
            msg = f"pdflatex failed with exit code {e.returncode}."
            self.log(msg)
            raise RuntimeError(msg) from e

        src = os.path.join(self.dirs['latex'], pdf_name)
        if not os.path.exists(src):
            msg = f"pdflatex did not produce {pdf_name}."
            self.log(msg)
            raise RuntimeError(msg)

        dst = os.path.join(self.dirs['04_figures'], pdf_name)
        shutil.copy2(src, dst)
        self._register_expected(os.path.join('04_figures', pdf_name))
        self._register_expected(os.path.join('latex', pdf_name))
        self._log_artifact_ok(src)

    def update_parity_with_pi(self):
        """Updates Parity Plots to include PI bands (Already handled in new generate_final_artifacts logic?)"""
        pass # The logic in generate_final_artifacts is currently scatter only. 
             # We should update generate_final_artifacts to add error bars if available.
             
    def plot_optuna_history(self, studies):
        """Generates Figure 9: Optuna Optimization History"""
        self.log("  Plotting Optuna History (Figure 9)...")
        if not studies: return
        
        for t, study in studies.items():
            try:
                # Extract history
                if hasattr(study, 'trials_dataframe'):
                    df_trials = study.trials_dataframe()
                    # Filter completed
                    df_trials = df_trials[df_trials['state'] == 'COMPLETE']
                    if df_trials.empty: continue
                    
                    # Plot
                    plt.figure(figsize=(10, 6))
                    plt.plot(df_trials['number'], df_trials['value'], 'o', color='lightblue', alpha=0.5, label='Trials')
                    
                    # Cummax line (Best so far)
                    best_so_far = df_trials['value'].cummax()
                    plt.plot(df_trials['number'], best_so_far, color='red', linewidth=2, label='Best So Far')
                    
                    plt.title(f"Optimization History: {TARGET_SHORT_NAMES.get(t, t)}")
                    plt.xlabel("Trial Number")
                    plt.ylabel("Validation R2 Score")
                    plt.legend()
                    plt.grid(True, linestyle=':', alpha=0.6)
                    self._save_plot(self._with_suffix(f"optuna_history_{TARGET_SHORT_NAMES.get(t,t).lower()}.pdf"))
            except Exception as e:
                self.log(f"    Could not plot Optuna history for {t}: {e}")

    def plot_outlier_index(self, models, feature_sets, final_df):
        """Generates Index vs Standardized Residual Plot to spot outlier samples"""
        self.log("  Plotting Outlier Index (Sample IDs)...")
        
        for t in TARGETS:
            if t not in models:
                continue
            info = models[t]
            oof = info.get('oof')
            if not oof:
                continue

            y_true = np.array(oof.get('y_true', []))
            y_pred = np.array(oof.get('y_pred', []))
            indices = oof.get('indices', [])
            if len(y_true) == 0 or len(y_pred) == 0:
                continue

            if 'Sample_Code' in final_df.columns and len(indices) == len(y_true):
                sample_codes = final_df.loc[indices, 'Sample_Code'].values
            else:
                sample_codes = np.arange(len(y_true))
            
            residuals = y_true - y_pred
            std_res = residuals / (residuals.std() + 1e-9)
            
            plt.figure(figsize=(10, 5))
            plt.scatter(range(len(std_res)), std_res, color='blue', alpha=0.6)
            plt.axhline(0, color='black', alpha=0.3, linestyle='--')
            plt.axhline(2, color='red', linestyle=':', label='2 SD')
            plt.axhline(-2, color='red', linestyle=':')
            
            # Annotate outliers > 2 SD
            outlier_indices = np.where(np.abs(std_res) > 2)[0]
            for idx in outlier_indices:
                code = str(sample_codes[idx]) if idx < len(sample_codes) else str(idx)
                plt.annotate(code, (idx, std_res[idx]), xytext=(5, 5), textcoords='offset points', fontsize=8, color='red')
            
            plt.ylabel("Standardized Residuals (Z-score)")
            plt.xlabel("Sample Index")
            plt.title(f"{TARGET_SHORT_NAMES.get(t,t)}: Residuals by Sample ID")
            plt.legend()
            self._save_plot(self._with_suffix(f"outlier_analysis_{TARGET_SHORT_NAMES.get(t,t).lower()}.pdf"))

    # =========================================================================
    # TASK 2: Table S1 (Outer Fold Summary) - Real Splits
    # =========================================================================
    def generate_table_s1_folds(self, final_df, models):
        """Generates Table S1 using stored outer fold indices"""
        if self.validation_mode != 'RIGOROUS':
            return

        self.log("  Generating Table S1 (Fold Summary)...")

        fold_indices = None
        for t in TARGETS:
            info = models.get(t, {})
            oof = info.get('oof', {})
            if oof.get('fold_indices'):
                fold_indices = oof.get('fold_indices')
                break

        if not fold_indices:
            self.log("    ERROR: No fold indices available for Table S1.")
            return

        stats = []
        for i, te_idx in enumerate(fold_indices):
            te_data = final_df.loc[te_idx]
            row = {'Fold': i + 1, 'N_Test': len(te_data)}

            counts = te_data['Layer_Count_from_code'].value_counts()
            for k in [1, 3, 5, 7]:
                row[f"{k}-L"] = int(counts.get(k, 0))

            for t in TARGETS:
                if t in te_data.columns:
                    short = TARGET_SHORT_NAMES.get(t, t)
                    row[f"{short}"] = f"{te_data[t].mean():.1f}"
            stats.append(row)

        df_folds = pd.DataFrame(stats)
        cols = ['Fold', 'N_Test', '1-L', '3-L', '5-L', '7-L'] + [TARGET_SHORT_NAMES.get(t, t) for t in TARGETS]
        cols = [c for c in cols if c in df_folds.columns]
        df_folds = df_folds[cols]

        tex = df_folds.to_latex(index=False)
        self._write_text(self._with_suffix('table_s1_folds.tex'), tex)

    # =========================================================================
    # TASK 5: Literature Comparison - From CSV
    # =========================================================================
    def generate_table_lit_comparison(self, models, run_config):
        """Generates Table 5 from CSV + Dynamic Row"""
        enable_lit = True
        if run_config is not None:
            enable_lit = bool(run_config.get('ENABLE_LIT_TABLE', True))

        if not enable_lit:
            self.log("[CONFIG] Table 5 disabled (ENABLE_LIT_TABLE=false). Skipping literature comparison.")
            return

        self.log("  Generating Table 5 (Literature Comparison)...")

        csv_name = 'literature_comparison.csv'
        lit_name = self._with_suffix('lit_comparison_full.tex')
        self._register_expected(os.path.join('latex', lit_name))
        possible_paths = [
            os.path.join(self._output_root, csv_name),
            os.path.join(os.path.dirname(self._output_root), csv_name),
            csv_name
        ]

        csv_path = None
        for p in possible_paths:
            if os.path.exists(p):
                csv_path = p
                break

        if not csv_path:
            msg = "Missing required file: literature_comparison.csv. Provide it or disable Table 5 in config."
            self.log(f"  {msg}")
            raise RuntimeError(msg)

        df_lit = pd.read_csv(csv_path)
        rows = df_lit.to_dict('records')

        if TARGETS[0] in models:
            info = models[TARGETS[0]]
            r2 = info.get('oof', {}).get('r2', 0)
            metric_val = f"{r2:.2f}"
            val_type = "Nested CV (5x3)" if self.validation_mode == 'RIGOROUS' else "Train/Test"
        else:
            metric_val = "-"
            val_type = "-"

        this_work = {
            "Ref": "This Work",
            "N": "43",
            "Inputs": "Process+Comp",
            "Targets": "4 Props",
            "Validation": val_type,
            "Metric_R2": metric_val,
            "Uncertainty": "Conformal",
            "LeakageCheck": "Yes"
        }

        rows.insert(0, this_work)

        df = pd.DataFrame(rows)
        tex = df.to_latex(index=False)
        if self.validation_mode == 'STANDARD':
            tex = "% DEBUG/DIAGNOSTIC (STANDARD MODE)\n" + tex
        self._write_text(lit_name, tex)

    # =========================================================================
    # TASK 3: Missing Layer Plot (Regex)
    # =========================================================================
    def plot_missing_layer_encoding(self, df):
        """Task 3: Visualizing Missing Layer Encoding"""
        self.log("  Plotting Missing Layer Encoding (Figure S3)...")
        
        # Select columns via Regex
        pat = re.compile(r'^Layer[1-7]_B4C_Rate$')
        layer_cols = sorted([c for c in df.columns if pat.match(c)])
        
        if not layer_cols: return
        
        # Prepare heatmap data
        # Rows: Samples, Cols: Layers
        # We need to visualize "Missing" vs "Value"
        
        # Subset
        dat = df[layer_cols].copy()
        dat['Layer_Count'] = df['Layer_Count_from_code']
        
        # Create matrix: 0=Missing, 1=Present (or actual value)
        # We explicitly set missing layers to NaN to distinguish them from 0.0 value
        for idx, row in dat.iterrows():
            lc = row['Layer_Count']
            for c in layer_cols:
                # Extract layer number "LayerX..."
                try:
                    l_num = int(re.match(r'Layer(\d+)', c).group(1))
                    if l_num > lc:
                        dat.at[idx, c] = np.nan
                except: pass

        plt.figure(figsize=(10, 12))
        
        # Sort by Layer Count then Code
        dat = dat.sort_values('Layer_Count')
        
        # Heatmap of values (NaNs will be white/gray depending on style)
        sns.heatmap(dat[layer_cols], cmap='viridis', cbar_kws={'label': 'B4C wt%'}, 
                    mask=dat[layer_cols].isna()) # Mask ensures they are not colored
        
        plt.title("Layer Encodings (White = Missing Layer)")
        # plt.tight_layout() # Conflict with seaborn colorbar layout engine
        
        self._save_plot(self._with_suffix("missing_layer_encoding.pdf"))

    def plot_residual_diagnostics(self, models):
        """Generates Figure S2: Residual Diagnostics"""
        self.log("  Plotting Residual Diagnostics (Figure S2)...")
        
        for t in TARGETS:
            if t not in models:
                continue
            info = models[t]
            oof = info.get('oof')
            if not oof:
                continue

            y_true = np.array(oof.get('y_true', []))
            y_pred = np.array(oof.get('y_pred', []))
            if len(y_true) == 0 or len(y_true) != len(y_pred):
                continue
                
            residuals = y_true - y_pred
            
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            
            # 1. Residuals vs Fitted
            axes[0].scatter(y_pred, residuals, alpha=0.6, edgecolor='k')
            axes[0].axhline(0, color='r', linestyle='--')
            axes[0].set_xlabel("Fitted Values")
            axes[0].set_ylabel("Residuals")
            axes[0].set_title(f"{TARGET_SHORT_NAMES.get(t,t)}: Residuals vs Fitted")
            
            # 2. Q-Q Plot
            import scipy.stats as stats
            stats.probplot(residuals, dist="norm", plot=axes[1])
            axes[1].set_title(f"{TARGET_SHORT_NAMES.get(t,t)}: Q-Q Plot")
            
            plt.tight_layout()
            self._save_plot(self._with_suffix(f"residual_diagnostics_{TARGET_SHORT_NAMES.get(t,t).lower()}.pdf"))

    def generate_table_s2_params(self, best_params_, run_config):
        """Generates Table S2: Hyperparameter Search Space & Best Params"""
        self.log("  Generating Table S2 (Hyperparameters)...")
        if not best_params_: return

        trials = "-"
        if run_config is not None:
            trials = run_config.get('n_trials', "-")
        
        rows = []
        for t, res in best_params_.items():
            short = TARGET_SHORT_NAMES.get(t, t)
            algo = res.get('algorithm', '-')
            params = res.get('params', {})
            # Format params as string
            p_str = ', '.join([f"{k}={v}" for k,v in params.items()])
            
            rows.append({
                'Target': short,
                'Algorithm': algo,
                'Best Parameters': p_str,
                'Trials': trials
            })
            
        df = pd.DataFrame(rows)
        tex = df.to_latex(index=False, column_format="lp{2cm}p{10cm}c")
        if self.validation_mode == 'STANDARD':
            tex = "% DEBUG/DIAGNOSTIC (STANDARD MODE)\n" + tex
        self._write_text(self._with_suffix('table_s2_params.tex'), tex)

    def generate_all_manuscript_artifacts(self, final_df, models, feature_sets, df_qc, stats_results, best_params_=None, studies=None, run_config=None):
        """High-level call to generate all required missing items"""
        self._set_run_config(run_config)

        try:
            # Core tables and figures
            self.generate_table_1_stats(df_qc)
            self.generate_table_2_features()
            self.generate_table_s1_folds(final_df, models)
            self.plot_permutation_hist(models)
            self.plot_feature_importance(models, feature_sets, final_df)

            # Supplementary
            if best_params_:
                self.generate_table_s2_params(best_params_, run_config)
            if studies:
                self.plot_optuna_history(studies)

            self.plot_residual_diagnostics(models)
            self.plot_outlier_index(models, feature_sets, final_df)
            if 'layer_vector_extended' in feature_sets:
                self.plot_missing_layer_encoding(feature_sets['layer_vector_extended'])

            enable_samplewise = True
            if run_config is not None:
                enable_samplewise = bool(run_config.get("enable_samplewise_table", True))
            if enable_samplewise:
                self.export_samplewise_prediction_table(
                    models=models, 
                    final_df=final_df, 
                    top_n=run_config.get('samplewise_table_top_n', 10) if run_config else 10,
                    validation_mode=self.validation_mode
                )

            self.generate_table_lit_comparison(models, run_config)
            self.generate_fig1_schematic_script()
        except Exception:
            self._write_manifest()
            raise

    def run_ablation_study(self, models, feature_sets, final_df, evaluate_cv_func, run_config):
        """
        Generates Table 4: Feature Set Ablation Study.
        Compares: Core vs Core+Ext vs Scenario B (Layer Vector)
        
        :param evaluate_cv_func: Callback to phase1._evaluate_cv (or wrapper) to ensure consistent validation
        """
        self._set_run_config(run_config)
        self.log(f"Running Ablation Study ({self.validation_mode})...")
        is_rigorous = (self.validation_mode == 'RIGOROUS')
        targets_ordered = list(TARGETS)
        
        sets_map = {
            'Core': 'core',
            'Core+Ext': 'core_extended',
            'Scenario B': 'layer_vector_extended'
        }
        
        final_table_rows = []
        
        for t in targets_ordered:
            if t not in models: continue
            info = models[t]
            model_tmpl = info['best_model']
            
            row = {'Property': TARGET_SHORT_NAMES[t]}

            base_key = None
            for k in sets_map.values():
                if k in feature_sets:
                    base_key = k
                    break

            splits = None
            if base_key is not None:
                mask_base = final_df[t].notna()
                X_base = feature_sets[base_key].loc[mask_base]
                y_base = final_df.loc[mask_base, t]
                n_outer = run_config.get('n_outer', 5) if run_config else 5
                seed = run_config.get('seed', 42) if run_config else 42
                splits = list(KFold(n_splits=n_outer, shuffle=True, random_state=seed).split(X_base, y_base))
            
            for table_col, feat_key in sets_map.items():
                if feat_key not in feature_sets:
                    row[table_col] = "-"
                    continue
                    
                X_all = feature_sets[feat_key]
                mask = final_df[t].notna()
                X = X_all.loc[mask]
                y = final_df.loc[mask, t]
                w = final_df.loc[mask, 'sample_weight']
                
                if is_rigorous:
                     # Using the passed callback function for CV evaluation
                     res = evaluate_cv_func(X, y, model_tmpl, weights=w, conformal=False, splits=splits)
                     val = res['r2']
                else:
                    m = clone(model_tmpl)
                    m.fit(X, y)
                    val = r2_score(y, m.predict(X))
                
                row[table_col] = f"{val:.3f}"
            
            final_table_rows.append(row)
            
        # Generate LaTeX
        df_abl = pd.DataFrame(final_table_rows)
        cols = ['Property', 'Core', 'Core+Ext', 'Scenario B']
        df_abl = df_abl[cols]
        
        tex = df_abl.to_latex(index=False, float_format="%.3f")
        if self.validation_mode == 'STANDARD':
            tex = "% DEBUG/DIAGNOSTIC (STANDARD MODE)\n" + tex
        self._write_text(self._with_suffix('ablation_table.tex'), tex)

    def run_calibration_analysis(self, models, feature_sets, final_df, run_config):
        """Generates Calibration Plots (Real vs Predicted)"""
        self._set_run_config(run_config)
        self.log("  Running Calibration Analysis...")
        
        for t in TARGETS:
            if t not in models:
                continue
            info = models[t]
            oof = info.get('oof')
            if not oof:
                continue

            y_true = np.array(oof.get('y_true', []))
            y_pred = np.array(oof.get('y_pred', []))
            if len(y_true) == 0 or len(y_true) != len(y_pred):
                self.log(f"    Skipping calibration for {t} due to length mismatch ({len(y_true)} vs {len(y_pred)})")
                continue
            
            plt.figure(figsize=(6, 6))
            plt.scatter(y_pred, y_true, alpha=0.5, edgecolor='k')
            dmin = min(y_true.min(), y_pred.min())
            dmax = max(y_true.max(), y_pred.max())
            plt.plot([dmin, dmax], [dmin, dmax], 'r--')
            
            plt.xlabel("Predicted")
            plt.ylabel("Observed")
            plt.title(f"Calibration: {TARGET_SHORT_NAMES.get(t, t)}")
            self._save_plot(self._with_suffix(f"calibration_{TARGET_SHORT_NAMES.get(t, t).lower()}.pdf"))

    def generate_final_artifacts(self, final_df, models, feature_sets, run_optuna, run_config):
        self._set_run_config(run_config)
        self.log("Generating Final Tables & Variables (via ArtifactGenerator)...")

        # 1. Combined Parity
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        axes = axes.flatten()

        for i, t in enumerate(TARGETS):
            if t not in models:
                continue
            info = models[t]
            oof = info.get('oof', {})
            y_true = np.array(oof.get('y_true', []))
            y_pred = np.array(oof.get('y_pred', []))
            if len(y_true) == 0 or len(y_true) != len(y_pred):
                continue

            unit = TARGET_UNITS.get(TARGET_SHORT_NAMES.get(t, 'Val'), '')
            width = info.get('pi', {}).get('width', 0)
            if width and width > 0:
                q = width / 2.0
                err_low = np.full_like(y_pred, q)
                err_high = np.full_like(y_pred, q)
                axes[i].errorbar(y_true, y_pred, yerr=[err_low, err_high],
                                 fmt='none', ecolor='gray', alpha=0.3, zorder=1)

            sns.scatterplot(x=y_true, y=y_pred, ax=axes[i], alpha=0.7, edgecolor='k', zorder=2)
            dmin = min(y_true.min(), y_pred.min())
            dmax = max(y_true.max(), y_pred.max())
            axes[i].plot([dmin, dmax], [dmin, dmax], 'r--', lw=1.5, zorder=3)

            r2_val = oof.get('r2', 0)
            rmse_val = oof.get('rmse', 0)
            algo_name = info.get('algo', 'Unknown')
            mode_lbl = "OOF (Honest)" if self.validation_mode == 'RIGOROUS' else "Train (Fit)"

            text_str = f"$R^2 = {r2_val:.3f}$\nRMSE = {rmse_val:.3f}\nAlgo: {algo_name}\nMode: {mode_lbl}"
            axes[i].text(0.05, 0.95, text_str, transform=axes[i].transAxes,
                         fontsize=11, verticalalignment='top',
                         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            axes[i].set_title(f"{TARGET_SHORT_NAMES[t]} Prediction", fontsize=14, fontweight='bold')
            axes[i].set_xlabel(f"Experimental ({unit})", fontsize=12)
            axes[i].set_ylabel(f"Predicted ({unit})", fontsize=12)
            axes[i].grid(True, linestyle=':', alpha=0.6)

        plt.tight_layout()
        self._save_plot(self._with_suffix("parity_plots.pdf"))

        # 2. latex_variables.tex
        content = "% Generated Variables\n"
        if self.validation_mode == 'STANDARD':
            content = "% DEBUG/DIAGNOSTIC (STANDARD MODE)\n" + content

        for t in TARGETS:
            short = TARGET_SHORT_NAMES[t].capitalize()
            if t == 'Max_Compressive_Strength_MPa': short = 'Strength'
            if t == 'Hardness_HB': short = 'Hardness'
            if t == 'Sintered_Density_g_cm3': short = 'Density'
            if t == 'Toughness_MJ_m3': short = 'Toughness'

            info = models.get(t, {})
            oof = info.get('oof', {})
            pi = info.get('pi', {})
            p_val = info.get('perm_test', {}).get('p_value', 1.0)
            p_str = "< 0.001" if p_val < 0.001 else f"{p_val:.3f}"

            content += f"\\newcommand{{\\{short}OOFR}}{{{oof.get('r2', 0):.2f}}}\n"
            content += f"\\newcommand{{\\{short}Coverage}}{{{pi.get('coverage', 0):.1%}}}\n"
            content += f"\\newcommand{{\\{short}CIWidth}}{{{pi.get('width', 0):.2f}}}\n"
            content += f"\\newcommand{{\\{short}CohensD}}{{{info.get('cohens_d', 0):.2f}}}\n"
            content += f"\\newcommand{{\\{short}SDDiff}}{{{info.get('sd_diff', 0):.2f}}}\n"
            content += f"\\newcommand{{\\{short}RMSE}}{{{oof.get('rmse', 0):.2f}}}\n"
            content += f"\\newcommand{{\\{short}RTwo}}{{{oof.get('r2', 0):.2f}}}\n"
            content += f"\\newcommand{{\\{short}PVal}}{{{p_str}}}\n"

        self._write_text(self._with_suffix("latex_variables.tex"), content)

        # 3. rigorous_results_table
        tex = r"\begin{tabular}{lccccccc}" + "\n\\toprule\nTarget & Algo & RMSE & $R^2$ & PI Cov. & PI Width & $d$ & p-value \\\\\n\\midrule\n"
        notes = []
        for t in TARGETS:
            info = models.get(t, {})
            name = TARGET_SHORT_NAMES.get(t, t)
            oof = info.get('oof', {})
            pi = info.get('pi', {})
            p_val = info.get('perm_test', {}).get('p_value', 1.0)
            p_str = "< 0.001" if p_val < 0.001 else f"{p_val:.3f}"
            tex += f"{name} & {info.get('algo','-')} & {oof.get('rmse',0):.2f} & {oof.get('r2',0):.2f} & {pi.get('coverage',0):.2f} & {pi.get('width',0):.2f} & {info.get('cohens_d',0):.2f} & {p_str} \\\\\n"
            if info.get('flag_not_significant'):
                notes.append(f"{name}: This target does not outperform the permutation null at $\\alpha=0.05$.")
        tex += "\\bottomrule\n\\end{tabular}\n"
        if notes:
            tex += "\\newline\\textit{Note:} " + " ".join(notes) + "\n"
        if self.validation_mode == 'STANDARD':
            tex = "% DEBUG/DIAGNOSTIC (STANDARD MODE)\n" + tex
        self._write_text(self._with_suffix("rigorous_results_table.tex"), tex)

        # Finalize manifest
        missing = self._write_manifest()
        if missing:
            raise RuntimeError(f"Missing artifacts: {missing}")
