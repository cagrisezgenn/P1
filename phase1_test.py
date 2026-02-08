#!/usr/bin/env python3
"""
phase1_test.py - Test harness for measuring R² impact of STEP_1..STEP_6

This script runs controlled experiments with different combinations of R2_IMPROVEMENTS
toggles to measure their impact on model performance.

Usage:
    python phase1_test.py --data Mekanik_Test_Cleaned.csv --outdir P1_TestSuite --targets Hardness_HB --suite minimal
    python phase1_test.py --data Mekanik_Test_Cleaned.csv --outdir P1_TestSuite --targets Hardness_HB --suite full
"""

import os
import sys
import argparse
import json
import csv
import re
import traceback
from pathlib import Path
from datetime import datetime
import importlib
import pandas as pd
import numpy as np

# Import phase1 module
import phase1


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Phase 1 Test Suite - Measure R² impact of STEP_1..STEP_6')
    
    # Determine default paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_data = os.path.join(script_dir, 'Mekanik_Test_Cleaned.csv')
    
    parser.add_argument('--data', default=default_data, help='Input CSV path')
    parser.add_argument('--outdir', default='P1_TestSuite', help='Root output directory for test suite')
    parser.add_argument('--targets', default='Hardness_HB', help='Target variables (comma-separated)')
    parser.add_argument('--suite', choices=['minimal', 'full'], default='minimal',
                       help='Test suite: minimal (~11 runs) or full (~23 runs)')
    parser.add_argument('--mode', choices=['rigorous', 'standard'], default='rigorous',
                       help='Validation mode')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--step7-mode', type=int, default=3, choices=[1, 2, 3],
                       help='STEP7_DATA_SLICE_MODE (1=pretest, 2=taguchi, 3=all)')
    
    return parser.parse_args()


def get_base_config(data_path, outdir, targets_list, mode, seed, step7_mode):
    """
    Get base configuration for fast, comparable test runs.
    
    Returns a dict of CONFIG overrides.
    """
    return {
        'VALIDATION_MODE': mode.upper(),
        'RUN_DUAL_MODE': False,
        'RUN_OPTUNA': False,
        'RUN_ENSEMBLES': False,
        'RUN_STATS': False,
        'PERMUTATION_COUNT': 0,
        'CV_SPLITS': 5,
        'CV_REPEATS': 1,
        'RIGOROUS_FINAL_MODEL_POLICY': 'TREE',
        'RUN_TARGETS': targets_list,
        'SEED': seed,
        'STEP7_DATA_SLICE_MODE': step7_mode,
        # Disable plotting/reports for speed (fixed config key)
        'RUN_VECTOR_PLOTS': False,
        'RUN_PDP': False,
        'RUN_SHAP': False,
        'RUN_ABLATION': False,  # Fixed: was ENABLE_ABLATION
        'ENABLE_LIT_TABLE': False,
        'REQUIRE_PDFLATEX_FOR_FIG1': False,
        'SHAP_STRICT': False,
        # Keep polynomial features as default
        'RUN_POLYNOMIAL_FEATURES': True,
    }


def get_experiment_cases(suite_type):
    """
    Define experiment cases (combinations of STEP toggles).
    
    Returns: list of dicts with keys: 'name', 'steps' (dict of STEP_X: bool)
    """
    if suite_type == 'minimal':
        cases = [
            # Baseline: all off
            {
                'name': 'baseline_all_off',
                'steps': {
                    'STEP_1_ENABLE_FEATURES': False,
                    'STEP_2_OPTIMUM_DEVIATION': False,
                    'STEP_3_MXENE_REGIME': False,
                    'STEP_4_QC_INCLUSIVE_THRESHOLD': False,
                    'STEP_5_DENSITY_HAT': False,
                    'STEP_6_OUTLIER_DOWNWEIGHT': False,
                }
            },
            # Individual steps only
            {
                'name': 'step1_only',
                'steps': {
                    'STEP_1_ENABLE_FEATURES': True,
                    'STEP_2_OPTIMUM_DEVIATION': False,
                    'STEP_3_MXENE_REGIME': False,
                    'STEP_4_QC_INCLUSIVE_THRESHOLD': False,
                    'STEP_5_DENSITY_HAT': False,
                    'STEP_6_OUTLIER_DOWNWEIGHT': False,
                }
            },
            {
                'name': 'step2_only',
                'steps': {
                    'STEP_1_ENABLE_FEATURES': False,
                    'STEP_2_OPTIMUM_DEVIATION': True,
                    'STEP_3_MXENE_REGIME': False,
                    'STEP_4_QC_INCLUSIVE_THRESHOLD': False,
                    'STEP_5_DENSITY_HAT': False,
                    'STEP_6_OUTLIER_DOWNWEIGHT': False,
                }
            },
            {
                'name': 'step3_only',
                'steps': {
                    'STEP_1_ENABLE_FEATURES': False,
                    'STEP_2_OPTIMUM_DEVIATION': False,
                    'STEP_3_MXENE_REGIME': True,
                    'STEP_4_QC_INCLUSIVE_THRESHOLD': False,
                    'STEP_5_DENSITY_HAT': False,
                    'STEP_6_OUTLIER_DOWNWEIGHT': False,
                }
            },
            {
                'name': 'step4_only',
                'steps': {
                    'STEP_1_ENABLE_FEATURES': False,
                    'STEP_2_OPTIMUM_DEVIATION': False,
                    'STEP_3_MXENE_REGIME': False,
                    'STEP_4_QC_INCLUSIVE_THRESHOLD': True,
                    'STEP_5_DENSITY_HAT': False,
                    'STEP_6_OUTLIER_DOWNWEIGHT': False,
                }
            },
            {
                'name': 'step5_only',
                'steps': {
                    'STEP_1_ENABLE_FEATURES': False,
                    'STEP_2_OPTIMUM_DEVIATION': False,
                    'STEP_3_MXENE_REGIME': False,
                    'STEP_4_QC_INCLUSIVE_THRESHOLD': False,
                    'STEP_5_DENSITY_HAT': True,
                    'STEP_6_OUTLIER_DOWNWEIGHT': False,
                }
            },
            {
                'name': 'step6_only',
                'steps': {
                    'STEP_1_ENABLE_FEATURES': False,
                    'STEP_2_OPTIMUM_DEVIATION': False,
                    'STEP_3_MXENE_REGIME': False,
                    'STEP_4_QC_INCLUSIVE_THRESHOLD': False,
                    'STEP_5_DENSITY_HAT': False,
                    'STEP_6_OUTLIER_DOWNWEIGHT': True,
                }
            },
            # All on
            {
                'name': 'all_on',
                'steps': {
                    'STEP_1_ENABLE_FEATURES': True,
                    'STEP_2_OPTIMUM_DEVIATION': True,
                    'STEP_3_MXENE_REGIME': True,
                    'STEP_4_QC_INCLUSIVE_THRESHOLD': True,
                    'STEP_5_DENSITY_HAT': True,
                    'STEP_6_OUTLIER_DOWNWEIGHT': True,
                }
            },
            # Leave-one-out for likely impactful steps (4, 5, 6)
            {
                'name': 'all_except_step4',
                'steps': {
                    'STEP_1_ENABLE_FEATURES': True,
                    'STEP_2_OPTIMUM_DEVIATION': True,
                    'STEP_3_MXENE_REGIME': True,
                    'STEP_4_QC_INCLUSIVE_THRESHOLD': False,
                    'STEP_5_DENSITY_HAT': True,
                    'STEP_6_OUTLIER_DOWNWEIGHT': True,
                }
            },
            {
                'name': 'all_except_step5',
                'steps': {
                    'STEP_1_ENABLE_FEATURES': True,
                    'STEP_2_OPTIMUM_DEVIATION': True,
                    'STEP_3_MXENE_REGIME': True,
                    'STEP_4_QC_INCLUSIVE_THRESHOLD': True,
                    'STEP_5_DENSITY_HAT': False,
                    'STEP_6_OUTLIER_DOWNWEIGHT': True,
                }
            },
            {
                'name': 'all_except_step6',
                'steps': {
                    'STEP_1_ENABLE_FEATURES': True,
                    'STEP_2_OPTIMUM_DEVIATION': True,
                    'STEP_3_MXENE_REGIME': True,
                    'STEP_4_QC_INCLUSIVE_THRESHOLD': True,
                    'STEP_5_DENSITY_HAT': True,
                    'STEP_6_OUTLIER_DOWNWEIGHT': False,
                }
            },
        ]
    
    elif suite_type == 'full':
        # Start with minimal cases
        cases = get_experiment_cases('minimal')
        
        # Add cumulative: step1, step1+2, step1+2+3, ..., step1+...+6
        cumulative = [
            {
                'name': 'cumulative_s1',
                'steps': {
                    'STEP_1_ENABLE_FEATURES': True,
                    'STEP_2_OPTIMUM_DEVIATION': False,
                    'STEP_3_MXENE_REGIME': False,
                    'STEP_4_QC_INCLUSIVE_THRESHOLD': False,
                    'STEP_5_DENSITY_HAT': False,
                    'STEP_6_OUTLIER_DOWNWEIGHT': False,
                }
            },
            {
                'name': 'cumulative_s1_s2',
                'steps': {
                    'STEP_1_ENABLE_FEATURES': True,
                    'STEP_2_OPTIMUM_DEVIATION': True,
                    'STEP_3_MXENE_REGIME': False,
                    'STEP_4_QC_INCLUSIVE_THRESHOLD': False,
                    'STEP_5_DENSITY_HAT': False,
                    'STEP_6_OUTLIER_DOWNWEIGHT': False,
                }
            },
            {
                'name': 'cumulative_s1_s2_s3',
                'steps': {
                    'STEP_1_ENABLE_FEATURES': True,
                    'STEP_2_OPTIMUM_DEVIATION': True,
                    'STEP_3_MXENE_REGIME': True,
                    'STEP_4_QC_INCLUSIVE_THRESHOLD': False,
                    'STEP_5_DENSITY_HAT': False,
                    'STEP_6_OUTLIER_DOWNWEIGHT': False,
                }
            },
            {
                'name': 'cumulative_s1_s2_s3_s4',
                'steps': {
                    'STEP_1_ENABLE_FEATURES': True,
                    'STEP_2_OPTIMUM_DEVIATION': True,
                    'STEP_3_MXENE_REGIME': True,
                    'STEP_4_QC_INCLUSIVE_THRESHOLD': True,
                    'STEP_5_DENSITY_HAT': False,
                    'STEP_6_OUTLIER_DOWNWEIGHT': False,
                }
            },
            {
                'name': 'cumulative_s1_s2_s3_s4_s5',
                'steps': {
                    'STEP_1_ENABLE_FEATURES': True,
                    'STEP_2_OPTIMUM_DEVIATION': True,
                    'STEP_3_MXENE_REGIME': True,
                    'STEP_4_QC_INCLUSIVE_THRESHOLD': True,
                    'STEP_5_DENSITY_HAT': True,
                    'STEP_6_OUTLIER_DOWNWEIGHT': False,
                }
            },
            {
                'name': 'cumulative_s1_s2_s3_s4_s5_s6',
                'steps': {
                    'STEP_1_ENABLE_FEATURES': True,
                    'STEP_2_OPTIMUM_DEVIATION': True,
                    'STEP_3_MXENE_REGIME': True,
                    'STEP_4_QC_INCLUSIVE_THRESHOLD': True,
                    'STEP_5_DENSITY_HAT': True,
                    'STEP_6_OUTLIER_DOWNWEIGHT': True,
                }
            },
        ]
        
        # Add leave-one-out for ALL steps
        leave_one_out = [
            {
                'name': 'all_except_step1',
                'steps': {
                    'STEP_1_ENABLE_FEATURES': False,
                    'STEP_2_OPTIMUM_DEVIATION': True,
                    'STEP_3_MXENE_REGIME': True,
                    'STEP_4_QC_INCLUSIVE_THRESHOLD': True,
                    'STEP_5_DENSITY_HAT': True,
                    'STEP_6_OUTLIER_DOWNWEIGHT': True,
                }
            },
            {
                'name': 'all_except_step2',
                'steps': {
                    'STEP_1_ENABLE_FEATURES': True,
                    'STEP_2_OPTIMUM_DEVIATION': False,
                    'STEP_3_MXENE_REGIME': True,
                    'STEP_4_QC_INCLUSIVE_THRESHOLD': True,
                    'STEP_5_DENSITY_HAT': True,
                    'STEP_6_OUTLIER_DOWNWEIGHT': True,
                }
            },
            {
                'name': 'all_except_step3',
                'steps': {
                    'STEP_1_ENABLE_FEATURES': True,
                    'STEP_2_OPTIMUM_DEVIATION': True,
                    'STEP_3_MXENE_REGIME': False,
                    'STEP_4_QC_INCLUSIVE_THRESHOLD': True,
                    'STEP_5_DENSITY_HAT': True,
                    'STEP_6_OUTLIER_DOWNWEIGHT': True,
                }
            },
        ]
        
        cases.extend(cumulative)
        cases.extend(leave_one_out)
    
    else:
        raise ValueError(f"Unknown suite type: {suite_type}")
    
    return cases


def extract_metrics_from_pipeline(pipeline, targets_list):
    """
    PRIMARY: Extract metrics from pipeline.models (authoritative source).
    
    Returns: dict of {target: {metric_name: value}}
    """
    metrics_by_target = {}
    
    for target in targets_list:
        metrics = {
            'algo': None,
            'oof_r2': None,
            'oof_rmse': None,
            'oof_mae': None,
            'pi_coverage': None,
            'pi_width': None,
            'p_value': None,
            'mode3_oof_r2': None,
            'mode3_oof_rmse': None,
        }
        
        if hasattr(pipeline, 'models') and target in pipeline.models:
            model_info = pipeline.models[target]
            
            # Extract algorithm
            metrics['algo'] = model_info.get('algo')
            
            # Extract OOF metrics
            oof = model_info.get('oof', {})
            metrics['oof_r2'] = oof.get('r2')
            metrics['oof_rmse'] = oof.get('rmse')
            metrics['oof_mae'] = oof.get('mae')
            
            # Extract PI metrics
            pi = model_info.get('pi', {})
            metrics['pi_coverage'] = pi.get('coverage')
            metrics['pi_width'] = pi.get('width')
            
            # Extract permutation test p-value
            perm = model_info.get('perm_test', {})
            metrics['p_value'] = perm.get('p_value')
            
            # Extract mode3 OOF metrics (if present)
            metrics['mode3_oof_r2'] = model_info.get('mode3_oof_r2')
            metrics['mode3_oof_rmse'] = model_info.get('mode3_oof_rmse')
        
        metrics_by_target[target] = metrics
    
    return metrics_by_target


def parse_latex_table_fallback(tex_path, target, target_short):
    """
    FALLBACK: Parse rigorous_results_table_*.tex to extract metrics.
    
    Column order: Target, Algo, RMSE, R^2, PI Cov, PI Width, d, p-value
    
    Returns dict with keys: oof_r2, oof_rmse, pi_coverage, pi_width, p_value
    """
    metrics = {
        'oof_r2': None,
        'oof_rmse': None,
        'pi_coverage': None,
        'pi_width': None,
        'p_value': None,
    }
    
    if not os.path.exists(tex_path):
        return metrics
    
    try:
        with open(tex_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Look for line containing target name (full or short)
        target_escaped = target.replace('_', '\\_')
        target_short_escaped = target_short.replace('_', '\\_') if target_short else None
        
        for line in content.split('\n'):
            if target_escaped in line or (target_short_escaped and target_short_escaped in line):
                # Split by '&' and extract columns
                # Format: Target & Algo & RMSE & R^2 & PI_Cov & PI_Width & d & p-value \\
                parts = [p.strip() for p in line.split('&')]
                
                if len(parts) >= 4:
                    # Index 2: RMSE, Index 3: R^2
                    try:
                        rmse_str = parts[2].strip()
                        r2_str = parts[3].strip()
                        
                        # Extract numeric values (handle LaTeX formatting)
                        rmse_match = re.search(r'[-+]?\d*\.\d+|\d+', rmse_str)
                        r2_match = re.search(r'[-+]?\d*\.\d+|\d+', r2_str)
                        
                        if rmse_match:
                            metrics['oof_rmse'] = float(rmse_match.group())
                        if r2_match:
                            metrics['oof_r2'] = float(r2_match.group())
                        
                        # Index 4: PI_Cov, Index 5: PI_Width, Index 7: p-value
                        if len(parts) > 4:
                            pi_cov_match = re.search(r'[-+]?\d*\.\d+|\d+', parts[4].strip())
                            if pi_cov_match:
                                metrics['pi_coverage'] = float(pi_cov_match.group())
                        
                        if len(parts) > 5:
                            pi_width_match = re.search(r'[-+]?\d*\.\d+|\d+', parts[5].strip())
                            if pi_width_match:
                                metrics['pi_width'] = float(pi_width_match.group())
                        
                        if len(parts) > 7:
                            p_val_match = re.search(r'[-+]?\d*\.\d+|\d+', parts[7].strip())
                            if p_val_match:
                                metrics['p_value'] = float(p_val_match.group())
                    except (ValueError, IndexError) as e:
                        pass  # Skip malformed lines
                
                break  # Found target row
    except Exception as e:
        print(f"  Warning: Could not parse {tex_path}: {e}")
    
    return metrics


def find_results_table(run_dir):
    """Find rigorous_results_table_*.tex in run directory."""
    # Prefer: <run_dir>/latex/rigorous_results_table_*.tex
    latex_dir = os.path.join(run_dir, 'latex')
    if os.path.exists(latex_dir):
        for fname in os.listdir(latex_dir):
            if fname.startswith('rigorous_results_table') and fname.endswith('.tex'):
                return os.path.join(latex_dir, fname)
    
    # Fallback: search entire run_dir
    for root, dirs, files in os.walk(run_dir):
        for fname in files:
            if fname.startswith('rigorous_results_table') and fname.endswith('.tex'):
                return os.path.join(root, fname)
    
    return None


def run_single_case(case_name, steps_dict, base_config, data_path, root_outdir, targets_list, step7_mode):
    """
    Run a single experiment case.
    
    Returns: list of result dicts (one per target)
    """
    print(f"\n{'='*80}")
    print(f"SUITE_RUN start case={case_name}")
    active_steps = [k for k, v in steps_dict.items() if v] + ['STEP_7_DATA_SLICE']
    print(f"  active_steps: {', '.join(active_steps)}")
    
    # Create run subdirectory
    run_dir = os.path.join(root_outdir, case_name)
    os.makedirs(run_dir, exist_ok=True)
    
    # Initialize results (one per target)
    results = []
    
    try:
        # Set R2_IMPROVEMENTS for this case
        r2_improvements = steps_dict.copy()
        r2_improvements['STEP_7_DATA_SLICE'] = True
        
        phase1.R2_IMPROVEMENTS = r2_improvements
        
        # Build config for this run
        config_override = base_config.copy()
        config_override['ARTIFACT_SUFFIX'] = case_name
        
        # Instantiate and run pipeline
        pipeline = phase1.Phase1Pipeline(
            input_file=data_path,
            outdir=run_dir,
            override_config=config_override
        )
        
        pipeline.run()
        
        # Extract metrics from pipeline (PRIMARY source)
        metrics_by_target = extract_metrics_from_pipeline(pipeline, targets_list)
        
        # Find common paths
        log_path = pipeline.log_path if hasattr(pipeline, 'log_path') else None
        results_table_path = find_results_table(run_dir)
        run_config_path = None
        manifest_path = None
        
        for fname in os.listdir(run_dir):
            if fname.startswith('run_config_') and fname.endswith('.json'):
                run_config_path = os.path.join(run_dir, fname)
            if fname.startswith('artifact_manifest_') and fname.endswith('.json'):
                manifest_path = os.path.join(run_dir, fname)
        
        # Create result for each target
        for target in targets_list:
            target_short = phase1.TARGET_SHORT_NAMES.get(target, target)
            metrics = metrics_by_target.get(target, {})
            
            # FALLBACK: If oof_r2 is None, try parsing LaTeX table
            if metrics.get('oof_r2') is None and results_table_path:
                fallback_metrics = parse_latex_table_fallback(results_table_path, target, target_short)
                # Merge fallback metrics (only if primary is None)
                for key, val in fallback_metrics.items():
                    if metrics.get(key) is None:
                        metrics[key] = val
            
            result = {
                'case_name': case_name,
                'target': target,
                'target_short': target_short,
                'step7_mode': step7_mode,
                'mode': base_config['VALIDATION_MODE'],
                'active_steps': ','.join(active_steps),
                'algo': metrics.get('algo'),
                'oof_r2': metrics.get('oof_r2'),
                'oof_rmse': metrics.get('oof_rmse'),
                'oof_mae': metrics.get('oof_mae'),
                'pi_coverage': metrics.get('pi_coverage'),
                'pi_width': metrics.get('pi_width'),
                'p_value': metrics.get('p_value'),
                'mode3_oof_r2': metrics.get('mode3_oof_r2'),
                'mode3_oof_rmse': metrics.get('mode3_oof_rmse'),
                'run_dir': run_dir,
                'log_path': log_path,
                'results_table_path': results_table_path,
                'run_config_path': run_config_path,
                'manifest_path': manifest_path,
                'status': 'OK',
                'error_message': '',
            }
            
            results.append(result)
            
            print(f"SUITE_RUN done case={case_name} target={target} " +
                  f"r2={metrics.get('oof_r2')} rmse={metrics.get('oof_rmse')}")
    
    except Exception as e:
        # Create FAIL result for all targets
        for target in targets_list:
            target_short = phase1.TARGET_SHORT_NAMES.get(target, target)
            
            result = {
                'case_name': case_name,
                'target': target,
                'target_short': target_short,
                'step7_mode': step7_mode,
                'mode': base_config['VALIDATION_MODE'],
                'active_steps': ','.join(active_steps),
                'algo': None,
                'oof_r2': None,
                'oof_rmse': None,
                'oof_mae': None,
                'pi_coverage': None,
                'pi_width': None,
                'p_value': None,
                'mode3_oof_r2': None,
                'mode3_oof_rmse': None,
                'run_dir': run_dir,
                'log_path': None,
                'results_table_path': None,
                'run_config_path': None,
                'manifest_path': None,
                'status': 'FAIL',
                'error_message': str(e)[:200],
            }
            results.append(result)
        
        print(f"SUITE_RUN FAILED case={case_name} error={str(e)}")
        traceback.print_exc()
    
    return results


def add_baseline_deltas(results_df):
    """
    Add delta_r2_vs_baseline column comparing to baseline_all_off.
    
    Computes delta per (target, step7_mode) group.
    """
    results_df['delta_r2_vs_baseline'] = np.nan
    
    # Group by target and step7_mode
    for (target, step7_mode), group in results_df.groupby(['target', 'step7_mode']):
        baseline_rows = group[group['case_name'].str.contains('baseline_all_off', na=False)]
        
        if len(baseline_rows) > 0:
            baseline_r2 = baseline_rows['oof_r2'].iloc[0]
            
            if pd.notna(baseline_r2):
                # Compute delta for all rows in this group
                group_indices = group.index
                results_df.loc[group_indices, 'delta_r2_vs_baseline'] = \
                    results_df.loc[group_indices, 'oof_r2'] - baseline_r2
    
    return results_df


def write_summary(results, output_dir):
    """Write summary CSV and JSON (LONG format + WIDE format)."""
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Add baseline deltas
    df = add_baseline_deltas(df)
    
    # Write LONG format
    summary_csv_long = os.path.join(output_dir, 'summary_results.csv')
    summary_json = os.path.join(output_dir, 'summary_results.json')
    
    df.to_csv(summary_csv_long, index=False)
    df.to_json(summary_json, orient='records', indent=2)
    
    # Write WIDE format (easier for comparison)
    summary_csv_wide = os.path.join(output_dir, 'summary_results_wide.csv')
    
    # Pivot: rows=case_name, columns=target metrics
    pivot_data = []
    for case_name in df['case_name'].unique():
        case_df = df[df['case_name'] == case_name]
        row = {'case_name': case_name}
        
        # Add step7_mode and active_steps (should be same for all targets in case)
        row['step7_mode'] = case_df['step7_mode'].iloc[0] if len(case_df) > 0 else None
        row['active_steps'] = case_df['active_steps'].iloc[0] if len(case_df) > 0 else None
        row['status'] = case_df['status'].iloc[0] if len(case_df) > 0 else None
        
        # Add metrics per target
        for _, target_row in case_df.iterrows():
            target_short = target_row['target_short']
            row[f'{target_short}_oof_r2'] = target_row['oof_r2']
            row[f'{target_short}_oof_rmse'] = target_row['oof_rmse']
            row[f'{target_short}_delta_r2'] = target_row['delta_r2_vs_baseline']
            
            if pd.notna(target_row['mode3_oof_r2']):
                row[f'{target_short}_mode3_oof_r2'] = target_row['mode3_oof_r2']
        
        pivot_data.append(row)
    
    wide_df = pd.DataFrame(pivot_data)
    wide_df.to_csv(summary_csv_wide, index=False)
    
    # Optional: R² leaderboard (sorted by first target's R²)
    if len(df) > 0:
        first_target = df['target'].iloc[0]
        leaderboard = df[df['target'] == first_target][['case_name', 'oof_r2', 'delta_r2_vs_baseline']].copy()
        leaderboard = leaderboard.sort_values('oof_r2', ascending=False)
        leaderboard_csv = os.path.join(output_dir, 'r2_leaderboard.csv')
        leaderboard.to_csv(leaderboard_csv, index=False)
        
        print(f"\nSummary written to:")
        print(f"  {summary_csv_long} (LONG format)")
        print(f"  {summary_csv_wide} (WIDE format)")
        print(f"  {summary_json}")
        print(f"  {leaderboard_csv}")
    
    return summary_csv_long, summary_csv_wide, summary_json


def print_summary_table(results):
    """Print concise console summary table."""
    df = pd.DataFrame(results)
    
    # If delta_r2_vs_baseline column is missing, compute it
    if 'delta_r2_vs_baseline' not in df.columns:
        df = add_baseline_deltas(df)
    
    print(f"\n{'='*120}")
    print("SUMMARY TABLE")
    print(f"{'='*120}")
    print(f"{'Case Name':<30} {'Target':<15} {'R²':<10} {'RMSE':<10} {'Δ R² vs Base':<15} {'Status':<10}")
    print(f"{'-'*120}")
    
    for _, row in df.iterrows():
        r2_str = f"{row['oof_r2']:.4f}" if pd.notna(row['oof_r2']) else 'N/A'
        rmse_str = f"{row['oof_rmse']:.4f}" if pd.notna(row['oof_rmse']) else 'N/A'
        delta_str = f"+{row['delta_r2_vs_baseline']:.4f}" if pd.notna(row.get('delta_r2_vs_baseline')) and row['delta_r2_vs_baseline'] >= 0 else f"{row['delta_r2_vs_baseline']:.4f}" if pd.notna(row.get('delta_r2_vs_baseline')) else 'N/A'
        
        print(f"{row['case_name']:<30} {row['target_short']:<15} {r2_str:<10} {rmse_str:<10} {delta_str:<15} {row['status']:<10}")
    
    print(f"{'='*120}\n")


def main():
    """Main test suite runner."""
    args = parse_args()
    
    print("="*80)
    print("PHASE 1 TEST SUITE")
    print("="*80)
    print(f"Data:       {args.data}")
    print(f"Outdir:     {args.outdir}")
    print(f"Targets:    {args.targets}")
    print(f"Suite:      {args.suite}")
    print(f"Mode:       {args.mode}")
    print(f"STEP7 Mode: {args.step7_mode}")
    print(f"Seed:       {args.seed}")
    print("="*80)
    
    # Create output directory
    os.makedirs(args.outdir, exist_ok=True)
    
    # Parse targets
    targets_list = [t.strip() for t in args.targets.split(',')]
    
    # Get base config
    base_config = get_base_config(args.data, args.outdir, targets_list, args.mode, args.seed, args.step7_mode)
    
    # Get experiment cases
    cases = get_experiment_cases(args.suite)
    
    print(f"\nRunning {len(cases)} cases...")
    
    # Run all cases (collect results as list of dicts, one per target per case)
    all_results = []
    for case in cases:
        case_results = run_single_case(
            case_name=case['name'],
            steps_dict=case['steps'],
            base_config=base_config,
            data_path=args.data,
            root_outdir=args.outdir,
            targets_list=targets_list,
            step7_mode=args.step7_mode
        )
        all_results.extend(case_results)
    
    # Write summary
    write_summary(all_results, args.outdir)
    
    # Print summary table
    print_summary_table(all_results)
    
    print(f"\nTest suite complete. Results in: {args.outdir}")


if __name__ == '__main__':
    main()
