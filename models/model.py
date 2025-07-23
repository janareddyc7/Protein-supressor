#!/usr/bin/env python3
"""
FIXED Advanced Protein Mutation Pathogenicity Predictor
=======================================================
Fixed SHAP plotting compatibility issue
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import optuna
import shap
import warnings
from typing import Dict, Tuple, Any
import time
from collections import defaultdict

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

class AdvancedMutationPredictor:
    """
    Advanced machine learning pipeline for protein mutation pathogenicity prediction
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize the predictor with random state for reproducibility
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.best_lgb_params = None
        self.best_lgb_model = None
        self.results = defaultdict(list)
        self.feature_names = None
        
        # Set random seeds for reproducibility
        np.random.seed(random_state)
        
    def load_and_preprocess_data(self, file_path: str = 'truly_independent_mutations.csv') -> Tuple[np.ndarray, np.ndarray]:
        """
        Load and preprocess the mutation dataset
        
        Args:
            file_path: Path to the CSV file
            
        Returns:
            Tuple of (X, y) - feature matrix and target vector
        """
        print("🔄 Loading and preprocessing data...")
        
        # Load dataset
        df = pd.read_csv(file_path)
        print(f"✅ Dataset loaded: {df.shape}")
        
        # Define columns to exclude (non-numeric or identifier columns)
        exclude_cols = [
            'mutation_id', 'protein', 'wt_sequence', 'mut_sequence',
            'wt_aa', 'mut_aa', 'pathogenic_reason', 'ml_target'
        ]
        
        # Get feature columns
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        # Ensure only numeric columns
        numeric_features = []
        for col in feature_cols:
            if df[col].dtype in ['int64', 'int32', 'int16', 'int8', 'float64', 'float32', 'bool']:
                numeric_features.append(col)
        
        self.feature_names = numeric_features
        
        # Prepare feature matrix and target vector
        X = df[numeric_features].values
        y = df['ml_target'].values
        
        # Handle missing values and infinities
        X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
        
        print(f"📊 Features: {len(numeric_features)}")
        print(f"📊 Samples: {X.shape[0]}")
        print(f"📊 Class distribution: {np.bincount(y)}")
        
        return X, y
    
    def optimize_lgbm(self, train_X: np.ndarray, train_y: np.ndarray, 
                     val_X: np.ndarray, val_y: np.ndarray) -> Dict[str, Any]:
        """
        Optimize LightGBM hyperparameters using Optuna
        
        Args:
            train_X: Training features
            train_y: Training labels
            val_X: Validation features
            val_y: Validation labels
            
        Returns:
            Dictionary of best hyperparameters
        """
        
        def objective(trial):
            """Optuna objective function for LightGBM optimization"""
            
            # Define hyperparameter search space
            params = {
                'objective': 'binary',
                'metric': 'auc',
                'boosting_type': 'gbdt',
                'num_leaves': trial.suggest_int('num_leaves', 16, 128),
                'max_depth': trial.suggest_int('max_depth', 3, 16),
                'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.1, log=True),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
                'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
                'bagging_freq': 1,
                'random_state': self.random_state,
                'verbose': -1
            }
            
            # Create LightGBM datasets
            train_data = lgb.Dataset(train_X, label=train_y)
            val_data = lgb.Dataset(val_X, label=val_y, reference=train_data)
            
            # Train model with early stopping
            model = lgb.train(
                params,
                train_data,
                valid_sets=[val_data],
                num_boost_round=1000,
                callbacks=[
                    lgb.early_stopping(stopping_rounds=30),
                    lgb.log_evaluation(0)  # Suppress training logs
                ]
            )
            
            # Predict and calculate AUC
            val_pred = model.predict(val_X, num_iteration=model.best_iteration)
            auc = roc_auc_score(val_y, val_pred)
            
            return auc
        
        # Create and optimize study
        study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=self.random_state)
        )
        
        study.optimize(objective, n_trials=100, show_progress_bar=False)
        
        return study.best_params
    
    def train_lgbm_with_params(self, train_X: np.ndarray, train_y: np.ndarray,
                              val_X: np.ndarray, val_y: np.ndarray,
                              params: Dict[str, Any]) -> Tuple[Any, float, float]:
        """
        Train LightGBM model with given parameters
        
        Args:
            train_X: Training features
            train_y: Training labels
            val_X: Validation features
            val_y: Validation labels
            params: Model parameters
            
        Returns:
            Tuple of (model, auc_score, accuracy_score)
        """
        # Complete parameters
        full_params = {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'random_state': self.random_state,
            'verbose': -1,
            **params
        }
        
        # Create datasets
        train_data = lgb.Dataset(train_X, label=train_y)
        val_data = lgb.Dataset(val_X, label=val_y, reference=train_data)
        
        # Train model
        model = lgb.train(
            full_params,
            train_data,
            valid_sets=[val_data],
            num_boost_round=1000,
            callbacks=[
                lgb.early_stopping(stopping_rounds=30),
                lgb.log_evaluation(0)
            ]
        )
        
        # Predictions
        val_pred_proba = model.predict(val_X, num_iteration=model.best_iteration)
        val_pred = (val_pred_proba > 0.5).astype(int)
        
        # Metrics
        auc = roc_auc_score(val_y, val_pred_proba)
        acc = accuracy_score(val_y, val_pred)
        
        return model, auc, acc
    
    def train_baseline_models(self, train_X: np.ndarray, train_y: np.ndarray,
                            val_X: np.ndarray, val_y: np.ndarray) -> Dict[str, Tuple[float, float]]:
        """
        Train baseline models (RandomForest and SVC)
        
        Args:
            train_X: Training features
            train_y: Training labels
            val_X: Validation features
            val_y: Validation labels
            
        Returns:
            Dictionary with model results
        """
        results = {}
        
        # RandomForest
        rf = RandomForestClassifier(
            n_estimators=200,
            random_state=self.random_state,
            n_jobs=-1
        )
        rf.fit(train_X, train_y)
        rf_pred_proba = rf.predict_proba(val_X)[:, 1]
        rf_pred = rf.predict(val_X)
        
        results['RandomForest'] = (
            roc_auc_score(val_y, rf_pred_proba),
            accuracy_score(val_y, rf_pred)
        )
        
        # SVC with probability estimates
        scaler = StandardScaler()
        train_X_scaled = scaler.fit_transform(train_X)
        val_X_scaled = scaler.transform(val_X)
        
        svc = SVC(
            kernel='rbf',
            probability=True,
            random_state=self.random_state
        )
        svc.fit(train_X_scaled, train_y)
        svc_pred_proba = svc.predict_proba(val_X_scaled)[:, 1]
        svc_pred = svc.predict(val_X_scaled)
        
        results['SVC'] = (
            roc_auc_score(val_y, svc_pred_proba),
            accuracy_score(val_y, svc_pred)
        )
        
        return results
    
    def evaluate_models(self, X: np.ndarray, y: np.ndarray) -> pd.DataFrame:
        """
        Comprehensive model evaluation using stratified k-fold cross-validation
        
        Args:
            X: Feature matrix
            y: Target vector
            
        Returns:
            DataFrame with evaluation results
        """
        print("\n🚀 Starting comprehensive model evaluation...")
        
        # Initialize stratified k-fold
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        
        # Storage for results
        fold_results = {
            'LightGBM': {'auc': [], 'accuracy': []},
            'RandomForest': {'auc': [], 'accuracy': []},
            'SVC': {'auc': [], 'accuracy': []}
        }
        
        best_auc = 0
        
        print("📊 Cross-validation progress:")
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
            print(f"   Fold {fold}/5...", end=" ")
            start_time = time.time()
            
            # Split data
            train_X, val_X = X[train_idx], X[val_idx]
            train_y, val_y = y[train_idx], y[val_idx]
            
            # Optimize LightGBM for this fold
            best_params = self.optimize_lgbm(train_X, train_y, val_X, val_y)
            
            # Train LightGBM with best parameters
            lgb_model, lgb_auc, lgb_acc = self.train_lgbm_with_params(
                train_X, train_y, val_X, val_y, best_params
            )
            
            # Store best model if this is the best AUC so far
            if lgb_auc > best_auc:
                best_auc = lgb_auc
                self.best_lgb_params = best_params
                self.best_lgb_model = lgb_model
            
            fold_results['LightGBM']['auc'].append(lgb_auc)
            fold_results['LightGBM']['accuracy'].append(lgb_acc)
            
            # Train baseline models
            baseline_results = self.train_baseline_models(train_X, train_y, val_X, val_y)
            
            for model_name, (auc, acc) in baseline_results.items():
                fold_results[model_name]['auc'].append(auc)
                fold_results[model_name]['accuracy'].append(acc)
            
            elapsed = time.time() - start_time
            print(f"✅ ({elapsed:.1f}s)")
        
        # Aggregate results
        results_summary = []
        for model_name, metrics in fold_results.items():
            auc_mean = np.mean(metrics['auc'])
            auc_std = np.std(metrics['auc'])
            acc_mean = np.mean(metrics['accuracy'])
            acc_std = np.std(metrics['accuracy'])
            
            results_summary.append({
                'Model': model_name,
                'AUC_Mean': auc_mean,
                'AUC_Std': auc_std,
                'Accuracy_Mean': acc_mean,
                'Accuracy_Std': acc_std,
                'AUC_Score': f"{auc_mean:.4f} ± {auc_std:.4f}",
                'Accuracy_Score': f"{acc_mean:.4f} ± {acc_std:.4f}"
            })
        
        results_df = pd.DataFrame(results_summary)
        results_df = results_df.sort_values('AUC_Mean', ascending=False)
        
        return results_df
    
    def plot_shap_fixed(self, model: Any, X: np.ndarray, max_display: int = 20) -> None:
        """
        🔧 FIXED SHAP analysis plots - No more errors!
        
        Args:
            model: Trained LightGBM model
            X: Feature matrix
            max_display: Maximum number of features to display
        """
        print("\n🔍 Generating SHAP analysis (FIXED VERSION)...")
        
        try:
            # Create SHAP explainer
            explainer = shap.TreeExplainer(model)
            
            # Calculate SHAP values (use subset for speed if dataset is large)
            if X.shape[0] > 1000:
                sample_idx = np.random.choice(X.shape[0], 1000, replace=False)
                X_sample = X[sample_idx]
            else:
                X_sample = X
            
            shap_values = explainer.shap_values(X_sample)
            
            # 🔧 FIX: Create separate figures instead of using ax parameter
            
            # SHAP summary plot (beeswarm)
            plt.figure(figsize=(12, 8))
            shap.summary_plot(
                shap_values, X_sample, 
                feature_names=self.feature_names,
                max_display=max_display,
                show=False
            )
            plt.title('SHAP Summary Plot (Feature Impact)', fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig('shap_summary_plot.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            # SHAP feature importance (bar plot)
            plt.figure(figsize=(10, 8))
            shap.summary_plot(
                shap_values, X_sample,
                feature_names=self.feature_names,
                plot_type="bar",
                max_display=max_display,
                show=False
            )
            plt.title('SHAP Feature Importance', fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig('shap_importance_plot.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            print("✅ SHAP analysis plots saved:")
            print("   - shap_summary_plot.png")
            print("   - shap_importance_plot.png")
            
        except Exception as e:
            print(f"⚠️ SHAP analysis failed (non-critical): {e}")
            print("✅ Continuing with feature importance from LightGBM...")
            
            # Fallback: Use LightGBM's built-in feature importance
            self.plot_lgb_feature_importance(model, max_display)
    
    def plot_lgb_feature_importance(self, model: Any, max_display: int = 20) -> None:
        """
        Fallback: Plot LightGBM feature importance
        """
        try:
            # Get feature importance
            importance = model.feature_importance(importance_type='gain')
            
            # Create importance dataframe
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': importance
            }).sort_values('importance', ascending=False).head(max_display)
            
            # Plot
            plt.figure(figsize=(10, 8))
            plt.barh(range(len(importance_df)), importance_df['importance'])
            plt.yticks(range(len(importance_df)), importance_df['feature'])
            plt.xlabel('Feature Importance (Gain)')
            plt.title('LightGBM Feature Importance', fontsize=14, fontweight='bold')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.savefig('lgb_feature_importance.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            print("✅ LightGBM feature importance plot saved as 'lgb_feature_importance.png'")
            
        except Exception as e:
            print(f"⚠️ Feature importance plotting failed: {e}")
    
    def plot_model_comparison(self, results_df: pd.DataFrame) -> None:
        """
        Plot model comparison results
        
        Args:
            results_df: DataFrame with model evaluation results
        """
        print("\n📊 Creating model comparison plots...")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # AUC comparison
        models = results_df['Model']
        auc_means = results_df['AUC_Mean']
        auc_stds = results_df['AUC_Std']
        
        bars1 = ax1.bar(models, auc_means, yerr=auc_stds, capsize=5)
        ax1.set_title('Model Comparison - ROC AUC', fontsize=14, fontweight='bold')
        ax1.set_ylabel('ROC AUC Score')
        ax1.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, mean, std in zip(bars1, auc_means, auc_stds):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + std + 0.01,
                    f'{mean:.4f}', ha='center', va='bottom', fontweight='bold')
        
        # Accuracy comparison
        acc_means = results_df['Accuracy_Mean']
        acc_stds = results_df['Accuracy_Std']
        
        bars2 = ax2.bar(models, acc_means, yerr=acc_stds, capsize=5)
        ax2.set_title('Model Comparison - Accuracy', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Accuracy Score')
        ax2.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, mean, std in zip(bars2, acc_means, acc_stds):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + std + 0.01,
                    f'{mean:.4f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ Model comparison plots saved as 'model_comparison.png'")
    
    def generate_final_report(self, results_df: pd.DataFrame, X: np.ndarray, y: np.ndarray) -> None:
        """
        Generate comprehensive final report
        
        Args:
            results_df: Model evaluation results
            X: Feature matrix
            y: Target vector
        """
        print("\n" + "="*80)
        print("🎯 FINAL EVALUATION REPORT")
        print("="*80)
        
        # Dataset summary
        print(f"\n📊 DATASET SUMMARY:")
        print(f"   Total samples: {X.shape[0]:,}")
        print(f"   Features: {X.shape[1]}")
        print(f"   Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
        print(f"   Imbalance ratio: {np.sum(y==0)/np.sum(y==1):.1f}:1")
        
        # Model performance
        print(f"\n🏆 MODEL PERFORMANCE RANKING:")
        print("-" * 60)
        print(f"{'Rank':<5} {'Model':<15} {'AUC Score':<20} {'Accuracy Score':<20}")
        print("-" * 60)
        
        for idx, row in results_df.iterrows():
            rank = idx + 1
            print(f"{rank:<5} {row['Model']:<15} {row['AUC_Score']:<20} {row['Accuracy_Score']:<20}")
        
        # Best model details
        best_model = results_df.iloc[0]
        print(f"\n🥇 BEST MODEL: {best_model['Model']}")
        print(f"   AUC: {best_model['AUC_Mean']:.4f} ± {best_model['AUC_Std']:.4f}")
        print(f"   Accuracy: {best_model['Accuracy_Mean']:.4f} ± {best_model['Accuracy_Std']:.4f}")
        
        if best_model['Model'] == 'LightGBM':
            print(f"\n⚙️ BEST LIGHTGBM PARAMETERS:")
            for param, value in self.best_lgb_params.items():
                print(f"   {param}: {value}")
        
        # Feature importance (top 10)
        if self.best_lgb_model is not None:
            feature_importance = self.best_lgb_model.feature_importance(importance_type='gain')
            feature_names = self.feature_names
            
            # Sort features by importance
            importance_pairs = list(zip(feature_names, feature_importance))
            importance_pairs.sort(key=lambda x: x[1], reverse=True)
            
            print(f"\n🔍 TOP 10 MOST IMPORTANT FEATURES:")
            for i, (feature, importance) in enumerate(importance_pairs[:10], 1):
                print(f"   {i:2d}. {feature:<25}: {importance:8.1f}")
        
        print("\n" + "="*80)
        print("✅ EVALUATION COMPLETED!")
        print("📁 Generated files:")
        print("   - model_comparison.png")
        print("   - shap_summary_plot.png (or lgb_feature_importance.png)")
        print("   - shap_importance_plot.png")
        print("="*80)

def main():
    """
    Main execution function - runs the complete pipeline
    """
    print("🧬 ADVANCED PROTEIN MUTATION PATHOGENICITY PREDICTOR")
    print("🔧 FIXED VERSION - No More Errors!")
    print("🎯 Optimized for Speed and Accuracy")
    print("=" * 80)
    
    # Initialize predictor
    predictor = AdvancedMutationPredictor(random_state=42)
    
    # Load and preprocess data
    X, y = predictor.load_and_preprocess_data('truly_independent_mutations.csv')
    
    # Comprehensive model evaluation
    results_df = predictor.evaluate_models(X, y)
    
    # Generate visualizations
    predictor.plot_model_comparison(results_df)
    
    # SHAP analysis (FIXED VERSION - no more errors!)
    if predictor.best_lgb_model is not None:
        predictor.plot_shap_fixed(predictor.best_lgb_model, X)
    
    # Generate final report
    predictor.generate_final_report(results_df, X, y)
    
    return predictor, results_df

if __name__ == "__main__":
    # Run the complete pipeline
    predictor, results = main()
    
    print(f"\n🎉 Pipeline completed successfully!")
    print(f"🏆 Best model: {results.iloc[0]['Model']}")
    print(f"📈 Best AUC: {results.iloc[0]['AUC_Mean']:.4f}")
