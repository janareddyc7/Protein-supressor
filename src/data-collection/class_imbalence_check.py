import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

# Sampling techniques
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
from imblearn.under_sampling import RandomUnderSampler, TomekLinks, EditedNearestNeighbours
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.pipeline import Pipeline as ImbPipeline

import warnings
warnings.filterwarnings('ignore')

class FixedImbalanceHandler:
    """
    FIXED: Class imbalance handler for truly_independent_mutations.csv
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.results = {}
    
    def load_and_prepare_data(self, file_path='../data/features/truly_independent_mutations.csv'):
        """Load and prepare the dataset - FIXED for truly_independent_mutations.csv"""
        print("🔄 Loading truly_independent_mutations.csv...")
        
        try:
            df = pd.read_csv(file_path)
            print(f"✅ Dataset loaded successfully: {df.shape}")
        except FileNotFoundError:
            print(f"❌ Error: Could not find {file_path}")
            print("Make sure 'truly_independent_mutations.csv' is in the current directory")
            return None, None, None
        
        print(f"📋 All columns: {list(df.columns)}")
        
        # ✅ FIX: Remove ALL non-numeric columns that can't be used in ML
        exclude_cols = [
            'mutation_id', 'protein', 'wt_sequence', 'mut_sequence',  # Text columns
            'wt_aa', 'mut_aa',  # ⭐ CRITICAL: Remove amino acid letters (strings)
            'pathogenic_reason',  # Text column with reasons
            'ml_target'  # Target variable (separate)
        ]
        
        # Get ONLY numeric columns for features
        all_cols = set(df.columns)
        exclude_set = set(exclude_cols)
        potential_features = list(all_cols - exclude_set)
        
        print(f"🔍 Potential feature columns: {potential_features}")
        
        # Double-check: only keep truly numeric columns
        numeric_features = []
        for col in potential_features:
            if df[col].dtype in ['int64', 'int32', 'int16', 'int8', 'float64', 'float32', 'bool']:
                numeric_features.append(col)
            else:
                print(f"⚠️  Skipping non-numeric column: {col} (type: {df[col].dtype})")
        
        print(f"✅ Final numeric features ({len(numeric_features)}): {numeric_features}")
        
        # Prepare features and target
        X = df[numeric_features].copy()
        y = df['ml_target'].copy()
        
        # ✅ FIX: Clean the data thoroughly
        print("🧹 Cleaning data...")
        
        # Replace infinite values with NaN
        X = X.replace([np.inf, -np.inf], np.nan)
        
        # Fill NaN values with median
        for col in X.columns:
            if X[col].isnull().any():
                median_val = X[col].median()
                X[col] = X[col].fillna(median_val)
                print(f"   Filled {X[col].isnull().sum()} NaN values in {col}")
        
        # Ensure target is clean
        y = y.fillna(0).astype(int)
        
        # Final verification - no string data should remain
        string_cols = X.select_dtypes(include=['object']).columns
        if len(string_cols) > 0:
            print(f"🚨 ERROR: Still found string columns: {list(string_cols)}")
            X = X.select_dtypes(exclude=['object'])
            print(f"   Removed string columns, final shape: {X.shape}")
        
        print(f"📊 Final dataset: {X.shape[0]} samples, {X.shape[1]} features")
        print(f"📊 Class distribution: {Counter(y)}")
        
        return X, y, list(X.columns)
    
    def analyze_class_distribution(self, y):
        """Analyze the class distribution"""
        class_counts = Counter(y)
        total = len(y)
        
        print("\n" + "="*50)
        print("📊 CLASS DISTRIBUTION ANALYSIS")
        print("="*50)
        
        for class_label, count in class_counts.items():
            percentage = (count / total) * 100
            class_name = "Benign" if class_label == 0 else "Pathogenic"
            print(f"{class_name} ({class_label}): {count:,} samples ({percentage:.1f}%)")
        
        imbalance_ratio = class_counts[0] / class_counts[1] if class_counts[1] > 0 else float('inf')
        print(f"Imbalance ratio: {imbalance_ratio:.1f}:1")
        
        # Create visualization
        try:
            plt.figure(figsize=(12, 5))
            
            # Bar plot
            plt.subplot(1, 2, 1)
            classes = ['Benign (0)', 'Pathogenic (1)']
            counts = [class_counts[0], class_counts[1]]
            colors = ['lightcoral', 'lightblue']
            
            bars = plt.bar(classes, counts, color=colors)
            plt.title('Class Distribution', fontsize=14, fontweight='bold')
            plt.xlabel('Class')
            plt.ylabel('Count')
            
            # Add count labels on bars
            for bar, count in zip(bars, counts):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(counts)*0.01,
                        f'{count:,}', ha='center', va='bottom', fontweight='bold')
            
            # Pie chart
            plt.subplot(1, 2, 2)
            plt.pie(counts, labels=classes, autopct='%1.1f%%',
                    colors=colors, startangle=90)
            plt.title('Class Distribution (%)', fontsize=14, fontweight='bold')
            
            plt.tight_layout()
            plt.savefig('class_distribution.png', dpi=300, bbox_inches='tight')
            print("📊 Class distribution plot saved as 'class_distribution.png'")
            plt.show()
        except Exception as e:
            print(f"⚠️ Could not create plot: {e}")
        
        return imbalance_ratio
    
    def safe_resample(self, X_train, y_train, method_name, resampler):
        """Safely apply resampling method with error handling"""
        try:
            print(f"🔄 Applying {method_name}...")
            X_resampled, y_resampled = resampler.fit_resample(X_train, y_train)
            print(f"✅ {method_name} successful")
            print(f"   Original: {Counter(y_train)}")
            print(f"   After {method_name}: {Counter(y_resampled)}")
            return X_resampled, y_resampled, True
        except Exception as e:
            print(f"❌ {method_name} failed: {e}")
            return X_train, y_train, False
    
    def evaluate_method(self, X_train, X_test, y_train, y_test, method_name):
        """Evaluate a resampling method with error handling"""
        print(f"\n--- Evaluating {method_name} ---")
        
        try:
            # Train model
            rf = RandomForestClassifier(
                n_estimators=100,
                random_state=self.random_state,
                n_jobs=-1,
                max_depth=10  # Prevent overfitting
            )
            
            rf.fit(X_train, y_train)
            
            # Predictions
            y_pred = rf.predict(X_test)
            y_pred_proba = rf.predict_proba(X_test)[:, 1]
            
            # Metrics
            f1 = f1_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_pred_proba)
            
            print(f"✅ F1 Score: {f1:.4f}")
            print(f"✅ AUC-ROC: {auc:.4f}")
            
            # Classification report
            report = classification_report(y_test, y_pred, output_dict=True)
            
            # Store results
            self.results[method_name] = {
                'f1_score': f1,
                'auc_roc': auc,
                'precision_class_1': report['1']['precision'],
                'recall_class_1': report['1']['recall'],
                'training_samples': len(X_train),
                'success': True
            }
            
            return f1, auc
            
        except Exception as e:
            print(f"❌ Evaluation failed for {method_name}: {e}")
            self.results[method_name] = {
                'f1_score': 0.0,
                'auc_roc': 0.0,
                'precision_class_1': 0.0,
                'recall_class_1': 0.0,
                'training_samples': len(X_train),
                'success': False
            }
            return 0.0, 0.0
    
    def compare_all_methods(self, X, y):
        """Compare all imbalance handling methods"""
        print("\n" + "="*60)
        print("🔬 COMPARING ALL IMBALANCE HANDLING METHODS")
        print("="*60)
        
        # Split data
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=self.random_state, stratify=y
            )
            print(f"✅ Data split successful")
            print(f"   Training: {len(X_train)} samples")
            print(f"   Testing: {len(X_test)} samples")
        except Exception as e:
            print(f"❌ Data splitting failed: {e}")
            return None
        
        # Method 1: Baseline (no resampling, just class weights)
        print("\n=== BASELINE: NO RESAMPLING (CLASS WEIGHTS) ===")
        try:
            rf_baseline = RandomForestClassifier(
                n_estimators=100,
                random_state=self.random_state,
                class_weight='balanced',
                max_depth=10
            )
            rf_baseline.fit(X_train, y_train)
            y_pred_baseline = rf_baseline.predict(X_test)
            y_pred_proba_baseline = rf_baseline.predict_proba(X_test)[:, 1]
            
            f1_baseline = f1_score(y_test, y_pred_baseline)
            auc_baseline = roc_auc_score(y_test, y_pred_proba_baseline)
            
            print(f"✅ F1 Score: {f1_baseline:.4f}")
            print(f"✅ AUC-ROC: {auc_baseline:.4f}")
            
            self.results['Baseline (Class Weights)'] = {
                'f1_score': f1_baseline,
                'auc_roc': auc_baseline,
                'training_samples': len(X_train),
                'success': True
            }
        except Exception as e:
            print(f"❌ Baseline evaluation failed: {e}")
            self.results['Baseline (Class Weights)'] = {
                'f1_score': 0.0,
                'auc_roc': 0.0,
                'training_samples': len(X_train),
                'success': False
            }
        
        # Test all resampling methods
        methods = [
            ('SMOTE', SMOTE(random_state=self.random_state)),
            ('Borderline SMOTE', BorderlineSMOTE(random_state=self.random_state)),
            ('ADASYN', ADASYN(random_state=self.random_state)),
            ('SMOTE + Tomek', SMOTETomek(random_state=self.random_state)),
            ('SMOTE + ENN', SMOTEENN(random_state=self.random_state)),
            ('Undersampling', RandomUnderSampler(random_state=self.random_state))
        ]
        
        for method_name, resampler in methods:
            print(f"\n=== {method_name.upper()} ===")
            
            # Apply resampling
            X_resampled, y_resampled, success = self.safe_resample(
                X_train, y_train, method_name, resampler
            )
            
            if success:
                # Evaluate method
                self.evaluate_method(X_resampled, X_test, y_resampled, y_test, method_name)
            else:
                print(f"⚠️ Skipping evaluation for {method_name} due to resampling failure")
        
        # Display results summary
        self.display_results_summary()
        
        return self.results
    
    def display_results_summary(self):
        """Display comprehensive results summary"""
        print("\n" + "="*80)
        print("🏆 RESULTS SUMMARY")
        print("="*80)
        
        # Filter successful results
        successful_results = {k: v for k, v in self.results.items() if v.get('success', False)}
        
        if not successful_results:
            print("❌ No methods completed successfully!")
            return
        
        # Convert results to DataFrame for easy comparison
        results_df = pd.DataFrame(successful_results).T
        results_df = results_df.sort_values('f1_score', ascending=False)
        
        print("\n🏆 RANKED BY F1 SCORE:")
        print("-" * 80)
        print(f"{'Method':<25} {'F1 Score':<10} {'AUC-ROC':<10} {'Precision':<12} {'Recall':<10} {'Samples':<10}")
        print("-" * 80)
        
        for method, row in results_df.iterrows():
            precision = row.get('precision_class_1', 0)
            recall = row.get('recall_class_1', 0)
            
            print(f"{method:<25} {row['f1_score']:<10.4f} {row['auc_roc']:<10.4f} "
                  f"{precision:<12.4f} {recall:<10.4f} {row['training_samples']:<10.0f}")
        
        # Recommendations
        print("\n" + "="*80)
        print("🎯 RECOMMENDATIONS")
        print("="*80)
        
        best_method = results_df.index[0]
        best_f1 = results_df.iloc[0]['f1_score']
        best_auc = results_df.iloc[0]['auc_roc']
        
        print(f"🏆 BEST METHOD: {best_method}")
        print(f"   F1 Score: {best_f1:.4f}")
        print(f"   AUC-ROC: {best_auc:.4f}")
        
        if best_f1 > 0.8 and best_auc > 0.9:
            print("   ✅ EXCELLENT performance!")
        elif best_f1 > 0.7 and best_auc > 0.85:
            print("   ✅ VERY GOOD performance!")
        elif best_f1 > 0.6 and best_auc > 0.8:
            print("   ✅ GOOD performance!")
        else:
            print("   ⚠️ Moderate performance - consider feature engineering")
        
        # Create comparison plot
        try:
            self.plot_results_comparison(results_df)
        except Exception as e:
            print(f"⚠️ Could not create comparison plot: {e}")
    
    def plot_results_comparison(self, results_df):
        """Create visualization comparing all methods"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # F1 Score comparison
            axes[0, 0].barh(results_df.index, results_df['f1_score'], color='skyblue')
            axes[0, 0].set_title('F1 Score Comparison', fontweight='bold')
            axes[0, 0].set_xlabel('F1 Score')
            
            # AUC-ROC comparison
            axes[0, 1].barh(results_df.index, results_df['auc_roc'], color='lightgreen')
            axes[0, 1].set_title('AUC-ROC Comparison', fontweight='bold')
            axes[0, 1].set_xlabel('AUC-ROC Score')
            
            # Training samples
            axes[1, 0].barh(results_df.index, results_df['training_samples'], color='orange')
            axes[1, 0].set_title('Training Samples', fontweight='bold')
            axes[1, 0].set_xlabel('Number of Samples')
            
            # Combined score (F1 + AUC) / 2
            combined_score = (results_df['f1_score'] + results_df['auc_roc']) / 2
            axes[1, 1].barh(results_df.index, combined_score, color='coral')
            axes[1, 1].set_title('Combined Score (F1 + AUC)/2', fontweight='bold')
            axes[1, 1].set_xlabel('Combined Score')
            
            plt.tight_layout()
            plt.savefig('imbalance_comparison.png', dpi=300, bbox_inches='tight')
            print("📊 Comparison plot saved as 'imbalance_comparison.png'")
            plt.show()
        except Exception as e:
            print(f"⚠️ Plotting failed: {e}")

def main():
    """Main execution function"""
    print("🧬 PROTEIN MUTATION CLASS IMBALANCE ANALYSIS")
    print("🎯 Dataset: truly_independent_mutations.csv")
    print("=" * 60)
    
    try:
        # Initialize handler
        handler = FixedImbalanceHandler()
        
        # Load and prepare data
        X, y, feature_cols = handler.load_and_prepare_data()
        
        if X is None:
            print("❌ Failed to load data. Exiting.")
            return None, None
        
        # Analyze class distribution
        imbalance_ratio = handler.analyze_class_distribution(y)
        
        # Compare all methods
        results = handler.compare_all_methods(X, y)
        
        if results and any(v.get('success', False) for v in results.values()):
            # Get best method
            successful_results = {k: v for k, v in results.items() if v.get('success', False)}
            results_df = pd.DataFrame(successful_results).T
            best_method = results_df.sort_values('f1_score', ascending=False).index[0]
            
            print(f"\n🎯 FINAL RECOMMENDATION: Use {best_method}")
            print("📁 Files created:")
            print("   - class_distribution.png")
            print("   - imbalance_comparison.png")
            
            return handler, best_method
        else:
            print("❌ No methods completed successfully!")
            return handler, None
            
    except Exception as e:
        print(f"❌ Critical error in main execution: {e}")
        import traceback
        traceback.print_exc()
        return None, None

if __name__ == "__main__":
    handler, best_method = main()
    
    if handler and best_method:
        print(f"\n✅ SUCCESS! Best method: {best_method}")
        print(f"🎯 Your dataset is ready for production modeling!")
    else:
        print(f"\n❌ FAILED! Check error messages above.")
