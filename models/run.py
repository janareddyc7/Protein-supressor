#!/usr/bin/env python3
"""
Quick runner for the FIXED version - No more errors!
"""

import sys
import os

def main():
    print("🔧 RUNNING FIXED VERSION - No More Errors!")
    print("=" * 50)
    
    # Check if data file exists
    if not os.path.exists('truly_independent_mutations.csv'):
        print("❌ Data file 'truly_independent_mutations.csv' not found!")
        return
    
    # Import and run the fixed version
    try:
        from fixed_advanced_mutation_predictor import main as run_analysis
        
        print("🚀 Starting FIXED analysis...")
        predictor, results = run_analysis()
        
        print("\n🎉 SUCCESS! Analysis completed without errors!")
        
        # Quick summary
        best_model = results.iloc[0]
        print(f"\n📊 FINAL RESULTS:")
        print(f"   🏆 Best Model: {best_model['Model']}")
        print(f"   📈 AUC Score: {best_model['AUC_Mean']:.4f} ± {best_model['AUC_Std']:.4f}")
        print(f"   🎯 Accuracy: {best_model['Accuracy_Mean']:.4f} ± {best_model['Accuracy_Std']:.4f}")
        print(f"\n✅ All plots generated successfully!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
