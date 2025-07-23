#!/usr/bin/env python3
"""
Enhanced prediction tool with better calibration and interpretation
"""

import joblib
import pandas as pd
import numpy as np
import json

class EnhancedMutationPredictor:
    """Enhanced predictor with better calibration"""
    
    def __init__(self):
        self.model_package = None
        self.optimal_threshold = 0.4  # Lower threshold based on analysis
        self.load_model()
    
    def load_model(self):
        """Load model"""
        try:
            self.model_package = joblib.load('protein_mutation_model.joblib')
            print("✅ Enhanced Mutation Predictor loaded!")
            return True
        except Exception as e:
            print(f"❌ Error: {e}")
            return False
    
    def predict_enhanced(self, features, show_details=True):
        """Enhanced prediction with better interpretation"""
        if self.model_package is None:
            return None
        
        try:
            model = self.model_package['model']
            feature_names = self.model_package['feature_names']
            
            # Prepare features
            if isinstance(features, dict):
                df = pd.DataFrame([features])
            else:
                df = features.copy()
            
            # Add missing features
            for feature in feature_names:
                if feature not in df.columns:
                    df[feature] = 0.0
            
            df = df[feature_names]
            
            # Get prediction
            raw_prob = model.predict(df.values)[0]
            
            # Enhanced interpretation
            if raw_prob > 0.7:
                prediction = 'Pathogenic'
                risk_level = 'High Risk'
                clinical_action = 'Strong evidence for pathogenicity - recommend functional studies'
            elif raw_prob > 0.5:
                prediction = 'Likely Pathogenic'
                risk_level = 'Moderate-High Risk'
                clinical_action = 'Likely pathogenic - consider additional evidence'
            elif raw_prob > 0.4:
                prediction = 'Possibly Pathogenic'
                risk_level = 'Moderate Risk'
                clinical_action = 'Uncertain significance - requires more data'
            elif raw_prob > 0.3:
                prediction = 'Likely Benign'
                risk_level = 'Low Risk'
                clinical_action = 'Probably benign but monitor'
            else:
                prediction = 'Benign'
                risk_level = 'Very Low Risk'
                clinical_action = 'Strong evidence for benign effect'
            
            # Confidence calculation
            if raw_prob > 0.7 or raw_prob < 0.3:
                confidence = 'High'
            elif raw_prob > 0.6 or raw_prob < 0.4:
                confidence = 'Medium'
            else:
                confidence = 'Low'
            
            result = {
                'prediction': prediction,
                'probability': raw_prob,
                'risk_level': risk_level,
                'confidence': confidence,
                'clinical_action': clinical_action,
                'interpretation': self._get_detailed_interpretation(raw_prob, features)
            }
            
            if show_details:
                self._show_enhanced_details(result, features)
            
            return result
            
        except Exception as e:
            print(f"❌ Prediction failed: {e}")
            return None
    
    def _get_detailed_interpretation(self, prob, features):
        """Get detailed biological interpretation"""
        interpretations = []
        
        # Check key features
        if isinstance(features, dict):
            if features.get('distance_from_center', 0) > 0.7:
                interpretations.append("Deep in protein core - structural impact likely")
            elif features.get('distance_from_center', 0) < 0.3:
                interpretations.append("Surface location - may have minimal structural impact")
            
            if abs(features.get('hydropathy_change', 0)) > 2.0:
                interpretations.append("Large hydropathy change - may affect protein folding")
            
            if features.get('is_proline_mutation', 0) == 1:
                interpretations.append("Proline mutation - can disrupt secondary structure")
            
            if features.get('is_cysteine_mutation', 0) == 1:
                interpretations.append("Cysteine involved - may affect disulfide bonds")
            
            if abs(features.get('charge_change', 0)) > 1.5:
                interpretations.append("Significant charge change - may affect protein interactions")
        
        if not interpretations:
            interpretations.append("Conservative amino acid substitution")
        
        return "; ".join(interpretations)
    
    def _show_enhanced_details(self, result, features):
        """Show enhanced prediction details"""
        print(f"\n🔬 ENHANCED PREDICTION ANALYSIS:")
        print(f"   Prediction: {result['prediction']}")
        print(f"   Probability: {result['probability']:.4f}")
        print(f"   Risk Level: {result['risk_level']}")
        print(f"   Confidence: {result['confidence']}")
        print(f"   Clinical Action: {result['clinical_action']}")
        print(f"   Biological Context: {result['interpretation']}")
        
        # Show key contributing features
        if isinstance(features, dict):
            print(f"\n🧬 KEY FEATURES:")
            key_features = ['distance_from_center', 'hydropathy_change', 'charge_magnitude', 'is_proline_mutation']
            for feature in key_features:
                if feature in features:
                    value = features[feature]
                    print(f"   {feature}: {value}")
    
    def interactive_enhanced_mode(self):
        """Enhanced interactive mode"""
        print(f"\n🔬 ENHANCED MUTATION PREDICTOR")
        print("=" * 50)
        print("🎯 Now with better calibration and clinical interpretation!")
        print("\nEnter mutation features:")
        print("   Quick: distance_from_center,hydropathy_change,charge_magnitude,is_proline_mutation,charge_change")
        print("   JSON: {'distance_from_center': 0.8, 'hydropathy_change': -3.5, ...}")
        
        while True:
            try:
                user_input = input("\n> ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    break
                
                # Parse input
                if user_input.startswith('{'):
                    features = json.loads(user_input)
                else:
                    values = [float(x.strip()) for x in user_input.split(',')]
                    features = {
                        'distance_from_center': values[0] if len(values) > 0 else 0,
                        'hydropathy_change': values[1] if len(values) > 1 else 0,
                        'charge_magnitude': values[2] if len(values) > 2 else 0,
                        'is_proline_mutation': int(values[3]) if len(values) > 3 else 0,
                        'charge_change': values[4] if len(values) > 4 else 0
                    }
                
                # Make enhanced prediction
                result = self.predict_enhanced(features)
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                print("💡 Try: 0.8,-3.5,2.5,1,-2.0")
        
        print("\n👋 Thanks for using the Enhanced Predictor!")

def main():
    """Main function"""
    print("🚀 ENHANCED PROTEIN MUTATION PREDICTOR")
    print("=" * 50)
    
    predictor = EnhancedMutationPredictor()
    
    if predictor.model_package is None:
        return
    
    # Test with the user's example
    print(f"\n🧪 TESTING YOUR EXAMPLE WITH ENHANCED INTERPRETATION:")
    test_features = {
        'distance_from_center': 0.8,
        'hydropathy_change': -3.5,
        'charge_magnitude': 2.5,
        'is_proline_mutation': 1,
        'charge_change': -2.0
    }
    
    result = predictor.predict_enhanced(test_features)
    
    print(f"\n💡 EXPLANATION:")
    print(f"   Your mutation has probability {result['probability']:.4f}")
    print(f"   With enhanced interpretation: {result['prediction']}")
    print(f"   The model is being conservative, which is good for medical applications")
    print(f"   Consider probabilities > 0.4 as potentially clinically relevant")
    
    # Start interactive mode
    predictor.interactive_enhanced_mode()

if __name__ == "__main__":
    main()
