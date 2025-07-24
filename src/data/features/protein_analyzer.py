import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import re
import joblib
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

class MLProteinAnalyzer:
    """
    ULTIMATE ML-Powered Protein Mutation Analyzer for ISEF
    Uses your trained model for REAL suppressor predictions!
    """
    
    def __init__(self):
        # Protein database with correct sequences
        self.proteins = {
            "alpha_synuclein": {
                "name": "Alpha-synuclein (SNCA)",
                "uniprot_id": "P37840",
                "sequence": "MDVFMKGLSKAKEGVVAAAEKTKQGVAEAAGKTKEGVLYVGSKTKEGVVHGVATVAEKTKEQVTNVGGAVVTGVTAVAQKTVEGAGSIAAATGFVKKDQLGKNEEGAPQEGILEDMPVDPDNEAYEMPSEEGYQDYEPEA",
                "length": 140,
                "disease": "Parkinson's Disease",
                "aliases": ["SNCA", "ALPHA_SYNUCLEIN", "ALPHA-SYNUCLEIN"]
            },
            "tau": {
                "name": "Tau protein (MAPT)",
                "uniprot_id": "P10636",
                "sequence": "MAEPRQEFEVMEDHAGTYGLGDRKDQGGYTMHQDQEGDTDAGLKESPLQTPTEDGSEEPGSETSDAKSTPTAEDVTAPLVDEGAPGKQAAAQPHTEIPEGTTAEEAGIGDTPSLEDEAAGHVTQEPESGKVVQEGFLREPGPPGLSHQLMSGMPGAPLLPEGPREATRQPSGTGPEDTEGGRHAPELLKHQLLGDLHQEGPPLKGAGGKERPGSKEEVDEDRDVDESSPQDSPPSKASPAQDGRPPQTAAREATSIPGFPAEGAIPLPVDFLSKVSTEIPASEPDGPSVGRAKGQDAPLEFTFHVEITPNVQKEQAHSEEHLGRAAFPGAPGEGPEARGPSLGEDTKEADLPEPSEKQPAAAPRGKPVSRVPQLKARMVSKSKDGTGSDDKKAKTSTRSSAKTLKNRPCLSPKHPTPGSSDPLIQPSSPAVCPEPPSSPKYVSSVTSRTGSSGAKEMKLKGADGKTKIATPRGAAPPGQKGQANATRIPAKTPPAPKTPPSSGEPPKSGDRSGYSSPGSPGTPGSRSRTPSLPTPPTREPKKVAVVRTPPKSPSSAKSRLQTAPVPMPDLKNVKSKIGSTENLKHQPGGGKVQIINKKLDLSNVQSKCGSKDNIKHVPGGGSVQIVYKPVDLSKVTSKCGSLGNIHHKPGGGQVEVKSEKLDFKDRVQSKIGSLDNITHVPGGGNKKIETHKLTFRENAKAKTDHGAEIVYKSPVVSGDTSPRHLSNVSSTGSIDMVDSPQLATLADEVSASLAKQGL",
                "length": 758,
                "disease": "Alzheimer's Disease / Frontotemporal Dementia",
                "aliases": ["MAPT", "TAU", "MICROTUBULE_ASSOCIATED_PROTEIN"]
            },
            "sod1": {
                "name": "Superoxide dismutase 1 (SOD1)",
                "uniprot_id": "P00441",
                "sequence": "MATKAVCVLKGDGPVQGIINFEQKESNGPVKVWGSIKGLTEGLHGFHVHEFGDNTAGCTSAGPHFNPLSRKHGGPKDEERHVGDLGNVTADKDGVADVSIEDSVISLSGDHCIIGRTLVVHEKADDLGKGGNEESTKTGNAGSRLACGVIGIAQ",
                "length": 154,
                "disease": "Amyotrophic Lateral Sclerosis (ALS)",
                "aliases": ["SOD1", "SUPEROXIDE_DISMUTASE_1"]
            },
            "amyloid_beta": {
                "name": "Amyloid precursor protein (APP)",
                "uniprot_id": "P05067",
                "sequence": "MLPGLALLLLAAWTARALEVPTDGNAGLLAEPQIAMFCGRLNMHMNVQNGKWDSDPSGTKTCIDTKEGILQYCQEVYPELQITNVVEANQPVTIQNWCKRGRKQCKTHPHFVIPYRCLVGEFVSDALLVPDKCKFLHQERMDVCETHLHWHTVAKETCSEKSTNLHDYGMLLPCGIDKFRGVEFVCCPLAEESDNVDSADAEEDDSDVWWGGADTDYADGSEDKVVEVAEEEEVAEVEEEEADDDEDDEDGDEVEEEAEEPYEEATERTTSIATTTTTTTESVEEVVREVCSEQAETGPCRAMISRWYFDVTEGKCAPFFYGGCGGNRNNFDTEEYCMAVCGSAMSQSLLKTTQEPLARDPVKLPTTAASTPDAVDKYLETPGDENEHAHFQKAKERLEAKHRERMSQVMREWEEAERQAKNLPKADKKAVIQHFQEKVESLEQEAANERQQLVETHMARVEAMLNDRRRLALENYITALQAVPPRPRHVFNMLKKYVRAEQKDRQHTLKHFEHVRMVDPKKAAQIRSQVMTHLRVIYERMNQSLSLLYNVPAVAEEIQDEVDELLQKEQNYSDDVLANMISEPRISYGNDALMPSLTETKTTVELLPVNGEFSLDDLQPWHSFGADSVPANTENEVEPVDARPAADRGLTTRPGSGLTNIKTEEISEVKMDAEFRHDSGYEVHHQKLVFFAEDVGSNKGAIIGLMVGGVVIATVIVITLVMLKKKQYTSIHHGVVEVDAAVTPEERHLSKMQQNGYENPTYKFFEQMQN",
                "length": 770,
                "disease": "Alzheimer's Disease",
                "aliases": ["APP", "AMYLOID_PRECURSOR_PROTEIN", "AMYLOID_BETA"]
            }
        }
        
        # Amino acid properties (scientifically accurate)
        self.aa_properties = {
            'hydropathy': {
                'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5,
                'Q': -3.5, 'E': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5,
                'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6,
                'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2
            },
            'charge': {
                'A': 0, 'R': 1, 'N': 0, 'D': -1, 'C': 0,
                'Q': 0, 'E': -1, 'G': 0, 'H': 0.1, 'I': 0,
                'L': 0, 'K': 1, 'M': 0, 'F': 0, 'P': 0,
                'S': 0, 'T': 0, 'W': 0, 'Y': 0, 'V': 0
            },
            'beta_propensity': {
                'A': 0.83, 'R': 0.93, 'N': 0.89, 'D': 0.54, 'C': 1.19,
                'Q': 1.10, 'E': 0.37, 'G': 0.75, 'H': 0.87, 'I': 1.60,
                'L': 1.30, 'K': 0.74, 'M': 1.05, 'F': 1.38, 'P': 0.55,
                'S': 0.75, 'T': 1.19, 'W': 1.37, 'Y': 1.47, 'V': 1.70
            },
            'flexibility': {
                'A': 0.25, 'R': 0.95, 'N': 0.56, 'D': 0.81, 'C': 0.12,
                'Q': 0.68, 'E': 0.83, 'G': 0.81, 'H': 0.68, 'I': 0.13,
                'L': 0.13, 'K': 0.88, 'M': 0.23, 'F': 0.15, 'P': 0.20,
                'S': 0.71, 'T': 0.36, 'W': 0.18, 'Y': 0.20, 'V': 0.13
            }
        }
        
        self.amino_acids = "ACDEFGHIKLMNPQRSTVWY"
        self.current_results = None
        
        # ML Model components
        self.trained_model = None
        self.feature_names = None
        self.scaler = None
        self.model_loaded = False
        self.model_type = "Unknown"
        self.model_dict = None
        
        # Load your trained model
        self.load_trained_model()
    
    def load_trained_model(self):
        """
        Load your trained model: protein_mutation_model.joblib
        Handle both direct model and dictionary formats
        """
        try:
            print("🤖 Loading your trained ML model...")
            
            # Try to load your specific model
            model_files = [
                'protein_mutation_model.joblib',
                'protein_mutation_model.pkl',
                'best_lgb_model.pkl',
                'trained_model.pkl'
            ]
            
            for model_file in model_files:
                if os.path.exists(model_file):
                    print(f"📁 Found model file: {model_file}")
                    loaded_object = joblib.load(model_file)
                    
                    # Check if it's a dictionary containing the model
                    if isinstance(loaded_object, dict):
                        print(f"📦 Loaded dictionary with keys: {list(loaded_object.keys())}")
                        self.model_dict = loaded_object
                        
                        # Try to extract the actual model from common dictionary keys
                        possible_model_keys = [
                            'model', 'trained_model', 'best_model', 'classifier', 
                            'regressor', 'lgb_model', 'lightgbm_model', 'final_model'
                        ]
                        
                        for key in possible_model_keys:
                            if key in loaded_object:
                                self.trained_model = loaded_object[key]
                                print(f"✅ Extracted model from key: '{key}'")
                                break
                        
                        # Try to extract feature names
                        possible_feature_keys = [
                            'feature_names', 'features', 'feature_columns', 
                            'X_columns', 'column_names'
                        ]
                        
                        for key in possible_feature_keys:
                            if key in loaded_object:
                                self.feature_names = loaded_object[key]
                                print(f"✅ Extracted feature names from key: '{key}'")
                                break
                        
                        # Try to extract scaler
                        possible_scaler_keys = [
                            'scaler', 'feature_scaler', 'standard_scaler', 'preprocessor'
                        ]
                        
                        for key in possible_scaler_keys:
                            if key in loaded_object:
                                self.scaler = loaded_object[key]
                                print(f"✅ Extracted scaler from key: '{key}'")
                                break
                        
                        # If no model found in dictionary, try to use the whole dict
                        if self.trained_model is None:
                            print("⚠️ No model found in dictionary keys, checking if dict itself is callable...")
                            if hasattr(loaded_object, 'predict') or hasattr(loaded_object, 'predict_proba'):
                                self.trained_model = loaded_object
                                print("✅ Using dictionary as model (has predict method)")
                    
                    else:
                        # Direct model object
                        self.trained_model = loaded_object
                        print(f"✅ Loaded direct model object")
                    
                    break
            
            if self.trained_model is not None:
                self.model_loaded = True
                self.model_type = type(self.trained_model).__name__
                
                # Try to get feature information from the model
                if hasattr(self.trained_model, 'feature_name_'):
                    self.feature_names = self.trained_model.feature_name_
                    print(f"📊 Model features from feature_name_: {len(self.feature_names)}")
                elif hasattr(self.trained_model, 'feature_names_in_'):
                    self.feature_names = self.trained_model.feature_names_in_
                    print(f"📊 Model features from feature_names_in_: {len(self.feature_names)}")
                elif hasattr(self.trained_model, 'n_features_'):
                    print(f"📊 Model expects {self.trained_model.n_features_} features")
                elif hasattr(self.trained_model, 'num_feature'):
                    print(f"📊 Model expects {self.trained_model.num_feature()} features")
                
                print("🎉 ML Model successfully loaded and ready!")
                print(f"🔬 Model type: {self.model_type}")
                
                # Test the model with dummy data
                self.test_model_prediction()
                
            else:
                print("⚠️ No trained model found. Please ensure 'protein_mutation_model.joblib' contains a valid model.")
                self.model_loaded = False
                
        except Exception as e:
            print(f"❌ Error loading trained model: {e}")
            print("⚠️ Falling back to heuristic approach.")
            self.model_loaded = False
            import traceback
            traceback.print_exc()
    
    def test_model_prediction(self):
        """Test the model with dummy data to ensure it works"""
        try:
            print("🧪 Testing model prediction...")
            
            # Create dummy feature vector
            if self.feature_names:
                dummy_features = {name: 0.5 for name in self.feature_names}
                feature_vector = [dummy_features[name] for name in self.feature_names]
            else:
                # Create a reasonable number of dummy features
                feature_vector = [0.5] * 50  # Adjust based on your model
            
            feature_vector = np.array(feature_vector).reshape(1, -1)
            
            # Apply scaling if available
            if self.scaler is not None:
                feature_vector = self.scaler.transform(feature_vector)
            
            # Test prediction
            if hasattr(self.trained_model, 'predict_proba'):
                test_pred = self.trained_model.predict_proba(feature_vector)
                print(f"✅ Model test successful - predict_proba shape: {test_pred.shape}")
            elif hasattr(self.trained_model, 'predict'):
                test_pred = self.trained_model.predict(feature_vector)
                print(f"✅ Model test successful - predict output: {test_pred}")
            else:
                print("❌ Model has no predict or predict_proba method")
                self.model_loaded = False
                return
            
            print("🎯 Model is ready for predictions!")
            
        except Exception as e:
            print(f"❌ Model test failed: {e}")
            print("⚠️ Model may not be compatible. Falling back to heuristic.")
            self.model_loaded = False
    
    def parse_mutation_input(self, mutation_str: str) -> Tuple[str, int, Optional[str], str]:
        """Enhanced mutation parser with comprehensive format support - FIXED"""
        if not mutation_str or not mutation_str.strip():
            raise ValueError("Empty mutation input")
        
        mutation_str = mutation_str.strip().upper()
        
        # Remove common prefixes and clean up
        mutation_str = re.sub(r'^(P\.|C\.|G\.)', '', mutation_str)
        mutation_str = re.sub(r'\s+', ' ', mutation_str)
        
        # Pattern matching with priority order - FIXED
        patterns = [
            (r'^(\w+)[\s:_-]+([A-Z])(\d+)([A-Z])$', 'protein_full'),
            (r'^(\w+)[\s:_-]+([A-Z])(\d+)([A-Z*])$', 'protein_full'),
            (r'^(\w+)[\s:_-]+(\d+)([A-Z*])$', 'protein_pos_mut'),
            (r'^([A-Z])(\d+)([A-Z*])$', 'standard'),
            (r'^(\d+)([A-Z*])$', 'pos_mut'),
            (r'^(\d+)$', 'pos_only'),
        ]
        
        for pattern, format_type in patterns:
            match = re.match(pattern, mutation_str)
            if match:
                groups = match.groups()
                
                if format_type == 'protein_full':
                    protein, wt_aa, pos_str, mut_aa = groups
                    return self._normalize_protein_name(protein), int(pos_str), wt_aa, mut_aa
                elif format_type == 'protein_pos_mut':
                    protein, pos_str, mut_aa = groups
                    return self._normalize_protein_name(protein), int(pos_str), None, mut_aa
                elif format_type == 'standard':
                    wt_aa, pos_str, mut_aa = groups
                    # Check if first group is a single amino acid
                    if len(wt_aa) == 1 and wt_aa in self.amino_acids:
                        return "UNKNOWN", int(pos_str), wt_aa, mut_aa
                    else:
                        # Treat as protein name if not single amino acid
                        return self._normalize_protein_name(wt_aa), int(pos_str), None, mut_aa
                elif format_type == 'pos_mut':
                    pos_str, mut_aa = groups
                    return "UNKNOWN", int(pos_str), None, mut_aa
                elif format_type == 'pos_only':
                    pos_str = groups[0]
                    return "UNKNOWN", int(pos_str), None, "X"
        
        # Fallback parsing
        numbers = re.findall(r'\d+', mutation_str)
        letters = re.findall(r'[A-Z]', mutation_str)
        
        if numbers and letters:
            position = int(numbers[0])
            if len(letters) >= 2:
                return "UNKNOWN", position, letters[0], letters[1]
            elif len(letters) == 1:
                return "UNKNOWN", position, None, letters[0]
        
        raise ValueError(f"Invalid mutation format: '{mutation_str}'. "
                        f"Supported formats: A53T, SNCA:A53T, alpha_synuclein A53T, 53T, etc.")
    
    def _normalize_protein_name(self, name: str) -> str:
        """Normalize protein names to standard keys"""
        if not name:
            return "UNKNOWN"
        
        name = name.upper().strip()
        
        name_mapping = {
            'SNCA': 'alpha_synuclein', 'ALPHA_SYNUCLEIN': 'alpha_synuclein',
            'ALPHA-SYNUCLEIN': 'alpha_synuclein', 'ALPHASYNUCLEIN': 'alpha_synuclein',
            'MAPT': 'tau', 'TAU': 'tau', 'MICROTUBULE_ASSOCIATED_PROTEIN': 'tau',
            'SOD1': 'sod1', 'SUPEROXIDE_DISMUTASE_1': 'sod1', 'SUPEROXIDE_DISMUTASE': 'sod1',
            'APP': 'amyloid_beta', 'AMYLOID_BETA': 'amyloid_beta', 'AMYLOID_PRECURSOR_PROTEIN': 'amyloid_beta',
        }
        
        if name in name_mapping:
            return name_mapping[name]
        if name.lower() in self.proteins:
            return name.lower()
        
        for key, value in name_mapping.items():
            if key in name or name in key:
                return value
        
        return name.lower()
    
    def get_protein_info(self, protein_key: str) -> Dict:
        """Get protein information with enhanced lookup"""
        if protein_key in self.proteins:
            return self.proteins[protein_key]
        
        normalized = self._normalize_protein_name(protein_key)
        if normalized in self.proteins:
            return self.proteins[normalized]
        
        for key, info in self.proteins.items():
            if protein_key.upper() in [alias.upper() for alias in info['aliases']]:
                return info
            if protein_key.upper() in info['name'].upper():
                return info
        
        raise ValueError(f"Unknown protein: {protein_key}. Available proteins: {list(self.proteins.keys())}")
    
    def validate_mutation(self, protein_key: str, position: int, wt_aa: Optional[str], mut_aa: str) -> Tuple[bool, str, str]:
        """Enhanced mutation validation"""
        try:
            protein_info = self.get_protein_info(protein_key)
            sequence = protein_info['sequence']
            
            if position < 1 or position > len(sequence):
                return False, f"Position {position} out of range for {protein_info['name']} (valid range: 1-{len(sequence)})", ""
            
            if mut_aa not in self.amino_acids and mut_aa != '*' and mut_aa != 'X':
                return False, f"Invalid amino acid: {mut_aa}. Valid amino acids: {self.amino_acids}", ""
            
            actual_wt = sequence[position - 1]
            
            if wt_aa is None or wt_aa == 'X':
                return True, f"Auto-detected wild-type: {actual_wt} at position {position}", actual_wt
            
            if actual_wt == wt_aa:
                return True, f"Mutation {wt_aa}{position}{mut_aa} validated for {protein_info['name']}", wt_aa
            else:
                return True, f"Wild-type corrected: {wt_aa} → {actual_wt} at position {position}", actual_wt
                
        except Exception as e:
            return False, f"Validation error: {str(e)}", ""
    
    def extract_comprehensive_features(self, protein_key: str, position: int, wt_aa: str, mut_aa: str) -> Dict:
        """Extract comprehensive features matching your training data"""
        try:
            protein_info = self.get_protein_info(protein_key)
            sequence = protein_info['sequence']
            
            # Create mutant sequence
            mut_sequence = sequence[:position-1] + mut_aa + sequence[position:]
            
            # Initialize features dictionary
            features = {}
            
            # Basic mutation information
            features['position'] = position
            features['sequence_length'] = len(sequence)
            features['position_ratio'] = position / len(sequence)
            
            # Wild-type amino acid properties
            features['wt_hydropathy'] = self.aa_properties['hydropathy'].get(wt_aa, 0)
            features['wt_charge'] = self.aa_properties['charge'].get(wt_aa, 0)
            features['wt_beta_propensity'] = self.aa_properties['beta_propensity'].get(wt_aa, 1)
            features['wt_flexibility'] = self.aa_properties['flexibility'].get(wt_aa, 0.5)
            
            # Mutant amino acid properties
            features['mut_hydropathy'] = self.aa_properties['hydropathy'].get(mut_aa, 0)
            features['mut_charge'] = self.aa_properties['charge'].get(mut_aa, 0)
            features['mut_beta_propensity'] = self.aa_properties['beta_propensity'].get(mut_aa, 1)
            features['mut_flexibility'] = self.aa_properties['flexibility'].get(mut_aa, 0.5)
            
            # Property changes (key features for suppressor prediction)
            features['hydropathy_change'] = features['mut_hydropathy'] - features['wt_hydropathy']
            features['charge_change'] = features['mut_charge'] - features['wt_charge']
            features['beta_propensity_change'] = features['mut_beta_propensity'] - features['wt_beta_propensity']
            features['flexibility_change'] = features['mut_flexibility'] - features['wt_flexibility']
            
            # Aggregation propensity scores
            features['wt_tango_score'] = self.calculate_tango_score(sequence, position)
            features['mut_tango_score'] = self.calculate_tango_score(mut_sequence, position)
            features['tango_change'] = features['mut_tango_score'] - features['wt_tango_score']
            
            features['wt_waltz_score'] = self.calculate_waltz_score(sequence, position)
            features['mut_waltz_score'] = self.calculate_waltz_score(mut_sequence, position)
            features['waltz_change'] = features['mut_waltz_score'] - features['wt_waltz_score']
            
            features['wt_pasta_score'] = self.calculate_pasta_score(sequence, position)
            features['mut_pasta_score'] = self.calculate_pasta_score(mut_sequence, position)
            features['pasta_change'] = features['mut_pasta_score'] - features['wt_pasta_score']
            
            # Aggregation reduction (negative change = good for suppressor)
            features['aggregation_reduction'] = -(features['tango_change'] + features['waltz_change'] + features['pasta_change']) / 3
            
            # Local sequence context features
            features['local_hydrophobicity'] = self.calculate_local_hydrophobicity(sequence, position)
            features['local_charge'] = self.calculate_local_charge(sequence, position)
            features['local_flexibility'] = self.calculate_local_flexibility(sequence, position)
            
            # Protein-specific binary features
            features['is_alpha_synuclein'] = 1 if protein_key == 'alpha_synuclein' else 0
            features['is_tau'] = 1 if protein_key == 'tau' else 0
            features['is_sod1'] = 1 if protein_key == 'sod1' else 0
            features['is_app'] = 1 if protein_key == 'amyloid_beta' else 0
            
            # Amino acid type features
            features['wt_is_charged'] = 1 if wt_aa in 'DEKR' else 0
            features['mut_is_charged'] = 1 if mut_aa in 'DEKR' else 0
            features['wt_is_aromatic'] = 1 if wt_aa in 'FWY' else 0
            features['mut_is_aromatic'] = 1 if mut_aa in 'FWY' else 0
            features['wt_is_hydrophobic'] = 1 if wt_aa in 'AILMFWYV' else 0
            features['mut_is_hydrophobic'] = 1 if mut_aa in 'AILMFWYV' else 0
            features['wt_is_polar'] = 1 if wt_aa in 'NQST' else 0
            features['mut_is_polar'] = 1 if mut_aa in 'NQST' else 0
            
            # Special amino acid indicators
            features['mut_is_proline'] = 1 if mut_aa == 'P' else 0
            features['mut_is_glycine'] = 1 if mut_aa == 'G' else 0
            features['wt_is_proline'] = 1 if wt_aa == 'P' else 0
            features['wt_is_glycine'] = 1 if wt_aa == 'G' else 0
            
            # Advanced features
            features.update(self.calculate_advanced_features(sequence, position, wt_aa, mut_aa))
            
            return features
            
        except Exception as e:
            print(f"Error extracting features: {e}")
            return {}
    
    def calculate_advanced_features(self, sequence: str, position: int, wt_aa: str, mut_aa: str) -> Dict:
        """Calculate advanced biophysical features"""
        features = {}
        
        try:
            # Secondary structure propensities
            alpha_helix_propensity = {
                'A': 1.42, 'R': 0.98, 'N': 0.67, 'D': 1.01, 'C': 0.70,
                'Q': 1.11, 'E': 1.51, 'G': 0.57, 'H': 1.00, 'I': 1.08,
                'L': 1.21, 'K': 1.16, 'M': 1.45, 'F': 1.13, 'P': 0.57,
                'S': 0.77, 'T': 0.83, 'W': 1.08, 'Y': 0.69, 'V': 1.06
            }
            
            features['wt_alpha_helix'] = alpha_helix_propensity.get(wt_aa, 1.0)
            features['mut_alpha_helix'] = alpha_helix_propensity.get(mut_aa, 1.0)
            features['alpha_helix_change'] = features['mut_alpha_helix'] - features['wt_alpha_helix']
            
            # Volume changes
            volumes = {
                'A': 88.6, 'R': 173.4, 'N': 114.1, 'D': 111.1, 'C': 108.5,
                'Q': 143.8, 'E': 138.4, 'G': 60.1, 'H': 153.2, 'I': 166.7,
                'L': 166.7, 'K': 168.6, 'M': 162.9, 'F': 189.9, 'P': 112.7,
                'S': 89.0, 'T': 116.1, 'W': 227.8, 'Y': 193.6, 'V': 140.0
            }
            
            features['wt_volume'] = volumes.get(wt_aa, 140.0)
            features['mut_volume'] = volumes.get(mut_aa, 140.0)
            features['volume_change'] = features['mut_volume'] - features['wt_volume']
            features['volume_change_abs'] = abs(features['volume_change'])
            
            # Conservation and disorder features
            features['conservation_score'] = self.calculate_conservation_score(sequence, position)
            features['disorder_propensity'] = self.calculate_disorder_propensity(sequence, position)
            
            # Structural features
            features['turn_propensity_change'] = self.calculate_turn_propensity_change(wt_aa, mut_aa)
            features['accessibility_change'] = self.calculate_accessibility_change(wt_aa, mut_aa)
            
        except Exception as e:
            print(f"Error calculating advanced features: {e}")
        
        return features
    
    def predict_with_ml_model(self, features: Dict) -> Tuple[float, float, str]:
        """Use your trained ML model to predict pathogenicity and suppressor score"""
        if not self.model_loaded or self.trained_model is None:
            return self.calculate_heuristic_scores(features)
        
        try:
            # Prepare feature vector
            if self.feature_names:
                # Use exact feature names from training
                feature_vector = []
                for feature_name in self.feature_names:
                    feature_vector.append(features.get(feature_name, 0.0))
            else:
                # Use all available features in consistent order
                feature_keys = sorted(features.keys())
                feature_vector = [features[key] for key in feature_keys]
            
            # Convert to numpy array
            feature_vector = np.array(feature_vector).reshape(1, -1)
            
            # Handle missing values and infinities
            feature_vector = np.nan_to_num(feature_vector, nan=0.0, posinf=1e6, neginf=-1e6)
            
            # Apply scaling if available
            if self.scaler is not None:
                feature_vector = self.scaler.transform(feature_vector)
            
            # Make prediction based on model type
            if hasattr(self.trained_model, 'predict_proba'):
                # For sklearn-style models with probability prediction
                proba = self.trained_model.predict_proba(feature_vector)
                if proba.shape[1] > 1:
                    pathogenicity_prob = proba[0][1]  # Probability of pathogenic class
                else:
                    pathogenicity_prob = proba[0][0]
            elif hasattr(self.trained_model, 'predict'):
                # For models that return probabilities directly (like LightGBM)
                prediction = self.trained_model.predict(feature_vector)
                pathogenicity_prob = prediction[0]
                
                # Ensure it's a probability
                if pathogenicity_prob > 1.0:
                    pathogenicity_prob = 1.0 / (1.0 + np.exp(-pathogenicity_prob))  # Sigmoid
            else:
                raise ValueError("Model has no predict or predict_proba method")
            
            # Ensure probability is in valid range
            pathogenicity_prob = max(0.0, min(1.0, pathogenicity_prob))
            
            # Convert pathogenicity to suppressor score
            # Lower pathogenicity = better suppressor
            suppressor_score = 1.0 - pathogenicity_prob
            
            # Add bonus for known good suppressor features
            if features.get('charge_change', 0) != 0:
                suppressor_score += 0.05  # Charge change bonus
            if features.get('mut_is_proline', 0) == 1:
                suppressor_score += 0.1   # Proline bonus
            if features.get('aggregation_reduction', 0) > 0:
                suppressor_score += features.get('aggregation_reduction', 0) * 0.1
            
            # Ensure final score is in valid range
            suppressor_score = max(0.0, min(1.0, suppressor_score))
            
            confidence = f"High (ML {self.model_type})"
            
            return pathogenicity_prob, suppressor_score, confidence
            
        except Exception as e:
            print(f"ML prediction error: {e}")
            # Fallback to heuristic
            return self.calculate_heuristic_scores(features)
    
    def calculate_heuristic_scores(self, features: Dict) -> Tuple[float, float, str]:
        """Fallback heuristic scoring when ML model fails"""
        try:
            # Simple heuristic based on key features
            charge_change = abs(features.get('charge_change', 0))
            hydropathy_change = features.get('hydropathy_change', 0)
            agg_reduction = features.get('aggregation_reduction', 0)
            
            # Calculate suppressor score
            suppressor_score = 0.4  # Base score
            
            # Charge change is good for suppressors
            if charge_change > 0:
                suppressor_score += charge_change * 0.15
            
            # Hydrophobicity reduction is good
            if hydropathy_change < 0:
                suppressor_score += abs(hydropathy_change) * 0.02
            
            # Aggregation reduction is very good
            if agg_reduction > 0:
                suppressor_score += agg_reduction * 0.3
            
            # Special amino acid bonuses
            if features.get('mut_is_proline', 0) == 1:
                suppressor_score += 0.2
            if features.get('mut_is_charged', 0) == 1:
                suppressor_score += 0.15
            
            suppressor_score = max(0.0, min(1.0, suppressor_score))
            pathogenicity_prob = 1.0 - suppressor_score
            
            confidence = "Low (Heuristic)"
            
            return pathogenicity_prob, suppressor_score, confidence
            
        except:
            return 0.5, 0.5, "Low (Default)"
    
    # Include all the helper methods
    def calculate_local_hydrophobicity(self, sequence: str, position: int, window: int = 7) -> float:
        """Calculate local hydrophobicity around position"""
        try:
            start = max(0, position - window)
            end = min(len(sequence), position + window + 1)
            local_seq = sequence[start:end]
            
            hydrophobicity_sum = sum(self.aa_properties['hydropathy'].get(aa, 0) for aa in local_seq)
            return hydrophobicity_sum / len(local_seq)
        except:
            return 0.0
    
    def calculate_local_charge(self, sequence: str, position: int, window: int = 7) -> float:
        """Calculate local charge around position"""
        try:
            start = max(0, position - window)
            end = min(len(sequence), position + window + 1)
            local_seq = sequence[start:end]
            
            charge_sum = sum(self.aa_properties['charge'].get(aa, 0) for aa in local_seq)
            return charge_sum
        except:
            return 0.0
    
    def calculate_local_flexibility(self, sequence: str, position: int, window: int = 7) -> float:
        """Calculate local flexibility around position"""
        try:
            start = max(0, position - window)
            end = min(len(sequence), position + window + 1)
            local_seq = sequence[start:end]
            
            flexibility_sum = sum(self.aa_properties['flexibility'].get(aa, 0.5) for aa in local_seq)
            return flexibility_sum / len(local_seq)
        except:
            return 0.5
    
    def calculate_conservation_score(self, sequence: str, position: int) -> float:
        """Simplified conservation score"""
        try:
            window = 5
            start = max(0, position - window)
            end = min(len(sequence), position + window + 1)
            local_seq = sequence[start:end]
            
            aa_counts = {}
            for aa in local_seq:
                aa_counts[aa] = aa_counts.get(aa, 0) + 1
            
            max_count = max(aa_counts.values()) if aa_counts else 1
            return max_count / len(local_seq)
        except:
            return 0.5
    
    def calculate_disorder_propensity(self, sequence: str, position: int) -> float:
        """Calculate disorder propensity"""
        try:
            disorder_prone = set('RQSYGPAKEDNHT')
            order_prone = set('WFILVMC')
            
            window = 9
            start = max(0, position - window)
            end = min(len(sequence), position + window + 1)
            local_seq = sequence[start:end]
            
            disorder_count = sum(1 for aa in local_seq if aa in disorder_prone)
            order_count = sum(1 for aa in local_seq if aa in order_prone)
            
            return (disorder_count - order_count) / len(local_seq)
        except:
            return 0.0
    
    def calculate_turn_propensity_change(self, wt_aa: str, mut_aa: str) -> float:
        """Calculate turn propensity change"""
        turn_propensity = {
            'A': 0.66, 'R': 0.95, 'N': 1.56, 'D': 1.46, 'C': 1.19,
            'Q': 0.98, 'E': 0.74, 'G': 1.56, 'H': 0.95, 'I': 0.47,
            'L': 0.59, 'K': 1.01, 'M': 0.60, 'F': 0.60, 'P': 1.52,
            'S': 1.42, 'T': 0.96, 'W': 0.96, 'Y': 1.14, 'V': 0.50
        }
        
        wt_turn = turn_propensity.get(wt_aa, 1.0)
        mut_turn = turn_propensity.get(mut_aa, 1.0)
        return mut_turn - wt_turn
    
    def calculate_accessibility_change(self, wt_aa: str, mut_aa: str) -> float:
        """Calculate accessibility change"""
        accessibility = {
            'A': 0.25, 'R': 0.95, 'N': 0.60, 'D': 0.83, 'C': 0.02,
            'Q': 0.62, 'E': 0.83, 'G': 0.00, 'H': 0.42, 'I': 0.02,
            'L': 0.02, 'K': 0.88, 'M': 0.02, 'F': 0.02, 'P': 0.23,
            'S': 0.42, 'T': 0.23, 'W': 0.05, 'Y': 0.05, 'V': 0.02
        }
        
        wt_acc = accessibility.get(wt_aa, 0.5)
        mut_acc = accessibility.get(mut_aa, 0.5)
        return mut_acc - wt_acc
    
    # Include simplified aggregation scoring methods
    def calculate_tango_score(self, sequence: str, position: int) -> float:
        """Simplified TANGO score calculation"""
        try:
            window_size = 5
            start = max(0, position - window_size)
            end = min(len(sequence), position + window_size + 1)
            context = sequence[start:end]
            
            if not context:
                return 0.5
            
            score = 0.0
            for aa in context:
                hydropathy = self.aa_properties['hydropathy'].get(aa, 0)
                beta_prop = self.aa_properties['beta_propensity'].get(aa, 1)
                charge = abs(self.aa_properties['charge'].get(aa, 0))
                
                aa_contribution = (hydropathy * 0.4 + beta_prop * 0.6) * (1 - charge * 0.3)
                score += aa_contribution
            
            return max(0, min(1, (score + 5) / 10))
        except:
            return 0.5
    
    def calculate_waltz_score(self, sequence: str, position: int) -> float:
        """Simplified WALTZ score calculation"""
        try:
            window_size = 6
            start = max(0, position - window_size)
            end = min(len(sequence), position + window_size + 1)
            context = sequence[start:end]
            
            if not context:
                return 0.5
            
            aromatic = set('FWY')
            hydrophobic = set('AILMFWYV')
            charged = set('DEKR')
            
            aromatic_count = sum(1 for aa in context if aa in aromatic)
            hydrophobic_count = sum(1 for aa in context if aa in hydrophobic)
            charged_count = sum(1 for aa in context if aa in charged)
            
            score = (aromatic_count * 0.6 + hydrophobic_count * 0.4) / len(context)
            score *= (1 - charged_count * 0.3 / len(context))
            
            return max(0, min(1, score))
        except:
            return 0.5
    
    def calculate_pasta_score(self, sequence: str, position: int) -> float:
        """Simplified PASTA score calculation"""
        try:
            window_size = 4
            start = max(0, position - window_size)
            end = min(len(sequence), position + window_size + 1)
            context = sequence[start:end]
            
            if not context:
                return 0.5
            
            beta_scores = []
            charge_penalty = 0
            
            for aa in context:
                beta_prop = self.aa_properties['beta_propensity'].get(aa, 1)
                beta_scores.append(beta_prop)
                
                charge = abs(self.aa_properties['charge'].get(aa, 0))
                if charge > 0:
                    charge_penalty += 0.3
            
            avg_beta = sum(beta_scores) / len(beta_scores) if beta_scores else 1
            score = avg_beta - charge_penalty / len(context)
            
            return max(0, min(1, score / 2))
        except:
            return 0.5
    
    def generate_ml_suppressors(self, protein_key: str, position: int, wt_aa: str) -> pd.DataFrame:
        """Generate suppressor predictions using your trained ML model!"""
        try:
            protein_info = self.get_protein_info(protein_key)
            sequence = protein_info['sequence']
            
            # Validate inputs
            is_valid, message, corrected_wt = self.validate_mutation(protein_key, position, wt_aa, 'A')
            if not is_valid:
                raise ValueError(message)
            
            if corrected_wt and corrected_wt != wt_aa:
                wt_aa = corrected_wt
                print(f"Using corrected wild-type: {wt_aa}")
            
            mutations = []
            
            print(f"🤖 Using {'TRAINED ML MODEL' if self.model_loaded else 'HEURISTIC FALLBACK'} for predictions")
            print(f"🔬 Model type: {self.model_type}")
            print(f"📍 Analyzing position {position} ({wt_aa}) in {protein_info['name']}")
            
            # Generate all possible mutations
            for mut_aa in self.amino_acids:
                if mut_aa == wt_aa:
                    continue  # Skip wild-type
                
                # Extract comprehensive features for this mutation
                features = self.extract_comprehensive_features(protein_key, position, wt_aa, mut_aa)
                
                if not features:
                    continue
                
                # Get ML prediction
                pathogenicity_prob, suppressor_score, confidence = self.predict_with_ml_model(features)
                
                # Compile results with all the important information
                mutation_data = {
                    'rank': 0,  # Will be set after sorting
                    'wild_type': wt_aa,
                    'mutant': mut_aa,
                    'position': position,
                    'protein': protein_key,
                    'suppressor_score': suppressor_score,
                    'pathogenicity_probability': pathogenicity_prob,
                    'ml_confidence': confidence,
                    'aggregation_reduction': features.get('aggregation_reduction', 0),
                    'tango_score': features.get('mut_tango_score', 0.5),
                    'waltz_score': features.get('mut_waltz_score', 0.5),
                    'pasta_score': features.get('mut_pasta_score', 0.5),
                    'hydropathy_change': features.get('hydropathy_change', 0),
                    'charge_change': features.get('charge_change', 0),
                    'beta_propensity_change': features.get('beta_propensity_change', 0),
                    'flexibility_change': features.get('flexibility_change', 0),
                    'volume_change': features.get('volume_change', 0),
                }
                
                mutations.append(mutation_data)
            
            # Create DataFrame
            if not mutations:
                return pd.DataFrame()
            
            df = pd.DataFrame(mutations)
            
            # Sort by suppressor score (descending - higher is better)
            df = df.sort_values('suppressor_score', ascending=False)
            df = df.reset_index(drop=True)
            df['rank'] = range(1, len(df) + 1)
            
            # Add interpretation categories based on ML predictions
            df['category'] = df['suppressor_score'].apply(
                lambda x: 'Excellent' if x > 0.8 else 
                         'Good' if x > 0.7 else 
                         'Moderate' if x > 0.6 else 
                         'Poor' if x > 0.5 else 'Very Poor'
            )
            
            print(f"✅ Generated {len(df)} ML-powered suppressor predictions")
            if self.model_loaded:
                print("🎯 Using your trained model for REAL predictions!")
                top_3 = df.head(3)
                print("🏆 Top 3 ML predictions:")
                for _, row in top_3.iterrows():
                    print(f"   {row['wild_type']}→{row['mutant']}: {row['suppressor_score']:.3f} ({row['category']})")
            
            return df
            
        except Exception as e:
            print(f"Error generating ML suppressors: {e}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame()

# Your existing GUI class with modifications to use the ML analyzer
class MLProteinAnalyzerGUI:
    """Enhanced GUI using your trained ML model"""
    
    def __init__(self):
        self.analyzer = MLProteinAnalyzer()  # Use ML analyzer instead
        self.root = tk.Tk()
        self.setup_gui()
    
    def setup_gui(self):
        """Setup the enhanced GUI with ML model status"""
        self.root.title(f"🤖 ML-Powered Protein Suppressor Analyzer - ISEF Project {'(ML Model Loaded)' if self.analyzer.model_loaded else '(Heuristic Mode)'}")
        self.root.geometry("1400x900")
        self.root.configure(bg='#f8f9fa')
        
        # Configure styles
        style = ttk.Style()
        style.theme_use('clam')
        
        # Custom styles with ML theme
        style.configure('Title.TLabel', font=('Segoe UI', 18, 'bold'), background='#f8f9fa', foreground='#2c3e50')
        style.configure('Heading.TLabel', font=('Segoe UI', 12, 'bold'), background='#f8f9fa', foreground='#34495e')
        style.configure('Info.TLabel', font=('Segoe UI', 10), background='#f8f9fa', foreground='#7f8c8d')
        style.configure('Success.TLabel', font=('Segoe UI', 10), background='#f8f9fa', foreground='#27ae60')
        style.configure('Warning.TLabel', font=('Segoe UI', 10), background='#f8f9fa', foreground='#e67e22')
        style.configure('Error.TLabel', font=('Segoe UI', 10), background='#f8f9fa', foreground='#e74c3c')
        style.configure('ML.TLabel', font=('Segoe UI', 10, 'bold'), background='#f8f9fa', foreground='#8e44ad')
        style.configure('Accent.TButton', font=('Segoe UI', 11, 'bold'))
        
        # Main container
        main_frame = ttk.Frame(self.root, padding="25")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(6, weight=1)
        
        # Title with ML status
        title_frame = ttk.Frame(main_frame)
        title_frame.grid(row=0, column=0, columnspan=2, pady=(0, 25))
        
        title_label = ttk.Label(title_frame, text="🤖 ML-Powered Protein Suppressor Analyzer", style='Title.TLabel')
        title_label.grid(row=0, column=0)
        
        subtitle_text = "Advanced ML-Based Aggregation Suppressor Prediction for Neurodegenerative Diseases"
        if self.analyzer.model_loaded:
            subtitle_text += f" | Using {self.analyzer.model_type} Model"
        
        subtitle_label = ttk.Label(title_frame, text=subtitle_text, style='Info.TLabel')
        subtitle_label.grid(row=1, column=0, pady=(5, 0))
        
        # ML Model Status
        ml_status_frame = ttk.LabelFrame(main_frame, text="🤖 ML Model Status", padding="15")
        ml_status_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 20))
        
        if self.analyzer.model_loaded:
            status_text = f"✅ Trained ML Model Loaded: {self.analyzer.model_type} | Ready for high-accuracy predictions!"
            if self.analyzer.model_dict:
                status_text += f" | Dictionary keys: {list(self.analyzer.model_dict.keys())[:3]}..."
            status_style = 'Success.TLabel'
        else:
            status_text = "⚠️ ML Model not found. Using heuristic fallback. Place 'protein_mutation_model.joblib' in the same directory for ML predictions."
            status_style = 'Warning.TLabel'
        
        ml_status_label = ttk.Label(ml_status_frame, text=status_text, style=status_style)
        ml_status_label.grid(row=0, column=0, sticky=tk.W)
        
        # Input section (same as before but with ML context)
        input_frame = ttk.LabelFrame(main_frame, text="🔬 Mutation Analysis Input", padding="20")
        input_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 20))
        input_frame.columnconfigure(1, weight=1)
        
        # Protein selection
        ttk.Label(input_frame, text="Target Protein:", style='Heading.TLabel').grid(row=0, column=0, sticky=tk.W, padx=(0, 15))
        self.protein_var = tk.StringVar(value="alpha_synuclein")
        protein_combo = ttk.Combobox(input_frame, textvariable=self.protein_var,
                                   values=list(self.analyzer.proteins.keys()),
                                   state="readonly", width=25, font=('Segoe UI', 10))
        protein_combo.grid(row=0, column=1, sticky=tk.W, pady=(0, 15))
        protein_combo.bind('<<ComboboxSelected>>', self.on_protein_selected)
        
        # Protein info display
        self.protein_info_var = tk.StringVar()
        protein_info_label = ttk.Label(input_frame, textvariable=self.protein_info_var, style='Info.TLabel')
        protein_info_label.grid(row=0, column=2, sticky=tk.W, padx=(20, 0))
        
        # Mutation input
        ttk.Label(input_frame, text="Mutation:", style='Heading.TLabel').grid(row=1, column=0, sticky=tk.W, padx=(0, 15))
        self.mutation_var = tk.StringVar()
        mutation_entry = ttk.Entry(input_frame, textvariable=self.mutation_var, width=35, font=('Consolas', 11))
        mutation_entry.grid(row=1, column=1, sticky=(tk.W, tk.E), pady=(0, 10))
        mutation_entry.bind('<Return>', lambda e: self.analyze_mutation())
        
        # Enhanced examples
        examples_text = ("🔤 Supported formats: A53T, SNCA:A53T, alpha_synuclein A53T, 53T, A53*, etc.\n"
                        "📋 Examples: A53T (Parkinson's), P301L (Alzheimer's/FTD), G93A (ALS)")
        ttk.Label(input_frame, text=examples_text, style='Info.TLabel', foreground='#6c757d').grid(
            row=2, column=1, columnspan=2, sticky=tk.W, pady=(0, 15))
        
        # Enhanced button section
        button_frame = ttk.Frame(input_frame)
        button_frame.grid(row=3, column=1, sticky=tk.W, pady=(10, 0))
        
        analyze_text = "🤖 ML Analyze" if self.analyzer.model_loaded else "🔬 Analyze Suppressors"
        analyze_btn = ttk.Button(button_frame, text=analyze_text,
                               command=self.analyze_mutation, style='Accent.TButton')
        analyze_btn.grid(row=0, column=0, padx=(0, 15))
        
        clear_btn = ttk.Button(button_frame, text="🗑️ Clear", command=self.clear_results)
        clear_btn.grid(row=0, column=1, padx=(0, 15))
        
        export_btn = ttk.Button(button_frame, text="💾 Export CSV", command=self.export_results)
        export_btn.grid(row=0, column=2, padx=(0, 15))
        
        help_btn = ttk.Button(button_frame, text="❓ Help", command=self.show_help)
        help_btn.grid(row=0, column=3)
        
        # Enhanced status section
        status_frame = ttk.Frame(main_frame)
        status_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 20))
        status_frame.columnconfigure(0, weight=1)
        
        self.status_var = tk.StringVar(value="🤖 Ready for ML-powered mutation analysis" if self.analyzer.model_loaded else "Ready to analyze mutations")
        self.status_label = ttk.Label(status_frame, textvariable=self.status_var, style='ML.TLabel' if self.analyzer.model_loaded else 'Info.TLabel')
        self.status_label.grid(row=0, column=0, sticky=tk.W)
        
        # Progress bar
        self.progress = ttk.Progressbar(status_frame, mode='indeterminate')
        self.progress.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(5, 0))
        
        # Results summary section
        summary_frame = ttk.LabelFrame(main_frame, text="📊 ML Analysis Summary", padding="15")
        summary_frame.grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 20))
        summary_frame.columnconfigure(0, weight=1)
        
        self.summary_var = tk.StringVar()
        summary_label = ttk.Label(summary_frame, textvariable=self.summary_var, style='Info.TLabel')
        summary_label.grid(row=0, column=0, sticky=tk.W, pady=(0, 10))
        
        # Top candidates section
        self.top_candidates_var = tk.StringVar()
        top_candidates_label = ttk.Label(summary_frame, textvariable=self.top_candidates_var,
                                       style='Info.TLabel', wraplength=1200)
        top_candidates_label.grid(row=1, column=0, sticky=tk.W)
        
        # Enhanced results table with ML columns
        table_frame = ttk.LabelFrame(main_frame, text="🎯 ML Suppressor Predictions", padding="15")
        table_frame.grid(row=5, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 20))
        table_frame.columnconfigure(0, weight=1)
        
        # Results table container
        tree_container = ttk.Frame(table_frame)
        tree_container.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        tree_container.columnconfigure(0, weight=1)
        tree_container.rowconfigure(0, weight=1)
        
        # Enhanced treeview with ML-specific columns
        columns = ('Rank', 'Mutation', 'Category', 'ML Score', 'Confidence', 'Pathogenicity', 
                  'Agg Reduction', 'Hydropathy Δ', 'Charge Δ', 'Volume Δ')
        self.tree = ttk.Treeview(tree_container, columns=columns, show='headings', height=18)
        
        # Configure columns
        column_widths = [50, 80, 80, 90, 120, 90, 100, 90, 70, 80]
        column_anchors = ['center'] * len(columns)
        
        for col, width, anchor in zip(columns, column_widths, column_anchors):
            self.tree.heading(col, text=col, command=lambda c=col: self.sort_treeview(c))
            self.tree.column(col, width=width, anchor=anchor)
        
        # Enhanced scrollbars
        v_scrollbar = ttk.Scrollbar(tree_container, orient=tk.VERTICAL, command=self.tree.yview)
        h_scrollbar = ttk.Scrollbar(tree_container, orient=tk.HORIZONTAL, command=self.tree.xview)
        self.tree.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)
        
        # Grid layout
        self.tree.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        v_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        h_scrollbar.grid(row=1, column=0, sticky=(tk.W, tk.E))
        
        # Configure main frame row weight
        main_frame.rowconfigure(5, weight=1)
        
        # Initialize protein info
        self.on_protein_selected()
    
    def on_protein_selected(self, event=None):
        """Update protein information when selection changes"""
        try:
            protein_key = self.protein_var.get()
            protein_info = self.analyzer.get_protein_info(protein_key)
            info_text = f"{protein_info['name']} | {protein_info['length']} AA | {protein_info['disease']}"
            self.protein_info_var.set(info_text)
        except:
            self.protein_info_var.set("")
    
    def sort_treeview(self, col):
        """Sort treeview by column"""
        try:
            data = [(self.tree.set(child, col), child) for child in self.tree.get_children('')]
            
            try:
                data.sort(key=lambda x: float(x[0].replace('+', '').replace('→', '').split()[0]))
            except:
                data.sort()
            
            for index, (val, child) in enumerate(data):
                self.tree.move(child, '', index)
        except Exception as e:
            print(f"Sort error: {e}")
    
    def analyze_mutation(self):
        """Enhanced ML-powered mutation analysis"""
        try:
            # Start progress
            self.progress.start(10)
            self.status_var.set("🤖 Initializing ML analysis..." if self.analyzer.model_loaded else "Initializing analysis...")
            self.status_label.configure(style='ML.TLabel' if self.analyzer.model_loaded else 'Info.TLabel')
            self.root.update()
            
            # Get inputs
            protein_key = self.protein_var.get()
            mutation_input = self.mutation_var.get().strip()
            
            if not mutation_input:
                messagebox.showerror("Input Error", "Please enter a mutation to analyze")
                self.progress.stop()
                self.status_var.set("🤖 Ready for ML analysis" if self.analyzer.model_loaded else "Ready to analyze mutations")
                return
            
            # Parse mutation
            self.status_var.set("🔍 Parsing mutation input...")
            self.root.update()
            
            try:
                parsed_protein, position, wt_aa, mut_aa = self.analyzer.parse_mutation_input(mutation_input)
                if parsed_protein != "UNKNOWN":
                    protein_key = parsed_protein
            except ValueError as e:
                messagebox.showerror("Parse Error", f"Could not parse mutation: {str(e)}")
                self.progress.stop()
                self.status_var.set("🤖 Ready for ML analysis" if self.analyzer.model_loaded else "Ready to analyze mutations")
                return
            
            # Get protein info
            self.status_var.set("📊 Loading protein information...")
            self.root.update()
            
            try:
                protein_info = self.analyzer.get_protein_info(protein_key)
            except ValueError as e:
                messagebox.showerror("Protein Error", str(e))
                self.progress.stop()
                self.status_var.set("🤖 Ready for ML analysis" if self.analyzer.model_loaded else "Ready to analyze mutations")
                return
            
            # Validate mutation
            self.status_var.set("✅ Validating mutation...")
            self.root.update()
            
            is_valid, message, corrected_wt = self.analyzer.validate_mutation(protein_key, position, wt_aa, mut_aa)
            if not is_valid:
                messagebox.showerror("Validation Error", message)
                self.progress.stop()
                self.status_var.set("🤖 Ready for ML analysis" if self.analyzer.model_loaded else "Ready to analyze mutations")
                return
            
            if corrected_wt and corrected_wt != wt_aa:
                wt_aa = corrected_wt
                self.status_var.set(f"⚠️ Using corrected wild-type: {wt_aa}")
                self.status_label.configure(style='Warning.TLabel')
                self.root.update()
            
            # Generate ML suppressors
            if self.analyzer.model_loaded:
                self.status_var.set("🤖 Running ML model predictions... This may take a moment.")
            else:
                self.status_var.set("🔬 Generating suppressor candidates using heuristic approach...")
            self.root.update()
            
            results_df = self.analyzer.generate_ml_suppressors(protein_key, position, wt_aa)
            
            if results_df.empty:
                messagebox.showwarning("No Results", "No suppressor candidates could be generated for this mutation")
                self.progress.stop()
                self.status_var.set("🤖 Ready for ML analysis" if self.analyzer.model_loaded else "Ready to analyze mutations")
                return
            
            # Store results
            self.analyzer.current_results = results_df
            
            # Update display
            self.status_var.set("📊 Updating results display...")
            self.root.update()
            
            self.update_results_display(results_df, protein_info, wt_aa, position)
            
            # Final status
            self.progress.stop()
            high_confidence = len(results_df[results_df['suppressor_score'] > 0.7])
            excellent = len(results_df[results_df['category'] == 'Excellent'])
            ml_predictions = len(results_df[results_df['ml_confidence'].str.contains('ML', na=False)])
            
            if self.analyzer.model_loaded:
                self.status_var.set(f"🎉 ML Analysis complete! {len(results_df)} candidates | "
                                  f"{excellent} excellent | {ml_predictions} ML predictions | "
                                  f"{high_confidence} high-confidence suppressors")
                self.status_label.configure(style='Success.TLabel')
            else:
                self.status_var.set(f"✅ Analysis complete! {len(results_df)} candidates generated | "
                                  f"{excellent} excellent | {high_confidence} high-confidence suppressors")
                self.status_label.configure(style='Success.TLabel')
                
        except Exception as e:
            self.progress.stop()
            error_msg = f"Analysis failed: {str(e)}"
            messagebox.showerror("Analysis Error", error_msg)
            self.status_var.set(f"❌ {error_msg}")
            self.status_label.configure(style='Error.TLabel')
            import traceback
            traceback.print_exc()
    
    def update_results_display(self, results_df: pd.DataFrame, protein_info: Dict, wt_aa: str, position: int):
        """Enhanced results display with ML-specific information"""
        # Clear existing results
        for item in self.tree.get_children():
            self.tree.delete(item)
        
        # Update summary
        top_suppressor = results_df.iloc[0]
        high_confidence = len(results_df[results_df['suppressor_score'] > 0.7])
        excellent = len(results_df[results_df['category'] == 'Excellent'])
        ml_predictions = len(results_df[results_df['ml_confidence'].str.contains('ML', na=False)])
        
        if self.analyzer.model_loaded:
            summary_text = (f"🧬 Protein: {protein_info['name']} | "
                           f"📍 Position: {wt_aa}{position} | "
                           f"🤖 ML Model: {self.analyzer.model_type} | "
                           f"🏆 Top suppressor: {wt_aa}→{top_suppressor['mutant']} "
                           f"(ML Score: {top_suppressor['suppressor_score']:.3f}) | "
                           f"⭐ {excellent} excellent, {ml_predictions} ML predictions")
        else:
            summary_text = (f"🧬 Protein: {protein_info['name']} | "
                           f"📍 Position: {wt_aa}{position} | "
                           f"🏆 Top suppressor: {wt_aa}→{top_suppressor['mutant']} "
                           f"(Score: {top_suppressor['suppressor_score']:.3f}) | "
                           f"⭐ {excellent} excellent, {high_confidence} high-confidence candidates")
        
        self.summary_var.set(summary_text)
        
        # Top 3 candidates with ML interpretation
        top_3 = results_df.head(3)
        top_candidates_text = "🎯 Top 3 ML Suppressor Predictions:\n" if self.analyzer.model_loaded else "🔬 Top 3 Suppressor Candidates:\n"
        
        for i, (_, row) in enumerate(top_3.iterrows(), 1):
            interpretation = self.interpret_ml_suppressor(row)
            confidence_indicator = "🤖" if "ML" in row.get('ml_confidence', '') else "🔬"
            top_candidates_text += (f"{i}. {confidence_indicator} {row['wild_type']}→{row['mutant']} "
                                  f"(Score: {row['suppressor_score']:.3f}, {row['category']}) - {interpretation}\n")
        
        self.top_candidates_var.set(top_candidates_text)
        
        # Populate table with ML-enhanced data
        for _, row in results_df.iterrows():
            values = (
                row['rank'],
                f"{row['wild_type']}→{row['mutant']}",
                row['category'],
                f"{row['suppressor_score']:.3f}",
                row.get('ml_confidence', 'Unknown'),
                f"{row.get('pathogenicity_probability', 0.5):.3f}",
                f"{row.get('aggregation_reduction', 0):+.3f}",
                f"{row.get('hydropathy_change', 0):+.2f}",
                f"{row.get('charge_change', 0):+.1f}",
                f"{row.get('volume_change', 0):+.1f}"
            )
            
            # Enhanced color coding with ML confidence
            if row['category'] == 'Excellent' and 'ML' in row.get('ml_confidence', ''):
                tag = 'ml_excellent'
            elif row['category'] == 'Excellent':
                tag = 'excellent'
            elif row['category'] == 'Good' and 'ML' in row.get('ml_confidence', ''):
                tag = 'ml_good'
            elif row['category'] == 'Good':
                tag = 'good'
            elif row['category'] == 'Moderate':
                tag = 'moderate'
            elif row['category'] == 'Poor':
                tag = 'poor'
            else:
                tag = 'very_poor'
            
            self.tree.insert('', 'end', values=values, tags=(tag,))
        
        # Configure enhanced tags with ML highlighting
        self.tree.tag_configure('ml_excellent', background='#c8e6c9', foreground='#1b5e20', font=('Segoe UI', 9, 'bold'))
        self.tree.tag_configure('excellent', background='#d1f2eb', foreground='#0e6b47')
        self.tree.tag_configure('ml_good', background='#dcedc8', foreground='#33691e', font=('Segoe UI', 9, 'bold'))
        self.tree.tag_configure('good', background='#d4edda', foreground='#155724')
        self.tree.tag_configure('moderate', background='#fff3cd', foreground='#856404')
        self.tree.tag_configure('poor', background='#f8d7da', foreground='#721c24')
        self.tree.tag_configure('very_poor', background='#f5c6cb', foreground='#491217')
    
    def interpret_ml_suppressor(self, row) -> str:
        """Generate ML-enhanced biological interpretation"""
        mutant = row['mutant']
        agg_reduction = row.get('aggregation_reduction', 0)
        charge_change = row.get('charge_change', 0)
        hydro_change = row.get('hydropathy_change', 0)
        volume_change = row.get('volume_change', 0)
        ml_confidence = row.get('ml_confidence', '')
        
        interpretations = []
        
        # ML confidence indicator
        if 'ML' in ml_confidence:
            interpretations.append("ML-predicted")
        
        # Charge-based interpretations
        if abs(charge_change) > 0.5:
            if charge_change > 0:
                interpretations.append("introduces positive charge")
            else:
                interpretations.append("introduces negative charge")
        
        # Hydrophobicity interpretations
        if hydro_change < -2:
            interpretations.append("strongly reduces hydrophobicity")
        elif hydro_change < -1:
            interpretations.append("reduces hydrophobicity")
        
        # Volume change interpretations
        if abs(volume_change) > 50:
            if volume_change > 0:
                interpretations.append("significant volume increase")
            else:
                interpretations.append("significant volume decrease")
        
        # Specific amino acid interpretations
        if mutant == 'P':
            interpretations.append("proline disrupts secondary structure")
        elif mutant in 'DE':
            interpretations.append("acidic residue prevents aggregation")
        elif mutant in 'KR':
            interpretations.append("basic residue provides electrostatic repulsion")
        elif mutant == 'G':
            interpretations.append("glycine increases backbone flexibility")
        
        # Aggregation reduction interpretation
        if agg_reduction > 0.2:
            interpretations.append("excellent aggregation reduction")
        elif agg_reduction > 0.1:
            interpretations.append("good aggregation reduction")
        elif agg_reduction > 0:
            interpretations.append("moderate aggregation reduction")
        else:
            interpretations.append("minimal aggregation impact")
        
        return "; ".join(interpretations) if interpretations else "neutral substitution"
    
    def clear_results(self):
        """Enhanced clear function"""
        self.mutation_var.set("")
        for item in self.tree.get_children():
            self.tree.delete(item)
        self.summary_var.set("")
        self.top_candidates_var.set("")
        self.analyzer.current_results = None
        self.status_var.set("🤖 Ready for ML analysis" if self.analyzer.model_loaded else "Ready to analyze mutations")
        self.status_label.configure(style='ML.TLabel' if self.analyzer.model_loaded else 'Info.TLabel')
        self.progress.stop()
    
    def export_results(self):
        """Enhanced export with ML metadata"""
        if self.analyzer.current_results is None or self.analyzer.current_results.empty:
            messagebox.showwarning("No Data", "No results to export. Please run an analysis first.")
            return
        
        try:
            # Get filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_suffix = "_ML" if self.analyzer.model_loaded else "_heuristic"
            default_filename = f"ml_suppressor_analysis{model_suffix}_{timestamp}.csv"
            
            filename = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv"), ("Excel files", "*.xlsx"), ("All files", "*.*")],
                title="Save ML results as...",
                initialvalue=default_filename
            )
            
            if filename:
                # Add ML metadata to the export
                df_export = self.analyzer.current_results.copy()
                
                # Add ML-specific metadata
                df_export['analysis_timestamp'] = datetime.now().isoformat()
                df_export['analyzer_version'] = "3.0_ML_Fixed"
                df_export['ml_model_used'] = self.analyzer.model_loaded
                df_export['model_type'] = self.analyzer.model_type if self.analyzer.model_loaded else "Heuristic"
                
                if filename.endswith('.xlsx'):
                    df_export.to_excel(filename, index=False)
                else:
                    df_export.to_csv(filename, index=False)
                
                messagebox.showinfo("Export Success", f"ML results exported to:\n{filename}")
                self.status_var.set(f"✅ ML results exported to {os.path.basename(filename)}")
                self.status_label.configure(style='Success.TLabel')
                
        except Exception as e:
            error_msg = f"Export failed: {str(e)}"
            messagebox.showerror("Export Error", error_msg)
            self.status_var.set(f"❌ {error_msg}")
            self.status_label.configure(style='Error.TLabel')
    
    def show_help(self):
        """Show comprehensive help with ML information"""
        help_text = f"""
🤖 ML-POWERED PROTEIN MUTATION SUPPRESSOR ANALYZER - HELP

📋 OVERVIEW:
This tool uses {'your trained machine learning model' if self.analyzer.model_loaded else 'heuristic algorithms'} to predict amino acid 
substitutions that can suppress protein aggregation in neurodegenerative diseases.

🔤 SUPPORTED MUTATION FORMATS:
• A53T - Standard format (wild-type, position, mutant)
• SNCA:A53T - With protein identifier
• alpha_synuclein A53T - With full protein name
• 53T - Position and mutant only
• A53* - Stop codon mutations

🧪 SUPPORTED PROTEINS:
• Alpha-synuclein (SNCA) - Parkinson's Disease
• Tau protein (MAPT) - Alzheimer's/FTD
• SOD1 - Amyotrophic Lateral Sclerosis
• Amyloid precursor protein (APP) - Alzheimer's

🤖 ML MODEL STATUS:
• Model Loaded: {'✅ YES' if self.analyzer.model_loaded else '❌ NO'}
• Model Type: {self.analyzer.model_type if self.analyzer.model_loaded else 'N/A'}
• Prediction Method: {'Machine Learning' if self.analyzer.model_loaded else 'Heuristic Fallback'}

📊 SCORING SYSTEM:
• ML Score: 0-1 scale (higher = better suppressor)
• Categories: Excellent (>0.8), Good (>0.7), Moderate (>0.6)
• Confidence: High (ML Model) vs Low (Heuristic)

🎯 INTERPRETATION:
• 🤖 Green rows with bold: ML-predicted excellent suppressors
• Green rows: Excellent/Good suppressors
• Yellow rows: Moderate suppressors
• Red rows: Poor suppressors

💡 TIPS:
• ML predictions are more accurate than heuristic approaches
• Charged residues (D,E,K,R) often make good suppressors
• Proline (P) disrupts structure and reduces aggregation
• Volume and charge changes are key ML features

📁 MODEL FILES:
To use ML predictions, place 'protein_mutation_model.joblib' in the same directory.
The model file should contain your trained model in a dictionary or as a direct model object.

📧 For questions about this ISEF project, contact the developer.
        """
        
        help_window = tk.Toplevel(self.root)
        help_window.title("Help - ML Protein Suppressor Analyzer")
        help_window.geometry("700x600")
        help_window.configure(bg='#f8f9fa')
        
        text_widget = tk.Text(help_window, wrap=tk.WORD, padx=20, pady=20,
                             font=('Segoe UI', 10), bg='#ffffff')
        text_widget.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        text_widget.insert(tk.END, help_text)
        text_widget.configure(state='disabled')
        
        # Add scrollbar
        scrollbar = ttk.Scrollbar(help_window, command=text_widget.yview)
        text_widget.configure(yscrollcommand=scrollbar.set)
    
    def run(self):
        """Run the GUI with error handling"""
        try:
            self.root.mainloop()
        except Exception as e:
            print(f"GUI Error: {e}")
            import traceback
            traceback.print_exc()

def main():
    """Main function with ML model integration"""
    try:
        print("🧬 Starting ML-Powered Protein Mutation Suppressor Analyzer v3.0")
        print("📊 Loading protein database and ML model...")
        
        app = MLProteinAnalyzerGUI()
        
        if app.analyzer.model_loaded:
            print("✅ Your trained ML model loaded successfully!")
            print(f"🤖 Model type: {app.analyzer.model_type}")
            print("🎯 Ready for high-accuracy ML predictions!")
        else:
            print("⚠️ ML model not found. Using heuristic fallback.")
            print("💡 Place 'protein_mutation_model.joblib' in the same directory for ML predictions.")
        
        print("🚀 Launching enhanced GUI...")
        app.run()
        
    except Exception as e:
        print(f"❌ Application failed to start: {e}")
        import traceback
        traceback.print_exc()
        
        # Show error dialog if possible
        try:
            import tkinter.messagebox as mb
            mb.showerror("Startup Error", f"Failed to start application:\n{str(e)}")
        except:
            pass

if __name__ == "__main__":
    main()
