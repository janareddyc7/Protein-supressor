#!/usr/bin/env python3
import os
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
import logging
from collections import Counter
import hashlib

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnhancedMutationDataProcessor:
    def __init__(self):
        # Standard amino acids (20 total)
        self.AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
        
        # Kyte-Doolittle hydropathy scale (REAL DATA)
        self.kyte_doolittle = {
            'A': 1.8, 'C': 2.5, 'D': -3.5, 'E': -3.5, 'F': 2.8,
            'G': -0.4, 'H': -3.2, 'I': 4.5, 'K': -3.9, 'L': 3.8,
            'M': 1.9, 'N': -3.5, 'P': -1.6, 'Q': -3.5, 'R': -4.5,
            'S': -0.8, 'T': -0.7, 'V': 4.2, 'W': -0.9, 'Y': -1.3
        }
        
        # Amino acid charges at physiological pH (REAL DATA)
        self.aa_charges = {
            'D': -1, 'E': -1, 'K': 1, 'R': 1, 'H': 1,
            'A': 0, 'C': 0, 'F': 0, 'G': 0, 'I': 0,
            'L': 0, 'M': 0, 'N': 0, 'P': 0, 'Q': 0,
            'S': 0, 'T': 0, 'V': 0, 'W': 0, 'Y': 0
        }
        
        # Amino acid volumes in Å³ from Zamyatnin (1972) (REAL DATA)
        self.aa_volumes = {
            'A': 88.6, 'C': 108.5, 'D': 111.1, 'E': 138.4, 'F': 189.9,
            'G': 60.1, 'H': 153.2, 'I': 166.7, 'K': 168.6, 'L': 166.7,
            'M': 162.9, 'N': 114.1, 'P': 112.7, 'Q': 143.8, 'R': 173.4,
            'S': 89.0, 'T': 116.1, 'V': 140.0, 'W': 227.8, 'Y': 193.6
        }
        
        # BLOSUM62 substitution matrix (REAL DATA)
        self.blosum62 = self._init_blosum62()
        
        # Amino acid properties for intelligent labeling (REAL DATA)
        self.aa_properties = {
            'A': {'hydrophobicity': 1.8, 'charge': 0, 'polarity': 'nonpolar', 'size': 'small', 'aromatic': False, 'aggregation_prone': False},
            'C': {'hydrophobicity': 2.5, 'charge': 0, 'polarity': 'polar', 'size': 'small', 'aromatic': False, 'aggregation_prone': False},
            'D': {'hydrophobicity': -3.5, 'charge': -1, 'polarity': 'charged', 'size': 'small', 'aromatic': False, 'aggregation_prone': False},
            'E': {'hydrophobicity': -3.5, 'charge': -1, 'polarity': 'charged', 'size': 'medium', 'aromatic': False, 'aggregation_prone': False},
            'F': {'hydrophobicity': 2.8, 'charge': 0, 'polarity': 'nonpolar', 'size': 'large', 'aromatic': True, 'aggregation_prone': True},
            'G': {'hydrophobicity': -0.4, 'charge': 0, 'polarity': 'nonpolar', 'size': 'small', 'aromatic': False, 'aggregation_prone': False},
            'H': {'hydrophobicity': -3.2, 'charge': 1, 'polarity': 'charged', 'size': 'medium', 'aromatic': True, 'aggregation_prone': False},
            'I': {'hydrophobicity': 4.5, 'charge': 0, 'polarity': 'nonpolar', 'size': 'large', 'aromatic': False, 'aggregation_prone': True},
            'K': {'hydrophobicity': -3.9, 'charge': 1, 'polarity': 'charged', 'size': 'large', 'aromatic': False, 'aggregation_prone': False},
            'L': {'hydrophobicity': 3.8, 'charge': 0, 'polarity': 'nonpolar', 'size': 'large', 'aromatic': False, 'aggregation_prone': True},
            'M': {'hydrophobicity': 1.9, 'charge': 0, 'polarity': 'nonpolar', 'size': 'large', 'aromatic': False, 'aggregation_prone': False},
            'N': {'hydrophobicity': -3.5, 'charge': 0, 'polarity': 'polar', 'size': 'medium', 'aromatic': False, 'aggregation_prone': False},
            'P': {'hydrophobicity': -1.6, 'charge': 0, 'polarity': 'nonpolar', 'size': 'small', 'aromatic': False, 'aggregation_prone': False},
            'Q': {'hydrophobicity': -3.5, 'charge': 0, 'polarity': 'polar', 'size': 'medium', 'aromatic': False, 'aggregation_prone': False},
            'R': {'hydrophobicity': -4.5, 'charge': 1, 'polarity': 'charged', 'size': 'large', 'aromatic': False, 'aggregation_prone': False},
            'S': {'hydrophobicity': -0.8, 'charge': 0, 'polarity': 'polar', 'size': 'small', 'aromatic': False, 'aggregation_prone': False},
            'T': {'hydrophobicity': -0.7, 'charge': 0, 'polarity': 'polar', 'size': 'small', 'aromatic': False, 'aggregation_prone': False},
            'V': {'hydrophobicity': 4.2, 'charge': 0, 'polarity': 'nonpolar', 'size': 'medium', 'aromatic': False, 'aggregation_prone': True},
            'W': {'hydrophobicity': -0.9, 'charge': 0, 'polarity': 'nonpolar', 'size': 'large', 'aromatic': True, 'aggregation_prone': True},
            'Y': {'hydrophobicity': -1.3, 'charge': 0, 'polarity': 'polar', 'size': 'large', 'aromatic': True, 'aggregation_prone': True}
        }
        
        # Known aggregation-prone proteins/regions with hotspots (REAL DATA from literature)
        self.aggregation_prone_proteins = {
            'alpha_synuclein': {
                'regions': [(61, 95)],  # NAC region
                'high_risk_positions': [1, 53, 80],
                'hotspot_regions': [(12, 16), (35, 42), (61, 95)]  # Literature-based hotspots
            },
            'amyloid_beta': {
                'regions': [(16, 23), (29, 40)], 
                'high_risk_positions': [16, 20, 22, 23],
                'hotspot_regions': [(16, 23), (29, 40)]
            },
            'tau': {
                'regions': [(306, 378)],  # Microtubule binding domain
                'high_risk_positions': [306, 311, 317],
                'hotspot_regions': [(306, 320), (350, 370)]
            },
        }
    
    def _init_blosum62(self):
        """Initialize BLOSUM62 substitution matrix (REAL DATA)"""
        amino_acids = "ARNDCQEGHILKMFPSTWYV"
        blosum_matrix = [
            [ 4, -1, -2, -2,  0, -1, -1,  0, -2, -1, -1, -1, -1, -2, -1,  1,  0, -3, -2,  0],
            [-1,  5,  0, -2, -3,  1,  0, -2,  0, -3, -2,  2, -1, -3, -2, -1, -1, -3, -2, -3],
            [-2,  0,  6,  1, -3,  0,  0,  0,  1, -3, -3,  0, -2, -3, -2,  1,  0, -4, -2, -3],
            [-2, -2,  1,  6, -3,  0,  2, -1, -1, -3, -4, -1, -3, -3, -1,  0, -1, -4, -3, -3],
            [ 0, -3, -3, -3,  9, -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -1, -1, -2, -2, -1],
            [-1,  1,  0,  0, -3,  5,  2, -2,  0, -3, -2,  1,  0, -3, -1,  0, -1, -2, -1, -2],
            [-1,  0,  0,  2, -4,  2,  5, -2,  0, -3, -3,  1, -2, -3, -1,  0, -1, -3, -2, -2],
            [ 0, -2,  0, -1, -3, -2, -2,  6, -2, -4, -4, -2, -3, -3, -2,  0, -2, -2, -3, -3],
            [-2,  0,  1, -1, -3,  0,  0, -2,  8, -3, -3, -1, -2, -1, -2, -1, -2, -2,  2, -3],
            [-1, -3, -3, -3, -1, -3, -3, -4, -3,  4,  2, -3,  1,  0, -3, -2, -1, -3, -1,  3],
            [-1, -2, -3, -4, -1, -2, -3, -4, -3,  2,  4, -2,  2,  0, -3, -2, -1, -2, -1,  1],
            [-1,  2,  0, -1, -3,  1,  1, -2, -1, -3, -2,  5, -1, -3, -1,  0, -1, -3, -2, -2],
            [-1, -1, -2, -3, -1,  0, -2, -3, -2,  1,  2, -1,  5,  0, -2, -1, -1, -1, -1,  1],
            [-2, -3, -3, -3, -2, -3, -3, -3, -1,  0,  0, -3,  0,  6, -4, -2, -2,  1,  3, -1],
            [-1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4,  7, -1, -1, -4, -3, -2],
            [ 1, -1,  1,  0, -1,  0,  0,  0, -1, -2, -2,  0, -1, -2, -1,  4,  1, -3, -2, -2],
            [ 0, -1,  0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1,  1,  5, -2, -2,  0],
            [-3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1,  1, -4, -3, -2, 11,  2, -3],
            [-2, -2, -2, -3, -2, -1, -2, -3,  2, -1, -1, -2, -1,  3, -3, -2, -2,  2,  7, -1],
            [ 0, -3, -3, -3, -1, -2, -2, -3, -3,  3,  1, -2,  1, -1, -2, -2,  0, -3, -1,  4]
        ]
        
        blosum_dict = {}
        for i, aa1 in enumerate(amino_acids):
            for j, aa2 in enumerate(amino_acids):
                blosum_dict[(aa1, aa2)] = blosum_matrix[i][j]
        
        return blosum_dict
    
    def generate_all_mutants(self):
        """Generate all single-point mutants with REAL features only"""
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        
        INPUT_JSON = os.path.join(project_root, 'data', 'proteins', 'all_sequences.json')
        OUTPUT_DIR = os.path.join(project_root, 'data', 'features')
        
        logger.info(f"Looking for input: {INPUT_JSON}")
        logger.info(f"Will save to: {OUTPUT_DIR}")
        
        # Check if input file exists
        if not os.path.exists(INPUT_JSON):
            logger.error(f"Input file not found: {INPUT_JSON}")
            logger.error("Please run data collection first!")
            return None
        
        # Create output directory if it doesn't exist
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        # Load protein sequences
        try:
            with open(INPUT_JSON, 'r') as f:
                proteins = json.load(f)
            logger.info(f"Loaded {len(proteins)} proteins")
        except Exception as e:
            logger.error(f"Error loading JSON: {e}")
            return None
        
        # Generate mutants for each protein separately
        for protein_name, protein_info in proteins.items():
            logger.info(f"Processing {protein_name}...")
            
            # Generate mutants
            rows = self._generate_mutant_rows_single_protein(protein_name, protein_info)
            
            # Create DataFrame and clean
            df = pd.DataFrame(rows)
            df_clean = self.clean_data(df)
            
            # Add REAL features only
            df_with_features = self.add_real_features(df_clean, protein_name)
            
            # Add intelligent labels
            df_labeled = self.add_intelligent_labels(df_with_features, protein_name)
            
            # Save individual protein dataset
            output_file = os.path.join(OUTPUT_DIR, f"{protein_name}_mutants_complete.csv")
            try:
                df_labeled.to_csv(output_file, index=False)
                logger.info(f"✅ {protein_name}: {len(df_labeled)} mutations saved to {output_file}")
                logger.info(f"File size: {os.path.getsize(output_file) / (1024*1024):.2f} MB")
                
                # Show summary for this protein
                self._print_protein_summary(df_labeled, protein_name)
                
            except Exception as e:
                logger.error(f"Error saving {protein_name}: {e}")
                continue
        
        logger.info("🎉 All proteins processed successfully!")
        logger.info("REAL FEATURES INCLUDED:")
        logger.info("✅ hydropathy_change (Kyte-Doolittle scale)")
        logger.info("✅ blosum_score (BLOSUM62 matrix)")
        logger.info("✅ charge_change (physiological pH)")
        logger.info("✅ size_change (amino acid volumes)")
        logger.info("✅ is_hotspot_region (literature-based)")
        logger.info("\nFEATURES NOT INCLUDED (would be fake/mock):")
        logger.info("❌ disorder_change (requires IUPred2A server)")
        logger.info("❌ conservation (requires MSA analysis)")
        logger.info("❌ AlphaFold_confidence (requires AlphaFold database)")
        logger.info("❌ is_disordered_region (requires IUPred2A)")
        
        return True
    
    def _generate_mutant_rows_single_protein(self, protein_name, protein_info):
        """Generate all mutation combinations for a single protein"""
        sequence = protein_info['sequence']
        length = protein_info['length']
        
        expected_mutants = length * (len(self.AMINO_ACIDS) - 1)
        logger.info(f"Generating {expected_mutants} mutants for {protein_name}")
        
        rows = []
        
        with tqdm(total=expected_mutants, desc=f"Generating {protein_name} mutants") as pbar:
            for position in range(length):
                wild_type_aa = sequence[position]
                
                for mutant_aa in self.AMINO_ACIDS:
                    if mutant_aa == wild_type_aa:
                        continue  # Skip wild-type (no mutation)
                    
                    # Create mutant sequence
                    mutant_sequence = sequence[:position] + mutant_aa + sequence[position+1:]
                    
                    # Create unique mutation ID
                    mutation_id = f"{protein_name}_{position+1}_{wild_type_aa}_{mutant_aa}"
                    
                    rows.append({
                        'mutation_id': mutation_id,
                        'protein': protein_name,
                        'position': position + 1,  # 1-based indexing
                        'wt_aa': wild_type_aa,
                        'mut_aa': mutant_aa,
                        'wt_sequence': sequence,
                        'mut_sequence': mutant_sequence,
                        'protein_length': length
                    })
                    
                    pbar.update(1)
        
        return rows
    
    def add_real_features(self, df, protein_name):
        """Add REAL computable features only"""
        logger.info(f"Adding REAL features for {protein_name}...")
        
        df_features = df.copy()
        
        # Initialize feature columns
        df_features['hydropathy_change'] = 0.0
        df_features['blosum_score'] = 0
        df_features['charge_change'] = 0
        df_features['size_change'] = 0.0
        df_features['is_hotspot_region'] = 0
        
        for idx, row in tqdm(df_features.iterrows(), total=len(df_features), desc="Computing real features"):
            wt_aa = row['wt_aa']
            mut_aa = row['mut_aa']
            position = row['position']
            
            # 1. Hydropathy change (Kyte-Doolittle scale)
            wt_hydropathy = self.kyte_doolittle.get(wt_aa, 0)
            mut_hydropathy = self.kyte_doolittle.get(mut_aa, 0)
            df_features.at[idx, 'hydropathy_change'] = mut_hydropathy - wt_hydropathy
            
            # 2. BLOSUM62 score
            blosum_key = (wt_aa, mut_aa)
            if blosum_key not in self.blosum62:
                blosum_key = (mut_aa, wt_aa)  # Try reverse
            df_features.at[idx, 'blosum_score'] = self.blosum62.get(blosum_key, 0)
            
            # 3. Charge change
            wt_charge = self.aa_charges.get(wt_aa, 0)
            mut_charge = self.aa_charges.get(mut_aa, 0)
            df_features.at[idx, 'charge_change'] = mut_charge - wt_charge
            
            # 4. Size change (volume difference)
            wt_volume = self.aa_volumes.get(wt_aa, 0)
            mut_volume = self.aa_volumes.get(mut_aa, 0)
            df_features.at[idx, 'size_change'] = mut_volume - wt_volume
            
            # 5. Is hotspot region (literature-based)
            if protein_name in self.aggregation_prone_proteins:
                protein_info = self.aggregation_prone_proteins[protein_name]
                is_hotspot = 0
                for start, end in protein_info.get('hotspot_regions', []):
                    if start <= position <= end:
                        is_hotspot = 1
                        break
                df_features.at[idx, 'is_hotspot_region'] = is_hotspot
        
        logger.info("REAL features computed successfully!")
        return df_features
    
    def clean_data(self, df):
        """Comprehensive data cleaning"""
        logger.info("Starting data cleaning...")
        original_count = len(df)
        
        # 1. Remove duplicates based on mutation_id
        df_clean = df.drop_duplicates(subset=['mutation_id'], keep='first')
        duplicates_removed = original_count - len(df_clean)
        if duplicates_removed > 0:
            logger.info(f"Removed {duplicates_removed} duplicate mutations")
        
        # 2. Check for missing values
        missing_counts = df_clean.isnull().sum()
        if missing_counts.any():
            logger.warning("Missing values found:")
            for col, count in missing_counts.items():
                if count > 0:
                    logger.warning(f"  {col}: {count} missing values")
        
        # Drop rows with missing critical fields
        critical_fields = ['protein', 'position', 'wt_aa', 'mut_aa', 'wt_sequence', 'mut_sequence']
        before_missing = len(df_clean)
        df_clean = df_clean.dropna(subset=critical_fields)
        missing_removed = before_missing - len(df_clean)
        if missing_removed > 0:
            logger.info(f"Removed {missing_removed} rows with missing critical fields")
        
        # 3. Validate amino acid sequences
        valid_aa_set = set(self.AMINO_ACIDS)
        
        def is_valid_sequence(seq):
            if not isinstance(seq, str) or len(seq) == 0:
                return False
            return all(aa in valid_aa_set for aa in seq.upper())
        
        # Check sequences
        invalid_wt = ~df_clean['wt_sequence'].apply(is_valid_sequence)
        invalid_mut = ~df_clean['mut_sequence'].apply(is_valid_sequence)
        invalid_seqs = invalid_wt | invalid_mut
        
        if invalid_seqs.sum() > 0:
            logger.warning(f"Found {invalid_seqs.sum()} rows with invalid sequences")
            df_clean = df_clean[~invalid_seqs]
        
        # 4. Validate mutation consistency
        def check_mutation_consistency(row):
            pos = int(row['position']) - 1  # Convert to 0-based
            wt_seq = row['wt_sequence']
            mut_seq = row['mut_sequence']
            wt_aa = row['wt_aa']
            mut_aa = row['mut_aa']
            
            if pos < 0 or pos >= len(wt_seq):
                return False
            if wt_seq[pos] != wt_aa:
                return False
            
            expected_mut_seq = wt_seq[:pos] + mut_aa + wt_seq[pos+1:]
            if mut_seq != expected_mut_seq:
                return False
            if len(wt_seq) != len(mut_seq):
                return False
                
            return True
        
        consistent_mask = df_clean.apply(check_mutation_consistency, axis=1)
        inconsistent_count = (~consistent_mask).sum()
        
        if inconsistent_count > 0:
            logger.warning(f"Found {inconsistent_count} inconsistent mutations")
            df_clean = df_clean[consistent_mask]
        
        # 5. Standardize formatting
        df_clean['protein'] = df_clean['protein'].str.lower().str.strip()
        df_clean['wt_aa'] = df_clean['wt_aa'].str.upper()
        df_clean['mut_aa'] = df_clean['mut_aa'].str.upper()
        df_clean['wt_sequence'] = df_clean['wt_sequence'].str.upper()
        df_clean['mut_sequence'] = df_clean['mut_sequence'].str.upper()
        df_clean['position'] = pd.to_numeric(df_clean['position'], errors='coerce')
        df_clean['protein_length'] = pd.to_numeric(df_clean['protein_length'], errors='coerce')
        
        # Remove rows where numeric conversion failed
        df_clean = df_clean.dropna(subset=['position', 'protein_length'])
        
        final_count = len(df_clean)
        total_removed = original_count - final_count
        
        logger.info(f"Cleaning complete: {original_count} → {final_count} ({total_removed} removed)")
        
        return df_clean.reset_index(drop=True)
    
    def add_intelligent_labels(self, df, protein_name):
        """Add intelligent labels based on domain knowledge"""
        logger.info(f"Adding intelligent labels for {protein_name}...")
        
        df_labeled = df.copy()
        
        # Initialize label columns
        df_labeled['aggregation_prone'] = 0
        df_labeled['destabilizing'] = 0
        df_labeled['charge_disrupting'] = 0
        df_labeled['hydrophobicity_disrupt'] = 0
        df_labeled['size_disrupt'] = 0
        df_labeled['ml_target'] = 0  # Main target for ML
        
        for idx, row in tqdm(df_labeled.iterrows(), total=len(df_labeled), desc="Adding labels"):
            protein = row['protein']
            position = row['position']
            wt_aa = row['wt_aa']
            mut_aa = row['mut_aa']
            
            wt_props = self.aa_properties.get(wt_aa, {})
            mut_props = self.aa_properties.get(mut_aa, {})
            
            # 1. Aggregation prone labeling
            aggregation_score = 0
            
            if mut_props.get('aggregation_prone', False):
                aggregation_score += 1
            
            if protein in self.aggregation_prone_proteins:
                protein_info = self.aggregation_prone_proteins[protein]
                for start, end in protein_info.get('regions', []):
                    if start <= position <= end:
                        aggregation_score += 2
                        break
                
                if position in protein_info.get('high_risk_positions', []):
                    aggregation_score += 2
            
            if (wt_props.get('polarity') in ['polar', 'charged'] and 
                mut_props.get('hydrophobicity', 0) > 2):
                aggregation_score += 1
            
            df_labeled.at[idx, 'aggregation_prone'] = 1 if aggregation_score >= 2 else 0
            
            # 2. Destabilizing mutations
            destab_score = 0
            
            if mut_aa == 'P' and position not in range(1, 6):
                destab_score += 2
            
            if wt_aa == 'G' and position > 10:
                destab_score += 1
            
            charge_change = abs(mut_props.get('charge', 0) - wt_props.get('charge', 0))
            if charge_change >= 2:
                destab_score += 1
            
            df_labeled.at[idx, 'destabilizing'] = 1 if destab_score >= 2 else 0
            
            # 3. Charge disrupting
            if charge_change >= 1:
                df_labeled.at[idx, 'charge_disrupting'] = 1
            
            # 4. Hydrophobicity disrupting
            hydro_change = abs(mut_props.get('hydrophobicity', 0) - wt_props.get('hydrophobicity', 0))
            if hydro_change > 3:
                df_labeled.at[idx, 'hydrophobicity_disrupt'] = 1
            
            # 5. Size disrupting
            size_map = {'small': 1, 'medium': 2, 'large': 3}
            wt_size = size_map.get(wt_props.get('size', 'medium'), 2)
            mut_size = size_map.get(mut_props.get('size', 'medium'), 2)
            if abs(mut_size - wt_size) >= 2:
                df_labeled.at[idx, 'size_disrupt'] = 1
            
            # 6. Main ML target (combined pathogenicity score)
            pathogenic_score = (df_labeled.at[idx, 'aggregation_prone'] * 2 +
                              df_labeled.at[idx, 'destabilizing'] * 2 +
                              df_labeled.at[idx, 'charge_disrupting'] * 1 +
                              df_labeled.at[idx, 'hydrophobicity_disrupt'] * 1 +
                              df_labeled.at[idx, 'size_disrupt'] * 1)
            
            df_labeled.at[idx, 'ml_target'] = 1 if pathogenic_score >= 3 else 0
        
        return df_labeled
    
    def _print_protein_summary(self, df, protein_name):
        """Print summary for individual protein"""
        logger.info(f"\n--- {protein_name.upper()} SUMMARY ---")
        logger.info(f"Total mutations: {len(df)}")
        logger.info(f"Sequence length: {df['protein_length'].iloc[0]}")
        
        # Feature statistics
        feature_cols = ['hydropathy_change', 'blosum_score', 'charge_change', 'size_change']
        logger.info("\n📊 FEATURE STATISTICS:")
        for col in feature_cols:
            if col in df.columns:
                mean_val = df[col].mean()
                std_val = df[col].std()
                min_val = df[col].min()
                max_val = df[col].max()
                logger.info(f"  {col}: μ={mean_val:.2f}, σ={std_val:.2f}, range=[{min_val:.2f}, {max_val:.2f}]")
        
        # Label statistics
        label_cols = ['aggregation_prone', 'destabilizing', 'charge_disrupting', 
                     'hydrophobicity_disrupt', 'size_disrupt', 'ml_target']
        logger.info("\n🏷️  LABEL STATISTICS:")
        for col in label_cols:
            if col in df.columns:
                positive_count = df[col].sum()
                positive_pct = (positive_count / len(df)) * 100
                logger.info(f"  {col}: {positive_count}/{len(df)} ({positive_pct:.1f}%)")
        
        # Hotspot regions
        if protein_name in self.aggregation_prone_proteins:
            hotspot_count = df['is_hotspot_region'].sum()
            hotspot_pct = (hotspot_count / len(df)) * 100
            logger.info(f"\n🔥 HOTSPOT REGIONS: {hotspot_count}/{len(df)} ({hotspot_pct:.1f}%)")
        
        # Most common mutations
        logger.info("\n🧬 MOST COMMON MUTATION TYPES:")
        mutation_types = df['wt_aa'].str.cat(df['mut_aa'], sep='→')
        top_mutations = Counter(mutation_types).most_common(5)
        for mut_type, count in top_mutations:
            pct = (count / len(df)) * 100
            logger.info(f"  {mut_type}: {count} ({pct:.1f}%)")
        
        logger.info("-" * 50)
    
    def generate_combined_dataset(self):
        """Combine all individual protein datasets into one master dataset"""
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        
        OUTPUT_DIR = os.path.join(project_root, 'data', 'features')
        COMBINED_FILE = os.path.join(OUTPUT_DIR, 'all_proteins_mutants_combined.csv')
        
        logger.info("🔄 Combining all protein datasets...")
        
        # Find all individual protein files
        protein_files = []
        for file in os.listdir(OUTPUT_DIR):
            if file.endswith('_mutants_complete.csv') and file != 'all_proteins_mutants_combined.csv':
                protein_files.append(os.path.join(OUTPUT_DIR, file))
        
        if not protein_files:
            logger.error("No individual protein files found!")
            return None
        
        logger.info(f"Found {len(protein_files)} protein files to combine")
        
        # Load and combine all datasets
        combined_dfs = []
        total_mutations = 0
        
        for file_path in tqdm(protein_files, desc="Loading protein files"):
            try:
                df = pd.read_csv(file_path)
                protein_name = os.path.basename(file_path).replace('_mutants_complete.csv', '')
                logger.info(f"  {protein_name}: {len(df)} mutations")
                combined_dfs.append(df)
                total_mutations += len(df)
            except Exception as e:
                logger.error(f"Error loading {file_path}: {e}")
                continue
        
        # Combine all DataFrames
        logger.info("Combining datasets...")
        combined_df = pd.concat(combined_dfs, ignore_index=True)
        
        # Final data validation
        logger.info("Performing final validation...")
        combined_df_clean = self._final_validation(combined_df)
        
        # Save combined dataset
        try:
            combined_df_clean.to_csv(COMBINED_FILE, index=False)
            file_size_mb = os.path.getsize(COMBINED_FILE) / (1024*1024)
            logger.info(f"✅ Combined dataset saved: {COMBINED_FILE}")
            logger.info(f"📊 Final dataset: {len(combined_df_clean)} mutations, {file_size_mb:.2f} MB")
            
            # Print overall summary
            self._print_combined_summary(combined_df_clean)
            
            return COMBINED_FILE
            
        except Exception as e:
            logger.error(f"Error saving combined dataset: {e}")
            return None
    
    def _final_validation(self, df):
        """Final validation of combined dataset"""
        logger.info("Running final validation checks...")
        original_count = len(df)
        
        # Check for duplicate mutation IDs
        duplicates = df['mutation_id'].duplicated()
        if duplicates.sum() > 0:
            logger.warning(f"Found {duplicates.sum()} duplicate mutation IDs - removing")
            df = df[~duplicates]
        
        # Validate all required columns exist
        required_cols = ['mutation_id', 'protein', 'position', 'wt_aa', 'mut_aa', 
                        'wt_sequence', 'mut_sequence', 'protein_length',
                        'hydropathy_change', 'blosum_score', 'charge_change', 
                        'size_change', 'is_hotspot_region', 'ml_target']
        
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            logger.error(f"Missing required columns: {missing_cols}")
            return df
        
        # Check for any remaining missing values in critical columns
        critical_cols = ['mutation_id', 'protein', 'position', 'wt_aa', 'mut_aa']
        missing_critical = df[critical_cols].isnull().any(axis=1)
        if missing_critical.sum() > 0:
            logger.warning(f"Removing {missing_critical.sum()} rows with missing critical data")
            df = df[~missing_critical]
        
        # Validate amino acid codes
        valid_aas = set(self.AMINO_ACIDS)
        invalid_wt = ~df['wt_aa'].isin(valid_aas)
        invalid_mut = ~df['mut_aa'].isin(valid_aas)
        
        if invalid_wt.sum() > 0:
            logger.warning(f"Found {invalid_wt.sum()} invalid wild-type amino acids")
            df = df[~invalid_wt]
        
        if invalid_mut.sum() > 0:
            logger.warning(f"Found {invalid_mut.sum()} invalid mutant amino acids")
            df = df[~invalid_mut]
        
        # Validate position ranges
        invalid_pos = (df['position'] < 1) | (df['position'] > df['protein_length'])
        if invalid_pos.sum() > 0:
            logger.warning(f"Found {invalid_pos.sum()} invalid positions")
            df = df[~invalid_pos]
        
        final_count = len(df)
        removed_count = original_count - final_count
        
        logger.info(f"Final validation complete: {original_count} → {final_count} ({removed_count} removed)")
        
        return df.reset_index(drop=True)
    
    def _print_combined_summary(self, df):
        """Print comprehensive summary of combined dataset"""
        logger.info("\n" + "="*60)
        logger.info("🎯 FINAL COMBINED DATASET SUMMARY")
        logger.info("="*60)
        
        # Basic statistics
        logger.info(f"📊 Dataset size: {len(df):,} mutations")
        logger.info(f"🧬 Proteins: {df['protein'].nunique()}")
        logger.info(f"📍 Average protein length: {df['protein_length'].mean():.1f} amino acids")
        
        # Protein distribution
        logger.info(f"\n📈 MUTATIONS PER PROTEIN:")
        protein_counts = df['protein'].value_counts()
        for protein, count in protein_counts.items():
            pct = (count / len(df)) * 100
            logger.info(f"  {protein}: {count:,} ({pct:.1f}%)")
        
        # Feature distributions
        logger.info(f"\n📊 FEATURE DISTRIBUTIONS:")
        feature_cols = ['hydropathy_change', 'blosum_score', 'charge_change', 'size_change']
        for col in feature_cols:
            mean_val = df[col].mean()
            std_val = df[col].std()
            logger.info(f"  {col}: μ={mean_val:.2f}, σ={std_val:.2f}")
        
        # Label distributions
        logger.info(f"\n🏷️  LABEL DISTRIBUTIONS:")
        label_cols = ['aggregation_prone', 'destabilizing', 'charge_disrupting', 
                     'hydrophobicity_disrupt', 'size_disrupt', 'ml_target']
        for col in label_cols:
            positive_count = df[col].sum()
            positive_pct = (positive_count / len(df)) * 100
            logger.info(f"  {col}: {positive_count:,}/{len(df):,} ({positive_pct:.1f}%)")
        
        # Class balance for ML target
        ml_target_dist = df['ml_target'].value_counts()
        logger.info(f"\n🎯 ML TARGET CLASS BALANCE:")
        logger.info(f"  Benign (0): {ml_target_dist.get(0, 0):,} ({ml_target_dist.get(0, 0)/len(df)*100:.1f}%)")
        logger.info(f"  Pathogenic (1): {ml_target_dist.get(1, 0):,} ({ml_target_dist.get(1, 0)/len(df)*100:.1f}%)")
        
        # Amino acid mutation preferences
        logger.info(f"\n🔄 TOP MUTATION PATTERNS:")
        mutation_patterns = df['wt_aa'].str.cat(df['mut_aa'], sep='→')
        top_patterns = Counter(mutation_patterns).most_common(10)
        for pattern, count in top_patterns:
            pct = (count / len(df)) * 100
            logger.info(f"  {pattern}: {count:,} ({pct:.1f}%)")
        
        # Data quality metrics
        logger.info(f"\n✅ DATA QUALITY:")
        logger.info(f"  Unique mutation IDs: {df['mutation_id'].nunique():,}/{len(df):,} ({df['mutation_id'].nunique()/len(df)*100:.1f}%)")
        logger.info(f"  Complete cases: {len(df):,}")
        logger.info(f"  No missing values in critical columns: ✅")
        
        logger.info("="*60)
        logger.info("🎉 Dataset generation completed successfully!")
        logger.info("="*60)
    
    def analyze_mutation_impacts(self, csv_file_path):
        """Analyze mutation impacts from generated dataset"""
        logger.info(f"🔬 Analyzing mutation impacts from: {csv_file_path}")
        
        try:
            df = pd.read_csv(csv_file_path)
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            return None
        
        logger.info(f"Loaded {len(df)} mutations for analysis")
        
        # Analysis by amino acid properties
        self._analyze_by_amino_acid_properties(df)
        
        # Analysis by position effects
        self._analyze_positional_effects(df)
        
        # Analysis by protein-specific patterns
        self._analyze_protein_specific_patterns(df)
        
        # Generate correlation matrix
        self._generate_feature_correlations(df)
        
        return True
    
    def _analyze_by_amino_acid_properties(self, df):
        """Analyze mutations by amino acid property changes"""
        logger.info("\n🧬 AMINO ACID PROPERTY ANALYSIS")
        logger.info("-" * 40)
        
        # Group mutations by property changes
        for prop in ['polarity', 'size', 'charge']:
            logger.info(f"\n{prop.upper()} CHANGES:")
            
            # Count mutations that change this property
            change_count = 0
            for idx, row in df.iterrows():
                wt_prop = self.aa_properties.get(row['wt_aa'], {}).get(prop)
                mut_prop = self.aa_properties.get(row['mut_aa'], {}).get(prop)
                if wt_prop != mut_prop:
                    change_count += 1
            
            change_pct = (change_count / len(df)) * 100
            logger.info(f"  Mutations changing {prop}: {change_count}/{len(df)} ({change_pct:.1f}%)")
            
            # Correlation with pathogenicity
            if change_count > 0:
                pathogenic_in_change = df[df.apply(lambda row: 
                    self.aa_properties.get(row['wt_aa'], {}).get(prop) != 
                    self.aa_properties.get(row['mut_aa'], {}).get(prop), axis=1)]['ml_target'].mean()
                logger.info(f"  Pathogenicity rate in {prop} changes: {pathogenic_in_change:.1%}")
    
    def _analyze_positional_effects(self, df):
        """Analyze position-dependent mutation effects"""
        logger.info("\n📍 POSITIONAL EFFECTS ANALYSIS")
        logger.info("-" * 40)
        
        # Analyze by relative position (N-terminal, middle, C-terminal)
        for protein in df['protein'].unique():
            protein_df = df[df['protein'] == protein]
            length = protein_df['protein_length'].iloc[0]
            
            # Divide into thirds
            n_term = protein_df[protein_df['position'] <= length/3]
            middle = protein_df[(protein_df['position'] > length/3) & (protein_df['position'] <= 2*length/3)]
            c_term = protein_df[protein_df['position'] > 2*length/3]
            
            logger.info(f"\n{protein.upper()}:")
            for region_df, region_name in [(n_term, 'N-terminal'), (middle, 'Middle'), (c_term, 'C-terminal')]:
                if len(region_df) > 0:
                    pathogenic_rate = region_df['ml_target'].mean()
                    logger.info(f"  {region_name}: {len(region_df)} mutations, {pathogenic_rate:.1%} pathogenic")
    
    def _analyze_protein_specific_patterns(self, df):
        """Analyze protein-specific mutation patterns"""
        logger.info("\n🔬 PROTEIN-SPECIFIC PATTERNS")
        logger.info("-" * 40)
        
        for protein in df['protein'].unique():
            protein_df = df[df['protein'] == protein]
            logger.info(f"\n{protein.upper()}:")
            
            # Most pathogenic positions
            pos_pathogenicity = protein_df.groupby('position')['ml_target'].mean()
            top_pathogenic_pos = pos_pathogenicity.nlargest(5)
            
            logger.info("  Top pathogenic positions:")
            for pos, rate in top_pathogenic_pos.items():
                if rate > 0:
                    logger.info(f"    Position {pos}: {rate:.1%} pathogenic")
            
            # Feature importance for this protein
            feature_cols = ['hydropathy_change', 'blosum_score', 'charge_change', 'size_change']
            logger.info("  Feature correlations with pathogenicity:")
            for feature in feature_cols:
                corr = protein_df[feature].corr(protein_df['ml_target'])
                logger.info(f"    {feature}: {corr:.3f}")
    
    def _generate_feature_correlations(self, df):
        """Generate correlation matrix for features"""
        logger.info("\n📊 FEATURE CORRELATION MATRIX")
        logger.info("-" * 40)
        
        # Select numeric columns for correlation
        numeric_cols = ['hydropathy_change', 'blosum_score', 'charge_change', 
                       'size_change', 'is_hotspot_region', 'aggregation_prone', 
                       'destabilizing', 'ml_target']
        
        correlation_matrix = df[numeric_cols].corr()
        
        # Print correlation with ML target
        logger.info("Correlations with ML target (pathogenicity):")
        ml_target_corrs = correlation_matrix['ml_target'].drop('ml_target').sort_values(key=abs, ascending=False)
        
        for feature, corr in ml_target_corrs.items():
            logger.info(f"  {feature}: {corr:.3f}")


# Usage example and main execution
def main():
    """Main execution function"""
    processor = EnhancedMutationDataProcessor()
    
    # Generate all mutants with real features
    logger.info("🚀 Starting mutation dataset generation...")
    success = processor.generate_all_mutants()
    
    if success:
        # Combine all protein datasets
        combined_file = processor.generate_combined_dataset()
        
        if combined_file:
            # Analyze the final dataset
            processor.analyze_mutation_impacts(combined_file)
            
            logger.info("✅ All processing completed successfully!")
            logger.info(f"📁 Final dataset: {combined_file}")
        else:
            logger.error("❌ Failed to create combined dataset")
    else:
        logger.error("❌ Failed to generate mutation datasets")


if __name__ == "__main__":
    main()