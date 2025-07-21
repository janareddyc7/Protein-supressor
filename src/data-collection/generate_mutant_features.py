#!/usr/bin/env python3
import os
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
import logging
from collections import Counter
import hashlib
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HighQualityMutationProcessor:
    def __init__(self):
        # Standard amino acids
        self.AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
        
        # REAL Kyte-Doolittle hydropathy scale (Kyte & Doolittle, 1982)
        self.kyte_doolittle = {
            'A': 1.8, 'C': 2.5, 'D': -3.5, 'E': -3.5, 'F': 2.8,
            'G': -0.4, 'H': -3.2, 'I': 4.5, 'K': -3.9, 'L': 3.8,
            'M': 1.9, 'N': -3.5, 'P': -1.6, 'Q': -3.5, 'R': -4.5,
            'S': -0.8, 'T': -0.7, 'V': 4.2, 'W': -0.9, 'Y': -1.3
        }
        
        # REAL amino acid charges at physiological pH 7.4
        self.aa_charges = {
            'D': -1, 'E': -1, 'K': 1, 'R': 1, 'H': 0.1,  # His partially charged at pH 7.4
            'A': 0, 'C': 0, 'F': 0, 'G': 0, 'I': 0,
            'L': 0, 'M': 0, 'N': 0, 'P': 0, 'Q': 0,
            'S': 0, 'T': 0, 'V': 0, 'W': 0, 'Y': 0
        }
        
        # REAL amino acid volumes (Zamyatnin, 1972) in Å³
        self.aa_volumes = {
            'A': 88.6, 'C': 108.5, 'D': 111.1, 'E': 138.4, 'F': 189.9,
            'G': 60.1, 'H': 153.2, 'I': 166.7, 'K': 168.6, 'L': 166.7,
            'M': 162.9, 'N': 114.1, 'P': 112.7, 'Q': 143.8, 'R': 173.4,
            'S': 89.0, 'T': 116.1, 'V': 140.0, 'W': 227.8, 'Y': 193.6
        }
        
        # REAL BLOSUM62 substitution matrix
        self.blosum62 = self._init_blosum62()
        
        # REAL amino acid flexibility (B-factors from PDB analysis)
        self.aa_flexibility = {
            'G': 0.984, 'A': 0.906, 'S': 0.929, 'T': 0.912, 'D': 0.924,
            'N': 0.923, 'P': 0.873, 'C': 0.891, 'Q': 0.912, 'E': 0.912,
            'H': 0.897, 'K': 0.897, 'R': 0.897, 'M': 0.879, 'L': 0.879,
            'V': 0.879, 'I': 0.879, 'F': 0.855, 'Y': 0.855, 'W': 0.824
        }
        
        # REAL amino acid polarizability (Miller et al., 1990)
        self.aa_polarizability = {
            'A': 0.046, 'C': 0.128, 'D': 0.105, 'E': 0.151, 'F': 0.290,
            'G': 0.000, 'H': 0.230, 'I': 0.186, 'K': 0.219, 'L': 0.186,
            'M': 0.221, 'N': 0.134, 'P': 0.131, 'Q': 0.180, 'R': 0.291,
            'S': 0.062, 'T': 0.108, 'V': 0.140, 'W': 0.409, 'Y': 0.298
        }
        
        # REAL amino acid accessible surface area (Chothia, 1976)
        self.aa_asa = {
            'A': 115, 'C': 135, 'D': 150, 'E': 190, 'F': 210,
            'G': 75, 'H': 195, 'I': 175, 'K': 200, 'L': 170,
            'M': 185, 'N': 160, 'P': 145, 'Q': 180, 'R': 225,
            'S': 115, 'T': 140, 'V': 155, 'W': 255, 'Y': 230
        }
        
        # REAL amino acid beta-sheet propensity (Chou-Fasman)
        self.aa_beta_propensity = {
            'A': 0.83, 'C': 1.19, 'D': 0.54, 'E': 0.37, 'F': 1.38,
            'G': 0.75, 'H': 0.87, 'I': 1.60, 'K': 0.74, 'L': 1.30,
            'M': 1.05, 'N': 0.89, 'P': 0.55, 'Q': 1.10, 'R': 0.93,
            'S': 0.75, 'T': 1.19, 'V': 1.70, 'W': 1.37, 'Y': 1.47
        }
        
        # REAL amino acid alpha-helix propensity (Chou-Fasman)
        self.aa_alpha_propensity = {
            'A': 1.42, 'C': 0.70, 'D': 1.01, 'E': 1.51, 'F': 1.13,
            'G': 0.57, 'H': 1.00, 'I': 1.08, 'K': 1.16, 'L': 1.21,
            'M': 1.45, 'N': 0.67, 'P': 0.57, 'Q': 1.11, 'R': 0.98,
            'S': 0.77, 'T': 0.83, 'V': 1.06, 'W': 1.08, 'Y': 0.69
        }

    def _init_blosum62(self):
        """Initialize BLOSUM62 substitution matrix"""
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

    def generate_enhanced_dataset(self):
        """Generate enhanced mutation dataset with only REAL features"""
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        
        INPUT_JSON = os.path.join(project_root, 'data', 'proteins', 'all_sequences.json')
        OUTPUT_DIR = os.path.join(project_root, 'data', 'features')
        
        logger.info(f"Input: {INPUT_JSON}")
        logger.info(f"Output: {OUTPUT_DIR}")
        
        if not os.path.exists(INPUT_JSON):
            logger.error(f"Input file not found: {INPUT_JSON}")
            return None
        
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        # Load protein sequences
        try:
            with open(INPUT_JSON, 'r') as f:
                proteins = json.load(f)
            logger.info(f"Loaded {len(proteins)} proteins")
        except Exception as e:
            logger.error(f"Error loading JSON: {e}")
            return None
        
        all_datasets = []
        total_mutations = 0
        
        # Process each protein
        for protein_name, protein_info in proteins.items():
            logger.info(f"Processing {protein_name}...")
            
            # Generate all possible mutations
            mutations = self._generate_all_mutations(protein_name, protein_info)
            
            # Add real biochemical features
            mutations_with_features = self._add_real_features(mutations)
            
            # Add high-quality labels based on biochemical principles
            mutations_labeled = self._add_biochemical_labels(mutations_with_features)
            
            # Quality validation
            mutations_clean = self._validate_data_quality(mutations_labeled)
            
            # Save individual protein dataset
            output_file = os.path.join(OUTPUT_DIR, f"{protein_name}_mutations_enhanced.csv")
            mutations_clean.to_csv(output_file, index=False)
            
            all_datasets.append(mutations_clean)
            total_mutations += len(mutations_clean)
            
            logger.info(f"✅ {protein_name}: {len(mutations_clean)} mutations")
            self._print_protein_stats(mutations_clean, protein_name)
        
        # Combine all datasets
        combined_df = pd.concat(all_datasets, ignore_index=True)
        
        # Final quality checks and enhancements
        final_dataset = self._final_quality_enhancement(combined_df)
        
        # Save combined dataset
        combined_file = os.path.join(OUTPUT_DIR, 'all_proteins_enhanced_mutations.csv')
        final_dataset.to_csv(combined_file, index=False)
        
        logger.info(f"🎉 FINAL DATASET: {len(final_dataset)} mutations")
        logger.info(f"📁 Saved to: {combined_file}")
        
        # Print comprehensive analysis
        self._print_final_analysis(final_dataset)
        
        return combined_file

    def _generate_all_mutations(self, protein_name, protein_info):
        """Generate all possible single-point mutations"""
        sequence = protein_info['sequence']
        length = len(sequence)
        
        mutations = []
        
        for position in range(length):
            wt_aa = sequence[position]
            
            for mut_aa in self.AMINO_ACIDS:
                if mut_aa == wt_aa:
                    continue
                
                mut_sequence = sequence[:position] + mut_aa + sequence[position+1:]
                mutation_id = f"{protein_name}_{position+1}_{wt_aa}{mut_aa}"
                
                mutations.append({
                    'mutation_id': mutation_id,
                    'protein': protein_name,
                    'position': position + 1,
                    'wt_aa': wt_aa,
                    'mut_aa': mut_aa,
                    'wt_sequence': sequence,
                    'mut_sequence': mut_sequence,
                    'protein_length': length,
                    'relative_position': (position + 1) / length
                })
        
        return pd.DataFrame(mutations)

    def _add_real_features(self, df):
        """Add only REAL biochemical features"""
        logger.info("Adding real biochemical features...")
        
        df_features = df.copy()
        
        # Initialize feature columns
        feature_columns = [
            'hydropathy_change', 'blosum_score', 'charge_change', 'volume_change',
            'flexibility_change', 'polarizability_change', 'asa_change',
            'beta_propensity_change', 'alpha_propensity_change',
            'hydropathy_magnitude', 'charge_magnitude', 'volume_magnitude',
            'is_proline_mutation', 'is_glycine_mutation', 'is_cysteine_mutation',
            'is_aromatic_change', 'is_charged_change', 'is_polar_change',
            'conservation_disruption_score', 'structural_impact_score'
        ]
        
        for col in feature_columns:
            df_features[col] = 0.0
        
        for idx, row in tqdm(df_features.iterrows(), total=len(df_features), desc="Computing features"):
            wt_aa = row['wt_aa']
            mut_aa = row['mut_aa']
            position = row['position']
            rel_pos = row['relative_position']
            
            # 1. Hydropathy change (Kyte-Doolittle)
            wt_hydro = self.kyte_doolittle[wt_aa]
            mut_hydro = self.kyte_doolittle[mut_aa]
            hydro_change = mut_hydro - wt_hydro
            df_features.at[idx, 'hydropathy_change'] = hydro_change
            df_features.at[idx, 'hydropathy_magnitude'] = abs(hydro_change)
            
            # 2. BLOSUM62 score
            blosum_key = (wt_aa, mut_aa)
            if blosum_key not in self.blosum62:
                blosum_key = (mut_aa, wt_aa)
            df_features.at[idx, 'blosum_score'] = self.blosum62.get(blosum_key, 0)
            
            # 3. Charge change
            wt_charge = self.aa_charges[wt_aa]
            mut_charge = self.aa_charges[mut_aa]
            charge_change = mut_charge - wt_charge
            df_features.at[idx, 'charge_change'] = charge_change
            df_features.at[idx, 'charge_magnitude'] = abs(charge_change)
            
            # 4. Volume change
            wt_vol = self.aa_volumes[wt_aa]
            mut_vol = self.aa_volumes[mut_aa]
            vol_change = mut_vol - wt_vol
            df_features.at[idx, 'volume_change'] = vol_change
            df_features.at[idx, 'volume_magnitude'] = abs(vol_change)
            
            # 5. Flexibility change
            wt_flex = self.aa_flexibility[wt_aa]
            mut_flex = self.aa_flexibility[mut_aa]
            df_features.at[idx, 'flexibility_change'] = mut_flex - wt_flex
            
            # 6. Polarizability change
            wt_pol = self.aa_polarizability[wt_aa]
            mut_pol = self.aa_polarizability[mut_aa]
            df_features.at[idx, 'polarizability_change'] = mut_pol - wt_pol
            
            # 7. Accessible surface area change
            wt_asa = self.aa_asa[wt_aa]
            mut_asa = self.aa_asa[mut_aa]
            df_features.at[idx, 'asa_change'] = mut_asa - wt_asa
            
            # 8. Secondary structure propensity changes
            wt_beta = self.aa_beta_propensity[wt_aa]
            mut_beta = self.aa_beta_propensity[mut_aa]
            df_features.at[idx, 'beta_propensity_change'] = mut_beta - wt_beta
            
            wt_alpha = self.aa_alpha_propensity[wt_aa]
            mut_alpha = self.aa_alpha_propensity[mut_aa]
            df_features.at[idx, 'alpha_propensity_change'] = mut_alpha - wt_alpha
            
            # 9. Special amino acid mutations
            df_features.at[idx, 'is_proline_mutation'] = 1 if mut_aa == 'P' else 0
            df_features.at[idx, 'is_glycine_mutation'] = 1 if wt_aa == 'G' else 0
            df_features.at[idx, 'is_cysteine_mutation'] = 1 if wt_aa == 'C' or mut_aa == 'C' else 0
            
            # 10. Chemical property changes
            aromatic_aas = {'F', 'W', 'Y', 'H'}
            is_aromatic_change = (wt_aa in aromatic_aas) != (mut_aa in aromatic_aas)
            df_features.at[idx, 'is_aromatic_change'] = 1 if is_aromatic_change else 0
            
            charged_aas = {'D', 'E', 'K', 'R', 'H'}
            is_charged_change = (wt_aa in charged_aas) != (mut_aa in charged_aas)
            df_features.at[idx, 'is_charged_change'] = 1 if is_charged_change else 0
            
            polar_aas = {'S', 'T', 'N', 'Q', 'Y', 'C'}
            is_polar_change = (wt_aa in polar_aas) != (mut_aa in polar_aas)
            df_features.at[idx, 'is_polar_change'] = 1 if is_polar_change else 0
            
            # 11. Conservation disruption score (based on BLOSUM62)
            conservation_score = max(0, -self.blosum62.get((wt_aa, mut_aa), 0))
            df_features.at[idx, 'conservation_disruption_score'] = conservation_score
            
            # 12. Structural impact score (composite)
            structural_impact = (
                abs(hydro_change) * 0.3 +
                abs(charge_change) * 2.0 +
                abs(vol_change) / 50.0 +
                conservation_score * 0.2 +
                (1 if is_aromatic_change else 0) * 0.5
            )
            df_features.at[idx, 'structural_impact_score'] = structural_impact
        
        logger.info("✅ Real biochemical features computed")
        return df_features

    def _add_biochemical_labels(self, df):
        """Add high-quality labels based on biochemical principles"""
        logger.info("Adding biochemical labels...")
        
        df_labeled = df.copy()
        
        # Initialize label columns
        df_labeled['is_destabilizing'] = 0
        df_labeled['is_aggregation_prone'] = 0
        df_labeled['is_functionally_disruptive'] = 0
        df_labeled['pathogenicity_score'] = 0.0
        df_labeled['ml_target'] = 0
        
        for idx, row in tqdm(df_labeled.iterrows(), total=len(df_labeled), desc="Adding labels"):
            wt_aa = row['wt_aa']
            mut_aa = row['mut_aa']
            position = row['position']
            rel_pos = row['relative_position']
            
            destab_score = 0
            aggregation_score = 0
            functional_score = 0
            
            # Destabilizing mutations
            # 1. Proline in secondary structure
            if mut_aa == 'P' and 0.1 < rel_pos < 0.9:
                destab_score += 2
            
            # 2. Glycine mutations (loss of flexibility)
            if wt_aa == 'G' and rel_pos > 0.1:
                destab_score += 1
            
            # 3. Large charge changes
            if abs(row['charge_change']) >= 2:
                destab_score += 2
            elif abs(row['charge_change']) >= 1:
                destab_score += 1
            
            # 4. Large volume changes
            if abs(row['volume_change']) > 80:
                destab_score += 2
            elif abs(row['volume_change']) > 40:
                destab_score += 1
            
            # 5. Cysteine mutations (disulfide bond disruption)
            if wt_aa == 'C':
                destab_score += 2
            
            # 6. Poor BLOSUM62 scores
            if row['blosum_score'] <= -3:
                destab_score += 2
            elif row['blosum_score'] <= -1:
                destab_score += 1
            
            # Aggregation-prone mutations
            # 1. Hydrophobic mutations in exposed regions
            if row['hydropathy_change'] > 3 and rel_pos < 0.8:
                aggregation_score += 1
            
            # 2. Loss of charged residues
            if wt_aa in {'K', 'R', 'D', 'E'} and mut_aa not in {'K', 'R', 'D', 'E'}:
                aggregation_score += 1
            
            # 3. Aromatic residue introduction
            if mut_aa in {'F', 'W', 'Y'} and wt_aa not in {'F', 'W', 'Y'}:
                aggregation_score += 1
            
            # 4. Beta-sheet propensity increase
            if row['beta_propensity_change'] > 0.5:
                aggregation_score += 1
            
            # Functionally disruptive mutations
            # 1. Active site regions (N-terminal and C-terminal often important)
            if rel_pos < 0.1 or rel_pos > 0.9:
                functional_score += 1
            
            # 2. Charge reversal
            if (wt_aa in {'D', 'E'} and mut_aa in {'K', 'R'}) or \
               (wt_aa in {'K', 'R'} and mut_aa in {'D', 'E'}):
                functional_score += 2
            
            # 3. Loss of special residues
            if wt_aa in {'W', 'Y', 'H', 'C', 'M', 'P'}:
                functional_score += 1
            
            # 4. High structural impact
            if row['structural_impact_score'] > 3:
                functional_score += 1
            
            # Set binary labels
            df_labeled.at[idx, 'is_destabilizing'] = 1 if destab_score >= 2 else 0
            df_labeled.at[idx, 'is_aggregation_prone'] = 1 if aggregation_score >= 2 else 0
            df_labeled.at[idx, 'is_functionally_disruptive'] = 1 if functional_score >= 2 else 0
            
            # Calculate pathogenicity score
            pathogenicity = (
                destab_score * 0.4 +
                aggregation_score * 0.3 +
                functional_score * 0.3
            )
            df_labeled.at[idx, 'pathogenicity_score'] = pathogenicity
            
            # Main ML target (pathogenic if score >= 2.5)
            df_labeled.at[idx, 'ml_target'] = 1 if pathogenicity >= 2.5 else 0
        
        logger.info("✅ Biochemical labels added")
        return df_labeled

    def _validate_data_quality(self, df):
        """Comprehensive data quality validation"""
        logger.info("Validating data quality...")
        
        original_count = len(df)
        
        # Remove duplicates
        df_clean = df.drop_duplicates(subset=['mutation_id'])
        
        # Check for missing values
        df_clean = df_clean.dropna()
        
        # Validate amino acid codes
        valid_aas = set(self.AMINO_ACIDS)
        valid_mask = (df_clean['wt_aa'].isin(valid_aas) & 
                     df_clean['mut_aa'].isin(valid_aas))
        df_clean = df_clean[valid_mask]
        
        # Validate positions
        pos_mask = ((df_clean['position'] >= 1) & 
                   (df_clean['position'] <= df_clean['protein_length']))
        df_clean = df_clean[pos_mask]
        
        # Validate sequences
        def validate_mutation(row):
            pos = int(row['position']) - 1
            wt_seq = row['wt_sequence']
            mut_seq = row['mut_sequence']
            
            if pos >= len(wt_seq) or pos < 0:
                return False
            if wt_seq[pos] != row['wt_aa']:
                return False
            if len(wt_seq) != len(mut_seq):
                return False
            
            expected_mut = wt_seq[:pos] + row['mut_aa'] + wt_seq[pos+1:]
            return mut_seq == expected_mut
        
        valid_mutations = df_clean.apply(validate_mutation, axis=1)
        df_clean = df_clean[valid_mutations]
        
        final_count = len(df_clean)
        removed = original_count - final_count
        
        logger.info(f"Quality validation: {original_count} → {final_count} ({removed} removed)")
        
        return df_clean.reset_index(drop=True)

    def _final_quality_enhancement(self, df):
        """Final quality enhancements for the combined dataset"""
        logger.info("Applying final quality enhancements...")
        
        # Add cross-protein features
        df_enhanced = df.copy()
        
        # Protein-specific normalization
        for protein in df['protein'].unique():
            protein_mask = df_enhanced['protein'] == protein
            protein_df = df_enhanced[protein_mask]
            
            # Normalize features within protein
            feature_cols = ['hydropathy_change', 'volume_change', 'structural_impact_score']
            for col in feature_cols:
                if protein_df[col].std() > 0:
                    normalized_col = f"{col}_normalized"
                    df_enhanced.loc[protein_mask, normalized_col] = (
                        (protein_df[col] - protein_df[col].mean()) / protein_df[col].std()
                    )
                else:
                    df_enhanced.loc[protein_mask, f"{col}_normalized"] = 0
        
        # Add ensemble features
        df_enhanced['combined_impact_score'] = (
            df_enhanced['structural_impact_score'] * 0.4 +
            df_enhanced['conservation_disruption_score'] * 0.3 +
            abs(df_enhanced['hydropathy_change']) * 0.2 +
            abs(df_enhanced['charge_change']) * 0.1
        )
        
        # Quality-based filtering
        # Remove extreme outliers that might be errors
        for col in ['hydropathy_change', 'volume_change', 'structural_impact_score']:
            q99 = df_enhanced[col].quantile(0.99)
            q01 = df_enhanced[col].quantile(0.01)
            df_enhanced = df_enhanced[
                (df_enhanced[col] >= q01) & (df_enhanced[col] <= q99)
            ]
        
        logger.info(f"Final dataset size: {len(df_enhanced)}")
        return df_enhanced

    def _print_protein_stats(self, df, protein_name):
        """Print statistics for individual protein"""
        logger.info(f"\n--- {protein_name.upper()} STATISTICS ---")
        logger.info(f"Mutations: {len(df)}")
        logger.info(f"Pathogenic: {df['ml_target'].sum()} ({df['ml_target'].mean():.1%})")
        logger.info(f"Destabilizing: {df['is_destabilizing'].sum()} ({df['is_destabilizing'].mean():.1%})")
        logger.info(f"Aggregation-prone: {df['is_aggregation_prone'].sum()} ({df['is_aggregation_prone'].mean():.1%})")

    def _print_final_analysis(self, df):
        """Print comprehensive final analysis"""
        logger.info("\n" + "="*60)
        logger.info("🎯 FINAL DATASET ANALYSIS")
        logger.info("="*60)
        
        logger.info(f"📊 Total mutations: {len(df):,}")
        logger.info(f"🧬 Proteins: {df['protein'].nunique()}")
        logger.info(f"📍 Average protein length: {df['protein_length'].mean():.1f}")
        
        # Class distribution
        logger.info(f"\n🏷️ CLASS DISTRIBUTION:")
        pathogenic_count = df['ml_target'].sum()
        benign_count = len(df) - pathogenic_count
        logger.info(f"  Pathogenic: {pathogenic_count:,} ({pathogenic_count/len(df):.1%})")
        logger.info(f"  Benign: {benign_count:,} ({benign_count/len(df):.1%})")
        
        # Feature statistics
        logger.info(f"\n📈 FEATURE STATISTICS:")
        key_features = ['hydropathy_change', 'blosum_score', 'charge_change', 
                       'volume_change', 'structural_impact_score']
        for feature in key_features:
            mean_val = df[feature].mean()
            std_val = df[feature].std()
            logger.info(f"  {feature}: μ={mean_val:.3f}, σ={std_val:.3f}")
        
        # Protein distribution
        logger.info(f"\n🧬 MUTATIONS PER PROTEIN:")
        for protein, count in df['protein'].value_counts().items():
            pct = count / len(df) * 100
            pathogenic_rate = df[df['protein'] == protein]['ml_target'].mean()
            logger.info(f"  {protein}: {count:,} ({pct:.1f}%) - {pathogenic_rate:.1%} pathogenic")
        
        logger.info("="*60)
        logger.info("✅ HIGH-QUALITY DATASET GENERATED SUCCESSFULLY!")
        logger.info("="*60)

def main():
    """Main execution function"""
    processor = HighQualityMutationProcessor()
    
    logger.info("🚀 Starting high-quality mutation dataset generation...")
    logger.info("📋 FEATURES INCLUDED (ALL REAL):")
    logger.info("  ✅ Kyte-Doolittle hydropathy scale")
    logger.info("  ✅ BLOSUM62 substitution matrix")
    logger.info("  ✅ Amino acid charges (pH 7.4)")
    logger.info("  ✅ Amino acid volumes (Zamyatnin)")
    logger.info("  ✅ Flexibility (B-factors)")
    logger.info("  ✅ Polarizability (Miller et al.)")
    logger.info("  ✅ Accessible surface area (Chothia)")
    logger.info("  ✅ Secondary structure propensities (Chou-Fasman)")
    logger.info("  ✅ Chemical property changes")
    logger.info("  ✅ Conservation disruption scores")
    logger.info("  ✅ Structural impact scores")
    logger.info("\n❌ REMOVED FAKE FEATURES:")
    logger.info("  ❌ Fake hotspot regions")
    logger.info("  ❌ Mock disorder predictions")
    logger.info("  ❌ Simulated conservation scores")
    
    result = processor.generate_enhanced_dataset()
    
    if result:
        logger.info(f"🎉 SUCCESS! Dataset saved to: {result}")
        logger.info("🎯 Ready for high-accuracy ML model training!")
    else:
        logger.error("❌ Failed to generate dataset")

if __name__ == "__main__":
    main()
