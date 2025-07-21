#!/usr/bin/env python3
import os
import json
import pandas as pd
from tqdm import tqdm

def generate_all_mutants():
    # Use absolute paths from project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    INPUT_JSON = os.path.join(project_root, 'data', 'proteins', 'all_sequences.json')
    OUTPUT_CSV = os.path.join(project_root, 'data', 'features', 'all_proteins_mutants.csv')
    
    print(f"Looking for input: {INPUT_JSON}")
    print(f"Will save to: {OUTPUT_CSV}")
    
    # Standard amino acids (20 total)
    AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
    
    # Check if input file exists
    if not os.path.exists(INPUT_JSON):
        print(f"ERROR: Input file not found: {INPUT_JSON}")
        print("Please run data collection first!")
        return
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    
    # Load protein sequences
    try:
        with open(INPUT_JSON, 'r') as f:
            proteins = json.load(f)
        print(f"Loaded {len(proteins)} proteins")
    except Exception as e:
        print(f"ERROR loading JSON: {e}")
        return
    
    # Calculate total mutations for progress bar
    total_positions = sum(info['length'] for info in proteins.values())
    expected_mutants = total_positions * (len(AMINO_ACIDS) - 1)  # -1 because we skip wild-type
    
    print(f"Total positions: {total_positions}")
    print(f"Expected mutants: {expected_mutants}")
    
    rows = []
    
    # Generate all single-point mutants with progress bar
    with tqdm(total=expected_mutants, desc="Generating mutants") as pbar:
        for protein_name, protein_info in proteins.items():
            sequence = protein_info['sequence']
            length = protein_info['length']
            
            for position in range(length):
                wild_type_aa = sequence[position]
                
                for mutant_aa in AMINO_ACIDS:
                    if mutant_aa == wild_type_aa:
                        continue  # Skip wild-type (no mutation)
                    
                    # Create mutant sequence
                    mutant_sequence = sequence[:position] + mutant_aa + sequence[position+1:]
                    
                    # Add to dataset
                    rows.append({
                        'protein': protein_name,
                        'position': position + 1,  # 1-based indexing
                        'wt_aa': wild_type_aa,
                        'mut_aa': mutant_aa,
                        'wt_sequence': sequence,
                        'mut_sequence': mutant_sequence,
                        'protein_length': length
                    })
                    
                    pbar.update(1)
    
    # Save to CSV
    try:
        df = pd.DataFrame(rows)
        df.to_csv(OUTPUT_CSV, index=False)
        print(f"\n✅ SUCCESS!")
        print(f"Generated {len(df)} mutants across {len(proteins)} proteins")
        print(f"Saved to: {OUTPUT_CSV}")
        print(f"File size: {os.path.getsize(OUTPUT_CSV) / (1024*1024):.2f} MB")
        
        # Show sample of data
        print(f"\nFirst 5 rows:")
        print(df.head().to_string())
        
        return df
        
    except Exception as e:
        print(f"ERROR saving CSV: {e}")
        return None

if __name__ == "__main__":
    generate_all_mutants()