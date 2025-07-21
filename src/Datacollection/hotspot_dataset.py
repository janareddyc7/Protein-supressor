#!/usr/bin/env python3
"""
Integration guide: How to add REAL hotspot detection to your mutation dataset
"""

import pandas as pd
import json
import os  # Import the above class

def upgrade_mutation_dataset_with_real_hotspots():
    """
    Replace fake hotspots with real ones in your existing dataset
    """
    
    # Initialize real hotspot detector
    detector = RealHotspotDetector(email="avatani")  # REPLACE WITH YOUR EMAIL
    
    # Load your protein sequences
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    INPUT_JSON = os.path.join(project_root, 'data', 'proteins', 'all_sequences.json')
    OUTPUT_DIR = os.path.join(project_root, 'data', 'features')
    
    with open(INPUT_JSON, 'r') as f:
        proteins = json.load(f)
    
    print("🔬 Upgrading datasets with REAL hotspot detection...")
    
    # Process each protein
    for protein_name, protein_info in proteins.items():
        print(f"\n{'='*60}")
        print(f"Processing {protein_name}")
        print(f"{'='*60}")
        
        sequence = protein_info['sequence']
        
        # Step 1: Detect REAL hotspots
        print("🔍 Detecting real hotspots...")
        real_hotspots = detector.detect_real_hotspots(protein_name, sequence)
        
        # Step 2: Create position hotspot map
        hotspot_map = detector.create_position_hotspot_map(real_hotspots, len(sequence))
        
        # Step 3: Load existing mutation dataset
        existing_file = os.path.join(OUTPUT_DIR, f"{protein_name}_mutants_complete.csv")
        
        if not os.path.exists(existing_file):
            print(f"⚠️  No existing dataset found for {protein_name}, skipping...")
            continue
        
        df = pd.read_csv(existing_file)
        print(f"📊 Loaded {len(df)} existing mutations")
        
        # Step 4: Replace fake hotspot column with real data
        print("🔄 Replacing hotspot data...")
        
        # Add new real hotspot columns
        df['is_hotspot_region_real'] = 0
        df['hotspot_confidence_score'] = 0.0
        df['hotspot_evidence_count'] = 0
        df['hotspot_sources'] = ''
        
        for idx, row in df.iterrows():
            position = int(row['position']) - 1  # Convert to 0-based
            
            if position < len(hotspot_map):
                df.at[idx, 'is_hotspot_region_real'] = hotspot_map[position]
                
                # Find which hotspot this position belongs to
                for hotspot in real_hotspots:
                    if hotspot['start'] <= row['position'] <= hotspot['end']:
                        df.at[idx, 'hotspot_confidence_score'] = hotspot['confidence_score']
                        df.at[idx, 'hotspot_evidence_count'] = hotspot['evidence_count']
                        df.at[idx, 'hotspot_sources'] = ','.join(hotspot['source']) if isinstance(hotspot['source'], list) else hotspot['source']
                        break
        
        # Step 5: Remove old fake hotspot column
        if 'is_hotspot_region' in df.columns:
            df = df.drop('is_hotspot_region', axis=1)
        
        # Step 6: Recalculate labels with real hotspots
        print("🏷️  Recalculating labels with real hotspot data...")
        df = recalculate_labels_with_real_hotspots(df)
        
        # Step 7: Save updated dataset
        output_file = os.path.join(OUTPUT_DIR, f"{protein_name}_mutants_real_hotspots.csv")
        df.to_csv(output_file, index=False)
        
        print(f"✅ Saved updated dataset: {output_file}")
        
        # Print summary
        print_real_hotspot_summary(df, real_hotspots, protein_name)

def recalculate_labels_with_real_hotspots(df):
    """
    Recalculate aggregation labels using real hotspot data
    """
    print("🧮 Recalculating aggregation labels...")
    
    for idx, row in df.iterrows():
        # Base aggregation score
        aggregation_score = 0
        
        # Real hotspot contribution (much more reliable now)
        if row['is_hotspot_region_real'] == 1:
            # Weight by confidence - high confidence hotspots get more weight
            confidence_weight = row['hotspot_confidence_score']
            aggregation_score += 3 * confidence_weight  # Strong contribution
        
        # Other factors (keep existing logic)
        wt_aa = row['wt_aa']
        mut_aa = row['mut_aa']
        
        # Mutation to aggregation-prone amino acids
        aggregation_prone_aas = ['F', 'I', 'L', 'V', 'W', 'Y']
        if mut_aa in aggregation_prone_aas and wt_aa not in aggregation_prone_aas:
            aggregation_score += 1
        
        # Hydrophobic mutations in hotspots are particularly dangerous
        if (row['is_hotspot_region_real'] == 1 and 
            row['hydropathy_change'] > 2):
            aggregation_score += 2
        
        # Charge loss in hotspots
        if (row['is_hotspot_region_real'] == 1 and 
            row['charge_change'] < -1):
            aggregation_score += 1.5
        
        # Update aggregation_prone label (more stringent threshold now)
        df.at[idx, 'aggregation_prone'] = 1 if aggregation_score >= 3 else 0
        
        # Update main ML target
        other_pathogenic_score = (df.at[idx, 'destabilizing'] * 2 +
                                df.at[idx, 'charge_disrupting'] * 1 +
                                df.at[idx, 'hydrophobicity_disrupt'] * 1 +
                                df.at[idx, 'size_disrupt'] * 1)
        
        total_pathogenic_score = aggregation_score + other_pathogenic_score
        df.at[idx, 'ml_target'] = 1 if total_pathogenic_score >= 4 else 0
    
    return df

def print_real_hotspot_summary(df, real_hotspots, protein_name):
    """
    Print summary of real hotspot detection results
    """
    print(f"\n📈 REAL HOTSPOT SUMMARY FOR {protein_name.upper()}")
    print("-" * 50)
    
    print(f"🔥 Detected hotspot regions: {len(real_hotspots)}")
    for i, hotspot in enumerate(real_hotspots, 1):
        sources = hotspot['source'] if isinstance(hotspot['source'], list) else [hotspot['source']]
        print(f"  {i}. Positions {hotspot['start']}-{hotspot['end']}")
        print(f"     Sources: {', '.join(sources)}")
        print(f"     Confidence: {hotspot['confidence_score']:.2f}")
        print(f"     Evidence: {hotspot['evidence_count']} sources")
    
    # Statistics
    real_hotspot_mutations = df['is_hotspot_region_real'].sum()
    total_mutations = len(df)
    hotspot_percentage = (real_hotspot_mutations / total_mutations) * 100
    
    print(f"\n📊 MUTATION STATISTICS:")
    print(f"  Total mutations: {total_mutations:,}")
    print(f"  Mutations in real hotspots: {real_hotspot_mutations:,} ({hotspot_percentage:.1f}%)")
    
    # Pathogenicity in hotspots vs non-hotspots
    hotspot_pathogenic = df[df['is_hotspot_region_real'] == 1]['ml_target'].mean()
    non_hotspot_pathogenic = df[df['is_hotspot_region_real'] == 0]['ml_target'].mean()
    
    print(f"  Pathogenic rate in hotspots: {hotspot_pathogenic:.1%}")
    print(f"  Pathogenic rate outside hotspots: {non_hotspot_pathogenic:.1%}")
    print(f"  Enrichment factor: {hotspot_pathogenic/non_hotspot_pathogenic:.1f}x")
    
    # Source breakdown
    print(f"\n🔍 HOTSPOT SOURCE BREAKDOWN:")
    source_counts = {}
    for idx, row in df.iterrows():
        if row['is_hotspot_region_real'] == 1 and row['hotspot_sources']:
            sources = row['hotspot_sources'].split(',')
            for source in sources:
                source = source.strip()
                source_counts[source] = source_counts.get(source, 0) + 1
    
    for source, count in source_counts.items():
        percentage = (count / real_hotspot_mutations) * 100 if real_hotspot_mutations > 0 else 0
        print(f"  {source}: {count:,} ({percentage:.1f}%)")

def create_combined_real_dataset():
    """
    Combine all proteins with real hotspots into final dataset
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    OUTPUT_DIR = os.path.join(project_root, 'data', 'features')
    
    print("🔗 Creating combined dataset with real hotspots...")
    
    # Find all real hotspot files
    real_files = []
    for file in os.listdir(OUTPUT_DIR):
        if file.endswith('_mutants_real_hotspots.csv'):
            real_files.append(os.path.join(OUTPUT_DIR, file))
    
    if not real_files:
        print("❌ No real hotspot files found!")
        return
    
    print(f"📁 Found {len(real_files)} protein datasets with real hotspots")
    
    # Combine datasets
    combined_dfs = []
    total_mutations = 0
    protein_counts = {}
    
    for file_path in real_files:
        protein_name = os.path.basename(file_path).replace('_mutants_real_hotspots.csv', '')
        print(f"  📄 Loading {protein_name}...")
        
        df = pd.read_csv(file_path)
        combined_dfs.append(df)
        
        mutations_count = len(df)
        protein_counts[protein_name] = mutations_count
        total_mutations += mutations_count
        
        print(f"     {mutations_count:,} mutations loaded")
    
    # Concatenate all dataframes
    print(f"\n🔄 Combining {len(combined_dfs)} datasets...")
    combined_df = pd.concat(combined_dfs, ignore_index=True)
    
    # Add dataset version information
    combined_df['dataset_version'] = 'real_hotspots_v1.0'
    combined_df['creation_date'] = pd.Timestamp.now().strftime('%Y-%m-%d')
    
    # Save final combined dataset
    final_output_path = os.path.join(OUTPUT_DIR, 'combined_mutations_with_real_hotspots.csv')
    combined_df.to_csv(final_output_path, index=False)
    
    # Create metadata file
    metadata = {
        'dataset_info': {
            'name': 'Combined Protein Mutations with Real Hotspots',
            'version': 'v1.0',
            'creation_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
            'total_mutations': int(total_mutations),
            'proteins_included': len(protein_counts),
            'hotspot_detection_method': 'Multi-source real hotspot detection'
        },
        'protein_breakdown': protein_counts,
        'features': {
            'position_features': ['position', 'wt_aa', 'mut_aa'],
            'biochemical_features': ['hydropathy_change', 'charge_change', 'size_disrupt'],
            'stability_features': ['destabilizing', 'charge_disrupting', 'hydrophobicity_disrupt'],
            'real_hotspot_features': [
                'is_hotspot_region_real', 
                'hotspot_confidence_score',
                'hotspot_evidence_count', 
                'hotspot_sources'
            ],
            'target_labels': ['aggregation_prone', 'ml_target']
        }
    }
    
    metadata_path = os.path.join(OUTPUT_DIR, 'combined_mutations_real_hotspots_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Print final summary
    print_combined_dataset_summary(combined_df, final_output_path, metadata)
    
    return final_output_path, metadata_path

def print_combined_dataset_summary(df, output_path, metadata):
    """
    Print comprehensive summary of the combined real hotspot dataset
    """
    print(f"\n{'='*70}")
    print("🎉 COMBINED REAL HOTSPOT DATASET CREATED SUCCESSFULLY!")
    print(f"{'='*70}")
    
    # Basic statistics
    print(f"\n📊 DATASET OVERVIEW:")
    print(f"  📄 File: {os.path.basename(output_path)}")
    print(f"  🧬 Total mutations: {len(df):,}")
    print(f"  🔬 Proteins included: {metadata['dataset_info']['proteins_included']}")
    print(f"  📅 Created: {metadata['dataset_info']['creation_date']}")
    
    # Hotspot statistics
    real_hotspot_count = df['is_hotspot_region_real'].sum()
    hotspot_percentage = (real_hotspot_count / len(df)) * 100
    
    print(f"\n🔥 REAL HOTSPOT STATISTICS:")
    print(f"  Mutations in real hotspots: {real_hotspot_count:,} ({hotspot_percentage:.1f}%)")
    print(f"  Mutations outside hotspots: {len(df) - real_hotspot_count:,} ({100-hotspot_percentage:.1f}%)")
    
    # Confidence score distribution
    hotspot_df = df[df['is_hotspot_region_real'] == 1]
    if len(hotspot_df) > 0:
        avg_confidence = hotspot_df['hotspot_confidence_score'].mean()
        high_conf_count = (hotspot_df['hotspot_confidence_score'] >= 0.8).sum()
        med_conf_count = ((hotspot_df['hotspot_confidence_score'] >= 0.5) & 
                         (hotspot_df['hotspot_confidence_score'] < 0.8)).sum()
        low_conf_count = (hotspot_df['hotspot_confidence_score'] < 0.5).sum()
        
        print(f"\n📈 HOTSPOT CONFIDENCE DISTRIBUTION:")
        print(f"  Average confidence: {avg_confidence:.3f}")
        print(f"  High confidence (≥0.8): {high_conf_count:,} ({high_conf_count/len(hotspot_df)*100:.1f}%)")
        print(f"  Medium confidence (0.5-0.8): {med_conf_count:,} ({med_conf_count/len(hotspot_df)*100:.1f}%)")
        print(f"  Low confidence (<0.5): {low_conf_count:,} ({low_conf_count/len(hotspot_df)*100:.1f}%)")
    
    # Target label distribution
    print(f"\n🏷️  TARGET LABEL DISTRIBUTION:")
    ml_target_dist = df['ml_target'].value_counts()
    aggregation_dist = df['aggregation_prone'].value_counts()
    
    print(f"  ML Target (pathogenic):")
    print(f"    Pathogenic (1): {ml_target_dist.get(1, 0):,} ({ml_target_dist.get(1, 0)/len(df)*100:.1f}%)")
    print(f"    Benign (0): {ml_target_dist.get(0, 0):,} ({ml_target_dist.get(0, 0)/len(df)*100:.1f}%)")
    
    print(f"  Aggregation Prone:")
    print(f"    Prone (1): {aggregation_dist.get(1, 0):,} ({aggregation_dist.get(1, 0)/len(df)*100:.1f}%)")
    print(f"    Not prone (0): {aggregation_dist.get(0, 0):,} ({aggregation_dist.get(0, 0)/len(df)*100:.1f}%)")
    
    # Hotspot vs pathogenicity analysis
    print(f"\n🔍 HOTSPOT ENRICHMENT ANALYSIS:")
    
    # ML target enrichment
    hotspot_pathogenic_rate = df[df['is_hotspot_region_real'] == 1]['ml_target'].mean()
    non_hotspot_pathogenic_rate = df[df['is_hotspot_region_real'] == 0]['ml_target'].mean()
    ml_enrichment = hotspot_pathogenic_rate / non_hotspot_pathogenic_rate if non_hotspot_pathogenic_rate > 0 else float('inf')
    
    # Aggregation enrichment
    hotspot_agg_rate = df[df['is_hotspot_region_real'] == 1]['aggregation_prone'].mean()
    non_hotspot_agg_rate = df[df['is_hotspot_region_real'] == 0]['aggregation_prone'].mean()
    agg_enrichment = hotspot_agg_rate / non_hotspot_agg_rate if non_hotspot_agg_rate > 0 else float('inf')
    
    print(f"  Pathogenic mutations:")
    print(f"    In hotspots: {hotspot_pathogenic_rate:.1%}")
    print(f"    Outside hotspots: {non_hotspot_pathogenic_rate:.1%}")
    print(f"    Enrichment factor: {ml_enrichment:.1f}x")
    
    print(f"  Aggregation-prone mutations:")
    print(f"    In hotspots: {hotspot_agg_rate:.1%}")
    print(f"    Outside hotspots: {non_hotspot_agg_rate:.1%}")
    print(f"    Enrichment factor: {agg_enrichment:.1f}x")
    
    # Data quality metrics
    print(f"\n✅ DATA QUALITY METRICS:")
    missing_values = df.isnull().sum().sum()
    complete_features = (~df[['is_hotspot_region_real', 'hotspot_confidence_score', 
                             'ml_target', 'aggregation_prone']].isnull()).all(axis=1).sum()
    
    print(f"  Missing values: {missing_values}")
    print(f"  Complete feature rows: {complete_features:,} ({complete_features/len(df)*100:.1f}%)")
    print(f"  Average evidence per hotspot mutation: {hotspot_df['hotspot_evidence_count'].mean():.1f}")
    
    # Per-protein breakdown
    print(f"\n🧬 PER-PROTEIN BREAKDOWN:")
    for protein, count in metadata['protein_breakdown'].items():
        protein_df = df[df['protein_name'] == protein] if 'protein_name' in df.columns else None
        if protein_df is not None and len(protein_df) > 0:
            hotspots_in_protein = protein_df['is_hotspot_region_real'].sum()
            pathogenic_in_protein = protein_df['ml_target'].sum()
            print(f"  {protein}: {count:,} mutations ({hotspots_in_protein} in hotspots, {pathogenic_in_protein} pathogenic)")
        else:
            print(f"  {protein}: {count:,} mutations")
    
    print(f"\n💾 FILES CREATED:")
    print(f"  📊 Main dataset: {output_path}")
    print(f"  📋 Metadata: {output_path.replace('.csv', '_metadata.json')}")
    
    print(f"\n🚀 Ready for machine learning! Use this dataset for:")
    print("   • Training pathogenicity prediction models")
    print("   • Aggregation propensity analysis")
    print("   • Hotspot-based feature engineering")
    print("   • Cross-protein generalization studies")

def validate_real_hotspot_dataset(dataset_path):
    """
    Validate the quality of the real hotspot dataset
    """
    print("🔍 Validating real hotspot dataset quality...")
    
    df = pd.read_csv(dataset_path)
    
    validation_results = {
        'total_mutations': len(df),
        'data_quality': {},
        'hotspot_quality': {},
        'target_balance': {},
        'warnings': [],
        'errors': []
    }
    
    # Data quality checks
    missing_critical = df[['position', 'wt_aa', 'mut_aa', 'is_hotspot_region_real', 'ml_target']].isnull().any(axis=1).sum()
    validation_results['data_quality']['missing_critical_features'] = missing_critical
    
    if missing_critical > 0:
        validation_results['errors'].append(f"Found {missing_critical} rows with missing critical features")
    
    # Hotspot quality checks
    hotspot_mutations = df[df['is_hotspot_region_real'] == 1]
    validation_results['hotspot_quality']['mutations_in_hotspots'] = len(hotspot_mutations)
    
    if len(hotspot_mutations) > 0:
        avg_confidence = hotspot_mutations['hotspot_confidence_score'].mean()
        low_confidence_count = (hotspot_mutations['hotspot_confidence_score'] < 0.3).sum()
        
        validation_results['hotspot_quality']['average_confidence'] = avg_confidence
        validation_results['hotspot_quality']['low_confidence_count'] = low_confidence_count
        
        if avg_confidence < 0.5:
            validation_results['warnings'].append("Average hotspot confidence is below 0.5")
        
        if low_confidence_count > len(hotspot_mutations) * 0.2:
            validation_results['warnings'].append("More than 20% of hotspot mutations have very low confidence")
    
    # Target balance checks
    pathogenic_rate = df['ml_target'].mean()
    aggregation_rate = df['aggregation_prone'].mean()
    
    validation_results['target_balance']['pathogenic_rate'] = pathogenic_rate
    validation_results['target_balance']['aggregation_rate'] = aggregation_rate
    
    if pathogenic_rate < 0.1 or pathogenic_rate > 0.9:
        validation_results['warnings'].append(f"Extreme class imbalance in pathogenic labels: {pathogenic_rate:.1%}")
    
    # Print validation report
    print(f"\n📋 VALIDATION REPORT")
    print("-" * 40)
    print(f"✅ Total mutations validated: {validation_results['total_mutations']:,}")
    print(f"✅ Missing critical features: {validation_results['data_quality']['missing_critical_features']}")
    print(f"✅ Mutations in hotspots: {validation_results['hotspot_quality']['mutations_in_hotspots']:,}")
    
    if 'average_confidence' in validation_results['hotspot_quality']:
        print(f"✅ Average hotspot confidence: {validation_results['hotspot_quality']['average_confidence']:.3f}")
    
    print(f"✅ Pathogenic rate: {validation_results['target_balance']['pathogenic_rate']:.1%}")
    print(f"✅ Aggregation rate: {validation_results['target_balance']['aggregation_rate']:.1%}")
    
    if validation_results['warnings']:
        print(f"\n⚠️  WARNINGS ({len(validation_results['warnings'])}):")
        for warning in validation_results['warnings']:
            print(f"  • {warning}")
    
    if validation_results['errors']:
        print(f"\n❌ ERRORS ({len(validation_results['errors'])}):")
        for error in validation_results['errors']:
            print(f"  • {error}")
    else:
        print(f"\n🎉 No critical errors found!")
    
    return validation_results

def main():
    """
    Main function to orchestrate the real hotspot integration process
    """
    print("🧬 STARTING REAL HOTSPOT INTEGRATION PIPELINE")
    print("=" * 60)
    
    try:
        # Step 1: Upgrade individual protein datasets
        print("📋 STEP 1: Upgrading individual protein datasets with real hotspots...")
        upgrade_mutation_dataset_with_real_hotspots()
        
        # Step 2: Create combined dataset
        print(f"\n📋 STEP 2: Creating combined dataset...")
        final_dataset_path, metadata_path = create_combined_real_dataset()
        
        # Step 3: Validate the final dataset
        print(f"\n📋 STEP 3: Validating final dataset quality...")
        validation_results = validate_real_hotspot_dataset(final_dataset_path)
        
        # Final success message
        print(f"\n{'='*60}")
        print("🎉 REAL HOTSPOT INTEGRATION COMPLETED SUCCESSFULLY!")
        print(f"{'='*60}")
        print(f"📊 Final dataset: {final_dataset_path}")
        print(f"📋 Metadata: {metadata_path}")
        print(f"🔬 Total mutations: {validation_results['total_mutations']:,}")
        print(f"🔥 Mutations in real hotspots: {validation_results['hotspot_quality']['mutations_in_hotspots']:,}")
        
        if not validation_results['errors']:
            print("✅ Dataset passed all quality checks!")
        else:
            print("⚠️  Dataset has some issues - please review validation report")
            
    except Exception as e:
        print(f"\n❌ PIPELINE FAILED: {str(e)}")
        print("Please check the error details and try again.")
        raise

if __name__ == "__main__":
    main()