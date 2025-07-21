import requests
import os
import json

def get_protein_sequence(uniprot_id):
    url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.fasta"
    response = requests.get(url)
    lines = response.text.strip().split('\n')
    sequence = ''.join(lines[1:])  # Skip the header line
    return sequence

def save_protein_sequences():
    proteins = {
        'alpha_synuclein': 'P37840',
        'tau': 'P10636', 
        'sod1': 'P00441',
        'amyloid_beta': 'P05067'
    }
    
    # Create data directory if it doesn't exist
    os.makedirs('data/proteins', exist_ok=True)
    
    sequences = {}
    for name, uniprot_id in proteins.items():
        print(f"Downloading {name}...")
        sequence = get_protein_sequence(uniprot_id)
        sequences[name] = {
            'uniprot_id': uniprot_id,
            'sequence': sequence,
            'length': len(sequence)
        }
        
        # Save individual FASTA file
        with open(f'data/proteins/{name}.fasta', 'w') as f:
            f.write(f">{name}|{uniprot_id}\n{sequence}\n")
    
    # Save all sequences as JSON
    with open('data/proteins/all_sequences.json', 'w') as f:
        json.dump(sequences, f, indent=2)
    
    print(f"Saved {len(sequences)} protein sequences to data/proteins/")
    return sequences

# Run this
if __name__ == "__main__":
    save_protein_sequences()