#!/usr/bin/env python3
"""
REAL Hotspot Detection Module
Uses actual scientific methods and databases
"""

import requests
import logging
import re
import numpy as np
from Bio import Entrez

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RealHotspotDetector:
    def __init__(self, email="your.email@domain.com"):
        """
        Initialize real hotspot detector
        Email required for NCBI queries (replace with your email)
        """
        self.email = email
        Entrez.email = email
        
        self.aggregation_propensity = {
            'A': 0.0673, 'C': 0.0359, 'D': -0.0954, 'E': -0.0820, 'F': 0.6794,
            'G': 0.0102, 'H': -0.0271, 'I': 0.7077, 'K': -0.1135, 'L': 0.5605,
            'M': 0.2631, 'N': -0.0492, 'P': -0.5448, 'Q': -0.0718, 'R': -0.1766,
            'S': -0.0359, 'T': -0.0280, 'V': 0.6107, 'W': 0.4331, 'Y': 0.3916
        }
        
        self.beta_propensity = {
            'A': 0.83, 'C': 1.19, 'D': 0.54, 'E': 0.37, 'F': 1.38,
            'G': 0.75, 'H': 0.87, 'I': 1.60, 'K': 0.74, 'L': 1.30,
            'M': 1.05, 'N': 0.89, 'P': 0.55, 'Q': 1.10, 'R': 0.93,
            'S': 0.75, 'T': 1.19, 'V': 1.70, 'W': 1.37, 'Y': 1.47
        }
        
        self.hydrophobicity = {
            'A': 1.8, 'C': 2.5, 'D': -3.5, 'E': -3.5, 'F': 2.8,
            'G': -0.4, 'H': -3.2, 'I': 4.5, 'K': -3.9, 'L': 3.8,
            'M': 1.9, 'N': -3.5, 'P': -1.6, 'Q': -3.5, 'R': -4.5,
            'S': -0.8, 'T': -0.7, 'V': 4.2, 'W': -0.9, 'Y': -1.3
        }

    def get_uniprot_data(self, protein_name):
        """Get protein data from UniProt"""
        logger.info(f"Querying UniProt for {protein_name}...")
        url = "https://rest.uniprot.org/uniprotkb/search"
        params = {
            'query': protein_name, 
            'format': 'json', 
            'size': 1
        }
        
        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            results = response.json().get('results', [])
            
            if not results:
                logger.warning(f"No UniProt results found for {protein_name}")
                return None
                
            accession = results[0]['primaryAccession']
            logger.info(f"UniProt accession: {accession}")
            
            # Get detailed information
            detail_url = f"https://rest.uniprot.org/uniprotkb/{accession}"
            detail_response = requests.get(detail_url, params={'format': 'json'}, timeout=30)
            detail_response.raise_for_status()
            
            return detail_response.json()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"UniProt request error: {e}")
            return None
        except Exception as e:
            logger.error(f"UniProt error: {e}")
            return None

    def extract_uniprot_domains(self, data):
        """Extract relevant domains from UniProt data"""
        if not data:
            return []
            
        domains = []
        features = data.get('features', [])
        
        for feature in features:
            feature_type = feature.get('type', '')
            if feature_type in ['DOMAIN', 'REGION', 'MOTIF', 'BINDING', 'COILED']:
                location = feature.get('location', {})
                start = location.get('start', {}).get('value')
                end = location.get('end', {}).get('value')
                description = feature.get('description', '')
                
                # Keywords associated with aggregation
                keywords = ['amyloid', 'aggregate', 'toxic', 'plaque', 'NAC', 'fibril']
                
                if start and end and any(keyword in description.lower() for keyword in keywords):
                    domains.append({
                        'start': start, 
                        'end': end, 
                        'source': 'uniprot',
                        'confidence': 0.8,  # High confidence for curated data
                        'description': description
                    })
                    
        logger.info(f"Found {len(domains)} relevant domains in UniProt")
        return domains

    def query_pubmed(self, protein_name, max_papers=20):
        """Query PubMed for aggregation-related papers"""
        search_term = f"{protein_name}[TIAB] AND (aggregation[TIAB] OR amyloid[TIAB])"
        
        try:
            # Search for papers
            search_handle = Entrez.esearch(
                db='pubmed', 
                term=search_term, 
                retmax=max_papers
            )
            search_results = Entrez.read(search_handle)
            search_handle.close()
            
            paper_ids = search_results['IdList']
            if not paper_ids:
                logger.info("No PubMed papers found")
                return []
                
            # Fetch abstracts
            fetch_handle = Entrez.efetch(
                db='pubmed', 
                id=paper_ids, 
                rettype='abstract', 
                retmode='xml'
            )
            articles = Entrez.read(fetch_handle)
            fetch_handle.close()
            
            regions = self._parse_abstracts(articles)
            logger.info(f"Found {len(regions)} regions from PubMed abstracts")
            return regions
            
        except Exception as e:
            logger.error(f"PubMed query error: {e}")
            return []

    def _parse_abstracts(self, articles):
        """Parse abstracts for amino acid ranges"""
        regions = []
        
        for article in articles.get('PubmedArticle', []):
            try:
                abstract_text = article['MedlineCitation']['Article'].get('Abstract', {}).get('AbstractText', '')
                
                if isinstance(abstract_text, list):
                    abstract_text = ' '.join(str(text) for text in abstract_text)
                
                pmid = str(article['MedlineCitation']['PMID'])
                
                # Look for amino acid ranges (e.g., "61-95", "residues 30-42")
                patterns = [
                    r'(?:residues?\s+)?(\d+)[-–](\d+)',
                    r'amino\s+acids?\s+(\d+)[-–](\d+)',
                    r'positions?\s+(\d+)[-–](\d+)'
                ]
                
                for pattern in patterns:
                    for match in re.finditer(pattern, abstract_text, re.IGNORECASE):
                        start, end = int(match.group(1)), int(match.group(2))
                        
                        # Validate range (reasonable protein length)
                        if start < end and end - start < 200 and start > 0:
                            regions.append({
                                'start': start, 
                                'end': end, 
                                'source': 'pubmed',
                                'confidence': 0.6,  # Medium confidence for literature
                                'pmid': pmid
                            })
                            
            except Exception as e:
                logger.warning(f"Error parsing abstract: {e}")
                continue
                
        return regions

    def compute_propensity(self, sequence, window=6):
        """Compute aggregation propensity for sequence windows"""
        if not sequence:
            return []
            
        # Clean sequence - remove non-amino acid characters
        clean_sequence = ''.join(char.upper() for char in sequence if char.upper() in self.aggregation_propensity)
        
        if len(clean_sequence) < window:
            logger.warning(f"Sequence too short ({len(clean_sequence)}) for window size {window}")
            return []
            
        propensities = []
        
        for i in range(len(clean_sequence) - window + 1):
            window_seq = clean_sequence[i:i + window]
            
            # Calculate average propensities for the window
            agg_scores = [self.aggregation_propensity.get(aa, 0) for aa in window_seq]
            beta_scores = [self.beta_propensity.get(aa, 0) for aa in window_seq]
            hydro_scores = [self.hydrophobicity.get(aa, 0) for aa in window_seq]
            
            agg_mean = np.mean(agg_scores)
            beta_mean = np.mean(beta_scores)
            hydro_mean = np.mean(hydro_scores)
            
            # Combined score (weighted average)
            combined_score = 0.5 * agg_mean + 0.3 * beta_mean + 0.2 * (hydro_mean / 5.0)
            
            propensities.append({
                'start': i + 1, 
                'end': i + window, 
                'confidence': combined_score,
                'source': 'computational'
            })
            
        logger.info(f"Computed propensity for {len(propensities)} windows")
        return propensities

    def merge_and_rank_hotspots(self, hotspots, sequence_length):
        """Merge overlapping hotspots and rank by confidence"""
        if not hotspots:
            return []
            
        # Ensure all hotspots have required fields
        for hotspot in hotspots:
            if 'source' not in hotspot:
                hotspot['source'] = 'unknown'
            if 'confidence' not in hotspot:
                hotspot['confidence'] = 0.5
                
        # Sort by start position
        hotspots.sort(key=lambda x: x['start'])
        
        merged = []
        for hotspot in hotspots:
            if not merged:
                hotspot['count'] = 1
                hotspot['sources'] = [hotspot['source']]
                merged.append(hotspot.copy())
            else:
                last = merged[-1]
                # Merge if overlapping or close (within 5 residues)
                if hotspot['start'] <= last['end'] + 5:
                    last['end'] = max(last['end'], hotspot['end'])
                    last['count'] += 1
                    last['sources'].append(hotspot['source'])
                    last['confidence'] = max(last['confidence'], hotspot['confidence'])
                else:
                    hotspot['count'] = 1
                    hotspot['sources'] = [hotspot['source']]
                    merged.append(hotspot.copy())
        
        # Calculate final confidence scores
        for hotspot in merged:
            # Boost confidence based on multiple sources
            source_bonus = min(0.2 * (hotspot['count'] - 1), 0.4)
            hotspot['final_confidence'] = min(hotspot['confidence'] + source_bonus, 1.0)
            
        # Sort by final confidence (descending)
        merged.sort(key=lambda x: x['final_confidence'], reverse=True)
        
        logger.info(f"Merged into {len(merged)} final hotspots")
        return merged

    def detect_real_hotspots(self, protein_name, sequence):
        """Main method to detect aggregation hotspots"""
        logger.info(f"Detecting hotspots for {protein_name}")
        
        # Get data from different sources
        uniprot_data = self.get_uniprot_data(protein_name)
        uniprot_domains = self.extract_uniprot_domains(uniprot_data)
        pubmed_regions = self.query_pubmed(protein_name)
        computational_propensities = self.compute_propensity(sequence)
        
        # Combine all hotspots
        all_hotspots = uniprot_domains + pubmed_regions + computational_propensities
        
        # Merge and rank
        final_hotspots = self.merge_and_rank_hotspots(all_hotspots, len(sequence))
        
        logger.info(f"Final result: {len(final_hotspots)} hotspots detected")
        return final_hotspots

    def create_position_hotspot_map(self, hotspots, sequence_length):
        """Create a binary map of hotspot positions"""
        position_map = [0] * sequence_length
        
        for hotspot in hotspots:
            start_idx = max(hotspot['start'] - 1, 0)  # Convert to 0-based indexing
            end_idx = min(hotspot['end'], sequence_length)
            
            for i in range(start_idx, end_idx):
                position_map[i] = 1
                
        return position_map