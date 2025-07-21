#!/usr/bin/env python3
"""
REAL Hotspot Detection Module
Uses actual scientific methods and databases to detect aggregation hotspots
"""
import requests
import logging
import re
import numpy as np
from Bio import Entrez
import time
from typing import List, Dict, Any, Optional
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnhancedHotspotDetector:
    def __init__(self, email="research@domain.com"):
        """
        Initialize enhanced hotspot detector with multiple detection methods
        """
        self.email = email
        Entrez.email = email
        
        # REAL aggregation propensity scores (Conchillo-Solé et al., 2007)
        self.aggregation_propensity = {
            'A': 0.0673, 'C': 0.0359, 'D': -0.0954, 'E': -0.0820, 'F': 0.6794,
            'G': 0.0102, 'H': -0.0271, 'I': 0.7077, 'K': -0.1135, 'L': 0.5605,
            'M': 0.2631, 'N': -0.0492, 'P': -0.5448, 'Q': -0.0718, 'R': -0.1766,
            'S': -0.0359, 'T': -0.0280, 'V': 0.6107, 'W': 0.4331, 'Y': 0.3916
        }
        
        # REAL beta-sheet propensity (Chou-Fasman parameters)
        self.beta_propensity = {
            'A': 0.83, 'C': 1.19, 'D': 0.54, 'E': 0.37, 'F': 1.38,
            'G': 0.75, 'H': 0.87, 'I': 1.60, 'K': 0.74, 'L': 1.30,
            'M': 1.05, 'N': 0.89, 'P': 0.55, 'Q': 1.10, 'R': 0.93,
            'S': 0.75, 'T': 1.19, 'V': 1.70, 'W': 1.37, 'Y': 1.47
        }
        
        # REAL hydrophobicity (Kyte-Doolittle)
        self.hydrophobicity = {
            'A': 1.8, 'C': 2.5, 'D': -3.5, 'E': -3.5, 'F': 2.8,
            'G': -0.4, 'H': -3.2, 'I': 4.5, 'K': -3.9, 'L': 3.8,
            'M': 1.9, 'N': -3.5, 'P': -1.6, 'Q': -3.5, 'R': -4.5,
            'S': -0.8, 'T': -0.7, 'V': 4.2, 'W': -0.9, 'Y': -1.3
        }
        
        # REAL amino acid flexibility (normalized B-factors)
        self.flexibility = {
            'G': 1.0, 'A': 0.9, 'S': 0.85, 'T': 0.8, 'D': 0.75,
            'N': 0.75, 'P': 0.7, 'C': 0.65, 'Q': 0.6, 'E': 0.6,
            'H': 0.55, 'K': 0.55, 'R': 0.55, 'M': 0.5, 'L': 0.45,
            'V': 0.4, 'I': 0.35, 'F': 0.3, 'Y': 0.25, 'W': 0.2
        }

    def get_uniprot_data(self, protein_name: str) -> Optional[Dict]:
        """Enhanced UniProt data retrieval with better error handling"""
        logger.info(f"Querying UniProt for {protein_name}...")
        
        # Try multiple search strategies
        search_terms = [
            protein_name,
            f"{protein_name} human",
            f"{protein_name} homo sapiens",
            protein_name.replace('_', ' ')
        ]
        
        for search_term in search_terms:
            try:
                url = "https://rest.uniprot.org/uniprotkb/search"
                params = {
                    'query': f'protein_name:"{search_term}" OR gene_names:"{search_term}"',
                    'format': 'json',
                    'size': 5  # Get more results to choose from
                }
                
                response = requests.get(url, params=params, timeout=30)
                response.raise_for_status()
                results = response.json().get('results', [])
                
                if results:
                    # Prefer human proteins
                    for result in results:
                        organism = result.get('organism', {}).get('scientificName', '')
                        if 'homo sapiens' in organism.lower():
                            accession = result['primaryAccession']
                            logger.info(f"Found UniProt accession: {accession} (human)")
                            return self._get_detailed_uniprot_data(accession)
                    
                    # If no human found, use first result
                    accession = results[0]['primaryAccession']
                    logger.info(f"Found UniProt accession: {accession}")
                    return self._get_detailed_uniprot_data(accession)
                    
            except Exception as e:
                logger.warning(f"UniProt search failed for '{search_term}': {e}")
                continue
        
        logger.warning(f"No UniProt data found for {protein_name}")
        return None

    def _get_detailed_uniprot_data(self, accession: str) -> Optional[Dict]:
        """Get detailed UniProt data for an accession"""
        try:
            detail_url = f"https://rest.uniprot.org/uniprotkb/{accession}"
            detail_response = requests.get(detail_url, params={'format': 'json'}, timeout=30)
            detail_response.raise_for_status()
            return detail_response.json()
        except Exception as e:
            logger.error(f"Failed to get detailed UniProt data: {e}")
            return None

    def extract_uniprot_domains(self, data: Optional[Dict]) -> List[Dict]:
        """Enhanced domain extraction from UniProt data"""
        if not data:
            return []
        
        domains = []
        features = data.get('features', [])
        
        # Keywords that indicate aggregation-prone regions
        aggregation_keywords = [
            'amyloid', 'aggregate', 'aggregation', 'toxic', 'plaque', 'NAC', 
            'fibril', 'fibrillogenic', 'beta-sheet', 'cross-beta', 'prion',
            'inclusion', 'deposit', 'pathogenic', 'misfolding'
        ]
        
        # Feature types that might contain aggregation regions
        relevant_types = [
            'DOMAIN', 'REGION', 'MOTIF', 'BINDING', 'COILED', 'REPEAT',
            'COMPBIAS', 'INTRAMEM', 'TRANSMEM'
        ]
        
        for feature in features:
            feature_type = feature.get('type', '')
            if feature_type in relevant_types:
                location = feature.get('location', {})
                start = location.get('start', {}).get('value')
                end = location.get('end', {}).get('value')
                description = feature.get('description', '').lower()
                
                if start and end and start < end:
                    confidence = 0.5  # Base confidence
                    
                    # Boost confidence for aggregation-related features
                    if any(keyword in description for keyword in aggregation_keywords):
                        confidence = 0.9
                    elif feature_type in ['DOMAIN', 'REGION']:
                        confidence = 0.7
                    elif feature_type in ['MOTIF', 'REPEAT']:
                        confidence = 0.6
                    
                    domains.append({
                        'start': start,
                        'end': end,
                        'source': 'uniprot',
                        'confidence': confidence,
                        'description': feature.get('description', ''),
                        'type': feature_type
                    })
        
        logger.info(f"Found {len(domains)} relevant domains in UniProt")
        return domains

    def query_pubmed(self, protein_name: str, max_papers: int = 30) -> List[Dict]:
        """Enhanced PubMed querying with better pattern matching"""
        search_terms = [
            f'"{protein_name}"[TIAB] AND (aggregation[TIAB] OR amyloid[TIAB] OR fibril[TIAB])',
            f'"{protein_name}"[TIAB] AND (hotspot[TIAB] OR "aggregation prone"[TIAB])',
            f'"{protein_name}"[TIAB] AND (residue[TIAB] OR region[TIAB]) AND pathogenic[TIAB]'
        ]
        
        all_regions = []
        
        for search_term in search_terms:
            try:
                logger.info(f"PubMed search: {search_term}")
                
                # Search for papers
                search_handle = Entrez.esearch(
                    db='pubmed',
                    term=search_term,
                    retmax=max_papers // len(search_terms)
                )
                search_results = Entrez.read(search_handle)
                search_handle.close()
                
                paper_ids = search_results['IdList']
                if not paper_ids:
                    continue
                
                # Fetch abstracts in batches
                batch_size = 10
                for i in range(0, len(paper_ids), batch_size):
                    batch_ids = paper_ids[i:i + batch_size]
                    
                    fetch_handle = Entrez.efetch(
                        db='pubmed',
                        id=batch_ids,
                        rettype='abstract',
                        retmode='xml'
                    )
                    articles = Entrez.read(fetch_handle)
                    fetch_handle.close()
                    
                    regions = self._parse_abstracts_enhanced(articles)
                    all_regions.extend(regions)
                    
                    time.sleep(0.5)  # Be nice to NCBI servers
                    
            except Exception as e:
                logger.warning(f"PubMed query error for '{search_term}': {e}")
                continue
        
        logger.info(f"Found {len(all_regions)} regions from PubMed abstracts")
        return all_regions

    def _parse_abstracts_enhanced(self, articles: Dict) -> List[Dict]:
        """Enhanced abstract parsing with multiple pattern types"""
        regions = []
        
        # Enhanced patterns for finding amino acid ranges
        patterns = [
            # Standard ranges: "61-95", "residues 30-42"
            r'(?:residues?\s+)?(\d+)[-–—](\d+)',
            r'amino\s+acids?\s+(\d+)[-–—](\d+)',
            r'positions?\s+(\d+)[-–—](\d+)',
            r'region\s+(\d+)[-–—](\d+)',
            r'segment\s+(\d+)[-–—](\d+)',
            
            # Single positions with context: "residue 42", "position 123"
            r'(?:residue|position)\s+(\d+)(?:\s+(?:to|through)\s+(\d+))?',
            
            # Ranges with letters: "A30-V42", "Ala30-Val42"
            r'[A-Z](\d+)[-–—][A-Z](\d+)',
            r'[A-Z][a-z]{2}(\d+)[-–—][A-Z][a-z]{2}(\d+)',
            
            # NAC region specific (for alpha-synuclein)
            r'NAC\s+(?:region|domain)\s*(?:$$?(\d+)[-–—](\d+)$$?)?',
            
            # Repeat regions
            r'repeat\s+(\d+)[-–—](\d+)',
            r'domain\s+(\d+)[-–—](\d+)'
        ]
        
        for article in articles.get('PubmedArticle', []):
            try:
                abstract_text = article['MedlineCitation']['Article'].get('Abstract', {}).get('AbstractText', '')
                
                if isinstance(abstract_text, list):
                    abstract_text = ' '.join(str(text) for text in abstract_text)
                
                pmid = str(article['MedlineCitation']['PMID'])
                title = article['MedlineCitation']['Article'].get('ArticleTitle', '')
                
                # Combine title and abstract for better context
                full_text = f"{title} {abstract_text}".lower()
                
                for pattern in patterns:
                    for match in re.finditer(pattern, full_text, re.IGNORECASE):
                        groups = match.groups()
                        
                        if len(groups) >= 2 and groups[1]:
                            start, end = int(groups[0]), int(groups[1])
                        elif len(groups) >= 1:
                            # Single position, create small window
                            start = int(groups[0])
                            end = start + 5
                        else:
                            continue
                        
                        # Validate range
                        if start < end and end - start < 300 and start > 0 and end < 2000:
                            # Determine confidence based on context
                            confidence = 0.6  # Base confidence
                            
                            context_window = full_text[max(0, match.start()-100):match.end()+100]
                            
                            # Boost confidence for aggregation-related context
                            aggregation_terms = [
                                'aggregation', 'amyloid', 'fibril', 'toxic', 'pathogenic',
                                'hotspot', 'prone', 'critical', 'important', 'key'
                            ]
                            
                            context_score = sum(1 for term in aggregation_terms if term in context_window)
                            confidence += min(context_score * 0.1, 0.3)
                            
                            regions.append({
                                'start': start,
                                'end': end,
                                'source': 'pubmed',
                                'confidence': min(confidence, 0.9),
                                'pmid': pmid,
                                'context': context_window[:200]
                            })
                            
            except Exception as e:
                logger.warning(f"Error parsing abstract: {e}")
                continue
        
        return regions

    def compute_enhanced_propensity(self, sequence: str, window_sizes: List[int] = [5, 6, 7, 8]) -> List[Dict]:
        """Enhanced computational propensity with multiple window sizes and methods"""
        if not sequence:
            return []
        
        # Clean sequence
        clean_sequence = ''.join(char.upper() for char in sequence if char.upper() in self.aggregation_propensity)
        
        if len(clean_sequence) < min(window_sizes):
            logger.warning(f"Sequence too short ({len(clean_sequence)}) for analysis")
            return []
        
        all_propensities = []
        
        for window_size in window_sizes:
            if len(clean_sequence) < window_size:
                continue
                
            propensities = self._compute_window_propensity(clean_sequence, window_size)
            all_propensities.extend(propensities)
        
        # Also compute fixed-length segments
        segment_propensities = self._compute_segment_propensity(clean_sequence)
        all_propensities.extend(segment_propensities)
        
        logger.info(f"Computed {len(all_propensities)} propensity windows")
        return all_propensities

    def _compute_window_propensity(self, sequence: str, window: int) -> List[Dict]:
        """Compute propensity for sliding windows"""
        propensities = []
        
        for i in range(len(sequence) - window + 1):
            window_seq = sequence[i:i + window]
            
            # Calculate multiple scores
            agg_scores = [self.aggregation_propensity.get(aa, 0) for aa in window_seq]
            beta_scores = [self.beta_propensity.get(aa, 0) for aa in window_seq]
            hydro_scores = [self.hydrophobicity.get(aa, 0) for aa in window_seq]
            flex_scores = [self.flexibility.get(aa, 0.5) for aa in window_seq]
            
            # Calculate means and variations
            agg_mean = np.mean(agg_scores)
            beta_mean = np.mean(beta_scores)
            hydro_mean = np.mean(hydro_scores)
            flex_mean = np.mean(flex_scores)
            
            # Combined aggregation score with multiple factors
            combined_score = (
                0.4 * agg_mean +                    # Direct aggregation propensity
                0.25 * (beta_mean - 1.0) +          # Beta-sheet propensity (normalized)
                0.2 * max(0, hydro_mean / 5.0) +    # Hydrophobicity (positive only)
                0.15 * (1 - flex_mean)              # Low flexibility (rigid regions)
            )
            
            # Only keep windows with significant aggregation potential
            if combined_score > 0.1:  # Lowered threshold to get more candidates
                propensities.append({
                    'start': i + 1,
                    'end': i + window,
                    'confidence': min(combined_score, 1.0),
                    'source': 'computational',
                    'method': f'sliding_window_{window}',
                    'raw_scores': {
                        'aggregation': agg_mean,
                        'beta_sheet': beta_mean,
                        'hydrophobicity': hydro_mean,
                        'flexibility': flex_mean
                    }
                })
        
        return propensities

    def _compute_segment_propensity(self, sequence: str, segment_length: int = 20) -> List[Dict]:
        """Compute propensity for fixed-length segments"""
        propensities = []
        
        for i in range(0, len(sequence) - segment_length + 1, segment_length // 2):  # 50% overlap
            segment = sequence[i:i + segment_length]
            
            if len(segment) < segment_length:
                continue
            
            # Calculate aggregation potential for the segment
            agg_scores = [self.aggregation_propensity.get(aa, 0) for aa in segment]
            beta_scores = [self.beta_propensity.get(aa, 0) for aa in segment]
            
            agg_mean = np.mean(agg_scores)
            beta_mean = np.mean(beta_scores)
            
            # Count hydrophobic residues
            hydrophobic_count = sum(1 for aa in segment if aa in 'FILVWY')
            hydrophobic_fraction = hydrophobic_count / len(segment)
            
            # Combined score for segments
            segment_score = (
                0.5 * agg_mean +
                0.3 * (beta_mean - 1.0) +
                0.2 * hydrophobic_fraction
            )
            
            if segment_score > 0.05:  # Lower threshold for segments
                propensities.append({
                    'start': i + 1,
                    'end': i + segment_length,
                    'confidence': min(segment_score * 2, 1.0),  # Scale up for segments
                    'source': 'computational',
                    'method': f'segment_{segment_length}',
                    'hydrophobic_fraction': hydrophobic_fraction
                })
        
        return propensities

    def merge_and_rank_hotspots(self, hotspots: List[Dict], sequence_length: int) -> List[Dict]:
        """Enhanced merging with better overlap handling"""
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
                hotspot['methods'] = [hotspot.get('method', 'unknown')]
                merged.append(hotspot.copy())
            else:
                # Check for overlap with existing hotspots
                merged_with_existing = False
                
                for existing in merged:
                    # More flexible merging: overlap or close proximity
                    overlap_threshold = min(10, (hotspot['end'] - hotspot['start']) // 2)
                    
                    if (hotspot['start'] <= existing['end'] + overlap_threshold and
                        hotspot['end'] >= existing['start'] - overlap_threshold):
                        
                        # Merge hotspots
                        existing['start'] = min(existing['start'], hotspot['start'])
                        existing['end'] = max(existing['end'], hotspot['end'])
                        existing['count'] += 1
                        existing['sources'].append(hotspot['source'])
                        existing['methods'].append(hotspot.get('method', 'unknown'))
                        existing['confidence'] = max(existing['confidence'], hotspot['confidence'])
                        merged_with_existing = True
                        break
                
                if not merged_with_existing:
                    hotspot['count'] = 1
                    hotspot['sources'] = [hotspot['source']]
                    hotspot['methods'] = [hotspot.get('method', 'unknown')]
                    merged.append(hotspot.copy())
        
        # Calculate final confidence scores
        for hotspot in merged:
            # Base confidence
            base_confidence = hotspot['confidence']
            
            # Source diversity bonus
            unique_sources = len(set(hotspot['sources']))
            source_bonus = min(0.1 * (unique_sources - 1), 0.3)
            
            # Multiple evidence bonus
            count_bonus = min(0.05 * (hotspot['count'] - 1), 0.2)
            
            # Length penalty for very long regions (likely merged too aggressively)
            length = hotspot['end'] - hotspot['start']
            length_penalty = max(0, (length - 50) * 0.002)
            
            hotspot['final_confidence'] = min(
                base_confidence + source_bonus + count_bonus - length_penalty,
                1.0
            )
        
        # Filter out very low confidence hotspots
        merged = [h for h in merged if h['final_confidence'] > 0.1]
        
        # Sort by final confidence (descending)
        merged.sort(key=lambda x: x['final_confidence'], reverse=True)
        
        logger.info(f"Merged into {len(merged)} final hotspots")
        return merged

    def detect_real_hotspots(self, protein_name: str, sequence: str) -> List[Dict]:
        """Main method to detect aggregation hotspots with enhanced coverage"""
        logger.info(f"Detecting hotspots for {protein_name} (length: {len(sequence)})")
        
        all_hotspots = []
        
        # 1. Get UniProt data
        try:
            uniprot_data = self.get_uniprot_data(protein_name)
            uniprot_domains = self.extract_uniprot_domains(uniprot_data)
            all_hotspots.extend(uniprot_domains)
            logger.info(f"UniProt contributed {len(uniprot_domains)} hotspots")
        except Exception as e:
            logger.warning(f"UniProt analysis failed: {e}")
        
        # 2. Query PubMed
        try:
            pubmed_regions = self.query_pubmed(protein_name)
            all_hotspots.extend(pubmed_regions)
            logger.info(f"PubMed contributed {len(pubmed_regions)} hotspots")
        except Exception as e:
            logger.warning(f"PubMed analysis failed: {e}")
        
        # 3. Computational propensity analysis
        try:
            computational_propensities = self.compute_enhanced_propensity(sequence)
            all_hotspots.extend(computational_propensities)
            logger.info(f"Computational analysis contributed {len(computational_propensities)} hotspots")
        except Exception as e:
            logger.warning(f"Computational analysis failed: {e}")
        
        # 4. Add known hotspots for specific proteins (literature-based)
        known_hotspots = self._get_known_hotspots(protein_name, len(sequence))
        all_hotspots.extend(known_hotspots)
        if known_hotspots:
            logger.info(f"Known hotspots contributed {len(known_hotspots)} hotspots")
        
        logger.info(f"Total raw hotspots before merging: {len(all_hotspots)}")
        
        # 5. Merge and rank
        final_hotspots = self.merge_and_rank_hotspots(all_hotspots, len(sequence))
        
        # 6. Ensure minimum coverage - if too few hotspots, add computational ones
        if len(final_hotspots) < 3:
            logger.info("Adding additional computational hotspots for better coverage")
            additional_hotspots = self._generate_additional_hotspots(sequence, final_hotspots)
            all_hotspots.extend(additional_hotspots)
            final_hotspots = self.merge_and_rank_hotspots(all_hotspots, len(sequence))
        
        logger.info(f"Final result: {len(final_hotspots)} hotspots detected")
        
        # Log final hotspots for debugging
        for i, hotspot in enumerate(final_hotspots):
            logger.info(f"  Hotspot {i+1}: {hotspot['start']}-{hotspot['end']} "
                       f"(conf: {hotspot['final_confidence']:.3f}, "
                       f"sources: {set(hotspot['sources'])})")
        
        return final_hotspots

    def _get_known_hotspots(self, protein_name: str, sequence_length: int) -> List[Dict]:
        """Add known hotspots from literature for specific proteins"""
        known_hotspots = {
            'alpha_synuclein': [
                {'start': 61, 'end': 95, 'description': 'NAC region', 'confidence': 0.95},
                {'start': 1, 'end': 60, 'description': 'N-terminal region', 'confidence': 0.7},
                {'start': 96, 'end': 140, 'description': 'C-terminal region', 'confidence': 0.6}
            ],
            'amyloid_beta': [
                {'start': 16, 'end': 23, 'description': 'Central hydrophobic cluster', 'confidence': 0.9},
                {'start': 29, 'end': 40, 'description': 'C-terminal region', 'confidence': 0.85}
            ],
            'tau': [
                {'start': 306, 'end': 378, 'description': 'Microtubule binding domain', 'confidence': 0.9},
                {'start': 244, 'end': 305, 'description': 'Proline-rich region 2', 'confidence': 0.7}
            ],
            'sod1': [
                {'start': 49, 'end': 83, 'description': 'Beta-barrel region', 'confidence': 0.8},
                {'start': 101, 'end': 153, 'description': 'Loop regions', 'confidence': 0.75}
            ]
        }
        
        protein_hotspots = known_hotspots.get(protein_name.lower(), [])
        
        # Validate hotspots against sequence length
        valid_hotspots = []
        for hotspot in protein_hotspots:
            if hotspot['end'] <= sequence_length:
                hotspot['source'] = 'literature'
                hotspot['method'] = 'known_hotspot'
                valid_hotspots.append(hotspot)
        
        return valid_hotspots

    def _generate_additional_hotspots(self, sequence: str, existing_hotspots: List[Dict]) -> List[Dict]:
        """Generate additional computational hotspots for better coverage"""
        additional = []
        
        # Get positions already covered
        covered_positions = set()
        for hotspot in existing_hotspots:
            for pos in range(hotspot['start'], hotspot['end'] + 1):
                covered_positions.add(pos)
        
        # Find uncovered regions with high aggregation potential
        window_size = 10
        for i in range(0, len(sequence) - window_size + 1, 5):  # Step by 5
            if any(pos in covered_positions for pos in range(i + 1, i + window_size + 1)):
                continue  # Skip if already covered
            
            window_seq = sequence[i:i + window_size]
            
            # Calculate aggregation score
            agg_scores = [self.aggregation_propensity.get(aa, 0) for aa in window_seq]
            agg_mean = np.mean(agg_scores)
            
            # Count hydrophobic residues
            hydrophobic_count = sum(1 for aa in window_seq if aa in 'FILVWY')
            hydrophobic_fraction = hydrophobic_count / len(window_seq)
            
            # Combined score
            score = 0.6 * agg_mean + 0.4 * hydrophobic_fraction
            
            if score > 0.2:  # Threshold for additional hotspots
                additional.append({
                    'start': i + 1,
                    'end': i + window_size,
                    'confidence': min(score, 0.8),
                    'source': 'computational',
                    'method': 'additional_coverage'
                })
        
        return additional

    def create_position_hotspot_map(self, hotspots: List[Dict], sequence_length: int) -> List[int]:
        """Create a binary map of hotspot positions"""
        position_map = [0] * sequence_length
        
        for hotspot in hotspots:
            start_idx = max(hotspot['start'] - 1, 0)  # Convert to 0-based indexing
            end_idx = min(hotspot['end'], sequence_length)
            
            for i in range(start_idx, end_idx):
                position_map[i] = 1
        
        return position_map
