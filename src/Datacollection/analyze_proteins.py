from real_hotspot import RealHotspotDetector

def main():
    detector = RealHotspotDetector(email="avatanishq00@gmail.com")  # Replace with your email
    
    proteins = {
        'alpha_synuclein': 'MDVFMKGLSKAKEGVVAAAEKTKQGVAEAAGKTKEGVLYVGSKTKEGVVHGVATVAEKTKEQVTNVGGAVVTGVTAVAQKTVEGAGSIAAATGFVKKDQLGKNEEGAPQEGILEDMPVDPDNEAYEMPSEEGYQDYEPEA',
        'amyloid_beta': 'DAEFRHDSGYEVHHQKLVFFAEDVGSNKGAIIGLMVGGVV'
    }
    
    for name, seq in proteins.items():
        print("\n" + "="*50)
        print(f"Analyzing {name}")
        print("="*50)
        
        try:
            hotspots = detector.detect_real_hotspots(name, seq)
            print(f"\nFound {len(hotspots)} hotspot regions:")
            
            for idx, h in enumerate(hotspots, 1):
                # Handle both old and new field names
                confidence = h.get('confidence_score', h.get('final_confidence', h.get('confidence', 0)))
                count = h.get('count', 1)
                sources = h.get('source', h.get('sources', ['unknown']))
                
                # Format sources properly
                if isinstance(sources, list):
                    sources_str = ', '.join(sources)
                else:
                    sources_str = str(sources)
                
                print(f"{idx}. {h['start']}-{h['end']} | conf={confidence:.2f} | count={count} | sources={sources_str}")
                
                # Show sequence fragment
                start_idx = max(0, h['start'] - 1)  # Convert to 0-based indexing
                end_idx = min(len(seq), h['end'])
                fragment = seq[start_idx:end_idx]
                print(f"   Sequence: {fragment}")
            
            # Create hotspot map
            hotspot_map = detector.create_position_hotspot_map(hotspots, len(seq))
            hotspot_percentage = sum(hotspot_map) / len(seq) * 100 if len(seq) > 0 else 0
            print(f"Hotspot positions: {sum(hotspot_map)}/{len(seq)} ({hotspot_percentage:.1f}%)")
            
            # Show hotspot map visualization
            print("\nHotspot Map (1=hotspot, 0=normal):")
            map_str = ''.join(str(x) for x in hotspot_map)
            # Break into chunks of 50 for readability
            for i in range(0, len(map_str), 50):
                chunk = map_str[i:i+50]
                print(f"{i+1:3d}: {chunk}")
                
        except Exception as e:
            print(f"Error analyzing {name}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
