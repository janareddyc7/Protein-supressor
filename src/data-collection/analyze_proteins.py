#!/usr/bin/env python3
"""
Script to detect real hotspot regions for specified protein sequences
and save the aggregated results into a JSON file under data/features.
"""
import os
import json
import argparse
import traceback
from pathlib import Path
from real_hotspot import RealHotspotDetector


def parse_args():
    parser = argparse.ArgumentParser(
        description="Detect protein hotspot regions and save results to JSON."
    )
    parser.add_argument(
        '--email',
        required=True,
        help="Email address for RealHotspotDetector registration"
    )
    parser.add_argument(
        '--output',
        default="data/features/hotspot_data.json",
        help=(
            "Relative path to output JSON file,"
            " will be created if it doesn't exist."
        )
    )
    return parser.parse_args()


def main():
    args = parse_args()
    detector = RealHotspotDetector(email=args.email)

    # Four target proteins with UniProt IDs
    proteins = {
        'alpha_synuclein': (
            'MDVFMKGLSKAKEGVVAAAEKTKQGVAEAAGKTKEGVLYVGSKTKEGVV'
            'HGVATVAEKTKEQVTNVGGAVVTGVTAVAQKTVEGAGSIAAATGFVKKDQL'
            'GKNEEGAPQEGILEDMPVDPDNEAYEMPSEEGYQDYEPEA'
        ),
        'amyloid_beta': (
            'DAEFRHDSGYEVHHQKLVFFAEDVGSNKGAIIGLMVGGVV'
        ),
        'sod1': (
            'MATKAVCVLKGDGPVQGIINFEQKESNGPVKVWGSIKGLTEGLHGFHVHEF'
            'GDNTAGCTSAGPHFNPLSRKHGGPKDEERHVGDLGNVTADKDGVADVSIEDSV'
            'ISLSGDHCIIGRTLVVHEKADDLGKGGNEESTKTGNAGSRLACGVIGIAQ'
        ),
        'tau': (
            'MAEPRQEFEVMEDHAGTYGLGDRKDQGGYTMHQDQEGDTDAGLKESPLQTP'
            'TEDGSEEPGSETSDAKSTPTAEDVTAPLVDEGAPGKQAAAQPHTEIPEGTTAEE'
            'AGIGDTPSLEDEAAGHVTQEPESGKVVQEGFLREPGPPGLSHQLMSGMPGAPLLP'
            'EGPREATRQPSGTGPEDTEGGRHAPELLKHQLLGDLHQEGPPLKGAGGKERPGSK'
            'EEVDEDRDVDESSPQDSPPSKASPAQDGRPPQTAAREATSIPGFPAEGAIPLPVDF'
            'LSKVSTEIPASEPDGPSVGRAKGQDAPLEFTFHVEITPNVQKEQAHSEEHLGRAAF'
            'PGAPGEGPEARGPSLGEDTKEADLPEPSEKQPAAAPRGKPVSRVPQLKARMVSKSK'
            'DGTGSDDKKAKTSTRSSAKTLKNRPCLSPKHPTPGSSDPLIQPSSPAVCPEPPSSPK'
            'YVSSVTSRTGSSGAKEMKLKGADGKTKIATPRGAAPPGQKGQANATRIPAKTPPAPK'
            'TPPSSGEPPKSGDRSGYSSPGSPGTPGSRSRTPSLPTPPTREPKKVAVVRTPPKSPS'
            'SAKSRLQTAPVPMPDLKNVKSKIGSTENLKHQPGGGKVQIINKKLDLSNVQSKCGSKD'
            'NIKHVPGGGSVQIVYKPVDLSKVTSKCGSLGNIHHKPGGGQVEVKSEKLDFKDRVQSKI'
            'GSLDNITHVPGGGNKKIETHKLTFRENAKAKTDHGAEIVYKSPVVSGDTSPRHLSNVS'
            'STGSIDMVDSPQLATLADEVSASLAKQGL'
        )
    }

    results = {}
    for name, seq in proteins.items():
        print(f"Analyzing {name} (length={len(seq)})...")
        try:
            hotspots = detector.detect_real_hotspots(name, seq)
            # Normalize hotspot records
            normalized = []
            for h in hotspots:
                start = int(h['start'])
                end = int(h['end'])
                fragment = seq[start-1:end]
                confidence = float(
                    h.get('confidence_score')
                    or h.get('final_confidence')
                    or h.get('confidence', 0)
                )
                count = int(h.get('count', 1))
                sources = h.get('source') or h.get('sources') or ['unknown']
                if not isinstance(sources, list):
                    sources = [sources]

                normalized.append({
                    'start': start,
                    'end': end,
                    'fragment': fragment,
                    'confidence': round(confidence, 4),
                    'count': count,
                    'sources': sources
                })

            # Create position map (0/1 list)
            hotspot_map = detector.create_position_hotspot_map(hotspots, len(seq))
            if len(hotspot_map) != len(seq):
                print(
                    f"⚠️ Warning: hotspot_map length ({len(hotspot_map)})"
                    f" != sequence length ({len(seq)}) for {name}"
                )

            results[name] = {
                'sequence': seq,
                'length': len(seq),
                'hotspots': normalized,
                'hotspot_map': hotspot_map
            }

        except Exception as e:
            print(f"Error processing {name}: {e}")
            traceback.print_exc()
            results[name] = {'error': str(e)}

    # Prepare output path
    script_dir = Path(__file__).parent
    out_path = script_dir / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Save to JSON
    with out_path.open('w') as f:
        json.dump(results, f, indent=2)

    print(f"\nAll results successfully saved to: {out_path.resolve()}")


if __name__ == '__main__':
    main()
