"""Check frozen training strata without model loading, GPU use or networking."""
import argparse
import json
from pathlib import Path
from search_r1.dynamic_sampling.stratified import load_manifest


def main():
    import pandas as pd
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True); parser.add_argument('--manifest', required=True)
    args = parser.parse_args()
    data = pd.read_parquet(args.dataset)
    manifest = load_manifest(args.manifest, args.dataset, [str(x['index']) for x in data.extra_info])
    print('BEHAVIOR_INPUTS PASS: '+json.dumps(manifest['role_counts'], sort_keys=True))


if __name__ == '__main__': main()
