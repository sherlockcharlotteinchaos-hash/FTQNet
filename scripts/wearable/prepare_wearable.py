#!/usr/bin/env python3
"""Make a ready-to-use 19-channel Wearable CSV from the released MixBench data.

The output is identical in channel selection and imputation to the released
Dataset_Wearable.  It preserves deviceId/ts_start, NOT a concatenated time series.
It deliberately does NOT repair irregular time gaps or train-only scaling.
"""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from data_provider.data_wearable import mixbench_prepare_frame, wearable_feature_names


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True,
                        help='Raw MixBench data/coupled/wearable.csv')
    parser.add_argument('--output', type=Path,
                        default=ROOT / 'dataset/wearable/wearable_mixbench19.csv')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    src, dst = args.input.expanduser().resolve(), args.output.expanduser().resolve()
    if not src.is_file():
        raise FileNotFoundError(src)
    if dst.exists() and not args.overwrite:
        raise FileExistsError(f'{dst} exists. Use --overwrite only intentionally.')
    prepared = mixbench_prepare_frame(src)
    columns = wearable_feature_names(src, 'HR')
    # For transparency retain metadata and *exactly* the original loader features.
    data = prepared[['deviceId', 'ts_start'] + columns]
    dst.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(dst, index=False)
    # This command intentionally creates only the requested input CSV.
    print(f'Created {dst} | {len(data)} rows, {len(columns)} features, '
          f'{data.deviceId.nunique()} subjects')



if __name__ == '__main__':
    main()
