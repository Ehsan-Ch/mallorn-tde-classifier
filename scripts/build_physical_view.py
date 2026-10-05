"""Compose a focused physical feature view without reading any class labels."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from mallorn.data import sha256
from mallorn.reporting import write_json


def transform(frame):
    out = frame.loc[:, frame.columns.str.startswith(
        ('phys_', 'gp_', 'bb_', 'morph_', 'flare_'))].copy()
    # Physical derivatives differ from fractional (logarithmic) derivatives:
    # e.g. two equal fractional radius changes can imply different expansion speeds.
    for parameter, scale in [('T', 1e4), ('R', 1e15), ('L', 1e44)]:
        for left, right in [(0, 10), (10, 30), (0, 60)]:
            a = np.power(10., frame[f'bb_log{parameter}_p{left}']) / scale
            b = np.power(10., frame[f'bb_log{parameter}_p{right}']) / scale
            out[f'derived_{parameter}_slope{left}_{right}'] = (b - a) / (right - left)
        out[f'derived_{parameter}_slope_change'] = (
            out[f'derived_{parameter}_slope10_30'] - out[f'derived_{parameter}_slope0_10'])
    for band in 'ugrizy':
        out[f'derived_{band}_peak_tail_ratio'] = (
            frame[f'gp_{band}_p30_norm'] / frame[f'gp_{band}_p90_norm'].where(
                frame[f'gp_{band}_p90_norm'] > .02))
        out[f'derived_{band}_width_ratio'] = (
            frame[f'gp_{band}_width0.2'] / frame[f'gp_{band}_width0.8'].where(
                frame[f'gp_{band}_width0.8'] > 0))
    return out.replace([np.inf, -np.inf], np.nan)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['train', 'test'], default='train')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    dest = root / 'artifacts/improvement'
    source = dest / f'complete_{args.mode}.csv'
    result = dest / f'physical_view_{args.mode}.csv'
    frame = transform(pd.read_csv(source, index_col='object_id'))
    frame.to_csv(result, float_format='%.15g')
    write_json(dest / f'physical_view_{args.mode}_manifest.json', {
        'source_sha256': sha256(source), 'code_sha256': sha256(Path(__file__)),
        'result_sha256': sha256(result), 'objects': len(frame), 'features': frame.shape[1]})
    print(frame.shape)
