#!/usr/bin/env python3
"""Batch local axial SVD diagnostics; no Gaussian fitting is performed."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from svd_analysis import load_stack
from local_svd_core import analyze_local
from svd_reporting import save_config_results, save_comparison


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parent / 'local_svd_results')
    parser.add_argument('--pattern', default='*.png')
    parser.add_argument('--resolution', nargs='+', type=int, default=[128, 64, 32, 16],
                        help='XY block side lengths in original pixels; images are not resized.')
    parser.add_argument('--slab', nargs='+', type=int, default=[4, 8, 16, 30],
                        help='Maximum slices per nonoverlapping axial segment; retain shorter tail.')
    parser.add_argument('--ranks', nargs='+', type=int, default=[1, 2, 4, 8, 12, 16])
    parser.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'])
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--dtype', choices=['float32', 'float64'], default='float64',
                        help='Gram accumulation and eigensolver precision (default: float64).')
    parser.add_argument('--energy-epsilon', type=float, default=1e-12,
                        help='Mean squared signal threshold for near-zero classification.')
    parser.add_argument('--examples', type=int, default=2, help='Maximum example blocks per analysis.')
    parser.add_argument('--no-plots', action='store_true')
    args = parser.parse_args(argv)
    if any(v <= 0 for v in args.resolution + args.slab) or args.batch_size <= 0:
        parser.error('resolution, slab and batch-size must be positive')
    if not args.ranks or any(r < 0 for r in args.ranks):
        parser.error('ranks must be nonnegative')
    if args.examples < 0 or args.energy_epsilon < 0:
        parser.error('examples and energy-epsilon must be nonnegative')
    args.resolution = list(dict.fromkeys(args.resolution))
    args.slab = list(dict.fromkeys(args.slab))
    return args


def main(argv=None):
    args = parse_args(argv)
    started = time.perf_counter()
    stack, files = load_stack(args.input.resolve(), args.pattern)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    config = {key: str(value) if isinstance(value, Path) else value
              for key, value in vars(args).items()}
    config.update(shape=list(stack.shape), axis_order='z,y,x',
                  input_files=[str(p.resolve()) for p in files],
                  partition='nonoverlapping; retain unpadded XY and Z tails',
                  energy_targets=[0.9, 0.95, 0.99],
                  capacity_note='Scalar factor count, not encoded bytes or Gaussian model size.')
    (output / 'experiment_config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    (output / 'slice_order.txt').write_text(
        '\n'.join(f'{i:02d}\t{p.name}' for i, p in enumerate(files)) + '\n', encoding='utf-8')
    rows = []
    count = len(args.resolution) * len(args.slab)
    print(f'Loaded {stack.shape} [Z,H,W]. Running {count} configurations.', flush=True)
    index = 0
    for resolution in args.resolution:
        for slab in args.slab:
            index += 1
            print(f'[{index}/{count}] resolution={resolution}, slab={slab}', flush=True)
            result = analyze_local(stack, resolution, slab, ranks=args.ranks,
                                   device=args.device, batch_size=args.batch_size,
                                   dtype=args.dtype, energy_epsilon=args.energy_epsilon)
            target = output / f'resolution_{resolution:03d}' / f'slab_{slab:02d}'
            target.mkdir(parents=True, exist_ok=True)
            (target / 'config.json').write_text(
                json.dumps(result['metadata'], indent=2), encoding='utf-8')
            rows.extend(save_config_results(result, target, stack=stack,
                                            examples=args.examples, make_plots=not args.no_plots))
            save_comparison(rows, output, make_plots=False)
            print(f"  {result['metadata']['device']}: {result['metadata']['timing_seconds']:.3f}s; saved {target}", flush=True)
    save_comparison(rows, output, make_plots=not args.no_plots)
    print(f'Completed in {time.perf_counter()-started:.2f}s. Results: {output}', flush=True)


if __name__ == '__main__':
    main()
