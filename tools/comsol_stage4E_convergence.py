#!/usr/bin/env python3
from __future__ import annotations

import argparse, sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))

from loudspeaker_axisym_fem.json_utils import dumps_json
from loudspeaker_axisym_fem.stage4E_convergence import build_stage4E_outputs


def main():
    ap=argparse.ArgumentParser(description='Stage 4E: mesh/HK/NRA convergence audit from Stage-4D split runs.')
    ap.add_argument('--outdir', default=str(ROOT/'outputs/stage4E_convergence'))
    ap.add_argument('--coarse-dir', default=str(ROOT/'outputs/stage4D_exterior_nra_final'))
    ap.add_argument('--refined-dir', default='')
    ap.add_argument('--refined-directivity-1000-dir', default='')
    ap.add_argument('--refined-directivity-5000-dir', default='')
    args=ap.parse_args()
    refined=args.refined_dir or str(ROOT/'outputs/stage4E_convergence/runs/refined_stage3_1000_5000_8000')
    d1000=args.refined_directivity_1000_dir or str(ROOT/'outputs/stage4E_convergence/runs/refined_directivity_1000')
    d5000=args.refined_directivity_5000_dir or str(ROOT/'outputs/stage4E_convergence/runs/refined_directivity_5000')
    summary=build_stage4E_outputs(args.outdir, args.coarse_dir, refined, d1000, d5000)
    print(dumps_json(summary, indent=2))

if __name__ == '__main__': main()
