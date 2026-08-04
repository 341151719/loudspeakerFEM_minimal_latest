#!/usr/bin/env python3
from pathlib import Path
import subprocess,sys
ROOT=Path(__file__).resolve().parent
commands=[[sys.executable,str(ROOT/'cli.py'),'self-test'],[sys.executable,str(ROOT/'tests/test_topology.py')],[sys.executable,str(ROOT/'tests/test_no_results_packaged.py')]]
for c in commands:
    r=subprocess.run(c,cwd=ROOT)
    if r.returncode:raise SystemExit(r.returncode)
print('FULL_SELF_TEST: PASS')
