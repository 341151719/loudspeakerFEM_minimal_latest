from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "best_model")]

from cli import _profile_for_frequency, _recommended_jobs, build_parser, parse_freqs


def test_sweep_controls():
    cfg = json.loads((ROOT / "configs/best_model.json").read_text(encoding="utf-8"))
    freqs = parse_freqs("comsol_126", cfg)

    assert len(freqs) == 126
    assert freqs[:12] == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 10.6, 11.2]
    assert freqs[-8:] == [5300.0, 5600.0, 6000.0, 6300.0, 6700.0, 7100.0, 7500.0, 8000.0]
    assert all(b > a for a, b in zip(freqs, freqs[1:]))

    assert _recommended_jobs(4, 2) == 2
    assert 1 <= _recommended_jobs(0, 126) <= 8

    hybrid = cfg["acoustics"]["hybrid_sweep"]
    assert _profile_for_frequency(None, hybrid, 3000) == "configs/fast_p1.json"
    assert _profile_for_frequency(None, hybrid, 8000) is None
    assert _profile_for_frequency(None, hybrid, 8000.1) == "configs/stage35_high_accuracy.json"

    parser = build_parser()
    magnetics = parser.parse_args(["magnetics"])
    assert magnetics.max_iter == 55
    assert magnetics.tol == 1e-5

    sweep = parser.parse_args(["sweep"])
    assert sweep.blas_threads == 1
    assert not sweep.single_profile
