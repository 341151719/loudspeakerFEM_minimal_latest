"""Atomic, input-bound checkpoints for the CLI frequency sweep."""
from __future__ import annotations

import json
import os
import fcntl
from pathlib import Path
from types import SimpleNamespace


COMPLEX_FIELDS = ("current_A_peak", "motional_impedance_ohm", "total_impedance_ohm", "p_axis_1m_Pa_peak")


def _atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    try:
        temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class SweepCheckpoint:
    def __init__(self, outdir: Path, run_key: str, frequencies: list[float]):
        self.outdir = outdir
        self.lock = (outdir / ".sweep.lock").open("a+")
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self.lock.close()
            raise RuntimeError(f"another sweep is using {outdir}") from None
        self.point_dir = outdir / "checkpoints" / "points"
        self.point_dir.mkdir(parents=True, exist_ok=True)
        self.state_path = outdir / "sweep_state.json"
        self.run_key = run_key
        self.frequencies = [float(freq) for freq in frequencies]
        if self.state_path.exists():
            state = json.loads(self.state_path.read_text(encoding="utf-8"))
            if state.get("run_key") != run_key or state.get("frequencies_Hz") != self.frequencies:
                raise ValueError("sweep output directory belongs to different inputs, code, or solver options; use a new --outdir")
            if len(state.get("points", [])) != len(frequencies):
                raise ValueError("sweep state has the wrong number of points")
            self.state = state
        else:
            if any(self.point_dir.iterdir()) or (outdir / "sweep_metrics.csv").exists():
                raise ValueError("sweep output exists without sweep_state.json; use a new --outdir")
            self.state = {
                "run_key": run_key,
                "frequencies_Hz": self.frequencies,
                "points": ["pending"] * len(frequencies),
                "errors": {},
                "status": "pending",
            }
        self.completed = {}
        for index, freq in enumerate(self.frequencies):
            point = self._read_point(index, freq)
            if point is not None:
                self.completed[index] = point
                self.state["points"][index] = "completed"
                self.state["errors"].pop(str(index), None)
            elif self.state["points"][index] in ("running", "completed"):
                self.state["points"][index] = "pending"
        self._save()

    def _point_path(self, index: int) -> Path:
        return self.point_dir / f"{index:06d}.json"

    def _read_point(self, index: int, freq: float):
        path = self._point_path(index)
        if not path.exists():
            return None
        point = json.loads(path.read_text(encoding="utf-8"))
        if point.get("run_key") != self.run_key or point.get("index") != index or point.get("freq_Hz") != freq:
            raise ValueError(f"checkpoint does not match this run: {path}")
        values = point["solution"]
        return SimpleNamespace(**{
            key: None if value is None else complex(*value) if key in COMPLEX_FIELDS else value
            for key, value in values.items()
        })

    def _save(self) -> None:
        states = self.state["points"]
        self.state["status"] = (
            "completed" if all(state == "completed" for state in states)
            else "partial" if any(state in ("completed", "failed", "interrupted") for state in states)
            else "running" if any(state == "running" for state in states)
            else "pending"
        )
        _atomic_json(self.state_path, self.state)

    def start(self, index: int) -> None:
        self.state["points"][index] = "running"
        self.state["errors"].pop(str(index), None)
        self._save()

    def finish(self, index: int, solution: dict) -> None:
        encoded = {
            key: [value.real, value.imag] if key in COMPLEX_FIELDS and value is not None else value
            for key, value in solution.items()
        }
        _atomic_json(self._point_path(index), {
            "run_key": self.run_key,
            "index": index,
            "freq_Hz": self.frequencies[index],
            "solution": encoded,
        })
        self.completed[index] = SimpleNamespace(**solution)
        self.state["points"][index] = "completed"
        self._save()

    def fail(self, index: int, error: BaseException) -> None:
        self.state["points"][index] = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        self.state["errors"][str(index)] = f"{type(error).__name__}: {error}"
        self._save()

    def ordered_solutions(self) -> list[SimpleNamespace]:
        return [self.completed[index] for index in range(len(self.frequencies)) if index in self.completed]

    def interrupt_running(self) -> None:
        for index, state in enumerate(self.state["points"]):
            if state == "running":
                self.state["points"][index] = "interrupted"
        self._save()

    def close(self) -> None:
        fcntl.flock(self.lock, fcntl.LOCK_UN)
        self.lock.close()
