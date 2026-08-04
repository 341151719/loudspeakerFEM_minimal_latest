from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None


def to_jsonable(obj: Any, *, nan_as_none: bool = True) -> Any:
    if obj is None or isinstance(obj, (str, bool)):
        return obj
    if np is not None:
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            val = float(obj)
            if not math.isfinite(val):
                return None if nan_as_none else val
            return val
        if isinstance(obj, np.complexfloating):
            z = complex(obj)
            return {"real": to_jsonable(z.real), "imag": to_jsonable(z.imag)}
        if isinstance(obj, np.ndarray):
            return to_jsonable(obj.tolist(), nan_as_none=nan_as_none)
    if isinstance(obj, int):
        return int(obj)
    if isinstance(obj, float):
        if not math.isfinite(obj):
            return None if nan_as_none else obj
        return float(obj)
    if isinstance(obj, complex):
        return {"real": to_jsonable(obj.real), "imag": to_jsonable(obj.imag)}
    if isinstance(obj, (list, tuple, set)):
        return [to_jsonable(x, nan_as_none=nan_as_none) for x in obj]
    if isinstance(obj, dict):
        return {str(to_jsonable(k, nan_as_none=nan_as_none)): to_jsonable(v, nan_as_none=nan_as_none) for k, v in obj.items()}
    if hasattr(obj, "item"):
        try:
            return to_jsonable(obj.item(), nan_as_none=nan_as_none)
        except Exception:
            pass
    return obj


def dumps_json(obj: Any, **kwargs: Any) -> str:
    kwargs.setdefault("ensure_ascii", False)
    kwargs.setdefault("allow_nan", False)
    return json.dumps(to_jsonable(obj), **kwargs)


def write_json(path: str | Path, obj: Any, **kwargs: Any) -> None:
    Path(path).write_text(dumps_json(obj, **kwargs), encoding="utf-8")
