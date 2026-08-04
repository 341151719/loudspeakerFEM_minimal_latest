#!/usr/bin/env python3
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "MANIFEST.json"
IGNORED_PARTS = {".venv", ".pytest_cache", "__pycache__"}


def main() -> None:
    entries = []
    total = 0
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file() or path == OUTPUT:
            continue
        rel = path.relative_to(ROOT)
        if any(part in IGNORED_PARTS for part in rel.parts):
            continue
        data = path.read_bytes()
        size = len(data)
        total += size
        entries.append(
            {
                "path": rel.as_posix(),
                "size_bytes": size,
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        )
    manifest = {
        "schema": "consistency-v4-blockedfix-20260716",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "file_count": len(entries),
        "total_size_bytes": total,
        "files": entries,
    }
    OUTPUT.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
