#!/usr/bin/env python3
"""Compare the frequency source with the time repository's bundled snapshot."""
from __future__ import annotations

import argparse
import ast
from collections import deque
import hashlib
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]


def commit(repo: Path) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def module_paths(base: Path) -> dict[str, Path]:
    mapping = {}
    for path in (base / "src/loudspeaker_axisym_fem").glob("*.py"):
        module = "loudspeaker_axisym_fem" if path.stem == "__init__" else f"loudspeaker_axisym_fem.{path.stem}"
        mapping[module] = path
    for path in (base / "best_model").glob("*.py"):
        if path.stem != "__init__":
            mapping[path.stem] = path
    return mapping


def imports(path: Path, module: str, known: set[str]) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    package = module.rsplit(".", 1)[0] if "." in module else ""
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            candidates = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            prefix = (package.split(".")[:len(package.split(".")) - node.level + 1]
                      if node.level else [])
            name = ".".join([*prefix, node.module] if node.module else prefix) if node.level else (node.module or "")
            candidates = [name, *(f"{name}.{alias.name}" for alias in node.names)]
        else:
            continue
        found.update(candidate for candidate in candidates if candidate in known)
    return found


def audit(time_repo: Path) -> dict:
    snapshot = time_repo / "inputs/frequency_mainline"
    if not (snapshot / "src/loudspeaker_axisym_fem").is_dir():
        raise FileNotFoundError(snapshot)
    frequency = module_paths(ROOT)
    bundled = module_paths(snapshot)
    known = set(frequency) | set(bundled)
    seeds = set()
    for path in (time_repo / "src/loudspeaker_time_fem").glob("*.py"):
        seeds.update(imports(path, "loudspeaker_time_fem." + path.stem, known))
    closure = set()
    queue = deque(sorted(seeds))
    while queue:
        module = queue.popleft()
        if module in closure:
            continue
        closure.add(module)
        path = bundled.get(module)
        if path is not None:
            queue.extend(sorted(imports(path, module, known) - closure))
    modules = []
    for name in sorted(known):
        fpath, tpath = frequency.get(name), bundled.get(name)
        status = "same" if fpath and tpath and sha(fpath) == sha(tpath) else (
            "different" if fpath and tpath else "frequency_only" if fpath else "time_only"
        )
        modules.append({
            "module": name,
            "frequency_path": str(fpath.relative_to(ROOT)) if fpath else None,
            "time_snapshot_path": str(tpath.relative_to(time_repo)) if tpath else None,
            "status": status,
            "time_import_closure": name in closure,
        })
    return {
        "frequency_commit": commit(ROOT),
        "time_commit": commit(time_repo),
        "direct_time_imports": sorted(seeds),
        "time_import_closure": sorted(closure),
        "modules": modules,
    }


def render_markdown(report: dict) -> str:
    rows = report["modules"]
    used_diffs = [row for row in rows if row["time_import_closure"] and row["status"] != "same"]
    return "\n".join([
        "# 频域代码与时域内置快照审计",
        "",
        f"频域基线：`{report['frequency_commit']}`。时域基线：`{report['time_commit']}`。",
        "本清单由 `tools/audit_time_snapshot.py` 基于 Python 静态导入和文件 SHA-256 生成；动态导入及运行时行为仍需单独核对。",
        "",
        "## 时域直接导入",
        "",
        *[f"- `{name}`" for name in report["direct_time_imports"]],
        "",
        "## 时域依赖闭包中的差异",
        "",
        *([f"- `{row['module']}`：{row['status']}；频域 `{row['frequency_path']}`，时域 `{row['time_snapshot_path']}`。" for row in used_diffs]
          or ["- 静态导入闭包内无文件差异。"]),
        "",
        "## 逐文件对照",
        "",
        "| 模块 | 状态 | 时域导入闭包 | 频域文件 | 时域快照文件 |",
        "|---|---|---|---|---|",
        *[f"| `{row['module']}` | {row['status']} | {'是' if row['time_import_closure'] else '否'} | `{row['frequency_path'] or '—'}` | `{row['time_snapshot_path'] or '—'}` |" for row in rows],
        "",
        "同步规则：仅当频域改动落在时域导入闭包内，才逐文件审查并在时域仓库单独提交；不得覆盖整个快照目录。",
        "函数级人工核对和时域基线记录见[时域仓库说明](https://github.com/341151719/loudspeakerTimeFEM_minimal_latest/blob/main/docs/FREQUENCY_SNAPSHOT_AUDIT_CN.md)。",
        "",
    ])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--time-repo", type=Path, default=ROOT.parent / "loudspeakerTimeFEM_minimal_latest")
    parser.add_argument("--json", type=Path)
    parser.add_argument("--markdown", type=Path)
    args = parser.parse_args()
    report = audit(args.time_repo.resolve())
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if args.markdown:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps({
        "frequency_commit": report["frequency_commit"],
        "time_commit": report["time_commit"],
        "modules": len(report["modules"]),
        "time_import_closure": len(report["time_import_closure"]),
        "different_in_closure": [row["module"] for row in report["modules"]
                                 if row["time_import_closure"] and row["status"] != "same"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
