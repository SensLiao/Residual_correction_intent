#!/usr/bin/env python3
"""
Hash-bound transfer manifest for the petct_textual_intent bundle.

Purpose (D-2026-08-05-02 / the 2026-07-31 `DOWNSTREAM_BUNDLE_SYNC_BLOCKED` finding):
the v2 six-class training chain already exists in the local canonical bundle, while the
A6000 still runs a 2026-07-18 v1 ADD-only deployment.  Repairing that is a *deployment*
problem, and the 07-31 audit is explicit that it must be driven by a fresh, hash-bound
transfer manifest -- never by hand-copied hashes and never by a blind overwrite.

Modes
-----
  scan     hash every file under a bundle root -> manifest JSON
  compare  local manifest x server manifest -> per-file verdict table

Verdicts produced by `compare`
------------------------------
  IDENTICAL      same sha256 on both sides                    -> no action
  SERVER_STALE   differs, local mtime newer                   -> candidate to push
  SERVER_NEWER   differs, server mtime newer                  -> DO NOT PUSH, inspect
  DIFFERS_AMBIG  differs, mtimes give no ordering             -> inspect by hand
  LOCAL_ONLY     absent on the server                         -> candidate to push
  SERVER_ONLY    absent locally                               -> NEVER delete, inspect

`SERVER_NEWER` / `SERVER_ONLY` exist because work happened directly on the A6000 after the
local mirror was taken (W0 wave, W1.1 corpus, W2.1 stroke channel).  A one-way rsync would
destroy it, so this tool refuses to express "sync" as a single direction.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

SKIP_DIRS = {"__pycache__", ".git", ".pytest_cache", ".mypy_cache", ".ipynb_checkpoints"}
SKIP_SUFFIX = {".pyc", ".pyo"}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def scan(root: Path, subdirs: list[str]) -> dict:
    files = {}
    for sub in subdirs:
        base = root / sub
        if not base.exists():
            continue
        for dirpath, dirnames, filenames in os.walk(base):
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
            for name in filenames:
                p = Path(dirpath) / name
                if p.suffix in SKIP_SUFFIX:
                    continue
                rel = p.relative_to(root).as_posix()
                st = p.stat()
                files[rel] = {
                    "sha256": sha256_file(p),
                    "bytes": st.st_size,
                    "mtime": int(st.st_mtime),
                }
    per_dir: dict[str, int] = {}
    for rel in files:
        per_dir[rel.split("/", 1)[0]] = per_dir.get(rel.split("/", 1)[0], 0) + 1
    return {
        "root": str(root),
        "subdirs": subdirs,
        "file_count": len(files),
        "files_per_top_dir": dict(sorted(per_dir.items())),
        "files": files,
    }


def compare(local: dict, server: dict) -> dict:
    lf, sf = local["files"], server["files"]
    rows = []
    for rel in sorted(set(lf) | set(sf)):
        a, b = lf.get(rel), sf.get(rel)
        if a and not b:
            verdict = "LOCAL_ONLY"
        elif b and not a:
            verdict = "SERVER_ONLY"
        elif a["sha256"] == b["sha256"]:
            verdict = "IDENTICAL"
        elif a["mtime"] > b["mtime"]:
            verdict = "SERVER_STALE"
        elif b["mtime"] > a["mtime"]:
            verdict = "SERVER_NEWER"
        else:
            verdict = "DIFFERS_AMBIG"
        rows.append({
            "path": rel,
            "verdict": verdict,
            "local_sha256": (a or {}).get("sha256"),
            "server_sha256": (b or {}).get("sha256"),
            "local_mtime": (a or {}).get("mtime"),
            "server_mtime": (b or {}).get("mtime"),
        })
    counts: dict[str, int] = {}
    for r in rows:
        counts[r["verdict"]] = counts.get(r["verdict"], 0) + 1
    needs_human = [r for r in rows if r["verdict"] in ("SERVER_NEWER", "SERVER_ONLY", "DIFFERS_AMBIG")]
    return {
        "counts": dict(sorted(counts.items())),
        "push_candidates": [r["path"] for r in rows if r["verdict"] in ("SERVER_STALE", "LOCAL_ONLY")],
        "needs_human_review": [
            {k: r[k] for k in ("path", "verdict", "local_mtime", "server_mtime")} for r in needs_human
        ],
        "rows": rows,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)

    s = sub.add_parser("scan")
    s.add_argument("--root", type=Path, required=True)
    s.add_argument("--out", type=Path, required=True)
    s.add_argument("--subdirs", nargs="*",
                   default=["scripts", "configs", "protocols", "schemas", "tests"])

    c = sub.add_parser("compare")
    c.add_argument("--local", type=Path, required=True)
    c.add_argument("--server", type=Path, required=True)
    c.add_argument("--out", type=Path)

    a = ap.parse_args()

    if a.mode == "scan":
        man = scan(a.root, list(a.subdirs))
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(man, indent=2), encoding="utf-8")
        print(json.dumps({
            "root": man["root"],
            "file_count": man["file_count"],
            "files_per_top_dir": man["files_per_top_dir"],
            "out": str(a.out),
        }, indent=2))
        return 0

    local = json.loads(a.local.read_text(encoding="utf-8"))
    server = json.loads(a.server.read_text(encoding="utf-8"))
    rep = compare(local, server)
    summary = {
        "counts": rep["counts"],
        "push_candidate_count": len(rep["push_candidates"]),
        "needs_human_review_count": len(rep["needs_human_review"]),
        "needs_human_review": rep["needs_human_review"][:40],
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(rep, indent=2), encoding="utf-8")
        print("full report -> " + str(a.out), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
