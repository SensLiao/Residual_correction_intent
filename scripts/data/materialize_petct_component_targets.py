#!/usr/bin/env python3
"""Materialize label-only multi-positive PET/CT component targets.

For an ADD-existing episode, the authorized residual first identifies exactly
one 18-connected GT lesion. Every current-M component that intersects that GT
lesion is then a positive pointer target. This is intentionally different
from intersecting current M with the ADD residual, which is empty by
construction. ADD-new and REMOVE use no learned pointer target.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np

SCRIPTS_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from common.petct_components import (  # noqa: E402
    COMPONENT_DESCRIPTOR_FIELDS,
    ENUMERATION_VERSION,
    component_key_for,
    label_components_18,
)

TARGET_SCHEMA_VERSION = "PETCT-COMPONENT-TARGETS-v1.0"
CANDIDATE_SCHEMA_VERSION = "PETCT-COMPONENT-CANDIDATES-v1.0"
_ADD_EXISTING_GOALS = {"ADD_SAME_LOCAL", "ADD_SAME_COMPLETE"}
_ADD_NEW_GOAL = "ADD_NEW_COMPLETE"
_REMOVE_GOALS = {
    "REMOVE_SAME_LOCAL",
    "REMOVE_SAME_COMPLETE",
    "REMOVE_NEW_COMPLETE",
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(value: Any) -> str:
    payload = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _sha256_mask(mask: np.ndarray) -> str:
    value = np.ascontiguousarray(np.asarray(mask, dtype=np.uint8))
    digest = hashlib.sha256()
    digest.update(b"axis_order=xyz;shape=")
    digest.update(",".join(str(int(v)) for v in value.shape).encode("ascii"))
    digest.update(b";data=")
    digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def _reject_constant(value: str) -> None:
    raise ValueError("non-standard JSON numeric constant: %s" % value)


def _reject_duplicate_pairs(pairs):
    output = {}
    for key, value in pairs:
        if key in output:
            raise ValueError("duplicate JSON object key: %s" % key)
        output[key] = value
    return output


def _loads_strict(payload: str, label: str) -> Any:
    try:
        return json.loads(
            payload,
            parse_constant=_reject_constant,
            object_pairs_hook=_reject_duplicate_pairs,
        )
    except (json.JSONDecodeError, ValueError) as error:
        raise ValueError("invalid strict JSON in %s: %s" % (label, error)) from error


def _load_json(path: Path) -> Dict[str, Any]:
    value = _loads_strict(path.read_text(encoding="utf-8"), str(path))
    if not isinstance(value, dict):
        raise ValueError("JSON root must be an object: %s" % path)
    return value


def _load_jsonl(path: Path, *, allow_empty: bool = False) -> list[Dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            value = _loads_strict(line, "%s line %d" % (path, line_number))
            if not isinstance(value, dict):
                raise ValueError("JSONL row must be an object: %s line %d" % (path, line_number))
            rows.append(value)
    if not rows and not allow_empty:
        raise ValueError("JSONL file is empty: %s" % path)
    return rows


def join_audit_source_evaluations(
    rows: Sequence[Mapping[str, Any]], audit_rows: Sequence[Mapping[str, Any]]
) -> list[Dict[str, Any]]:
    """Restore the private source mapping only for label-lane target derivation.

    The R12 controlled manifest deliberately omits ``source_evaluation``.  Its
    audit manifest keeps the same record content behind an opaque episode ID.
    This join is confined to this target materializer and never writes the
    restored mapping into a visible manifest or candidate sidecar.
    """

    audit_by_episode: Dict[str, Mapping[str, Any]] = {}
    for audit_row in audit_rows:
        episode_id = str(audit_row.get("episode_id") or "")
        source_record = audit_row.get("source_record")
        if not episode_id or not isinstance(source_record, Mapping):
            raise ValueError("audit row lacks episode_id/source_record")
        if episode_id in audit_by_episode:
            raise ValueError("duplicate audit episode_id: %s" % episode_id)
        if str(source_record.get("episode_id") or "") != episode_id:
            raise ValueError("audit source record episode_id mismatch: %s" % episode_id)
        if str(audit_row.get("source_record_sha256") or "") != _sha256_json(source_record):
            raise ValueError("audit source record hash mismatch: %s" % episode_id)
        audit_by_episode[episode_id] = source_record

    joined: list[Dict[str, Any]] = []
    for row in rows:
        episode_id = str(row.get("episode_id") or "")
        source_record = audit_by_episode.get(episode_id)
        if source_record is None:
            raise ValueError("missing audit source record: %s" % episode_id)
        for field in ("partition", "operation", "goal"):
            if str(row.get(field) or "") != str(source_record.get(field) or ""):
                raise ValueError("audit/source %s mismatch for %s" % (field, episode_id))
        source_evaluation = source_record.get("source_evaluation")
        if not isinstance(source_evaluation, Mapping):
            raise ValueError("audit source_evaluation mapping missing: %s" % episode_id)
        merged = dict(row)
        merged["source_evaluation"] = dict(source_evaluation)
        joined.append(merged)
    return joined


def _write_json_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(
            value,
            stream,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        stream.write("\n")


def _write_jsonl_exclusive(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(
                json.dumps(
                    row,
                    allow_nan=False,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    sort_keys=True,
                )
                + "\n"
            )


def _verified_regular_file(raw: Any, expected_sha256: Any, label: str) -> Path:
    path = Path(str(raw))
    if path.is_symlink():
        raise ValueError("%s must not be a symlink: %s" % (label, path))
    path = path.resolve()
    if not path.is_file():
        raise ValueError("missing %s: %s" % (label, path))
    actual = _sha256_file(path)
    expected = str(expected_sha256 or "")
    if not expected or actual != expected:
        raise ValueError("%s sha256 mismatch: %s" % (label, path))
    return path


def _load_aligned_binary(paths: Sequence[Path]) -> list[np.ndarray]:
    try:
        import nibabel as nib
    except ImportError:
        raise ImportError("nibabel is required to read NIfTI masks") from None
    images = [nib.load(str(path)) for path in paths]
    reference = images[0]
    if len(reference.shape) != 3:
        raise ValueError("NIfTI masks must be 3D [X,Y,Z]")
    for image in images[1:]:
        if image.shape != reference.shape or not np.allclose(
            image.affine, reference.affine, atol=1e-3, rtol=0
        ):
            raise ValueError("m0/gt/authorized NIfTI geometry mismatch")
    return [(np.asarray(image.dataobj) > 0).astype(np.uint8) for image in images]


def _validated_candidate(
    row: Mapping[str, Any],
    summary: Mapping[str, Any],
    m0: np.ndarray,
) -> tuple[Dict[str, Any], list[str]]:
    episode_id = str(row["episode_id"])
    candidate_path = _verified_regular_file(
        summary.get("candidate_path"),
        summary.get("candidate_sha256"),
        "candidate record for %s" % episode_id,
    )
    candidate = _load_json(candidate_path)
    if candidate.get("schema_version") != CANDIDATE_SCHEMA_VERSION:
        raise ValueError("candidate schema mismatch for %s" % episode_id)
    if str(candidate.get("episode_id")) != episode_id:
        raise ValueError("candidate episode_id mismatch for %s" % episode_id)
    if candidate.get("axis_order") != "xyz" or int(candidate.get("axial_axis", -1)) != 2:
        raise ValueError("candidate axis contract mismatch for %s" % episode_id)
    if candidate.get("enumeration_version") != ENUMERATION_VERSION:
        raise ValueError("candidate enumeration version mismatch for %s" % episode_id)
    if candidate.get("descriptor_order") != list(COMPONENT_DESCRIPTOR_FIELDS):
        raise ValueError("candidate descriptor order mismatch for %s" % episode_id)
    if str(candidate.get("partition")) != str(row["partition"]):
        raise ValueError("candidate partition mismatch for %s" % episode_id)
    if str(candidate.get("operation")) != str(row["operation"]):
        raise ValueError("candidate operation mismatch for %s" % episode_id)
    source = row["source_evaluation"]
    m_sha256 = str(source["m0_sha256"])
    if str(candidate.get("m_sha256")) != m_sha256:
        raise ValueError("candidate current-mask hash mismatch for %s" % episode_id)

    components = candidate.get("components")
    if not isinstance(components, list):
        raise ValueError("candidate components must be a list for %s" % episode_id)
    labels, actual_count = label_components_18(m0)
    del labels
    if int(candidate.get("component_count", -1)) != actual_count:
        raise ValueError("candidate component count disagrees with current M for %s" % episode_id)
    if len(components) != actual_count:
        raise ValueError("candidate component list length mismatch for %s" % episode_id)
    keys = []
    for position, component in enumerate(components):
        if not isinstance(component, dict) or int(
            component.get("candidate_position", -1)
        ) != position:
            raise ValueError("candidate positions must be contiguous 0-based for %s" % episode_id)
        if int(component.get("position", position)) != position:
            raise ValueError("candidate enumeration position mismatch for %s" % episode_id)
        expected_key = component_key_for(episode_id, m_sha256, position)
        if str(component.get("component_key")) != expected_key:
            raise ValueError("candidate component_key mismatch for %s" % episode_id)
        vector = component.get("descriptor_vector")
        if not isinstance(vector, list) or len(vector) != len(COMPONENT_DESCRIPTOR_FIELDS):
            raise ValueError("candidate descriptor vector shape mismatch for %s" % episode_id)
        if not all(np.isfinite(float(value)) for value in vector):
            raise ValueError("candidate descriptor contains non-finite value for %s" % episode_id)
        for name, value in zip(COMPONENT_DESCRIPTOR_FIELDS, vector):
            if not np.isclose(float(component[name]), float(value), atol=1e-9, rtol=0):
                raise ValueError("candidate descriptor field/vector mismatch for %s" % episode_id)
        keys.append(expected_key)
    if len(set(keys)) != len(keys):
        raise ValueError("candidate component keys are not unique for %s" % episode_id)
    keys_hash = _sha256_json(keys)
    if str(candidate.get("component_keys_sha256")) != keys_hash:
        raise ValueError("candidate key hash mismatch for %s" % episode_id)

    summary_checks = {
        "m_sha256": m_sha256,
        "enumeration_version": ENUMERATION_VERSION,
        "component_count": actual_count,
        "component_keys": keys,
        "component_keys_sha256": keys_hash,
        "descriptor_order": list(COMPONENT_DESCRIPTOR_FIELDS),
    }
    for name, expected in summary_checks.items():
        if summary.get(name) != expected:
            raise ValueError("candidate summary %s mismatch for %s" % (name, episode_id))
    return candidate, keys


def materialize_target_record(
    row: Mapping[str, Any], candidate_summary: Mapping[str, Any]
) -> Dict[str, Any]:
    """Create one label-only target record with a verified candidate join."""

    episode_id = str(row.get("episode_id") or "")
    if not episode_id:
        raise ValueError("learning row lacks episode_id")
    partition = str(row.get("partition") or "")
    if partition not in ("train", "val"):
        raise ValueError("label-only targets refuse non train/val row: %s" % episode_id)
    operation = str(row.get("operation") or "")
    goal = str(row.get("goal") or "")
    if operation == "ADD" and goal not in (_ADD_EXISTING_GOALS | {_ADD_NEW_GOAL}):
        raise ValueError("ADD row has an incompatible goal for %s" % episode_id)
    if operation == "REMOVE" and goal not in _REMOVE_GOALS:
        raise ValueError("REMOVE row has an incompatible goal for %s" % episode_id)
    if operation not in ("ADD", "REMOVE"):
        raise ValueError("invalid operation for %s" % episode_id)
    source = row.get("source_evaluation")
    if not isinstance(source, dict):
        raise ValueError("source_evaluation mapping missing for %s" % episode_id)
    m0_path = _verified_regular_file(
        source.get("m0_path"), source.get("m0_sha256"), "m0 for %s" % episode_id
    )
    gt_path = _verified_regular_file(
        source.get("gt_path"), source.get("gt_sha256"), "gt for %s" % episode_id
    )
    authorized_path = _verified_regular_file(
        source.get("authorized_path"),
        source.get("authorized_sha256"),
        "authorized residual for %s" % episode_id,
    )
    m0, gt, authorized = _load_aligned_binary([m0_path, gt_path, authorized_path])
    legal_residual = (gt > 0) & ~(m0 > 0) if operation == "ADD" else (m0 > 0) & ~(gt > 0)
    if not authorized.any() or np.any((authorized > 0) & ~legal_residual):
        raise ValueError("authorized mask is not a non-empty legal residual subset for %s" % episode_id)

    candidate, component_keys = _validated_candidate(
        row, candidate_summary, m0
    )
    pointer_positions: list[int] = []
    target_gt_lesion_position = None
    target_gt_lesion_sha256 = None
    if operation == "ADD":
        gt_labels, _ = label_components_18(gt)
        touched_labels = np.unique(gt_labels[authorized > 0])
        touched_labels = touched_labels[touched_labels > 0]
        if len(touched_labels) != 1:
            raise ValueError(
                "authorized ADD residual must identify exactly one GT lesion for %s"
                % episode_id
            )
        gt_label = int(touched_labels[0])
        target_gt_lesion = gt_labels == gt_label
        target_gt_lesion_position = gt_label - 1
        target_gt_lesion_sha256 = _sha256_mask(target_gt_lesion)
        current_labels, current_count = label_components_18(m0)
        if current_count != int(candidate["component_count"]):
            raise ValueError("candidate/current-M count changed for %s" % episode_id)
        pointer_positions = [
            position
            for position in range(current_count)
            if np.any((current_labels == position + 1) & target_gt_lesion)
        ]
        if goal in _ADD_EXISTING_GOALS and not pointer_positions:
            raise ValueError("ADD-existing target GT lesion has no current-M fragment for %s" % episode_id)
        if goal == _ADD_NEW_GOAL and pointer_positions:
            raise ValueError("ADD-new target GT lesion already intersects current M for %s" % episode_id)
        if goal == _ADD_NEW_GOAL:
            pointer_positions = []

    pointer_keys = [component_keys[position] for position in pointer_positions]
    return {
        "schema_version": TARGET_SCHEMA_VERSION,
        "episode_id": episode_id,
        "partition": partition,
        "operation": operation,
        "goal": goal,
        "lane": "label_only",
        "axis_order": "xyz",
        "axial_axis": 2,
        "pointer_targets": pointer_positions,
        "pointer_target_positions": pointer_positions,
        "pointer_target_component_keys": pointer_keys,
        "target_count": len(pointer_positions),
        "target_gt_lesion_position": target_gt_lesion_position,
        "target_gt_lesion_sha256": target_gt_lesion_sha256,
        "target_derivation": "authorized_residual_to_gt_lesion_to_current_m_fragments",
        "candidate_sha256": str(candidate_summary["candidate_sha256"]),
        "candidate_m_sha256": str(candidate["m_sha256"]),
        "candidate_component_count": int(candidate["component_count"]),
        "candidate_component_keys_sha256": str(candidate["component_keys_sha256"]),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--episodes",
        "--learning-manifest",
        dest="episodes",
        type=Path,
        required=True,
        help="learning tensor manifest (jsonl)",
    )
    parser.add_argument("--candidate-summary", type=Path, required=True)
    parser.add_argument(
        "--audit-manifest",
        type=Path,
        default=None,
        help="optional audit JSONL supplying source_evaluation for a redacted manifest",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args(argv)
    if not args.episodes.is_file():
        parser.error("missing learning manifest: %s" % args.episodes)
    if not args.candidate_summary.is_file():
        parser.error("missing candidate summary: %s" % args.candidate_summary)
    if args.audit_manifest is not None and not args.audit_manifest.is_file():
        parser.error("missing audit manifest: %s" % args.audit_manifest)
    if os.path.lexists(str(args.output)):
        parser.error("output already exists: %s" % args.output)
    if os.path.lexists(str(args.summary)):
        parser.error("summary already exists: %s" % args.summary)

    try:
        rows = _load_jsonl(args.episodes)
        if args.audit_manifest is not None:
            rows = join_audit_source_evaluations(rows, _load_jsonl(args.audit_manifest))
        episode_ids = [str(row.get("episode_id") or "") for row in rows]
        if any(not value for value in episode_ids) or len(set(episode_ids)) != len(episode_ids):
            raise ValueError("learning manifest episode_id values must be non-empty and unique")
        candidate_rows = _load_jsonl(args.candidate_summary)
        candidate_by_episode = {}
        for candidate_row in candidate_rows:
            episode_id = str(candidate_row.get("episode_id") or "")
            if not episode_id or episode_id in candidate_by_episode:
                raise ValueError("candidate summary episode_id values must be non-empty and unique")
            candidate_by_episode[episode_id] = candidate_row
        if set(candidate_by_episode) != set(episode_ids):
            missing = sorted(set(episode_ids) - set(candidate_by_episode))
            extra = sorted(set(candidate_by_episode) - set(episode_ids))
            raise ValueError("candidate/learning episode set mismatch; missing=%s extra=%s" % (missing, extra))

        args.output.mkdir(parents=True, exist_ok=False)
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        summary_rows = []
        for row in rows:
            episode_id = str(row["episode_id"])
            candidate_summary = candidate_by_episode[episode_id]
            record = materialize_target_record(row, candidate_summary)
            output_path = (args.output / (episode_id + ".json")).resolve()
            _write_json_exclusive(output_path, record)
            summary_rows.append(
                {
                    "schema_version": TARGET_SCHEMA_VERSION,
                    "episode_id": episode_id,
                    "partition": str(record["partition"]),
                    "operation": str(record["operation"]),
                    "target_path": str(output_path),
                    "target_sha256": _sha256_file(output_path),
                    "pointer_targets": list(record["pointer_targets"]),
                    "pointer_target_component_keys": list(
                        record["pointer_target_component_keys"]
                    ),
                    "candidate_sha256": str(record["candidate_sha256"]),
                    "candidate_component_count": int(
                        record["candidate_component_count"]
                    ),
                    "candidate_component_keys_sha256": str(
                        record["candidate_component_keys_sha256"]
                    ),
                }
            )
        _write_jsonl_exclusive(args.summary, summary_rows)
    except (KeyError, TypeError, ValueError, RuntimeError, OSError) as error:
        parser.error(str(error))
    print(
        json.dumps(
            {
                "episodes": len(summary_rows),
                "output": str(args.output.resolve()),
                "summary": str(args.summary.resolve()),
            },
            allow_nan=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
