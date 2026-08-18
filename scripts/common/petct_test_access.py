#!/usr/bin/env python3
"""Issue and consume the one formal PET/CT test-partition authorization.

The final-freeze grant is an explicit, no-clobber director/Codex act.  It is
valid only after the separately built FINAL_DEVELOPMENT_FREEZE.json has bound
and revalidated every required scientific artifact.  Consumption is claimed
with O_EXCL in a project-global ledger, so copying or recreating a run
directory cannot reset the exactly-once decision.  Leaf CLIs accept only the
resulting consumed receipt and revalidate its config, split, freeze, ledger
entry, and run-root binding before touching test truth.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

try:
    from common.petct_development_freeze import (
        DevelopmentFreezeError,
        validate_final_development_freeze,
    )
except ModuleNotFoundError:  # direct ``python scripts/common/...`` execution
    from petct_development_freeze import (  # type: ignore[no-redef]
        DevelopmentFreezeError,
        validate_final_development_freeze,
    )


PROJECT_ROOT = Path(__file__).resolve().parents[2]
GLOBAL_LEDGER_ROOT = PROJECT_ROOT / "records" / "test-access" / "global-consumption-ledger"

GRANT_SCHEMA = "PETCT-TEST-ACCESS-FINAL-FREEZE-GRANT-v2.0"
LEDGER_SCHEMA = "PETCT-TEST-ACCESS-GLOBAL-CONSUMPTION-v2.0"
RECEIPT_SCHEMA = "PETCT-TEST-ACCESS-CONSUMED-v2.0"
FINAL_FREEZE_CONFIRMATION = "I_CONFIRM_ALL_DEVELOPMENT_FREEZES_ARE_FINAL"
ALLOWED_AUTHORIZERS = ("director", "codex-on-director-authority")
TEST_PARTITION = "test"

TEST_ACCESS_POLICIES = {
    None: (
        "exactly-once-after-all-development-freezes",
        "exactly-once-after-freeze",
    ),
    "PETCT-ROUTE-A-EXPERIMENT-v1.0": (
        "exactly-once-after-all-development-freezes",
        "exactly-once-after-freeze",
    ),
    "PETCT-ROUTE-A-EXPERIMENT-v2.0": (
        "exactly-once-after-all-v2-development-freezes",
        "exactly-once-after-v2-freeze",
    ),
    # v3.0 mirrors the sealed v3 config exactly (registered 2026-08-18,
    # audit-data-corpus MEDIUM-1).  statistics.test_access is None in the
    # sealed v3 config: the statistics partition rides the SAME exactly-once
    # final-freeze grant as the dataset partition rather than a separate
    # policy string.  The formal closed-loop protocol amendment remains a
    # pending director decision; this registration only unblocks the grant
    # path for the config that is actually frozen.
    "PETCT-ROUTE-A-EXPERIMENT-v3.0": (
        "exactly-once-after-all-v2-development-freezes",
        None,
    ),
}


class TestAccessError(RuntimeError):
    """Raised when formal test access is missing, stale, replayed, or tampered."""

    __test__ = False


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _canonical_sha256(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _regular_file(path: Path, *, label: str) -> Path:
    raw = Path(path)
    if raw.is_symlink():
        raise TestAccessError(f"{label} must be a regular non-symlink file: {raw}")
    resolved = raw.resolve()
    if not resolved.is_file():
        raise TestAccessError(
            f"{label} must be a regular non-symlink file: {resolved}"
        )
    return resolved


def _file_record(path: Path, *, label: str) -> dict[str, Any]:
    path = _regular_file(path, label=label)
    return {
        "path": str(path),
        "sha256": _sha256_file(path),
        "bytes": path.stat().st_size,
    }


def _load_json(path: Path, *, label: str) -> tuple[Path, dict[str, Any]]:
    path = _regular_file(path, label=label)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise TestAccessError(f"{label} is not valid UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise TestAccessError(f"{label} must be a JSON object")
    return path, value


def _sealed(payload: Mapping[str, Any], field: str) -> dict[str, Any]:
    unsigned = dict(payload)
    unsigned.pop(field, None)
    return {**unsigned, field: _canonical_sha256(unsigned)}


def _verify_seal(payload: Mapping[str, Any], field: str, *, label: str) -> str:
    observed = payload.get(field)
    unsigned = {key: value for key, value in payload.items() if key != field}
    expected = _canonical_sha256(unsigned)
    if observed != expected:
        raise TestAccessError(f"{label} self-hash mismatch")
    return expected


def _write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    """Durably publish one no-clobber JSON file using an atomic O_EXCL claim."""

    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(
        payload,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8") + b"\n"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(str(path), flags, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            descriptor = -1
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
    except Exception:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        raise
    if hasattr(os, "O_DIRECTORY"):
        directory_fd = os.open(str(path.parent), os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)


def _validate_config_and_split(
    experiment_config: Path, learning_split: Path
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    config_path, config = _load_json(experiment_config, label="experiment config")
    split_path, split = _load_json(learning_split, label="learning split")
    try:
        development_policy = config["dataset"]["learning_split"]["test_access"]
        statistics_policy = config["statistics"]["test_access"]
        split_schema = config["dataset"]["learning_split"]["schema_version"]
    except (KeyError, TypeError) as exc:
        raise TestAccessError("experiment config omits the frozen test-access policy") from exc
    schema_version = config.get("schema_version")
    if schema_version not in TEST_ACCESS_POLICIES:
        raise TestAccessError("experiment config test-access schema is unknown")
    expected_development, expected_statistics = TEST_ACCESS_POLICIES[schema_version]
    if development_policy != expected_development:
        raise TestAccessError("experiment config development test-access policy is not frozen")
    if statistics_policy != expected_statistics:
        raise TestAccessError("experiment config statistics test-access policy is not frozen")
    if split.get("schema_version") != split_schema:
        raise TestAccessError("learning split schema differs from the experiment config")
    if split.get("status") != "FROZEN_BEFORE_MODEL_SELECTION":
        raise TestAccessError("learning split is not frozen before model selection")
    return (
        config,
        split,
        _file_record(config_path, label="experiment config"),
        _file_record(split_path, label="learning split"),
    )


def _grant_id(config_sha256: str, split_sha256: str, freeze_sha256: str) -> str:
    return _canonical_sha256(
        {
            "schema_version": GRANT_SCHEMA,
            "experiment_config_sha256": config_sha256,
            "learning_split_sha256": split_sha256,
            "final_development_freeze_sha256": freeze_sha256,
            "allowed_partition": TEST_PARTITION,
            "final_freeze": True,
        }
    )


def _consumption_key(
    config_sha256: str, split_sha256: str, freeze_sha256: str
) -> str:
    # Deliberately excludes grant path/id and run root.  A copied or re-issued
    # grant for the same frozen experiment therefore cannot reset consumption.
    return _canonical_sha256(
        {
            "schema_version": LEDGER_SCHEMA,
            "experiment_config_sha256": config_sha256,
            "learning_split_sha256": split_sha256,
            "final_development_freeze_sha256": freeze_sha256,
            "allowed_partition": TEST_PARTITION,
        }
    )


def create_final_freeze_grant(
    *,
    experiment_config: Path,
    learning_split: Path,
    final_development_freeze: Path,
    grant_path: Path,
    authorized_by: str,
    confirmation: str,
    now: Callable[[], str] = _utc_now,
) -> dict[str, Any]:
    """Publish an explicit final-freeze grant without consuming test access."""

    if authorized_by not in ALLOWED_AUTHORIZERS:
        raise TestAccessError("authorized_by must identify the director or delegated Codex")
    if confirmation != FINAL_FREEZE_CONFIRMATION:
        raise TestAccessError("exact final-freeze confirmation phrase is required")
    _, _, config_record, split_record = _validate_config_and_split(
        experiment_config, learning_split
    )
    try:
        freeze = validate_final_development_freeze(
            final_development_freeze,
            experiment_config=experiment_config,
            learning_split=learning_split,
        )
    except DevelopmentFreezeError as exc:
        raise TestAccessError(f"final development freeze is invalid: {exc}") from exc
    freeze_record = _file_record(
        final_development_freeze, label="final development freeze"
    )
    payload = _sealed(
        {
            "schema_version": GRANT_SCHEMA,
            "status": "FINAL_FREEZE_GRANTED",
            "authorized_by": authorized_by,
            "authorization_kind": "explicit-final-freeze-command",
            "confirmation": confirmation,
            "final_freeze": True,
            "allowed_partition": TEST_PARTITION,
            "experiment_config": config_record,
            "learning_split": split_record,
            "final_development_freeze": freeze_record,
            "checkpoint_inventory_sha256": freeze[
                "checkpoint_inventory_sha256"
            ],
            "grant_id": _grant_id(
                config_record["sha256"],
                split_record["sha256"],
                freeze_record["sha256"],
            ),
            "granted_at_utc": now(),
        },
        "grant_sha256",
    )
    try:
        _write_json_exclusive(grant_path, payload)
    except FileExistsError as exc:
        raise TestAccessError(f"final-freeze grant already exists: {grant_path}") from exc
    return payload


def _validate_grant(grant_path: Path) -> tuple[Path, dict[str, Any]]:
    grant_path, grant = _load_json(grant_path, label="final-freeze grant")
    _verify_seal(grant, "grant_sha256", label="final-freeze grant")
    expected_header = {
        "schema_version": GRANT_SCHEMA,
        "status": "FINAL_FREEZE_GRANTED",
        "authorization_kind": "explicit-final-freeze-command",
        "confirmation": FINAL_FREEZE_CONFIRMATION,
        "final_freeze": True,
        "allowed_partition": TEST_PARTITION,
    }
    if any(grant.get(key) != value for key, value in expected_header.items()):
        raise TestAccessError("final-freeze grant contract is invalid")
    if grant.get("authorized_by") not in ALLOWED_AUTHORIZERS:
        raise TestAccessError("final-freeze grant authorizer is invalid")
    config_record = grant.get("experiment_config")
    split_record = grant.get("learning_split")
    freeze_record = grant.get("final_development_freeze")
    if (
        not isinstance(config_record, dict)
        or not isinstance(split_record, dict)
        or not isinstance(freeze_record, dict)
    ):
        raise TestAccessError("final-freeze grant omits config/split/freeze records")
    _, _, current_config, current_split = _validate_config_and_split(
        Path(str(config_record.get("path") or "")),
        Path(str(split_record.get("path") or "")),
    )
    if current_config != config_record or current_split != split_record:
        raise TestAccessError("final-freeze grant config or split changed after grant")
    current_freeze = _file_record(
        Path(str(freeze_record.get("path") or "")), label="final development freeze"
    )
    if current_freeze != freeze_record:
        raise TestAccessError("final development freeze changed after grant")
    try:
        freeze = validate_final_development_freeze(
            Path(freeze_record["path"]),
            experiment_config=Path(config_record["path"]),
            learning_split=Path(split_record["path"]),
        )
    except DevelopmentFreezeError as exc:
        raise TestAccessError(f"final development freeze is invalid: {exc}") from exc
    if grant.get("checkpoint_inventory_sha256") != freeze.get(
        "checkpoint_inventory_sha256"
    ):
        raise TestAccessError("final-freeze grant checkpoint inventory mismatch")
    expected_id = _grant_id(
        config_record["sha256"], split_record["sha256"], freeze_record["sha256"]
    )
    if grant.get("grant_id") != expected_id:
        raise TestAccessError("final-freeze grant id mismatch")
    return grant_path, grant


def consume_final_freeze_grant(
    *,
    grant_path: Path,
    run_root: Path,
    receipt_path: Path,
    ledger_root: Path = GLOBAL_LEDGER_ROOT,
    now: Callable[[], str] = _utc_now,
) -> dict[str, Any]:
    """Atomically consume the global test decision and emit its run receipt."""

    grant_path, grant = _validate_grant(grant_path)
    raw_run_root = Path(run_root)
    if raw_run_root.is_symlink():
        raise TestAccessError("run root must already be a real non-symlink directory")
    run_root = raw_run_root.resolve()
    if not run_root.is_dir():
        raise TestAccessError("run root must already be a real non-symlink directory")
    receipt_path = Path(receipt_path).resolve()
    if receipt_path == run_root or not receipt_path.is_relative_to(run_root):
        raise TestAccessError("consumed receipt must be written inside its bound run root")
    config_record = dict(grant["experiment_config"])
    split_record = dict(grant["learning_split"])
    freeze_record = dict(grant["final_development_freeze"])
    key = _consumption_key(
        config_record["sha256"], split_record["sha256"], freeze_record["sha256"]
    )
    ledger_root = Path(ledger_root).resolve()
    ledger_path = ledger_root / f"{key}.json"
    consumed_at = now()
    core = {
        "consumption_key": key,
        "grant_id": grant["grant_id"],
        "grant": _file_record(grant_path, label="final-freeze grant"),
        "experiment_config": config_record,
        "learning_split": split_record,
        "final_development_freeze": freeze_record,
        "checkpoint_inventory_sha256": grant["checkpoint_inventory_sha256"],
        "allowed_partition": TEST_PARTITION,
        "run_root": str(run_root),
        "receipt_path": str(receipt_path),
        "consumed_at_utc": consumed_at,
    }
    core_sha256 = _canonical_sha256(core)
    ledger = _sealed(
        {
            "schema_version": LEDGER_SCHEMA,
            "status": "CONSUMED",
            "consumption": core,
            "consumption_sha256": core_sha256,
        },
        "ledger_record_sha256",
    )
    try:
        _write_json_exclusive(ledger_path, ledger)
    except FileExistsError as exc:
        raise TestAccessError(
            "test access was already consumed globally for this config/split"
        ) from exc
    ledger_record = _file_record(ledger_path, label="global consumption ledger record")
    receipt = _sealed(
        {
            "schema_version": RECEIPT_SCHEMA,
            "status": "CONSUMED",
            "consumption": core,
            "consumption_sha256": core_sha256,
            "global_ledger": ledger_record,
        },
        "receipt_sha256",
    )
    try:
        _write_json_exclusive(receipt_path, receipt)
    except Exception as exc:
        # The global claim deliberately remains: a partial publication must
        # fail closed, never make the test decision reusable.
        raise TestAccessError(
            "global test access was consumed but the run receipt could not be published"
        ) from exc
    return receipt


def validate_consumed_receipt(
    receipt_path: Path,
    *,
    experiment_config: Path,
    learning_split: Path,
    run_root: Path,
    ledger_root: Path = GLOBAL_LEDGER_ROOT,
) -> dict[str, Any]:
    """Revalidate a consumed receipt and its immutable global ledger anchor."""

    receipt_path, receipt = _load_json(receipt_path, label="consumed test receipt")
    _verify_seal(receipt, "receipt_sha256", label="consumed test receipt")
    if receipt.get("schema_version") != RECEIPT_SCHEMA or receipt.get("status") != "CONSUMED":
        raise TestAccessError("consumed test receipt contract is invalid")
    core = receipt.get("consumption")
    if not isinstance(core, dict) or receipt.get("consumption_sha256") != _canonical_sha256(core):
        raise TestAccessError("consumed test receipt core hash mismatch")
    _, _, config_record, split_record = _validate_config_and_split(
        experiment_config, learning_split
    )
    if core.get("experiment_config") != config_record:
        raise TestAccessError("consumed test receipt binds a different experiment config")
    if core.get("learning_split") != split_record:
        raise TestAccessError("consumed test receipt binds a different learning split")
    if core.get("allowed_partition") != TEST_PARTITION:
        raise TestAccessError("consumed test receipt does not allow exactly the test partition")
    raw_run_root = Path(run_root)
    if raw_run_root.is_symlink():
        raise TestAccessError("bound run root must be a real non-symlink directory")
    run_root = raw_run_root.resolve()
    if not run_root.is_dir():
        raise TestAccessError("bound run root must be a real non-symlink directory")
    if core.get("run_root") != str(run_root):
        raise TestAccessError("consumed test receipt binds a different run root")
    if core.get("receipt_path") != str(receipt_path):
        raise TestAccessError("consumed test receipt path differs from its global claim")
    freeze_record = core.get("final_development_freeze")
    if not isinstance(freeze_record, dict):
        raise TestAccessError("consumed receipt omits final development freeze")
    current_freeze = _file_record(
        Path(str(freeze_record.get("path") or "")), label="final development freeze"
    )
    if current_freeze != freeze_record:
        raise TestAccessError("final development freeze changed after consumption")
    try:
        freeze = validate_final_development_freeze(
            Path(freeze_record["path"]),
            experiment_config=experiment_config,
            learning_split=learning_split,
        )
    except DevelopmentFreezeError as exc:
        raise TestAccessError(f"final development freeze is invalid: {exc}") from exc
    if core.get("checkpoint_inventory_sha256") != freeze.get(
        "checkpoint_inventory_sha256"
    ):
        raise TestAccessError("consumed receipt checkpoint inventory mismatch")
    expected_key = _consumption_key(
        config_record["sha256"], split_record["sha256"], freeze_record["sha256"]
    )
    if core.get("consumption_key") != expected_key:
        raise TestAccessError("consumed test receipt key mismatch")
    ledger_path = Path(ledger_root).resolve() / f"{expected_key}.json"
    ledger_record = receipt.get("global_ledger")
    if not isinstance(ledger_record, dict) or ledger_record != _file_record(
        ledger_path, label="global consumption ledger record"
    ):
        raise TestAccessError("consumed test receipt global ledger record changed")
    _, ledger = _load_json(ledger_path, label="global consumption ledger record")
    _verify_seal(ledger, "ledger_record_sha256", label="global consumption ledger record")
    if (
        ledger.get("schema_version") != LEDGER_SCHEMA
        or ledger.get("status") != "CONSUMED"
        or ledger.get("consumption") != core
        or ledger.get("consumption_sha256") != receipt["consumption_sha256"]
    ):
        raise TestAccessError("global consumption ledger differs from the run receipt")
    grant_record = core.get("grant")
    if not isinstance(grant_record, dict) or grant_record != _file_record(
        Path(str(grant_record.get("path") or "")), label="final-freeze grant"
    ):
        raise TestAccessError("final-freeze grant changed after consumption")
    _, grant = _validate_grant(Path(grant_record["path"]))
    if grant.get("grant_id") != core.get("grant_id"):
        raise TestAccessError("consumed receipt grant id mismatch")
    return receipt


def enforce_partition_access(
    partitions: str | Iterable[str],
    *,
    receipt_path: Path | None,
    experiment_config: Path,
    learning_split: Path,
    run_root: Path | None,
    output_paths: Sequence[Path] = (),
    ledger_root: Path = GLOBAL_LEDGER_ROOT,
) -> dict[str, Any] | None:
    """Apply the reusable leaf-CLI rule before any test truth is opened."""

    selected = {partitions} if isinstance(partitions, str) else set(partitions)
    if not selected or not selected.issubset({"train", "val", "test"}):
        raise TestAccessError("partitions must be an explicit subset of train/val/test")
    if TEST_PARTITION not in selected:
        if receipt_path is not None:
            raise TestAccessError("validation/development access rejects a test receipt")
        return None
    if receipt_path is None:
        raise TestAccessError("test partition requires a consumed test-access receipt")
    if run_root is None:
        raise TestAccessError("test partition requires the receipt-bound --run-root")
    receipt = validate_consumed_receipt(
        receipt_path,
        experiment_config=experiment_config,
        learning_split=learning_split,
        run_root=run_root,
        ledger_root=ledger_root,
    )
    resolved_root = Path(run_root).resolve()
    for output in output_paths:
        resolved = Path(output).resolve()
        if resolved == resolved_root or not resolved.is_relative_to(resolved_root):
            raise TestAccessError(
                f"test output escapes the receipt-bound run root: {resolved}"
            )
    return receipt


def add_leaf_test_access_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--test-access-receipt",
        type=Path,
        help="consumed exactly-once receipt; required for test and rejected for val",
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        help="formal run root hash-bound by the consumed test receipt",
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    grant = subparsers.add_parser("grant", help="publish the explicit final-freeze grant")
    grant.add_argument("--experiment-config", type=Path, required=True)
    grant.add_argument("--learning-split", type=Path, required=True)
    grant.add_argument("--final-development-freeze", type=Path, required=True)
    grant.add_argument("--grant", type=Path, required=True)
    grant.add_argument("--authorized-by", choices=ALLOWED_AUTHORIZERS, required=True)
    grant.add_argument("--confirm-final-freeze", required=True)
    consume = subparsers.add_parser("consume", help="atomically consume the global grant")
    consume.add_argument("--grant", type=Path, required=True)
    consume.add_argument("--run-root", type=Path, required=True)
    consume.add_argument("--receipt", type=Path, required=True)
    validate = subparsers.add_parser("validate", help="revalidate a consumed receipt")
    validate.add_argument("--receipt", type=Path, required=True)
    validate.add_argument("--experiment-config", type=Path, required=True)
    validate.add_argument("--learning-split", type=Path, required=True)
    validate.add_argument("--run-root", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "grant":
            payload = create_final_freeze_grant(
                experiment_config=args.experiment_config,
                learning_split=args.learning_split,
                final_development_freeze=args.final_development_freeze,
                grant_path=args.grant,
                authorized_by=args.authorized_by,
                confirmation=args.confirm_final_freeze,
            )
        elif args.command == "consume":
            payload = consume_final_freeze_grant(
                grant_path=args.grant,
                run_root=args.run_root,
                receipt_path=args.receipt,
            )
        else:
            payload = validate_consumed_receipt(
                args.receipt,
                experiment_config=args.experiment_config,
                learning_split=args.learning_split,
                run_root=args.run_root,
            )
    except TestAccessError as exc:
        parser.error(str(exc))
    sys.stdout.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
