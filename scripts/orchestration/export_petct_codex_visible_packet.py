#!/usr/bin/env python3
"""Export a deterministic visible-only PET/CT packet for the Codex probe."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import sys
import zipfile
from pathlib import Path
from typing import Any

from PIL import Image


VISIBLE_SCHEMA_VERSION = "PETCT-EPISODE-VISIBLE-v2.0"
INTENT_SCHEMA_VERSION = "PETCT-INTENT-v2.0"
PROMPT_VERSION = "PETCT-CODEX-PROMPT-v2.0"
OPAQUE_EPISODE_PATTERN = re.compile(r"ep-[0-9a-f]{6,64}")
PACKET_ID_PATTERN = re.compile(r"[a-z0-9][a-z0-9-]{2,63}")
VISIBLE_ALLOWED_FIELDS = {
    "schema_version",
    "episode_id",
    "lane",
    "patient_group_hash",
    "montage_reference",
    "m0_provenance",
    "scribble",
    "expected_model_output_schema",
}
FORBIDDEN_KEY_FRAGMENTS = (
    "gt",
    "gold",
    "residual",
    "component",
    "authorized",
    "target",
    "source_case",
    "source_patient",
)


class PacketContractError(RuntimeError):
    """Raised when a local packet could leak the evaluation plane."""


def _json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _scan_forbidden_keys(value: Any, *, path: str = "visible") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            lowered = str(key).casefold()
            if any(fragment in lowered for fragment in FORBIDDEN_KEY_FRAGMENTS):
                raise PacketContractError(
                    f"forbidden evaluation field in visible JSON at {path}.{key}"
                )
            _scan_forbidden_keys(child, path=f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _scan_forbidden_keys(child, path=f"{path}[{index}]")


def _validate_visible(payload: dict[str, Any], *, expected_id: str) -> None:
    _scan_forbidden_keys(payload)
    unexpected = sorted(set(payload) - VISIBLE_ALLOWED_FIELDS)
    if unexpected:
        raise PacketContractError(f"unexpected visible fields: {unexpected}")
    missing = sorted(VISIBLE_ALLOWED_FIELDS - set(payload))
    if missing:
        raise PacketContractError(f"visible JSON missing fields: {missing}")
    if payload.get("schema_version") != VISIBLE_SCHEMA_VERSION:
        raise PacketContractError("visible schema version mismatch")
    if payload.get("episode_id") != expected_id:
        raise PacketContractError("visible filename/episode_id mismatch")
    if payload.get("expected_model_output_schema") != INTENT_SCHEMA_VERSION:
        raise PacketContractError("visible intent output schema mismatch")
    if payload.get("lane") not in {"controlled", "natural"}:
        raise PacketContractError("visible lane must be controlled or natural")
    if len(str(payload.get("patient_group_hash", ""))) != 64:
        raise PacketContractError("patient_group_hash must be a SHA-256 digest")
    scribble = payload.get("scribble")
    if not isinstance(scribble, dict) or scribble.get("polarity") not in {
        "foreground",
        "background",
    }:
        raise PacketContractError(
            "visible packet requires one explicit foreground/background scribble polarity"
        )
    montage_reference = payload.get("montage_reference")
    if not isinstance(montage_reference, str):
        raise PacketContractError("montage_reference must be a string")
    if Path(montage_reference).name != f"{expected_id}.png":
        raise PacketContractError("montage reference does not match episode_id")


def _validate_montage(path: Path) -> None:
    if path.is_symlink():
        raise PacketContractError(f"montage symlink is forbidden: {path}")
    try:
        with Image.open(path) as image:
            if image.format != "PNG":
                raise PacketContractError(f"montage must be PNG: {path}")
            forbidden_metadata = {
                key
                for key in image.info
                if key.casefold() in {"comment", "description", "exif", "xml", "xmp"}
            }
            if forbidden_metadata:
                raise PacketContractError(
                    f"montage contains forbidden metadata: {sorted(forbidden_metadata)}"
                )
            image.verify()
    except PacketContractError:
        raise
    except Exception as exc:
        raise PacketContractError(f"montage is unreadable: {path}: {exc}") from exc


def _zip_bytes(entries: list[tuple[str, bytes]]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, mode="w") as archive:
        for archive_path, payload in entries:
            info = zipfile.ZipInfo(
                filename=archive_path,
                date_time=(1980, 1, 1, 0, 0, 0),
            )
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            archive.writestr(info, payload, compresslevel=9)
    return buffer.getvalue()


def export_visible_packet(
    *,
    visible_root: Path,
    montage_root: Path,
    protocol_path: Path,
    output_zip: Path,
    packet_id: str,
) -> dict[str, Any]:
    """Create an allowlisted archive with no evaluation-plane files."""
    if not PACKET_ID_PATTERN.fullmatch(packet_id):
        raise PacketContractError("packet_id must be a lowercase opaque slug")
    visible_root = Path(visible_root).resolve()
    montage_root = Path(montage_root).resolve()
    protocol_path = Path(protocol_path).resolve()
    output_zip = Path(output_zip).resolve()
    if output_zip.exists():
        raise FileExistsError(f"refusing to overwrite existing packet: {output_zip}")
    if not visible_root.is_dir():
        raise PacketContractError(f"visible root not found: {visible_root}")
    if not montage_root.is_dir():
        raise PacketContractError(f"montage root not found: {montage_root}")
    if not protocol_path.is_file() or protocol_path.is_symlink():
        raise PacketContractError(f"protocol file not found or unsafe: {protocol_path}")

    visible_files = sorted(visible_root.glob("*.json"))
    if not visible_files:
        raise PacketContractError("visible packet has no episodes")
    unexpected_visible_files = [
        path for path in visible_root.iterdir() if not path.is_file() or path.suffix != ".json"
    ]
    if unexpected_visible_files:
        raise PacketContractError(
            f"visible root contains unexpected entries: {unexpected_visible_files}"
        )

    episode_payloads: list[tuple[str, bytes]] = []
    montage_payloads: list[tuple[str, bytes]] = []
    referenced_montages: set[str] = set()
    episode_ids: list[str] = []
    for path in visible_files:
        if path.is_symlink():
            raise PacketContractError(f"visible symlink is forbidden: {path}")
        episode_id = path.stem
        if not OPAQUE_EPISODE_PATTERN.fullmatch(episode_id):
            raise PacketContractError(f"visible filename is not opaque: {path.name}")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise PacketContractError(f"visible JSON is invalid: {path}") from exc
        if not isinstance(payload, dict):
            raise PacketContractError(f"visible JSON is not an object: {path}")
        _validate_visible(payload, expected_id=episode_id)
        episode_ids.append(episode_id)
        episode_payloads.append((f"episodes/{episode_id}.json", _json_bytes(payload)))

        montage_name = Path(payload["montage_reference"]).name
        montage_path = montage_root / montage_name
        if not montage_path.is_file():
            raise PacketContractError(f"montage missing for {episode_id}: {montage_path}")
        _validate_montage(montage_path)
        referenced_montages.add(montage_name)
        montage_payloads.append((f"montages/{montage_name}", montage_path.read_bytes()))

    actual_montages = {path.name for path in montage_root.glob("*.png")}
    unexpected_montage_entries = [
        path for path in montage_root.iterdir() if not path.is_file() or path.suffix.casefold() != ".png"
    ]
    if unexpected_montage_entries:
        raise PacketContractError(
            f"montage root contains unexpected entries: {unexpected_montage_entries}"
        )
    unreferenced = sorted(actual_montages - referenced_montages)
    if unreferenced:
        raise PacketContractError(f"unreferenced montage files: {unreferenced}")

    try:
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PacketContractError("prompt protocol is invalid JSON") from exc
    if protocol.get("prompt_version") != PROMPT_VERSION:
        raise PacketContractError("prompt protocol version mismatch")
    if protocol.get("intent_schema_version") != INTENT_SCHEMA_VERSION:
        raise PacketContractError("prompt intent schema mismatch")
    protocol_bytes = _json_bytes(protocol)
    content_entries = sorted(episode_payloads) + sorted(montage_payloads) + [
        ("protocol/petct_codex_prompt_v2.json", protocol_bytes)
    ]
    manifest = {
        "contract_version": "PETCT-CODEX-VISIBLE-PACKET-v2.0",
        "packet_id": packet_id,
        "episode_count": len(episode_ids),
        "episode_ids": sorted(episode_ids),
        "prompt_version": PROMPT_VERSION,
        "intent_schema_version": INTENT_SCHEMA_VERSION,
        "contains_eval_plane": False,
        "contains_gt": False,
        "contains_residual": False,
        "contains_gold_intent": False,
        "files": [
            {"path": name, "bytes": len(payload), "sha256": _sha256(payload)}
            for name, payload in content_entries
        ],
    }
    entries = [("MANIFEST.json", _json_bytes(manifest))] + content_entries
    archive_bytes = _zip_bytes(entries)
    output_zip.parent.mkdir(parents=True, exist_ok=True)
    try:
        with output_zip.open("xb") as stream:
            stream.write(archive_bytes)
            stream.flush()
            os.fsync(stream.fileno())
    except Exception:
        try:
            output_zip.unlink()
        except FileNotFoundError:
            pass
        raise
    return {
        "status": "COMMITTED",
        "contract_version": manifest["contract_version"],
        "packet_id": packet_id,
        "path": str(output_zip),
        "bytes": len(archive_bytes),
        "sha256": _sha256(archive_bytes),
        "episode_count": len(episode_ids),
        "contains_eval_plane": False,
        "contains_gt": False,
        "contains_residual": False,
        "contains_gold_intent": False,
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--visible-root", type=Path, required=True)
    parser.add_argument("--montage-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--packet-id", required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    receipt = export_visible_packet(
        visible_root=args.visible_root,
        montage_root=args.montage_root,
        protocol_path=args.protocol,
        output_zip=args.output,
        packet_id=args.packet_id,
    )
    sys.stdout.write(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
