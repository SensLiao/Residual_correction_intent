from pathlib import Path
import hashlib


PROJECT = Path(__file__).resolve().parents[1]


def test_core_environment_pins_cc3d_and_preflights_official_autopet_modules() -> None:
    script = (PROJECT / "scripts/setup/setup_petct_nnunet_env.sh").read_text(
        encoding="utf-8"
    )
    assert '"connected-components-3d==4.0.0"' in script
    assert '"schema_version": "PETCT-NNUNET-ENV-v1.1"' in script
    assert 'PETCT_SKIP_INSTALL' in script
    assert '"VERIFY_EXISTING_NO_INSTALL"' in script
    assert 'export PYTHONDONTWRITEBYTECODE=1' in script
    assert '-p no:cacheprovider' in script
    assert '"setup_mode": os.environ["PETCT_ENV_SETUP_MODE"]' in script
    assert 'metadata.version("connected-components-3d")' in script
    assert 'Path(cc3d.__file__).resolve()' in script
    assert 'PETCT_AUTOPETV_PROTOCOL_ROOT' in script
    assert 'PETCT_AUTOPETV_PROTOCOL_MANIFEST' in script
    assert '"PETCT-AUTOPETV-PROTOCOL-RUNTIME-v2.0"' in script
    assert '"FROZEN_MINIMAL_RUNTIME_SIX_CLASS_POLARITY_ADAPTER_NOT_EXECUTED"' in script
    assert '"simulator": (' in script
    assert '"metrics": (' in script
    assert 'receipt["official_autopetv_preflight"][receipt_key]' in script
    assert "observed_runtime_files" in script
    assert "observed_runtime_directories" in script
    assert "protocol root contains a symlink" in script
    assert "files outside the frozen minimal package" in script
    assert '"interactive/simulate_scribbles.py"' in script
    assert '"simulate_scribble_from_label"' in script
    assert '"metrics.py"' in script
    assert '"MetricEvaluator"' in script
    assert '"import_status": "PASS"' in script
    assert '"PREFLIGHT_PASS_PENDING_ATOMIC_EVIDENCE_PUBLICATION"' in script
    assert '"expected_nnunet_runtime_tree_sha256"' in script
    assert 'resolve_head(expected_source)' in script
    assert 'expected_source.joinpath("nnunetv2").rglob("*.py")' in script
    assert '"pinned-runtime-tree-without-git-metadata"' in script
    assert '"git-head-and-pinned-runtime-tree"' in script
    assert 'receipt["nnunet_source_identity_mode"] = source_identity_mode' in script
    assert 'receipt["nnunet_git_metadata_present"] = git_dir.is_dir()' in script
    assert script.index(
        'observed_tree_sha256 != receipt["expected_nnunet_runtime_tree_sha256"]'
    ) < script.index("if git_dir.exists():")
    assert '"a2124e8aa4207e53ac93259214a35b7cf74626f83ab164e519769f86557d7cd2"' in script
    assert '"93e303219deb46b10fc5e5532873a42745aec1ecd6f78335f36cebba62104b83"' in script
    assert '"1eb85fc97224598dad1852b5d6483bbcf0aa8608790dcc657a5a2a761ae9c8c6"' in script
    assert 'PETCT_ENV_EVIDENCE_STAGE' in script
    assert '"PETCT-NNUNET-ENV-EVIDENCE-BUNDLE-v1.0"' in script
    assert '"bundle_sha256": bundle_sha256' in script
    assert '"ENVIRONMENT_EVIDENCE_COMPLETE"' in script
    assert 'os.rename(stage, final_bundle)' in script
    assert 'os.replace(temporary_marker, marker)' in script
    assert 'touch "${EXP_ROOT}/envs/ENV_READY.done"' not in script
    assert 'subprocess' not in script
    assert 'git rev-parse' not in script


def test_autopetv_runtime_manifest_is_the_three_file_minimal_package() -> None:
    import json

    manifest = json.loads(
        (PROJECT / "protocols/autopetv_protocol_runtime.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["schema_version"] == "PETCT-AUTOPETV-PROTOCOL-RUNTIME-v2.0"
    assert (
        manifest["status"]
        == "FROZEN_MINIMAL_RUNTIME_SIX_CLASS_POLARITY_ADAPTER_NOT_EXECUTED"
    )
    assert manifest["upstream_commit"] == "4a2026866bfacc812492cfc7e6a8c54ac3c4f703"
    assert {record["path"] for record in manifest["files"]} == {
        "interactive/simulate_scribbles.py",
        "metrics.py",
        "LICENSE",
    }
    assert "checkpoint" not in manifest

    runtime_root = PROJECT / "external_runners" / "autopetv_protocol"
    if not runtime_root.is_dir():
        runtime_root = PROJECT / "upstream" / "autoPETV"
    for record in manifest["files"]:
        path = runtime_root / record["path"]
        assert path.is_file() and not path.is_symlink()
        assert hashlib.sha256(path.read_bytes()).hexdigest() == record["sha256"]
