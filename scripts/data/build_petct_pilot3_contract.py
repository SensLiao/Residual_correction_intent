#!/usr/bin/env python3
"""Legacy Pilot-3 contract marker; v1 ADD-only compilation is disabled."""

from __future__ import annotations


LEGACY_SCHEMA = "PETCT-PILOT3-LEGACY-PROVENANCE-v1.0"


class Pilot3ContractError(RuntimeError):
    """Raised for every attempted execution of the superseded contract."""


def _removed() -> None:
    raise Pilot3ContractError(
        "Pilot-3 ADD-only is superseded by the six-class Pilot-6 matched-state "
        "contract and cannot be compiled or published"
    )


def compile_pilot3_contract(*args, **kwargs):  # noqa: ANN002, ANN003
    _removed()


def publish_pilot3_contract(*args, **kwargs):  # noqa: ANN002, ANN003
    _removed()


def main(argv=None):  # noqa: ANN001
    _removed()


if __name__ == "__main__":
    raise SystemExit(main())
