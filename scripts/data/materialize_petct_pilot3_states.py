#!/usr/bin/env python3
"""Legacy Pilot-3 provenance marker; the v1 ADD-only entry is disabled."""

from __future__ import annotations


LEGACY_SCHEMA = "PETCT-PILOT3-LEGACY-PROVENANCE-v1.0"
LEGACY_GOALS = ("SAME_LOCAL", "SAME_COMPLETE", "NEW_COMPLETE")


class Pilot3MaterializationError(RuntimeError):
    """Raised for every attempted execution of the superseded v1 contract."""


def _removed() -> None:
    raise Pilot3MaterializationError(
        "Pilot-3 ADD-only is legacy provenance and cannot execute; use "
        "materialize_petct_pilot6_states.construct_pilot6_states"
    )


def construct_pilot3_states(*args, **kwargs):  # noqa: ANN002, ANN003
    _removed()


def publish_pilot3_materialization(*args, **kwargs):  # noqa: ANN002, ANN003
    _removed()


def main(argv=None):  # noqa: ANN001
    _removed()


if __name__ == "__main__":
    raise SystemExit(main())
