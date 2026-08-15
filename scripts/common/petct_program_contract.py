#!/usr/bin/env python3
"""v3 program grammar: operation-conditioned legal operator calls (SCEP).

This module is the single source of truth for the v3 correction-program
grammar.  It deliberately does NOT import or modify any frozen v2 module:
the v2 six-goal ontology and ``petct_intent_v2.schema.json`` stay untouched;
the six frozen goals are rendered here as deterministic projections of legal
operator calls.

Design authority: the 2026-08-16 SCEP strict-research redesign.  Where the
MS-LCP blueprint used ``PATCH_LOCAL``, SCEP renames it ``GROW_LOCAL``; SCEP
names win.  REMOVE keeps a conditional sixth family
``REPAIR_OVERSEGMENTED_COMPONENT`` whose retention is decided by
construct-validity gates, not by symmetry aesthetics.
"""

from __future__ import annotations

from typing import Dict, FrozenSet, List, Mapping, Tuple

GRAMMAR_VERSION = "PETCT-PROGRAM-GRAMMAR-v1.0"
ARCHITECTURE_ID = "matched_legal_component_program_v1"
SCHEMA_VERSION = "PETCT-PROGRAM-v1.0"

ADD_FAMILIES: Tuple[str, ...] = ("GROW_LOCAL", "COMPLETE_EXISTING", "CREATE_NEW")
REMOVE_FAMILIES: Tuple[str, ...] = (
    "TRIM_LOCAL",
    "DELETE_COMPONENT",
    "REPAIR_OVERSEGMENTED_COMPONENT",
)
REMOVE_CORE_FAMILIES: Tuple[str, ...] = ("TRIM_LOCAL", "DELETE_COMPONENT")

# The conditional REMOVE class exists so the six frozen v2 goals keep a
# one-to-one deterministic rendering; its retention as a learned class is
# gated (see REMOVE_CORE_FAMILIES and config v3 ``grammar.include_repair``).
FAMILY_TO_GOAL: Dict[str, str] = {
    "GROW_LOCAL": "ADD_SAME_LOCAL",
    "COMPLETE_EXISTING": "ADD_SAME_COMPLETE",
    "CREATE_NEW": "ADD_NEW_COMPLETE",
    "TRIM_LOCAL": "REMOVE_SAME_LOCAL",
    "DELETE_COMPONENT": "REMOVE_NEW_COMPLETE",
    "REPAIR_OVERSEGMENTED_COMPONENT": "REMOVE_SAME_COMPLETE",
}
GOAL_TO_FAMILY: Dict[str, str] = {goal: family for family, goal in FAMILY_TO_GOAL.items()}

FAMILIES_BY_OPERATION: Dict[str, Tuple[str, ...]] = {
    "ADD": ADD_FAMILIES,
    "REMOVE": REMOVE_FAMILIES,
}

NEW_CUE_SENTINEL = "NEW_CUE"

# Rendering of the frozen six goals.  SAME/NEW and LOCAL/COMPLETE are NOT
# predicted by independent heads in v3; they are deterministic projections.
GOAL_TO_RENDERED_SLOTS: Dict[str, Tuple[str, str, str]] = {
    "ADD_SAME_LOCAL": ("ADD", "SAME", "LOCAL"),
    "ADD_SAME_COMPLETE": ("ADD", "SAME", "COMPLETE"),
    "ADD_NEW_COMPLETE": ("ADD", "NEW", "COMPLETE"),
    "REMOVE_SAME_LOCAL": ("REMOVE", "SAME", "LOCAL"),
    "REMOVE_SAME_COMPLETE": ("REMOVE", "SAME", "COMPLETE"),
    "REMOVE_NEW_COMPLETE": ("REMOVE", "NEW", "COMPLETE"),
}

OPERATIONS: Tuple[str, ...] = ("ADD", "REMOVE")


class ProgramContractError(ValueError):
    """Raised when a program construction violates the v3 grammar."""


def operation_from_cue_sign(cue_fg_any: bool, cue_bg_any: bool) -> str:
    """Derive operation authority from signed cue presence.

    Exactly one polarity may be present.  The operation is never predicted
    by the compiler; the signed cue hard-conditions it (SCEP 5.1).
    """

    if cue_fg_any and cue_bg_any:
        raise ProgramContractError("both cue polarities present; ambiguous operation")
    if not cue_fg_any and not cue_bg_any:
        raise ProgramContractError("empty signed cue; operation undefined")
    return "ADD" if cue_fg_any else "REMOVE"


def family_ids(operation: str, include_repair: bool = True) -> List[str]:
    """Return the legal family id list for one operation."""

    if operation not in OPERATIONS:
        raise ProgramContractError("unknown operation: %s" % operation)
    families = list(FAMILIES_BY_OPERATION[operation])
    if operation == "REMOVE" and not include_repair:
        families.remove("REPAIR_OVERSEGMENTED_COMPONENT")
    return families


def validate_legal_call(operation: str, family: str, operand: str) -> None:
    """Fail closed on any call that is not legal-by-construction."""

    if operation not in OPERATIONS:
        raise ProgramContractError("unknown operation: %s" % operation)
    if family not in FAMILIES_BY_OPERATION[operation]:
        raise ProgramContractError(
            "family %s is not legal for operation %s" % (family, operation)
        )
    if operation == "ADD":
        if family == "CREATE_NEW":
            if operand != NEW_CUE_SENTINEL:
                raise ProgramContractError("CREATE_NEW requires NEW_CUE operand")
        elif operand == NEW_CUE_SENTINEL:
            raise ProgramContractError(
                "existing-component ADD family requires a component operand"
            )
    else:
        if operand == NEW_CUE_SENTINEL:
            raise ProgramContractError("REMOVE families require a component operand")


def render_goal(operation: str, family: str) -> str:
    """Deterministic projection of a legal call onto the frozen six goals."""

    validate_legal_call(operation, family, NEW_CUE_SENTINEL if family == "CREATE_NEW" else "C")
    return FAMILY_TO_GOAL[family]


def rendered_slots(operation: str, family: str) -> Tuple[str, str, str]:
    """Deterministic (operation, target, scope) rendering of a legal call."""

    return GOAL_TO_RENDERED_SLOTS[render_goal(operation, family)]


def compile_legal_call(
    operation: str, family_id: int, operand: str, *, include_repair: bool = True
) -> Dict[str, str]:
    """Compile a scored family id and operand into a validated call record."""

    family = family_ids(operation, include_repair)[family_id]
    validate_legal_call(operation, family, operand)
    return {
        "grammar_version": GRAMMAR_VERSION,
        "operation": operation,
        "family": family,
        "operand": operand,
        "goal": render_goal(operation, family),
    }


def family_to_id(family: str) -> int:
    """Global family id used by embedding tables and training targets."""

    order = ADD_FAMILIES + REMOVE_FAMILIES
    try:
        return order.index(family)
    except ValueError:
        raise ProgramContractError("unknown family: %s" % family) from None


def id_to_family(family_id: int) -> str:
    order = ADD_FAMILIES + REMOVE_FAMILIES
    if not 0 <= family_id < len(order):
        raise ProgramContractError("family id out of range: %d" % family_id)
    return order[family_id]


def goal_to_family_id(goal: str) -> int:
    if goal not in GOAL_TO_FAMILY:
        raise ProgramContractError("goal is not in the frozen six: %s" % goal)
    return family_to_id(GOAL_TO_FAMILY[goal])


VISIBILITY_LANES: FrozenSet[str] = frozenset(
    {
        "PET5", "CT5", "M5", "signed_cue", "full_current_mask",
        "component_enumeration", "spacing", "component_descriptors",
    }
)
LABEL_ONLY_FIELDS: FrozenSet[str] = frozenset(
    {
        "gt", "authorized_residual", "gold_program", "pointer_targets",
        "residual_maps", "lesion_correspondence",
    }
)
AUDIT_ONLY_FIELDS: FrozenSet[str] = frozenset(
    {
        "checkpoint_hash", "grammar_version", "enumeration_version",
        "m_sha256", "manifest_sha256", "visibility_lane",
    }
)


def assert_visibility_lane(field: str) -> None:
    """Reject label-only or audit-only fields entering a visible dataset path."""

    if field in LABEL_ONLY_FIELDS or field in AUDIT_ONLY_FIELDS:
        raise ProgramContractError(
            "field %r belongs to the label/evaluator lane and must not enter "
            "an inference-visible dataset path" % field
        )


def protected_refs_policy(operation: str, operand: str) -> Mapping[str, object]:
    """Return the execution constraint record for one legal call.

    Non-selected components are always hard-protected.  For ADD the union
    update additionally preserves every existing foreground voxel.
    """

    validate_legal_call(operation, "CREATE_NEW" if operand == NEW_CUE_SENTINEL else "GROW_LOCAL", operand)
    return {
        "protected_complement": True,
        "monotone_update": operation == "ADD",
        "selected_component_scope": operand != NEW_CUE_SENTINEL,
        "new_cue_channel_all_zero": operand == NEW_CUE_SENTINEL,
    }
