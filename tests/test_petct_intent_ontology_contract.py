from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest


PROJECT = Path(__file__).resolve().parents[1]
SCRIPTS = PROJECT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from common.petct_models import (  # noqa: E402
    LEGAL_GOALS as MODEL_GOALS,
    LEGAL_GOAL_SLOTS,
    editor_condition_ids,
)
from common.petct_route_a_core import (  # noqa: E402
    LEGAL_JOINT_GOALS,
    OPERATIONS,
    SCOPES,
    TARGETS,
    ContractError,
    intent_slots_from_goal,
    joint_goal,
)
from data.build_petct_scribble_episode import canonical_intent_frame  # noqa: E402
from data.materialize_petct_pilot6_states import GOALS as PILOT6_GOALS  # noqa: E402
from editor.infer_petct_residual_editor import LEGAL_PREDICTIONS  # noqa: E402
from p2t.validate_petct_intent_response import GOAL_MAPPING  # noqa: E402


FROZEN_GOALS = (
    "ADD_SAME_LOCAL",
    "REMOVE_SAME_LOCAL",
    "ADD_SAME_COMPLETE",
    "REMOVE_SAME_COMPLETE",
    "ADD_NEW_COMPLETE",
    "REMOVE_NEW_COMPLETE",
)
FROZEN_SLOTS = (
    ("ADD", "SAME", "LOCAL"),
    ("REMOVE", "SAME", "LOCAL"),
    ("ADD", "SAME", "COMPLETE"),
    ("REMOVE", "SAME", "COMPLETE"),
    ("ADD", "NEW", "COMPLETE"),
    ("REMOVE", "NEW", "COMPLETE"),
)


def _json(path: str) -> dict:
    return json.loads((PROJECT / path).read_text(encoding="utf-8"))


def test_all_machine_contracts_share_the_frozen_six_class_order() -> None:
    config = _json("configs/petct_route_a_experiment.json")
    protocol = _json("protocols/petct_codex_prompt_v2.json")
    schema = _json("schemas/petct_intent_v2.schema.json")
    schema_goals = tuple(
        goal for goal in schema["properties"]["goal"]["enum"] if goal is not None
    )

    assert OPERATIONS == ("ADD", "REMOVE")
    assert TARGETS == ("SAME", "NEW")
    assert SCOPES == ("LOCAL", "COMPLETE")
    assert LEGAL_JOINT_GOALS == FROZEN_GOALS
    assert MODEL_GOALS == FROZEN_GOALS
    assert PILOT6_GOALS == FROZEN_GOALS
    assert tuple(GOAL_MAPPING) == FROZEN_GOALS
    assert tuple(LEGAL_PREDICTIONS) == FROZEN_GOALS
    assert tuple(protocol["goal_contract"]) == FROZEN_GOALS
    assert tuple(config["intent_ontology"]["legal_joint_goals"]) == FROZEN_GOALS
    assert schema_goals == FROZEN_GOALS


def test_every_joint_goal_has_exact_operation_target_scope_slots() -> None:
    expected_id_slots = (
        (0, 0, 0),
        (1, 0, 0),
        (0, 0, 1),
        (1, 0, 1),
        (0, 1, 1),
        (1, 1, 1),
    )
    assert LEGAL_GOAL_SLOTS == expected_id_slots
    for goal, slots in zip(FROZEN_GOALS, FROZEN_SLOTS):
        assert intent_slots_from_goal(goal) == slots
        assert GOAL_MAPPING[goal] == slots
        assert LEGAL_PREDICTIONS[goal] == slots
        assert joint_goal(*slots) == goal
        frame = canonical_intent_frame(goal)
        assert (frame["operation"], frame["target"], frame["scope"]) == slots


@pytest.mark.parametrize("operation", ("ADD", "REMOVE"))
def test_new_local_is_structurally_illegal_everywhere(operation: str) -> None:
    with pytest.raises(ContractError, match="NEW_LOCAL"):
        joint_goal(operation, "NEW", "LOCAL")
    with pytest.raises(ValueError, match="NEW_LOCAL"):
        editor_condition_ids(operation, "NEW", "LOCAL", "CORRECT")
    config = _json("configs/petct_route_a_experiment.json")
    assert f"{operation}_NEW_LOCAL" in config["intent_ontology"][
        "forbidden_joint_goals"
    ]


def test_retired_relation_and_target_labels_are_absent_from_production_contracts() -> None:
    retired_target = re.compile(r"\b(?:ATTACHED|STANDALONE)\b")
    retired_field = re.compile(r"[\"']relation[\"']")
    offenders: list[str] = []
    for root_name in ("configs", "protocols", "schemas", "scripts"):
        for path in (PROJECT / root_name).rglob("*"):
            if not path.is_file() or path.suffix not in {".json", ".py", ".sh", ".yaml", ".yml"}:
                continue
            text = path.read_text(encoding="utf-8")
            if retired_target.search(text) or retired_field.search(text):
                offenders.append(str(path.relative_to(PROJECT)))
    assert offenders == []
