"""The spec rejects configurations that would make the checks meaningless."""

import copy

import pytest
from pydantic import ValidationError

from simbench.scenario import load_scenario
from simbench.spec import SimUserSpec

from conftest import SPEC_DATA


def variant(mutate) -> dict:
    """Return a copy of the base spec data after applying ``mutate`` to it."""
    data = copy.deepcopy(SPEC_DATA)
    mutate(data)
    return data


def test_markers_and_phrases_are_normalized(spec: SimUserSpec) -> None:
    data = variant(lambda d: d["facts"][2].update(markers=["  Jordan.Lee@Example.COM "]))
    assert SimUserSpec.model_validate(data).facts[2].markers == ["jordan.lee@example.com"]


def test_on_request_fact_needs_unlock_phrases() -> None:
    data = variant(lambda d: d["facts"][2].update(unlock_phrases=[]))
    with pytest.raises(ValidationError, match="needs unlock_phrases"):
        SimUserSpec.model_validate(data)


def test_volunteered_fact_cannot_carry_unlock_phrases() -> None:
    data = variant(lambda d: d["facts"][0].update(unlock_phrases=["order"]))
    with pytest.raises(ValidationError, match="would be ignored"):
        SimUserSpec.model_validate(data)


def test_on_request_fact_cannot_be_required() -> None:
    data = variant(lambda d: d["facts"][2].update(required=True))
    with pytest.raises(ValidationError, match="cannot be required"):
        SimUserSpec.model_validate(data)


def test_blank_marker_is_rejected() -> None:
    data = variant(lambda d: d["facts"][0].update(markers=["  "]))
    with pytest.raises(ValidationError, match="must contain text"):
        SimUserSpec.model_validate(data)


def test_duplicate_ids_across_facts_and_unknown_facts_are_rejected() -> None:
    data = variant(lambda d: d["unknown_facts"][0].update(id="order_number"))
    with pytest.raises(ValidationError, match="duplicate fact ids: order_number"):
        SimUserSpec.model_validate(data)


def test_fact_ids_must_be_slugs() -> None:
    data = variant(lambda d: d["facts"][0].update(id="Order Number"))
    with pytest.raises(ValidationError):
        SimUserSpec.model_validate(data)


def test_bundled_scenario_spec_is_valid_and_exercises_every_feature() -> None:
    bundled = load_scenario("late_refund").sim_user
    assert bundled is not None
    assert any(fact.disclosure == "on_request" for fact in bundled.facts)
    assert any(fact.required for fact in bundled.facts)
    assert bundled.unknown_facts
    assert bundled.goal.markers and bundled.goal.abandon_markers
