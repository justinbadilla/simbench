"""SimUser spec: what a simulated customer wants, knows, and must not reveal.

Everything the checks and the state tracker need is written here as data, so a
new scenario is a new file and the code stays the same. Matching is plain
case-insensitive substring search. That is deterministic and easy to explain,
and it is crude: a paraphrase that avoids every marker is not seen. Choose
markers with that in mind.
"""

from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator


def normalize(text: str) -> str:
    """Lowercase ``text`` and collapse runs of whitespace into single spaces.

    This is the single normalization rule for the project. Markers are
    normalized when a spec loads, and message text is normalized before
    matching, so comparisons are case- and spacing-insensitive.

    Returns:
        The normalized string. An empty or whitespace-only input gives ``""``.

    Called by:
        ``_clean_phrases`` (spec load time), ``SimUserState.observe`` and the
        goal checks in ``checks.py`` (match time).
    """
    return " ".join(text.lower().split())


def _clean_phrases(values: list[str]) -> list[str]:
    """Normalize every marker or phrase in a list and reject blank entries.

    Raises:
        ValueError: If any entry is empty after normalization. A blank marker
            would match every message and silently break the checks.

    Called by:
        The field validators on ``Fact``, ``UnknownFact`` and ``GoalSpec``.
    """
    cleaned = []
    for value in values:
        phrase = normalize(value)
        if not phrase:
            raise ValueError("markers and phrases must contain text")
        cleaned.append(phrase)
    return cleaned


class Fact(BaseModel):
    """Something the SimUser knows.

    ``volunteer`` facts may be said at any time. ``on_request`` facts are held
    back until an agent message contains one of ``unlock_phrases``. ``markers``
    are the strings that count as "this fact was said" in a customer message.
    """

    id: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    text: str = Field(min_length=1)
    markers: list[str] = Field(min_length=1)
    disclosure: Literal["volunteer", "on_request"] = "volunteer"
    unlock_phrases: list[str] = Field(default_factory=list)
    required: bool = False

    @field_validator("markers", "unlock_phrases")
    @classmethod
    def phrases_are_clean(cls, values: list[str]) -> list[str]:
        """Pydantic hook: normalize ``markers`` and ``unlock_phrases`` on load."""
        return _clean_phrases(values)

    @model_validator(mode="after")
    def disclosure_rules_agree(self) -> "Fact":
        """Pydantic hook: reject combinations of fields that contradict each other.

        Rules enforced:
            * ``on_request`` needs at least one unlock phrase, or it could never be unlocked.
            * ``volunteer`` must not have unlock phrases, since they would be ignored.
            * ``on_request`` cannot be ``required``, because it is only sayable after the
              agent asks. ``answered_when_asked`` covers that case instead.

        Raises:
            ValueError: On any violation, naming the fact id.
        """
        if self.disclosure == "on_request" and not self.unlock_phrases:
            raise ValueError(f"fact {self.id!r} is on_request and needs unlock_phrases")
        if self.disclosure == "volunteer" and self.unlock_phrases:
            raise ValueError(f"fact {self.id!r} is volunteered, so unlock_phrases would be ignored")
        if self.disclosure == "on_request" and self.required:
            raise ValueError(
                f"fact {self.id!r} is on_request, so it cannot be required up front; "
                "the answered_when_asked check covers it"
            )
        return self


class UnknownFact(BaseModel):
    """Something the SimUser must not know until the agent says it.

    Saying a marker before an agent message has contained it counts as using a
    fact the SimUser should not have. Repeating it afterward is fine.
    """

    id: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    label: str = Field(min_length=1)
    markers: list[str] = Field(min_length=1)

    @field_validator("markers")
    @classmethod
    def markers_are_clean(cls, values: list[str]) -> list[str]:
        """Pydantic hook: normalize ``markers`` on load."""
        return _clean_phrases(values)


class GoalSpec(BaseModel):
    """The goal text goes in the prompt. The markers feed the goal checks.

    Leave ``markers`` empty to skip the goal-stated check, and
    ``abandon_markers`` empty to skip the abandonment check.
    """

    text: str = Field(min_length=1)
    markers: list[str] = Field(default_factory=list)
    abandon_markers: list[str] = Field(default_factory=list)

    @field_validator("markers", "abandon_markers")
    @classmethod
    def phrases_are_clean(cls, values: list[str]) -> list[str]:
        """Pydantic hook: normalize ``markers`` and ``abandon_markers`` on load."""
        return _clean_phrases(values)


class SimUserSpec(BaseModel):
    """The full description of one simulated customer.

    Combines the goal, tone, stopping rule, known facts, and unknown facts. It
    lives on ``Scenario.sim_user``, drives prompt building (``sim_user.py``),
    state tracking (``state.py``) and the checks (``checks.py``), and is copied
    onto each ``Transcript`` so a stored run records the spec it used.
    """

    goal: GoalSpec
    style: str = Field(min_length=1)
    done_when: str = Field(min_length=1)
    facts: list[Fact] = Field(min_length=1)
    unknown_facts: list[UnknownFact] = Field(default_factory=list)

    @model_validator(mode="after")
    def ids_are_unique(self) -> "SimUserSpec":
        """Pydantic hook: require unique ids across both ``facts`` and ``unknown_facts``.

        Ids key the state dictionaries and the event log, so a duplicate would
        merge two different facts.

        Raises:
            ValueError: Listing the duplicated ids.
        """
        ids = [fact.id for fact in self.facts] + [item.id for item in self.unknown_facts]
        duplicates = sorted({item for item in ids if ids.count(item) > 1})
        if duplicates:
            raise ValueError(f"duplicate fact ids: {', '.join(duplicates)}")
        return self

    def fact(self, fact_id: str) -> Fact:
        """Look up a known fact by id.

        Raises:
            KeyError: If no fact has that id. Unknown facts are not searched here.

        Called by:
            ``checks.py`` when it turns a logged event back into fact text.
        """
        for fact in self.facts:
            if fact.id == fact_id:
                return fact
        raise KeyError(fact_id)
