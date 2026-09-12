"""Pure reduction of observed event facts; repeated claims cannot reset novelty."""

from dataclasses import dataclass, field

from app.services.theme_evaluation.translation_normalization import (
    normalize_text,
    quantity_counter,
)

STATUS_STRENGTH = {"unknown": 0, "rumored": 1, "announced": 2, "confirmed": 3}
CONFLICTING_STATUSES = {"denied", "cancelled"}


def quantity_signature(quantities):
    return tuple(
        sorted(
            quantity_counter(normalize_text(" ".join(quantities)).quantities).items()
        )
    )


@dataclass
class EventState:
    quantities: dict[str, set[tuple]] = field(default_factory=dict)

    @property
    def statuses(self):
        return self.quantities.keys()

    def observe(self, facts):
        self.quantities.setdefault(facts["status"], set()).add(
            quantity_signature(facts.get("quantities", []))
        )

    def classify(self, facts, *, correction=None):
        # Source corrections retain their contradiction even when the corrected
        # claim is no longer part of the current event state.
        if (
            correction is not None
            and correction.status != facts.status
            and ({correction.status, facts.status} & CONFLICTING_STATUSES)
        ):
            return "contradiction"
        if not self.quantities:
            return "new_event" if facts.reference or facts.event_time else "uncertain"
        strength = max(
            (STATUS_STRENGTH.get(status, 0) for status in self.statuses), default=0
        )
        if facts.status in STATUS_STRENGTH and STATUS_STRENGTH[facts.status] < strength:
            return "additional_detail"
        quantities = quantity_signature(facts.quantities)
        previous_quantities = self.quantities.get(facts.status, set())
        if quantities in previous_quantities:
            return "repeated_coverage"
        if facts.status not in self.statuses:
            if (
                facts.status in CONFLICTING_STATUSES
                or self.statuses & CONFLICTING_STATUSES
            ):
                return "contradiction"
            return (
                "material_update" if facts.status != "unknown" else "additional_detail"
            )
        return (
            "material_update"
            if quantities and any(previous_quantities)
            else "additional_detail"
        )


def classify(facts, previous):
    state = EventState()
    for observation in previous:
        state.observe(observation.facts)
    return state.classify(facts)
