"""Select current evidence explicitly while retaining immutable attempt history."""

from .preparation_records import PreparationManifest


class PreparationState:
    def __init__(self, base, store, handoff, prior_id=None):
        self.store = store
        self.manifest = (
            store.load(base, prior_id)
            if prior_id
            else PreparationManifest(bundle_id=base.name, handoff=handoff)
        )
        if self.manifest.handoff != handoff:
            raise ValueError("prior_handoff_mismatch")

    def record(self, binding):
        result = self.store.load_result(binding.result_id)
        if binding.stage != result.stage:
            raise ValueError("preparation_stage_mismatch")
        history = {b.binding_id: b for b in self.manifest.bindings}
        old_id = self.manifest.current.get(binding.slot_id)
        old = self.store.load_result(history[old_id].result_id) if old_id else None
        history[binding.binding_id] = binding
        current = dict(self.manifest.current)
        # An unavailable retry cannot discard usable evidence for the same input.
        # Changed captured inputs start a new version, even when processing fails.
        retain = (
            old
            and old.status != "unavailable"
            and result.status == "unavailable"
            and (
                not result.has_input
                or result.request.input_sha256 == old.request.input_sha256
            )
        )
        if not retain:
            current[binding.slot_id] = binding.binding_id
        roots = {
            (
                history[i].source_kind,
                history[i].source_id,
                history[i].result_id,
                history[i].input_locator,
            )
            for i in current.values()
            if history[i].parent_result_id is None
        }
        current = {
            slot: selected
            for slot, selected in current.items()
            if history[selected].parent_result_id is None
            or (
                history[selected].source_kind,
                history[selected].source_id,
                history[selected].parent_result_id,
                history[selected].input_locator,
            )
            in roots
        }
        self.manifest = PreparationManifest(
            bundle_id=self.manifest.bundle_id,
            handoff=self.manifest.handoff,
            bindings=list(history.values()),
            current=current,
        )
