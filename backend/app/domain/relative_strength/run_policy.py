from collections.abc import Mapping


BALANCED_RS_PRICE_BASIS = "adj_close_only"


def balanced_run_has_required_price_basis(run) -> bool:
    diagnostics = getattr(run, "diagnostics_json", None)
    return (
        isinstance(diagnostics, Mapping)
        and diagnostics.get("price_basis") == BALANCED_RS_PRICE_BASIS
    )
