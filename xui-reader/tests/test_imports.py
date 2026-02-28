"""Import smoke tests for scaffold modules."""

from importlib import import_module


MODULES = [
    "xui_reader.config",
    "xui_reader.models",
    "xui_reader.errors",
    "xui_reader.logging",
    "xui_reader.browser.session",
    "xui_reader.collectors.base",
    "xui_reader.extract.base",
    "xui_reader.store.base",
    "xui_reader.render.base",
    "xui_reader.scheduler.base",
    "xui_reader.diagnostics.base",
]


def test_core_modules_import_cleanly() -> None:
    for module in MODULES:
        assert import_module(module) is not None
