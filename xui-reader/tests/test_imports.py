"""Import smoke tests for scaffold modules."""

from importlib import import_module

from xui_reader.models import Checkpoint


MODULES = [
    "xui_reader.config",
    "xui_reader.models",
    "xui_reader.errors",
    "xui_reader.logging",
    "xui_reader.browser.policy",
    "xui_reader.browser.session",
    "xui_reader.collectors.base",
    "xui_reader.collectors.timeline",
    "xui_reader.extract.base",
    "xui_reader.extract.normalize",
    "xui_reader.extract.selectors",
    "xui_reader.extract.tweets",
    "xui_reader.store.base",
    "xui_reader.render.base",
    "xui_reader.scheduler.base",
    "xui_reader.diagnostics.base",
]


def test_core_modules_import_cleanly() -> None:
    for module in MODULES:
        assert import_module(module) is not None


def test_checkpoint_defaults_updated_at_timestamp() -> None:
    checkpoint = Checkpoint(source_id="src-1")
    assert checkpoint.updated_at is not None
