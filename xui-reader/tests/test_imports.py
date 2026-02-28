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
    "xui_reader.store.checkpoints",
    "xui_reader.store.sqlite",
    "xui_reader.render.base",
    "xui_reader.scheduler.base",
    "xui_reader.scheduler.merge",
    "xui_reader.scheduler.read",
    "xui_reader.scheduler.timing",
    "xui_reader.scheduler.watch",
    "xui_reader.diagnostics.base",
    "xui_reader.diagnostics.artifacts",
    "xui_reader.diagnostics.doctor",
    "xui_reader.diagnostics.events",
]


def test_core_modules_import_cleanly() -> None:
    for module in MODULES:
        assert import_module(module) is not None


def test_checkpoint_defaults_updated_at_timestamp() -> None:
    checkpoint = Checkpoint(source_id="src-1")
    assert checkpoint.updated_at is not None
