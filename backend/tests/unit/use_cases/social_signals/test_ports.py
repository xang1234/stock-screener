from inspect import signature

from app.use_cases.social_signals.ports import (
    ConfirmationReader,
    ProviderReadLease,
    PublishedReader,
    SocialDispatcher,
    SocialProvider,
    SocialWriter,
)


def test_provider_exposes_the_single_provider_neutral_ingestion_method():
    public_operations = {
        name for name, value in SocialProvider.__dict__.items()
        if not name.startswith("_") and callable(value)
    }
    assert public_operations == {"read_source"}
    assert list(signature(SocialProvider.read_source).parameters) == ["self", "request"]


def test_writer_contract_separates_observation_persistence_from_publication():
    assert {"persist_observations", "publish"} <= set(SocialWriter.__dict__)
    assert list(signature(SocialWriter.publish).parameters) == [
        "self",
        "run_id",
        "expected_mode_version",
    ]


def test_remaining_ports_publish_only_required_operations():
    assert {"read"} <= set(ConfirmationReader.__dict__)
    assert {"queue"} <= set(PublishedReader.__dict__)
    assert {"acquire", "release"} <= set(ProviderReadLease.__dict__)
    assert {"refresh", "test_source"} <= set(SocialDispatcher.__dict__)
