"""Test that public API is importable from package root."""


def test_main_imports():
    from data_layer import (
        DataLayer,
        DataLayerConfig,
        RedisConfig,
        PostgresConfig,
        ElasticsearchConfig,
        FaissConfig,
        DataEvent,
        InProcessEventBus,
        Health,
        VectorIndex,
        VectorMatch,
    )

    # Just verify they're importable
    assert DataLayer is not None
    assert DataLayerConfig is not None
    assert DataEvent is not None
