"""Tests for FAISS vector index adapter."""

import pytest
import tempfile
from pathlib import Path
from data_layer.adapters.faiss.adapter import VectorIndex, VectorMatch


@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as td:
        yield Path(td)


@pytest.fixture
def vector_index(temp_dir: Path) -> VectorIndex:
    return VectorIndex(index_path=temp_dir, dimension=4)


async def test_upsert_and_query(vector_index: VectorIndex):
    await vector_index.upsert(
        namespace="test",
        vector_id="vec-1",
        vector=[1.0, 0.0, 0.0, 0.0],
        metadata={"label": "first"},
    )

    results = await vector_index.query(
        namespace="test",
        vector=[1.0, 0.0, 0.0, 0.0],
        top_k=1,
    )

    assert len(results) == 1
    assert results[0].id == "vec-1"
    assert results[0].metadata == {"label": "first"}


async def test_query_returns_nearest(vector_index: VectorIndex):
    await vector_index.upsert("test", "vec-1", [1.0, 0.0, 0.0, 0.0])
    await vector_index.upsert("test", "vec-2", [0.0, 1.0, 0.0, 0.0])
    await vector_index.upsert("test", "vec-3", [0.9, 0.1, 0.0, 0.0])  # closest to vec-1

    results = await vector_index.query("test", [1.0, 0.0, 0.0, 0.0], top_k=2)

    assert len(results) == 2
    # vec-1 should be first (exact match), vec-3 second (closest)
    assert results[0].id == "vec-1"
    assert results[1].id == "vec-3"


async def test_delete(vector_index: VectorIndex):
    await vector_index.upsert("test", "vec-1", [1.0, 0.0, 0.0, 0.0])
    deleted = await vector_index.delete("test", "vec-1")
    assert deleted is True

    results = await vector_index.query("test", [1.0, 0.0, 0.0, 0.0], top_k=1)
    assert len(results) == 0


async def test_delete_nonexistent(vector_index: VectorIndex):
    deleted = await vector_index.delete("test", "nonexistent")
    assert deleted is False


async def test_separate_namespaces(vector_index: VectorIndex):
    await vector_index.upsert("ns1", "vec-1", [1.0, 0.0, 0.0, 0.0])
    await vector_index.upsert("ns2", "vec-1", [0.0, 1.0, 0.0, 0.0])

    results_ns1 = await vector_index.query("ns1", [1.0, 0.0, 0.0, 0.0], top_k=1)
    results_ns2 = await vector_index.query("ns2", [1.0, 0.0, 0.0, 0.0], top_k=1)

    # Same ID, different namespaces, different vectors
    assert results_ns1[0].id == "vec-1"
    assert results_ns2[0].id == "vec-1"


async def test_save_and_load(temp_dir: Path):
    index1 = VectorIndex(index_path=temp_dir, dimension=4)
    await index1.upsert("test", "vec-1", [1.0, 0.0, 0.0, 0.0], {"label": "saved"})
    await index1.save()

    # Create new instance and load
    index2 = VectorIndex(index_path=temp_dir, dimension=4)
    await index2.load()

    results = await index2.query("test", [1.0, 0.0, 0.0, 0.0], top_k=1)
    assert len(results) == 1
    assert results[0].id == "vec-1"
    assert results[0].metadata == {"label": "saved"}


def test_vector_match_dataclass():
    match = VectorMatch(id="vec-1", score=0.95, metadata={"key": "value"})
    assert match.id == "vec-1"
    assert match.score == 0.95
    assert match.metadata == {"key": "value"}
