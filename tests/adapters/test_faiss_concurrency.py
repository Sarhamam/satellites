"""Tests for FAISS adapter thread safety under concurrent operations."""

import asyncio
import tempfile
from pathlib import Path

import pytest

from data_layer.adapters.faiss.adapter import VectorIndex


@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as td:
        yield Path(td)


@pytest.fixture
def vector_index(temp_dir: Path) -> VectorIndex:
    return VectorIndex(index_path=temp_dir / "vectors", dimension=128)


async def test_concurrent_upserts_maintain_consistency(vector_index: VectorIndex):
    """Test that concurrent upserts don't cause ID mapping corruption."""
    namespace = "test"
    num_vectors = 100

    # Create test vectors
    def make_vector(i: int) -> list[float]:
        return [float(i)] + [0.0] * 127

    # Upsert many vectors concurrently
    tasks = [
        vector_index.upsert(
            namespace=namespace,
            vector_id=f"vec-{i}",
            vector=make_vector(i),
            metadata={"index": i},
        )
        for i in range(num_vectors)
    ]

    await asyncio.gather(*tasks)

    # Verify all vectors are present with correct IDs and metadata
    query_vector = make_vector(0)
    results = await vector_index.query(namespace=namespace, vector=query_vector, top_k=num_vectors)

    # Should have all vectors
    assert len(results) == num_vectors

    # Check that IDs are unique
    result_ids = [r.id for r in results]
    assert len(result_ids) == len(set(result_ids)), "Duplicate IDs found!"

    # Check that all expected IDs are present
    expected_ids = {f"vec-{i}" for i in range(num_vectors)}
    actual_ids = set(result_ids)
    assert actual_ids == expected_ids, f"Missing IDs: {expected_ids - actual_ids}"

    # Check metadata is correct
    for result in results:
        expected_index = int(result.id.split("-")[1])
        assert result.metadata == {"index": expected_index}


async def test_concurrent_upsert_same_id(vector_index: VectorIndex):
    """Test that concurrent upserts to the same ID are handled correctly."""
    namespace = "test"
    vector_id = "same-vec"
    num_concurrent = 50

    # Create different vectors
    def make_vector(i: int) -> list[float]:
        return [float(i)] + [0.0] * 127

    # Upsert the same ID concurrently with different vectors
    tasks = [
        vector_index.upsert(
            namespace=namespace,
            vector_id=vector_id,
            vector=make_vector(i),
            metadata={"version": i},
        )
        for i in range(num_concurrent)
    ]

    await asyncio.gather(*tasks)

    # Should have exactly one vector (last writer wins)
    results = await vector_index.query(namespace=namespace, vector=make_vector(0), top_k=10)
    assert len(results) == 1
    assert results[0].id == vector_id
    # Metadata should be from one of the concurrent writes
    assert "version" in results[0].metadata


async def test_concurrent_operations_mixed(vector_index: VectorIndex):
    """Test concurrent upserts, queries, and deletes."""
    namespace = "test"
    num_vectors = 50

    def make_vector(i: int) -> list[float]:
        return [float(i)] + [0.0] * 127

    # First, insert some initial vectors
    for i in range(num_vectors):
        await vector_index.upsert(
            namespace=namespace,
            vector_id=f"vec-{i}",
            vector=make_vector(i),
            metadata={"index": i},
        )

    # Now do mixed operations concurrently
    upsert_tasks = [
        vector_index.upsert(
            namespace=namespace,
            vector_id=f"new-{i}",
            vector=make_vector(i + 1000),
            metadata={"new": True},
        )
        for i in range(20)
    ]

    query_tasks = [
        vector_index.query(namespace=namespace, vector=make_vector(i), top_k=5)
        for i in range(20)
    ]

    delete_tasks = [
        vector_index.delete(namespace=namespace, vector_id=f"vec-{i}")
        for i in range(10)  # Delete first 10
    ]

    # Run all concurrently
    await asyncio.gather(*upsert_tasks, *query_tasks, *delete_tasks)

    # Verify consistency: should have original - deleted + new
    # Original: 50, Deleted: 10, Added: 20 = 60 total
    all_results = await vector_index.query(
        namespace=namespace, vector=make_vector(0), top_k=100
    )

    assert len(all_results) == 60

    # Verify deleted vectors are gone
    result_ids = {r.id for r in all_results}
    for i in range(10):
        assert f"vec-{i}" not in result_ids

    # Verify new vectors are present
    for i in range(20):
        assert f"new-{i}" in result_ids
