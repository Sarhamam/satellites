"""FAISS vector index adapter with async wrapper and metadata sidecar."""

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import faiss
import numpy as np


@dataclass(frozen=True)
class VectorMatch:
    """Result from vector similarity search."""

    id: str
    score: float
    metadata: Optional[Mapping[str, Any]] = None


class VectorIndex:
    """Async wrapper for FAISS with metadata storage."""

    def __init__(self, index_path: Path, dimension: int) -> None:
        self._index_path = index_path
        self._dimension = dimension
        self._namespaces: dict[str, faiss.IndexFlatIP] = {}
        self._id_to_idx: dict[str, dict[str, int]] = {}  # namespace -> {id -> idx}
        self._idx_to_id: dict[str, dict[int, str]] = {}  # namespace -> {idx -> id}
        self._metadata: dict[str, dict[str, Mapping[str, Any]]] = {}  # namespace -> {id -> meta}
        self._index_path.mkdir(parents=True, exist_ok=True)

    def _get_or_create_namespace(self, namespace: str) -> faiss.IndexFlatIP:
        if namespace not in self._namespaces:
            self._namespaces[namespace] = faiss.IndexFlatIP(self._dimension)
            self._id_to_idx[namespace] = {}
            self._idx_to_id[namespace] = {}
            self._metadata[namespace] = {}
        return self._namespaces[namespace]

    async def upsert(
        self,
        namespace: str,
        vector_id: str,
        vector: Sequence[float],
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> None:
        """Insert or update a vector with optional metadata."""

        def _upsert() -> None:
            # Delete existing if present
            if namespace in self._id_to_idx and vector_id in self._id_to_idx[namespace]:
                self._delete_sync(namespace, vector_id)

            index = self._get_or_create_namespace(namespace)
            vec = np.array([vector], dtype=np.float32)
            faiss.normalize_L2(vec)  # Normalize for cosine similarity

            idx = index.ntotal
            index.add(vec)

            self._id_to_idx[namespace][vector_id] = idx
            self._idx_to_id[namespace][idx] = vector_id
            if metadata:
                self._metadata[namespace][vector_id] = metadata

        await asyncio.to_thread(_upsert)

    def _delete_sync(self, namespace: str, vector_id: str) -> bool:
        """Sync delete - rebuilds index without the deleted vector."""
        if namespace not in self._id_to_idx:
            return False
        if vector_id not in self._id_to_idx[namespace]:
            return False

        # Get all vectors except the one to delete
        old_index = self._namespaces[namespace]
        old_id_to_idx = self._id_to_idx[namespace]
        old_idx_to_id = self._idx_to_id[namespace]

        # Rebuild
        new_index = faiss.IndexFlatIP(self._dimension)
        new_id_to_idx: dict[str, int] = {}
        new_idx_to_id: dict[int, str] = {}

        for vid, old_idx in old_id_to_idx.items():
            if vid == vector_id:
                continue
            vec = old_index.reconstruct(old_idx).reshape(1, -1)
            new_idx = new_index.ntotal
            new_index.add(vec)
            new_id_to_idx[vid] = new_idx
            new_idx_to_id[new_idx] = vid

        self._namespaces[namespace] = new_index
        self._id_to_idx[namespace] = new_id_to_idx
        self._idx_to_id[namespace] = new_idx_to_id
        self._metadata[namespace].pop(vector_id, None)

        return True

    async def delete(self, namespace: str, vector_id: str) -> bool:
        """Delete a vector by ID."""
        return await asyncio.to_thread(self._delete_sync, namespace, vector_id)

    async def query(
        self,
        namespace: str,
        vector: Sequence[float],
        top_k: int = 10,
    ) -> list[VectorMatch]:
        """Find nearest neighbors."""

        def _query() -> list[VectorMatch]:
            if namespace not in self._namespaces:
                return []

            index = self._namespaces[namespace]
            if index.ntotal == 0:
                return []

            vec = np.array([vector], dtype=np.float32)
            faiss.normalize_L2(vec)

            k = min(top_k, index.ntotal)
            scores, indices = index.search(vec, k)

            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx == -1:
                    continue
                vid = self._idx_to_id[namespace].get(int(idx))
                if vid:
                    meta = self._metadata[namespace].get(vid)
                    results.append(VectorMatch(id=vid, score=float(score), metadata=meta))

            return results

        return await asyncio.to_thread(_query)

    async def save(self) -> None:
        """Persist all indexes and metadata to disk."""

        def _save() -> None:
            for namespace, index in self._namespaces.items():
                ns_path = self._index_path / namespace
                ns_path.mkdir(exist_ok=True)

                faiss.write_index(index, str(ns_path / "index.faiss"))

                state = {
                    "id_to_idx": self._id_to_idx[namespace],
                    "idx_to_id": {str(k): v for k, v in self._idx_to_id[namespace].items()},
                    "metadata": dict(self._metadata[namespace]),
                }
                with open(ns_path / "state.json", "w") as f:
                    json.dump(state, f)

        await asyncio.to_thread(_save)

    async def load(self) -> None:
        """Load all indexes and metadata from disk."""

        def _load() -> None:
            if not self._index_path.exists():
                return

            for ns_path in self._index_path.iterdir():
                if not ns_path.is_dir():
                    continue

                index_file = ns_path / "index.faiss"
                state_file = ns_path / "state.json"

                if not index_file.exists() or not state_file.exists():
                    continue

                namespace = ns_path.name
                self._namespaces[namespace] = faiss.read_index(str(index_file))

                with open(state_file) as f:
                    state = json.load(f)

                self._id_to_idx[namespace] = state["id_to_idx"]
                self._idx_to_id[namespace] = {int(k): v for k, v in state["idx_to_id"].items()}
                self._metadata[namespace] = state.get("metadata", {})

        await asyncio.to_thread(_load)
