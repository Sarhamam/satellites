"""DataLayer façade - unified access to all backends."""

from types import TracebackType
from typing import Optional, Self

import redis.asyncio as redis_lib
import asyncpg
from elasticsearch import AsyncElasticsearch

from data_layer.config import DataLayerConfig
from data_layer.core.events import InProcessEventBus
from data_layer.core.health import Health
from data_layer.adapters.faiss.adapter import VectorIndex


class DataLayer:
    """Unified façade - lifecycle + access to all backends."""

    def __init__(self, config: DataLayerConfig) -> None:
        self._config = config
        self._redis: Optional[redis_lib.Redis] = None
        self._postgres: Optional[asyncpg.Pool] = None
        self._elasticsearch: Optional[AsyncElasticsearch] = None
        self._vectors = VectorIndex(
            index_path=config.faiss.index_path,
            dimension=config.faiss.dimension,
        )
        self._events = InProcessEventBus()

    @property
    def redis(self) -> redis_lib.Redis:
        """Native redis-py async client."""
        if self._redis is None:
            raise RuntimeError("DataLayer not started. Call start() first.")
        return self._redis

    @property
    def postgres(self) -> asyncpg.Pool:
        """Native asyncpg pool."""
        if self._postgres is None:
            raise RuntimeError("DataLayer not started. Call start() first.")
        return self._postgres

    @property
    def elasticsearch(self) -> AsyncElasticsearch:
        """Native Elasticsearch async client."""
        if self._elasticsearch is None:
            raise RuntimeError("DataLayer not started. Call start() first.")
        return self._elasticsearch

    @property
    def vectors(self) -> VectorIndex:
        """FAISS vector index."""
        return self._vectors

    @property
    def events(self) -> InProcessEventBus:
        """Event bus for pub/sub."""
        return self._events

    async def _connect_redis(self) -> None:
        cfg = self._config.redis
        self._redis = redis_lib.Redis(
            host=cfg.host,
            port=cfg.port,
            db=cfg.db,
            password=cfg.password,
        )

    async def _connect_postgres(self) -> None:
        cfg = self._config.postgres
        self._postgres = await asyncpg.create_pool(
            host=cfg.host,
            port=cfg.port,
            database=cfg.database,
            user=cfg.user,
            password=cfg.password,
            min_size=cfg.min_connections,
            max_size=cfg.max_connections,
        )

    async def _connect_elasticsearch(self) -> None:
        cfg = self._config.elasticsearch
        self._elasticsearch = AsyncElasticsearch(hosts=cfg.hosts)

    async def _disconnect_redis(self) -> None:
        if self._redis:
            await self._redis.aclose()
            self._redis = None

    async def _disconnect_postgres(self) -> None:
        if self._postgres:
            await self._postgres.close()
            self._postgres = None

    async def _disconnect_elasticsearch(self) -> None:
        if self._elasticsearch:
            await self._elasticsearch.close()
            self._elasticsearch = None

    async def start(self) -> None:
        """Connect all backends."""
        await self._connect_redis()
        await self._connect_postgres()
        await self._connect_elasticsearch()
        await self._vectors.load()

    async def stop(self) -> None:
        """Graceful shutdown."""
        await self._vectors.save()
        await self._disconnect_redis()
        await self._disconnect_postgres()
        await self._disconnect_elasticsearch()

    async def health(self) -> Health:
        """Aggregate health check across all backends."""
        details: dict[str, str] = {}
        all_ok = True

        # Redis
        try:
            if self._redis:
                await self._redis.ping()
                details["redis"] = "ok"
            else:
                details["redis"] = "not connected"
                all_ok = False
        except Exception as e:
            details["redis"] = f"error: {e}"
            all_ok = False

        # Postgres
        try:
            if self._postgres:
                await self._postgres.fetchval("SELECT 1")
                details["postgres"] = "ok"
            else:
                details["postgres"] = "not connected"
                all_ok = False
        except Exception as e:
            details["postgres"] = f"error: {e}"
            all_ok = False

        # Elasticsearch
        try:
            if self._elasticsearch:
                await self._elasticsearch.ping()
                details["elasticsearch"] = "ok"
            else:
                details["elasticsearch"] = "not connected"
                all_ok = False
        except Exception as e:
            details["elasticsearch"] = f"error: {e}"
            all_ok = False

        # FAISS (always ok if initialized)
        details["faiss"] = "ok"

        return Health(ok=all_ok, details=details)

    async def __aenter__(self) -> Self:
        await self.start()
        return self

    async def __aexit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        await self.stop()
