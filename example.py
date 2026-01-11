"""
Example demonstrating all data_layer features.

This example shows how to use the unified data layer for:
- Redis caching
- Postgres database operations
- Elasticsearch full-text search
- FAISS vector similarity search
- Event-driven state synchronization
- Health monitoring

Prerequisites:
    cd docker && docker compose up -d
"""

import asyncio
from pathlib import Path

from data_layer import (
    DataLayer,
    DataLayerConfig,
    RedisConfig,
    PostgresConfig,
    ElasticsearchConfig,
    FaissConfig,
    DataEvent,
)


async def main():
    print("=" * 60)
    print("Data Layer Example")
    print("=" * 60)

    # Configure the data layer
    config = DataLayerConfig(
        redis=RedisConfig(),
        postgres=PostgresConfig(password="postgres"),
        elasticsearch=ElasticsearchConfig(),
        faiss=FaissConfig(index_path=Path("./data/example_vectors")),
    )

    # Use async context manager for automatic lifecycle
    async with DataLayer(config) as data:
        # =================================================================
        # 1. Health Check
        # =================================================================
        print("\n[1] Health Check")
        health = await data.health()
        print(f"  All services healthy: {health.ok}")
        for service, status in health.details.items():
            print(f"  - {service}: {status}")

        # =================================================================
        # 2. Redis Operations
        # =================================================================
        print("\n[2] Redis Operations (Caching)")

        # Set and get
        await data.redis.set("user:123:name", b"Alice")
        name = await data.redis.get("user:123:name")
        print(f"  Retrieved name: {name.decode()}")

        # Hash operations
        await data.redis.hset("user:123", mapping={
            "name": "Alice",
            "email": "alice@example.com",
            "age": "30",
        })
        user = await data.redis.hgetall("user:123")
        print(f"  User hash: {user}")

        # Expiration
        await data.redis.setex("session:abc", 3600, b"session-data")
        ttl = await data.redis.ttl("session:abc")
        print(f"  Session TTL: {ttl} seconds")

        # =================================================================
        # 3. Postgres Operations
        # =================================================================
        print("\n[3] Postgres Operations (Relational Data)")

        # Create a temporary table
        await data.postgres.execute("""
            CREATE TEMPORARY TABLE users (
                id SERIAL PRIMARY KEY,
                name TEXT NOT NULL,
                email TEXT UNIQUE NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        print("  Created users table")

        # Insert data
        user_id = await data.postgres.fetchval(
            "INSERT INTO users (name, email) VALUES ($1, $2) RETURNING id",
            "Alice Johnson",
            "alice@example.com",
        )
        print(f"  Inserted user with ID: {user_id}")

        # Query data
        user = await data.postgres.fetchrow(
            "SELECT * FROM users WHERE id = $1",
            user_id,
        )
        print(f"  Retrieved user: {dict(user)}")

        # Batch insert
        await data.postgres.executemany(
            "INSERT INTO users (name, email) VALUES ($1, $2)",
            [
                ("Bob Smith", "bob@example.com"),
                ("Charlie Brown", "charlie@example.com"),
            ],
        )

        # Query all users
        users = await data.postgres.fetch("SELECT name, email FROM users ORDER BY id")
        print(f"  All users: {[dict(u) for u in users]}")

        # =================================================================
        # 4. Elasticsearch Operations
        # =================================================================
        print("\n[4] Elasticsearch Operations (Full-Text Search)")

        index_name = "articles"

        # Index documents
        articles = [
            {
                "id": "1",
                "title": "Introduction to Python",
                "content": "Python is a powerful programming language",
                "tags": ["python", "programming", "tutorial"],
            },
            {
                "id": "2",
                "title": "Advanced Python Techniques",
                "content": "Learn about decorators, generators, and metaclasses",
                "tags": ["python", "advanced", "programming"],
            },
            {
                "id": "3",
                "title": "Web Development with FastAPI",
                "content": "FastAPI is a modern web framework for building APIs",
                "tags": ["python", "web", "fastapi"],
            },
        ]

        for article in articles:
            await data.elasticsearch.index(
                index=index_name,
                id=article["id"],
                document=article,
            )
        print(f"  Indexed {len(articles)} articles")

        # Refresh to make documents searchable
        await data.elasticsearch.indices.refresh(index=index_name)

        # Full-text search
        search_results = await data.elasticsearch.search(
            index=index_name,
            query={
                "match": {
                    "content": "programming"
                }
            },
        )
        hits = search_results["hits"]["hits"]
        print(f"  Search 'programming' found {len(hits)} results:")
        for hit in hits:
            print(f"    - {hit['_source']['title']} (score: {hit['_score']:.2f})")

        # Complex query with filters
        filtered_results = await data.elasticsearch.search(
            index=index_name,
            query={
                "bool": {
                    "must": [
                        {"match": {"content": "python"}}
                    ],
                    "filter": [
                        {"term": {"tags": "advanced"}}
                    ]
                }
            },
        )
        print(f"  Filtered search found {len(filtered_results['hits']['hits'])} results")

        # Cleanup
        await data.elasticsearch.indices.delete(index=index_name)

        # =================================================================
        # 5. FAISS Vector Operations
        # =================================================================
        print("\n[5] FAISS Vector Operations (Similarity Search)")

        # Create sample embeddings (normally from an embedding model)
        # Using 1536 dimensions (OpenAI embedding size)
        def create_embedding(base_value: float) -> list[float]:
            """Create a simple embedding for demonstration."""
            embedding = [base_value] + [0.0] * 1535
            return embedding

        # Upsert vectors with metadata
        docs = [
            {
                "id": "doc-1",
                "embedding": create_embedding(1.0),
                "metadata": {"title": "Python Tutorial", "category": "programming"},
            },
            {
                "id": "doc-2",
                "embedding": create_embedding(0.9),
                "metadata": {"title": "Python Best Practices", "category": "programming"},
            },
            {
                "id": "doc-3",
                "embedding": create_embedding(0.2),
                "metadata": {"title": "Cooking Recipes", "category": "food"},
            },
        ]

        for doc in docs:
            await data.vectors.upsert(
                namespace="documents",
                vector_id=doc["id"],
                vector=doc["embedding"],
                metadata=doc["metadata"],
            )
        print(f"  Stored {len(docs)} document embeddings")

        # Query for similar vectors
        query_embedding = create_embedding(0.95)  # Similar to doc-1 and doc-2
        results = await data.vectors.query(
            namespace="documents",
            vector=query_embedding,
            top_k=3,
        )

        print(f"  Top {len(results)} similar documents:")
        for i, match in enumerate(results, 1):
            print(f"    {i}. {match.metadata['title']} (score: {match.score:.4f})")

        # Delete a vector
        deleted = await data.vectors.delete("documents", "doc-3")
        print(f"  Deleted doc-3: {deleted}")

        # =================================================================
        # 6. Event System
        # =================================================================
        print("\n[6] Event System (Pub/Sub)")

        # Track received events
        received_events: list[DataEvent] = []

        # Define event handlers
        async def log_all_events(event: DataEvent) -> None:
            """Log all events to console."""
            print(f"  [LOG] {event.type}: {event.resource}/{event.key}")

        async def handle_user_created(event: DataEvent) -> None:
            """Handle user.created events."""
            received_events.append(event)
            user_name = event.payload.get("name", "Unknown")
            print(f"  [HANDLER] New user created: {user_name}")

        async def handle_document_indexed(event: DataEvent) -> None:
            """Handle document.indexed events."""
            received_events.append(event)
            doc_id = event.key
            print(f"  [HANDLER] Document indexed: {doc_id}")

        # Subscribe to events
        data.events.subscribe_all(log_all_events)
        data.events.subscribe("user.created", handle_user_created)
        data.events.subscribe("document.indexed", handle_document_indexed)

        # Publish events
        print("  Publishing events...")

        await data.events.publish(DataEvent(
            type="user.created",
            resource="users",
            key="user-456",
            payload={"name": "Bob", "email": "bob@example.com"},
            source="postgres",
        ))

        await data.events.publish(DataEvent(
            type="document.indexed",
            resource="documents",
            key="doc-789",
            payload={"title": "Sample Document"},
            source="elasticsearch",
        ))

        await data.events.publish(DataEvent(
            type="cache.updated",
            resource="cache",
            key="config",
            payload={"setting": "value"},
            source="redis",
        ))

        print(f"  Received {len(received_events)} specific events")

        # =================================================================
        # 7. Cross-Service Workflow
        # =================================================================
        print("\n[7] Cross-Service Workflow Example")

        # Simulate a complete workflow using all services
        print("  Workflow: Store user, index profile, cache preferences")

        # Step 1: Store in Postgres
        new_user_id = await data.postgres.fetchval(
            "INSERT INTO users (name, email) VALUES ($1, $2) RETURNING id",
            "Diana Prince",
            "diana@example.com",
        )
        print(f"    1. Stored user in Postgres (ID: {new_user_id})")

        # Step 2: Index in Elasticsearch
        await data.elasticsearch.index(
            index="user_profiles",
            id=str(new_user_id),
            document={
                "name": "Diana Prince",
                "bio": "Software engineer interested in distributed systems",
                "skills": ["python", "databases", "elasticsearch"],
            },
            refresh=True,
        )
        print(f"    2. Indexed user profile in Elasticsearch")

        # Step 3: Cache preferences in Redis
        await data.redis.hset(
            f"user:{new_user_id}:prefs",
            mapping={
                "theme": "dark",
                "notifications": "email",
                "language": "en",
            },
        )
        print(f"    3. Cached user preferences in Redis")

        # Step 4: Store user embedding in FAISS
        user_embedding = create_embedding(0.85)
        await data.vectors.upsert(
            namespace="users",
            vector_id=f"user-{new_user_id}",
            vector=user_embedding,
            metadata={"user_id": new_user_id, "name": "Diana Prince"},
        )
        print(f"    4. Stored user embedding in FAISS")

        # Step 5: Publish event
        await data.events.publish(DataEvent(
            type="user.onboarded",
            resource="users",
            key=f"user-{new_user_id}",
            payload={"user_id": new_user_id, "name": "Diana Prince"},
            source="system",
        ))
        print(f"    5. Published user.onboarded event")

        # Cleanup Elasticsearch index
        await data.elasticsearch.indices.delete(index="user_profiles")

    print("\n" + "=" * 60)
    print("Example Complete!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
