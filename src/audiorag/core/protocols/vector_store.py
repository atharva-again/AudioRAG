from typing import Protocol, runtime_checkable


@runtime_checkable
class VectorStoreProvider(Protocol):
    async def add(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict],
        documents: list[str],
    ) -> None: ...

    async def query(self, embedding: list[float], top_k: int = 10) -> list[dict]: ...

    async def delete_by_source(self, source_url: str) -> None: ...


@runtime_checkable
class VerifiableVectorStoreProvider(Protocol):
    async def verify(self, ids: list[str]) -> bool: ...
