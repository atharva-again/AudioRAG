from typing import Literal, Protocol, runtime_checkable

VectorIdDefaultFormat = Literal["sha256", "uuid5"]


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

    async def delete_by_source_id(self, source_id: str) -> None: ...


@runtime_checkable
class VerifiableVectorStoreProvider(Protocol):
    async def verify(self, ids: list[str]) -> bool: ...


@runtime_checkable
class VectorIdFormatAwareProvider(Protocol):
    vector_id_default_format: VectorIdDefaultFormat
