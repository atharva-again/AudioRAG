from typing import Protocol, runtime_checkable


@runtime_checkable
class RerankerProvider(Protocol):
    async def rerank(
        self, query: str, documents: list[str], top_n: int = 3
    ) -> list[tuple[int, float]]: ...
