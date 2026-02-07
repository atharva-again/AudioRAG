from typing import Protocol, runtime_checkable


@runtime_checkable
class GenerationProvider(Protocol):
    async def generate(self, query: str, context: list[str]) -> str: ...
