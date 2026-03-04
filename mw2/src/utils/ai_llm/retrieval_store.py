"""Retrieval store abstraction for embedding + vector search + exact match."""

from abc import ABC, abstractmethod
from typing import Any


class RetrievalStore(ABC):
    """Abstract base class for retrieval: embed, retrieve, exact_match, store."""

    @abstractmethod
    def embed(self, text: str) -> list[float]:
        """Calculate embedding for text."""
        ...

    @abstractmethod
    def retrieve(self, embedding: list[float], top_k: int, **kwargs: Any) -> list[Any]:
        """Retrieve top-k results from vector store."""
        ...

    @abstractmethod
    def exact_match(self, results: list[Any]) -> tuple[bool, Any] | None:
        """Return (True, parsed) if top-1 is exact match, else (False, None) or None."""
        ...

    def store(
        self,
        query: str,
        embedding: list[float],
        result: Any,
        **kwargs: Any,
    ) -> None:
        """Store to DB when exact match not found. Override in subclass; default no-op."""
        pass

    def get(
        self,
        query: str,
        top_k: int = 4,
        **kwargs: Any,
    ) -> tuple[Any | None, list[Any], list[float]]:
        """Embed, retrieve, check exact_match. Returns (parsed_if_exact_match, results, embedding)."""
        emb = self.embed(query)
        results = self.retrieve(emb, top_k, **kwargs)
        match = self.exact_match(results)
        if match and match[0]:
            return match[1], results, emb
        return None, results, emb
