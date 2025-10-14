from __future__ import annotations

"""
baseline_rag:: LLM
naive_rag: top-k retrieval + LLM
advanced_rag: top-k retrieval + cross-encoder rerank + LLM
"""

from typing import Callable, List, Optional, Sequence

import faiss
import numpy as np


class RAG:
    def __init__(
        self,
        embedder,
        llm: Callable[..., str],
        *,
        reranker: Optional[Callable[[list[tuple[str, str]]], List[float]]] = None,
    ) -> None:
        self.embedder = embedder
        self.llm = llm
        self.reranker = reranker

        self._docs: list[str] = []
        self._index: Optional[faiss.Index] = None

    # Embeddings & index helpers
    def _encode(self, texts: Sequence[str], *, normalise: bool = True) -> np.ndarray:
        """Vectorise texts"""
        vecs: np.ndarray = np.asarray(self.embedder.encode(list(texts)), dtype="float32")
        if normalise:
            vecs /= np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12
        return vecs

    def build_index(self, docs: Sequence[str]) -> None:
        """Create a fresh FAISS inner-product index and add *docs*."""
        self._docs = list(docs)
        vecs = self._encode(self._docs)
        dim = int(vecs.shape[1])
        self._index = faiss.IndexFlatIP(dim)
        self._index.add(vecs)

    # Core retrieval & reranking
    def retrieve(self, query: str, k: int = 20) -> list[str]:
        if self._index is None:
            raise RuntimeError("Index not initialised - call build_index() first.")
        qv = self._encode([query])
        _dists, indices = self._index.search(qv, k)
        return [self._docs[i] for i in indices[0] if i != -1]

    def rerank_cross(self, query: str, chunks: list[str], top_n: int = 5) -> list[str]:
        """Rerank chunks with a cross-encoder"""
        if not self.reranker or not chunks:
            return chunks[:top_n]
        scores = self.reranker.predict([(query, c) for c in chunks])
        ranked_pairs = sorted(zip(scores, chunks), key=lambda p: p[0], reverse=True)
        return [c for _s, c in ranked_pairs[:top_n]]

    # Prompt & generation helpers
    @staticmethod
    def _build_prompt(
        query: str,
        context_chunks: Sequence[str],
        system_msg: str | None = None,
    ) -> str:
        if system_msg is None:
            system_msg = (
                "You are a helpful assistant. Use the supplied context as the "
                "primary source. If necessary, draw on your broader knowledge "
                "to provide a complete answer."
            )
        context_block = "\n\n".join(context_chunks) if context_chunks else "(no relevant context retrieved)"
        return (
            f"{system_msg}\n\n"
            f"### Question:\n{query}\n\n"
            f"### Context:\n{context_block}\n\n"
            f"### Answer:"
        )

    def generate(self, prompt: str, max_tokens: int = 256, **gen_kwargs) -> str:
        """Thin wrapper over the underlying llm callable."""
        if self.llm is None:
            raise RuntimeError("No LLM/generator supplied to RAG.")

        if "max_new_tokens" not in gen_kwargs and "max_tokens" not in gen_kwargs:
            gen_kwargs["max_new_tokens"] = max_tokens

        out = self.llm(prompt, **gen_kwargs)

        if isinstance(out, list):
            text = out[0].get("generated_text", str(out[0])).strip()
        else:
            text = str(out).strip()
        return text

    # RAG 
    def baseline_rag(
        self,
        query: str,
        *,
        system_msg: str | None = None,
        **gen_kwargs,
    ) -> str:
        """Pure LLM generation - no retrieval, serves as memorisation baseline."""
        prompt = self._build_prompt(query, [], system_msg=system_msg)
        return self.generate(prompt, **gen_kwargs)

    def naive_rag(
        self,
        query: str,
        *,
        k_ctx: int = 3,
    ) -> str:
        """Retrieve top-k context chunks and generate an answer."""
        chunks = self.retrieve(query, k_ctx)
        prompt = self._build_prompt(query, chunks)
        from pprint import pprint
        pprint(prompt)
        return self.generate(prompt)

    def advanced_rag(
        self,
        query: str,
        *,
        k_retrieval: int = 20,
        k_ctx: int = 5,
    ) -> str:
        """Retrieve many, rerank, then answer."""
        retrieved = self.retrieve(query, k_retrieval)
        ranked = self.rerank_cross(query, retrieved, top_n=k_ctx)
        prompt = self._build_prompt(query, ranked)
        return self.generate(prompt)
