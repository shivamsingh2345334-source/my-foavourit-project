"""
Supabase (Postgres + pgvector) fact store, plus the local fastembed embedder
(Groq has no embeddings endpoint, so vectors are computed locally with ONNX).
"""
import asyncio
from typing import Dict, List, Optional

try:
    from supabase import acreate_client as _create_async_client
except ImportError:  # older/newer supabase-py naming
    from supabase import create_async_client as _create_async_client

from core.config import settings
from core.latency_tracker import LatencyTracker, get_tracker


class Embedder:
    def __init__(self):
        self._model = None

    def _load_sync(self) -> None:
        from fastembed import TextEmbedding

        self._model = TextEmbedding(model_name=settings.EMBED_MODEL)
        list(self._model.embed(["warm up"]))  # first call is slow; pay it at startup

    async def load(self, tracker: Optional[LatencyTracker] = None) -> None:
        tracker = get_tracker(tracker)
        loop = asyncio.get_running_loop()
        async with tracker.astep("Embedding model load (fastembed ONNX)"):
            await loop.run_in_executor(None, self._load_sync)

    def _embed_sync(self, texts: List[str]) -> List[List[float]]:
        if self._model is None:
            self._load_sync()
        return [v.tolist() for v in self._model.embed(texts)]

    async def embed(
        self, texts: List[str], tracker: Optional[LatencyTracker] = None, label: str = "Embedding"
    ) -> List[List[float]]:
        tracker = get_tracker(tracker)
        loop = asyncio.get_running_loop()
        async with tracker.astep(f"{label} (local ONNX, {len(texts)} text)"):
            return await loop.run_in_executor(None, self._embed_sync, texts)


class FactStore:
    def __init__(self):
        self.client = None

    async def connect(self) -> None:
        self.client = await _create_async_client(settings.SUPABASE_URL, settings.SUPABASE_SERVICE_KEY)

    async def ping(self, tracker: Optional[LatencyTracker] = None) -> None:
        tracker = get_tracker(tracker)
        async with tracker.astep("Supabase warm-up (select 1 row)"):
            await self.client.table(settings.SUPABASE_FACTS_TABLE).select("id").limit(1).execute()

    async def insert_facts(self, rows: List[Dict], tracker: Optional[LatencyTracker] = None) -> None:
        if self.client is None:
            await self.connect()
        tracker = get_tracker(tracker)
        if not rows:
            return
        async with tracker.astep(f"Supabase Insert ({len(rows)} rows)"):
            await self.client.table(settings.SUPABASE_FACTS_TABLE).insert(rows).execute()

    async def search(
        self, embedding: List[float], tracker: Optional[LatencyTracker] = None,
        k: Optional[int] = None, threshold: Optional[float] = None,
    ) -> List[Dict]:
        tracker = get_tracker(tracker)
        async with tracker.astep("Supabase pgvector RPC (match_facts)") as s:
            res = await self.client.rpc(
                settings.SUPABASE_MATCH_RPC,
                {
                    "query_embedding": embedding,
                    "match_count": k or settings.MATCH_COUNT,
                    "match_threshold": threshold if threshold is not None else settings.MATCH_THRESHOLD,
                },
            ).execute()
            rows = res.data or []
            s.extra = f" ({len(rows)} rows)"
        return rows
