"""
Upstash Redis (serverless, REST) hot-key cache.

Key layout (Redis LIST per key, capped + TTL):
    fact:e:<entity_slug>   -> JSON facts mentioning that entity
    fact:d:<YYYY-MM-DD>    -> JSON facts whose event_date is that day

Pipelines are used so a push or lookup costs ONE HTTP round-trip regardless of key count.
"""
import json
import re
from typing import Dict, Iterable, List, Optional

from upstash_redis.asyncio import Redis

from core.config import settings
from core.latency_tracker import LatencyTracker, get_tracker


def slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


class HotCache:
    def __init__(self):
        self.redis = Redis(
            url=settings.UPSTASH_REDIS_REST_URL.strip().replace("rediss://default:" + settings.UPSTASH_REDIS_REST_TOKEN + "@", "https://").rstrip(":6379") if (settings.UPSTASH_REDIS_REST_URL.strip().startswith("redis://") or settings.UPSTASH_REDIS_REST_URL.strip().startswith("rediss://")) else settings.UPSTASH_REDIS_REST_URL.strip(),
            token=settings.UPSTASH_REDIS_REST_TOKEN,
        )

    @staticmethod
    def keys_for_fact(fact: Dict) -> List[str]:
        keys = [f"fact:e:{slug(e)}" for e in fact.get("entities", []) if slug(e)]
        if fact.get("event_date"):
            keys.append(f"fact:d:{fact['event_date']}")
        return list(dict.fromkeys(keys))

    @staticmethod
    def keys_for_query(entities: Iterable[str], dates: Iterable[str]) -> List[str]:
        keys = [f"fact:e:{slug(e)}" for e in entities if slug(e)]
        keys += [f"fact:d:{d}" for d in dates if d]
        return list(dict.fromkeys(keys))

    async def ping(self, tracker: Optional[LatencyTracker] = None) -> None:
        tracker = get_tracker(tracker)
        async with tracker.astep("Redis PING (Upstash warm-up)"):
            await self.redis.ping()

    async def push_facts(self, facts: List[Dict], tracker: Optional[LatencyTracker] = None) -> int:
        """Push each fact under all of its hot keys in a single pipelined round-trip."""
        tracker = get_tracker(tracker)
        pipe = self.redis.pipeline()
        n_keys = 0
        for f in facts:
            payload = json.dumps(f, ensure_ascii=False)
            for k in self.keys_for_fact(f):
                pipe.rpush(k, payload)
                pipe.ltrim(k, -settings.REDIS_MAX_PER_KEY, -1)
                pipe.expire(k, settings.REDIS_TTL_SECONDS)
                n_keys += 1
        if n_keys == 0:
            return 0
        async with tracker.astep(f"Redis Push Hot Keys ({n_keys} key writes)"):
            await pipe.exec()
        return n_keys

    async def lookup(
        self, entities: Iterable[str], dates: Iterable[str],
        tracker: Optional[LatencyTracker] = None, limit: int = 8,
    ) -> List[Dict]:
        tracker = get_tracker(tracker)
        keys = self.keys_for_query(entities, dates)
        if not keys:
            return []

        pipe = self.redis.pipeline()
        for k in keys:
            pipe.lrange(k, 0, -1)

        facts: Dict[str, Dict] = {}
        async with tracker.astep("Redis Cache Check") as s:
            results = await pipe.exec()
            for lst in results:
                for item in lst or []:
                    try:
                        obj = json.loads(item) if isinstance(item, (str, bytes)) else item
                        facts[obj.get("id", obj["content"])] = obj
                    except Exception:
                        continue
            s.name = "Redis Cache Hit" if facts else "Redis Cache Miss"
            s.extra = f" ({len(keys)} keys, {len(facts)} facts)"

        ordered = sorted(facts.values(), key=lambda f: f.get("created_at", ""), reverse=True)
        return ordered[:limit]
