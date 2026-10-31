"""
Groq wrappers: Whisper STT, JSON-mode chat (router / Agent Alpha), streaming chat (synthesis).
Every external call is wrapped in latency tracking.
"""
import json
import re
import time
from typing import AsyncIterator, Dict, List, Optional

from groq import AsyncGroq

from core.config import settings
from core.latency_tracker import LatencyTracker, get_tracker
from core.silero_vad import pcm_to_wav


def _g(obj, key, default=None):
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _parse_json(raw: str) -> Dict:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", raw, re.S)
        if m:
            return json.loads(m.group(0))
        raise


class GroqService:
    def __init__(self):
        self.client = AsyncGroq(api_key=settings.GROQ_API_KEY, timeout=20.0, max_retries=1)

    async def warmup(self, tracker: Optional[LatencyTracker] = None) -> None:
        """Primes DNS + TLS + HTTP/2 so the first real call is not penalised."""
        tracker = get_tracker(tracker)
        async with tracker.astep("Groq warm-up (DNS+TLS+auth)"):
            await self.client.models.list()

    # ------------------------------------------------------------------- STT
    async def transcribe(self, pcm: bytes, tracker: Optional[LatencyTracker], label: str) -> Dict:
        tracker = get_tracker(tracker)
        wav = pcm_to_wav(pcm)
        kwargs = dict(
            file=("chunk.wav", wav),
            model=settings.GROQ_STT_MODEL,
            response_format="verbose_json",
            temperature=0.0,
        )
        if settings.STT_LANGUAGE:
            kwargs["language"] = settings.STT_LANGUAGE

        t0 = time.time()
        async with tracker.astep(f"STT Groq ({label})"):
            resp = await self.client.audio.transcriptions.create(**kwargs)
        stt_ms = (time.time() - t0) * 1000

        text = (getattr(resp, "text", "") or "").strip()
        segs = getattr(resp, "segments", None) or []
        nsp = [_g(s, "no_speech_prob") for s in segs if _g(s, "no_speech_prob") is not None]
        lps = [_g(s, "avg_logprob") for s in segs if _g(s, "avg_logprob") is not None]
        return {
            "text": text,
            "stt_ms": stt_ms,
            "no_speech_prob": max(nsp) if nsp else None,
            "avg_logprob": (sum(lps) / len(lps)) if lps else None,
        }

    # -------------------------------------------------------------- JSON chat
    async def chat_json(
        self, model: str, messages: List[Dict], tracker: Optional[LatencyTracker],
        label: str, max_tokens: int = 400,
    ) -> Dict:
        tracker = get_tracker(tracker)
        async with tracker.astep(label):
            resp = await self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.0,
                max_tokens=max_tokens,
                response_format={"type": "json_object"},
            )
        return _parse_json(resp.choices[0].message.content or "{}")

    # ---------------------------------------------------------- streaming chat
    async def chat_stream(
        self, model: str, messages: List[Dict], tracker: Optional[LatencyTracker],
        label: str, temperature: float = 0.3, max_tokens: int = 300,
    ) -> AsyncIterator[str]:
        tracker = get_tracker(tracker)
        t0 = time.time()
        first = True
        try:
            stream = await self.client.chat.completions.create(
                model=model, messages=messages, temperature=temperature,
                max_tokens=max_tokens, stream=True,
            )
            async for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta.content
                if delta:
                    if first:
                        tracker.record(f"{label} TTFT", (time.time() - t0) * 1000)
                        first = False
                    yield delta
        finally:
            tracker.record(f"{label} full stream", (time.time() - t0) * 1000)


_service: Optional[GroqService] = None


def get_groq() -> GroqService:
    global _service
    if _service is None:
        _service = GroqService()
    return _service
