"""
FastAPI application.

WebSocket protocol  (ws://host:8000/ws/voice)
  client -> server
    {"type":"start","tts":true}         begin a turn
    <binary>                            raw PCM int16, mono, 16 kHz, any packet size
    {"type":"end"}                      user stopped speaking
  server -> client
    {"type":"transcript","boxes":[...],"dropped":[...]}
    {"type":"intent","items":[...]}
    {"type":"sentence","index":n,"text":"..."}
    {"type":"audio_start","index":n} <binary mp3 chunks> {"type":"audio_end","index":n}
    {"type":"latency_report","steps":[...]}
    {"type":"done"}  |  {"type":"error","message":"..."}

REST
  GET  /health
  POST /query/text   {"text": "..."}   (skips audio layers; handy for testing)
"""
import asyncio
import json
import time
import traceback
import uuid
from contextlib import asynccontextmanager
from typing import List, Optional

import edge_tts
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from agents.intent_router import IntentRouter
from agents.langgraph_pipeline import Services, build_pipeline
from core.config import settings
from core.latency_tracker import LatencyTracker
from core.silero_vad import EdgeVAD, SpeechChunk, build_text_boxes, load_base_model
from infrastructure.groq_client import get_groq
from infrastructure.redis_client import HotCache
from infrastructure.supabase_client import Embedder, FactStore


# =============================================================================
# Startup: load models, open connections, warm everything
# =============================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    settings.validate()
    tracker = LatencyTracker("startup")
    loop = asyncio.get_running_loop()

    groq, cache, store, embedder = get_groq(), HotCache(), FactStore(), Embedder()
    async with tracker.astep("Supabase client connect"):
        await store.connect()

    # essential model loads (fail loudly)
    await asyncio.gather(
        loop.run_in_executor(None, load_base_model),
        embedder.load(tracker),
    )
    # warm-ups (best effort)
    results = await asyncio.gather(
        groq.warmup(tracker), cache.ping(tracker), store.ping(tracker),
        return_exceptions=True,
    )
    for name, r in zip(("Groq", "Redis", "Supabase"), results):
        if isinstance(r, Exception):
            print(f"[WARN] {name} warm-up failed: {r!r}")

    svc = Services(groq=groq, cache=cache, store=store, embedder=embedder, router=IntentRouter(groq))
    app.state.graph = build_pipeline(svc)
    tracker.print_summary()
    print(f"[READY] ws://{settings.SERVER_HOST}:{settings.SERVER_PORT}/ws/voice")
    yield


app = FastAPI(title="Voice-Native AI Pipeline", lifespan=lifespan)


# =============================================================================
# WebSocket helpers
# =============================================================================
class WSSender:
    """Serialises concurrent sends (TTS consumer + graph events) on one socket."""

    def __init__(self, ws: WebSocket):
        self.ws = ws
        self._lock = asyncio.Lock()

    async def json(self, payload: dict) -> None:
        async with self._lock:
            await self.ws.send_text(json.dumps(payload, ensure_ascii=False))

    async def bytes(self, data: bytes) -> None:
        async with self._lock:
            await self.ws.send_bytes(data)


async def stream_tts(sender: WSSender, text: str, idx: int, tracker: LatencyTracker) -> None:
    """Layer 6: edge-tts streaming. Audio goes to the client as soon as the first bytes arrive."""
    t0 = time.time()
    first = True
    await sender.json({"type": "audio_start", "index": idx})
    comm = edge_tts.Communicate(text, settings.TTS_VOICE, rate=settings.TTS_RATE)
    async for chunk in comm.stream():
        if chunk["type"] != "audio":
            continue
        if first:
            tracker.record(f"TTS edge-tts first audio byte (sentence {idx})", (time.time() - t0) * 1000)
            first = False
        await sender.bytes(chunk["data"])
        if idx == 1:
            tracker.milestone("FIRST AUDIO BYTE SENT (end-of-speech -> first audio)", once=True)
    tracker.record(f"TTS edge-tts full sentence {idx}", (time.time() - t0) * 1000)
    await sender.json({"type": "audio_end", "index": idx})


async def tts_consumer(q: asyncio.Queue, sender: WSSender, tracker: LatencyTracker, tts_enabled: bool):
    idx = 0
    while True:
        text = await q.get()
        if text is None:
            break
        idx += 1
        await sender.json({"type": "sentence", "index": idx, "text": text})
        if tts_enabled:
            try:
                await stream_tts(sender, text, idx, tracker)
            except Exception as e:
                print(f"[ERROR] TTS failed for sentence {idx}: {e!r}")
                await sender.json({"type": "error", "where": "tts", "message": str(e)})


# =============================================================================
# One user turn: streaming VAD -> concurrent STT per chunk
# =============================================================================
class TurnSession:
    def __init__(self, loop: asyncio.AbstractEventLoop, tts_enabled: bool):
        self.loop = loop
        self.tts_enabled = tts_enabled
        self.tracker = LatencyTracker(uuid.uuid4().hex[:8])
        self.groq = get_groq()
        self.vad: Optional[EdgeVAD] = None
        self.stt_tasks: List[asyncio.Task] = []

    async def init(self) -> None:
        self.vad = await self.loop.run_in_executor(None, EdgeVAD, self.tracker)

    async def feed(self, pcm: bytes) -> None:
        chunks = await self.loop.run_in_executor(None, self.vad.feed, pcm)
        self._spawn(chunks)

    def _spawn(self, chunks: List[SpeechChunk]) -> None:
        for c in chunks:
            self.stt_tasks.append(asyncio.create_task(self._stt(c)))

    async def _stt(self, chunk: SpeechChunk) -> dict:
        try:
            r = await self.groq.transcribe(chunk.pcm, self.tracker, f"chunk {chunk.index}")
        except Exception as e:
            print(f"[ERROR] STT chunk {chunk.index} failed: {e!r}")
            r = {"text": "", "stt_ms": 0.0, "no_speech_prob": None, "avg_logprob": None}
        r["chunk"] = chunk.index
        r["duration_ms"] = chunk.duration_ms
        return r

    async def finish(self) -> List[dict]:
        tail = await self.loop.run_in_executor(None, self.vad.flush)
        self._spawn(tail)
        return list(await asyncio.gather(*self.stt_tasks)) if self.stt_tasks else []

    def cancel(self) -> None:
        for t in self.stt_tasks:
            t.cancel()


async def run_turn(app: FastAPI, session: TurnSession, sender: WSSender) -> None:
    tracker = session.tracker
    tracker.set_anchor("end-of-speech (client sent 'end')")

    results = await session.finish()
    tracker.milestone("STT layer complete (all chunks transcribed)")

    boxes, dropped = build_text_boxes(results)
    await sender.json({"type": "transcript", "boxes": boxes, "dropped": dropped})
    if not boxes:
        tracker.print_summary()
        await sender.json({"type": "done", "note": "no valid speech detected"})
        return

    q: asyncio.Queue = asyncio.Queue()
    consumer = asyncio.create_task(tts_consumer(q, sender, tracker, session.tts_enabled))

    async def emit(sentence: str) -> None:
        await q.put(sentence)

    try:
        await app.state.graph.ainvoke(
            {"boxes": boxes},
            config={"configurable": {"tracker": tracker, "emit": emit, "send_event": sender.json}},
        )
    except Exception as e:
        traceback.print_exc()
        await sender.json({"type": "error", "where": "graph", "message": str(e)})
    finally:
        await q.put(None)
        await consumer

    tracker.milestone("TURN COMPLETE (end-of-speech -> last audio byte sent)")
    tracker.print_summary()
    await sender.json({"type": "latency_report", "steps": tracker.report()})
    await sender.json({"type": "done"})


# =============================================================================
# Endpoints
# =============================================================================
@app.websocket("/ws/voice")
async def ws_voice(ws: WebSocket):
    await ws.accept()
    sender = WSSender(ws)
    loop = asyncio.get_running_loop()
    session: Optional[TurnSession] = None
    print("[WS] client connected")
    try:
        while True:
            msg = await ws.receive()
            if msg["type"] == "websocket.disconnect":
                break

            if msg.get("bytes") is not None:
                if session is not None:
                    await session.feed(msg["bytes"])
                continue

            if msg.get("text") is not None:
                data = json.loads(msg["text"])
                kind = data.get("type")
                if kind == "start":
                    if session:
                        session.cancel()
                    session = TurnSession(loop, bool(data.get("tts", True)))
                    await session.init()
                    print(f"[WS] turn started request={session.tracker.request_id}")
                elif kind == "end" and session is not None:
                    try:
                        await run_turn(app, session, sender)
                    except Exception as e:
                        traceback.print_exc()
                        await sender.json({"type": "error", "message": str(e)})
                        await sender.json({"type": "done"})
                    session = None
    except WebSocketDisconnect:
        pass
    finally:
        if session:
            session.cancel()
        print("[WS] client disconnected")


class TextQuery(BaseModel):
    text: str


@app.post("/query/text")
async def query_text(body: TextQuery):
    tracker = LatencyTracker(uuid.uuid4().hex[:8])
    tracker.set_anchor("text request")
    boxes = [{"box": 1, "text": body.text, "chunk": 1, "stt_ms": 0.0, "duration_ms": 0.0}]
    sentences: List[str] = []

    async def emit(s: str) -> None:
        sentences.append(s)

    await app.state.graph.ainvoke(
        {"boxes": boxes},
        config={"configurable": {"tracker": tracker, "emit": emit}},
    )
    tracker.milestone("TEXT TURN COMPLETE")
    tracker.print_summary()
    return {"answer": " ".join(sentences), "latency": tracker.report()}


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    # Keep workers=1: the in-process models (VAD, embedder) are loaded once per process.
    uvicorn.run("main:app", host=settings.SERVER_HOST, port=settings.SERVER_PORT, workers=1)
