"""
Layers 3-5 as a LangGraph StateGraph:

    START -> route --+--> ingest   (Agent Alpha)  --+
                     |                              +--> synth (Groq 70B, streaming) -> END
                     +--> retrieve (Redis -> Supabase)+
                     +--> synth (chit-chat only)

Per-request objects (latency tracker, sentence emitter, websocket event sender) are passed
through config["configurable"], so the compiled graph is shared across all connections.
"""
import asyncio
import re
import uuid
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Any, Dict, List, Tuple, TypedDict

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph

from agents.intent_router import IntentRouter
from core.config import settings, today_str
from infrastructure.groq_client import GroqService
from infrastructure.redis_client import HotCache
from infrastructure.supabase_client import Embedder, FactStore


class PipelineState(TypedDict, total=False):
    boxes: List[Dict]
    classified: List[Dict]
    facts: List[Dict]
    queries: List[Dict]
    chitchat: List[Dict]
    ingested_facts: List[Dict]
    ingest_status: Dict[str, Any]
    context: List[Dict]
    context_source: str
    answer: str


@dataclass
class Services:
    groq: GroqService
    cache: HotCache
    store: FactStore
    embedder: Embedder
    router: IntentRouter


# --------------------------------------------------------------------------- prompts
ALPHA_SYSTEM = """You are Agent Alpha, the memory-ingestion agent of a voice assistant.
Convert the numbered transcript boxes into atomic, self-contained facts.
For each fact output:
- "content": one standalone sentence (resolve pronouns, keep the speaker's language)
- "entities": canonical names of people/companies/projects/places (max 6)
- "event_date": YYYY-MM-DD or null (resolve relative dates using TODAY)
- "category": one of meeting, task, person, finance, location, preference, other
- "importance": integer 1-5
- "boxes": list of source box numbers
Respond with ONLY valid JSON: {"facts":[{"content":"...","entities":[],"event_date":null,"category":"other","importance":3,"boxes":[1]}]}
"""

SYNTH_SYSTEM = """You are a real-time voice assistant. Your reply is spoken aloud.
Rules:
- For questions about stored information, answer ONLY from CONTEXT. If CONTEXT lacks the answer, say you do not have that information. Never invent facts.
- Be brief: one to three short sentences. Plain spoken language. No markdown, lists, emojis or abbreviations.
- Say dates and numbers the way a person would say them aloud.
- Reply in the same language the user spoke."""

# --------------------------------------------------------------------------- helpers
_BOUNDARY = re.compile(r"(?<=[.!?\u0964])\s+")
_EAGER = re.compile(r"(?<=[,;:])\s+")


def pop_sentences(buf: str, eager: bool) -> Tuple[List[str], str]:
    """Split a token buffer into speakable sentences. 'eager' lets the FIRST chunk break at a
    comma (>= 25 chars) so TTS can start earlier."""
    out: List[str] = []
    while True:
        cut = _BOUNDARY.search(buf)
        if cut is None and eager and not out:
            for m in _EAGER.finditer(buf):
                if m.start() >= 25:
                    cut = m
                    break
        if cut is None:
            break
        head = buf[:cut.start()].strip()
        buf = buf[cut.end():]
        if head:
            out.append(head)
            eager = False
    return out, buf


def _valid_date(s: Any):
    try:
        return date.fromisoformat(str(s)).isoformat()
    except Exception:
        return None


def _uniq(items) -> List[str]:
    return list(dict.fromkeys(i for i in items if i))


def build_pipeline(svc: Services):
    # ------------------------------------------------------------------ node: route
    async def route_node(state: PipelineState, config: RunnableConfig) -> Dict:
        cfg = config["configurable"]
        tracker = cfg["tracker"]
        classified = await svc.router.classify(state["boxes"], tracker)
        facts = [c for c in classified if c["intent"] == "fact"]
        queries = [c for c in classified if c["intent"] == "query"]
        chitchat = [c for c in classified if c["intent"] == "chitchat"]
        tracker.milestone("Intent routing done", once=True)
        send = cfg.get("send_event")
        if send:
            await send({"type": "intent", "items": classified})
        return {"classified": classified, "facts": facts, "queries": queries, "chitchat": chitchat}

    def route_next(state: PipelineState) -> List[str]:
        nxt = []
        if state.get("facts"):
            nxt.append("ingest")
        if state.get("queries"):
            nxt.append("retrieve")
        return nxt or ["synth"]

    # ---------------------------------------------------- node: ingest (Agent Alpha)
    async def ingest_node(state: PipelineState, config: RunnableConfig) -> Dict:
        tracker = config["configurable"]["tracker"]
        facts_in = state["facts"]
        listing = "\n".join(f"#{f['box']}: {f['text']}" for f in facts_in)
        items: List[Dict] = []
        try:
            parsed = await svc.groq.chat_json(
                settings.GROQ_ROUTER_MODEL,
                [{"role": "system", "content": ALPHA_SYSTEM},
                 {"role": "user", "content": f"TODAY: {today_str()}\n\nBOXES:\n{listing}"}],
                tracker, "Agent Alpha entity/date parse (JSON)", max_tokens=700,
            )
            items = parsed.get("facts", []) or []
        except Exception as e:
            print(f"[ERROR] Agent Alpha parse failed ({e!r}); using router metadata as fallback")

        if not items:  # fallback: store raw boxes with router-extracted metadata
            items = [{"content": f["text"], "entities": f["entities"],
                      "event_date": (f["dates"] or [None])[0], "category": "other",
                      "importance": 3, "boxes": [f["box"]]} for f in facts_in]

        now_iso = datetime.now(timezone.utc).isoformat()
        facts: List[Dict] = []
        for it in items:
            content = str(it.get("content", "")).strip()
            if not content:
                continue
            try:
                importance = max(1, min(5, int(it.get("importance", 3))))
            except Exception:
                importance = 3
            facts.append({
                "id": str(uuid.uuid4()),
                "content": content,
                "entities": [str(e).strip() for e in (it.get("entities") or []) if str(e).strip()][:6],
                "event_date": _valid_date(it.get("event_date")),
                "category": str(it.get("category", "other")),
                "importance": importance,
                "source_boxes": it.get("boxes", []),
                "created_at": now_iso,
            })

        # ---- push to both sinks concurrently (hot key -> Redis, row -> Supabase)
        async def redis_sink():
            await svc.cache.push_facts(facts, tracker)

        async def supabase_sink():
            embs = await svc.embedder.embed([f["content"] for f in facts], tracker, "Embedding (ingest)")
            rows = [{
                "id": f["id"], "content": f["content"], "entities": f["entities"],
                "event_date": f["event_date"], "created_at": f["created_at"],
                "metadata": {"category": f["category"], "importance": f["importance"],
                             "source_boxes": f["source_boxes"]},
                "embedding": e,
            } for f, e in zip(facts, embs)]
            await svc.store.insert_facts(rows, tracker)

        status = {"redis": False, "supabase": False, "count": len(facts)}
        if facts:
            r_redis, r_supa = await asyncio.gather(redis_sink(), supabase_sink(), return_exceptions=True)
            status["redis"] = not isinstance(r_redis, Exception)
            status["supabase"] = not isinstance(r_supa, Exception)
            if isinstance(r_redis, Exception):
                print(f"[ERROR] Redis sink failed: {r_redis!r}")
            if isinstance(r_supa, Exception):
                print(f"[ERROR] Supabase sink failed: {r_supa!r}")
        tracker.milestone("Ingestion (Agent Alpha) done", once=True)
        return {"ingested_facts": facts, "ingest_status": status}

    # ------------------------------------------------------------- node: retrieve
    async def retrieve_node(state: PipelineState, config: RunnableConfig) -> Dict:
        tracker = config["configurable"]["tracker"]
        queries = state["queries"]
        entities = _uniq(e for q in queries for e in q.get("entities", []))
        dates = _uniq(d for q in queries for d in q.get("dates", []))
        query_text = " ".join(q["text"] for q in queries)

        facts: List[Dict] = []
        source = "none"
        if entities or dates:
            try:  # Step 1: Redis hot cache
                facts = await svc.cache.lookup(entities, dates, tracker)
                if facts:
                    source = "redis"
            except Exception as e:
                print(f"[ERROR] Redis lookup failed ({e!r}); falling back to Supabase")

        if not facts:  # Step 2: Supabase pgvector fallback
            try:
                emb = await svc.embedder.embed([query_text], tracker, "Embedding (query)")
                facts = await svc.store.search(emb[0], tracker)
                source = "supabase" if facts else "none"
            except Exception as e:
                print(f"[ERROR] Supabase retrieval failed: {e!r}")

        tracker.milestone(f"Context retrieved (source={source})", once=True)
        return {"context": facts, "context_source": source}

    # --------------------------------------------------------------- node: synth
    async def synth_node(state: PipelineState, config: RunnableConfig) -> Dict:
        cfg = config["configurable"]
        tracker, emit = cfg["tracker"], cfg["emit"]
        queries = state.get("queries", [])
        chitchat = state.get("chitchat", [])
        ingested = state.get("ingested_facts", [])
        status = state.get("ingest_status", {})

        # Pure ingestion turn: template acknowledgement, zero LLM latency.
        if not queries and not chitchat:
            if ingested and (status.get("redis") or status.get("supabase")):
                n = len(ingested)
                await emit(f"Got it. I have saved {n} {'fact' if n == 1 else 'facts'}.")
            else:
                await emit("Sorry, I could not save that. Please try again.")
            return {"answer": "(ack)"}

        ctx_facts = state.get("context", []) + ingested
        lines = []
        for f in ctx_facts:
            d = f" [date: {f['event_date']}]" if f.get("event_date") else ""
            lines.append(f"- {f['content']}{d}")
        context = "\n".join(lines) if lines else "(no stored information found)"
        note = "\n(The user also just shared new information, which has been saved.)" if ingested else ""

        user_boxes = sorted(queries + chitchat, key=lambda x: x["box"])
        user_text = "\n".join(f"#{b['box']}: {b['text']}" for b in user_boxes)
        messages = [
            {"role": "system", "content": f"{SYNTH_SYSTEM}\n\nTODAY: {today_str()}\n\nCONTEXT:\n{context}{note}"},
            {"role": "user", "content": user_text},
        ]

        buf, sent, spoken = "", 0, []
        async for tok in svc.groq.chat_stream(
            settings.GROQ_SYNTH_MODEL, messages, tracker, "Groq LLM Synthesis"
        ):
            buf += tok
            ready, buf = pop_sentences(buf, eager=(sent == 0))
            for s in ready:
                if sent == 0:
                    tracker.milestone("LLM first speakable sentence ready", once=True)
                sent += 1
                spoken.append(s)
                await emit(s)
        tail = buf.strip()
        if tail:
            if sent == 0:
                tracker.milestone("LLM first speakable sentence ready", once=True)
            spoken.append(tail)
            await emit(tail)
        return {"answer": " ".join(spoken)}

    # ------------------------------------------------------------------- compile
    g = StateGraph(PipelineState)
    g.add_node("route", route_node)
    g.add_node("ingest", ingest_node)
    g.add_node("retrieve", retrieve_node)
    g.add_node("synth", synth_node)
    g.add_edge(START, "route")
    g.add_conditional_edges("route", route_next, ["ingest", "retrieve", "synth"])
    g.add_edge("ingest", "synth")
    g.add_edge("retrieve", "synth")
    g.add_edge("synth", END)
    return g.compile()
