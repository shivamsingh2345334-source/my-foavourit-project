"""
Layer 3 - Intent router (Llama 8B via Groq, JSON mode).
Classifies EVERY text box as fact | query | chitchat and extracts entities + absolute dates
in the same call, so retrieval needs no extra LLM hop.
"""
from typing import Dict, List, Optional

from core.config import settings, today_str
from core.latency_tracker import LatencyTracker
from infrastructure.groq_client import GroqService

ROUTER_SYSTEM = """You are the intent router of a real-time voice assistant with long-term memory.
You receive numbered transcript boxes (#1, #2, ...). Each box is one spoken sentence-chunk.

Classify EVERY box as exactly one of:
- "fact": the speaker is stating or uploading information that should be remembered
          (events, meetings, deadlines, numbers, names, decisions, tasks, preferences).
- "query": the speaker asks a question or asks to recall, look up or summarise stored information.
- "chitchat": greeting, thanks, small talk, or a request that needs no stored memory.

For each box also extract:
- "entities": people, companies, projects, places, products mentioned (canonical short names, max 6).
- "dates": absolute dates as YYYY-MM-DD, resolved from relative expressions
           (today, tomorrow, last Friday, next Monday...) using TODAY. Empty list if none.

Respond with ONLY valid JSON, no prose:
{"items":[{"box":1,"intent":"fact","entities":["..."],"dates":["YYYY-MM-DD"]}]}
"""

_VALID = {"fact", "query", "chitchat"}


def _box_no(v) -> Optional[int]:
    try:
        return int(str(v).strip().lstrip("#"))
    except Exception:
        return None


class IntentRouter:
    def __init__(self, groq: GroqService):
        self.groq = groq

    async def classify(self, boxes: List[Dict], tracker: Optional[LatencyTracker]) -> List[Dict]:
        listing = "\n".join(f"#{b['box']}: {b['text']}" for b in boxes)
        messages = [
            {"role": "system", "content": ROUTER_SYSTEM},
            {"role": "user", "content": f"TODAY: {today_str()}\n\nBOXES:\n{listing}"},
        ]
        items: List[Dict] = []
        try:
            data = await self.groq.chat_json(
                settings.GROQ_ROUTER_MODEL, messages, tracker,
                "Intent Router Groq (JSON)", max_tokens=500,
            )
            items = data.get("items", []) or []
        except Exception as e:
            print(f"[ERROR] Intent router failed ({e!r}); defaulting every box to 'query'")

        by_box = {}
        for it in items:
            n = _box_no(it.get("box")) if isinstance(it, dict) else None
            if n is not None:
                by_box[n] = it

        out: List[Dict] = []
        for b in boxes:
            it = by_box.get(b["box"], {})
            intent = str(it.get("intent", "query")).lower().strip()
            if intent not in _VALID:
                intent = "query"
            ents = [str(e).strip() for e in (it.get("entities") or []) if str(e).strip()]
            dates = [str(d).strip() for d in (it.get("dates") or []) if str(d).strip()]
            out.append({"box": b["box"], "text": b["text"], "intent": intent,
                        "entities": ents, "dates": dates})

        for o in out:
            print(f"[ROUTER] #{o['box']} -> {o['intent']:<8} entities={o['entities']} dates={o['dates']}")
        return out
