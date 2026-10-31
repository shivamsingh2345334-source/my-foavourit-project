# Voice-Native AI Pipeline: Setup & Run

## 0. Prerequisites
- Python 3.10 - 3.12
- A microphone and speakers (for `voice_tester.py`)
- Linux: `sudo apt install libportaudio2 ffmpeg`   |   macOS: `brew install portaudio`

## 1. Create the workspace
```bash
mkdir project_root && cd project_root
mkdir core infrastructure agents
touch core/__init__.py infrastructure/__init__.py agents/__init__.py
# paste every file from the chat into its path
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
```

## 2. requirements.txt
```text
fastapi>=0.110
uvicorn[standard]>=0.29
websockets>=12.0
python-dotenv>=1.0
groq>=0.11
upstash-redis>=1.1
supabase>=2.5
langgraph>=0.2
edge-tts>=6.1.9
silero-vad>=5.1
numpy>=1.26
sounddevice>=0.4.6
miniaudio>=1.59
fastembed>=0.3
```
```bash
# CPU-only PyTorch (much smaller, plenty for Silero):
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

## 3. API keys (`.env` in project_root)
```env
GROQ_API_KEY=gsk_...
UPSTASH_REDIS_REST_URL=https://xxxx.upstash.io
UPSTASH_REDIS_REST_TOKEN=AX...
SUPABASE_URL=https://xxxx.supabase.co
SUPABASE_SERVICE_KEY=eyJ...        # service_role key. Server-side ONLY, never ship to clients

# optional overrides
# GROQ_STT_MODEL=whisper-large-v3-turbo
# GROQ_ROUTER_MODEL=llama-3.1-8b-instant
# GROQ_SYNTH_MODEL=llama-3.3-70b-versatile
# STT_LANGUAGE=en
# TTS_VOICE=en-US-AriaNeural
# VAD_PAUSE_MS=700
```
- **Groq:** console.groq.com -> API Keys. Confirm the model IDs against the Models page.
- **Upstash:** console.upstash.com -> Create Redis DB -> *REST API* section -> copy URL + token. Pick the region nearest your server.
- **Supabase:** Project Settings -> API -> Project URL and `service_role` key. Pick the region nearest your server.

## 4. Supabase schema (SQL Editor -> run once)
```sql
create extension if not exists vector;

create table if not exists public.facts (
  id          uuid primary key,
  content     text        not null,
  entities    text[]      not null default '{}',
  event_date  date,
  metadata    jsonb       not null default '{}'::jsonb,
  embedding   vector(384) not null,          -- must equal EMBED_DIM
  created_at  timestamptz not null default now()
);

create index if not exists facts_embedding_hnsw on public.facts
  using hnsw (embedding vector_cosine_ops);
create index if not exists facts_entities_gin on public.facts using gin (entities);
create index if not exists facts_event_date_idx on public.facts (event_date);

create or replace function public.match_facts(
  query_embedding vector(384),
  match_count     int   default 5,
  match_threshold float default 0.25
)
returns table (
  id uuid, content text, entities text[], event_date date,
  metadata jsonb, created_at timestamptz, similarity float
)
language sql stable as $$
  select f.id, f.content, f.entities, f.event_date, f.metadata, f.created_at,
         1 - (f.embedding <=> query_embedding) as similarity
  from public.facts f
  where 1 - (f.embedding <=> query_embedding) > match_threshold
  order by f.embedding <=> query_embedding
  limit match_count;
$$;
```
If you change `EMBED_MODEL` (e.g. a multilingual one), change `vector(384)` and `EMBED_DIM` to match.

## 5. Run the server (single worker)
```bash
python main.py
# or: uvicorn main:app --host 0.0.0.0 --port 8000
```
Startup prints warm-up latencies (Groq TLS, Redis PING, Supabase, model loads) and `[READY]`.
Smoke test without audio:
```bash
curl -X POST localhost:8000/query/text -H "content-type: application/json" \
     -d '{"text":"What is the capital of France?"}'
```

## 6. Run the voice tester (second terminal)
```bash
python voice_tester.py                 # streaming mode (lowest perceived latency)
python voice_tester.py --batch         # record fully, then send
python voice_tester.py --client-tts    # server returns text; edge-tts runs on the client
python voice_tester.py --list-devices  # find your mic index, then --device N
```
Press ENTER to start, speak, press ENTER to stop.

**Try this sequence**
1. Fact upload: *"Tomorrow at three PM I have a meeting with Acme about the budget. Priya will present the numbers."* -> expect `Got it. I have saved 2 facts.`
2. Query: *"When is my meeting with Acme?"* -> Redis hit (hot key `fact:e:acme`) -> spoken answer.
3. Query about something stored long ago / with different wording -> Redis miss -> pgvector fallback.

## 7. Reading the latency output
Every external call prints a line such as:
```
[LATENCY] STT Groq (chunk 2): 180ms
[LATENCY] Intent Router Groq (8B JSON): 150ms
[LATENCY] Redis Cache Hit: 18ms (2 keys, 1 facts)
[LATENCY] Groq LLM Synthesis (70B) TTFT: 210ms
[LATENCY] TTS edge-tts first audio byte (sentence 1): 260ms
[LATENCY] * FIRST AUDIO BYTE SENT (end-of-speech -> first audio): 980ms (cumulative since end-of-speech)
```
Lines starting with `*` are cumulative milestones measured from the moment you stopped speaking. The most important one is **end-of-speech -> first audio**. The numbers above are illustrative; your values depend on region, network and Groq load.

| Stage | Where the time goes | Knob |
|---|---|---|
| VAD + chunking | pause timeout (not paid on the last chunk because `end` flushes it) | `VAD_PAUSE_MS` |
| STT | one Groq Whisper call on the final chunk | `GROQ_STT_MODEL` |
| Router | 8B JSON call | smaller `max_tokens`, `GROQ_ROUTER_MODEL` |
| Retrieval | Redis (1 pipelined HTTP call) vs embed + pgvector | co-locate regions |
| Synthesis | 70B TTFT; first sentence emitted at the first comma or period | `GROQ_SYNTH_MODEL` |
| TTS | edge-tts websocket handshake plus first bytes, per sentence | `TTS_VOICE` |

## 8. Design notes and limits
- **Streaming STT:** each pause-delimited chunk is transcribed while you are still talking. Only the last chunk is transcribed after you stop.
- **Filters:** Silero probability gate + RMS gate (noise), fewer than `MIN_WORDS` words dropped, plus a small list of known Whisper hallucinations ("thank you", "you") and a no-speech/logprob check.
- **Router:** one 8B call tags every box as fact/query/chitchat and extracts entities and absolute dates. Facts go to Agent Alpha (second 8B call that normalises them into atomic facts) and are written to Redis and Supabase concurrently. Queries go Redis first (entity/date hot keys) and then Supabase pgvector.
- **Pure-ingestion turns** use a template acknowledgement (no 70B call) to save latency.
- **Redis semantics:** any hit on a hot key counts as a hit. A partial hit can therefore skip older Supabase facts. To always merge both, call both in `retrieve_node` instead of falling back.
- **Concurrency:** each WebSocket gets its own Silero model copy. Run one uvicorn worker per CPU core behind a load balancer. Sticky routing is not needed because turns are independent.
- **No barge-in:** the server handles one turn at a time per connection.
- **Conversation memory:** there is no multi-turn chat memory beyond the stored facts. Add a short rolling history to `synth_node` if needed.
- **Edge-tts** uses Microsoft's unofficial Edge endpoint. It has no SLA, so swap in a licensed TTS for production.
- **Security:** put auth (token in the WebSocket query string or first message) in front of `/ws/voice` before exposing it publicly.

## 9. Troubleshooting
| Symptom | Fix |
|---|---|
| `PortAudio library not found` | install `libportaudio2` / `portaudio` (step 0) |
| `Missing required env vars` | check `.env` is in the directory you launch from |
| Groq 404 / model not found | update `GROQ_*_MODEL` to a model listed in the Groq console |
| `vector` dimension mismatch | `EMBED_DIM`, `vector(N)` and `EMBED_MODEL` must agree |
| Everything filtered as noise | lower `VAD_RMS_GATE` (e.g. 80) or `VAD_THRESHOLD` (0.4) |
| Sentences cut mid-thought | raise `VAD_PAUSE_MS` (e.g. 900) |
| No playback sound | check default output device; `python -c "import sounddevice as sd; print(sd.query_devices())"` |
| `miniaudio` decode error | edge-tts returned an empty body (voice name typo or network); check `TTS_VOICE` |
