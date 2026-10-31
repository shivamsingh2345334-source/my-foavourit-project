"""
Central configuration. Every tunable lives here and can be overridden via .env.
"""
import os
from dataclasses import dataclass
from datetime import datetime

from dotenv import load_dotenv

load_dotenv()


def _f(name: str, default: float) -> float:
    return float(os.getenv(name, default))


def _i(name: str, default: int) -> int:
    return int(os.getenv(name, default))


@dataclass(frozen=True)
class Settings:
    # ---------------- Groq ----------------
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    GROQ_STT_MODEL: str = os.getenv("GROQ_STT_MODEL", "whisper-large-v3-turbo")
    GROQ_ROUTER_MODEL: str = os.getenv("GROQ_ROUTER_MODEL", "qwen/qwen3.8-27b")
    GROQ_SYNTH_MODEL: str = os.getenv("GROQ_SYNTH_MODEL", "qwen/qwen3.8-27b")
    STT_LANGUAGE: str = os.getenv("STT_LANGUAGE", "")  # "" = auto-detect, or "en", "hi", ...

    # ---------------- Upstash Redis ----------------
    UPSTASH_REDIS_REST_URL: str = os.getenv("UPSTASH_REDIS_REST_URL", "")
    UPSTASH_REDIS_REST_TOKEN: str = os.getenv("UPSTASH_REDIS_REST_TOKEN", "")
    REDIS_TTL_SECONDS: int = _i("REDIS_TTL_SECONDS", 86400)
    REDIS_MAX_PER_KEY: int = _i("REDIS_MAX_PER_KEY", 50)

    # ---------------- Supabase ----------------
    SUPABASE_URL: str = os.getenv("SUPABASE_URL", "")
    SUPABASE_SERVICE_KEY: str = os.getenv("SUPABASE_SERVICE_KEY", "")
    SUPABASE_FACTS_TABLE: str = os.getenv("SUPABASE_FACTS_TABLE", "facts")
    SUPABASE_MATCH_RPC: str = os.getenv("SUPABASE_MATCH_RPC", "match_facts")
    MATCH_COUNT: int = _i("MATCH_COUNT", 5)
    MATCH_THRESHOLD: float = _f("MATCH_THRESHOLD", 0.25)

    # ---------------- Embeddings (local, fastembed) ----------------
    EMBED_MODEL: str = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
    EMBED_DIM: int = _i("EMBED_DIM", 384)  # must match vector(N) in the SQL schema

    # ---------------- Edge ingestion / VAD ----------------
    SAMPLE_RATE: int = 16000  # Silero supports 8000 / 16000
    VAD_THRESHOLD: float = _f("VAD_THRESHOLD", 0.5)
    VAD_PAUSE_MS: int = _i("VAD_PAUSE_MS", 700)          # silence that closes a sentence-chunk
    VAD_MIN_SPEECH_MS: int = _i("VAD_MIN_SPEECH_MS", 250)
    VAD_PRE_ROLL_MS: int = _i("VAD_PRE_ROLL_MS", 300)
    VAD_MAX_CHUNK_S: int = _i("VAD_MAX_CHUNK_S", 20)
    VAD_RMS_GATE: float = _f("VAD_RMS_GATE", 150.0)      # int16 RMS below this = noise floor
    MIN_WORDS: int = _i("MIN_WORDS", 2)                  # word-threshold filter (< 2 words dropped)
    NO_SPEECH_PROB_MAX: float = _f("NO_SPEECH_PROB_MAX", 0.6)

    # ---------------- TTS ----------------
    TTS_VOICE: str = os.getenv("TTS_VOICE", "en-US-AriaNeural")
    TTS_RATE: str = os.getenv("TTS_RATE", "+0%")

    # ---------------- Server ----------------
    SERVER_HOST: str = os.getenv("SERVER_HOST", "0.0.0.0")
    SERVER_PORT: int = _i("SERVER_PORT", 8000)

    def validate(self) -> None:
        required = [
            "GROQ_API_KEY",
            "UPSTASH_REDIS_REST_URL",
            "UPSTASH_REDIS_REST_TOKEN",
            "SUPABASE_URL",
            "SUPABASE_SERVICE_KEY",
        ]
        missing = [k for k in required if not getattr(self, k)]
        if missing:
            raise RuntimeError(f"Missing required env vars in .env: {', '.join(missing)}")


settings = Settings()


def today_str() -> str:
    """Used inside prompts so the LLMs can resolve 'tomorrow', 'last Friday', etc."""
    return datetime.now().strftime("%A, %Y-%m-%d")
