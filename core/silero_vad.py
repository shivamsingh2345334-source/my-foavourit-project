"""
Layer 1 - Edge ingestion & filtering.

* Silero VAD (PyTorch) run frame-by-frame (512 samples = 32 ms @ 16 kHz).
* collections.deque RAM buffers: a pre-roll ring buffer and the in-flight speech buffer.
* Async-friendly pause-based sentence chunking: a chunk closes after VAD_PAUSE_MS of silence.
* Environmental-noise filter: speech-probability gate + RMS energy gate.
* Word-threshold filter + numbering of transcripts into text boxes (#1, #2, ...).
"""
import collections
import copy
import io
import re
import time
import wave
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np
import torch

from core.config import settings
from core.latency_tracker import LatencyTracker, get_tracker

torch.set_num_threads(1)  # tiny per-frame inference: 1 thread is fastest

SR = settings.SAMPLE_RATE
FRAME_SAMPLES = 512 if SR == 16000 else 256
FRAME_BYTES = FRAME_SAMPLES * 2
FRAME_MS = FRAME_SAMPLES * 1000.0 / SR  # 32 ms

_BASE_MODEL = None


def load_base_model():
    """Load Silero once per process (call from startup)."""
    global _BASE_MODEL
    if _BASE_MODEL is None:
        from silero_vad import load_silero_vad

        t0 = time.time()
        _BASE_MODEL = load_silero_vad()
        print(f"[LATENCY] Silero VAD model load: {(time.time() - t0) * 1000:.0f}ms")
    return _BASE_MODEL


def _new_session_model():
    """Silero's model is stateful (LSTM), so each session needs its own copy."""
    base = load_base_model()
    try:
        return copy.deepcopy(base)
    except Exception:
        from silero_vad import load_silero_vad

        return load_silero_vad()


def pcm_to_wav(pcm: bytes, sr: int = SR) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm)
    return buf.getvalue()


@dataclass
class SpeechChunk:
    index: int
    pcm: bytes
    duration_ms: float
    rms: float
    reason: str
    created_at: float


class EdgeVAD:
    """
    Feed raw int16 mono PCM of any size via feed(); returns finished SpeechChunks.
    Call flush() when the user stops speaking to close the in-flight chunk immediately
    (no need to wait for the pause timeout).
    """

    def __init__(self, tracker: Optional[LatencyTracker] = None):
        self.tracker = get_tracker(tracker)
        self.model = _new_session_model()
        self._reset_model()

        self._carry = bytearray()  # partial frame bytes between feed() calls
        pre_roll_frames = max(1, int(round(settings.VAD_PRE_ROLL_MS / FRAME_MS)))
        self._pre_roll: Deque[bytes] = collections.deque(maxlen=pre_roll_frames)
        self._speech: Deque[bytes] = collections.deque()
        self._in_speech = False
        self._silence = 0
        self._voiced = 0
        self._neg_thr = max(settings.VAD_THRESHOLD - 0.15, 0.01)
        self._chunk_count = 0
        self.compute_ms = 0.0

    # ------------------------------------------------------------------ public
    def feed(self, pcm: bytes) -> List[SpeechChunk]:
        t0 = time.time()
        self._carry.extend(pcm)
        n_frames = len(self._carry) // FRAME_BYTES
        out: List[SpeechChunk] = []
        if n_frames:
            buf = bytes(self._carry[: n_frames * FRAME_BYTES])
            del self._carry[: n_frames * FRAME_BYTES]
            for i in range(n_frames):
                chunk = self._process_frame(buf[i * FRAME_BYTES:(i + 1) * FRAME_BYTES])
                if chunk:
                    out.append(chunk)
        self.compute_ms += (time.time() - t0) * 1000
        return out

    def flush(self) -> List[SpeechChunk]:
        out: List[SpeechChunk] = []
        if self._in_speech:
            chunk = self._finalize("end_of_stream")
            if chunk:
                out.append(chunk)
        self.tracker.record("Silero VAD compute (all frames this turn)", self.compute_ms)
        return out

    # ----------------------------------------------------------------- private
    def _reset_model(self) -> None:
        if hasattr(self.model, "reset_states"):
            self.model.reset_states()

    def _process_frame(self, frame: bytes) -> Optional[SpeechChunk]:
        x = torch.from_numpy(np.frombuffer(frame, dtype=np.int16).astype(np.float32) / 32768.0)
        with torch.no_grad():
            prob = float(self.model(x, SR).item())

        thr = settings.VAD_THRESHOLD
        if not self._in_speech:
            if prob >= thr:
                self._in_speech = True
                self._speech = collections.deque(self._pre_roll)  # include pre-roll
                self._speech.append(frame)
                self._pre_roll.clear()
                self._silence = 0
                self._voiced = 1
            else:
                self._pre_roll.append(frame)
            return None

        self._speech.append(frame)
        if prob >= self._neg_thr:
            self._silence = 0
            if prob >= thr:
                self._voiced += 1
        else:
            self._silence += 1

        too_long = len(self._speech) * FRAME_MS >= settings.VAD_MAX_CHUNK_S * 1000
        if self._silence * FRAME_MS >= settings.VAD_PAUSE_MS:
            return self._finalize("pause")
        if too_long:
            return self._finalize("max_length")
        return None

    def _finalize(self, reason: str) -> Optional[SpeechChunk]:
        frames = list(self._speech)
        keep_tail = int(200 / FRAME_MS)  # keep ~200 ms of trailing silence
        if self._silence > keep_tail:
            del frames[len(frames) - (self._silence - keep_tail):]
        voiced_ms = self._voiced * FRAME_MS

        # reset state for the next chunk
        self._in_speech = False
        self._speech = collections.deque()
        self._pre_roll.clear()
        self._silence = 0
        self._voiced = 0
        self._reset_model()

        if voiced_ms < settings.VAD_MIN_SPEECH_MS:
            print(f"[VAD] dropped blip ({voiced_ms:.0f}ms voiced < {settings.VAD_MIN_SPEECH_MS}ms)")
            return None

        pcm = b"".join(frames)
        samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
        rms = float(np.sqrt(np.mean(samples ** 2))) if samples.size else 0.0
        if rms < settings.VAD_RMS_GATE:
            print(f"[VAD] dropped as environmental noise (rms={rms:.0f} < gate {settings.VAD_RMS_GATE:.0f})")
            return None

        self._chunk_count += 1
        dur = len(pcm) / 2 / SR * 1000
        print(f"[VAD] chunk #{self._chunk_count} closed ({reason}): {dur:.0f}ms audio, rms={rms:.0f}")
        return SpeechChunk(self._chunk_count, pcm, dur, rms, reason, time.time())


# ----------------------------------------------------------------------------
# Word-threshold filter + text-box numbering (end of Layer 1/2)
# ----------------------------------------------------------------------------
_HALLUCINATIONS = {
    "thank you", "thanks for watching", "thank you for watching",
    "please subscribe", "you", "bye", "subtitles by the amaraorg community",
}


def build_text_boxes(results: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
    """
    results: dicts from GroqService.transcribe() + {"chunk": int, "duration_ms": float}.
    Returns (boxes, dropped). Boxes are numbered #1, #2, ... in speaking order.
    """
    boxes: List[Dict] = []
    dropped: List[Dict] = []
    for r in sorted(results, key=lambda x: x["chunk"]):
        text = (r.get("text") or "").strip()
        words = len(text.split())
        norm = re.sub(r"[^\w\s]", "", text.lower()).strip()
        lp = r.get("avg_logprob")
        lp = 0.0 if lp is None else lp
        nsp = r.get("no_speech_prob") or 0.0

        reason = None
        if words < settings.MIN_WORDS:
            reason = f"fewer than {settings.MIN_WORDS} words"
        elif norm in _HALLUCINATIONS:
            reason = "whisper hallucination"
        elif nsp > settings.NO_SPEECH_PROB_MAX and lp < -1.0:
            reason = f"likely noise (no_speech={nsp:.2f}, logprob={lp:.2f})"

        if reason:
            dropped.append({"chunk": r["chunk"], "text": text, "reason": reason})
            print(f"[FILTER] dropped chunk {r['chunk']} ({reason}): {text!r}")
            continue

        boxes.append({
            "box": len(boxes) + 1,
            "text": text,
            "chunk": r["chunk"],
            "stt_ms": round(r.get("stt_ms", 0.0), 1),
            "duration_ms": round(r.get("duration_ms", 0.0), 1),
        })

    print(f"[FILTER] {len(boxes)} text box(es) kept, {len(dropped)} dropped")
    for b in boxes:
        print(f"         #{b['box']}: {b['text']}")
    return boxes, dropped
