"""
End-to-end voice tester.

  ENTER -> start recording   |   ENTER again -> stop and send

Default mode streams mic frames over the WebSocket WHILE you speak (server runs VAD + STT
per sentence-chunk in parallel). Use --batch to record first and send afterwards.
The strict timer `start_time = time.time()` starts the moment you press the stop-ENTER
(streaming mode) or right before the first byte is sent (batch mode).

Usage:
    python voice_tester.py
    python voice_tester.py --batch
    python voice_tester.py --client-tts          # server sends text only; edge-tts runs locally
    python voice_tester.py --device 1            # pick input device (see --list-devices)
"""
import argparse
import asyncio
import json
import queue
import threading
import time

import edge_tts
import miniaudio
import numpy as np
import sounddevice as sd
import websockets

SR = 16000
BLOCK = 1024            # 64 ms mic blocks
PLAY_SR = 24000         # edge-tts mp3 output rate


# =============================================================================
# Audio playback (decode mp3 -> PCM -> speakers) on a background thread
# =============================================================================
class Player:
    def __init__(self):
        self.q: "queue.Queue[bytes]" = queue.Queue()
        self.pending = 0
        self.first_audible = None
        self._lock = threading.Lock()
        threading.Thread(target=self._run, daemon=True).start()

    def reset(self):
        self.first_audible = None

    def enqueue(self, mp3: bytes):
        with self._lock:
            self.pending += 1
        self.q.put(mp3)

    def _run(self):
        while True:
            mp3 = self.q.get()
            try:
                dec = miniaudio.decode(
                    mp3, output_format=miniaudio.SampleFormat.SIGNED16,
                    nchannels=1, sample_rate=PLAY_SR,
                )
                pcm = np.frombuffer(dec.samples, dtype=np.int16)
                if self.first_audible is None:
                    self.first_audible = time.time()
                sd.play(pcm, PLAY_SR)
                sd.wait()
            except Exception as e:
                print(f"[PLAYBACK ERROR] {e!r}")
            finally:
                with self._lock:
                    self.pending -= 1

    def wait_done(self):
        while True:
            with self._lock:
                if self.pending <= 0:
                    return
            time.sleep(0.02)


async def synth_local(text: str, voice: str) -> bytes:
    """--client-tts: synthesise with edge-tts on the client."""
    t0 = time.time()
    buf = bytearray()
    async for ch in edge_tts.Communicate(text, voice).stream():
        if ch["type"] == "audio":
            buf.extend(ch["data"])
    print(f"[CLIENT LATENCY] edge-tts local synth: {(time.time() - t0) * 1000:.0f}ms")
    return bytes(buf)


async def ainput(prompt: str) -> str:
    return await asyncio.get_running_loop().run_in_executor(None, input, prompt)


# =============================================================================
# Record + send
# =============================================================================
async def record_and_send(ws, args) -> float:
    loop = asyncio.get_running_loop()
    q: asyncio.Queue = asyncio.Queue()
    frames = []

    def callback(indata, n_frames, time_info, status):
        if status:
            print(f"[MIC] {status}")
        loop.call_soon_threadsafe(q.put_nowait, bytes(indata))

    stream = sd.RawInputStream(
        samplerate=SR, channels=1, dtype="int16", blocksize=BLOCK,
        callback=callback, device=args.device,
    )

    await ainput("\n>>> Press ENTER to START recording... ")
    await ws.send(json.dumps({"type": "start", "tts": not args.client_tts}))

    async def pump():
        while True:
            data = await q.get()
            if data is None:
                return
            if args.batch:
                frames.append(data)
            else:
                await ws.send(data)

    stream.start()
    pump_task = asyncio.create_task(pump())
    await ainput(">>> RECORDING... press ENTER to STOP ")
    stream.stop()
    stream.close()

    if args.batch:
        q.put_nowait(None)
        await pump_task
        start_time = time.time()                 # strict timer starts right before sending
        blob = b"".join(frames)
        for i in range(0, len(blob), 32000):
            await ws.send(blob[i:i + 32000])
    else:
        start_time = time.time()                 # streaming: timer starts at stop
        q.put_nowait(None)
        await pump_task

    await ws.send(json.dumps({"type": "end"}))
    print(f"[CLIENT] sent end-of-speech marker (recorded {len(blob) / 2 / SR:.1f}s)" if args.batch
          else "[CLIENT] sent end-of-speech marker")
    return start_time


# =============================================================================
# Receive response
# =============================================================================
async def receive_turn(ws, player: Player, start_time: float, args) -> None:
    player.reset()
    t_transcript = t_sentence = t_audio = None
    audio_bufs = {}
    cur_idx = None

    async for raw in ws:
        now = time.time()

        if isinstance(raw, (bytes, bytearray)):
            if t_audio is None:
                t_audio = now
                end_time = time.time()
                print(f"[CLIENT LATENCY] FIRST AUDIO BYTE received: {(end_time - start_time) * 1000:.0f}ms")
            if cur_idx is not None:
                audio_bufs.setdefault(cur_idx, bytearray()).extend(raw)
            continue

        msg = json.loads(raw)
        kind = msg.get("type")

        if kind == "transcript":
            t_transcript = now
            print(f"[CLIENT LATENCY] transcript received: {(now - start_time) * 1000:.0f}ms")
            for b in msg["boxes"]:
                print(f"   #{b['box']}: {b['text']}")
            for d in msg.get("dropped", []):
                print(f"   (dropped: {d['text']!r} - {d['reason']})")
        elif kind == "intent":
            for it in msg["items"]:
                print(f"   intent #{it['box']}: {it['intent']}  entities={it['entities']} dates={it['dates']}")
        elif kind == "sentence":
            if t_sentence is None:
                t_sentence = now
                print(f"[CLIENT LATENCY] first response text: {(now - start_time) * 1000:.0f}ms")
            print(f"   BOT: {msg['text']}")
            if args.client_tts:
                mp3 = await synth_local(msg["text"], args.voice)
                player.enqueue(mp3)
        elif kind == "audio_start":
            cur_idx = msg["index"]
        elif kind == "audio_end":
            data = audio_bufs.pop(msg["index"], None)
            if data:
                player.enqueue(bytes(data))
        elif kind == "latency_report":
            print("\n---- server-side breakdown ----")
            for s in msg["steps"]:
                mark = "*" if s["kind"] == "milestone" else "-"
                print(f"  {mark} {s['name']:<58} {s['ms']:>8.0f} ms")
        elif kind == "error":
            print(f"[SERVER ERROR] {msg}")
        elif kind == "done":
            if msg.get("note"):
                print(f"[SERVER] {msg['note']}")
            break

    t_done = time.time()
    await asyncio.get_running_loop().run_in_executor(None, player.wait_done)
    t_played = time.time()

    print("\n================ CLIENT LATENCY REPORT ================")
    def line(label, t):
        if t is not None:
            print(f"  {label:<42} {(t - start_time) * 1000:>8.0f} ms")
    line("stop -> transcript", t_transcript)
    line("stop -> first response text", t_sentence)
    line("stop -> first audio byte", t_audio)
    line("stop -> first audible sound", player.first_audible)
    line("stop -> server done", t_done)
    line("stop -> playback finished", t_played)
    print("=======================================================")


# =============================================================================
async def main(args):
    if args.list_devices:
        print(sd.query_devices())
        return
    player = Player()
    print(f"Connecting to {args.url} ...")
    async with websockets.connect(args.url, max_size=None, ping_interval=20) as ws:
        print("Connected. Ctrl+C to quit.")
        while True:
            start_time = await record_and_send(ws, args)
            await receive_turn(ws, player, start_time, args)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="ws://localhost:8000/ws/voice")
    ap.add_argument("--batch", action="store_true", help="record fully, then send")
    ap.add_argument("--client-tts", action="store_true", help="synthesise speech locally with edge-tts")
    ap.add_argument("--voice", default="en-US-AriaNeural")
    ap.add_argument("--device", type=int, default=None, help="input device index")
    ap.add_argument("--list-devices", action="store_true")
    try:
        asyncio.run(main(ap.parse_args()))
    except (KeyboardInterrupt, EOFError):
        print("\nBye.")
