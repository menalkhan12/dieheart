"""TTS: edge-tts only. Groq skipped."""
from __future__ import annotations
import asyncio
import threading
import uuid
import io
import time
import logging
from groq_utils import get_client, get_next_key_index, num_keys, GROQ_KEYS

logger = logging.getLogger(__name__)

GROQ_MODEL = "canopylabs/orpheus-v1-english"
GROQ_VOICE = "hannah"

_pending = {}
_pending_lock = threading.Lock()

_ios_cache = {}
_ios_cache_lock = threading.Lock()
_ios_cache_max_age = 120
_ios_cache_times = {}
_ios_generating = set()
_ios_generating_lock = threading.Lock()

# Shared cache for all clients: serves retries when token was already consumed (avoids 404)
_tts_cache = {}
_tts_cache_lock = threading.Lock()
_tts_cache_max_age = 300  # 5 min
_tts_cache_times = {}


def generate_tts(text: str, session_id: str) -> str:
    token = str(uuid.uuid4())
    with _pending_lock:
        _pending[token] = text
    logger.info(f"TTS token: {token[:8]}… text len {len(text)}")
    return f"/api/tts_stream/{token}"


def get_and_clear(token: str):
    with _pending_lock:
        return _pending.pop(token, None)


def set_tts_cached(token: str, audio: bytes, mimetype: str = "audio/mpeg"):
    with _tts_cache_lock:
        now = time.time()
        for k in list(_tts_cache_times.keys()):
            if now - _tts_cache_times[k] > _tts_cache_max_age:
                _tts_cache.pop(k, None)
                _tts_cache_times.pop(k, None)
        _tts_cache[token] = (audio, mimetype)
        _tts_cache_times[token] = now


def get_tts_cached(token: str):
    with _tts_cache_lock:
        return _tts_cache.get(token)


def _get_ios_cached(token: str):
    with _ios_cache_lock:
        return _ios_cache.get(token)


def _is_ios_generating(token: str) -> bool:
    with _ios_generating_lock:
        return token in _ios_generating


def _mark_ios_generating(token: str):
    with _ios_generating_lock:
        _ios_generating.add(token)


def _clear_ios_generating(token: str):
    with _ios_generating_lock:
        _ios_generating.discard(token)


def _set_ios_cached(token: str, audio: bytes, mimetype: str = "audio/wav"):
    with _ios_generating_lock:
        _ios_generating.discard(token)
    with _ios_cache_lock:
        now = time.time()
        for k in list(_ios_cache_times.keys()):
            if now - _ios_cache_times[k] > _ios_cache_max_age:
                _ios_cache.pop(k, None)
                _ios_cache_times.pop(k, None)
        _ios_cache[token] = (audio, mimetype)
        _ios_cache_times[token] = now


def _wait_for_ios_cache(token: str, timeout: float = 15.0):
    step = 0.2
    elapsed = 0.0
    while elapsed < timeout:
        cached = _get_ios_cached(token)
        if cached:
            return cached
        time.sleep(step)
        elapsed += step
    return None


def _groq_tts_bytes(text: str) -> bytes:
    if not GROQ_KEYS or not text.strip():
        return b""
    first_key = get_next_key_index()
    key_order = [first_key] + [i for i in range(num_keys()) if i != first_key]
    for key_idx in key_order:
        try:
            client = get_client(key_idx)
            resp = client.audio.speech.create(
                model=GROQ_MODEL,
                voice=GROQ_VOICE,
                input=text.strip(),
                response_format="wav",
            )
            data = None
            if hasattr(resp, "content") and resp.content:
                data = resp.content
            elif hasattr(resp, "read") and callable(resp.read):
                data = resp.read()
            elif hasattr(resp, "write_to_file"):
                buf = io.BytesIO()
                resp.write_to_file(buf)
                data = buf.getvalue()
            if data:
                logger.info(f"Groq TTS: {len(data)} bytes")
                return data
        except Exception as e:
            err = str(e).lower()
            if "429" in err or "rate" in err:
                logger.warning(f"Groq TTS key {key_idx+1} rate limited")
                continue
            if "400" in err or "terms" in err:
                logger.warning("Groq TTS terms not accepted — use edge-tts fallback. Accept at: console.groq.com/playground?model=canopylabs/orpheus-v1-english")
            logger.error(f"Groq TTS: {e}")
            break
    return b""


# Single consistent voice: en-US-JennyNeural (female, neutral, clear for phone)
EDGE_VOICE = "en-US-JennyNeural"
EDGE_RATE = "+15%"  # Faster for ~1 sec response feel; reduces pauses after dots


def _edge_tts_bytes(text: str) -> bytes:
    try:
        import edge_tts
        chunks = []

        async def _run():
            comm = edge_tts.Communicate(text, EDGE_VOICE, rate=EDGE_RATE, pitch="+0Hz")
            async for c in comm.stream():
                if c.get("type") == "audio":
                    chunks.append(c["data"])

        asyncio.run(_run())
        return b"".join(chunks) if chunks else b""
    except Exception as e:
        logger.error(f"edge-tts error: {e}")
        return b""


def get_full_audio_bytes(text: str) -> tuple[bytes, str]:
    """Returns (audio_bytes, mimetype). edge-tts only."""
    data = _edge_tts_bytes(text)
    if data:
        return (data, "audio/mpeg")
    return (b"", "audio/mpeg")


def stream_tts_chunks(text: str):
    """Yield full audio in one chunk (Groq WAV or edge-tts MP3)."""
    data, _ = get_full_audio_bytes(text)
    if data:
        yield data

