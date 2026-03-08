"""Shared Groq client with multi-key fallback. Use GROQ_API_KEYS (comma-separated) or GROQ_API_KEY in env."""
import os
import random
import logging
import threading
from groq import Groq

logger = logging.getLogger(__name__)

def _get_keys():
    keys_str = os.getenv("GROQ_API_KEYS")
    if keys_str:
        keys = [k.strip() for k in keys_str.replace("\n", ",").split(",") if k.strip()]
        if keys:
            return keys
    single = os.getenv("GROQ_API_KEY")
    return [single.strip()] if single and single.strip() else []

GROQ_KEYS = _get_keys()

# With multiple Gunicorn workers, each process has its own memory so round-robin would
# make every worker start at key 0 → all concurrent calls hit same key → 429s.
# Use random selection so each request (from any worker) picks a random key → even spread.
_key_lock = threading.Lock()

def get_next_key_index():
    """Return a key index to use for first attempt. With 2+ keys: random spread (works across workers)."""
    if not GROQ_KEYS:
        return 0
    n = len(GROQ_KEYS)
    if n == 1:
        return 0
    with _key_lock:
        return random.randint(0, n - 1)

def get_client(key_index=None):
    """Get Groq client. If key_index is None and multiple keys exist, use round-robin for first call."""
    if not GROQ_KEYS:
        raise ValueError("No GROQ API keys. Set GROQ_API_KEY or GROQ_API_KEYS in env.")
    if key_index is None:
        key_index = get_next_key_index()
    idx = key_index % len(GROQ_KEYS)
    # 1 key: allow retries (only option is to wait on 429). 2+ keys: fail fast, try next key.
    max_retries = 1 if len(GROQ_KEYS) == 1 else 0
    return Groq(api_key=GROQ_KEYS[idx], timeout=25.0, max_retries=max_retries)

def num_keys():
    return len(GROQ_KEYS)
