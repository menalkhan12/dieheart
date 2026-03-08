# Gevent must patch before any other imports (fixes MonkeyPatchWarning / ssl issues)
from gevent import monkey
monkey.patch_all()

import os
import re
import json
import uuid
import threading
import logging
from flask import Flask, request, jsonify, render_template, Response, stream_with_context
from dotenv import load_dotenv

from rag import generate_answer, generate_answer_stream, initialize_rag
from stt import transcribe_audio
from tts import (
    generate_tts, get_and_clear, get_full_audio_bytes,
    _get_ios_cached, _set_ios_cached, _is_ios_generating, _wait_for_ios_cache,
    _mark_ios_generating, _clear_ios_generating,
    set_tts_cached, get_tts_cached,
)
from groq_utils import GROQ_KEYS
from livekit_utils import generate_livekit_token
from utils import (
    init_call_record,
    update_call_record,
    end_call_record,
    append_lead_log,
    detect_phone_number,
    get_last_user_query,
    get_recent_turns,
)

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
log_lock = threading.Lock()

try:
    initialize_rag()
    logger.info("RAG system initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize RAG: {e}")


def _is_ios(user_agent: str) -> bool:
    ua = (user_agent or "").lower()
    return any(x in ua for x in ("iphone", "ipad", "ipod", "crios", "fxios"))


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/health")
def health():
    ok = bool(GROQ_KEYS)
    return jsonify({
        "status": "OK" if ok else "degraded",
        "groq_configured": ok,
        "hint": None if ok else "Set GROQ_API_KEY in Render Dashboard → Environment"
    }), 200


# ── TTS Stream endpoint ────────────────────────────────────────────────────
@app.route("/api/tts_stream/<token>")
def tts_stream(token):
    """
    Streams TTS audio to the client.

    Android/Chrome: true chunked streaming — first bytes in ~300ms.
    iOS Safari:     buffers full audio in memory then sends as one response.
                    (iOS doesn't support chunked-transfer MP3 reliably)

    Either way this is faster than the old disk-write approach.
    """
    ua = request.headers.get("User-Agent", "")
    ios = _is_ios(ua)

    # iOS Safari often requests same URL 2x (range/seek). Serve from cache.
    if ios:
        cached = _get_ios_cached(token)
        if cached:
            audio_bytes, mimetype = cached
            return Response(
                audio_bytes,
                mimetype=mimetype,
                headers={
                    "Content-Length": str(len(audio_bytes)),
                    "Cache-Control": "public, max-age=3600",
                    "Accept-Ranges": "bytes",
                }
            )

    text = get_and_clear(token)
    if not text:
        # iOS: second request may arrive before first finishes; wait for cache
        if ios and _is_ios_generating(token):
            cached = _wait_for_ios_cache(token)
            if cached:
                audio_bytes, mimetype = cached
                return Response(
                    audio_bytes,
                    mimetype=mimetype,
                    headers={
                        "Content-Length": str(len(audio_bytes)),
                        "Cache-Control": "public, max-age=3600",
                        "Accept-Ranges": "bytes",
                    }
                )
        # Retry / duplicate request: serve from shared cache if we generated it before
        cached = get_tts_cached(token)
        if cached:
            audio_bytes, mimetype = cached
            return Response(
                audio_bytes,
                mimetype=mimetype,
                headers={
                    "Content-Length": str(len(audio_bytes)),
                    "Cache-Control": "public, max-age=3600",
                    "Accept-Ranges": "bytes",
                }
            )
        logger.warning(f"TTS token not found: {token[:8]}")
        return Response("Token not found or expired", status=404)

    if ios:
        # Buffer entire audio in memory, send as complete response
        # Mark generating so concurrent Safari requests can wait for cache
        _mark_ios_generating(token)
        logger.info(f"TTS buffered mode (iOS): token {token[:8]}")
        try:
            audio_bytes, mimetype = get_full_audio_bytes(text)
            if not audio_bytes:
                _clear_ios_generating(token)
                return Response("TTS generation failed", status=500)
            _set_ios_cached(token, audio_bytes, mimetype)
            set_tts_cached(token, audio_bytes, mimetype)
            return Response(
                audio_bytes,
                mimetype=mimetype,
                headers={
                    "Content-Length": str(len(audio_bytes)),
                    "Cache-Control": "public, max-age=3600",
                    "Accept-Ranges": "bytes",
                }
            )
        except Exception as e:
            logger.error(f"iOS TTS buffer error: {e}")
            _clear_ios_generating(token)
            return Response("TTS error", status=500)
    else:
        # Non-iOS: single chunk (Groq WAV or edge-tts MP3)
        logger.info(f"TTS stream (non-iOS): token {token[:8]}")
        audio_bytes, mimetype = get_full_audio_bytes(text)
        if not audio_bytes:
            return Response("TTS generation failed", status=500)
        set_tts_cached(token, audio_bytes, mimetype)
        return Response(
            audio_bytes,
            mimetype=mimetype,
            headers={
                "Content-Length": str(len(audio_bytes)),
                "Cache-Control": "public, max-age=3600",
                "Accept-Ranges": "bytes",
            }
        )


@app.route("/api/start_call", methods=["POST"])
def start_call():
    if not GROQ_KEYS:
        return jsonify({"error": "GROQ_API_KEY not set. Add it in Render Dashboard → Environment."}), 503
    try:
        session_id = str(uuid.uuid4())
        room_name = f"room_{session_id}"
        init_call_record(session_id)
        token = generate_livekit_token(room_name, session_id)
        greeting_text = "Hello, this is Institute of Space Technology. How can I help you today?"
        audio_url = generate_tts(greeting_text, session_id)
        logger.info(f"Call started: {session_id}")
        return jsonify({
            "session_id": session_id,
            "livekit_token": token,
            "room_name": room_name,
            "greeting_audio_url": audio_url
        }), 200
    except Exception as e:
        logger.error(f"Error starting call: {e}")
        return jsonify({"error": "Failed to start call"}), 500


@app.route("/api/query", methods=["POST"])
def query():
    try:
        session_id = request.form.get("session_id")
        audio_file = request.files.get("audio")

        if not session_id or not audio_file:
            return jsonify({"error": "Invalid request"}), 400

        filename = audio_file.filename or ""
        content_type = audio_file.content_type or ""
        if filename.endswith(".wav") or "wav" in content_type:
            ext = "wav"
        elif filename.endswith(".mp4") or filename.endswith(".m4a") or "mp4" in content_type or "m4a" in content_type:
            ext = "mp4"
        elif filename.endswith(".ogg") or "ogg" in content_type:
            ext = "ogg"
        else:
            ext = "webm"
        temp_path = f"temp_{uuid.uuid4()}.{ext}"
        audio_file.save(temp_path)

        try:
            user_text = transcribe_audio(temp_path)
            logger.info(f"User query [{session_id}]: {user_text}")
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

        NOISE_PHRASES = {
            "thank you", "thanks", "you're welcome", "ok", "okay",
            "yes", "no", "hmm", "uh", "um", "uh huh", "mm hmm",
            "bye", "hello", "hi", "hey",
        }
        stripped = (user_text or "").strip()
        is_noise = (
            not stripped
            or len(stripped) < 5
            or stripped.lower().rstrip(".!?,") in NOISE_PHRASES
        )
        if is_noise:
            return jsonify({"ignored": True, "user_text": stripped}), 200

        # Phone number
        phone = detect_phone_number(user_text)
        if phone:
            unanswered_query = get_last_user_query(session_id) or user_text
            append_lead_log(session_id, phone, unanswered_query)
            reply = "Thank you. Our admissions office will contact you soon."
            audio_url = generate_tts(reply, session_id)
            update_call_record(session_id, user_text, reply, escalated=True, phone=phone)
            logger.info(f"Phone detected: {phone}")
            return jsonify({"user_text": user_text, "response": reply, "audio_url": audio_url}), 200

        # RAG
        history = get_recent_turns(session_id, n=10)
        reply, escalated = generate_answer(user_text, conversation_history=history)
        audio_url = generate_tts(reply, session_id)
        update_call_record(session_id, user_text, reply, escalated=escalated)
        logger.info(f"Agent response [{session_id}]: {reply} (escalated: {escalated})")

        return jsonify({"user_text": user_text, "response": reply, "audio_url": audio_url}), 200

    except Exception as e:
        logger.error(f"Error processing query: {e}")
        return jsonify({"error": "Failed to process query"}), 500


@app.route("/api/end_call", methods=["POST"])
def end_call():
    try:
        session_id = request.json.get("session_id")
        if not session_id:
            return jsonify({"error": "Invalid request"}), 400
        end_call_record(session_id)
        logger.info(f"Call ended: {session_id}")
        return jsonify({"status": "ended"}), 200
    except Exception as e:
        logger.error(f"Error ending call: {e}")
        return jsonify({"error": "Failed to end call"}), 500


def _split_sentences(text):
    """Split reply into sentences for streaming TTS to reduce perceived latency."""
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    return [p.strip() for p in parts if p.strip()]


def _sse(obj):
    """Format a dict as an SSE data line."""
    return "data: " + json.dumps(obj) + "\n\n"


def _detect_ext(audio_file):
    filename = audio_file.filename or ""
    content_type = audio_file.content_type or ""
    if filename.endswith(".wav") or "wav" in content_type:
        return "wav"
    if filename.endswith(".mp4") or filename.endswith(".m4a") or "mp4" in content_type:
        return "mp4"
    if filename.endswith(".ogg") or "ogg" in content_type:
        return "ogg"
    return "webm"


@app.route("/api/query_stream", methods=["POST"])
def query_stream():
    """
    SSE endpoint for Android/Desktop VAD mode.
    Streams: transcript → sentence(s) with audio_url → done
    Client plays sentence 1 while sentence 2 TTS is still generating.
    """
    session_id = request.form.get("session_id")
    audio_file = request.files.get("audio")

    if not session_id or not audio_file:
        return jsonify({"error": "Invalid request"}), 400

    ext = _detect_ext(audio_file)
    temp_path = f"temp_{uuid.uuid4()}.{ext}"
    audio_file.save(temp_path)

    try:
        user_text = transcribe_audio(temp_path)
        logger.info(f"[stream] Transcript [{session_id}]: {user_text}")
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

    def generate():
        try:
            stripped = user_text.strip() if user_text else ""

            # Reject empty or too-short transcripts
            NOISE_PHRASES = {
                "thank you", "thanks", "you're welcome", "ok", "okay",
                "yes", "no", "hmm", "uh", "um", "uh huh", "mm hmm",
                "bye", "hello", "hi", "hey",
            }
            if not stripped or len(stripped) < 5:
                yield _sse({"type": "ignored"})
                yield _sse({"type": "done"})
                return
            if stripped.lower().rstrip(".!?,") in NOISE_PHRASES:
                yield _sse({"type": "ignored"})
                yield _sse({"type": "done"})
                return

            lower = stripped.lower()

            # Fix 4: Detect end call BEFORE sending to RAG
            from rag import _is_end_call
            if _is_end_call(lower):
                reply = "Thank you for calling IST. Have a great day! Goodbye."
                audio_url = generate_tts(reply, session_id)
                end_call_record(session_id)
                yield _sse({"type": "end_call", "text": reply, "audio_url": audio_url})
                return

            # Fix 3: Block off-topic STT hallucinations
            # These are phrases Whisper hallucinates that are clearly NOT IST admission queries
            OFF_TOPIC_TRIGGERS = [
                "terrestrial", "relationship between", "solar system", "milky way",
                "planet earth", "mars", "jupiter", "saturn", "black hole",
                "nasa", "spacex", "elon musk", "quantum physics",
                "theory of relativity", "big bang", "evolution",
                "world war", "history of", "recipe", "cook",
            ]
            IST_KEYWORDS = [
                "ist", "admission", "fee", "merit", "scholarship", "hostel",
                "transport", "program", "department", "engineering", "aerospace",
                "electrical", "computer", "avionics", "mechanical", "mathematics",
                "physics", "biotechnology", "apply", "eligibility", "entry test",
            ]
            is_off_topic = any(t in lower for t in OFF_TOPIC_TRIGGERS)
            has_ist_context = any(k in lower for k in IST_KEYWORDS)
            if is_off_topic and not has_ist_context:
                logger.warning(f"[stream] Off-topic/hallucinated STT blocked: '{stripped}'")
                yield _sse({"type": "ignored"})
                yield _sse({"type": "done"})
                return

            yield _sse({"type": "transcript", "text": user_text})

            phone = detect_phone_number(user_text)
            if phone:
                unanswered_query = get_last_user_query(session_id) or user_text
                append_lead_log(session_id, phone, unanswered_query)
                reply = "Thank you. Our admissions office will contact you soon."
                update_call_record(session_id, user_text, reply, escalated=True, phone=phone)
                audio_url = generate_tts(reply, session_id)
                yield _sse({"type": "sentence", "text": reply, "audio_url": audio_url})
                yield _sse({"type": "done"})
                return

            history = get_recent_turns(session_id, n=10)
            reply_parts = []
            escalated = False
            for sentence, esc in generate_answer_stream(user_text, conversation_history=history):
                if sentence:
                    reply_parts.append(sentence)
                    audio_url = generate_tts(sentence, session_id)
                    yield _sse({"type": "sentence", "text": sentence, "audio_url": audio_url})
                if esc is not None:
                    escalated = esc
            full_reply = " ".join(reply_parts)
            update_call_record(session_id, user_text, full_reply, escalated=escalated)
            if full_reply:
                logger.info(f"[stream] Agent reply [{session_id}]: {full_reply}")

            yield _sse({"type": "done"})

        except Exception as e:
            logger.error(f"query_stream error: {e}")
            yield _sse({"type": "error", "text": "Server error"})
            yield _sse({"type": "done"})

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        }
    )


@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404


@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal error: {error}")
    return jsonify({"error": "Internal server error"}), 500


if __name__ == "__main__":
    app.run(debug=os.getenv("FLASK_ENV") == "development")