import os
import logging
from groq_utils import get_client, num_keys, GROQ_KEYS

logger = logging.getLogger(__name__)

# Max 896 characters allowed by Groq Whisper
WHISPER_PROMPT = (
    "IST, Institute of Space Technology. fee structure, semester fee, "
    "BS Electrical Engineering, BS Computer Engineering, BS Aerospace Engineering, "
    "BS Avionics Engineering, BS Mechanical Engineering, BS Software Engineering, "
    "BS Computer Science, BS Space Science, BS Mathematics, BS Biotechnology, "
    "BS Data Science, BS Artificial Intelligence, BS Materials Science, "
    "merit based scholarship, closing merit, merit aggregate, merit 2024, "
    "admission open, last date to apply, entry test, FSc, matric, DAE, A-level, "
    "hostel charges, transport, bus route, lakh, rupees, eligibility"
)

MIN_AUDIO_BYTES = 1000


def transcribe_audio(file_path):
    try:
        if not GROQ_KEYS:
            logger.error("GROQ_API_KEY / GROQ_API_KEYS not set")
            return ""

        if not os.path.exists(file_path):
            logger.error(f"Audio file not found: {file_path}")
            return ""

        file_size = os.path.getsize(file_path)
        logger.info(f"Audio file size: {file_size} bytes")

        if file_size < MIN_AUDIO_BYTES:
            logger.warning(f"Audio file too small ({file_size} bytes) — skipping.")
            return ""

        for key_idx in range(num_keys()):
            try:
                client = get_client(key_idx)
                with open(file_path, "rb") as audio:
                    transcription = client.audio.transcriptions.create(
                        file=(os.path.basename(file_path), audio),
                        model="whisper-large-v3",
                        language="en",
                        prompt=WHISPER_PROMPT
                    )
                text = transcription.text.strip()
                logger.info(f"Transcription result: '{text}'")
                return text

            except Exception as e:
                err_str = str(e).lower()
                logger.error(f"STT error (key {key_idx+1}): {e}")

                if "400" in err_str or "bad request" in err_str:
                    logger.error(f"Groq 400 error — check prompt length or audio format. File size: {file_size} bytes")
                    return ""

                if "429" in err_str or "rate" in err_str or "quota" in err_str:
                    logger.warning(f"Key {key_idx+1} rate limited, trying next...")
                    continue

                if "401" in err_str or "invalid" in err_str or "unauthorized" in err_str:
                    logger.warning(f"Key {key_idx+1} unauthorized, trying next...")
                    continue

                continue

        logger.error("All keys exhausted for STT")
        return ""

    except Exception as e:
        logger.error(f"STT transcription error: {e}")
        return ""