import os
import io
import json
import base64
import urllib.request
import urllib.error
from gtts import gTTS

SARVAM_API_KEY = (os.environ.get("SARVAM_API_KEY") or "").strip()
SARVAM_TTS_URL = "https://api.sarvam.ai/text-to-speech"

BULBUL_MODEL = "bulbul:v4-flash"
TAMIL_SPEAKER = "gokul_ta_narration"
TAMIL_LANG_CODE = "ta-IN"


def _clean_text_for_tts(text: str, max_chars: int = 2000) -> str:
    lines = []
    for line in (text or "").splitlines():
        cleaned = line.lstrip("•*- \t").replace("*", "").replace("#", "").strip()
        if cleaned:
            lines.append(cleaned)
    return ". ".join(lines)[:max_chars].strip()


def _synthesize_sarvam_tamil_mp3(clean_text: str) -> str:
    """Calls Sarvam bulbul:v4-flash with ta-IN and gokul_ta_narration; returns base64 MP3."""
    if not SARVAM_API_KEY:
        raise RuntimeError("SARVAM_API_KEY is not set")

    payload = {
        "text": clean_text,
        "target_language_code": TAMIL_LANG_CODE,
        "model": BULBUL_MODEL,
        "speaker": TAMIL_SPEAKER,
        "output_audio_codec": "mp3",
        "speech_sample_rate": 22050,
        "pace": 0.95,
        "enable_preprocessing": True,
    }

    req = urllib.request.Request(
        SARVAM_TTS_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "api-subscription-key": SARVAM_API_KEY,
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as err:
        detail = err.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"Sarvam TTS HTTP {err.code}: {detail}") from err

    audios = body.get("audios") or []
    if not audios or not isinstance(audios[0], str) or not audios[0].strip():
        raise RuntimeError("Sarvam TTS returned empty audio payload")

    return audios[0].strip()


def _synthesize_gtts_mp3(clean_text: str, gtts_lang: str = "ta"):
    """Fallback MP3 synthesis via gTTS; returns base64 MP3 or None."""
    try:
        buf = io.BytesIO()
        tts = gTTS(text=clean_text, lang=gtts_lang, slow=False)
        tts.write_to_fp(buf)
        buf.seek(0)
        encoded = base64.b64encode(buf.read()).decode("ascii")
        return encoded if encoded else None
    except Exception:
        return None


def synthesize_speech_base64(text: str, language: str = "ta-IN"):
    """
    Generates base64-encoded MP3 audio.
    For Tamil (ta-IN): tries Sarvam bulbul:v4-flash with gokul_ta_narration first,
    then cleanly falls back to gTTS ('ta'), and returns None if both fail so
    browser speechSynthesis can handle playback on the client.
    """
    clean_text = _clean_text_for_tts(text)
    if not clean_text:
        return None

    lang = (language or "ta-IN").strip()
    is_tamil = lang.lower().startswith("ta")

    if is_tamil:
        try:
            return _synthesize_sarvam_tamil_mp3(clean_text)
        except Exception:
            # Clean fallback to gTTS if Sarvam is missing a key, rate-limited, or errors out
            return _synthesize_gtts_mp3(clean_text, gtts_lang="ta")

    fallback_lang = "hi" if lang.lower().startswith("hi") else "en"
    return _synthesize_gtts_mp3(clean_text, gtts_lang=fallback_lang)
