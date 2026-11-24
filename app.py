import os
import time
from flask import Flask, render_template, request, jsonify
from google import genai
from google.genai import types

app = Flask(__name__)

raw_key = os.environ.get("GEMINI_API_KEY")
clean_key = (raw_key or "").strip()
client = genai.Client(api_key=clean_key) if clean_key else None

ACTIVE_MODEL = "gemini-3.8-flash"

FALLBACK_ANSWERS = {
    "ta-IN": (
        "• தற்காலிகமாக AI சேவையில் அதிக நெரிசல் உள்ளது (503/429).\n"
        "• மண்ணின் ஈரப்பதம் மற்றும் பயிர் நிலையைப் பொறுத்து நீர்ப்பாசனத்தைத் திட்டமிடுங்கள்.\n"
        "• உரம் அல்லது பூச்சிக்கொல்லி அளவுகளை அனுமானிக்காமல் உள்ளூர் வேளாண் அலுவலரிடம் ஆலோசிக்கவும்.\n"
        "• ஒரு நிமிடம் கழித்து மீண்டும் கேள்வியை முயற்சிக்கவும்."
    ),
    "en-IN": (
        "• Gemini's free-tier capacity is temporarily busy or rate-limited right now.\n"
        "• Check soil moisture and crop stage before scheduling irrigation or fieldwork.\n"
        "• Do not guess fertilizer or pesticide dosages; consult your local agricultural officer for serious crop issues.\n"
        "• Wait a minute and try your question again."
    ),
}


def call_gemini(prompt, max_tokens=450):
    """Calls gemini-3.8-flash with brief retries only on temporary 503 spikes."""
    last_err = None
    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model=ACTIVE_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    max_output_tokens=max_tokens,
                    temperature=0.4,
                ),
            )
            text = (response.text or "").strip()
            if text:
                return text
        except Exception as err:
            last_err = err
            if getattr(err, "code", None) == 503 and attempt < 2:
                time.sleep(2 * (attempt + 1))
                continue
            raise err
    raise last_err or RuntimeError("Empty response from Gemini.")


def get_key_fingerprint():
    if not clean_key:
        return None
    return {
        "length": len(clean_key),
        "prefix": clean_key[:6],
        "suffix": clean_key[-4:],
        "has_whitespace_issue": clean_key != raw_key,
    }


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/test")
def test_api():
    fingerprint = get_key_fingerprint()
    if not client:
        return jsonify({
            "ok": False,
            "key_fingerprint": fingerprint,
            "error": "GEMINI_API_KEY is missing or empty in Vercel Environment Variables."
        }), 500
    try:
        reply = call_gemini("Reply with the single word: Working", max_tokens=20)
        return jsonify({
            "ok": True,
            "key_fingerprint": fingerprint,
            "model": ACTIVE_MODEL,
            "reply": reply,
        })
    except Exception as e:
        return jsonify({
            "ok": False,
            "key_fingerprint": fingerprint,
            "error": str(e),
        }), 500


@app.route("/ask", methods=["POST"])
def ask():
    language = "ta-IN"
    try:
        data = request.get_json(silent=True) or {}
        question = (data.get("question") or "").strip()
        language = data.get("language", "ta-IN")

        if not question:
            return jsonify({"error": "Please enter your farming question."}), 400

        if not client:
            return jsonify({"error": "GEMINI_API_KEY is missing in Vercel settings."}), 500

        language_name = {
            "ta-IN": "Tamil",
            "en-IN": "English",
            "hi-IN": "Hindi",
            "te-IN": "Telugu",
            "ml-IN": "Malayalam",
            "kn-IN": "Kannada",
        }.get(language, "Tamil")

        prompt = f"""You are AI Farmer Friend helping Indian farmers.
Answer ONLY in {language_name}.
Give a short, practical answer in 4 to 5 brief bullet points.
Never invent fertilizer or pesticide dosages; advise consulting local agricultural officers for serious diseases.

Farmer's question: {question}"""

        answer = call_gemini(prompt, max_tokens=450)
        return jsonify({"answer": answer})

    except Exception as e:
        app.logger.exception("AI request failed")
        error_code = getattr(e, "code", None)
        if error_code in (429, 503) or "429" in str(e) or "503" in str(e):
            fallback = FALLBACK_ANSWERS.get(language, FALLBACK_ANSWERS["en-IN"])
            return jsonify({
                "answer": fallback,
                "fallback": True,
                "notice": "Gemini free tier is busy/rate-limited; showing a safe backup tip."
            })
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=False)
