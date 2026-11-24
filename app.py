import os
import time
from flask import Flask, render_template, request, jsonify
from google import genai
from google.genai import types

app = Flask(__name__)

api_key = os.environ.get("GEMINI_API_KEY")
client = genai.Client(api_key=api_key) if api_key else None

# Current active models (in order of preference)
ACTIVE_MODELS = [
    "gemini-3.8-flash",
    "gemini-3.8-flash-lite",
    "gemini-3.0-flash",
    "gemini-2.0-flash",
]


def ask_gemini(prompt, max_tokens=450):
    """Tries active Gemini models with quick retries on temporary 503 demand spikes."""
    attempts_log = []

    for model_name in ACTIVE_MODELS:
        # Try each model up to 2 times if it returns a temporary 503 busy error
        for attempt in range(2):
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        max_output_tokens=max_tokens,
                        temperature=0.4,
                    ),
                )
                text = (response.text or "").strip()
                if text:
                    return text, model_name, attempts_log
            except Exception as err:
                code = getattr(err, "code", None)
                attempts_log.append(f"{model_name} (try {attempt + 1}): {code or err}")

                # If busy (503), wait 1.5s and retry once before moving to next model
                if code == 503 and attempt == 0:
                    time.sleep(1.5)
                    continue
                # If 404 (not available) or 429 (quota 0), move immediately to next model
                if code in (404, 429, 503):
                    break
                raise err

    raise RuntimeError("All models failed: " + " | ".join(attempts_log))


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/test")
def test_api():
    if not client:
        return jsonify({
            "ok": False,
            "error": "GEMINI_API_KEY is missing in Vercel Environment Variables."
        }), 500
    try:
        reply, used_model, log = ask_gemini(
            "Reply with the single word: Working",
            max_tokens=20
        )
        return jsonify({
            "ok": True,
            "working_model": used_model,
            "reply": reply,
            "attempts": log
        })
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/ask", methods=["POST"])
def ask():
    try:
        if not client:
            return jsonify({"error": "GEMINI_API_KEY is missing in Vercel settings."}), 500

        data = request.get_json(silent=True) or {}
        question = (data.get("question") or "").strip()
        language = data.get("language", "ta-IN")

        if not question:
            return jsonify({"error": "Please enter your farming question."}), 400

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

        answer, _, _ = ask_gemini(prompt, max_tokens=450)
        return jsonify({"answer": answer})

    except Exception as e:
        app.logger.exception("AI request failed")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=False)
