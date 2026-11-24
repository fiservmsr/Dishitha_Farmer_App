import os
import time
from flask import Flask, render_template, request, jsonify
from google import genai
from google.genai import types

app = Flask(__name__)

api_key = os.environ.get("GEMINI_API_KEY")
client = genai.Client(api_key=api_key) if api_key else None

ACTIVE_MODEL = "gemini-3.8-flash"


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
                time.sleep(2 * (attempt + 1))  # Wait 2s, then 4s
                continue
            raise err
    raise last_err or RuntimeError("Empty response from Gemini.")


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
        reply = call_gemini("Reply with the single word: Working", max_tokens=20)
        return jsonify({
            "ok": True,
            "working_model": ACTIVE_MODEL,
            "reply": reply,
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

        answer = call_gemini(prompt, max_tokens=450)
        return jsonify({"answer": answer})

    except Exception as e:
        app.logger.exception("AI request failed")
        error_code = getattr(e, "code", None)
        if error_code == 429:
            return jsonify({
                "error": "Free tier quota reached for this project. Please wait or switch to another key/project."
            }), 429
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=False)
