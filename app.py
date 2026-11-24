import os
from flask import Flask, render_template, request, jsonify
from google import genai
from google.genai import types

app = Flask(__name__)

api_key = os.environ.get("GEMINI_API_KEY")
client = genai.Client(api_key=api_key) if api_key else None

# Standard models that support Free Tier (ordered newest to oldest)
FREE_TIER_MODELS = [
    "gemini-3.8-flash-lite",
    "gemini-3.8-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.5-flash",
]


def generate_with_available_model(prompt, max_tokens=450):
    last_error = None
    for model_name in FREE_TIER_MODELS:
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
                return text, model_name
        except Exception as err:
            last_error = err
            # If this model is busy (503), quota=0 (429), or retired (404), try the next one
            if getattr(err, "code", None) in (404, 429, 503):
                continue
            raise err
    raise last_error or RuntimeError("No free-tier model responded.")


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
        reply, used_model = generate_with_available_model(
            "Reply with the single word: Working",
            max_tokens=20
        )
        return jsonify({
            "ok": True,
            "working_model": used_model,
            "reply": reply
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

        answer, _ = generate_with_available_model(prompt, max_tokens=450)
        return jsonify({"answer": answer})

    except Exception as e:
        app.logger.exception("AI request failed")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=False)
