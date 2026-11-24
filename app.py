import os
from flask import Flask, render_template, request, jsonify
from google import genai
from google.genai import types

app = Flask(__name__)

api_key = os.environ.get("GEMINI_API_KEY")
client = genai.Client(api_key=api_key) if api_key else None

# Use only the active Free-tier model (1 request per question to save quota)
ACTIVE_MODELS = ["gemini-3.8-flash"]


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
        response = client.models.generate_content(
            model=ACTIVE_MODELS[0],
            contents="Reply with the single word: Working",
            config=types.GenerateContentConfig(max_output_tokens=20),
        )
        return jsonify({
            "ok": True,
            "working_model": ACTIVE_MODELS[0],
            "reply": (response.text or "").strip(),
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

        response = client.models.generate_content(
            model=ACTIVE_MODELS[0],
            contents=prompt,
            config=types.GenerateContentConfig(
                max_output_tokens=450,
                temperature=0.4,
            ),
        )

        answer = (response.text or "").strip()
        if not answer:
            return jsonify({"error": "No answer received. Please try again."}), 500

        return jsonify({"answer": answer})

    except Exception as e:
        app.logger.exception("AI request failed")
        error_code = getattr(e, "code", None)
        if error_code == 429:
            return jsonify({
                "error": "Daily/minute free quota reached on Gemini API. Please wait 1 minute or use a fresh API key."
            }), 429
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=False)
