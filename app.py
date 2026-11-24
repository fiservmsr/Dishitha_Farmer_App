import os
from flask import Flask, render_template, request, jsonify
from google import genai
from google.genai import types

app = Flask(__name__)

api_key = os.environ.get("GEMINI_API_KEY")
client = genai.Client(api_key=api_key) if api_key else None


def get_best_flash_model():
    """Finds the best available Gemini Flash model enabled on your API key."""
    if not client:
        return "gemini-2.5-flash"
    try:
        available = [
            m.name.replace("models/", "")
            for m in client.models.list()
            if "gemini" in m.name.lower() and "flash" in m.name.lower()
        ]
        # Prefer standard flash models over experimental/preview ones if available
        standard = [m for m in available if "exp" not in m and "thinking" not in m]
        if standard:
            return sorted(standard)[-1]
        if available:
            return sorted(available)[-1]
    except Exception:
        pass
    return "gemini-2.5-flash"


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/test")
def test_api():
    """Open /test in your browser to verify your API key and active Gemini models."""
    if not client:
        return jsonify({
            "ok": False,
            "error": "GEMINI_API_KEY is missing in Vercel Environment Variables."
        }), 500

    try:
        models = [
            m.name for m in client.models.list()
            if "gemini" in m.name.lower()
        ]
        chosen_model = get_best_flash_model()

        response = client.models.generate_content(
            model=chosen_model,
            contents="Reply with the single word: Working"
        )

        return jsonify({
            "ok": True,
            "working_model": chosen_model,
            "reply": (response.text or "").strip(),
            "available_gemini_models": models
        })

    except Exception as e:
        return jsonify({
            "ok": False,
            "error": str(e)
        }), 500


@app.route("/ask", methods=["POST"])
def ask():
    try:
        if not client:
            return jsonify({
                "error": "GEMINI_API_KEY is missing in Vercel settings."
            }), 500

        data = request.get_json(silent=True) or {}
        question = (data.get("question") or "").strip()
        language = data.get("language", "ta-IN")

        if not question:
            return jsonify({
                "error": "Please enter your farming question."
            }), 400

        language_name = {
            "ta-IN": "Tamil",
            "en-IN": "English",
            "hi-IN": "Hindi",
            "te-IN": "Telugu",
            "ml-IN": "Malayalam",
            "kn-IN": "Kannada",
        }.get(language, "Tamil")

        model_to_use = get_best_flash_model()

        prompt = f"""You are AI Farmer Friend helping Indian farmers.
Answer ONLY in {language_name}.
Give a short, practical answer in 4 to 5 brief bullet points.
Never invent fertilizer or pesticide dosages; advise consulting local agricultural officers for serious diseases.

Farmer's question: {question}"""

        response = client.models.generate_content(
            model=model_to_use,
            contents=prompt,
            config=types.GenerateContentConfig(
                max_output_tokens=450,
                temperature=0.4,
            ),
        )

        answer = (response.text or "").strip()
        if not answer:
            return jsonify({
                "error": "No answer received from Gemini. Please try again."
            }), 500

        return jsonify({"answer": answer})

    except Exception as e:
        app.logger.exception("AI request failed")
        return jsonify({
            "error": str(e)
        }), 500


if __name__ == "__main__":
    app.run(debug=False)
