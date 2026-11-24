import os
from flask import Flask, render_template, request, jsonify
from google import genai
from google.genai import types

app = Flask(__name__)

api_key = os.environ.get("GEMINI_API_KEY")
client = genai.Client(api_key=api_key) if api_key else None

# Primary model first, backup models if the primary is busy (503/429)
MODELS_TO_TRY = [
    "gemini-3.8-flash",
    "gemini-3.8-flash-lite",
    "gemini-3.0-flash",
]


@app.route("/")
def home():
    return render_template("index.html")


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

        last_error = None

        for model_name in MODELS_TO_TRY:
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        max_output_tokens=450,
                        temperature=0.4,
                    ),
                )
                answer = (response.text or "").strip()
                if answer:
                    return jsonify({"answer": answer})
            except Exception as model_err:
                last_error = model_err
                error_code = getattr(model_err, "code", None)
                # Retry with next model if busy (503), rate-limited (429), or unavailable (404)
                if error_code in (503, 429, 404):
                    continue
                raise model_err

        return jsonify({
            "error": "AI service is currently busy across all models. Please wait 15 seconds and try again."
        }), 503

    except Exception as e:
        app.logger.exception("AI request failed")
        return jsonify({"error": f"AI Error: {str(e)}"}), 500
