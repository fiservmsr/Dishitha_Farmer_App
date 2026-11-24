import os
from flask import Flask, render_template, request, jsonify
from google import genai

app = Flask(__name__)

api_key = os.environ.get("GEMINI_API_KEY")
client = genai.Client(api_key=api_key) if api_key else None


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/ask", methods=["POST"])
def ask():
    try:
        if not client:
            return jsonify({
                "error": "Gemini API key is not configured. Please check your environment variables."
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

        prompt = f"""
You are AI Farmer Friend, a knowledgeable agricultural
assistant helping Indian farmers.

IMPORTANT LANGUAGE RULE:
Answer ONLY in {language_name}.

Your responsibilities:
1. Crop selection and cultivation advice.
2. Fertilizer and nutrient management.
3. Crop disease identification guidance.
4. Pest management and prevention.
5. Irrigation and water management.
6. Harvesting and post-harvest advice.
7. Organic and sustainable farming.
8. Weather-related farming precautions.

Give simple, practical and safe advice.

IMPORTANT SAFETY:
- Never invent fertilizer or pesticide dosages.
- Ask for crop, location, growth stage and soil-test
  details when required.
- Explain safe handling of agricultural chemicals.
- Recommend local agricultural experts for serious
  crop diseases or pesticide decisions.
- Clearly mention when you are uncertain.

Farmer's question:
{question}
"""

        response = client.models.generate_content(
            model="gemini-3.8-flash",
            contents=prompt,
        )

        answer = (response.text or "").strip()
        if not answer:
            return jsonify({
                "error": "No answer received. Please try again."
            }), 500

        return jsonify({"answer": answer})

    except Exception as e:
        app.logger.exception("AI request failed")
        error_code = getattr(e, "code", None)

        if error_code == 503:
            return jsonify({
                "error": "Google AI is currently busy. Please wait 30 seconds and try again."
            }), 503
        if error_code == 429:
            return jsonify({
                "error": "AI request limit reached. Please try again later."
            }), 429
        if error_code == 404:
            return jsonify({
                "error": "Requested Gemini model was not found. Check model ID or SDK version."
            }), 404

        return jsonify({
            "error": "AI service is temporarily unavailable. Please try again."
        }), 500


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000)
