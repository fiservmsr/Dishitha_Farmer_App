
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
            return jsonify({"error": "Gemini API key is not configured"}), 500

        data = request.get_json()
        question = data.get("question", "").strip()
        language = data.get("language", "ta-IN")

        if not question:
            return jsonify({"error": "Please enter a question"}), 400

        prompt = f"""
        You are AI Farmer Friend, an agricultural assistant
        for Indian farmers.

        Answer in the language requested: {language}.

        Give simple, practical and safe farming advice.
        Do not guess fertilizer or pesticide quantities.
        Ask for crop, location, growth stage and soil-test
        details when needed.
        Recommend local agricultural experts for serious
        crop disease or pesticide decisions.

        Farmer's question: {question}
        """

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )

        return jsonify({"answer": response.text})

    except Exception as e:
        app.logger.exception("AI request failed")
        return jsonify({"error": "Unable to get an AI answer. Please try again."}), 500

if __name__ == "__main__":
    app.run(debug=True)
