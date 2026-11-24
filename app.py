import os
import urllib.request
import json
from flask import Flask, render_template, request, jsonify
from google import genai
from google.genai import types

app = Flask(__name__)

GEMINI_KEY = (os.environ.get("GEMINI_API_KEY") or "").strip()
GROQ_KEY = (os.environ.get("GROQ_API_KEY") or "").strip()

gemini_client = genai.Client(api_key=GEMINI_KEY) if GEMINI_KEY else None
PRIMARY_MODEL = "gemini-3.8-flash"
GROQ_MODEL = "qwen/qwen3.8-27b"


def build_prompt(question: str, language: str) -> str:
    lang_label = "Tamil" if language.startswith("ta") else "English"
    return (
        f"You are AI Farmer Friend helping an Indian farmer with general fertilizer and crop care.\n"
        f"Reply ONLY in {lang_label} using 4 short bullet points.\n"
        f"Do not guess exact chemical pesticide/fertilizer dosages; recommend soil testing and local agriculture officer guidance.\n\n"
        f"Farmer input: {question}"
    )


def ask_gemini(prompt: str) -> str:
    if not gemini_client:
        raise RuntimeError("GEMINI_API_KEY not set")
    resp = gemini_client.models.generate_content(
        model=PRIMARY_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(max_output_tokens=350, temperature=0.3),
    )
    return (resp.text or "").strip()


def ask_groq(prompt: str) -> str:
    if not GROQ_KEY:
        raise RuntimeError("GROQ_API_KEY not set")
    payload = {
        "model": GROQ_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 350,
        "temperature": 0.3,
    }
    req = urllib.request.Request(
        "https://api.groq.com/openai/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {GROQ_KEY}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=15) as res:
        data = json.loads(res.read().decode("utf-8"))
    return (data["choices"][0]["message"]["content"] or "").strip()


def demo_fertilizer_reply(question: str, language: str) -> str:
    q_lower = question.lower()
    crop = "your crop"
    for name in ["paddy", "rice", "banana", "tomato", "brinjal", "cotton", "sugarcane", "maize", "groundnut"]:
        if name in q_lower:
            crop = name.title()
            break

    if language.startswith("ta"):
        return (
            f"• {crop} பயிருக்கான பொதுவான உர மற்றும் பராமரிப்பு வழிகாட்டுதல் (மாதிரி பதில்).\n"
            "• அடித்தள உரமாக நன்கு மக்கிய தொழுஉரம் (FYM) மற்றும் மண் பரிசோதனை அடிப்படையில் NPK உரங்களைப் பயன்படுத்தவும்.\n"
            "• இலை மஞ்சள் அல்லது பூச்சித் தாக்குதல் இருந்தால் சொந்தமாக அளவு கணிக்காமல் உள்ளூர் வேளாண் விரிவாக்க மையத்தை அணுகவும்.\n"
            "• மண்ணின் ஈரப்பதத்திற்கு ஏற்ப சீரான இடைவெளியில் நீர்ப்பாசனம் செய்யவும்."
        )

    return (
        f"• General fertilizer & crop guidance for {crop} (demo fallback):\n"
        "• Apply well-decomposed farmyard manure (FYM) and balance NPK based on a soil test report.\n"
        "• Avoid guessing exact chemical fertilizer or pesticide doses; check with your local agricultural officer for field-specific rates.\n"
        "• Maintain steady moisture appropriate to the crop growth stage and avoid waterlogging."
    )


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/test")
def test_route():
    fp = {
        "gemini_key_present": bool(GEMINI_KEY),
        "prefix": GEMINI_KEY[:6] if GEMINI_KEY else None,
        "suffix": GEMINI_KEY[-4:] if GEMINI_KEY else None,
        "groq_key_present": bool(GROQ_KEY),
    }
    try:
        out = ask_gemini("Reply with one word: OK")
        return jsonify({"ok": True, "provider": "gemini", "model": PRIMARY_MODEL, "reply": out, "key_check": fp})
    except Exception as e:
        return jsonify({"ok": False, "provider": "gemini", "error": str(e), "key_check": fp}), 503


@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json(silent=True) or {}
    question = (data.get("question") or "").strip()
    language = data.get("language", "ta-IN")

    if not question:
        return jsonify({"error": "Please enter a crop or fertilizer question."}), 400

    prompt = build_prompt(question, language)

    for fn in (ask_gemini, ask_groq):
        try:
            text = fn(prompt)
            if text:
                return jsonify({"answer": text})
        except Exception:
            continue

    return jsonify({"answer": demo_fertilizer_reply(question, language), "demo_mode": True})


if __name__ == "__main__":
    app.run(debug=False)
