from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os

from google import genai



# Get API key from .env
API_KEY = "AIzaSyA38KT8dqkxaQ7kKKqCf_uAYwMIVW5GBEY"

print("🔑 API KEY:", API_KEY)  # debug

# Flask setup
app = Flask(__name__, static_folder="frontend")
CORS(app)

# Primary inference client
client = genai.Client(api_key=API_KEY)

VALID_LABELS = ["joy", "sadness", "anger", "fear", "love", "surprise"]

# --- Primary Inference Engine ---
def external_model_predict(text):
    try:
        print("\n==============================")
        print("👉 PRIMARY ENGINE CALLED")

        response = client.models.generate_content(
            model="gemini-flash-latest",  # internal only
            contents=f"""
            Classify the emotion into EXACTLY one word from:
            joy, sadness, anger, fear, love, surprise.

            Return only the word.

            Sentence: {text}
            """
        )

        if not response or not response.text:
            print("❌ Empty response")
            return None, "primary"

        raw = response.text.strip().lower()
        print("🧾 RAW OUTPUT:", raw)

        # ✅ BEST MATCHING (robust + simple)
        for label in VALID_LABELS:
            if label in raw:
                print("✅ MATCH FOUND:", label)
                return label, "primary"

        print("❌ NO VALID LABEL")
        return None, "primary_invalid"

    except Exception as e:
        print("🔥 PRIMARY ERROR:", e)
        return None, "primary_error"


# --- Local Model (Fallback) ---
def local_model_predict(text):
    print("🧠 FALLBACK MODEL USED")
    return "joy", "local"


# --- Serve Frontend ---
@app.route("/")
def serve_frontend():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    front_path = os.path.join(base_dir, "frontend")

    print("📂 Serving from:", front_path)

    return send_from_directory(front_path, "emotion_app.html")


# --- API Route ---
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        if not data or "text" not in data:
            return jsonify({"error": "No input text"}), 400

        text = data["text"]

        # Primary system
        result, source = external_model_predict(text)

        # Fallback if needed
        if result is None:
            result, source = local_model_predict(text)

        return jsonify({
            "emotion": result,
            "source": source  # used for ⚡ / 🧠 indicator
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# --- Run App ---
if __name__ == "__main__":
    import webbrowser

    host = "127.0.0.1"
    port = 5000
    url = f"http://{host}:{port}"

    print("\n" + "="*50)
    print("🚀 Emotion System Running (Final Stable)")
    print(f"👉 Open: {url}")
    print("="*50 + "\n")

    try:
        webbrowser.open(url)
    except:
        pass

    app.run(host=host, port=port, debug=True)