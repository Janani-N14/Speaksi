import os
import uuid
import random

# Load .env file if present
env_file = os.path.join(os.path.dirname(__file__), ".env")
if os.path.exists(env_file):
    with open(env_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, val = line.split("=", 1)
                os.environ[key.strip()] = val.strip().strip('"').strip("'")

from flask import Flask, request, jsonify, render_template, send_file
from flask_cors import CORS

from services.audio_processor import (
    convert_to_wav,
    generate_tts_audio,
    calculate_acoustic_similarity
)
from services.transcriber import transcribe_and_compare
from services.therapist import get_speech_therapy_tips

app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app)

TEMP_DIR = os.path.join(os.path.dirname(__file__), "temp_audio")
os.makedirs(TEMP_DIR, exist_ok=True)

@app.route('/', methods=['GET'])
def index():
    """Render the pronunciation testing interface."""
    return render_template('index.html')

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok", "message": "Pronunciation Checker Service is running."})

@app.route('/tts', methods=['GET', 'POST'])
def tts_preview():
    """
    Endpoint to stream reference pronunciation audio for a given word or phrase.
    """
    if request.method == 'POST':
        data = request.get_json(silent=True) or request.form
        text = data.get('text', '').strip()
    else:
        text = request.args.get('text', '').strip()

    if not text:
        return jsonify({"error": "No text provided"}), 400

    unique_id = str(uuid.uuid4())
    tts_file = os.path.join(TEMP_DIR, f"tts_preview_{unique_id}.wav")
    try:
        generate_tts_audio(text, tts_file)
        return send_file(tts_file, mimetype="audio/wav", as_attachment=False)
    except Exception as e:
        return jsonify({"error": f"TTS generation failed: {str(e)}"}), 500

@app.route('/check-pronunciation', methods=['POST'])
def check_pronunciation():
    """
    Pronunciation evaluation endpoint.
    Accepts:
        - text: Target phrase or word (form field)
        - audio / file: User audio recording file/blob (form file)
    Returns:
        JSON with similarity score, transcriptions, letter/word mismatches, AI therapy tips, and video guides.
    """
    target_text = request.form.get('text', '').strip()
    user_audio_file = request.files.get('audio') or request.files.get('file')

    if not target_text:
        return jsonify({"error": "Target 'text' parameter is required."}), 400

    if not user_audio_file or user_audio_file.filename == '':
        return jsonify({"error": "An 'audio' file must be provided."}), 400

    session_id = str(uuid.uuid4())
    temp_raw_path = os.path.join(TEMP_DIR, f"raw_{session_id}_{user_audio_file.filename}")
    temp_wav_path = os.path.join(TEMP_DIR, f"user_{session_id}.wav")
    temp_tts_path = os.path.join(TEMP_DIR, f"tts_{session_id}.wav")

    try:
        # 1. Save uploaded audio
        user_audio_file.save(temp_raw_path)

        # 2. Convert user audio to standard 16kHz PCM WAV
        convert_to_wav(temp_raw_path, temp_wav_path, sample_rate=16000)

        # 3. Generate reference TTS audio
        generate_tts_audio(target_text, temp_tts_path)

        # 4. Calculate Acoustic Similarity (MFCC)
        raw_similarity = calculate_acoustic_similarity(temp_tts_path, temp_wav_path)

        # 5. Speech-to-Text Transcription & Mismatch Analysis
        user_transcription, comparison = transcribe_and_compare(temp_wav_path, target_text)

        mismatched_letters = comparison["mismatched_letters"]
        mismatched_words = comparison["mismatched_words"]
        video_links = comparison["video_links"]

        # 6. Score Calibration
        has_mismatch = len(mismatched_letters) > 0 or (user_transcription != target_text.lower())
        
        if not has_mismatch:
            # Flawless match
            final_similarity = max(88.0, min(100.0, raw_similarity + random.uniform(2.0, 6.0)))
        elif has_mismatch and raw_similarity >= 85.0:
            final_similarity = max(60.0, min(85.0, raw_similarity - random.uniform(8.0, 15.0)))
        else:
            final_similarity = max(10.0, min(90.0, raw_similarity - 5.0))

        final_similarity = round(float(final_similarity), 1)

        # 7. Get AI Speech Therapist ("Speaky") Coaching Tips
        mismatched_items = mismatched_words if mismatched_words else mismatched_letters
        tips = get_speech_therapy_tips(target_text, user_transcription, mismatched_items)

        response_data = {
            "similarity": final_similarity,
            "target_text": target_text,
            "transcription": user_transcription,
            "mis_matchings": mismatched_letters,
            "mismatched_words": mismatched_words,
            "tips": tips,
            "videos": video_links
        }

        return jsonify(response_data), 200

    except Exception as e:
        print(f"[Error in /check-pronunciation]: {e}")
        return jsonify({"error": f"Pronunciation check failed: {str(e)}"}), 500

    finally:
        # Cleanup temporary files
        for path in [temp_raw_path, temp_wav_path, temp_tts_path]:
            if os.path.exists(path):
                try:
                    os.remove(path)
                except Exception:
                    pass

if __name__ == '__main__':
    port = int(os.getenv("PORT", 5000))
    print(f"Starting Pronunciation Checker server on http://localhost:{port}")
    app.run(host='0.0.0.0', port=port, debug=False)
