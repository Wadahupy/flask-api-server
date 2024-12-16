from flask import Blueprint, request, jsonify
from server.controllers.emotion_detection import predict_text_emotion, predict_audio_emotion, cleanup_file

prediction = Blueprint('predict', __name__)

@prediction.route('/text', methods=['POST'])
def text_prediction():

    text = request.json['text']

    predicted_emotion, confidence = predict_text_emotion(text)

    return jsonify({"type": "text", "predicted_emotion": predicted_emotion, "confidence": confidence})

@prediction.route('/audio', methods=['POST'])
def audio_prediction():

    file = request.files['file']

    # Validate file type
    if not file.mimetype.startswith('audio/'):
        return jsonify({"error": "Unsupported file type. Please upload an audio file."}), 400

    file_path = "temp_audio.wav"
    file.save(file_path)
    print(f"Saved file for debugging at: {file_path}")

    predicted_emotion, confidence = predict_audio_emotion(file_path)
    cleanup_file(file_path)

    if confidence.get("error"):
        return jsonify({"error": confidence["error"]}), 500

    return jsonify({"type": "audio", "predicted_emotion": predicted_emotion, "confidence": confidence})
