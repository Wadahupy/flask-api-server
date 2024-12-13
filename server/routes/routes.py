from flask import Blueprint, request, jsonify
from server.controllers.emotion_detection import predict_text_emotion, predict_audio_emotion, cleanup_file

prediction = Blueprint('predict', __name__)

@prediction.route('/predict', methods=['POST'])
def predict_emotion():
    if 'text' in request.json:
        text = request.json['text']
        predicted_emotion, confidence = predict_text_emotion(text)
        return jsonify({"type": "text", "predicted_emotion": predicted_emotion, "confidence": confidence})

    if 'file' in request.files:
        file = request.files['file']
        file_path = "temp_audio.wav"
        file.save(file_path)

        predicted_emotion, confidence = predict_audio_emotion(file_path)
        cleanup_file(file_path)

        if confidence.get("error"):
            return jsonify({"error": confidence["error"]}), 500

        return jsonify({"type": "audio", "predicted_emotion": predicted_emotion, "confidence": confidence})

    return jsonify({"error": "Invalid request. Provide either 'text' or 'file'."}), 400
