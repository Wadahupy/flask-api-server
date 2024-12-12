import os
import librosa
import numpy as np
import pandas as pd
import pickle
from keras.models import load_model
from flask import Flask, request, render_template, jsonify
import joblib
import altair as alt
from werkzeug.utils import secure_filename

# Initialize Flask app
app = Flask(__name__)

# Emojis for emotions
emotions_emoji_dict = {
    "anger": "😠", "disgust": "🤮", "fear": "😨😱", "happy": "🤗", "joy": "😂",
    "neutral": "😐", "sad": "😔", "sadness": "😔", "shame": "😳", "surprise": "😮"
}

# Load models
pipe_lr = joblib.load("text_emotion.pkl")

def load_emotion_recognition_model(model_path, scaler_path):
    model = load_model(model_path)
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

# Feature extraction functions for audio
def extract_features(data, sample_rate):
    result = np.array([])

    zcr = np.mean(librosa.feature.zero_crossing_rate(y=data).T, axis=0)
    result = np.hstack((result, zcr))

    stft = np.abs(librosa.stft(data))
    chroma_stft = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
    result = np.hstack((result, chroma_stft))

    mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mfcc))

    rms = np.mean(librosa.feature.rms(y=data).T, axis=0)
    result = np.hstack((result, rms))

    mel = np.mean(librosa.feature.melspectrogram(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mel))

    return result

def get_features_from_audio(file_path, chunk_size=2.5):
    data, sample_rate = librosa.load(file_path)
    total_duration = librosa.get_duration(y=data, sr=sample_rate)

    if total_duration < chunk_size:
        return extract_features(data, sample_rate).reshape(1, -1)

    all_features = []
    for i in range(0, int(total_duration // chunk_size)):
        start = int(i * chunk_size * sample_rate)
        end = int((i + 1) * chunk_size * sample_rate)
        chunk = data[start:end]
        all_features.append(extract_features(chunk, sample_rate))

    return np.vstack(all_features) if all_features else np.array([]).reshape(0, 0)

def predict_emotions(docx):
    return pipe_lr.predict([docx])[0]

def predict_emotion_from_audio(file_path, model, scaler):
    emotion_mapping = {
        0: 'angry', 1: 'calm', 2: 'disgust', 3: 'fear',
        4: 'happy', 5: 'neutral', 6: 'sad', 7: 'surprise'
    }

    features = get_features_from_audio(file_path)
    scaled_features = scaler.transform(features)
    reshaped_features = np.expand_dims(scaled_features, axis=2)

    predictions = model.predict(reshaped_features)
    averaged_predictions = np.mean(predictions, axis=0)

    predicted_index = np.argmax(averaged_predictions)
    predicted_emotion = emotion_mapping[predicted_index]
    confidence = averaged_predictions[predicted_index]

    emotion_probabilities = {emotion_mapping[i]: prob for i, prob in enumerate(averaged_predictions)}

    return predicted_emotion, confidence, emotion_probabilities

# Flask Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict_text', methods=['POST'])
def predict_text():
    text_input = request.form['text_input']
    predicted_text_emotion = predict_emotions(text_input)
    prediction_proba = pipe_lr.predict_proba([text_input])[0]
    max_proba = np.max(prediction_proba)

    emoji = emotions_emoji_dict.get(predicted_text_emotion, "😐")
    return jsonify({
        'emotion': predicted_text_emotion,
        'emoji': emoji,
        'probability': max_proba
    })

@app.route('/predict_audio', methods=['POST'])
def predict_audio():
    if 'audio_file' not in request.files:
        return jsonify({'error': 'No audio file uploaded'})

    audio_file = request.files['audio_file']
    if audio_file.filename == '':
        return jsonify({'error': 'No selected file'})

    file_path = os.path.join('uploads', secure_filename(audio_file.filename))
    audio_file.save(file_path)

    model, scaler = load_emotion_recognition_model('emotion_recognition_model(1).h5', 'scaler(1).pkl')
    predicted_audio_emotion, confidence, emotion_probabilities = predict_emotion_from_audio(file_path, model, scaler)

    proba_df = pd.DataFrame(list(emotion_probabilities.items()), columns=["Emotion", "Probability"])

    # Plot the probabilities with Altair and save as an HTML file
    chart = alt.Chart(proba_df).mark_bar().encode(
        x=alt.X('Emotion', sort=None),
        y='Probability',
        color='Emotion'
    )
    chart.save(os.path.join('static', 'audio_chart.html'))

    return jsonify({
        'emotion': predicted_audio_emotion,
        'confidence': confidence,
        'probabilities_chart': 'static/audio_chart.html'
    })

if __name__ == '__main__':
    app.run(debug=True)
