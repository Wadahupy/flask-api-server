import librosa
import numpy as np
from tensorflow.keras.models import load_model
import joblib
import os

# Load models and labels
text_model = joblib.load("text_emotion.pkl")
audio_model = load_model('lstm_model.keras')

text_labels = text_model.classes_
audio_labels = ['angry', 'disgust', 'happy', 'fear', 'neutral', 'ps', 'sad']

# Preprocess audio for prediction
def preprocess_audio(file_path):
    """
    Preprocess audio file for emotion detection.
    Extracts MFCC features and reshapes them for the model.
    """
    try:
        y, sr = librosa.load(file_path, duration=3, offset=0.5)
        mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
        return np.expand_dims(np.expand_dims(mfccs, axis=-1), axis=0)
    except Exception as e:
        return str(e)

# Text emotion prediction
def predict_text_emotion(text):
    """
    Predict emotion from text input.
    Returns the predicted emotion and confidence scores for all classes.
    """
    predicted_probabilities = text_model.predict_proba([text])[0]
    confidence = {emotion: float(prob) for emotion, prob in zip(text_labels, predicted_probabilities)}
    predicted_emotion = text_labels[np.argmax(predicted_probabilities)]
    return predicted_emotion, confidence

# Audio emotion prediction
def predict_audio_emotion(file_path):
    """
    Predict emotion from audio file.
    Returns the predicted emotion and confidence scores for all classes.
    """
    features = preprocess_audio(file_path)
    if isinstance(features, str):  # Error occurred
        return None, {"error": features}

    prediction = audio_model.predict(features)
    predicted_class = np.argmax(prediction)
    confidence = {label: float(prediction[0][i]) for i, label in enumerate(audio_labels)}
    predicted_emotion = audio_labels[predicted_class]

    return predicted_emotion, confidence

# Utility function to clean up temporary files
def cleanup_file(file_path):
    """Delete temporary files if they exist."""
    if os.path.exists(file_path):
        os.remove(file_path)
