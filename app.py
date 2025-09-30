# streamlit_app.py
import streamlit as st
import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model
import io
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from collections import Counter
import soundfile as sf  # pip install soundfile

# Configure page
st.set_page_config(
    page_title="Speech Emotion Recognition",
    page_icon="🎤",
    layout="wide"
)

@st.cache_resource
def load_models_and_artifacts():
    try:
        rf_model = joblib.load('random_forest_model.pkl')
        lr_model = joblib.load('logistic_regression_model.pkl')
        cnn_model = load_model('cnn_emotion_model.h5')
        label_encoder = joblib.load('label_encoder.pkl')
        scaler = joblib.load('scaler.pkl')
        return rf_model, lr_model, cnn_model, label_encoder, scaler
    except FileNotFoundError as e:
        st.error(f"Model files not found. Please train the models first. Error: {e}")
        return None, None, None, None, None
    except Exception as e:
        st.error(f"Unexpected error loading artifacts: {e}")
        return None, None, None, None, None

class AudioProcessor:
    def __init__(self, target_sr=16000, duration=3.0):
        self.target_sr = target_sr
        self.duration = duration
        self.target_length = int(target_sr * duration)
    
    def preprocess_audio(self, audio_data, sr):
        if audio_data.ndim > 1:
            audio_data = librosa.to_mono(audio_data.T)
        if sr != self.target_sr:
            audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=self.target_sr)
            sr = self.target_sr
        audio_data, _ = librosa.effects.trim(audio_data, top_db=20)
        audio_data = librosa.util.normalize(audio_data)
        if len(audio_data) > self.target_length:
            audio_data = audio_data[:self.target_length]
        else:
            audio_data = np.pad(audio_data, (0, self.target_length - len(audio_data)), mode='constant')
        return audio_data, sr

    def extract_features(self, audio):
        mfcc = np.mean(librosa.feature.mfcc(y=audio, sr=self.target_sr, n_mfcc=13).T, axis=0)
        chroma = np.mean(librosa.feature.chroma_stft(y=audio, sr=self.target_sr).T, axis=0)
        spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=self.target_sr)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=self.target_sr)[0]
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=self.target_sr)[0]
        zero_crossing_rate = librosa.feature.zero_crossing_rate(y=audio)[0]
        spectral_features = np.array([
            np.mean(spectral_centroids), np.std(spectral_centroids),
            np.mean(spectral_rolloff), np.std(spectral_rolloff),
            np.mean(spectral_bandwidth), np.std(spectral_bandwidth),
            np.mean(zero_crossing_rate), np.std(zero_crossing_rate)
        ])
        features = np.concatenate([mfcc, chroma, spectral_features])
        mel_spec = librosa.power_to_db(librosa.feature.melspectrogram(y=audio, sr=self.target_sr, n_mels=128), ref=np.max)
        return features, mel_spec

def create_audio_visualizations(audio_data, sr, mel_spec_db):
    fig = make_subplots(rows=3, cols=1, subplot_titles=('Waveform', 'Mel-Spectrogram', 'Feature Distribution'), 
                        specs=[[{"secondary_y": False}],
                               [{"secondary_y": False}],
                               [{"secondary_y": False}]])
    time = np.linspace(0, len(audio_data)/sr, len(audio_data))
    fig.add_trace(go.Scatter(x=time, y=audio_data, mode='lines', name='Waveform'), row=1, col=1)
    fig.add_trace(go.Heatmap(z=mel_spec_db, name='Mel-Spectrogram'), row=2, col=1)
    mel_mean = np.mean(mel_spec_db, axis=1)
    top_n = 20
    x = list(range(min(top_n, len(mel_mean))))
    y = mel_mean[:top_n]
    fig.add_trace(go.Bar(x=x, y=y, name='Mel mean (first bins)'), row=3, col=1)
    fig.update_layout(height=900, showlegend=False)
    fig.update_xaxes(title_text="Time (s)", row=1, col=1)
    fig.update_yaxes(title_text="Amplitude", row=1, col=1)
    fig.update_xaxes(title_text="Time Frames", row=2, col=1)
    fig.update_yaxes(title_text="Mel Frequency Bins", row=2, col=1)
    return fig

def plot_probability_bars(probabilities):
    prob_items = list(probabilities.items())
    prob_items.sort(key=lambda x: x[1], reverse=True)
    emotions = [k.replace('_', ' ').title() for k, v in prob_items]
    probs = [v for k, v in prob_items]
    fig = go.Figure(go.Bar(
        x=probs,
        y=emotions,
        orientation='h',
        text=[f"{p:.1%}" for p in probs],
        textposition='outside'
    ))
    fig.update_layout(xaxis_tickformat='.0%')
    return fig

def predict_emotion(audio_data, sr, models, processor, scaler, label_encoder):
    processed_audio, new_sr = processor.preprocess_audio(audio_data, sr)
    features, mel_spec = processor.extract_features(processed_audio)
    rf_model, lr_model, cnn_model = models
    features_scaled = scaler.transform(features.reshape(1, -1))
    mel_spec_reshaped = mel_spec.reshape(1, mel_spec.shape[0], mel_spec.shape[1], 1)
    predictions = {}

    rf_pred = rf_model.predict(features_scaled)[0]
    rf_proba = rf_model.predict_proba(features_scaled)[0]
    predictions['Random Forest'] = {
        'prediction': label_encoder.inverse_transform([int(rf_pred)])[0],
        'confidence': float(np.max(rf_proba)),
        'probabilities': dict(zip(label_encoder.classes_, rf_proba))
    }

    lr_pred = lr_model.predict(features_scaled)[0]
    lr_proba = lr_model.predict_proba(features_scaled)[0]
    predictions['Logistic Regression'] = {
        'prediction': label_encoder.inverse_transform([int(lr_pred)])[0],
        'confidence': float(np.max(lr_proba)),
        'probabilities': dict(zip(label_encoder.classes_, lr_proba))
    }

    cnn_proba = cnn_model.predict(mel_spec_reshaped)[0]
    cnn_pred = int(np.argmax(cnn_proba))
    predictions['CNN'] = {
        'prediction': label_encoder.inverse_transform([cnn_pred])[0],
        'confidence': float(np.max(cnn_proba)),
        'probabilities': dict(zip(label_encoder.classes_, cnn_proba))
    }

    return predictions, processed_audio, mel_spec, new_sr

def main():
    st.title("🎤 Speech Emotion Recognition")
    st.markdown("""
    Upload an audio file to predict the emotion using multiple machine learning models.
    The system uses MFCC, Chroma, and Spectral features with Random Forest, Logistic Regression, and CNN models.
    """)

    with st.spinner("Loading models..."):
        rf_model, lr_model, cnn_model, label_encoder, scaler = load_models_and_artifacts()
    if rf_model is None:
        st.error("Please ensure model files exist in the app folder.")
        return

    models = (rf_model, lr_model, cnn_model)
    processor = AudioProcessor()

    # File upload section
    st.header("📁 Upload Audio File")
    uploaded_file = st.file_uploader("Choose an audio file", type=['wav', 'mp3', 'ogg', 'flac'])
    st.info("Audio recording feature is removed. Please upload an audio file for analysis.")

    if uploaded_file is not None:
        uploaded_file.seek(0)
        data, sr = sf.read(uploaded_file)
        uploaded_file.seek(0)
        st.audio(uploaded_file)

        # Audio info
        st.subheader("📊 Audio Information")
        col1, col2, col3 = st.columns(3)
        duration = data.shape[0] / sr
        with col1: st.metric("Duration", f"{duration:.2f}s")
        with col2: st.metric("Sample Rate", f"{sr} Hz")
        with col3: st.metric("Samples", data.shape[0])

        with st.spinner("Analyzing audio and predicting emotion..."):
            predictions, processed_audio, mel_spec, proc_sr = predict_emotion(data, sr, models, processor, scaler, label_encoder)

            # Display predictions
            st.header("🎯 Emotion Predictions")
            col1, col2, col3 = st.columns(3)
            columns = [col1, col2, col3]
            model_names = ['Random Forest', 'Logistic Regression', 'CNN']
            model_icons = ['🌳', '📊', '🧠']
            emotion_emojis = {
                'sad': '😢💔', 'pleasant_surprise': '😲✨', 'neutral': '😐',
                'happy': '😊🌟', 'fear': '😰⚡', 'disgust': '🤢💀', 'angry': '😡🔥'
            }

            for i, (model_name, icon) in enumerate(zip(model_names, model_icons)):
                if model_name in predictions:
                    pred_data = predictions[model_name]
                    with columns[i]:
                        st.subheader(f"{icon} {model_name}")
                        emotion = pred_data.get('prediction', 'unknown')
                        confidence = pred_data.get('confidence', 0.0)
                        symbol = emotion_emojis.get(emotion, '❓')
                        st.markdown(f"### {symbol} {emotion.replace('_', ' ').title()}")
                        st.markdown(f"**Confidence:** {confidence:.2%}")
                        probabilities = pred_data.get('probabilities', {})
                        if probabilities:
                            st.plotly_chart(plot_probability_bars(probabilities), use_container_width=True)

            # Consensus prediction
            st.header("🤝 Model Consensus")
            all_predictions = [pred['prediction'] for pred in predictions.values() if 'prediction' in pred]
            all_confidences = [pred['confidence'] for pred in predictions.values() if 'confidence' in pred]
            if all_predictions:
                prediction_counts = Counter(all_predictions)
                consensus_emotion = prediction_counts.most_common(1)[0][0]
                consensus_count = prediction_counts.most_common(1)[0][1]
                consensus_confidences = [pred['confidence'] for pred in predictions.values() 
                                         if pred.get('prediction') == consensus_emotion]
                avg_confidence = float(np.mean(consensus_confidences)) if consensus_confidences else 0.0
                emoji = emotion_emojis.get(consensus_emotion, '❓')
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Consensus Emotion", f"{emoji} {consensus_emotion.replace('_', ' ').title()}",
                              f"{consensus_count}/{len(predictions)} models agree")
                with col2:
                    st.metric("Average Confidence", f"{avg_confidence:.2%}",
                              f"Range: {min(all_confidences):.2%} - {max(all_confidences):.2%}" if all_confidences else "")
            else:
                st.warning("No valid model predictions were produced.")

            # Visualizations
            st.header("📈 Audio Analysis Visualizations")
            st.plotly_chart(create_audio_visualizations(processed_audio, proc_sr, mel_spec), use_container_width=True)

            # Model comparison
            st.subheader("🔍 Model Comparison")
            comparison_df = pd.DataFrame([{
                'Model': k,
                'Predicted Emotion': v['prediction'].replace('_', ' ').title(),
                'Confidence': v['confidence']
            } for k, v in predictions.items()])
            fig_comparison = px.bar(comparison_df, x='Model', y='Confidence', color='Predicted Emotion', 
                                    text='Confidence', title='Model Predictions Comparison')
            fig_comparison.update_traces(texttemplate='%{text:.1%}', textposition='outside')
            fig_comparison.update_layout(yaxis_tickformat='.0%')
            st.plotly_chart(fig_comparison, use_container_width=True)

            # Download predictions
            results_df = pd.DataFrame([{
                'File': uploaded_file.name,
                'Duration_seconds': duration,
                'Sample_rate': sr,
                'Consensus_emotion': consensus_emotion,
                'Consensus_confidence': avg_confidence
            }])
            for model_name, pred_data in predictions.items():
                model_key = model_name.replace(' ', '_').lower()
                results_df[f'{model_key}_prediction'] = pred_data.get('prediction', '')
                results_df[f'{model_key}_confidence'] = pred_data.get('confidence', 0.0)

            st.download_button(label="📥 Download Prediction Results (CSV)", data=results_df.to_csv(index=False),
                               file_name=f"emotion_prediction_{uploaded_file.name}.csv", mime="text/csv")

if __name__ == "__main__":
    main()
