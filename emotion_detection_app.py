import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from transformers import pipeline
import pickle

# Load Sentiment Analysis Model
path_to_pickle_file = 'D:/New folder/Folder Kuliah Khansa Fairuz Zakiyah/KPKhansa/end2end-nlp-project-main/notebooks/emotiondetector.pickle'
with open(path_to_pickle_file, 'rb') as f:
    emotion_detector_model = pickle.load(f)

# Function to Detect Emotion
def detect_emotion(text):
    result = emotion_detector_model([text])[0]  # Ubah input teks menjadi list dengan satu elemen
    return result["label"], result["score"]

# Streamlit App
def main():
    st.title("Aplikasi Pendeteksi Emosi NLP")

    # Input Text
    text_input = st.text_area("Masukkan teks di sini:", "")

    if st.button("Deteksi Emosi"):
        if text_input:
            emotion, confidence = detect_emotion(text_input)
            st.write(f"Emosi: {emotion}")
            st.write(f"Kepercayaan: {confidence:.2f}")
            show_chart(emotion, confidence)
        else:
            st.warning("Mohon masukkan teks sebelum menekan tombol Deteksi Emosi.")

def show_chart(emotion, confidence):
    data = {'Emotion': [emotion], 'Confidence': [confidence]}
    df = pd.DataFrame(data)

    fig, ax = plt.subplots()
    ax.bar(df['Emotion'], df['Confidence'])
    ax.set_ylabel('Confidence')
    ax.set_title('Hasil Deteksi Emosi')

    st.pyplot(fig)

if __name__ == "__main__":
    main()
