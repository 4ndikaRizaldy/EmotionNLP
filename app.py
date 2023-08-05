# Core Pkgs
import streamlit as st 
import sqlite3
import altair as alt
import plotly.express as px 

# EDA Pkgs
import pandas as pd 
import numpy as np 
from datetime import datetime

from PIL import Image
st.set_page_config(page_icon="📉", 
                       page_title="Dashboard Aplikasi Kepuasan Pelanggan", 
                       layout = 'wide', 
                       initial_sidebar_state = 'auto')
gambar_lokal = 'foto/logo.png'
st.image(gambar_lokal)


# Utils
import joblib 
pipe_lr = joblib.load(open("models/emotiondetector.pickle","rb"))

conn = sqlite3.connect('data.db')

# Track Utils
from track_utils import create_page_visited_table,add_page_visited_details,view_all_page_visited_details,add_prediction_details,view_all_prediction_details,create_emotionclf_table


emotions_emoji_dict = {"kesal":"😠", "Senang":"🤗", "puas":"😂", "biasa saja":"😐", "sedih":"😔", "kecewa":"😔"}
def predict_emotions(docx):
    results = pipe_lr.predict([docx])
    prediction = results[0].capitalize()
    return prediction

def get_prediction_proba(docx):
    results = pipe_lr.predict_proba([docx])
    return results

def main():
    
    st.title("Aplikasi Kepuasan Pelanggan")
    menu = ["Home", "Monitor", "About"]
    choice = st.sidebar.selectbox("Menu", menu)
    create_page_visited_table()
    create_emotionclf_table()
    if choice == "Home":
        add_page_visited_details("Home", datetime.now())
        st.subheader("Home-Emotion In Text")

        with st.form(key='emotion_clf_form'):
            raw_text = st.text_area("Type Here")
            submit_text = st.form_submit_button(label='Submit')

            if raw_text.isdigit():
                st.error("Hasil tidak bisa di deteksi. Input harus berupa teks, bukan angka.")
            else:
                col1, col2 = st.columns(2)
                # Apply Fxn Here
                try:
                    prediction = predict_emotions(raw_text)
                    prediction = prediction.strip()
                    probability = get_prediction_proba(raw_text)
                    add_prediction_details(raw_text, prediction, np.max(probability), datetime.now())

                    with col1:
                        st.success("Original Text")
                        st.write(raw_text)

                        st.success("Prediction")
                        if prediction in emotions_emoji_dict:
                            emoji_icon = emotions_emoji_dict[prediction]
                            st.write("{}:{}".format(prediction, emoji_icon))
                        else:
                            st.write("Prediction: {}".format(prediction))
                        st.write("Confidence:{}".format(np.max(probability)))

                    with col2:
                        st.success("Prediction Probability")
                        # st.write(probability)
                        proba_df = pd.DataFrame(probability, columns=pipe_lr.classes_)
                        # st.write(proba_df.T)
                        proba_df_clean = proba_df.T.reset_index()
                        proba_df_clean.columns = ["emotions", "probability"]

                        fig = alt.Chart(proba_df_clean).mark_bar().encode(x='emotions', y='probability', color='emotions')
                        st.altair_chart(fig, use_container_width=True)
                except ValueError:
                    st.error("The input must be a string.")


    elif choice == "Monitor":
        add_page_visited_details("Monitor", datetime.now())
        st.subheader("Monitor App")

        with st.expander("Page Metrics"):
            page_visited_details = pd.DataFrame(view_all_page_visited_details(), columns=['Pagename', 'Time_of_Visit'])
            st.dataframe(page_visited_details)	

            pg_count = page_visited_details['Pagename'].value_counts().rename_axis('Pagename').reset_index(name='Counts')
            c = alt.Chart(pg_count).mark_bar().encode(x='Pagename', y='Counts', color='Pagename')
            st.altair_chart(c, use_container_width=True)	

            p = px.pie(pg_count, values='Counts', names='Pagename')
            st.plotly_chart(p, use_container_width=True)

        with st.expander('Emotion Classifier Metrics'):
            df_emotions = pd.DataFrame(view_all_prediction_details(), columns=['Rawtext', 'Prediction', 'Probability', 'Time_of_Visit'])
            df_emotions = df_emotions.apply(lambda col: col.astype(str, errors='ignore'))
            for entry in df_emotions.to_dict('records'):
                for key, value in entry.items():
                    if isinstance(value, str):
                        entry[key] = value.encode('utf-8', 'ignore').decode('utf-8')

            # Widget interaktif untuk memilih prediksi
            all_predictions = df_emotions['Prediction'].unique()
            selected_predictions = st.multiselect("Pilih Prediksi:", all_predictions, default=all_predictions)

            # Memfilter data berdasarkan prediksi yang dipilih
            filtered_data = df_emotions[df_emotions['Prediction'].isin(selected_predictions)]

            st.dataframe(filtered_data)

            prediction_count = filtered_data['Prediction'].value_counts().rename_axis('Prediction').reset_index(name='Counts')
            pc = alt.Chart(prediction_count).mark_bar().encode(x='Prediction', y='Counts', color='Prediction')
            st.altair_chart(pc, use_container_width=True)

    else:
        st.subheader("About")
        add_page_visited_details("About", datetime.now())

if __name__ == '__main__':
    
    main()
