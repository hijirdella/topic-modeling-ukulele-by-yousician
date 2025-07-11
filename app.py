import streamlit as st
import pandas as pd
import re
import joblib
import gensim
from gensim.corpora import Dictionary
from gensim.models import LdaModel
from datetime import datetime
import pytz

# === Load Model dan Dictionary ===
lda_model = LdaModel.load("best_lda_ukelele_by_yousician.gensim")
dictionary = Dictionary.load("best_lda_ukelele_by_yousician.gensim.id2word")

# === Preprocessing ===
def clean_text(text):
    text = text.lower()
    text = re.sub(r'<.*?>', ' ', text)
    text = re.sub(r'http\S+|www.\S+', ' ', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def tokenize(text):
    return clean_text(text).split()

def get_dominant_topic(text):
    bow = dictionary.doc2bow(tokenize(text))
    topics = lda_model.get_document_topics(bow)
    if topics:
        dominant_topic, prob = max(topics, key=lambda x: x[1])
        return dominant_topic, round(prob, 3)
    else:
        return None, 0.0

# === Streamlit UI ===
st.set_page_config(page_title="Topic Modeling - Ukulele by Yousician", layout="wide")
st.title("🧠 Topic Modeling - Ukulele by Yousician")

input_mode = st.radio("Pilih Mode Input:", ["📝 Input Manual", "📁 Upload CSV"])

# === Input Manual ===
if input_mode == "📝 Input Manual":
    name = st.text_input("Nama Pengguna:")
    review = st.text_area("Masukkan Review:")
    
    if st.button("Deteksi Topik"):
        if review.strip() == "":
            st.warning("Review tidak boleh kosong.")
        else:
            topic, prob = get_dominant_topic(review)
            now = datetime.now(pytz.timezone("Asia/Jakarta"))
            df = pd.DataFrame([{
                "Name": name if name else "(Anonim)",
                "Review": review,
                "Predicted Topic": topic,
                "Probability": prob,
                "Datetime (WIB)": now.strftime("%Y-%m-%d %H:%M")
            }])
            st.success("✅ Topik berhasil diprediksi!")
            st.dataframe(df)

            # Tombol download
            st.download_button(
                label="📥 Download Hasil",
                data=df.to_csv(index=False),
                file_name="topic_prediction_manual.csv",
                mime="text/csv"
            )

# === Input Batch CSV ===
else:
    uploaded_file = st.file_uploader("Upload file CSV dengan kolom 'review':", type="csv")
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            if 'review' not in df.columns:
                st.error("❌ Kolom 'review' tidak ditemukan.")
            else:
                df['cleaned_review'] = df['review'].fillna("").apply(clean_text)
                df['tokens'] = df['cleaned_review'].apply(lambda x: x.split())
                df['bow'] = df['tokens'].apply(lambda x: dictionary.doc2bow(x))
                df['Predicted Topic'] = df['bow'].apply(lambda x: max(lda_model.get_document_topics(x), key=lambda t: t[1])[0] if x else None)
                df['Probability'] = df['bow'].apply(lambda x: round(max(lda_model.get_document_topics(x), key=lambda t: t[1])[1], 3) if x else 0.0)

                st.success("✅ Topik berhasil diprediksi!")
                st.dataframe(df[['review', 'Predicted Topic', 'Probability']])

                st.download_button(
                    label="📥 Download Hasil Prediksi",
                    data=df.to_csv(index=False),
                    file_name="topic_prediction_batch.csv",
                    mime="text/csv"
                )
        except Exception as e:
            st.error(f"❌ Terjadi kesalahan: {e}")
