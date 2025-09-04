import streamlit as st
import tensorflow as tf
from transformers import AutoTokenizer
from huggingface_hub import hf_hub_download
import zipfile
import os

st.set_page_config(page_title="Analisis Sentimen Sosial Media", page_icon="💬")
st.title("💬 Analisis Sentimen Komentar Sosial Media")

# --- Download model & tokenizer dari HuggingFace Hub ---
REPO_ID = "zahralitha/sentimenteks"

# Model h5
MODEL_FILE = hf_hub_download(repo_id=REPO_ID, filename="sentiment_model.h5")

# Tokenizer (diunggah dalam bentuk zip, jadi perlu ekstrak)
TOKENIZER_ZIP = hf_hub_download(repo_id=REPO_ID, filename="tokenizer.zip")
if not os.path.exists("tokenizer"):
    with zipfile.ZipFile(TOKENIZER_ZIP, "r") as zip_ref:
        zip_ref.extractall("tokenizer")

# --- Load model dan tokenizer ---
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(MODEL_FILE)
    tokenizer = AutoTokenizer.from_pretrained("tokenizer")
    return model, tokenizer

model, tokenizer = load_model()

# --- Label mapping (ganti sesuai dataset kamu) ---
id2label = {
    0: "SADNESS",
    1: "ANGER",
    2: "SUPPORT",
    3: "HOPE",
    4: "DISAPPOINTMENT"
}

# --- Fungsi prediksi ---
def predict(text):
    enc = tokenizer(
        [text],
        truncation=True,
        padding="max_length",
        max_length=128,
        return_tensors="np"
    )
    preds = model.predict([enc["input_ids"], enc["attention_mask"]], verbose=0)
    label_id = preds.argmax(axis=1)[0]
    return id2label[label_id], float(preds.max())

# --- UI Streamlit ---
st.subheader("Masukkan komentar:")
user_input = st.text_area("Tulis komentar di sini...")

if st.button("🔍 Analisis"):
    if user_input.strip():
        label, score = predict(user_input)
        st.success(f"Prediksi: **{label}** (confidence: {score:.2f})")
    else:
        st.warning("Masukkan teks terlebih dahulu.")
