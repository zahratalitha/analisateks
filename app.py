import streamlit as st
import tensorflow as tf
from transformers import AutoTokenizer
from huggingface_hub import hf_hub_download
import os, re

st.set_page_config(page_title="Analisis Sentimen", page_icon="💬")
st.title("💬 Analisis Komentar Pada Kasus Tom Lembong")

REPO_ID = "zahratalitha/cnn2"

MODEL_PATH = hf_hub_download(repo_id=REPO_ID, filename="best_model_tf.h5")

# Tokenizer
TOKENIZER_PATH = hf_hub_download(repo_id=REPO_ID, filename="best_model_tokenizer.zip")
TOKENIZER_DIR = os.path.dirname(TOKENIZER_PATH)

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(MODEL_PATH)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR)
    return model, tokenizer

model, tokenizer = load_model()

# --- Cleaning teks sederhana ---
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# Label mapping
id2label = {0: "NEGATIVE", 1: "NEUTRAL", 2: "POSITIVE"}

def predict(text):
    clean = clean_text(text)
    enc = tokenizer(
        [clean],
        padding="max_length",
        truncation=True,
        max_length=128,
        return_tensors="np"
    )
    preds = model.predict(
        {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]},
        verbose=0
    )
    label_id = preds.argmax(axis=1)[0]
    return id2label[label_id], float(preds.max()), clean

# --- UI ---
st.subheader("Masukkan teks komentar:")
user_input = st.text_area("Teks:", "")

if st.button("🔍 Analisis Sentimen"):
    if user_input.strip():
        label, score, cleaned = predict(user_input)
        st.info(f"📝 Preprocessed: {cleaned}")
        st.success(f"**Prediksi:** {label} (confidence: {score:.2f})")
    else:
        st.warning("Silakan masukkan teks terlebih dahulu.")
