import streamlit as st
import tensorflow as tf
from transformers import AutoTokenizer
from huggingface_hub import hf_hub_download
import zipfile
import os
import re

st.set_page_config(page_title="Analisis Sentimen Teks", page_icon="💬")
st.title("💬 Analisis Sentimen Komentar Sosial Media")

from tensorflow import keras
from keras.utils import get_custom_objects

custom_objects = {"TFOpLambda": lambda x: x}
model = keras.models.load_model("sentiment_model.h5", custom_objects=custom_objects)

REPO_ID = "zahratalitha/sentimenteks"  
MODEL_ZIP = "sentiment_model_tf.zip"
TOKENIZER_ZIP = "tokenizer.zip"

MODEL_DIR = "sentiment_model_tf"
TOKENIZER_DIR = "tokenizer"

# ================================
# Download & Extract
# ================================
if not os.path.exists(MODEL_DIR):
    model_zip = hf_hub_download(repo_id=REPO_ID, repo_type="dataset", filename=MODEL_ZIP)
    with zipfile.ZipFile(model_zip, "r") as zip_ref:
        zip_ref.extractall(MODEL_DIR)

if not os.path.exists(TOKENIZER_DIR):
    tok_zip = hf_hub_download(repo_id=REPO_ID, repo_type="dataset", filename=TOKENIZER_ZIP)
    with zipfile.ZipFile(tok_zip, "r") as zip_ref:
        zip_ref.extractall(TOKENIZER_DIR)

# ================================
# Load Model & Tokenizer
# ================================
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(MODEL_DIR)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR)
    return model, tokenizer

model, tokenizer = load_model()

# ================================
# Preprocessing
# ================================
slang_dict = {
    "yg": "yang", "ga": "tidak", "gk": "tidak", "ngga": "tidak",
    "nggak": "tidak", "tdk": "tidak", "dgn": "dengan", "aja": "saja",
    "gmn": "gimana", "bgt": "banget", "dr": "dari", "utk": "untuk",
    "dlm": "dalam", "tp": "tapi", "krn": "karena"
}

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    tokens = text.split()
    tokens = [slang_dict.get(tok, tok) for tok in tokens]
    return " ".join(tokens)

# label mapping (ubah sesuai training kamu)
id2label = {
    0: "SADNESS",
    1: "ANGER",
    2: "SUPPORT",
    3: "HOPE",
    4: "DISAPPOINTMENT"
}

# ================================
# Fungsi Prediksi
# ================================
def predict(text):
    clean = clean_text(text)
    enc = tokenizer(clean, truncation=True, padding="max_length", max_length=128, return_tensors="np")
    preds = model.predict(
        {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]},
        verbose=0
    )
    label_id = preds.argmax(axis=1)[0]
    confidence = float(preds.max())
    return id2label[label_id], confidence, clean

# ================================
# Streamlit UI
# ================================
st.subheader("Masukkan komentar:")
user_input = st.text_area("Komentar:", "")

if st.button("🔍 Analisis Sentimen"):
    if user_input.strip():
        label, score, cleaned = predict(user_input)
        st.info(f"📝 Teks setelah preprocessing: {cleaned}")
        st.success(f"**Prediksi:** {label} (confidence: {score:.2f})")
    else:
        st.warning("Masukkan teks komentar terlebih dahulu!")
