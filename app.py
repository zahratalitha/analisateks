import streamlit as st
import tensorflow as tf
from transformers import AutoTokenizer
import zipfile
import os
import re

# === Setup halaman ===
st.set_page_config(page_title="Analisis Sentimen Tom Lembong", page_icon="💬")
st.title("💬 Analisis Sentimen Komentar Kasus Tom Lembong")

# === Ekstrak model kalau belum ada ===
MODEL_ZIP = "best_model_full.zip"
MODEL_DIR = "best_model_full"

if not os.path.exists(MODEL_DIR):
    with zipfile.ZipFile(MODEL_ZIP, "r") as zip_ref:
        zip_ref.extractall(MODEL_DIR)

# Path model & tokenizer
MODEL_PATH = os.path.join(MODEL_DIR, "model")
TOKENIZER_PATH = os.path.join(MODEL_DIR, "tokenizer")

# === Load model & tokenizer ===
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(MODEL_PATH)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
    return model, tokenizer

model, tokenizer = load_model()

# === Preprocessing ===
important_mentions = ["tomlembong", "jokowi", "prabowo"]
important_hashtags = ["savetomlembong", "respect", "ripjustice", "justicefortomlembong"]

slang_dict = {
    "yg": "yang", "ga": "tidak", "gk": "tidak", "ngga": "tidak", "nggak": "tidak",
    "tdk": "tidak", "dgn": "dengan", "aja": "saja", "gmn": "gimana", "bgt": "banget",
    "dr": "dari", "utk": "untuk", "dlm": "dalam", "tp": "tapi", "krn": "karena"
}

def clean_text(text, normalize_slang=True):
    if text is None or text == "":
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)

    def mention_repl(match):
        mention = match.group(1)
        return mention if mention in important_mentions else ""
    text = re.sub(r"@(\w+)", mention_repl, text)

    def hashtag_repl(match):
        hashtag = match.group(1)
        return hashtag if hashtag in important_hashtags else ""
    text = re.sub(r"#(\w+)", hashtag_repl, text)

    text = re.sub(r"[^a-z0-9\s]", " ", text)

    if normalize_slang:
        tokens = text.split()
        tokens = [slang_dict.get(tok, tok) for tok in tokens]
        text = " ".join(tokens)

    text = re.sub(r"\s+", " ", text).strip()
    return text

# === Mapping label ===
id2label = {
    0: "ANGER",
    1: "DISAPPOINTMENT",
    2: "NEUTRAL",
    3: "SATISFACTION",
    4: "HAPPINESS"
}

# === Fungsi prediksi ===
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

# === UI ===
st.subheader("Masukkan teks komentar:")
option = st.radio("Pilih input:", ["Ketik manual", "Pilih contoh"])

if option == "Ketik manual":
    user_input = st.text_area("Teks:", "")
else:
    examples = [
        "Saya kecewa dengan pernyataan ini.",
        "Tom Lembong sangat profesional.",
        "Netral saja, tidak terlalu penting.",
        "Senang sekali mendengarnya!"
    ]
    user_input = st.selectbox("Pilih contoh komentar:", examples)

if st.button("🔍 Analisis Sentimen"):
    if user_input.strip():
        label, score, cleaned = predict(user_input)
        st.info(f"📝 **Teks setelah preprocessing:** {cleaned}")
        st.success(f"**Prediksi:** {label} (confidence: {score:.2f})")
    else:
        st.warning("Silakan masukkan teks terlebih dahulu.")
