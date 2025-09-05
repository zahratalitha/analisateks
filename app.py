import streamlit as st
import tensorflow as tf
from transformers import AutoTokenizer
from huggingface_hub import hf_hub_download
import zipfile
import os
import re

st.set_page_config(page_title="Analisis Sentimen Teks", page_icon="💬")
st.title("💬 Analisis Sentimen Komentar Sosial Media")

REPO_ID = "zahratalitha/sentimenteks"
MODEL_ZIP = "sentiment_model_tf.zip"
TOKENIZER_ZIP = "tokenizer.zip"

MODEL_DIR = "sentiment_model_tf"
TOKENIZER_DIR = "tokenizer"

if not os.path.exists(MODEL_DIR):
    model_zip = hf_hub_download(repo_id=REPO_ID, repo_type="dataset", filename=MODEL_ZIP)
    with zipfile.ZipFile(model_zip, "r") as zip_ref:
        zip_ref.extractall(MODEL_DIR)

if not os.path.exists(TOKENIZER_DIR):
    tok_zip = hf_hub_download(repo_id=REPO_ID, repo_type="dataset", filename=TOKENIZER_ZIP)
    with zipfile.ZipFile(tok_zip, "r") as zip_ref:
        zip_ref.extractall(TOKENIZER_DIR)




