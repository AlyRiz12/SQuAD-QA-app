print("✅ app.py berhasil dijalankan")

from flask import Flask, request, jsonify, send_from_directory
import os

from flask import Flask, request, jsonify
import pandas as pd
import re
from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering

# ====== Inisialisasi Flask App ======
app = Flask(__name__)

# ====== Load dan Bersihkan Dataset ======
df = pd.read_csv("SQuAD_small.csv")  # Pastikan file ini ada di folder yg sama
df.dropna(inplace=True)
df = df.drop(columns=['Unnamed: 0', 'id', 'answer_start'])
df.rename(columns={'text': 'answer'}, inplace=True)

# Bersihkan teks
def clean_text(text):
    text = re.sub(r"\(.*?\)", "", text)
    text = re.sub(r'\s+', ' ', text.strip().lower())
    return text

for col in ['context', 'question', 'answer']:
    df[col] = df[col].apply(clean_text)

# ====== Load Model Terbaik (RoBERTa) ======
MODEL_CHECKPOINT = "deepset/roberta-base-squad2"
tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)
model = AutoModelForQuestionAnswering.from_pretrained(MODEL_CHECKPOINT)
qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer)

# ====== Fungsi untuk Mencari Konteks yang Relevan ======
def get_relevant_context(question):
    match = df[df['question'].str.lower() == question.lower()]
    if not match.empty:
        return match.iloc[0]['context']
    else:
        return None

# ====== Endpoint Root ======
@app.route("/", methods=["GET"])
def home():
    return "Welcome to the QA API! Use POST /qa with question and context or just question."

# ====== Endpoint untuk QA langsung (pakai context) ======
@app.route("/qa", methods=["POST"])
def answer_question():
    data = request.get_json()
    question = data.get("question")
    context = data.get("context")

    if not question:
        return jsonify({"error": "Question is required"}), 400

    # Jika tidak diberikan konteks, ambil dari dataset
    if not context:
        context = get_relevant_context(question)
        if context is None:
            return jsonify({"error": "Konteks tidak ditemukan dalam dataset"}), 404

    result = qa_pipeline(question=question, context=context)
    return jsonify({"answer": result['answer']})

@app.route('/frontend')
def frontend():
    return send_from_directory(os.getcwd(), 'index.html')

# ====== Jalankan Flask App ======
if __name__ == "__main__":
    print("Running Flask app...")
    app.run(debug=True, host='127.0.0.1', port=5000, threaded=False)