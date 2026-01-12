import os
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from dotenv import load_dotenv


import streamlit as st
import google.generativeai as genai


# .envファイルをロード
load_dotenv()

# APIキー取得
api_key = st.secrets["GOOGLE_API_KEY"]
genai.configure(api_key=api_key)
if not api_key:
    st.error("APIキーが設定されていません。Google CloudのAPIキーを設定してください。")
    st.stop()

# ================================
# 関数の設定
# ================================
# ================================
# Geminiモデル取得
# ================================
@st.cache_resource
def get_gemini_model():
    return genai.GenerativeModel("models/gemini-flash-latest")



# ================================
# CSV読み込み
# ================================
@st.cache_data
def load_data(csv_file_path):
    df = pd.read_csv(csv_file_path)
    df = df.dropna(subset=["text"])
    return df


# ================================
# TF-IDFモデル構築
# ================================
@st.cache_resource
def build_tfidf_model(texts):
    tfidf_vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        stop_words="english"
    )
    tfidf_matrix = tfidf_vectorizer.fit_transform(texts)
    return tfidf_vectorizer, tfidf_matrix



# ================================
# SentenceTransformerモデル取得
# ================================
@st.cache_resource
def get_embedding_model():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return model


# ================================
# 埋め込みベクトル構築
# ================================
@st.cache_resource
def build_embedding_model(texts):
    model = get_embedding_model()
    embeddings = model.encode(texts, show_progress_bar=True)
    return embeddings


# ================================
# ハイブリッド検索
# ================================
def hybrid_search(query, tfidf_matrix, tfidf_vectorizer, embeddings, top_n=5):
    # TF-IDF類似度
    query_tfidf = tfidf_vectorizer.transform([query])
    tfidf_scores = cosine_similarity(query_tfidf, tfidf_matrix)[0]

    # Embedding類似度
    embed_model = get_embedding_model()
    query_embedding = embed_model.encode([query])
    embed_scores = cosine_similarity(query_embedding, embeddings)[0]

    # 正規化
    tfidf_scores = (tfidf_scores - tfidf_scores.min()) / (tfidf_scores.max() - tfidf_scores.min() + 1e-8)
    embed_scores = (embed_scores - embed_scores.min()) / (embed_scores.max() - embed_scores.min() + 1e-8)

    # ハイブリッドスコア
    hybrid_scores = (tfidf_scores + embed_scores) / 2

    top_indices = np.argsort(hybrid_scores)[::-1][:top_n]

    return top_indices, hybrid_scores[top_indices]


# ================================
# チャット履歴初期化
# ================================
def init_chat_history():
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []


# ================================
# チャット履歴表示
# ================================
def display_chat_history():
    for role, message in st.session_state.chat_history:
        with st.chat_message(role):
            st.markdown(message)


# ================================
# Geminiによる応答生成
# ================================
def respond_with_gemini(query, results, texts, top_n=3):
    model = get_gemini_model()

    context = "\n\n".join(
        [f"記事{i+1}:\n{texts[idx]}" for i, idx in enumerate(results[:top_n])]
    )

    prompt = f"""
以下はニュース記事の抜粋です。

{context}

これらの情報を参考に、次の質問に日本語で簡潔かつ正確に答えてください。

質問:
{query}
"""

    response = model.generate_content(prompt)
    return response.text


# ================================
# Streamlitアプリのメイン
# ================================
st.title("RAG System")

# ----------------
# データロード
# ----------------
csv_file_path = "yahoo_news_articles_preprocessed.csv"
df = load_data(csv_file_path)

# ニュース本文をリスト化
texts = df["text"].tolist()

# ----------------
# モデル構築
# ----------------
tfidf_vectorizer, tfidf_matrix = build_tfidf_model(texts)
embeddings = build_embedding_model(texts)

# ----------------
# チャット履歴初期化
# ----------------
init_chat_history()
display_chat_history()

# ----------------
# ユーザー入力
# ----------------
user_input = st.chat_input("質問を入力してください")

if user_input:
    # ユーザー発言を履歴に追加
    st.session_state.chat_history.append(("user", user_input))
    with st.chat_message("user"):
        st.markdown(user_input)

    # --------
    # 検索
    # --------
    top_indices, scores = hybrid_search(
        query=user_input,
        tfidf_matrix=tfidf_matrix,
        tfidf_vectorizer=tfidf_vectorizer,
        embeddings=embeddings,
        top_n=5
    )

    # --------
    # Gemini応答生成
    # --------
    answer = respond_with_gemini(
        query=user_input,
        results=top_indices,
        texts=texts,
        top_n=3
    )

    # --------
    # AI応答表示
    # --------
    st.session_state.chat_history.append(("assistant", answer))
    with st.chat_message("assistant"):
        st.markdown(answer)

