import os
import sys
import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer
from openai import OpenAI

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

MODEL_NAME = "all-MiniLM-L6-v2"


@st.cache_resource
def get_openai_client():
    if "OPENAI_API_KEY" not in st.secrets:
        st.error("請在 Streamlit Secrets 中設定 OPENAI_API_KEY")
        st.stop()
    return OpenAI(api_key=st.secrets["OPENAI_API_KEY"])


@st.cache_resource
def get_rag_engine():
    return RAGEngine()


class RAGEngine:
    def __init__(self):
        from preprocess import load_articles
        self.model = SentenceTransformer(MODEL_NAME)
        chunks_with_metadata = load_articles()
        self.documents = [chunk for chunk, _ in chunks_with_metadata]
        self.sources = [source for _, source in chunks_with_metadata]
        self.embeddings = self.model.encode(self.documents, convert_to_numpy=True)
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        self.embeddings = self.embeddings / np.maximum(norms, 1e-10)

    def retrieve_docs(self, query, top_k=3):
        query_vec = self.model.encode([query], convert_to_numpy=True)
        query_vec = query_vec / np.maximum(np.linalg.norm(query_vec), 1e-10)
        scores = self.embeddings @ query_vec.T
        top_indices = np.argsort(scores[:, 0])[::-1][:top_k]
        docs = [self.documents[i] for i in top_indices]
        sources = [self.sources[i] for i in top_indices]
        return docs, sources


def generate_answer(query):
    rag_engine = get_rag_engine()
    client = get_openai_client()

    context_docs, context_sources = rag_engine.retrieve_docs(query)
    context_text = "\n\n".join(context_docs)
    unique_sources = list(dict.fromkeys(context_sources))

    system_prompt = """你是一個名為「機車保險智多星」的AI助理。你的任務是根據提供的「參考資料」，用專業、親切且深入淺出的方式回答使用者的「問題」。請盡量詳細說明，並在適當的時候舉例。如果參考資料不足以回答問題，請誠實地回答「根據我所擁有的資料，目前無法回答這個問題，建議您洽詢專業的保險業務員喔。」"""

    user_prompt = f"""以下是我的問題以及相關的參考資料，請根據這些資料回答我的問題。

---[參考資料]---
{context_text}
---[參考資料]---

問題：「{query}」"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7,
        )
        answer = response.choices[0].message.content.strip()
    except Exception as e:
        answer = f"呼叫 API 時發生錯誤: {e}"
        unique_sources = []

    return answer, unique_sources
