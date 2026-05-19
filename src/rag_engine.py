import os
import sys
import streamlit as st
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from openai import OpenAI

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHROMA_DIR = os.path.join(BASE_DIR, "embeddings", "chroma_db")

MODEL_NAME = "all-MiniLM-L6-v2"


@st.cache_resource
def get_openai_client():
    if "OPENAI_API_KEY" not in st.secrets:
        st.error("請在 Streamlit Secrets 中設定 OPENAI_API_KEY")
        st.stop()
    return OpenAI(api_key=st.secrets["OPENAI_API_KEY"])


class RAGEngine:
    def __init__(self, model_name=MODEL_NAME):
        self.embedding_fn = SentenceTransformerEmbeddingFunction(
            model_name=model_name
        )
        self.collection = self._load_chroma_db()

    def _build_db(self):
        from preprocess import load_articles
        with st.spinner("首次啟動，正在建立向量資料庫，請稍候..."):
            os.makedirs(CHROMA_DIR, exist_ok=True)
            client = chromadb.PersistentClient(path=CHROMA_DIR)
            try:
                client.delete_collection(name="insurance_kb")
            except Exception:
                pass
            collection = client.get_or_create_collection(
                name="insurance_kb",
                embedding_function=self.embedding_fn
            )
            chunks_with_metadata = load_articles()
            ids = [f"doc_{i}" for i in range(len(chunks_with_metadata))]
            documents = [chunk for chunk, _ in chunks_with_metadata]
            metadatas = [{"source": source} for _, source in chunks_with_metadata]
            collection.add(ids=ids, documents=documents, metadatas=metadatas)

    def _load_chroma_db(self):
        if not os.path.exists(CHROMA_DIR):
            self._build_db()

        client = chromadb.PersistentClient(path=CHROMA_DIR)
        try:
            collection = client.get_collection(
                name="insurance_kb",
                embedding_function=self.embedding_fn
            )
        except Exception as e:
            st.error(f"無法載入 ChromaDB collection：{e}")
            st.stop()

        return collection

    def retrieve_docs(self, query, top_k=3):
        """回傳 (documents, sources) tuple"""
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k
        )

        retrieved_docs = results["documents"][0] if results["documents"] else []
        retrieved_sources = [
            meta.get("source", "未知來源")
            for meta in (results["metadatas"][0] if results["metadatas"] else [])
        ]

        return retrieved_docs, retrieved_sources


@st.cache_resource
def get_rag_engine():
    return RAGEngine()


def generate_answer(query):
    """回傳 (answer, sources) tuple，sources 為去重複後的來源檔名 list"""
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
