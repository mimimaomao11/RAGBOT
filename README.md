# 🏍️ 機車保險知識型 RAG 問答助手

部署網站：[ragbot](https://ragbot-lgmgc4jvyetfrdzzrbpfq4.streamlit.app/)

這是一個基於 RAG（Retrieval-Augmented Generation）架構的問答應用程式，專門回答關於台灣機車保險（強制險與第三人責任險）的相關問題。

使用者輸入問題後，系統會先從向量資料庫檢索最相關的知識片段，再將這些片段連同問題送給 OpenAI GPT 模型生成精確回答，並在回答下方標示參考來源檔名。

## ✨ 功能特色

- **領域專注問答**：聚焦機車保險領域，提供比通用模型更精確的回答
- **來源透明**：每則回答下方顯示「參考來源：xxx.txt」，答案可追溯
- **即時互動介面**：Streamlit 打造的輕量網頁介面
- **持久化向量庫**：ChromaDB PersistentClient，索引重啟後無需重建
- **易於擴充**：將新 `.txt` 知識文件放入 `data/raw_html/` 後重新執行建庫腳本即可

## 🛠️ 技術棧

| 層級 | 技術 |
|------|------|
| 前端框架 | Streamlit |
| 語言模型 | OpenAI API（gpt-4o-mini） |
| 向量嵌入模型 | sentence-transformers（all-MiniLM-L6-v2） |
| 向量資料庫 | ChromaDB（PersistentClient） |
| 主要套件 | `chromadb`, `openai`, `streamlit`, `sentence-transformers` |

## 🚀 本地端執行步驟

### 1. 建立並啟用虛擬環境

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

### 2. 安裝依賴套件

```bash
pip install -r requirements.txt
```

### 3. 設定 OpenAI API 金鑰

建立 `.streamlit/secrets.toml` 並填入金鑰：

```toml
OPENAI_API_KEY = "sk-..."
```

### 4. 建立向量資料庫

執行一次即可，知識庫更新時再次執行：

```bash
python src/build_vector_db.py
```

### 5. 啟動應用程式

```bash
streamlit run src/app.py
```

瀏覽器會自動開啟應用程式。

## 📂 專案結構

```
AIoT-HW4/
├── data/
│   └── raw_html/              # 知識庫 .txt 文件
├── embeddings/
│   └── chroma_db/             # ChromaDB 自動生成的持久化索引（本地執行後產生）
├── src/
│   ├── app.py                 # Streamlit 前端主程式
│   ├── build_vector_db.py     # 建立 ChromaDB 向量索引
│   ├── preprocess.py          # 文件讀取與文字切分
│   ├── rag_engine.py          # RAG 核心：向量檢索 + LLM 生成
│   └── generate_new_article.py
├── .gitignore
├── README.md
└── requirements.txt
```

## ☁️ Streamlit Cloud 部署

1. 將專案推送到 GitHub
2. 前往 [Streamlit Community Cloud](https://streamlit.io/cloud) 連結儲存庫
3. 在 App Settings → Secrets 填入：
   ```toml
   OPENAI_API_KEY = "sk-..."
   ```
4. 部署前先在本地執行 `build_vector_db.py` 並將 `embeddings/chroma_db/` 一併推送（或移除 .gitignore 中的對應行）
