__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
from rag_engine import generate_answer

st.title("🏍️ 機車保險知識型助手 (RAG)")

query = st.text_input("請輸入你的問題：")

if st.button("詢問"):
    if query.strip() != "":
        with st.spinner("AI 正在思考..."):
            answer, sources = generate_answer(query)
        st.markdown("**回答:**")
        st.write(answer)
        
        # 顯示參考來源
        if sources:
            st.markdown("---")
            sources_text = "、".join(sources)
            st.markdown(f"📄 **參考來源：** {sources_text}")
    else:
        st.warning("請輸入問題！")
