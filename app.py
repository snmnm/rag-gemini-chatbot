# -*- coding: utf-8 -*-
import os

# 🧼 İzin hatalarını önlemek için ortam değişkenlerini EN ÜSTE koyuyoruz
os.environ["STREAMLIT_HOME"] = "/tmp/streamlit"
os.environ["STREAMLIT_CACHE_DIR"] = "/tmp/streamlit_cache"
os.environ["HAYSTACK_TELEMETRY_ENABLED"] = "false"
os.environ["HAYSTACK_HOME"] = "/tmp/haystack_home"
os.environ["HF_HOME"] = "/tmp/hf_cache"
os.environ["HF_DATASETS_CACHE"] = "/tmp/hf_datasets_cache"

# 📌 Haystack Secret objesi
from haystack.utils import Secret

# 📦 Geri kalan importlar
import streamlit as st
from dotenv import load_dotenv
from datasets import load_dataset

# 🧠 Haystack importları
from haystack import Pipeline
from haystack.dataclasses import Document
from haystack.components.preprocessors import DocumentSplitter
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.embedders import (
    SentenceTransformersDocumentEmbedder,
    SentenceTransformersTextEmbedder,
)
from haystack.components.writers import DocumentWriter
from haystack.document_stores.types import DuplicatePolicy
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack.components.builders import ChatPromptBuilder
from haystack_integrations.components.generators.google_genai import GoogleGenAIChatGenerator

# ------------------------------------------------------------------------
# 🌱 Ortam Değişkenleri
# ------------------------------------------------------------------------
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    st.error("🚨 GOOGLE_API_KEY eksik! (Spaces > Settings > Secrets)")
    st.stop()

# ------------------------------------------------------------------------
# 📚 Veri Seti Yükleme ve Belge Hazırlama
# ------------------------------------------------------------------------
@st.cache_resource
def load_and_prepare_data():
    with st.spinner("📥 SCIQ veri seti yükleniyor..."):
        dataset = load_dataset("allenai/sciq", split="train")
        docs = []
        for i, row in enumerate(dataset):
            content = f"Soru: {row['question']}\nCevap: {row['correct_answer']}"
            docs.append(Document(id=f"sciq-{i}", content=content))

        splitter = DocumentSplitter(split_by="word", split_length=128, split_overlap=30)
        split_docs = splitter.run(docs)["documents"]
        return split_docs

# ------------------------------------------------------------------------
# 🧭 Vektör Veritabanı
# ------------------------------------------------------------------------
@st.cache_resource
def create_vector_store(docs):
    store = InMemoryDocumentStore()
    embedder = SentenceTransformersDocumentEmbedder(
        model="sentence-transformers/all-MiniLM-L6-v2"
    )
    writer = DocumentWriter(document_store=store, policy=DuplicatePolicy.OVERWRITE)

    indexing = Pipeline()
    indexing.add_component("embedder", embedder)
    indexing.add_component("writer", writer)
    indexing.connect("embedder.documents", "writer.documents")
    indexing.run({"embedder": {"documents": docs}})
    return store

# ------------------------------------------------------------------------
# 🧠 RAG Pipeline
# ------------------------------------------------------------------------
@st.cache_resource
def build_rag_pipeline(_store):
    retriever = InMemoryEmbeddingRetriever(document_store=_store, top_k=3)
    text_embedder = SentenceTransformersTextEmbedder(
        model="sentence-transformers/all-MiniLM-L6-v2"
    )
    template = """Belgelerden yararlanarak aşağıdaki soruyu yanıtla.
Eğer cevap bulunamazsa: "Belgelerde bu konuda bilgi yok." yaz.
Belgeler:
{% for doc in documents %}
{{ doc.content }}
{% endfor %}
Soru: {{question}}
Yanıt:
"""
    prompt_builder = ChatPromptBuilder(template=template)

    generator = GoogleGenAIChatGenerator(
        model="gemini-2.5-flash",  # İstersen "gemini-2.5-pro" da kullanabilirsin
        api_key=Secret.from_token(GOOGLE_API_KEY)
    )

    pipeline = Pipeline()
    pipeline.add_component("text_embedder", text_embedder)
    pipeline.add_component("retriever", retriever)
    pipeline.add_component("prompt_builder", prompt_builder)
    pipeline.add_component("generator", generator)

    pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
    pipeline.connect("retriever.documents", "prompt_builder.documents")
    pipeline.connect("prompt_builder.prompt", "generator.messages")

    return pipeline

# ------------------------------------------------------------------------
# 🧑‍💻 Arayüz
# ------------------------------------------------------------------------
def main():
    st.set_page_config(page_title="Bilim Soru-Cevap Chatbotu", page_icon="🧪")
    st.title("🧪 Bilim Soru-Cevap Chatbotu (RAG + Gemini)")
    st.caption("Veri seti: allenai/sciq | Retriever: Sentence Transformers | LLM: Gemini 2.5 Flash")

    docs = load_and_prepare_data()
    store = create_vector_store(docs)
    rag_pipeline = build_rag_pipeline(store)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # geçmiş mesajları yazdır
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # chat input alanı
    user_q = st.chat_input("Bilimsel bir soru sor...")
    if user_q:
        if not user_q.strip():
            st.warning("⚠️ Lütfen boş olmayan bir soru yazın.")
        else:
            st.session_state.messages.append({"role": "user", "content": user_q})
            with st.chat_message("user"):
                st.markdown(user_q)

            with st.spinner("Düşünüyorum 🤔..."):
                try:
                    result = rag_pipeline.run({
                        "text_embedder": {"text": user_q},
                        "prompt_builder": {"question": user_q}
                    })
                    answer = result.get("generator", {}).get("replies", ["Bir hata oluştu 😕"])[0]
                except Exception as e:
                    answer = f"❌ Hata: {e}"

            st.session_state.messages.append({"role": "assistant", "content": answer})
            with st.chat_message("assistant"):
                st.markdown(answer)

# ------------------------------------------------------------------------
# 🚀 Çalıştırma
# ------------------------------------------------------------------------
if __name__ == "__main__":
    main()
