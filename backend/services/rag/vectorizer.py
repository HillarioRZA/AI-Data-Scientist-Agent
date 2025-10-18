# File: backend/services/rag/vectorizer.py
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
load_dotenv()

# (Pastikan GOOGLE_API_KEY sudah di-set di environment Anda)
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

def create_vector_store(text_content: str):
    """
    Memecah teks, membuat embeddings, dan membangun vector store FAISS.

    Args:
        text_content: Teks mentah yang diekstrak dari PDF.

    Returns:
        Objek FAISS vector store atau pesan error.
    """
    try:
        # 1. Memecah teks menjadi potongan (chunks)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=8000, # Ukuran setiap potongan
            chunk_overlap=500, # Jumlah overlap antar potongan
            length_function=len
        )
        chunks = text_splitter.split_text(text_content)

        if not chunks:
            return "Error: Tidak ada teks yang bisa diindeks dari PDF."

        # 2. Membuat vector store FAISS dari chunks
        vector_store = FAISS.from_texts(chunks, embedding=embeddings)

        return vector_store

    except Exception as e:
        print(f"Error saat membuat vector store: {e}")
        return f"Error: Gagal membuat index vector. Detail: {str(e)}"