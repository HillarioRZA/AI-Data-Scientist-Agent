# File: backend/services/memory_manager.py

"""
Modul ini bertindak sebagai penyimpanan memori (state) sederhana untuk sesi pengguna.
Ia menyimpan metrik dari model yang telah dilatih.
"""

from langchain.memory import ConversationBufferMemory
from typing import Dict, Any

# Ini adalah "buku catatan" atau memori sesi kita.
# Ini adalah dictionary global di dalam modul, 
# yang berarti ia akan tetap ada selama server berjalan (shared state).
_session_memory: Dict[str, Dict[str, Any]] = {}

def _get_session_data(session_id: str) -> Dict[str, Any]:
    """Mendapatkan atau membuat dictionary data untuk session_id tertentu."""
    if session_id not in _session_memory:
        _session_memory[session_id] = {
            "model_registry": {},
            "active_vector_store": None,
            "chat_memory": None # Akan dibuat saat get_or_create_memory dipanggil
        }
    return _session_memory[session_id]

# --- Fungsi untuk Metrik Model (Per Sesi) ---
def save_model_data(session_id: str, model_name: str, metrics: dict, model_path: str, preprocessor_path: str):
    """Menyimpan atau memperbarui metrik model untuk sesi dan nama model tertentu."""
    session_data = _get_session_data(session_id)
    if model_name:
        session_data["model_registry"][model_name] = { # <-- Diubah dari model_metrics
            "metrics": metrics,
            "model_path": model_path,
            "preprocessor_path": preprocessor_path
        }
        print(f"--- Data Model untuk sesi {session_id}, model '{model_name}' diperbarui ---")

def get_model_data(session_id: str, model_name: str) -> dict | None:
    """Mengambil metrik model yang tersimpan untuk sesi dan nama model tertentu."""
    session_data = _get_session_data(session_id)
    if model_name:
        return session_data["model_registry"].get(model_name)
    return None

def clear_all_model_data_for_session(session_id: str):
    """Menghapus semua metrik model HANYA untuk sesi ini."""
    if session_id in _session_memory:
        _session_memory[session_id]["model_registry"] = {}
        print(f"--- Metrik Model untuk sesi {session_id} dihapus ---")

# --- Fungsi untuk Vector Store (Per Sesi) ---
def save_vector_store(session_id: str, vector_store: Any):
    """Menyimpan objek vector store aktif untuk sesi ini."""
    session_data = _get_session_data(session_id)
    session_data["active_vector_store"] = vector_store
    print(f"--- Vector Store untuk sesi {session_id} disimpan ---")

def get_vector_store(session_id: str) -> Any | None:
    """Mengambil objek vector store aktif untuk sesi ini."""
    session_data = _get_session_data(session_id)
    return session_data.get("active_vector_store")

def clear_vector_store(session_id: str):
    """Menghapus vector store aktif untuk sesi ini."""
    if session_id in _session_memory:
        _session_memory[session_id]["active_vector_store"] = None
        print(f"--- Vector Store untuk sesi {session_id} dihapus ---")

# --- Fungsi untuk Memori Chat (Per Sesi) ---
def get_or_create_memory(session_id: str) -> ConversationBufferMemory:
    """Mendapatkan memori chat untuk session_id, atau membuat yang baru jika belum ada."""
    session_data = _get_session_data(session_id)
    if session_data.get("chat_memory") is None:
        print(f"Membuat objek memori chat baru untuk sesi: {session_id}")
        session_data["chat_memory"] = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
    return session_data["chat_memory"]

def clear_chat_memory(session_id: str):
    """Menghapus memori chat untuk sesi ini."""
    if session_id in _session_memory:
        _session_memory[session_id]["chat_memory"] = None # Atau hapus objeknya jika perlu
        print(f"--- Memori Chat untuk sesi {session_id} dihapus ---")

# --- Fungsi Pembersihan Total untuk Satu Sesi ---
def clear_all_memory_for_session(session_id: str):
    """Menghapus SEMUA data memori (metrik, vector store, chat) HANYA untuk sesi ini."""
    if session_id in _session_memory:
        del _session_memory[session_id]
        print(f"--- SEMUA memori untuk sesi {session_id} dihapus ---")
    else:
        print(f"--- Tidak ada memori ditemukan untuk sesi {session_id} ---")
