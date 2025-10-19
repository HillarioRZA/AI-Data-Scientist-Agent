# File: backend/services/memory/memory_manager.py

"""
Modul ini bertindak sebagai penyimpanan memori (state) sederhana untuk sesi pengguna.
Ia menyimpan metrik dari model yang telah dilatih.
"""

from langchain.memory import ConversationBufferMemory
from typing import Dict, Any
from backend.services.memory import persistent_memory

# Ini adalah "buku catatan" atau memori sesi kita.
# Ini adalah dictionary global di dalam modul, 
# yang berarti ia akan tetap ada selama server berjalan (shared state).
_session_memory: Dict[str, Dict[str, Any]] = {}

def _get_session_data(session_id: str) -> Dict[str, Any]:
    """Mendapatkan atau membuat dictionary data cache (STM) untuk session_id tertentu."""
    if session_id not in _session_memory:
        _session_memory[session_id] = {
            # Registri model sekarang ditangani LTM, dihapus dari sini.
            "active_vector_store": None,
            "chat_memory": None # Akan diisi oleh get_or_create_memory
        }
    return _session_memory[session_id]

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
# --- Fungsi untuk Memori Chat (Interaksi STM + LTM) ---
def get_or_create_memory(session_id: str) -> ConversationBufferMemory:
    """
    Mendapatkan memori chat dari cache (STM). 
    Jika tidak ada di cache, fungsi ini akan mencoba memuatnya dari LTM (TinyDB).
    """
    session_data = _get_session_data(session_id)
    
    # 1. Cek cache (STM) dulu
    if session_data.get("chat_memory") is None:
        print(f"--- [STM] Memori chat tidak ada di cache. Mencoba memuat dari LTM... ---")
        
        # 2. Jika tidak ada di cache, panggil LTM untuk memuat riwayat
        ltm_memory = persistent_memory.load_chat_history(session_id)
        
        # 3. Simpan objek yang dimuat dari LTM ke cache (STM)
        session_data["chat_memory"] = ltm_memory
        return ltm_memory
    else:
        # 4. Jika ada di cache, langsung kembalikan
        print(f"--- [STM] Memori chat ditemukan di cache. ---")
        return session_data["chat_memory"]

def clear_chat_memory(session_id: str):
    """Menghapus memori chat HANYA dari cache (STM)."""
    if session_id in _session_memory:
        _session_memory[session_id]["chat_memory"] = None
        print(f"--- [STM] Memori Chat untuk sesi {session_id} dihapus dari cache ---")

# --- Fungsi Pembersihan Total (STM + LTM) ---
def clear_all_memory_for_session(session_id: str):
    """
    Menghapus SEMUA data memori (cache STM dan data persisten LTM) 
    yang terkait dengan sesi ini.
    """
    # 1. Hapus dari cache (STM)
    if session_id in _session_memory:
        del _session_memory[session_id]
        print(f"--- [STM] SEMUA memori cache untuk sesi {session_id} dihapus ---")
    else:
        print(f"--- [STM] Tidak ada memori cache ditemukan untuk sesi {session_id} ---")
        
    # 2. Panggil pembersihan LTM (yang akan menghapus dari TinyDB dan disk)
    persistent_memory.clear_all_memory_for_session(session_id)
