# File: backend/services/memory_manager.py

"""
Modul ini bertindak sebagai penyimpanan memori (state) sederhana untuk sesi pengguna.
Ia menyimpan metrik dari model yang telah dilatih.
"""

# Ini adalah "buku catatan" atau memori sesi kita.
# Ini adalah dictionary global di dalam modul, 
# yang berarti ia akan tetap ada selama server berjalan (shared state).
_session_memory = {
    "model_metrics": {},  # Misal: {"model_v1": {"accuracy": 0.76, ...}}
    "active_vector_store": None
}

def save_model_metrics(model_name: str, metrics: dict):
    """
    Menyimpan atau memperbarui metrik untuk model tertentu.
    """
    if model_name:
        _session_memory["model_metrics"][model_name] = metrics
        print(f"--- Memori Sesi Diperbarui ---")
        print(_session_memory)
        print("------------------------------")

def get_model_metrics(model_name: str) -> dict | None:
    """
    Mengambil metrik yang tersimpan untuk model tertentu.
    """
    if model_name:
        return _session_memory["model_metrics"].get(model_name)
    return None

def clear_all_metrics():
    """
    Menghapus semua metrik yang tersimpan (berguna untuk reset sesi).
    """
    _session_memory["model_metrics"] = {}
    print("--- Memori Sesi Dikosongkan ---")

def save_vector_store(vector_store):
    """Menyimpan objek vector store aktif."""
    _session_memory["active_vector_store"] = vector_store
    print(f"--- Vector Store aktif disimpan ---")

def get_vector_store():
    """Mengambil objek vector store aktif."""
    return _session_memory.get("active_vector_store")

def clear_vector_store():
    """Menghapus vector store aktif."""
    _session_memory["active_vector_store"] = None
    print(f"--- Vector Store aktif dihapus ---")