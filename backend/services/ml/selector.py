import pandas as pd
from typing import Literal

def detect_problem_type(target_column: pd.Series) -> Literal["Klasifikasi", "Regresi"]:
    """
    Mendeteksi tipe masalah machine learning berdasarkan karakteristik kolom target.

    Args:
        target_column: Seri pandas dari kolom target.

    Returns:
        String "Klasifikasi" atau "Regresi".
    """
    # Menghapus nilai yang hilang untuk analisis
    target_column = target_column.dropna()

    # Aturan 1: Jika tipe data adalah object (teks), pasti Klasifikasi.
    if pd.api.types.is_object_dtype(target_column):
        return "Klasifikasi"

    # Aturan 2: Jika tipe data numerik, lakukan pengecekan lebih detail.
    if pd.api.types.is_numeric_dtype(target_column):
        unique_values = target_column.nunique()

        # --- PERBAIKAN UTAMA DI SINI ---
        # Jika hanya ada 2 nilai unik (misal: 0 dan 1, atau 0.0 dan 1.0), ini adalah Klasifikasi Biner.
        if unique_values == 2:
            return "Klasifikasi"
        # --------------------------------

        # Heuristik lama untuk kasus multikelas (misal: rating 1-5)
        if unique_values <= 15 and pd.api.types.is_integer_dtype(target_column):
            return "Klasifikasi"
        else:
            # Jika semua syarat di atas tidak terpenuhi, ini adalah Regresi.
            return "Regresi"
    
    # Default jika tipe data tidak terduga
    return "Klasifikasi"