from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score
)
import pandas as pd

def evaluate_model(model, X_test, y_test, problem_type: str) -> dict:
    """
    Mengevaluasi performa model dan mengembalikan metriknya.

    Args:
        model: Model yang sudah dilatih.
        X_test: Fitur dari data uji.
        y_test: Target dari data uji.
        problem_type: "Klasifikasi" atau "Regresi".

    Returns:
        Sebuah dictionary berisi metrik performa.
    """
    # 1. Buat prediksi pada data uji
    y_pred = model.predict(X_test)

    # 2. Hitung metrik berdasarkan tipe masalah
    if problem_type == "Klasifikasi":
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, average='weighted', zero_division=0),
            "recall": recall_score(y_test, y_pred, average='weighted', zero_division=0),
            "f1_score": f1_score(y_test, y_pred, average='weighted', zero_division=0)
        }
    elif problem_type == "Regresi":
        metrics = {
            "mean_squared_error": mean_squared_error(y_test, y_pred),
            "mean_absolute_error": mean_absolute_error(y_test, y_pred),
            "r2_score": r2_score(y_test, y_pred)
        }
    else:
        return {"error": "Tipe masalah tidak didukung."}

    return metrics

def get_feature_importance(model, preprocessor) -> list[dict]:
    """
    Mengekstrak dan menyusun skor kepentingan fitur dari model.

    Args:
        model: Model yang sudah dilatih (misal: RandomForest).
        preprocessor: Objek preprocessor yang digunakan saat pelatihan.

    Returns:
        Daftar dictionary yang berisi fitur dan skor kepentingannya,
        diurutkan dari yang paling penting.
    """
    try:
        # Dapatkan nama fitur setelah preprocessing (termasuk kolom one-hot encoded)
        feature_names = preprocessor.get_feature_names_out()
        
        # Dapatkan skor kepentingan dari model
        importances = model.feature_importances_
        
        # Gabungkan nama dan skor, lalu urutkan
        feature_importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values(by='importance', ascending=False)
        
        # Ambil 10 fitur teratas dan ubah ke format dictionary
        top_10_features = feature_importance_df.head(10)
        
        return top_10_features.to_dict('records')
        
    except AttributeError:
        # Menangani kasus jika model tidak memiliki atribut 'feature_importances_'
        return [{"error": "Model yang digunakan tidak mendukung ekstraksi kepentingan fitur."}]
    except Exception as e:
        return [{"error": f"Terjadi kesalahan: {str(e)}"}]
