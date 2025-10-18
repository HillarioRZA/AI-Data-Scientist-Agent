# File: backend/services/ml/predictor.py
import pandas as pd

def predict_new_data(new_data: dict, model, preprocessor):
    """
    Melakukan prediksi pada satu baris data baru menggunakan model dan preprocessor yang dimuat.
    
    Args:
        new_data: Sebuah dictionary yang merepresentasikan data baru (satu baris).
        model: Objek model yang sudah dilatih dan dimuat.
        preprocessor: Objek preprocessor yang sudah di-fit dan dimuat.

    Returns:
        Hasil prediksi.
    """
    try:
        # Ubah dictionary data baru menjadi DataFrame pandas
        new_df = pd.DataFrame([new_data])
        
        # Terapkan preprocessor yang sama persis seperti saat pelatihan
        new_data_processed = preprocessor.transform(new_df)
        
        # Lakukan prediksi
        prediction = model.predict(new_data_processed)
        
        # Kembalikan hasil prediksi (ambil elemen pertama dari array)
        return {"prediction": prediction[0].tolist()}

    except Exception as e:
        return {"error": f"Gagal melakukan prediksi. Pastikan data baru memiliki semua kolom yang dibutuhkan. Detail: {str(e)}"}