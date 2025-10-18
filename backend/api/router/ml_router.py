# File: backend/api/ml_router.py

from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse
import pandas as pd

from backend.services.ml import selector, preprocessor, trainer, evaluator
from backend.services.eda.main import _read_csv_with_fallback

router = APIRouter(
    prefix="/api/ml",
    tags=["Machine Learning"]
)

model_artifacts = {
    "model": None,
    "preprocessor": None,
    "problem_type": None
}

@router.post("/run-pipeline")
async def run_full_ml_pipeline(
    file: UploadFile = File(...),
    target_column: str = Form(...)
):
    """
    Menjalankan pipeline machine learning end-to-end:
    Membaca data -> Deteksi Masalah -> Preprocessing -> Training -> Evaluasi.
    """
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Format file tidak valid.")

    contents = await file.read()
    df = _read_csv_with_fallback(contents)

    if df is None:
        raise HTTPException(status_code=500, detail="Gagal membaca atau memproses file CSV.")
    
    if target_column not in df.columns:
        raise HTTPException(status_code=404, detail=f"Kolom target '{target_column}' tidak ditemukan.")

    # --- PERBAIKAN DI SINI ---
    # Hapus semua baris di mana nilai pada kolom target adalah NaN.
    # Ini memastikan model hanya belajar dari data yang memiliki "kunci jawaban".
    df.dropna(subset=[target_column], inplace=True)
    # -------------------------

    try:
        # --- Alur Kerja Pipeline ---
        
        # 1. Deteksi Masalah
        problem_type = selector.detect_problem_type(df[target_column])
        
        # 2. Persiapan Data
        X_processed, y, pipeline_object = preprocessor.preprocess_data(df, target_column)
        
        # 3. Latih Model
        trained_model, X_test, y_test, model_name = trainer.train_model(X_processed, y, problem_type)
        
        # 4. Evaluasi Model
        evaluation_metrics = evaluator.evaluate_model(trained_model, X_test, y_test, problem_type)

        model_artifacts["model"] = trained_model
        model_artifacts["preprocessor"] = pipeline_object
        model_artifacts["problem_type"] = problem_type
        
        # 5. Kembalikan Laporan Akhir
        return JSONResponse(content={
            "message": "Pipeline ML berhasil dijalankan.",
            "problem_type_detected": problem_type,
            "model_trained": model_name,
            "evaluation_metrics": evaluation_metrics
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Terjadi kesalahan saat menjalankan pipeline ML: {str(e)}")

@router.post("/predict")
def make_prediction(
    new_data: dict # FastAPI akan otomatis mengubah body JSON menjadi dictionary
):
    """
    Menerima data baru dalam format JSON dan mengembalikan hasil prediksi
    dari model yang sudah dilatih.
    """
    if model_artifacts["model"] is None:
         raise HTTPException(status_code=400, detail="Model belum dilatih. Jalankan '/run-pipeline' terlebih dahulu.")

    try:
        result = predictor.predict_new_data(new_data, model_artifacts)
        if "error" in result:
             raise HTTPException(status_code=500, detail=result["error"])
        
        return JSONResponse(content=result)
    
    except Exception as e:
        # Menangkap error jika data baru tidak memiliki kolom yang sesuai
        raise HTTPException(status_code=400, detail=f"Error saat prediksi: Pastikan semua kolom fitur yang dibutuhkan ada dalam data. Detail: {str(e)}")

@router.post("/run-tuned-pipeline")
async def run_tuned_ml_pipeline(
    file: UploadFile = File(...),
    target_column: str = Form(...)
):
    """
    Menjalankan pipeline ML lengkap DENGAN hyperparameter tuning
    untuk mendapatkan performa model yang lebih baik.
    """
    # ... (Kode validasi file dan pembersihan target sama seperti di /run-pipeline) ...
    contents = await file.read()
    df = _read_csv_with_fallback(contents)
    df.dropna(subset=[target_column], inplace=True)

    try:
        problem_type = selector.detect_problem_type(df[target_column])
        X_processed, y, pipeline_object = preprocessor.preprocess_data(df, target_column)
        
        # --- PERBEDAAN UTAMA DI SINI ---
        # Panggil fungsi train_model dengan perform_tuning=True
        trained_model, X_test, y_test, model_name, best_params = trainer.train_model(
            X_processed, y, problem_type, perform_tuning=True
        )
        # -------------------------------
        
        evaluation_metrics = evaluator.evaluate_model(trained_model, X_test, y_test, problem_type)
        
        # Simpan artefak (opsional, jika Anda ingin menggunakan model hasil tuning untuk prediksi)
        model_artifacts["model"] = trained_model
        model_artifacts["preprocessor"] = pipeline_object
        model_artifacts["problem_type"] = problem_type
        
        return JSONResponse(content={
            "message": "Pipeline ML dengan hyperparameter tuning berhasil dijalankan.",
            "problem_type_detected": problem_type,
            "model_trained": model_name,
            "best_params_found": best_params, # Kembalikan parameter terbaik yang ditemukan
            "evaluation_metrics": evaluation_metrics
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Terjadi kesalahan saat menjalankan pipeline tuning: {str(e)}")

@router.post("/feature-importance")
def get_model_feature_importance():
    """
    Mengembalikan daftar fitur yang paling berpengaruh dari model
    yang terakhir dilatih.
    """
    if model_artifacts["model"] is None:
         raise HTTPException(status_code=400, detail="Model belum dilatih. Jalankan '/run-pipeline' terlebih dahulu.")

    try:
        importance_results = evaluator.get_feature_importance(
            model_artifacts["model"],
            model_artifacts["preprocessor"]
        )
        
        if "error" in importance_results[0]:
             raise HTTPException(status_code=500, detail=importance_results[0]["error"])
        
        return JSONResponse(content={
            "message": "10 fitur paling berpengaruh berhasil diekstrak.",
            "feature_importances": importance_results
        })
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saat mengekstrak fitur: {str(e)}")

@router.get("/download/{model_name}")
def download_model_artifacts(model_name: str, type: str = "model"):
    """
    Mengunduh file artefak model yang tersimpan (model atau preprocessor).
    Gunakan query parameter ?type=model atau ?type=preprocessor
    """
    
    if type == "model":
        file_name = f"{model_name}_model.joblib"
    elif type == "preprocessor":
        file_name = f"{model_name}_preprocessor.joblib"
    else:
        raise HTTPException(status_code=400, detail="Tipe tidak valid. Pilih 'model' atau 'preprocessor'.")

    path = f"saved_models/{file_name}"
    
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"File artefak '{file_name}' tidak ditemukan.")
    
    return FileResponse(
        path, 
        media_type='application/octet-stream', 
        filename=file_name
    )