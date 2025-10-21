from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from fastapi.responses import StreamingResponse
import io
from backend.services.eda import main as eda_main
from backend.services.agent import main as agent_main


router = APIRouter(
    prefix="/api/eda", 
    tags=["EDA"] 
)

@router.post("/describe")
async def describe_data(file: UploadFile = File(...)):
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Format file tidak valid. Harap unggah file CSV.")
    contents = await file.read()
    description = eda_main.get_csv_description(contents)
    if description is None:
        raise HTTPException(status_code=500, detail="Gagal memproses file CSV.")
    return {
        "filename": file.filename,
        "statistics": description
    }

@router.post("/correlation-heatmap", response_class=StreamingResponse)
async def get_correlation_heatmap(file: UploadFile = File(...)):
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Format file tidak valid. Harap unggah file CSV.")
    
    contents = await file.read()
    image_bytes = eda_main.generate_correlation_heatmap(contents)
    if image_bytes is None:
        raise HTTPException(status_code=500, detail="Gagal membuat heatmap.")

    return StreamingResponse(io.BytesIO(image_bytes), media_type="image/png")

@router.post("/histogram", response_class=StreamingResponse)
async def get_histogram(
    file: UploadFile = File(...),
    column_name: str = Form(...)
):

    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Format file tidak valid.")

    contents = await file.read()

    result = eda_main.generate_histogram(contents, column_name)

    if result == "column_not_found":
        raise HTTPException(status_code=404, detail=f"Kolom '{column_name}' tidak ditemukan di dalam file.")
    if result == "column_not_numeric":
        raise HTTPException(status_code=400, detail=f"Kolom '{column_name}' bukan numerik dan tidak bisa dibuatkan histogram.")
    if result is None:
        raise HTTPException(status_code=500, detail="Gagal membuat histogram.")

    return StreamingResponse(io.BytesIO(result), media_type="image/png")

@router.post("/missing-value-heatmap", response_class=StreamingResponse)
async def get_missing_value_heatmap(file: UploadFile = File(...)):
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Format file tidak valid.")
    
    contents = await file.read()
    image_bytes = eda_main.generate_missing_value_heatmap(contents)
    if image_bytes is None:
        raise HTTPException(status_code=500, detail="Gagal membuat heatmap missing values.")

    return StreamingResponse(io.BytesIO(image_bytes), media_type="image/png")

@router.post("/outliers")
async def find_outliers(
    file: UploadFile = File(...),
    column_name: str = Form(...)
):
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Format file tidak valid.")

    contents = await file.read()
    result = eda_main.get_outliers(contents, column_name)

    if result == "column_not_found":
        raise HTTPException(status_code=404, detail=f"Kolom '{column_name}' tidak ditemukan.")
    if result == "column_not_numeric":
        raise HTTPException(status_code=400, detail=f"Kolom '{column_name}' bukan numerik.")
    if result is None:
        raise HTTPException(status_code=500, detail="Gagal mendeteksi outliers.")

    return result

@router.post("/skewness")
async def calculate_skewness(file: UploadFile = File(...)):
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Format file tidak valid.")
    
    contents = await file.read()
    skew_data = eda_main.get_skewness(contents)
    if skew_data is None:
        raise HTTPException(status_code=500, detail="Gagal menghitung skewness.")
        
    return {"skewness": skew_data}

@router.post("/categorical-insights")
async def get_categorical_analysis(
    file: UploadFile = File(...),
    column_name: str = Form(...)
):
    contents = await file.read()
    result = eda_main.get_categorical_insights(contents, column_name)

    if result == "column_not_found":
        raise HTTPException(status_code=404, detail=f"Kolom '{column_name}' tidak ditemukan.")
    if result == "column_is_highly_numeric":
        raise HTTPException(status_code=400, detail=f"Kolom '{column_name}' memiliki terlalu banyak nilai unik untuk dianalisis sebagai kategori.")
    if result is None:
        raise HTTPException(status_code=500, detail="Gagal menganalisis kolom kategorikal.")

    return result

@router.post("/target-analysis")
async def get_target_analysis(
    file: UploadFile = File(...),
    target_column: str = Form(...)
):
    contents = await file.read()
    result = eda_main.analyze_target(contents, target_column)

    if result == "column_not_found":
        raise HTTPException(status_code=404, detail=f"Kolom target '{target_column}' tidak ditemukan.")
    if result is None:
        raise HTTPException(status_code=500, detail="Gagal menganalisis variabel target.")

    return result

@router.post("/target-feature-plot", response_class=StreamingResponse)
async def get_target_feature_relationship_plot(
    file: UploadFile = File(...),
    target_column: str = Form(...),
    feature_column: str = Form(...)
):
    contents = await file.read()
    image_bytes = eda_main.generate_target_feature_plot(contents, target_column, feature_column)

    if image_bytes == "column_not_found":
        raise HTTPException(status_code=404, detail="Satu atau kedua kolom tidak ditemukan.")
    if image_bytes is None:
        raise HTTPException(status_code=500, detail="Gagal membuat visualisasi.")

    return StreamingResponse(io.BytesIO(image_bytes), media_type="image/png")

@router.post("/full-profile")
async def get_full_data_profile(file: UploadFile = File(...)):
    contents = await file.read()
    profile_data = eda_main.run_full_data_profile(contents)
    
    if profile_data is None:
        raise HTTPException(status_code=500, detail="Gagal membuat profil data.")
        
    return profile_data

@router.post("/vif")
async def get_vif_analysis(file: UploadFile = File(...)):
    contents = await file.read()
    result = eda_main.calculate_vif(contents)

    if result is None:
        raise HTTPException(status_code=500, detail="Gagal menghitung VIF.")
    
    return JSONResponse(content={"vif_results": result})
