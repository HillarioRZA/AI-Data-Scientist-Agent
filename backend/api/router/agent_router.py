import io
import json
import base64
from fastapi import APIRouter, HTTPException, Body, UploadFile, File, Form
from fastapi.responses import StreamingResponse
from fastapi.responses import Response, JSONResponse
from backend.services.eda import main as eda_main
from backend.services.agent import main as agent_main
from typing import Optional,List,Dict

router = APIRouter(
    prefix="/api/agent",
    tags=["Agent"]
)

@router.post("/execute")
async def execute_agent_action(
    file: Optional[UploadFile] = File(None),
    prompt: str = Form(...)
):
    """
    Menerima prompt, menjalankan tool yang sesuai, dan mengembalikan
    hasil (gambar) beserta interpretasi AI dalam satu respons multipart.
    """

    contents = None
    file_type = None
    available_columns = []

    if file and file.filename:
        if file.filename.endswith('.csv'):
            file_type = 'csv'
            contents = await file.read()
            # Hanya baca kolom jika filenya CSV
            df = eda_main._read_csv_with_fallback(contents)
            if df is None:
                raise HTTPException(status_code=400, detail="Tidak dapat membaca file CSV.")
            available_columns = df.columns.tolist()

        elif file.filename.endswith('.pdf'):
            file_type = 'pdf'
            contents = await file.read()
        else:
            # Jika ada file tapi formatnya salah
            raise HTTPException(status_code=400, detail="Format file tidak valid. Harap unggah file CSV atau PDF.")
    
    plan = agent_main.get_agent_plan(prompt,available_columns)
    if "error" in plan:
        raise HTTPException(status_code=500, detail=plan.get("detail", plan["error"]))
    
    result = agent_main.execute_tool(plan, contents)
    
    if "error" in result:
        raise HTTPException(status_code=500, detail=result.get("detail", result["error"]))

    return JSONResponse(content=result)

@router.post("/custom-visualize", response_class=JSONResponse)
async def create_custom_visualization(
    file: UploadFile = File(...),
    prompt: str = Form(...)
):
    """
    Endpoint khusus untuk membuat visualisasi kustom berdasarkan prompt detail.
    Ini mengimplementasikan Fitur #12: Customization Visualization.
    """
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Format file tidak valid.")
        
    contents = await file.read()

    # 1. Agen mengekstrak parameter plot dari prompt
    plot_plan = agent_main.get_plot_plan(prompt)
    if "error" in plot_plan:
        raise HTTPException(status_code=500, detail=plot_plan.get("detail", plot_plan["error"]))

    # 2. Buat plot kustom menggunakan parameter yang diekstrak
    image_bytes = eda_main.generate_custom_plot(
        file_contents=contents,
        plot_type=plot_plan.get("plot_type"),
        x_col=plot_plan.get("x_col"),
        y_col=plot_plan.get("y_col"),
        hue_col=plot_plan.get("hue_col"),
        orientation=plot_plan.get("orientation", 'v')
    )

    if isinstance(image_bytes, str): # Penanganan error dari service
        raise HTTPException(status_code=500, detail=f"Gagal membuat plot: {image_bytes}")

    # 3. Minta AI untuk menginterpretasikan plot yang baru dibuat
    summary = agent_main.get_interpretation(
        tool_name=plot_plan.get("plot_type"),
        tool_output=plot_plan, # Berikan rencana sebagai konteks
        image_bytes=image_bytes
    )

    # 4. Encode gambar ke Base64 dan kembalikan semuanya dalam JSON
    image_b64 = base64.b64encode(image_bytes).decode("utf-8")

    return JSONResponse(content={
        "plan": plot_plan,
        "summary": summary,
        "image_base64": image_b64,
        "image_format": "png"
    })