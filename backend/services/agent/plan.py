from typing import Optional,List,Dict
from pydantic import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from backend.services.memory import memory_manager
from dotenv import load_dotenv
load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

class ToolPlan(BaseModel):
    tool_name: str = Field(description="Nama alat yang harus digunakan.")
    reasoning: str = Field(description="Alasan singkat pemilihan alat.")
    column_name: Optional[str] = Field(default=None, description="Nama kolom jika dibutuhkan.")
    target_column: Optional[str] = Field(default=None, description="Nama kolom target jika dibutuhkan oleh tool.")
    columns_to_drop: Optional[List[str]] = Field(default=None, description="Daftar nama kolom yang akan dibuang sebelum pelatihan.")
    new_data: Optional[Dict] = Field(default=None, description="Data baru dalam format dictionary untuk prediksi.")
    model_name: Optional[str] = Field(default=None, description="Nama unik untuk menyimpan atau memuat model.")
    baseline_model_name: Optional[str] = Field(default=None, description="Nama model baseline untuk perbandingan tuning.") #
    pdf_question: Optional[str] = Field(default=None, description="Pertanyaan spesifik yang akan diajukan ke dokumen PDF.")

AVAILABLE_TOOLS = [
    {"name": "describe", "description": "Untuk ringkasan statistik (mean, std, etc.)."},
    {"name": "correlation-heatmap", "description": "Untuk visualisasi korelasi antar kolom."},
    {"name": "histogram", "description": "Untuk visualisasi distribusi satu kolom."},
    {"name": "full-profile", "description": "Untuk analisis umum atau profil lengkap dataset."},
    {"name": "outliers", "description": "Untuk mendeteksi pencilan di satu kolom."},
    {"name": "skewness", "description": "Untuk memeriksa kemiringan distribusi data."},
    {"name": "missing-value-heatmap", "description": "Untuk visualisasi data yang hilang."},
    {"name": "target-feature-plot", "description": "Untuk visualisasi hubungan fitur dengan target."},
    {"name": "vif", "description": "Untuk menghitung VIF guna mendeteksi multikolinearitas antar fitur."},
    {"name": "target-analysis", "description": "Untuk analisis variabel target."},
    {"name": "categorical-insights", "description": "Untuk analisis kategorikal."},
    {"name": "run_ml_pipeline", "description": "Gunakan ini untuk melatih model prediksi pada dataset. Butuh 'target_column'."},
    {"name": "run_tuned_ml_pipeline", "description": "Untuk melatih model dengan hyperparameter tuning agar lebih akurat. Butuh 'target_column'."},
    {"name": "get_feature_importance", "description": "Untuk mengetahui fitur apa yang paling penting dari model yang sudah dilatih."},
    {"name": "predict_new_data", "description": "Gunakan setelah model dilatih untuk memprediksi hasil dari satu data baru."},
    {"name": "index_pdf", "description": "Gunakan ini satu kali saat pengguna mengunggah file PDF. Ini akan membaca dan mengindeks dokumen agar siap ditanyai."},
    {"name": "answer_pdf_question", "description": "Gunakan ini untuk menjawab pertanyaan spesifik dari dokumen PDF yang telah diindeks sebelumnya. Butuh 'pdf_question'."},
    {"name": "conversational_recall", "description": "Gunakan ini untuk menjawab pertanyaan kontekstual sederhana yang jawabannya dapat ditemukan di riwayat chat atau memori sesi. JANGAN gunakan tool ini untuk pertanyaan yang memerlukan perhitungan data baru atau tool ML/EDA."}
]

def get_agent_plan(session_id: str,user_prompt: str, column_list: list[str]) -> dict:
    tools_as_string = "\n".join([f"- {tool['name']}: {tool['description']}" for tool in AVAILABLE_TOOLS])
    parser = JsonOutputParser(pydantic_object=ToolPlan)
    prompt_template = ChatPromptTemplate.from_template(
        """Anda adalah AI data analyst. Pilih alat yang paling tepat dari daftar berikut berdasarkan permintaan pengguna.

        --- RIWAYAT PERCAKAPAN SEBELUMNYA ---
        {chat_history}
        --- AKHIR RIWAYAT ---

        {tools}
        PENTING: Semua nama kolom yang Anda pilih HARUS berasal dari daftar kolom yang valid berikut ini. Jangan menciptakan nama kolom baru.
        Daftar Kolom yang Valid: {columns}
    
        ATURAN PENTING UNTUK EKSTRAKSI KOLOM:
        - Untuk tool 'histogram', 'outliers', atau 'categorical_insights', ekstrak satu nama kolom ke dalam field 'column_name'.
        - Untuk tool 'analyze_target', ekstrak satu nama kolom ke dalam field 'target_column'.
        - Untuk tool 'target-feature-plot', Anda HARUS mengekstrak DUA nama kolom. Kolom yang menjadi fitur utama masuk ke 'column_name', dan kolom yang menjadi target atau pembanding masuk ke 'target_column'.

        CONTOH 1 (Plotting):
        Permintaan Pengguna: "buat plot hubungan antara fitur 'training_hours' dan target 'gender'"
        JSON Output:
        {{
            "tool_name": "target-feature-plot", "reasoning": "...",
            "column_name": "training_hours", "target_column": "gender"
        }}
        
        CONTOH 2 (ML Training):
        Permintaan Pengguna: "latih model untuk prediksi target, tapi buang kolom enrollee_id dan simpan dengan nama 'model_churn_v1'"
        JSON Output:
        {{
            "tool_name": "run_ml_pipeline", "reasoning": "...",
            "target_column": "target",
            "columns_to_drop": ["enrollee_id"],
            "model_name": "model_churn_v1"
        }}

        CONTOH 3A (ML Tuning - Eksplisit):
        Permintaan Pengguna: "coba tuning model 'model_v1' untuk prediksi target dan simpan sebagai 'model_v1_tuned'"
        JSON Output:
        {{
            "tool_name": "run_tuned_ml_pipeline",
            "reasoning": "Pengguna meminta untuk tuning model 'model_v1' dan menyimpannya sebagai nama baru.",
            "target_column": "target",
            "model_name": "model_v1_tuned",
            "baseline_model_name": "model_v1"
        }}
        
        CONTOH 3B (ML Tuning - Implisit/Ambigue):
        Permintaan Pengguna: "coba lakukan tuning pada 'model_v1' untuk kolom target"
        JSON Output:
        {{
            "tool_name": "run_tuned_ml_pipeline",
            "reasoning": "Pengguna meminta untuk tuning 'model_v1'. Saya akan menyimpan hasilnya sebagai 'model_v1_tuned' untuk perbandingan.",
            "target_column": "target",
            "model_name": "model_v1_tuned",
            "baseline_model_name": "model_v1"
        }}

        CONTOH 4 (ML Feature Importance):
        Permintaan Pengguna: "fitur apa yang paling penting dari model 'model_churn_v1'?"
        JSON Output:
        {{
            "tool_name": "get_feature_importance", "reasoning": "...",
            "model_name": "model_churn_v1"
        }}

        CONTOH 5 (ML Prediksi):
        Permintaan Pengguna: "gunakan 'model_churn_v1' untuk memprediksi data ini: {{'gender': 'Male', 'experience': '>20'}}"
        JSON Output:
        {{
            "tool_name": "predict_new_data", "reasoning": "...",
            "model_name": "model_churn_v1",
            "new_data": {{"gender": "Male", "experience": ">20"}}
        }}

        CONTOH 6A (Mengindeks PDF):
        Permintaan Pengguna: "Tolong baca dan siapkan laporan PDF ini untuk ditanyai."
        JSON Output:
        {{
            "tool_name": "index_pdf",
            "reasoning": "Pengguna mengunggah PDF dan ingin mengindeksnya.",
            "pdf_question": null
            // ... (field lain null) ...
        }}

        CONTOH 6B (Bertanya ke PDF yang Sudah Diindeks):
        Permintaan Pengguna: "Dari laporan tadi, apa kesimpulan utamanya?"
        JSON Output:
        {{
            "tool_name": "answer_pdf_question",
            "reasoning": "Pengguna bertanya tentang PDF yang sudah diindeks sebelumnya.",
            "pdf_question": "Apa kesimpulan utama dari laporan?"
            // ... (field lain null) ...
        }}


        {format_instructions}
        Permintaan Pengguna: {user_input}"""
    )
    chain = prompt_template | llm | parser
    try:
        memory = memory_manager.get_or_create_memory(session_id)

        memory_variables = memory.load_memory_variables({})

        return chain.invoke({
            "tools": tools_as_string,
            "user_input": user_prompt,
            "columns": ", ".join(column_list),
            "format_instructions": parser.get_format_instructions(),
            "chat_history": memory_variables.get("chat_history", [])
        })
    except Exception as e:
        return {"error": "Gagal membuat rencana.", "detail": str(e)}
