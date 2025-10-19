import os
import json
import base64
from typing import Optional,List,Dict
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.output_parsers import JsonOutputParser
from backend.services.eda import main as eda_main
from backend.services.ml import selector, preprocessor, trainer, evaluator,predictor
from langchain.memory import ConversationBufferMemory
import pandas as pd
import joblib
from backend.services.memory import memory_manager
from backend.services.rag import parser as rag_parser
from backend.services.rag import vectorizer as rag_vectorizer
from backend.services.rag import retriever as rag_retriever
load_dotenv()


llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
model_artifacts = {"model": None, "preprocessor": None, "problem_type": None}

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
# Daftar tool yang diperbarui dan lebih lengkap
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
    {"name": "answer_pdf_question", "description": "Gunakan ini untuk menjawab pertanyaan spesifik dari dokumen PDF yang telah diindeks sebelumnya. Butuh 'pdf_question'."}
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
        # ----------------------------------

        # Muat variabel memori
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

def get_interpretation(session_id: str,tool_name: str, tool_output, image_bytes: Optional[bytes] = None, baseline_metrics: Optional[dict] = None) -> str:
    """Fungsi interpretasi universal untuk data dan gambar."""
    if image_bytes:
        # Logika Multimodal untuk Gambar
        image_b64 = base64.b64encode(image_bytes).decode("utf-8")
        message = HumanMessage(content=[
            {"type": "text", "text": f"Anda adalah AI data analyst. Jelaskan insight utama dari gambar {tool_name} ini. Jika ada anomali atau pola menarik, sebutkan. Berikan juga rekomendasi langkah selanjutnya berdasarkan visualisasi ini."},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
        ])
        response = llm.invoke([message])
        return response.content
    else:
        # Logika Teks untuk JSON/Data
        output_str = json.dumps(tool_output, indent=2)

        baseline_str = json.dumps(baseline_metrics, indent=2) if baseline_metrics else "Tidak ada data baseline."

        if tool_name == "run_tuned_ml_pipeline":
             baseline_name = tool_output.get("plan", {}).get("baseline_model_name") # Ambil dari rencana
             baseline_metrics = memory_manager.get_model_metrics(baseline_name) # Ambil dari memori
             if baseline_metrics:
                 baseline_str = json.dumps(baseline_metrics, indent=2, default=str)
        
        # Template prompt spesifik untuk setiap tool data
        prompt_templates = {
            "full-profile": """Anda adalah AI data analyst yang sedang menyajikan temuan utama kepada klien.
            Berdasarkan data profil berikut: {data}.
            Buatlah sebuah ringkasan naratif ('Cerita dari Data'). Sorot 3-4 poin paling krusial seperti masalah kualitas data (missing values), fitur yang distribusinya paling aneh (skewed), dan temuan menarik lainnya.
            Akhiri dengan memberikan 2 rekomendasi konkret untuk langkah analisis selanjutnya.""",
            "skewness": """Berikut adalah data skewness: {data}.
            Identifikasi kolom dengan skewness paling tinggi (positif atau negatif). Jelaskan artinya secara sederhana.
            Jika ada kolom dengan skewness di luar rentang -1 dan 1, berikan rekomendasi spesifik (misal: 'pertimbangkan transformasi logaritma pada kolom X').""",
            "vif": """Berikut adalah hasil perhitungan VIF: {data}.
            Jelaskan apa itu VIF secara singkat. Identifikasi fitur dengan VIF di atas 10 (jika ada) dan jelaskan mengapa ini bisa menjadi masalah (multikolinearitas).
            Berikan rekomendasi yang jelas, seperti 'pertimbangkan untuk menghapus fitur X atau menggabungkannya dengan fitur lain'.""",
            "run_ml_pipeline": """Anda adalah seorang AI data scientist yang sedang melaporkan hasil pemodelan kepada manajer.
            Berikut adalah laporan metrik evaluasi dari model yang baru dilatih: {data}.
            Jelaskan arti dari hasil ini dalam beberapa poin:
            1.  Sebutkan tipe masalah dan model yang digunakan.
            2.  Jelaskan metrik utamanya (Akurasi untuk klasifikasi, R2 Score untuk regresi) dengan bahasa yang mudah dimengerti.
            3.  Berikan kesimpulan akhir tentang seberapa baik performa model tersebut.""",
           "run_tuned_ml_pipeline": """Anda adalah AI data scientist yang melaporkan hasil tuning.
            
            Ini adalah metrik dari model BARU yang sudah di-tuning:
            {data}
            
            Ini adalah metrik dari model SEBELUMNYA (baseline):
            {baseline_data}
            
            Jelaskan hasilnya:
            1. Sebutkan model dan parameter terbaik yang ditemukan.
            2. Bandingkan metrik utamanya (Akurasi/R2 Score) dari model BARU dengan model BASELINE.
            3. Jelaskan dengan angka spesifik (misal: "ada peningkatan akurasi sebesar 1.5%")
            4. Berikan kesimpulan apakah tuning ini berhasil.""",
            
            "get_feature_importance": """Anda adalah seorang analis yang menjelaskan faktor-faktor penting kepada klien. Berikut adalah daftar fitur paling berpengaruh dalam model: {data}.
            
            Sebutkan 3 fitur teratas dan jelaskan kemungkinan artinya dalam konteks bisnis secara sederhana. Contoh: 'Fitur 'masa_berlangganan' paling penting, artinya semakin lama seseorang berlangganan, semakin besar pengaruhnya terhadap prediksi.'"""
        }
        
        template = prompt_templates.get(tool_name, "Berikut adalah hasil analisis: {data}. Ringkas hasilnya.")
        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | llm
        return chain.invoke({"data": output_str, "baseline_data": baseline_str}).content

def run_agent_flow(session_id: str, prompt: str, file_contents: Optional[bytes], dataset_name: Optional[str]):
    """
    Fungsi master yang mengelola seluruh alur agen, kini dengan memori.
    """
    column_list = []
    # (Logika memuat kolom dari memori bisa ditambahkan nanti jika diperlukan)
    if file_contents and dataset_name and dataset_name.endswith('.csv'):
         df = eda_main._read_csv_with_fallback(file_contents)
         if df is not None:
             column_list = df.columns.tolist()

    # 1. Buat Rencana (menggunakan session_id untuk mendapatkan memori)
    #    Fungsi get_agent_plan sekarang menerima session_id
    plan = get_agent_plan(session_id, prompt, column_list)
    if "error" in plan:
        return plan

    # 2. Eksekusi Tool
    #    Fungsi execute_tool juga perlu session_id (misal untuk RAG)
    result = execute_tool(session_id, plan, file_contents)

    # --- SIMPAN INTERAKSI KE MEMORI ---
    if "error" not in result:
        # Ambil ringkasan dari hasil, atau gunakan pesan default
        # Pastikan ada output yang disimpan, terutama untuk tool seperti index_pdf
        agent_response = result.get("summary")
        if agent_response: # Hanya simpan jika ada ringkasan/jawaban
            # Dapatkan objek memori lagi dari manajer
            memory = memory_manager.get_or_create_memory(session_id)
            # Format input/output untuk disimpan
            inputs = {"input": prompt}
            outputs = {"output": agent_response}
            # Simpan ke memori menggunakan save_context
            memory.save_context(inputs, outputs)
            print(f"--- Konteks disimpan ke memori sesi {session_id} ---")
        else:
             print(f"--- Tidak ada output summary untuk disimpan ke memori sesi {session_id} ---")
    # ------------------------------------

    return result

def execute_tool(session_id: str,plan: dict, file_contents: Optional[bytes]):
    """
    Fungsi master tunggal untuk mengeksekusi semua tool, baik visual maupun data.
    """
    tool_to_use = plan.get("tool_name")
    column_name = plan.get("column_name")
    target_column = plan.get("target_column")
    columns_to_drop = plan.get("columns_to_drop")
    model_name = plan.get("model_name")
    new_data = plan.get("new_data")
    baseline_model_name = plan.get("baseline_model_name")
    pdf_question = plan.get("pdf_question")

    if tool_to_use == "target-analysis":
        tool_to_use = "analyze_target"

    if tool_to_use == "categorical-insights":
        tool_to_use = "categorical_insights"

    visual_tools = ["histogram", "correlation-heatmap", "missing-value-heatmap", "target-feature-plot"]
    data_tools_eda = ["describe", "skewness", "outliers", "full-profile", "vif", "analyze_target", "categorical_insights"]
    data_tools_ml = ["run_ml_pipeline", "run_tuned_ml_pipeline", "get_feature_importance"]
    data_tools_rag = ["index_pdf", "answer_pdf_question"]

    if tool_to_use == "predict_new_data":
        if not model_name: return {"error": "Anda perlu menyebutkan nama model yang ingin digunakan."}
        if not new_data: return {"error": "Tidak ada data baru yang diberikan."}
        
        try:
            model = joblib.load(f"saved_models/{model_name}_model.joblib")
            preprocessor_obj = joblib.load(f"saved_models/{model_name}_preprocessor.joblib")
        except FileNotFoundError:
            return {"error": f"Model dengan nama '{model_name}' tidak ditemukan."}

        raw_data = predictor.predict_new_data(new_data, model, preprocessor_obj)
        
        return {"plan": plan, "summary": f"Prediksi menggunakan model '{model_name}' adalah: {raw_data.get('prediction')}", "data": raw_data}

    # Inisialisasi untuk tool lain
    raw_data = None
    image_bytes = None    

    # Eksekusi tool untuk menghasilkan output (gambar atau data)
    if tool_to_use in visual_tools:
        if tool_to_use == "histogram":
            if not column_name: return {"error": "Nama kolom dibutuhkan untuk histogram."}
            image_bytes = eda_main.generate_histogram(file_contents, column_name)
        elif tool_to_use == "correlation-heatmap":
            image_bytes = eda_main.generate_correlation_heatmap(file_contents)
        elif tool_to_use == "missing-value-heatmap":
            image_bytes = eda_main.generate_missing_value_heatmap(file_contents)
        elif tool_to_use == "target-feature-plot":
            if not column_name or not target_column: return {"error": "Butuh kolom fitur dan target untuk plot ini."}
            image_bytes = eda_main.generate_target_feature_plot(file_contents, column_name, target_column)

        if image_bytes is None or isinstance(image_bytes, str):
            return {"error": f"Gagal menghasilkan gambar untuk tool '{tool_to_use}'.", "detail": image_bytes}

    elif tool_to_use in data_tools_eda:
        raw_data = None
        if tool_to_use == "full-profile":
            raw_data = eda_main.run_full_data_profile(file_contents)
        elif tool_to_use == "skewness":
            raw_data = eda_main.get_skewness(file_contents)
        elif tool_to_use == "vif":
            raw_data = eda_main.calculate_vif(file_contents)
        elif tool_to_use == "categorical_insights":
            if not column_name: return {"error": "Nama kolom dibutuhkan."}
            raw_data = eda_main.get_categorical_insights(file_contents, column_name)
        elif tool_to_use == "analyze_target":
            if not column_name: return {"error": "Nama kolom dibutuhkan."}
            raw_data = eda_main.analyze_target(file_contents, column_name)
        elif tool_to_use == "describe":
            raw_data = eda_main.get_csv_description(file_contents)
        elif tool_to_use == "outliers":
            if not column_name: return {"error": "Nama kolom dibutuhkan."}
            raw_data = eda_main.get_outliers(file_contents, column_name)
       
    elif tool_to_use in data_tools_ml:
        if tool_to_use in ["run_ml_pipeline", "run_tuned_ml_pipeline"]:
            if not target_column: return {"error": "Kolom target dibutuhkan."}
            
            df = eda_main._read_csv_with_fallback(file_contents)
            if columns_to_drop:
                # Pastikan kolom yang akan didrop ada di DataFrame untuk menghindari error
                existing_cols_to_drop = [col for col in columns_to_drop if col in df.columns]
                df.drop(columns=existing_cols_to_drop, inplace=True)
                print(f"Dropped columns: {existing_cols_to_drop}")

            df.dropna(subset=[target_column], inplace=True)
            
            problem_type = selector.detect_problem_type(df[target_column])
            X_processed, y, pipeline_object = preprocessor.preprocess_data(df, target_column)
            
            perform_tuning = tool_to_use == "run_tuned_ml_pipeline"
            
            model, X_test, y_test, type_model, best_params = trainer.train_model(
                X_processed, y, problem_type, perform_tuning=perform_tuning
            )
            raw_data = evaluator.evaluate_model(model, X_test, y_test, problem_type)

            save_name = model_name if model_name else "model_terakhir"
            model_path = f"saved_models/{save_name}_model.joblib"
            preprocessor_path = f"saved_models/{save_name}_preprocessor.joblib"
            joblib.dump(model, model_path)
            joblib.dump(pipeline_object, preprocessor_path)

            raw_data.update({
                "problem_type": problem_type,
                "model_name": type_model,
                "best_params": best_params,
                "saved_model_name": save_name
            })

            # --- MODIFIKASI: Gunakan Manajer Memori ---
            memory_manager.save_model_metrics(session_id, save_name, raw_data)
            # -----------------------------------------
            

        
        elif tool_to_use == "get_feature_importance":
            if not model_name: return {"error": "Anda perlu menyebutkan nama model yang ingin dianalisis."}
            try:
                model = joblib.load(f"saved_models/{model_name}_model.joblib")
                preprocessor_obj = joblib.load(f"saved_models/{model_name}_preprocessor.joblib")
            except FileNotFoundError:
                return {"error": f"Model dengan nama '{model_name}' tidak ditemukan."}
            
            raw_data = evaluator.get_feature_importance(model, preprocessor_obj)

    elif tool_to_use in data_tools_rag:
        if tool_to_use == "index_pdf":
            if not file_contents:
                return {"error": "Tidak ada file PDF yang diunggah untuk diindeks."}
            
            text_content = rag_parser.parse_pdf(file_contents)

            print("--- Teks yang Diekstrak dari PDF ---")
            print(text_content[:2000]) # Cetak 2000 karakter pertama
            print("------------------------------------")
            
            if isinstance(text_content, str) and text_content.startswith("Error:"):
                return {"error": "Gagal mem-parsing PDF.", "detail": text_content}
                
            vector_store = rag_vectorizer.create_vector_store(text_content)
            if isinstance(vector_store, str) and vector_store.startswith("Error:"):
                 return {"error": "Gagal membuat vector store.", "detail": vector_store}
            
            memory_manager.save_vector_store(vector_store)
            
            # Khusus untuk index_pdf, tidak ada data/summary interpretasi
            return {"plan": plan, "summary": "Dokumen PDF berhasil diindeks dan siap untuk ditanyai.", "data": None}

        elif tool_to_use == "answer_pdf_question":
            if not pdf_question:
                return {"error": "Tidak ada pertanyaan yang diberikan untuk PDF."}
            
            vector_store = memory_manager.get_vector_store()
            if not vector_store:
                return {"error": "PDF belum diindeks. Silakan unggah dan indeks PDF terlebih dahulu."}
            
            answer = rag_retriever.get_rag_answer(pdf_question, vector_store, llm)
            
            
            if isinstance(answer, str) and answer.startswith("Error:"):
                return {"error": "Gagal menjawab pertanyaan dari PDF.", "detail": answer}
            
            # Jawaban RAG adalah 'summary', tidak ada 'data' mentah
            # Kita tidak perlu interpretasi tambahan dari get_interpretation
            return {"plan": plan, "summary": answer, "data": None}
    else:
        return {"error": f"Tool '{tool_to_use}' tidak dikenali."}

    # Jika tool visual berhasil, kembalikan gambar
    if image_bytes:
        summary = get_interpretation(session_id,tool_to_use, plan, image_bytes=image_bytes)
        image_b64 = base64.b64encode(image_bytes).decode("utf-8")
        return {
            "plan": plan, "summary": summary,
            "image_base64": image_b64, "image_format": "png"
        }

    # Jika tool data berhasil, kembalikan data
    if raw_data is not None:
        baseline_metrics_data = None
        if tool_to_use == "run_tuned_ml_pipeline":
             # Gunakan nama baseline dari plan
             baseline_name = plan.get("baseline_model_name")
             baseline_metrics_data = memory_manager.get_model_metrics(session_id, baseline_name) # Ambil per sesi
        summary = get_interpretation(session_id, tool_to_use, raw_data, baseline_metrics=baseline_metrics_data)
        return {"plan": plan, "summary": summary, "data": raw_data}

    if tool_to_use in visual_tools or tool_to_use in data_tools_eda or tool_to_use in data_tools_ml:
        return {"error": f"Gagal mengeksekusi '{tool_to_use}' karena hasil kosong."}
        
    # Fallback jika terjadi kesalahan logika
    return {"error": f"Gagal mengeksekusi '{tool_to_use}' karena hasil kosong."}

class PlotPlan(BaseModel):
    """Struktur untuk merencanakan visualisasi kustom."""
    plot_type: str = Field(description="Tipe plot, harus salah satu dari: bar, box, histogram, scatter.")
    x_col: str = Field(description="Nama kolom untuk sumbu X.")
    y_col: Optional[str] = Field(default=None, description="Nama kolom untuk sumbu Y.")
    hue_col: Optional[str] = Field(default=None, description="Nama kolom untuk pewarnaan (hue).")
    orientation: str = Field(default='v', description="Orientasi plot, 'v' untuk vertikal, 'h' untuk horizontal.")

# Buat fungsi perencanaan baru, khusus untuk plot
def get_plot_plan(user_prompt: str) -> dict:
    parser = JsonOutputParser(pydantic_object=PlotPlan)
    prompt = ChatPromptTemplate.from_template(
        """Anda adalah asisten yang tugasnya mengekstrak parameter untuk membuat plot dari permintaan pengguna.
        {format_instructions}
        Permintaan Pengguna: {user_input}"""
    )
    chain = prompt | llm | parser
    try:
        return chain.invoke({
            "user_input": user_prompt,
            "format_instructions": parser.get_format_instructions()
        })
    except Exception as e:
        return {"error": "Gagal mengekstrak parameter plot.", "detail": str(e)}

