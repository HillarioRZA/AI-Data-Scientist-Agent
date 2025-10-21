import os
import base64
from backend.services.eda import main as eda_main
from backend.services.visualization import main as visualization_main
from backend.services.ml import selector, preprocessor, trainer, evaluator,predictor
from backend.utils.read_csv import _read_csv_with_fallback
import joblib
from backend.services.memory import memory_manager
from backend.services.memory import persistent_memory
from backend.services.rag import parser as rag_parser
from backend.services.rag import vectorizer as rag_vectorizer
from backend.services.rag import retriever as rag_retriever
from backend.services.agent.interpretation import get_interpretation
from typing import Optional,List,Dict
from langchain_google_genai import ChatGoogleGenerativeAI
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

def execute_tool(session_id: str,plan: dict,file_path: Optional[str],original_prompt: str):
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
    tools_need_file = visual_tools + data_tools_eda + ["run_ml_pipeline", "run_tuned_ml_pipeline", "index_pdf"]

    file_path_to_use = file_path
    file_contents = None

    if file_path_to_use is None and tool_to_use in tools_need_file:
        file_key_to_load = "__latest_pdf" if tool_to_use == "index_pdf" else "__latest_csv"
        
        dataset_info = persistent_memory.get_dataset_path(session_id, file_key_to_load)
        
        if dataset_info and os.path.exists(dataset_info['path']):
            file_path_to_use = dataset_info['path']
            print(f"--- [LTM] Memuat file tersimpan '{file_path_to_use}' untuk tool '{tool_to_use}' ---")
        else:
             return {"error": f"Tool '{tool_to_use}' membutuhkan file CSV/PDF, tapi tidak ada file yang diunggah atau tersimpan di LTM untuk sesi ini."}
    
    if file_path_to_use:
        try:
            with open(file_path_to_use, 'rb') as f:
                file_contents = f.read()
        except Exception as e:
            return {"error": f"Gagal membaca file di path: {file_path_to_use}", "detail": str(e)}

    if tool_to_use == "predict_new_data":
        if not model_name: return {"error": "Anda perlu menyebutkan nama model yang ingin digunakan."}
        if not new_data: return {"error": "Tidak ada data baru yang diberikan."}
        
        model_data = persistent_memory.get_model_data(session_id, model_name)
        if not model_data:
             return {"error": f"Model dengan nama '{model_name}' tidak ditemukan untuk sesi ini."}

        try:
            if not os.path.exists(model_data["model_path"]) or not os.path.exists(model_data["preprocessor_path"]):
                return {"error": f"File model atau preprocessor untuk '{model_name}' (path: {model_data['model_path']}) tidak ditemukan di disk."}
            
            model = joblib.load(model_data["model_path"])
            preprocessor_obj = joblib.load(model_data["preprocessor_path"])
        except FileNotFoundError:
            return {"error": f"Model dengan nama '{model_name}' tidak ditemukan."}

        raw_data = predictor.predict_new_data(new_data, model, preprocessor_obj)
        
        return {"plan": plan, "summary": f"Prediksi menggunakan model '{model_name}' adalah: {raw_data.get('prediction')}", "data": raw_data}
    elif tool_to_use == "conversational_recall":
        memory_stm = memory_manager.get_or_create_memory(session_id)
        
        chat_history_str = memory_stm.load_memory_variables({})['chat_history']
        
        full_prompt = f"""Anda adalah AI asisten yang sangat cerdas. Jawab pertanyaan pengguna HANYA berdasarkan RIWAYAT CHAT di bawah ini dan konteks umum yang Anda ketahui tentang sesi ini. Jawab dengan ringkas dan langsung, tanpa format JSON.
        
        RIWAYAT CHAT: {chat_history_str}
        
        PERMINTAAN PENGGUNA: {original_prompt}
        
        JAWABAN:"""

        try:
            answer = llm.invoke(full_prompt).content
            return {"plan": plan, "summary": answer, "data": {"type": "conversational_answer"}}
        except Exception as e:
            return {"error": "Gagal menghasilkan jawaban percakapan.", "detail": str(e)}

    raw_data = None
    image_bytes = None    

    if tool_to_use in visual_tools:
        if tool_to_use == "histogram":
            if not column_name: return {"error": "Nama kolom dibutuhkan untuk histogram."}
            image_bytes = visualization_main.generate_histogram(file_contents, column_name)
        elif tool_to_use == "correlation-heatmap":
            image_bytes = visualization_main.generate_correlation_heatmap(file_contents)
        elif tool_to_use == "missing-value-heatmap":
            image_bytes = visualization_main.generate_missing_value_heatmap(file_contents)
        elif tool_to_use == "target-feature-plot":
            if not column_name or not target_column: return {"error": "Butuh kolom fitur dan target untuk plot ini."}
            image_bytes = visualization_main.generate_target_feature_plot(file_contents, column_name, target_column)

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
            
            df = _read_csv_with_fallback(file_contents)
            if columns_to_drop:
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
            raw_metrics = evaluator.evaluate_model(model, X_test, y_test, problem_type)
            save_name = model_name if model_name else f"model_{session_id}_terakhir"

            session_model_dir = f"saved_models/{session_id}"
            os.makedirs(session_model_dir, exist_ok=True) # Buat folder sesi jika belum ada
            model_path = os.path.join(session_model_dir, f"{save_name}_model.joblib")
            preprocessor_path = os.path.join(session_model_dir, f"{save_name}_preprocessor.joblib")

            joblib.dump(model, model_path)
            joblib.dump(pipeline_object, preprocessor_path)

            raw_data = raw_metrics.copy()
            raw_data.update({
                "problem_type": problem_type,
                "model_name": type_model,
                "best_params": best_params,
                "saved_model_name": save_name,
                "model_path": model_path,
                "preprocessor_path": preprocessor_path
            })

            persistent_memory.save_model_data(session_id, save_name, raw_metrics, model_path, preprocessor_path)
             
        elif tool_to_use == "get_feature_importance":
            if not model_name: return {"error": "Anda perlu menyebutkan nama model yang ingin dianalisis."}

            model_data = persistent_memory.get_model_data(session_id, model_name)
            if not model_data:
                return {"error": f"Model dengan nama '{model_name}' tidak ditemukan untuk sesi ini."}

            try:
                if not os.path.exists(model_data["model_path"]) or not os.path.exists(model_data["preprocessor_path"]):
                    return {"error": f"File model atau preprocessor untuk '{model_name}' (path: {model_data['model_path']}) tidak ditemukan di disk."}
                model = joblib.load(model_data["model_path"])
                preprocessor_obj = joblib.load(model_data["preprocessor_path"])
            except FileNotFoundError:
                return {"error": f"Model dengan nama '{model_name}' tidak ditemukan."}
            
            raw_data = evaluator.get_feature_importance(model, preprocessor_obj)

    elif tool_to_use in data_tools_rag:
        if tool_to_use == "index_pdf":
            if not file_contents:
                return {"error": "Tidak ada file PDF yang diunggah untuk diindeks."}
            
            text_content = rag_parser.parse_pdf(file_contents)

            
            if isinstance(text_content, str) and text_content.startswith("Error:"):
                return {"error": "Gagal mem-parsing PDF.", "detail": text_content}
                
            vector_store = rag_vectorizer.create_vector_store(text_content)
            if isinstance(vector_store, str) and vector_store.startswith("Error:"):
                 return {"error": "Gagal membuat vector store.", "detail": vector_store}
            
            memory_manager.save_vector_store(session_id,vector_store)

            return {"plan": plan, "summary": "Dokumen PDF berhasil diindeks dan siap untuk ditanyai.", "data": None}

        elif tool_to_use == "answer_pdf_question":
            if not pdf_question:
                return {"error": "Tidak ada pertanyaan yang diberikan untuk PDF."}
            
            vector_store = memory_manager.get_vector_store(session_id)
            if not vector_store:
                return {"error": "PDF belum diindeks. Silakan unggah dan indeks PDF terlebih dahulu."}
            
            answer = rag_retriever.get_rag_answer(pdf_question, vector_store, llm)
            
            if isinstance(answer, str) and answer.startswith("Error:"):
                return {"error": "Gagal menjawab pertanyaan dari PDF.", "detail": answer}
            return {"plan": plan, "summary": answer, "data": None}
    else:
        return {"error": f"Tool '{tool_to_use}' tidak dikenali."}

    if image_bytes:
        summary = get_interpretation(session_id,tool_to_use, plan, image_bytes=image_bytes)
        image_b64 = base64.b64encode(image_bytes).decode("utf-8")
        return {
            "plan": plan, "summary": summary,
            "image_base64": image_b64, "image_format": "png"
        }

    if raw_data is not None:
        baseline_metrics_data = None
        if tool_to_use == "run_tuned_ml_pipeline":
             baseline_name = plan.get("baseline_model_name")
             baseline_model_info = persistent_memory.get_model_data(session_id, baseline_name)
             if baseline_model_info:
                 baseline_metrics_data = baseline_model_info.get("metrics")
        summary = get_interpretation(session_id, tool_to_use, raw_data, baseline_metrics=baseline_metrics_data)
        return {"plan": plan, "summary": summary, "data": raw_data}

    if tool_to_use in visual_tools or tool_to_use in data_tools_eda or tool_to_use in data_tools_ml:
        return {"error": f"Gagal mengeksekusi '{tool_to_use}' karena hasil kosong."}

    return {"error": f"Gagal mengeksekusi '{tool_to_use}' karena hasil kosong."}
