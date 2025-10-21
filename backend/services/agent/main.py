# File: backend/services/agemt/main.py
import os
import json
import base64
from typing import Optional,List,Dict
from pydantic import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from backend.utils.read_csv import _read_csv_with_fallback
import joblib
from backend.services.memory import memory_manager
from backend.services.memory import persistent_memory
from backend.services.agent.plan import get_agent_plan
from backend.services.agent.execute_tools import execute_tool

def run_agent_flow(session_id: str, prompt: str, new_file_path: Optional[str], new_dataset_name: Optional[str]):
    column_list = []
    file_path_to_use = new_file_path
    file_type = None

    if new_file_path and new_dataset_name:
        if new_dataset_name.endswith('.csv'):
            file_type = 'csv'

            try:
                with open(new_file_path, 'rb') as f:
                    contents = f.read()
                df = _read_csv_with_fallback(contents)
                if df is not None:
                    column_list = df.columns.tolist()

                persistent_memory.save_dataset_path(session_id, "__latest_csv", new_file_path)
            except Exception as e:
                print(f"Gagal membaca file CSV baru untuk kolom: {e}")
        
        elif new_dataset_name.endswith('.pdf'):
            file_type = 'pdf'
            persistent_memory.save_dataset_path(session_id, "__latest_pdf", new_file_path)

    elif not new_file_path:
        dataset_info = persistent_memory.get_dataset_path(session_id, "__latest_csv")
        if dataset_info and os.path.exists(dataset_info['path']):
            try:
                file_path_to_use = dataset_info['path']
                file_type = 'csv'
                with open(file_path_to_use, 'rb') as f:
                    csv_contents_bytes = f.read()
                df = _read_csv_with_fallback(csv_contents_bytes)
                if df is not None:
                    column_list = df.columns.tolist()
            except Exception as e:
                print(f"Gagal memuat kolom dari file CSV di LTM: {e}")

    plan = get_agent_plan(session_id, prompt, column_list)
    if "error" in plan:
        return plan

    result = execute_tool(session_id, plan, file_path_to_use,prompt)

    if "error" not in result:
        agent_response = result.get("summary")
        if agent_response:
            memory_stm = memory_manager.get_or_create_memory(session_id)
            inputs = {"input": prompt}
            outputs = {"output": agent_response}
            memory_stm.save_context(inputs, outputs)
            print(f"--- [STM] Konteks disimpan ke cache memori sesi {session_id} ---")
            persistent_memory.save_chat_history(session_id, memory_stm)
        else:
             print(f"--- Tidak ada output summary untuk disimpan ke memori sesi {session_id} ---")

    return result


class PlotPlan(BaseModel):
    plot_type: str = Field(description="Tipe plot, harus salah satu dari: bar, box, histogram, scatter.")
    x_col: str = Field(description="Nama kolom untuk sumbu X.")
    y_col: Optional[str] = Field(default=None, description="Nama kolom untuk sumbu Y.")
    hue_col: Optional[str] = Field(default=None, description="Nama kolom untuk pewarnaan (hue).")
    orientation: str = Field(default='v', description="Orientasi plot, 'v' untuk vertikal, 'h' untuk horizontal.")

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