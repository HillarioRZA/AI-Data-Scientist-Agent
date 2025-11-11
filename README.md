# Data Whisperer v1.0
## AI-Powered Data Analysis Platform

---
[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 🎯 Overview

**Data Whisperer** adalah platform AI-powered yang mengintegrasikan Exploratory Data Analysis (EDA), Machine Learning (ML), dan Retrieval Augmented Generation (RAG) dalam satu sistem terpadu. Platform ini dirancang untuk memudahkan analisis data dengan bantuan AI Agent yang dapat memahami konteks data dan memberikan insights yang actionable.

### ✨ Core Capabilities
| Area | Fitur Utama |
|------|--------------|
| 🤖 **AI Reasoning Agent** | Memahami intent user dan memilih tools (EDA, ML, RAG) secara otomatis |
| 📊 **Exploratory Data Analysis** | Menjalankan 12+ analisis statistik lengkap dan visualisasi dinamis |
| 🧠 **Machine Learning Automation** | Otomatis mendeteksi problem (klasifikasi/regresi) dan melatih model |
| 📄 **RAG Engine** | Menganalisis PDF dan menjawab pertanyaan berbasis isi dokumen |
| 💬 **Context Memory** | Mengingat sesi percakapan dan hasil analisis sebelumnya |

---

## ✨ Features

### 🤖 AI Agent Services
- **ReAct Architecture**: Reasoning + Acting untuk intelligent decision making
- **Intent Classification**: Automatic tool selection berdasarkan user intent
- **Multimodal Interpretation**: JSON dan image interpretation
- **Data-Aware Agent**: Mencegah halusinasi dengan data context awareness
- **Conversational Interface**: Natural language interaction

### 📊 Exploratory Data Analysis (EDA)
- **Data Description**: Statistical summaries dan data profiling
- **Visualization**: Histograms, heatmaps, correlation matrices
- **Outlier Detection**: Statistical outlier identification
- **Skewness Analysis**: Distribution analysis
- **Categorical Insights**: Categorical variable analysis
- **Target Analysis**: Target variable analysis
- **VIF Analysis**: Variance Inflation Factor calculation
- **Custom Visualization**: User-defined plot generation

### 🧠 Machine Learning Pipeline
- **Automatic Problem Detection**: Classification vs Regression detection
- **Data Preprocessing**: Imputation, encoding, scaling
- **Model Training**: RandomForest dengan hyperparameter tuning
- **Model Evaluation**: Comprehensive metrics calculation
- **Feature Importance**: Explainable AI (XAI) features
- **Model Persistence**: Session-based model storage
- **Prediction**: New data prediction dengan trained models

### 📄 RAG (Retrieval Augmented Generation)
- **PDF Processing**: Text extraction dari PDF documents
- **Vector Store**: FAISS-based document indexing
- **Question Answering**: Context-aware Q&A system
- **Document Analysis**: Multi-document analysis capabilities

### 🧠 Memory Management
- **Short-Term Memory**: Session-based conversation context
- **Long-Term Memory**: Persistent data storage dengan TinyDB
- **Session Management**: Isolated user sessions
- **Data Persistence**: Model dan dataset storage

---

## 🏗️ Architecture

### System Architecture
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │   FastAPI       │    │   AI Services   │
│   (React)       │◄──►│   Backend       │◄──►│   (LangChain)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                              │
                              ▼
                       ┌─────────────────┐
                       │   Core Services │
                       │   - EDA         │
                       │   - ML          │
                       │   - RAG         │
                       │   - Memory      │
                       └─────────────────┘
```

### Service Architecture
- **API Layer**: FastAPI dengan modular routers
- **Service Layer**: Core business logic
- **Memory Layer**: Session dan persistent storage
- **Integration Layer**: External services (LLM, embeddings)

---

## 🧩 Tech Stack
| Layer | Technology |
|--------|-------------|
| Backend | FastAPI |
| AI Orchestration | LangChain (ReAct Pattern) |
| ML/EDA | Scikit-learn, Pandas, Seaborn, Plotly |
| Vectorstore | FAISS / ChromaDB |
| PDF Parsing | PyMuPDF, pdfplumber |
| Memory | ConversationBufferMemory, TinyDB |
| Frontend (Planned) | React + Tailwind |

---

## 🚀 Installation

### Prerequisites
- Python 3.10+
- pip (Python package manager)

### Step 1: Clone Repository
```bash
git clone https://github.com/your-username/data-whisperer.git
cd data-whisperer
```

### Step 2: Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Environment Setup
Create `.env` file:
```env
GOOGLE_API_KEY=your_google_api_key
OPENAI_API_KEY=your_openai_api_key
```

### Step 5: Run Application
```bash
# Development server
uvicorn backend.api.main:app --reload --host 0.0.0.0 --port 8000

# Production server
uvicorn backend.api.main:app --host 0.0.0.0 --port 8000
```

---

## 🚀 Quick Start

### 1. Start the Server
```bash
uvicorn backend.api.main:app --reload
```

### 2. Access the API
- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/

### 3. Upload Data
```python
import requests

# Upload CSV file
with open('data.csv', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/eda/upload',
        files={'file': f}
    )
```

### 4. Run EDA Analysis
```python
# Get data description
response = requests.get('http://localhost:8000/eda/describe')
print(response.json())
```

### 5. Interact with AI Agent
```python
# Ask AI agent
response = requests.post(
    'http://localhost:8000/agent/execute',
    json={
        'session_id': 'your-session-id',
        'prompt': 'Analyze this data and create a visualization'
    }
)
```

---

## 📚 API Documentation

### Base URL
```
http://localhost:8000
```

### Main Endpoints

#### 🏠 Health Check
```http
GET /
```

#### 📊 EDA Endpoints
```http
POST /eda/upload                    # Upload CSV file
GET  /eda/describe                  # Data description
GET  /eda/correlation-heatmap       # Correlation heatmap
GET  /eda/histogram/{column}         # Histogram for column
GET  /eda/outliers                  # Outlier detection
GET  /eda/skewness                  # Skewness analysis
GET  /eda/categorical               # Categorical analysis
GET  /eda/target-analysis           # Target variable analysis
GET  /eda/full-profile              # Complete data profile
```

#### 🤖 Agent Endpoints
```http
POST /agent/execute                 # Execute agent action
POST /agent/visualization           # Create custom visualization
```

#### 🧠 ML Endpoints
```http
POST /ml/run-pipeline               # Run ML pipeline
POST /ml/predict                    # Make prediction
POST /ml/tuned-pipeline             # Run tuned pipeline
GET  /ml/feature-importance         # Get feature importance
GET  /ml/download-model             # Download model artifacts
```

---

## 🧠 Skills Demonstrated

LLM Integration (OpenAI API)

LangChain ReAct Reasoning

Backend Architecture (FastAPI)

Data Science Automation (EDA & ML)

RAG Implementation (FAISS)

Session & Memory Management

Clean Modular Codebase Design

---

## 🧩 Example Output

Task	Output

- “Analyze churn data”	“83% churn pada pelanggan tenure < 2 tahun.”
- “Visualize correlation”	Generates heatmap chart
- “Summarize PDF report”	Extracts and summarizes sections automatically
---

## 🧑‍💻 About the Developer

👋 Built by Reza Pratama

> Informatics Student & AI Engineer Enthusiast
Focused on AI Agents, LLM Integration, and Backend Automation.

---

## 📝 License

MIT License © 2025 — Reza Pratama


---

❤️ Acknowledgments

LangChain

FastAPI

Scikit-learn

Pandas

FAISS

---

💬 Data Whisperer v1.0 — Empowering AI-driven Data Analysis
“An intelligent assistant that speaks the language of data.”

---
