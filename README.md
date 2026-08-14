# 🤖 RevOps Multi-Agent Sales Qualification & Outreach Engine

An enterprise-grade, autonomous multi-agent RevOps system built with **LangGraph**, **FastAPI**, **Streamlit**, **PyTorch**, **ChromaDB**, and **MLflow**. The system automates inbound sales qualification, web scraping, predictive conversion modeling, qualitative ICP evaluation, vector database RAG search, and multi-modal asset generation.

---

## 🎓 Course Module Application Matrix

This project directly translates each course module into a production system component:

| Module | Topic / Focus | System Implementation | Key Code File(s) |
| :--- | :--- | :--- | :--- |
| **Module 1** | **State Schema & Architecture** | Designed RevOpsState typed memory structure passed cleanly between LangGraph state nodes. | src/agents/state.py |
| **Module 2** | **Web Scraping & Intent Analysis** | Scrapes target domain content and classifies commercial buying intent signals. | src/agents/scraper_agent.py |
| **Module 3** | **Predictive Neural Networks** | Computes quantitative lead conversion probability using a PyTorch classification model. | src/agents/scoring_agent.py |
| **Module 4** | **ReAct & Prompt Engineering** | Implements ReAct (Reasoning + Action) framework for qualitative ICP fit reasoning. | src/agents/icp_agent.py |
| **Module 5** | **Vector Search & RAG Knowledge** | Seeds ChromaDB vector store with case studies & objection battlecards for semantic context retrieval. | src/agents/rag_agent.py |
| **Module 6** | **Multi-Modal Vision Assets** | Constructs Stable Diffusion / ControlNet prompt structures to auto-render personalized pitch graphics. | src/agents/media_agent.py |
| **Module 7** | **Guardrails, MLOps & Deployment** | Implements adversarial prompt injection defense, MLflow tracking, LangGraph supervisor, FastAPI REST, & Streamlit UI. | src/utils/guardrails.py<br>src/agents/supervisor.py<br>src/api.py<br>app.py |

---

## 📂 Project Structure

```text
revops-multi-agent/
├── app.py                     # Streamlit interactive dashboard UI [Module 7]
├── simulate_production.py     # End-to-end integration & security simulation [Module 7]
├── requirements.txt           # Environment dependencies
├── README.md                  # System architecture & documentation
├── .gitignore                 # Exclusion configuration for ML artifacts/secrets
└── src/
    ├── api.py                 # FastAPI REST endpoint (/api/v1/qualify-lead) [Module 7]
    ├── agents/
    │   ├── state.py           # RevOpsState memory schema [Module 1]
    │   ├── scraper_agent.py   # Web scraper & intent classification [Module 2]
    │   ├── scoring_agent.py   # PyTorch quantitative classifier [Module 3]
    │   ├── icp_agent.py       # ReAct qualitative reasoning engine [Module 4]
    │   ├── rag_agent.py       # ChromaDB vector store retriever [Module 5]
    │   ├── media_agent.py     # Multi-modal visual asset generator [Module 6]
    │   └── supervisor.py      # LangGraph state machine orchestrator [Module 7]
    └── utils/
        └── guardrails.py      # Prompt safety interceptor & MLflow logger [Module 7]
```

---

## 🛠️ Quickstart & Execution Guide

### 1. Setup Virtual Environment & Dependencies
```bash
source venv/Scripts/activate  # Windows Git Bash
pip install -r requirements.txt
```

### 2. Launch FastAPI REST Server
```bash
uvicorn src.api:app --reload --port 8000
```
* **Interactive Docs**: Open `http://127.0.0.1:8000/docs`

### 3. Launch Streamlit Interactive UI
```bash
streamlit run app.py
```
* **Dashboard**: Open `http://localhost:8501`

### 4. Run Production Integration Simulation
```bash
python simulate_production.py
```

### 5. Launch MLflow Experiment Dashboard
```bash
mlflow ui --port 5000
```
* **MLflow UI**: Open `http://localhost:5000`

---

## 🛡️ Security & Quality Assurance

* **Adversarial Injection Guardrail**: Intercepts jailbreaks and prompt override attempts (returns HTTP 400).
* **Early-Exit Router**: Saves compute by terminating pipeline execution when lead quantitative score falls below 0.3.
* **MLflow Metrics Tracking**: Logs conversion probability scores, qualification decisions, and RAG precision/faithfulness evaluations.

---

## 📜 License
This project is licensed under the MIT License.
