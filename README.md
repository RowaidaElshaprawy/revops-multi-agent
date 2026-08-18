# 🤖 Autonomous RevOps Sales Lead Qualification & Outreach Orchestrator
> **An Enterprise Multi-Agent System built with LangGraph, PyTorch, Transformers, ChromaDB, and FastAPI.**

## 📌 Business Overview
An autonomous, multi-agent RevOps engine designed to qualify inbound sales leads in real time, extract intent, compute conversion probabilities, query domain knowledge, generate multi-modal outreach assets, and log metrics—guarded by human-in-the-loop controls.

---

## 🏛️ System Architecture

```text
               ┌─────────────────────────────────────────┐
               │    Inbound Lead (REST API / Streamlit)   │
               └────────────────────┬────────────────────┘
                                    │
                                    ▼
               ┌─────────────────────────────────────────┐
               │   Supervisor Agent (LangGraph State)    │
               └─┬──────────────┬──────────────┬────────┬┘
                 │              │              │        │
                 ▼              ▼              ▼        ▼
          ┌────────────┐  ┌───────────┐  ┌───────────┐┌───────────┐
          │ Agent 1:   │  │ Agent 2:  │  │ Agent 3:  ││ Agent 4:  │
          │ Scraper &  │  │ PyTorch   │  │ ICP ReAct ││ RAG &     │
          │ Transformer│  │ Neural Net│  │ Reasoner  ││ Multi-    │
          │ Intent     │  │ Scorer    │  │ (QLoRA)   ││ Modal     │
          └────────────┘  └───────────┘  └───────────┘└───────────┘