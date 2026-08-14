import streamlit as st
from src.agents.supervisor import build_revops_graph
from src.agents.state import RevOpsState

st.set_page_config(page_title="RevOps AI Multi-Agent Dashboard", layout="wide")

st.title("🤖 RevOps Agentic AI: Lead Qualification & Outreach Hub")
st.markdown("---")

# Sidebar - Controls
st.sidebar.header("Input Target Prospect")
domain_input = st.sidebar.text_input("Company Domain", value="stripe.com")

st.sidebar.subheader("Normalized Tabular Features")
f1 = st.sidebar.slider("Company Size Index", 0.0, 1.0, 0.85)
f2 = st.sidebar.slider("Revenue Scale Index", 0.0, 1.0, 0.90)
f3 = st.sidebar.slider("Tech Stack Match", 0.0, 1.0, 0.80)
f4 = st.sidebar.slider("Inbound Activity Level", 0.0, 1.0, 0.70)

if st.sidebar.button("Run Multi-Agent Pipeline", type="primary"):
    with st.spinner("Executing 5-Agent LangGraph Workflow..."):
        graph = build_revops_graph()
        
        initial_state: RevOpsState = {
            "domain": domain_input,
            "raw_features": [f1, f2, f3, f4],
            "scraped_text": None,
            "intent_data": None,
            "pytorch_score": None,
            "icp_reasoning": None,
            "is_qualified": None,
            "rag_context": None,
            "media_asset_info": None,
            "current_step": "START",
            "audit_logs": []
        }

        res = graph.invoke(initial_state)

    intent_info = res.get("intent_data") or {}
    media_info = res.get("media_asset_info") or {}

    # Metrics Columns layout
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Detected Intent", intent_info.get("top_intent", "N/A"))
    with col2:
        st.metric("PyTorch Prob Score", res.get("pytorch_score", 0.0))
    with col3:
        status_label = "✅ QUALIFIED" if res.get("is_qualified") else "❌ DISQUALIFIED"
        st.metric("Final Status", status_label)

    st.markdown("---")
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "🧠 ReAct Reasoning", 
        "📚 ChromaDB RAG Context", 
        "🎨 Multi-Modal Vision", 
        "📋 Agent Audit Logs"
    ])

    with tab1:
        st.subheader("Agent 3: Qualitative ICP Evaluation")
        st.code(res.get("icp_reasoning", "No qualitative reasoning generated."))

    with tab2:
        st.subheader("Agent 4: Retrieved Knowledge Case Studies")
        if res.get("rag_context"):
            for doc in res["rag_context"]:
                st.info(f"📄 {doc}")
        else:
            st.warning("No RAG context retrieved.")

    with tab3:
        st.subheader("Agent 5: Generated Diffusion Prompt")
        if media_info:
            st.success(f"Prompt: {media_info.get('diffusion_prompt', 'N/A')}")
            st.text(f"Asset File Spec: {media_info.get('asset_path', 'N/A')}")
        else:
            st.info("No media asset generated for this run.")

    with tab4:
        st.subheader("LangGraph State Audit Logs")
        for log in res.get("audit_logs", []):
            st.text(log)
