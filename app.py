import streamlit as st

from src.agents.supervisor import run_qualification

st.set_page_config(page_title="RevOps AI Multi-Agent Dashboard", layout="wide")
st.title("🤖 RevOps Agentic AI: Lead Qualification & Outreach Hub")

domain_input = st.sidebar.text_input("Company Domain", value="stripe.com")

if st.sidebar.button("Run Multi-Agent Pipeline", type="primary"):
    with st.spinner("Scraping site and running the full pipeline..."):
        state = run_qualification(domain_input)

    if state.get("blocked"):
        st.error(f"Blocked: {state.get('block_reason')}")
    else:
        col1, col2, col3 = st.columns(3)
        col1.metric("Detected Intent", (state.get("intent_data") or {}).get("top_intent", "N/A"))
        col2.metric("PyTorch Score", f"{state.get('pytorch_score', 0):.2f}")
        col3.metric("Status", "✅ QUALIFIED" if state.get("is_qualified") else "❌ DISQUALIFIED")

        tab1, tab2, tab3, tab4 = st.tabs(["🧠 ICP Reasoning", "📚 RAG Context", "🎨 Media Asset", "📋 Audit Logs"])
        with tab1:
            st.write(state.get("icp_reasoning"))
        with tab2:
            for doc in state.get("rag_context") or []:
                st.info(f"📄 {doc}")
        with tab3:
            media = state.get("media_asset_info")
            if media:
                st.success(f"Prompt: {media['diffusion_prompt']}")
                st.image(media["asset_path"])
            else:
                st.info("No media asset generated (lead not qualified).")
        with tab4:
            for log in state.get("audit_logs", []):
                st.text(log)