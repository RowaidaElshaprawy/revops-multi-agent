from src.agents.supervisor import build_revops_graph
from src.agents.state import RevOpsState

def run_pipeline(domain: str, raw_features: list):
    print(f"\n==================================================")
    print(f"🚀 RUNNING 5-AGENT REVOPS SYSTEM FOR: {domain}")
    print(f"==================================================")

    graph = build_revops_graph()

    initial_state: RevOpsState = {
        "domain": domain,
        "raw_features": raw_features,
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

    final_state = graph.invoke(initial_state)

    print("\n📋 System Audit Logs:")
    for log in final_state["audit_logs"]:
        print(f"  ├─ {log}")

    print("\n🎯 Complete Multi-Agent Outputs:")
    print(f"  ├─ Domain: {final_state['domain']}")
    print(f"  ├─ Intent Detected: {final_state.get('intent_data', {}).get('top_intent')}")
    print(f"  ├─ PyTorch Score: {final_state['pytorch_score']}")
    print(f"  ├─ ICP Qualified: {final_state['is_qualified']}")
    
    if final_state.get("rag_context"):
        print(f"\n  📚 ChromaDB RAG Context Retrieved:")
        for idx, doc in enumerate(final_state["rag_context"], 1):
            print(f"     {idx}. {doc}")

    if final_state.get("media_asset_info"):
        print(f"\n  🎨 Multi-Modal Diffusion Asset:")
        print(f"     ├─ Prompt: {final_state['media_asset_info']['diffusion_prompt']}")
        print(f"     └─ File Spec: {final_state['media_asset_info']['asset_path']}")

if __name__ == "__main__":
    # Test high-value qualified lead (should trigger all 5 agents)
    run_pipeline(domain="stripe.com", raw_features=[0.85, 0.90, 0.80, 0.70])