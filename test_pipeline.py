import sys
import os

# Ensure src module is on path
sys.path.append(os.path.abspath(os.path.curdir))

from src.agents.supervisor import build_revops_graph

def run_test_cases():
    print("="*60)
    print("🚀 INITIALIZING REVOPS MULTI-AGENT GRAPH TEST")
    print("="*60)
    
    app = build_revops_graph()

    # --- Test Case 1: High-value lead (Should complete full pipeline) ---
    print("\n[TEST 1] Testing Qualified Lead Payload (High Score Expected)...")
    qualified_payload = {
        "domain": "stripe.com",
        "raw_features": {
            "company_size": 1500,
            "revenue_tier": 4,
            "tech_match": 1,
            "engagement": 9
        },
        "audit_logs": []
    }
    
    res1 = app.invoke(qualified_payload)
    print(f"-> Domain: {res1.get('domain')}")
    print(f"-> PyTorch Score: {res1.get('pytorch_score'):.4f}")
    print(f"-> Qualified Status: {res1.get('is_qualified')}")
    print(f"-> ICP Reasoning: {res1.get('icp_reasoning')}")
    print(f"-> Retrived RAG Context Count: {len(res1.get('rag_context', []))}")
    print(f"-> Media Spec Prompt: {res1.get('media_asset_info', {}).get('prompt')}")
    print("\nAudit Logs:")
    for log in res1.get("audit_logs", []):
        print(f"  • {log}")

    # --- Test Case 2: Low-value lead (Should trigger Early Exit) ---
    print("\n" + "="*60)
    print("[TEST 2] Testing Low-Value Lead Payload (Early-Exit Expected)...")
    disqualified_payload = {
        "domain": "unknown-tiny-blog.org",
        "raw_features": {
            "company_size": 1,
            "revenue_tier": 0,
            "tech_match": 0,
            "engagement": 0
        },
        "audit_logs": []
    }
    
    res2 = app.invoke(disqualified_payload)
    print(f"-> Domain: {res2.get('domain')}")
    print(f"-> PyTorch Score: {res2.get('pytorch_score'):.4f}")
    print(f"-> Qualified Status: {res2.get('is_qualified')}")
    print(f"-> ICP Reasoning: {res2.get('icp_reasoning')}")
    print("\nAudit Logs:")
    for log in res2.get("audit_logs", []):
        print(f"  • {log}")
        
    print("\n" + "="*60)
    print("✅ PIPELINE EXECUTION TEST COMPLETED SUCCESSFULLY!")
    print("="*60)

if __name__ == "__main__":
    run_test_cases()