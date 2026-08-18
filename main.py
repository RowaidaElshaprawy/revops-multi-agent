import sys

from src.agents.supervisor import run_qualification


def run_pipeline(domain: str):
    print(f"\n🚀 RUNNING REVOPS SYSTEM FOR: {domain}")
    state = run_qualification(domain)
    print("\n📋 Audit Logs:")
    for log in state["audit_logs"]:
        print(f" ├─ {log}")
    print(f"\n🎯 Intent: {(state.get('intent_data') or {}).get('top_intent')}")
    print(f"🎯 PyTorch Score: {state.get('pytorch_score')}")
    print(f"🎯 Qualified: {state.get('is_qualified')}")


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "stripe.com"
    run_pipeline(target)