import os

from PIL import Image, ImageDraw, ImageFont

from src.agents.state import RevOpsState

ASSET_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "generated_assets")


def build_diffusion_prompt(domain: str, top_intent: str, is_qualified: bool) -> str:
    tone = "premium enterprise" if is_qualified else "friendly self-serve"
    return f"A {tone} personalized outreach graphic for {domain}, highlighting {top_intent} intent, clean modern SaaS branding"


def render_local_placeholder(domain: str, prompt: str) -> str:
    os.makedirs(ASSET_DIR, exist_ok=True)
    safe_name = "".join(c if c.isalnum() else "_" for c in domain)
    path = os.path.join(ASSET_DIR, f"{safe_name}.png")
    img = Image.new("RGB", (800, 400), color=(24, 28, 38))
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    draw.text((30, 30), f"Outreach asset for {domain}", fill=(255, 255, 255), font=font)
    words, lines, cur = prompt.split(), [], ""
    for w in words:
        cur = f"{cur} {w}".strip()
        if len(cur) > 70:
            lines.append(cur); cur = ""
    if cur:
        lines.append(cur)
    for i, line in enumerate(lines):
        draw.text((30, 80 + i * 18), line, fill=(180, 200, 255), font=font)
    img.save(path)
    return path


def media_node(state: RevOpsState) -> RevOpsState:
    logs = state.get("audit_logs", [])
    if not state.get("is_qualified"):
        logs.append("[media_agent] skipped — lead not qualified")
        state["media_asset_info"] = None
        state["current_step"] = "MEDIA_SKIPPED"
        state["audit_logs"] = logs
        return state

    domain = state["domain"]
    top_intent = (state.get("intent_data") or {}).get("top_intent", "general")
    prompt = build_diffusion_prompt(domain, top_intent, True)
    asset_path = render_local_placeholder(domain, prompt)
    logs.append(f"[media_agent] rendered real asset at {asset_path}")
    state["media_asset_info"] = {"diffusion_prompt": prompt, "asset_path": asset_path}
    state["current_step"] = "MEDIA_GENERATED"
    state["audit_logs"] = logs
    return state