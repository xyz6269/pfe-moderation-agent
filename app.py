import os
import gradio as gr
import spaces
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import transforms
from transformers import AutoTokenizer, BertModel, CLIPModel
from huggingface_hub import hf_hub_download
from groq import Groq
import easyocr
from datetime import datetime


from huggingface_hub import login
login(token=os.environ.get("HF_TOKEN", ""))


# ── Config ────────────────────────────────────────────────────────────
TRANSFORMER_NAME = "UBC-NLP/MARBERTv2"
CLIP_PATH        = "openai/clip-vit-base-patch32"
MODEL_REPO       = "xyz6269/arabic-hateful-memes-model"
MODEL_FILE       = "model.pt"
THRESHOLD        = 0.53
GROQ_API_KEY     = os.environ["GROQ_API_KEY"]

MEMES = [f"img/meme{i}.jpg" for i in range(1, 9)]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(device)

# ── Model ─────────────────────────────────────────────────────────────
class MultiModalHatefulMemeModel(nn.Module):
    def __init__(self, clip_path=CLIP_PATH):
        super().__init__()
        self.text_encoder = BertModel.from_pretrained(TRANSFORMER_NAME)
        self.visual_model = CLIPModel.from_pretrained(clip_path)
        for param in self.visual_model.parameters():
            param.requires_grad = False
        self.text_proj = nn.Sequential(
            nn.Linear(768, 512), nn.LayerNorm(512), nn.GELU(), nn.Linear(512, 512)
        )
        self.image_proj = nn.Sequential(
            nn.Linear(512, 512), nn.LayerNorm(512), nn.GELU(), nn.Linear(512, 512)
        )
        self.classifier = nn.Sequential(
            nn.Linear(1024, 512), nn.GELU(), nn.Dropout(0.5),
            nn.Linear(512, 256),  nn.GELU(), nn.Dropout(0.5),
            nn.Linear(256, 1)
        )

    def forward(self, input_ids, attention_mask, pixel_values):
        text_out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        mask = attention_mask.unsqueeze(-1).float()
        text_features = (text_out.last_hidden_state * mask).sum(1) / mask.sum(1)
        with torch.no_grad():
            vision_out = self.visual_model.vision_model(pixel_values=pixel_values)
            visual_features = self.visual_model.visual_projection(vision_out.pooler_output).float()
        t = self.text_proj(text_features)
        v = self.image_proj(visual_features)
        return self.classifier(torch.cat((t, v), dim=1))

# ── Load ──────────────────────────────────────────────────────────────
print("Loading model from HF Hub...")
model_path = hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILE)
model = MultiModalHatefulMemeModel()
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()
print("Model ready")

tokenizer = AutoTokenizer.from_pretrained(TRANSFORMER_NAME)
reader    = easyocr.Reader(['en', 'ar'], gpu=torch.cuda.is_available())
groq_client = Groq(api_key=GROQ_API_KEY)

clip_transform = transforms.Compose([
    transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                         std=[0.26862954, 0.26130258, 0.27577711])
])

moderation_log = []

# ── Tools ─────────────────────────────────────────────────────────────
@spaces.GPU
def tool_ocr(image: Image.Image) -> str:
    results = reader.readtext(np.array(image))
    text = " ".join([r[1] for r in results]).strip()
    return text if text else "[no text detected]"

@spaces.GPU
def tool_classify(image: Image.Image, text: str) -> float:
    enc = tokenizer(text, return_tensors="pt", padding="max_length",
                    truncation=True, max_length=128)
    input_ids      = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    pixel_values   = clip_transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(input_ids, attention_mask, pixel_values)
    return torch.sigmoid(logits).item()

def tool_llm_judge(text: str, prob: float) -> str:
    prompt = f"""You are a content moderation judge. Analyze the following text extracted from a meme.

Text: {text}
Model hate confidence: {prob:.2f}

Decide:
- REMOVE if it contains hate speech, discrimination, or harmful content
- WARN if it is borderline or context-dependent
- ALLOW if it is harmless

Respond in this exact format:
Decision: [REMOVE/WARN/ALLOW]
Reason: [one sentence explanation in English]"""
    try:
        resp = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}]
        )
        return resp.choices[0].message.content
    except Exception as e:
        print(f"Groq error: {e}")  # add this
        return "Decision: REMOVE\nReason: API unavailable, defaulting to safe removal."

# ── Agent ─────────────────────────────────────────────────────────────
def agent_moderate(image: Image.Image, meme_id: int) -> dict:
    log = {"meme_id": meme_id, "timestamp": datetime.now().strftime("%H:%M:%S")}

    text = tool_ocr(image)
    log["text"] = text

    prob = tool_classify(image, text)
    log["confidence"] = f"{prob*100:.1f}%"
    model_decision = "REMOVE" if prob >= THRESHOLD else "ALLOW"
    log["model_decision"] = f"{model_decision} ({prob*100:.1f}%)"

    if 0.4 <= prob < THRESHOLD:
        llm_raw  = tool_llm_judge(text, prob)
        lines    = llm_raw.split("\n")
        decision = next((l.split(":")[-1].strip() for l in lines if "Decision:" in l), "WARN")
        reason   = next((l.split(":", 1)[-1].strip() for l in lines if "Reason:" in l), llm_raw)
        if decision not in ("REMOVE", "WARN", "ALLOW"):
            decision = "WARN"
        judge = "LLM judge"
    else:
        decision = model_decision
        reason   = "Model confidence is high."
        judge    = "fusion model"

    log["decision"]    = decision
    log["explanation"] = reason
    log["judge"]       = judge
    moderation_log.append(log)
    return log

# ── UI helpers ────────────────────────────────────────────────────────
def get_badge(decision):
    color = {"REMOVE": "#ff4444", "WARN": "#ff9900", "ALLOW": "#00c851"}.get(decision, "#888")
    return f"<div style='background:{color};color:white;padding:12px;border-radius:8px;text-align:center;font-size:18px;font-weight:bold'>{decision}</div>"

def format_log():
    if not moderation_log:
        return "<p style='color:#888;text-align:center'>No decisions yet</p>"
    rows = ""
    for l in moderation_log[-10:]:
        color = {"REMOVE": "#ff4444", "WARN": "#ff9900", "ALLOW": "#00c851"}.get(l["decision"], "#ccc")
        rows += f"<tr><td>{l['timestamp']}</td><td>Meme {l['meme_id']+1}</td><td style='color:{color};font-weight:bold'>{l['decision']}</td><td>{l['confidence']}</td><td>{l['judge']}</td></tr>"
    return f"""<table style='width:100%;color:#ccc;border-collapse:collapse;font-size:14px'>
        <tr style='color:#888;border-bottom:1px solid #333'><th>Time</th><th>Meme</th><th>Decision</th><th>Confidence</th><th>Judge</th></tr>
        {rows}
    </table>"""

def flag_meme(meme_idx):
    image = Image.open(MEMES[meme_idx]).convert("RGB")
    log   = agent_moderate(image, meme_idx)
    return (
        get_badge(log["decision"]),
        log["text"],
        f"Model: {log['model_decision']} | Final: {log['decision']} by {log['judge']}",
        log["explanation"],
        format_log()
    )

# ── CSS ───────────────────────────────────────────────────────────────
css = """
body { background: #0a0a0a !important; }
.gradio-container { background: #0a0a0a !important; max-width: 100% !important; padding: 20px !important; }
.meme-card {
    background: #1a1a1a;
    border-radius: 16px;
    padding: 10px;
    border: 1px solid #2a2a2a;
    box-shadow: 0 2px 12px rgba(0,0,0,0.4);
    margin: 6px;
}
.meme-card img { border-radius: 10px; width: 100%; object-fit: cover; }
.flag-btn { width: 100% !important; margin-top: 8px !important; border-radius: 8px !important; }
.results-panel { background: #1a1a1a; border-radius: 16px; padding: 20px; border: 1px solid #2a2a2a; }
.log-panel { background: #111; border-radius: 12px; padding: 16px; border: 1px solid #222; }
"""

# ── UI ────────────────────────────────────────────────────────────────
with gr.Blocks(css=css, title="Multimodal Hate Speech Detection") as demo:
    gr.Markdown("# Arabic Multimodal Hate Speech Detection\nClick **Flag** on any meme to run the moderation pipeline.\n---")

    buttons = []
    with gr.Row():
        for i in range(4):
            with gr.Column(scale=1, min_width=200, elem_classes="meme-card"):
                gr.Image(value=MEMES[i], show_label=False, interactive=False, height=220)
                btn = gr.Button(f"Flag Meme {i+1}", elem_classes="flag-btn", variant="secondary")
                buttons.append(btn)

    with gr.Row():
        for i in range(4, 8):
            with gr.Column(scale=1, min_width=200, elem_classes="meme-card"):
                gr.Image(value=MEMES[i], show_label=False, interactive=False, height=220)
                btn = gr.Button(f"Flag Meme {i+1}", elem_classes="flag-btn", variant="secondary")
                buttons.append(btn)

    gr.Markdown("---\n### Analysis Result")
    with gr.Row(elem_classes="results-panel"):
        with gr.Column(scale=1):
            result_badge = gr.HTML("<div style='height:52px'></div>")
            result_conf  = gr.Textbox(label="Model vs Final Decision", lines=1)
        with gr.Column(scale=2):
            result_text    = gr.Textbox(label="Extracted Text", lines=2)
            result_explain = gr.Textbox(label="Explanation", lines=3)

    gr.Markdown("---\n### Moderation Log")
    log_display = gr.HTML(elem_classes="log-panel", value="<p style='color:#888;text-align:center'>No decisions yet</p>")

    outputs = [result_badge, result_text, result_conf, result_explain, log_display]
    for i, btn in enumerate(buttons):
        btn.click(fn=lambda idx=i: flag_meme(idx), inputs=[], outputs=outputs)

demo.launch()