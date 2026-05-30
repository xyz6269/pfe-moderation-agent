---
title: Arabic Hate Speech Detection
emoji: 🛡
colorFrom: red
colorTo: gray
sdk: gradio
sdk_version: "4.44.1"
app_file: app.py
pinned: false
---


# Arabic Multimodal Hate Speech Detection

A multimodal hate speech detection system for memes, combining Arabic text understanding with visual reasoning. The system fuses a fine-tuned MARBERTv2 text encoder with a CLIP ViT-B/32 vision encoder via cross-modal projection, and routes borderline cases to a VLM judge for final moderation decisions.

---

## How It Works

### Pipeline Overview

```
Meme Image
    │
    ├──► EasyOCR (text extraction)
    │         │
    │         ▼
    │    NLLB-200 Translation (EN → AR)
    │         │
    │         ▼
    │    MARBERTv2 (text encoder) ──► text features (512d)
    │                                        │
    └──► CLIP ViT-B/32 (vision encoder) ──► visual features (512d)
                                                    │
                                             Concatenation (1024d)
                                                    │
                                            Binary Classifier
                                                    │
                                          Confidence Score (0–1)
                                                    │
                              ┌─────────────────────┴──────────────────────┐
                         conf >= 0.53                              0.4 <= conf < 0.53
                              │                                            │
                        REMOVE / ALLOW                            VLM Judge (Groq)
                        (fusion model)                           REMOVE / WARN / ALLOW
```

### Components

**Text Encoder — MARBERTv2**
A BERT-based model pre-trained on Arabic dialectal and Modern Standard Arabic. Handles the Arabic text extracted and translated from meme images. Token embeddings are mean-pooled with attention masking and projected to 512 dimensions.

**Vision Encoder — CLIP ViT-B/32**
OpenAI's CLIP vision transformer, frozen during training. Extracts visual semantics from the meme image and projects them to 512 dimensions via CLIP's visual projection head.

**Fusion**
Text and visual features are concatenated into a 1024-dimensional vector and passed through a 3-layer MLP classifier with GELU activations and dropout (0.5), producing a single hate speech probability score.

**Agentic Decision Layer**
The agent applies a confidence-based routing policy:
- `conf >= 0.53` → **REMOVE** (fusion model is confident)
- `conf < 0.4` → **ALLOW** (fusion model is confident)
- `0.4 <= conf < 0.53` → escalate to **VLM Judge** (Groq, meta-llama/llama-4-scout-17b-16e-instruct)

The VLM judge receives the meme image and the extracted text, then returns one of: `REMOVE`, `WARN`, or `ALLOW`, with a one-sentence explanation.

**Fallback policy**: if the Groq API is unavailable, the system defaults to `REMOVE` to err on the side of safety.

---

## Performance

| Metric | Score |
|--------|-------|
| Test Accuracy | 0.7333 |
| AUROC | 0.7830 |
| Macro F1 | 0.71 |
| Not-Hate F1 | 0.79 |
| Hate F1 | 0.62 |
| Decision Threshold | 0.53 |

Evaluated on the Facebook Hateful Memes dataset (900 test samples).

---

## Running Locally

### Requirements

- Python 3.10+
- CUDA-capable GPU recommended (CPU inference is slow)
- A Groq API key (free tier available at [console.groq.com](https://console.groq.com))

### Installation

```bash
git clone https://huggingface.co/spaces/xyz6269/arabic-hate-speech-detection
cd arabic-hate-speech-detection
pip install -r requirements.txt
```

### Environment Variables

```bash
export GROQ_API_KEY="your_groq_api_key_here"
```

### Model Weights

The model weights are hosted on HuggingFace Hub and downloaded automatically at startup from `xyz6269/arabic-hateful-memes-model`. No manual download needed.

### Launch

```bash
python app.py
```

The Gradio interface will be available at `http://127.0.0.1:7860`.

---

## Requirements

```
gradio
spaces
torch
torchvision
transformers
huggingface_hub
easyocr
groq
Pillow
```
* please ignore the requirements.txt file in the repo as that one is different for a reason being that it's what running the model on huggingfaces requires
---

## Dataset

Training data sourced from:
- [Facebook Hateful Memes Dataset](https://ai.facebook.com/tools/hatefulmemes/) via Kaggle (`parthplc/facebook-hateful-meme-dataset`)
- [Hateful Memes Expanded](https://huggingface.co/datasets/limjiayi/hateful_memes_expanded)

Combined dataset: ~9000 training samples after filtering and balancing.

---

## Citation

If you use this system in your research, please cite the accompanying paper (forthcoming, SoftwareX, Elsevier).

---

## License

MIT