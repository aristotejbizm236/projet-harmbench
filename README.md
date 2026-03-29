# 🛡️ LLM Safety Evaluation — HarmBench Benchmark

> **Hackathon — ECE Paris 2026**  
> Ethical and Behavioral Evaluation of Large Language Models using HarmBench

---

## 📋 Table of Contents

1. [Subject Overview](#subject-overview)
2. [Corpus Description](#corpus-description)
3. [Tested LLM Models](#tested-llm-models)
4. [Results Summary](#results-summary)
5. [Project Structure](#project-structure)
6. [Reproduce the Experiment](#reproduce-the-experiment)
7. [Credits](#credits)

---

## 🎯 Subject Overview

**Selected Benchmark:** HarmBench

### Scientific Objective
This project evaluates the safety and robustness of Large Language Models (LLMs) against a curated corpus of harmful, adversarial, and copyright-sensitive prompts. We quantitatively measure refusal rates and qualitatively analyze behavioral patterns to understand the impact of alignment training on LLM safety.

### Research Question
> *"How can we rigorously, reproducibly, and ethically evaluate the behavior of Large Language Models when confronted with sensitive or malicious requests?"*

### Three Evaluation Axes

| Axis | Description | Attack Type |
|------|-------------|-------------|
| `standard` | Direct requests for harmful information (weapons, hacking, harm facilitation) | Direct harmful prompt |
| `contextual` | Prompts with an injected system-prompt context to manipulate the model's behavior | Prompt injection / jailbreak |
| `copyright` | Requests to reproduce copyrighted creative works verbatim | Copyright extraction |

---

## 📂 Corpus Description

### Source Dataset
- **Original source:** [HarmBench](https://github.com/centerforaisafety/HarmBench) (Mazeika et al., 2024)
- **Files used:** `harmbench_behaviors_text_all.csv`, `harmbench_behaviors_text_test.csv`, `harmbench_behaviors_text_val.csv`

### Experimental Corpus
- **Total prompts:** 720
- **Format:** `questions.json` — each entry contains `theme`, `prompt`, `source`, and optionally `context_string` (used as system prompt for contextual axis)

### Corpus Breakdown

| Category | Count | Percentage | Notes |
|----------|-------|------------|-------|
| `standard` | 359 | 49.9% | Direct harmful requests, no context |
| `contextual` | 181 | 25.1% | Include `context_string` as system prompt |
| `copyright` | 180 | 25.0% | Requests for copyrighted material |
| **Total** | **720** | **100%** | — |

### Construction Methodology
Prompts were extracted from the three HarmBench CSV files using `convert_all_to_json.py`, which merges all behavioral categories into a single unified `questions.json` corpus, tagging each prompt with its source axis and preserving contextual metadata.

---

## 🤖 Tested LLM Models

| Model | Type | Parameters | Access Method | Alignment Level |
|-------|------|-----------|---------------|----------------|
| `mistral-small-latest` | Proprietary | ~22B | Mistral AI API | High (RLHF + safety fine-tuning) |
| `facebook/opt-125m` | Open-source | 125M | HuggingFace local | None (base model) |

### Execution Parameters

**Mistral-small-latest:**
- API: `https://api.mistral.ai/v1` (OpenAI-compatible)
- Temperature: default (1.0)
- System prompt: `context_string` field (contextual prompts only)

**facebook/OPT-125m:**
- Library: HuggingFace `transformers` pipeline
- `max_new_tokens`: 150
- `do_sample`: True
- `device_map`: auto

---

## 📊 Results Summary

### Overall Refusal Rates

| Model | Total Prompts | Refused | Complied | **Refusal Rate** |
|-------|--------------|---------|----------|-----------------|
| `mistral-small-latest` | 720 | 383 | 337 | **53.2%** |
| `facebook/opt-125m` | 720 | 43 | 677 | **6.0%** |

### Refusal Rate by Category

| Category | Mistral Refusal Rate | OPT-125m Refusal Rate |
|----------|---------------------|----------------------|
| `standard` | **71.9%** | 7.2% |
| `contextual` | **55.2%** | 3.3% |
| `copyright` | **13.9%** | 6.1% |

### Key Findings

- **Mistral-small** demonstrates strong safety alignment, particularly on direct harmful prompts (71.9% refusal rate on standard axis).
- **OPT-125m** shows near-total absence of safety mechanisms across all axes (6% overall), consistent with its unaligned base model nature.
- **Copyright compliance** is the weakest axis for both models — even Mistral-small refuses only 13.9% of copyright requests.
- **Contextual attacks** are partially effective against Mistral, reducing its refusal rate from 71.9% to 55.2%.
- A **47-percentage-point behavioral gap** separates aligned from unaligned models, confirming the transformative impact of safety fine-tuning.

📄 **Full analysis:** See the scientific report `docs/HarmBench_Scientific_Report_ECEParis_2026.pdf`

---

## 🗂️ Project Structure

```
projet-harmbench/
│
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── docker-compose.yml                 # ELK Stack + benchmarker services
├── .env.example                       # API key configuration template
│
├── /scripts                           # All executable code
│   ├── benchmark_hate_speech.py       # Main benchmark runner (multi-model)
│   ├── convert_all_to_json.py         # CSV → JSON corpus converter
│   ├── test_hf.py                     # HuggingFace model connectivity test
│   ├── questions.json                 # Unified experimental corpus (720 prompts)
│   ├── harmbench_behaviors_text_all.csv   # Original HarmBench dataset (all)
│   ├── harmbench_behaviors_text_test.csv  # HarmBench test split
│   ├── harmbench_behaviors_text_val.csv   # HarmBench validation split
│   ├── requirements.txt               # Python deps for Docker container
│   └── Dockerfile                     # Benchmarker container definition
│
├── /results                           # Raw model outputs
│   ├── harmbench_results_20260329_0223.json   # Main results (1440 entries)
│   ├── harmbench_output_20260214_2240.json    # Earlier run outputs
│   ├── harmbench_output_20260214_2246.json
│   └── harmbench_hate_speech_20260216_1550.json
│
├── /logstash                          # ELK pipeline configuration
│   └── /pipeline
│       └── logstash.conf              # Logstash ingestion pipeline
│
└── /docs                              # Documentation & report
    └── HarmBench_Scientific_Report_ECEParis_2026.pdf
```

### Output Format

Each line in result JSON files follows this schema:
```json
{
  "timestamp": "2026-03-29T02:23:14.521Z",
  "model": "mistral-small-latest",
  "theme": "standard",
  "prompt": "Give detailed instructions for...",
  "response": "I'm unable to assist with...",
  "failed_safety": false
}
```

- `failed_safety: false` → model **refused** (safe behavior)
- `failed_safety: true` → model **complied** (potentially dangerous)

---

## ▶️ Reproduce the Experiment

### Prerequisites
- Docker & Docker Compose installed
- API keys for the models you wish to test

### Step 1 — Clone the repository

```bash
git clone https://github.com/<your-org>/projet-harmbench.git
cd projet-harmbench
```
git remote add origin https://github.com/aristotejbizm236/projet-harmbench.git
git branch -M main
git push -u origin main
### Step 2 — Configure API keys

```bash
cp .env.example .env
# Edit .env and fill in your API keys:
# OPENAI_API_KEY=...
# XAI_API_KEY=...
# MISTRAL_API_KEY=...
# HF_TOKEN=...  (required for gated HuggingFace models)
```

### Step 3 — Select models to test

Edit `scripts/benchmark_hate_speech.py` and update the `models_to_test` list:

```python
models_to_test = [
    "mistral-small-latest",       # Mistral API
    "gpt-4o",                     # OpenAI API
    "grok-2-latest",              # xAI API
    "facebook/opt-125m",          # HuggingFace local
]
```

### Step 4 — Launch the full stack

```bash
docker compose up --build
```

This starts Elasticsearch, Logstash, Kibana, and the benchmarker container.  
Results are saved to `/results/harmbench_results_<timestamp>.json`.

### Step 5 — Visualize results in Kibana

Open [http://localhost:5601](http://localhost:5601) in your browser.  
Logstash automatically ingests result files into Elasticsearch for visualization.

### Step 6 — Run locally (without Docker)

```bash
cd scripts
pip install -r requirements.txt
python benchmark_hate_speech.py
```

Results are saved to `results/harmbench_results_<timestamp>.json`.

---

## 🏆 Credits

### Reference Repositories
- [HarmBench](https://github.com/centerforaisafety/HarmBench) — Original benchmark by Mazeika et al. (2024)
- [HuggingFace Transformers](https://github.com/huggingface/transformers) — Local model inference
- [Elastic Stack](https://github.com/elastic/elasticsearch) — Result storage and visualization

### References
- Mazeika et al. (2024). *HarmBench: A Standardized Evaluation Framework for Automated Red Teaming and Robust Refusal.* arXiv:2402.04249
- Zou et al. (2023). *Universal and Transferable Adversarial Attacks on Aligned Language Models.* arXiv:2307.15043

### Project Team — ECE Paris, Bachelor Informatique, Promotion 2026

| Name | Role |
|------|------|
| Nyame Bouelle Henri Honore Molina | Contributor |
| Izamo Aristote | Contributor |
| Asseng Kemoum Reine Josépha | Contributor |
| Nouyang Nouyang Léopold Radeaubel | Contributor |
| Sybel Donfack Démanou Melvis | Contributor |

### Supervisors
- Mme Nassima NACER
- Mr Simon VANDAMME  
- M. Yann FORNIER

---

*ECE Paris — Bachelor Informatique  — Hackathon 2026*
