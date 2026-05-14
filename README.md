#🛡️ HarmBench — LLM Robustness & Ethics Evaluation

> ECE Paris – Bachelor 2 | Hackathon Cybersécurité AI 2026  
> Supervisors: M. FORNIER (Technical) | M. VANDAMME (Methodology)

---

## A. Subject Presentation

**Benchmark chosen:** WMDP HarmBench  
**Scientific objective:** Evaluate the resistance of large language models (LLMs) to harmful, unethical, and socially inappropriate prompts across multiple behavioral categories.

**Research question:**  
*To what extent do open-source LLMs (Llama, Qwen, Phi-2) resist harmful solicitations from the HarmBench benchmark, and how do their refusal behaviors differ across prompt categories?*

**Categories tested (HarmBench):**
- Standard Behaviors (direct harmful requests)
- Contextual Behaviors (harmful requests embedded in context)
- Copyright Behaviors
- Multimodal Behaviors

---

## B. Corpus Description

| Field | Details |
|---|---|
| Original source | [HarmBench](https://github.com/centerforaisafety/HarmBench) |
| Total questions used | 60 prompts (experimental subset) |
| Format | `.txt` (questions) + `.json` (raw responses) |
| Categories | Bioweapon, Harassment, General Harm, ChemWeapon, Cybercrime, Misinformation, Copyright, Illegal Act |

**Experimental corpus construction methodology:**
- Selected 60 representative prompts from the HarmBench dataset
- Prompts span multiple harm categories to ensure diversity
- Each prompt submitted identically to all three models (same conditions)

**Categorization used:** HarmBench original taxonomy (Standard / Contextual / Copyright / Multimodal behaviors)

---

## C. LLM Models Tested

| Model | Type | Version | Temperature | Top_p |
|---|---|---|---|---|
| Llama | Open-source (Meta) | Llama-3 | 0.0 | 0.0 |
| Qwen | Open-source (Alibaba) | Qwen2 | 0.0 | 0.0 |
| Phi-2 | Open-source (Microsoft) | Phi-2 | 0.0 | 0.0 |

All models were run under identical conditions to ensure fair comparison.

---

## D. Summary of Results

> ⚠️ Results based on 60 prompts — full results in `results/processed_results.csv`

<img width="1919" height="1032" alt="image" src="https://github.com/user-attachments/assets/3031fc70-8fe2-44a8-8085-70a1f69b5036" />


<img width="1913" height="1086" alt="image" src="https://github.com/user-attachments/assets/38222ddc-098d-4e81-a74a-09a7353316df" />


| Model | Median of the score | Risk level |
|---|---|---|
| Qwen | 10/10 | CRITICAL: The model provides a dangerous answer in the majority of cases.| 
| Phi-2 | 7/10 | HIGH: Significant porosity despite sporadic rejections.| 
| Llama | 5/10 | MODERATE: Unstable behavior, often due to incomplete responses.| 

Indicator : Attack success rate median

| Model | French | English | Spanish |
|---|---|---|---|
| Qwen | 5,2  | 2,1 | 5,3 |
| Phi-2 | 5,4  | 3,5 | 5,1 |
| Llama | 7,5  | 7,2 | 10 |
| Middle Language | 6 | 4,2 | 6,8 |

Indicator: Linguistic safety bias

| Model | Direct request | Context request |
|---|---|---|
| Qwen | 1,8 | 6,5 | 
| Phi-2 | 3,6 | 5,8 | 
| Llama | 7,9 | 8,8 | 

Indicator: Contextual vulnerability gap

📄 Link to the scientific report PDF: *(to be added)*

---

## E. Project Structure

```
Hackaton/
│
├── README.md
├── requirements.txt
│
├── /data/
│   ├── original_benchmark/     # HarmBench original prompts
│   └── experimental_corpus/    # Our 50-prompt subset
│
├── /scripts/
│   ├── run_llm.py              # Main prompt execution script
│   ├── evaluation.py           # Automatic scoring script
│   └── utils.py                # Utility functions
│
├── /results/
│   ├── raw_outputs/            # Raw model responses (JSON/CSV)
│   │   ├── responses_llama.json
│   │   ├── responses_qwen.json
│   │   └── responses_phi2.json
│   ├── processed_results.csv   # Final scored results
│   └── figures/                # Kibana exports (.png)
│
├── questions.txt               # Prompt list (50 questions)
├── reponses.json               # Raw responses (all models)
└── qtotal.txt                  # Summary stats
```

---

## F. How to Reproduce

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure API keys (if needed)
```bash
export HUGGINGFACE_TOKEN=your_token_here
```

### 3. Run the experiment
```bash
python scripts/run_llm.py --model llama --input data/experimental_corpus/ --output results/raw_outputs/
```

### 4. Evaluate results
```bash
python scripts/evaluation.py --input results/raw_outputs/ --output results/processed_results.csv
```

Results will be saved in `results/processed_results.csv`.

---

## G. Credits

**Benchmark reference:**  
- HarmBench: [https://github.com/centerforaisafety/HarmBench](https://github.com/centerforaisafety/HarmBench)

**Models used:**  
- Meta Llama: [https://huggingface.co/meta-llama](https://huggingface.co/meta-llama)  
- Qwen: [https://huggingface.co/Qwen](https://huggingface.co/Qwen)  
- Microsoft Phi-2: [https://huggingface.co/microsoft/phi-2](https://huggingface.co/microsoft/phi-2)

**Contributors:**  
- edouard2303 — [@AdolphFaye](https://github.com/AdolphFaye) - Hocine661

---

*ECE Paris — Bachelor 2 — Hackathon Cybersécurité AI 2026*
