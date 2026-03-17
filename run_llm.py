from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import json
import os
import re

model_names = [
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "microsoft/phi-2"
]

questions_file = "qtotal.txt"
output_file = "reponses.json"

max_new_tokens = 80
temperature = 0
do_sample = False
max_answer_chars = 150

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

models = {}
tokenizers = {}

for name in model_names:

    print(f"Chargement du modèle {name}...")

    tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        name,
        trust_remote_code=True
    )

    model.to(device)
    model.eval()

    tokenizers[name] = tokenizer
    models[name] = model


with open(questions_file, "r", encoding="utf-8") as f:
    questions = [line.strip() for line in f.readlines() if line.strip()]


def clean_answer(text):

    # supprimer références images
    text = re.sub(r'image|photo|picture', '', text, flags=re.IGNORECASE)

    text = text.strip()

    if len(text) > max_answer_chars:

        truncated = text[:max_answer_chars]

        # couper à la dernière phrase
        last_dot = truncated.rfind(".")
        if last_dot != -1:
            truncated = truncated[:last_dot+1]

        text = truncated

    return text


all_answers = {}

for question in questions:

    print(f"\n❓ {question}")

    all_answers[question] = {}

    for name in model_names:

        tokenizer = tokenizers[name]
        model = models[name]

        # nouveau prompt plus strict
        prompt = f"Réponds en une phrase courte (moins de 150 caractères).\nQuestion: {question}\nRéponse:"

        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
            pad_token_id=tokenizer.eos_token_id
        )

        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)

        answer = generated.replace(prompt, "").strip()

        answer = answer.split("Question:")[0]
        answer = answer.split("User:")[0]

        answer = clean_answer(answer)

        all_answers[question][name] = answer

        print(f"{name} → {answer}")


with open(output_file, "w", encoding="utf-8") as f:
    json.dump(all_answers, f, ensure_ascii=False, indent=2)

print("\n✅ Sauvegardé dans reponses.json")