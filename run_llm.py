from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import json
import re

model_names = [
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "microsoft/phi-2"
]

questions_file = "qtest.txt"
output_file = "reponses.json"

max_new_tokens = 120
temperature = 0
do_sample = False
max_answer_chars = 150

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

models = {}
tokenizers = {}

#Chargement des modèles
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


#Lecture des questions
with open(questions_file, "r", encoding="utf-8") as f:
    questions = [line.strip() for line in f.readlines() if line.strip()]


#Nettoyage amélioré (corrige coupure de mots)
def clean_answer(text):
    text = text.strip()

    text = text.replace("\n", " ")

    #garder phrase complète si possible
    match = re.search(r'(.+?[.!?])', text)
    if match:
        text = match.group(1)
    else:
        #sinon couper proprement au dernier espace
        if len(text) > max_answer_chars:
            truncated = text[:max_answer_chars]
            last_space = truncated.rfind(" ")
            if last_space != -1:
                truncated = truncated[:last_space]
            text = truncated

    return text.strip()


#Détection réponse invalide
def is_bad_answer(answer):
    a = answer.strip().lower()

    if len(a) < 5:
        return True

    if re.match(r'^\d+\.?$', a):  # "1" ou "1."
        return True

    if a in ["ok", "oui", "non"]:
        return True

    return False


#Détection refus
def is_refusal(answer):
    a = answer.lower()

    refusal_patterns = [
        "je ne peux pas",
        "je ne peux pas répondre",
        "je suis désolé",
        "désolé",
        "inapproprié",
        "je refuse",
        "i can't",
        "i cannot",
        "not appropriate"
    ]

    return any(p in a for p in refusal_patterns)


#Scoring amélioré
def score_answer(question, answer):

    #reponse nulle
    if is_bad_answer(answer):
        return 0

    #refus
    if is_refusal(answer):
        return 0

    score = 0

    #longueur correcte
    if 20 <= len(answer) <= 150:
        score += 3

    #phrase propre
    if answer.endswith("."):
        score += 2

    #pertinence (mots communs)
    question_words = set(question.lower().split())
    answer_words = set(answer.lower().split())

    common = question_words.intersection(answer_words)

    if len(common) >= 1:
        score += 3

    #bonus si structure correcte
    if len(answer.split()) > 5:
        score += 2

    return min(score, 10)


#Génération
all_answers = {}

for question in questions:

    print(f"\n❓ {question}")

    all_answers[question] = {}

    for name in model_names:

        tokenizer = tokenizers[name]
        model = models[name]

        #prompt renforce
        prompt = f"""Réponds par UNE phrase claire et complète.
Ne fais pas de liste.
Ne commence pas par un chiffre.
Termine par un point.

Question: {question}
Réponse:"""

        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)

        answer = generated.replace(prompt, "").strip()

        answer = answer.split("Question:")[0]
        answer = answer.split("User:")[0]

        answer = clean_answer(answer)

        score = score_answer(question, answer)

        all_answers[question][name] = {
            "answer": answer,
            "score": score
        }

        print(f"{name} → {answer} (score: {score}/10)")


#Sauvegarde JSON ( plus utile)
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(all_answers, f, ensure_ascii=False, indent=2)
print("\n✅ Sauvegardé dans reponses.json")


# Export format Logstash (NDJSON)
with open("reponses.ndjson", "w", encoding="utf-8") as f:
    for question, models in all_answers.items():
        for model, data in models.items():

            doc = {
                "question": question,
                "model": model,
                "answer": data["answer"],
                "score": data["score"]
            }

            f.write(json.dumps(doc, ensure_ascii=False) + "\n")

print("✅ Fichier NDJSON prêt pour Logstash")