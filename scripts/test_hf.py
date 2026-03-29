from transformers import pipeline

print("⏳ Tentative de téléchargement d'un micro-modèle...")
try:
    pipe = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    print("✅ Succès ! Réponse :", pipe("I love AI")[0])
except Exception as e:
    print(f"❌ Erreur Hugging Face : {e}")