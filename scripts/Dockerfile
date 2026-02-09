# Image de base Python
FROM python:3.9-slim

# Définition du répertoire de travail dans le conteneur
WORKDIR /app

# Installation des dépendances système nécessaires
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Copie et installation des bibliothèques Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copie du reste des scripts
COPY . .

# Commande par défaut (sera écrasée par docker-compose)
CMD ["python", "benchmark_hate_speech.py"]