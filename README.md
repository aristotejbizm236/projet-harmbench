# 🛡️ Analyse de Sûreté des LLM (HarmBench)

Ce projet implémente un laboratoire d'audit pour évaluer la résistance des modèles d'IA face à des requêtes malveillantes.

## 🏗️ Architecture
- **ELK Stack** (Elasticsearch, Logstash, Kibana) pour la visualisation.
- **Docker Compose** pour l'isolation complète de l'environnement.
- **Python / Transformers** pour l'inférence des modèles.

## 📈 Résultats Clés
- **Mistral-Small** : Excellente gestion des refus éthiques.
- **OPT-125M** : Vulnérabilité critique identifiée (absence de filtres).
- 
