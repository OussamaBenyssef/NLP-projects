# Détection Darija + Code-Switching (Darija / FR / EN / MSA)

Pipeline complet et reproductible pour la détection du Darija marocain et du Code-Switching multilingue.  
**Classes cibles** : `AR_MSA`, `AR_DAR`, `FR`, `EN`, `MIX`.

## Structure du Projet

```
darija_cs_detection/
├── app.py                  # Serveur Flask (chat multilingue + panneau technique)
├── config.yaml             # Paramètres globaux du projet
├── requirements.txt        # Dépendances Python
├── pipeline.ipynb          # Notebook complet de la pipeline NLP
│
├── services/               # Services métier
│   ├── detector_service.py #   Détection langue (Transformer fine-tuné)
│   └── generator_service.py#   Génération via Qwen3.5 (API locale)
│
├── src/                    # Code utilitaire
│   ├── normalizer.py       #   Normalisation texte (arabe, arabizi, emojis)
│   ├── post_processing.py  #   Post-traitement des prédictions
│   └── utils.py            #   Fonctions utilitaires
│
├── scripts/
│   └── rebalance_perclass.py  # Rééquilibrage des classes
│
├── data/
│   └── gold/test_gold.parquet  # Jeu de test annoté manuellement (296 Ko)
│
├── models/
│   └── transformer_final_v2/
│       ├── config.json          # Architecture du modèle (métadonnées)
│       └── tokenizer_config.json# Configuration du tokenizer
│
├── templates/index.html    # Interface web du chat
└── static/                 # Assets frontend (CSS + JS)
```

> **Note :** Les fichiers volumineux (modèles `.safetensors`, `.pkl`, données brutes) sont exclus via `.gitignore`.

## Installation

```bash
pip install -r requirements.txt
```

## Utilisation

### 1. Lancer le serveur de génération (Qwen3.5)

```bash
transformers serve \
    --force-model Qwen/Qwen3.5-397B-A17B \
    --port 8000 \
    --continuous-batching
```

> Au premier lancement, le modèle (~70 Go) est téléchargé automatiquement depuis HuggingFace.

### 2. Lancer l'application Flask

```bash
python app.py
```

L'interface est accessible sur **http://localhost:5001**.

### API

| Endpoint       | Méthode | Description                          |
|----------------|---------|--------------------------------------|
| `/`            | GET     | Interface chat                       |
| `/api/chat`    | POST    | Détection + Génération → JSON        |

**Exemple :**
```bash
curl -X POST http://localhost:5001/api/chat \
  -H "Content-Type: application/json" \
  -d '{"text": "Salam, kifach ndir had l-exercice?"}'
```

## Pipeline NLP

Le notebook `pipeline.ipynb` documente la pipeline complète :

1. **Collecte** — Téléchargement des datasets (Atlaset, NADI 2022, LangID FR/EN)
2. **Nettoyage** — Normalisation du texte (arabe, arabizi, URL, mentions)
3. **Séparation** — Train / Valid / Test (70/15/15)
4. **Auto-annotation** — Silver labels via heuristiques linguistiques
5. **Validation** — Annotation manuelle d'un sous-ensemble (Gold)
6. **Baseline SVM** — Classification TF-IDF + SVM linéaire
7. **Transformer** — Fine-tuning XLM-RoBERTa-base
8. **Évaluation** — Métriques, matrices de confusion, analyse d'erreurs

## Modèles

| Modèle               | Rôle              | F1-macro |
|-----------------------|-------------------|----------|
| SVM (TF-IDF)          | Baseline           | ~0.78    |
| XLM-RoBERTa (fine-tuné) | Détecteur final  | ~0.91    |
| Qwen3.5-397B-A17B     | Génération réponse | —        |

## Auteurs

Projet académique — Master NLP.
