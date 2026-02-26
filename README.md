# Détection Darija + Code-switching (Darija/FR/EN/MSA)

Pipeline complet et reproductible pour la détection du Darija et du Code-Switching (langues cibles : AR_MSA, AR_DAR, FR, EN, MIX). 

## Structure du Projet

- `data/` : Contient les données brutes, nettoyées, les splits, annotations silver et gold.
  - `raw/nadi2022/` : (À remplir par l'utilisateur avec le dataset NADI 2022).
- `models/` : Modèles entraînés (Baseline, Transformer).
- `src/` : Code utilitaire et normalisation du texte.
- `scripts/` : Scripts d'exécution de la pipeline.
- `docs/` : Rapports et analyses.

## Installation

1. Installer les dépendances :
```bash
pip install -r requirements.txt
```

> **Note relative à HuggingFace :** Le dataset `atlasia/Atlaset` est limité en accès (gated). Veuillez vous authentifier via `huggingface-cli login` ou définir la variable d'environnement `HF_TOKEN` avant de lancer le script de téléchargement, et valider l'accès au dataset sur sa page web.

2. Configurer le projet en modifiant `config.yaml` si nécessaire.

3. Fournir le dataset NADI 2022 :
Placer un fichier `nadi_texts.csv` (colonnes: `id,text,label`) dans le sous-dossier `data/raw/nadi2022/`. Si vous ne l'avez pas ou si vous n'avez que les IDs Twitter, le script 01 fournira des instructions.

## Pipeline d'Exécution

```bash
# Etape 1 : Téléchargement datasets
python scripts/01_download_datasets.py

# Etape 2 : Chargement et Nettoyage
python scripts/02_load_and_clean.py

# Etape 3 : Séparation (Train/Valid/Test)
python scripts/03_split_data.py

# Etape 4 : Auto-annotation (Silver Labels)
python scripts/04_auto_label_silver.py

# Etape 5 : Pack de Validation Manuelle
python scripts/05_manual_validation_pack.py
# -> /!\ Annoter 'data/annotation/to_label.csv' et sauvegarder sous 'data/annotation/labeled.csv'

# Etape 6 : Entraînement Baseline SVM
python scripts/06_train_baseline_svm.py

# Etape 7 : Entraînement Transformer
python scripts/07_train_transformer.py

# Etape 8 : Evaluation
python scripts/08_evaluate.py
```

## Tâches en attente (TODO)
- **NADI 2022 (TODO)** : Si le dataset est fourni sous forme d'IDs Twitter, vous devrez écrire/utiliser un script d'hydratation (rehydrate) pour les récupérer via l'API Twitter.
- **Annotation Humaine (TODO)** : Lors de l'étape 5, le fichier `to_label.csv` est généré. Vous devez annoter manuellement les lignes et l'enregistrer sous `labeled.csv` avant d'entraîner/évaluer les modèles.
