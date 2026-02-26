"""
04b_train_dar_msa_classifier.py — Entraîne un classifieur DAR/MSA sur les annotations gold.

Ce classifieur est utilisé par 04_auto_label_silver.py pour améliorer la distinction
entre l'arabe dialectal marocain (Darija) et l'arabe standard (MSA) dans les cas
où les heuristiques lexicales ne suffisent pas.

Approche : TF-IDF (character n-grams 2-5) + LinearSVC
- Les char n-grams capturent les patterns morphologiques du Darija :
  - Préfixes verbaux : كي/كا (présent), غ/غا (futur)
  - Particules : ديال (possession), هاد (démonstratif), واش (interrogatif)
- Entraîné sur 755 échantillons gold (385 DAR, 370 MSA)
- 5-fold CV accuracy : ~88%
"""
import os
import sys
import joblib
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import load_config, setup_logger, ensure_dir


def load_gold_arabic(config, logger):
    """Charge tous les échantillons gold AR_DAR et AR_MSA."""
    labeled_path = config["paths"]["data"]["labeled"]
    test_labeled_path = labeled_path.replace("labeled.csv", "test_labeled.csv")
    
    dfs = []
    for path in [labeled_path, test_labeled_path]:
        if os.path.exists(path):
            df = pd.read_csv(path)
            if "label_gold" in df.columns and "text_norm" in df.columns:
                arabic = df[df["label_gold"].isin(["AR_DAR", "AR_MSA"])][["text_norm", "label_gold"]]
                dfs.append(arabic)
                logger.info(f"Chargé {len(arabic)} échantillons arabes depuis {os.path.basename(path)}")
    
    if not dfs:
        raise ValueError("Aucun fichier d'annotations gold trouvé")
    
    gold = pd.concat(dfs, ignore_index=True).dropna(subset=["text_norm"])
    logger.info(f"Total: {len(gold)} échantillons ({gold['label_gold'].value_counts().to_dict()})")
    return gold


def train_classifier(gold, logger):
    """Entraîne et évalue le classifieur DAR/MSA."""
    X = gold["text_norm"].values
    y = gold["label_gold"].values
    
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(2, 5),
            max_features=20000,
            sublinear_tf=True
        )),
        ("svm", LinearSVC(C=1.0, max_iter=5000, random_state=42))
    ])
    
    # Cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(pipe, X, y, cv=cv, scoring="accuracy")
    logger.info(f"5-fold CV Accuracy: {scores.mean():.1%} ± {scores.std():.1%}")
    logger.info(f"Per-fold: {[f'{s:.1%}' for s in scores]}")
    
    # Train on full dataset
    pipe.fit(X, y)
    
    # Classification report on train (sanity check)
    y_pred = pipe.predict(X)
    logger.info(f"Train report:\n{classification_report(y, y_pred)}")
    
    return pipe


def main():
    config = load_config()
    logger = setup_logger("04b_dar_msa", log_level=config["project"]["log_level"])
    
    logger.info("Entraînement du classifieur DAR/MSA")
    
    # 1. Charger les données gold
    gold = load_gold_arabic(config, logger)
    
    # 2. Entraîner le classifieur
    classifier = train_classifier(gold, logger)
    
    # 3. Sauvegarder le modèle
    out_path = config["paths"]["models"].get("dar_msa_classifier", "models/dar_msa_classifier.pkl")
    ensure_dir(out_path)
    joblib.dump(classifier, out_path)
    logger.info(f"Classifieur sauvegardé dans {out_path}")
    
    logger.info("Terminé.")


if __name__ == "__main__":
    main()
