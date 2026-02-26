import os
import sys
import re
import argparse
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix

# Ajouter le dossier parent au PATH pour importer src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import load_config, setup_logger, ensure_dir

def plot_confusion_matrix(y_true, y_pred, labels, out_path, title="Confusion Matrix"):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    ensure_dir(out_path)
    plt.savefig(out_path)
    plt.close()


def compute_script_ratios(texts):
    """Calcule arab_ratio, latin_ratio, cs_ratio pour chaque texte."""
    arab_pat = re.compile(r'[\u0600-\u06FF]+')
    latin_pat = re.compile(r'[a-zA-Z]+')
    
    features = []
    for text in texts:
        tokens = str(text).split()
        if not tokens:
            features.append([0.0, 0.0, 0.0])
            continue
        n = len(tokens)
        arab = sum(1 for t in tokens if arab_pat.search(t))
        latin = sum(1 for t in tokens if latin_pat.search(t))
        arab_r = arab / n
        latin_r = latin / n
        cs_r = min(arab_r, latin_r)
        features.append([arab_r, latin_r, cs_r])
    
    return np.array(features)


def main():
    parser = argparse.ArgumentParser(description="Baseline SVM")
    parser.add_argument("--filter_confidence", type=float, default=0.6, help="Filtrer les données train_silver avec confidence < X")
    args = parser.parse_args()

    config = load_config()
    logger = setup_logger("06_svm", log_level=config["project"]["log_level"])
    
    logger.info("Début de l'étape 6 : Modèle Baseline SVM")
    
    # 1. Chargement des données
    path_train = config["paths"]["data"]["train_silver"]
    path_valid = config["paths"]["data"]["valid_silver"]
    path_test_gold = config["paths"]["data"]["test_gold"]
    
    if not os.path.exists(path_train) or not os.path.exists(path_valid):
        logger.error("Fichiers silver introuvables. Lancez l'étape 4 d'abord.")
        return
        
    df_train = pd.read_parquet(path_train)
    df_valid = pd.read_parquet(path_valid)
    
    # Filtrage confiance
    if args.filter_confidence > 0:
        logger.info(f"Filtrage des données d'entraînement (confiance >= {args.filter_confidence})...")
        len_before = len(df_train)
        df_train = df_train[df_train["confidence"] >= args.filter_confidence]
        logger.info(f"Données conservées : {len(df_train)} / {len_before} ({len(df_train)/len_before:.1%})")
        
    if len(df_train) > 100000:
        logger.info(f"Sous-échantillonnage de df_train de {len(df_train)} à 100000 exemples pour l'entraînement rapide.")
        df_train = df_train.sample(n=100000, random_state=config["project"]["seed"])

    X_train, y_train = df_train["text_norm"], df_train["label_silver"]
    X_valid, y_valid = df_valid["text_norm"], df_valid["label_silver"]
    
    # 2. Pipeline combinée : TF-IDF word + TF-IDF char + script ratios
    logger.info("Construction et entraînement du modèle (features combinées)...")
    ngram_range = tuple(config["training"]["svm"]["ngram_range"])
    max_features = config["training"]["svm"]["max_features"]
    C = config["training"]["svm"]["C"]
    
    pipeline = Pipeline([
        ("features", FeatureUnion([
            ("tfidf_word", TfidfVectorizer(
                analyzer="word",
                ngram_range=ngram_range,
                max_features=max_features,
                sublinear_tf=True
            )),
            ("tfidf_char", TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=(2, 4),
                max_features=max_features,
                sublinear_tf=True
            )),
            ("script_ratios", Pipeline([
                ("compute", FunctionTransformer(compute_script_ratios, validate=False)),
                ("scale", StandardScaler())
            ])),
        ])),
        ("svm", LinearSVC(C=C, random_state=config["project"]["seed"], dual="auto", max_iter=2000, class_weight="balanced"))
    ])
    
    pipeline.fit(X_train, y_train)
    logger.info("Entraînement terminé.")
    
    # 3. Validation
    logger.info("Évaluation sur VALID_SILVER :")
    y_pred_valid = pipeline.predict(X_valid)
    acc_valid = accuracy_score(y_valid, y_pred_valid)
    f1_valid = f1_score(y_valid, y_pred_valid, average="macro")
    
    logger.info(f"Accuracy Validation: {acc_valid:.2%}")
    logger.info(f"Macro F1 Validation: {f1_valid:.2%}")
    logger.info("\n" + classification_report(y_valid, y_pred_valid))
    
    labels_order = sorted(y_valid.unique())
    plot_cm_path = "docs/figures/cm_svm_baseline_valid.png"
    plot_confusion_matrix(y_valid, y_pred_valid, labels_order, plot_cm_path, "SVM - Validation (Silver)")
    logger.info(f"Matrice de confusion générée : {plot_cm_path}")
    
    # 4. Test (si test_gold existe)
    if os.path.exists(path_test_gold):
        logger.info("Évaluation sur TEST_GOLD :")
        df_test = pd.read_parquet(path_test_gold)
        if not df_test.empty and "label_gold" in df_test.columns:
            # S'assurer qu'on évalue sur les phrases annotées
            X_test, y_test = df_test["text_norm"], df_test["label_gold"]
            y_pred_test = pipeline.predict(X_test)
            
            acc_test = accuracy_score(y_test, y_pred_test)
            f1_test = f1_score(y_test, y_pred_test, average="macro")
            logger.info(f"Accuracy Test (Gold): {acc_test:.2%}")
            logger.info(f"Macro F1 Test (Gold): {f1_test:.2%}")
            logger.info("\n" + classification_report(y_test, y_pred_test))
            
            # Matrice test gold
            plot_cm_test_path = "docs/figures/cm_svm_baseline_test_gold.png"
            plot_confusion_matrix(y_test, y_pred_test, labels_order, plot_cm_test_path, "SVM - Test (Gold)")
            logger.info(f"Matrice de confusion Test générée : {plot_cm_test_path}")
    else:
        logger.warning(f"Fichier test_gold ({path_test_gold}) introuvable. Évaluation Test ignorée pour l'instant. (Veuillez faire l'étape 5 d'abord).")
        
    # 5. Sauvegarde
    model_path = config["paths"]["models"]["baseline_svm"]
    ensure_dir(model_path)
    joblib.dump(pipeline, model_path)
    logger.info(f"Modèle sauvegardé dans : {model_path}")
    
    logger.info("Fin de l'étape 6")

if __name__ == "__main__":
    main()
