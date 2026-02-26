"""
05b_generate_annotation_pack.py — Génère un nouveau pack d'annotation équilibré.

Produit 2 fichiers CSV :
  - data/annotation/to_label.csv       (échantillon train, 500 lignes)
  - data/annotation/test_to_label.csv   (échantillon test, 500 lignes)

Chaque fichier contient des colonnes :
  id, text_raw, text_norm, label_silver, cs_ratio, label_gold

L'annotateur doit remplir la colonne "label_gold" avec :
  AR_DAR, AR_MSA, FR, EN, ou MIX
"""
import os
import sys
import pandas as pd
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import load_config, setup_logger, ensure_dir


def stratified_sample(df, label_col, n_total, min_per_class=30, seed=42):
    """Échantillonnage stratifié avec minimum garanti par classe."""
    rng = np.random.RandomState(seed)
    classes = df[label_col].value_counts()
    n_classes = len(classes)
    
    # Garantir min_per_class pour chaque classe
    samples = []
    remaining = n_total
    
    for cls in classes.index:
        cls_df = df[df[label_col] == cls]
        n_cls = min(min_per_class, len(cls_df), remaining)
        samples.append(cls_df.sample(n=n_cls, random_state=seed))
        remaining -= n_cls
    
    # Distribuer le reste proportionnellement
    if remaining > 0:
        already_sampled_ids = set(pd.concat(samples)["id"] if samples else pd.Series())
        leftover = df[~df.index.isin(pd.concat(samples).index)]
        
        if len(leftover) > 0:
            # Sur-représenter MIX dans le reste (utile pour améliorer l'évaluation)
            mix_leftover = leftover[leftover[label_col] == "MIX"]
            other_leftover = leftover[leftover[label_col] != "MIX"]
            
            n_mix_extra = min(len(mix_leftover), remaining // 3)
            n_other = remaining - n_mix_extra
            
            if n_mix_extra > 0:
                samples.append(mix_leftover.sample(n=n_mix_extra, random_state=seed+1))
            if n_other > 0 and len(other_leftover) > 0:
                samples.append(other_leftover.sample(n=min(n_other, len(other_leftover)), random_state=seed+2))
    
    result = pd.concat(samples, ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)
    return result


def main():
    config = load_config()
    logger = setup_logger("05b_annotation", log_level=config["project"]["log_level"])
    seed = config["project"]["seed"]
    
    logger.info("Génération du nouveau pack d'annotation")
    
    # Charger les splits silver
    train_path = config["paths"]["data"]["train_silver"]
    test_path = config["paths"]["data"].get("test_silver", "data/silver/test_silver.parquet")
    
    if not os.path.exists(train_path):
        logger.error(f"Fichier {train_path} introuvable. Lancez les étapes 04 + 04c d'abord.")
        return
    
    df_train = pd.read_parquet(train_path)
    
    # Pour le test, utiliser le test silver s'il existe, sinon valid silver
    if os.path.exists(test_path):
        df_test = pd.read_parquet(test_path)
    else:
        df_test = pd.read_parquet(config["paths"]["data"]["valid_silver"])
        logger.info("Utilisation du valid_silver comme source de test")
    
    # Colonnes à conserver
    keep_cols = ["id", "text_raw", "text_norm", "label_silver", "cs_ratio"]
    
    # S'assurer que les colonnes existent
    for col in keep_cols:
        if col not in df_train.columns:
            if col == "cs_ratio":
                df_train[col] = 0.0
                df_test[col] = 0.0
            elif col == "text_raw":
                df_train[col] = df_train["text_norm"]
                df_test[col] = df_test["text_norm"]
    
    # === TRAIN ANNOTATION PACK ===
    logger.info(f"Distribution train silver: {df_train['label_silver'].value_counts().to_dict()}")
    
    df_train_sample = stratified_sample(
        df_train, "label_silver", n_total=500, min_per_class=50, seed=seed
    )
    df_train_sample = df_train_sample[keep_cols].copy()
    df_train_sample["label_gold"] = ""  # À remplir par l'annotateur
    
    train_out = config["paths"]["data"]["to_label"]
    ensure_dir(train_out)
    df_train_sample.to_csv(train_out, index=False)
    
    dist_train = df_train_sample["label_silver"].value_counts().to_dict()
    logger.info(f"Échantillon train exporté ({len(df_train_sample)} lignes): {dist_train}")
    logger.info(f"  → {train_out}")
    
    # === TEST ANNOTATION PACK ===
    logger.info(f"Distribution test silver: {df_test['label_silver'].value_counts().to_dict()}")
    
    # Pour le test, plus de diversité sur MIX
    df_test_sample = stratified_sample(
        df_test, "label_silver", n_total=500, min_per_class=30, seed=seed+10
    )
    df_test_sample = df_test_sample[keep_cols].copy()
    df_test_sample["label_gold"] = ""
    
    test_out = config["paths"]["data"].get("test_to_label", "data/annotation/test_to_label.csv")
    if isinstance(test_out, str) and not test_out.endswith(".csv"):
        test_out = "data/annotation/test_to_label.csv"
    ensure_dir(test_out)
    df_test_sample.to_csv(test_out, index=False)
    
    dist_test = df_test_sample["label_silver"].value_counts().to_dict()
    logger.info(f"Échantillon test exporté ({len(df_test_sample)} lignes): {dist_test}")
    logger.info(f"  → {test_out}")
    
    logger.info("Pack d'annotation généré avec succès.")
    logger.info("Prochaine étape : annoter label_gold dans les 2 fichiers CSV")


if __name__ == "__main__":
    main()
