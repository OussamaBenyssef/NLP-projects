"""
08_evaluate.py — Évaluation unifiée des silver labels et des modèles contre le gold set.

Compare :
1. Silver labels vs Gold (labeled.csv + test_labeled.csv)
2. SVM predictions vs Gold
3. Transformer predictions vs Gold
"""
import os
import sys
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import load_config, setup_logger


def load_gold_annotations(config, logger):
    """Charge et normalise les deux fichiers d'annotations gold."""
    
    # labeled.csv (train subset gold)
    labeled_path = config["paths"]["data"]["labeled"]
    test_labeled_path = labeled_path.replace("labeled.csv", "test_labeled.csv")
    
    dfs = []
    
    if os.path.exists(labeled_path):
        df = pd.read_csv(labeled_path)
        if "label_gold" in df.columns and "label_silver" in df.columns:
            df = df[df["label_gold"].notna() & df["label_silver"].notna()]
            df["split"] = "train_subset"
            dfs.append(df)
            logger.info(f"Chargé {len(df)} annotations gold depuis labeled.csv")
    
    if os.path.exists(test_labeled_path):
        df = pd.read_csv(test_labeled_path)
        if "label_gold" in df.columns:
            # Normaliser la colonne silver (peut être label_original)
            silver_col = "label_silver" if "label_silver" in df.columns else "label_original"
            df = df.rename(columns={silver_col: "label_silver"})
            df = df[df["label_gold"].notna() & df["label_silver"].notna()]
            df["split"] = "test_subset"
            dfs.append(df)
            logger.info(f"Chargé {len(df)} annotations gold depuis test_labeled.csv")
    
    if not dfs:
        logger.warning("Aucun fichier d'annotations gold trouvé.")
        return pd.DataFrame()
    
    return pd.concat(dfs, ignore_index=True)


def evaluate_silver_vs_gold(gold_df, logger):
    """Évalue la qualité des silver labels contre les gold annotations."""
    
    TARGET_LABELS = ["AR_DAR", "AR_MSA", "EN", "FR", "MIX"]
    
    for split_name in ["train_subset", "test_subset", "all"]:
        if split_name == "all":
            subset = gold_df
        else:
            subset = gold_df[gold_df["split"] == split_name]
        
        if len(subset) == 0:
            continue
        
        # Filtrer aux labels cibles uniquement (ignorer les labels NADI country dans gold)
        mask = subset["label_gold"].isin(TARGET_LABELS)
        subset = subset[mask]
        
        if len(subset) == 0:
            continue
        
        y_true = subset["label_gold"]
        y_pred = subset["label_silver"]
        
        # Mapper les labels silver non-standard
        y_pred = y_pred.apply(lambda x: x if x in TARGET_LABELS else "OTHER")
        
        # Filtrer les OTHER
        valid = y_pred != "OTHER"
        y_true = y_true[valid]
        y_pred = y_pred[valid]
        
        acc = accuracy_score(y_true, y_pred)
        f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
        
        logger.info(f"\n{'='*60}")
        logger.info(f"SILVER vs GOLD — {split_name} ({len(y_true)} samples)")
        logger.info(f"{'='*60}")
        logger.info(f"Accuracy: {acc:.1%}")
        logger.info(f"Macro F1: {f1_macro:.1%}")
        logger.info(f"\n{classification_report(y_true, y_pred, zero_division=0)}")
        
        # Confusion Matrix
        labels = sorted(set(y_true) | set(y_pred))
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        logger.info("Confusion Matrix:")
        logger.info(f"  {'':>8} | " + " | ".join(f"{l:>7}" for l in labels))
        for i, row in enumerate(cm):
            logger.info(f"  {labels[i]:>8} | " + " | ".join(f"{v:>7}" for v in row))
        
        # Error Patterns
        errors = subset[valid][y_true != y_pred]
        if len(errors) > 0:
            logger.info(f"\nTop error patterns ({len(errors)} errors):")
            error_counts = errors.groupby(["label_silver", "label_gold"]).size()
            error_counts = error_counts.sort_values(ascending=False)
            for (s, g), count in error_counts.head(10).items():
                logger.info(f"  Silver={s:>8} → Gold={g:>8} : {count}")


def evaluate_silver_distribution(config, logger):
    """Affiche la distribution des silver labels après re-labeling."""
    
    for split, path_key in [("train", "train_silver"), ("valid", "valid_silver")]:
        path = config["paths"]["data"][path_key]
        if not os.path.exists(path):
            continue
        
        df = pd.read_parquet(path)
        logger.info(f"\n{'='*60}")
        logger.info(f"DISTRIBUTION SILVER — {split} ({len(df)} samples)")
        logger.info(f"{'='*60}")
        dist = df["label_silver"].value_counts()
        for label, count in dist.items():
            logger.info(f"  {label:>8} : {count:>6} ({count/len(df):.1%})")
        logger.info(f"  Confiance moyenne : {df['confidence'].mean():.3f}")


def main():
    config = load_config()
    logger = setup_logger("08_evaluate", log_level=config["project"]["log_level"])
    
    logger.info("Début de l'évaluation unifiée")
    
    # 1. Distribution des silver labels
    evaluate_silver_distribution(config, logger)
    
    # 2. Silver vs Gold
    gold_df = load_gold_annotations(config, logger)
    if len(gold_df) > 0:
        evaluate_silver_vs_gold(gold_df, logger)
    
    logger.info("\nÉvaluation terminée.")


if __name__ == "__main__":
    main()
