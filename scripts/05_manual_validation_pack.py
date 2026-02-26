import os
import sys
import argparse
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report

# Ajouter le dossier parent au PATH pour importer src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import load_config, setup_logger, ensure_dir

def export_samples(config, logger):
    """Exporte des échantillons pour l'annotation humaine."""
    path_train_silver = config["paths"]["data"]["train_silver"]
    path_test = config["paths"]["data"]["test_split"]
    
    path_to_label = config["paths"]["data"]["to_label"] # train sample
    # Fichier test à annoter :
    path_test_to_label = os.path.join(os.path.dirname(path_to_label), "test_to_label.csv")
    
    nb_train_samples = config["data"]["validation_samples"]
    nb_test_samples = config["data"]["test_gold_samples"]
    seed = config["project"]["seed"]
    
    if os.path.exists(path_to_label) and os.path.exists(path_test_to_label):
        logger.info("Les fichiers à annoter existent déjà. Saute l'export.")
        return
        
    logger.info("Génération des fichiers à annoter manuellement...")
    
    # 1. Export Train Validation Sample
    if os.path.exists(path_train_silver):
        df_train = pd.read_parquet(path_train_silver)
        # Échantillonnage diversifié (stratifié par label silver)
        try:
            sample_train = df_train.groupby("label_silver", group_keys=False).apply(
                lambda x: x.sample(int(nb_train_samples/len(df_train['label_silver'].unique())), random_state=seed)
            ).copy()
        except:
            sample_train = df_train.sample(nb_train_samples, random_state=seed).copy()
            
        sample_train["label_gold"] = ""
        cols_out = ["id", "text_raw", "text_norm", "label_silver", "confidence", "label_gold"]
        ensure_dir(path_to_label)
        sample_train[cols_out].to_csv(path_to_label, index=False)
        logger.info(f"Échantillon train ({len(sample_train)} lignes) exporté vers : {path_to_label}")
    else:
        logger.warning(f"Fichier {path_train_silver} introuvable.")
        
    # 2. Export Test Gold Sample
    if os.path.exists(path_test):
        df_test = pd.read_parquet(path_test)
        # Échantillonnage stratifié si possible sur l'original
        strat = df_test["label_original"] if "label_original" in df_test.columns else None
        try:
            from sklearn.model_selection import train_test_split
            _, sample_test = train_test_split(df_test, test_size=min(nb_test_samples, len(df_test)), stratify=strat, random_state=seed)
        except:
            sample_test = df_test.sample(min(nb_test_samples, len(df_test)), random_state=seed)
            
        sample_test = sample_test.copy()
        sample_test["label_gold"] = ""
        cols_out_test = [c for c in ["id", "text_raw", "text_norm", "label_original", "label_gold"] if c in sample_test.columns]
        ensure_dir(path_test_to_label)
        sample_test[cols_out_test].to_csv(path_test_to_label, index=False)
        logger.info(f"Échantillon test ({len(sample_test)} lignes) exporté vers : {path_test_to_label}")
    else:
        logger.warning(f"Fichier {path_test} introuvable.")

def import_and_evaluate(config, logger):
    """Importe les annotations manuelles, évalue la qualité silver et crée test_gold."""
    path_labeled = config["paths"]["data"]["labeled"] # train annotation
    path_test_labeled = os.path.join(os.path.dirname(path_labeled), "test_labeled.csv")
    
    path_test_gold = config["paths"]["data"]["test_gold"]
    
    labels_ok = True
    
    # 1. Évaluation Train (Silver vs Gold)
    if not os.path.exists(path_labeled):
        logger.warning(f"Fichier annotté {path_labeled} introuvable. Veuillez annoter 'to_label.csv' et le renommer en 'labeled.csv'.")
        labels_ok = False
    else:
        df_gold = pd.read_csv(path_labeled)
        df_gold = df_gold[df_gold["label_gold"].notna() & (df_gold["label_gold"] != "")]
        if not df_gold.empty:
            df_gold["label_gold"] = df_gold["label_gold"].str.upper().str.strip()
            # on filtre si label_silver n'est pas nan non plus
            df_eval = df_gold[df_gold["label_silver"].notna() & df_gold["label_gold"].notna()]
            if not df_eval.empty:
                acc = accuracy_score(df_eval["label_gold"], df_eval["label_silver"])
                f1 = f1_score(df_eval["label_gold"], df_eval["label_silver"], average='macro')
                logger.info("=== EVALUATION SILVER vs GOLD (Train Valid. Sub-split) ===")
                logger.info(f"Accuracy : {acc:.2%}")
                logger.info(f"Macro F1 : {f1:.2%}")
                logger.info("\n" + classification_report(df_eval["label_gold"], df_eval["label_silver"]))
        else:
            logger.warning("Le fichier `labeled.csv` est vide ou n'a pas été annoté.")
            labels_ok = False
            
    # 2. Import Test Gold
    if not os.path.exists(path_test_labeled):
        logger.warning(f"Fichier annotté {path_test_labeled} introuvable. Veuillez annoter 'test_to_label.csv' et le renommer en 'test_labeled.csv'.")
        labels_ok = False
    else:
        df_test_gold = pd.read_csv(path_test_labeled)
        df_test_gold = df_test_gold[df_test_gold["label_gold"].notna() & (df_test_gold["label_gold"] != "")]
        if not df_test_gold.empty:
            df_test_gold["label_gold"] = df_test_gold["label_gold"].str.upper().str.strip()
            
            # 2b. Enrichissement avec mix_to_label.csv (annotations MIX supplémentaires)
            path_mix_labeled = os.path.join(os.path.dirname(path_labeled), "mix_to_label.csv")
            if os.path.exists(path_mix_labeled):
                df_mix = pd.read_csv(path_mix_labeled)
                df_mix = df_mix[df_mix["label_gold"].notna() & (df_mix["label_gold"] != "")]
                if not df_mix.empty:
                    df_mix["label_gold"] = df_mix["label_gold"].str.upper().str.strip()
                    # Garder uniquement les colonnes communes
                    common_cols = [c for c in df_test_gold.columns if c in df_mix.columns]
                    df_mix_aligned = df_mix[common_cols].copy()
                    # Ajouter les colonnes manquantes avec NaN
                    for c in df_test_gold.columns:
                        if c not in df_mix_aligned.columns:
                            df_mix_aligned[c] = pd.NA
                    # Fusionner en dédupliquant par id
                    existing_ids = set(df_test_gold["id"].values)
                    df_mix_new = df_mix_aligned[~df_mix_aligned["id"].isin(existing_ids)]
                    if len(df_mix_new) > 0:
                        df_test_gold = pd.concat([df_test_gold, df_mix_new[df_test_gold.columns]], ignore_index=True)
                        logger.info(f"Enrichi avec {len(df_mix_new)} annotations depuis mix_to_label.csv")
                    else:
                        logger.info("Tous les IDs de mix_to_label.csv sont déjà dans le gold set.")
                    logger.info(f"Distribution gold enrichi: {df_test_gold['label_gold'].value_counts().to_dict()}")
            
            ensure_dir(path_test_gold)
            df_test_gold.to_parquet(path_test_gold, index=False)
            logger.info(f"Fichier de test gold exporté ({len(df_test_gold)} lignes) vers : {path_test_gold}")
        else:
            logger.warning("Le fichier `test_labeled.csv` est vide ou n'a pas été annoté.")
            labels_ok = False
            
    if not labels_ok:
        logger.error("Vérifiez les annotations. L'étape 5 nécessite une validation humaine.")

def main():
    parser = argparse.ArgumentParser(description="Pack de Validation Manuelle")
    parser.add_argument("--export", action="store_true", help="Génère les CSV à annoter")
    parser.add_argument("--import_eval", action="store_true", help="Importe les CSV annotés et évalue")
    args = parser.parse_args()

    config = load_config()
    logger = setup_logger("05_validation", log_level=config["project"]["log_level"])
    
    logger.info("Début de l'étape 5 : Pack Validation")
    
    # Par défaut on fait les deux / selon si les fichiers existent ou pas
    if not args.export and not args.import_eval:
        path_labeled = config["paths"]["data"]["labeled"]
        path_test_labeled = os.path.join(os.path.dirname(path_labeled), "test_labeled.csv")
        
        if os.path.exists(path_labeled) and os.path.exists(path_test_labeled):
            logger.info("Les fichiers annotés existent déjà, mode IMPORT activé par défaut.")
            import_and_evaluate(config, logger)
        else:
            logger.info("Les fichiers annotés n'existent pas encore, mode EXPORT activé par défaut.")
            export_samples(config, logger)
    else:
        if args.export:
            export_samples(config, logger)
        if args.import_eval:
            import_and_evaluate(config, logger)
            
    logger.info("Fin de l'étape 5")

if __name__ == "__main__":
    main()
