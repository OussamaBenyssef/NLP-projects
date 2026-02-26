import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split

# Ajouter le dossier parent au PATH pour importer src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import load_config, setup_logger, ensure_dir

def main():
    config = load_config()
    logger = setup_logger("03_split", log_level=config["project"]["log_level"])
    
    logger.info("Début de l'étape 3 : Split des données")
    
    in_path = config["paths"]["data"]["processed"]
    if not os.path.exists(in_path):
        logger.error(f"Fichier de données {in_path} introuvable. Lancez l'étape 2 d'abord.")
        return
        
    df = pd.read_parquet(in_path)
    logger.info(f"Données chargées : {len(df)} phrases.")
    
    # Ratios
    r_train = config["data"]["splits"]["train"]
    r_valid = config["data"]["splits"]["valid"]
    r_test = config["data"]["splits"]["test"]
    
    seed = config["project"]["seed"]
    
    # 1. On sépare test (test) vs le reste (temp)
    # Stratification sur label_original si disponible
    strat = df["label_original"] if "label_original" in df.columns else None
    try:
        df_temp, df_test = train_test_split(df, test_size=r_test, random_state=seed, stratify=strat)
    except ValueError as e:
        logger.warning(f"Stratification impossible ({e}). Split aléatoire.")
        df_temp, df_test = train_test_split(df, test_size=r_test, random_state=seed)
        
    # 2. On sépare train et valid à partir du reste
    # La proportion de valid par rapport à (train + valid)
    r_valid_relative = r_valid / (r_train + r_valid)
    
    strat_temp = df_temp["label_original"] if "label_original" in df_temp.columns else None
    try:
        df_train, df_valid = train_test_split(df_temp, test_size=r_valid_relative, random_state=seed, stratify=strat_temp)
    except ValueError as e:
        logger.warning(f"Stratification temp impossible ({e}). Split aléatoire.")
        df_train, df_valid = train_test_split(df_temp, test_size=r_valid_relative, random_state=seed)
        
    logger.info(f"Tailles finales - Train: {len(df_train)} ({(len(df_train)/len(df)):.1%}) | Valid: {len(df_valid)} ({(len(df_valid)/len(df)):.1%}) | Test: {len(df_test)} ({(len(df_test)/len(df)):.1%})")
    
    # Sauvegarde
    path_train = config["paths"]["data"]["train_split"]
    path_valid = config["paths"]["data"]["valid_split"]
    path_test = config["paths"]["data"]["test_split"]
    
    for path, df_split in zip([path_train, path_valid, path_test], [df_train, df_valid, df_test]):
        ensure_dir(path)
        df_split.to_parquet(path, index=False)
        
    logger.info("Splits sauvegardés avec succès.")
    logger.info("Fin de l'étape 3")

if __name__ == "__main__":
    main()
