import os
import sys
import pandas as pd
from datasets import load_dataset
from pathlib import Path

# Ajouter le dossier parent au PATH pour importer src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import load_config, setup_logger, ensure_dir

def download_atlaset(config, logger):
    out_path = config["paths"]["data"]["atlaset"]
    ensure_dir(out_path)
    
    if os.path.exists(out_path):
        logger.info(f"Atlaset déjà présent : {out_path}")
        return

    logger.info("Téléchargement de Atlaset depuis HuggingFace...")
    try:
        ds = load_dataset("atlasia/Atlaset", split="train")
        df = ds.to_pandas()
        
        # On va garder toutes les colonnes brutes
        # Export
        df.to_parquet(out_path, index=False)
        logger.info(f"Atlaset sauvegardé ({len(df)} lignes) : {out_path}")
    except Exception as e:
        logger.error(f"Erreur lors du téléchargement de Atlaset: {e}")

def download_fr_en(config, logger):
    out_path = config["paths"]["data"]["fr_en"]
    ensure_dir(out_path)
    
    if os.path.exists(out_path):
        logger.info(f"Dataset FR/EN déjà présent : {out_path}")
        return

    logger.info("Téléchargement du dataset langid_fr_en depuis HuggingFace...")
    try:
        # On charge toutes les splits disponibles et on combine
        ds = load_dataset("papluca/language-identification")
        
        dfs = []
        for split_name in ds.keys():
            df_split = ds[split_name].to_pandas()
            # Filtrer uniqement les langues "fr" et "en"
            df_filt = df_split[df_split['labels'].isin(['fr', 'en'])].copy()
            # Renommer le label pour correspondre au format du projet
            df_filt['labels'] = df_filt['labels'].str.upper() # 'FR' ou 'EN'
            dfs.append(df_filt)
            
        df_combined = pd.concat(dfs, ignore_index=True)
        # On peut renommer 'text' et 'labels' -> 'text', 'label'
        df_combined = df_combined.rename(columns={'text': 'text_raw', 'labels': 'label_original'})
        
        # Exporter
        df_combined.to_parquet(out_path, index=False)
        logger.info(f"Dataset FR/EN sauvegardé ({len(df_combined)} lignes) : {out_path}")
    except Exception as e:
        logger.error(f"Erreur lors du téléchargement de langid_fr_en: {e}")

def verifier_nadi(config, logger):
    nadi_dir = config["paths"]["data"]["nadi"]
    nadi_csv = os.path.join(nadi_dir, "nadi_texts.csv")
    ensure_dir(nadi_csv)
    
    if not os.path.exists(nadi_csv):
        logger.warning(f"Le fichier NADI n'a pas été trouvé à l'emplacement : {nadi_csv}")
        with open(os.path.join(nadi_dir, "README_NADI.txt"), "w", encoding="utf-8") as f:
            f.write("INSTRUCTIONS NADI\n")
            f.write("================\n")
            f.write("Veuillez placer un fichier nommé 'nadi_texts.csv' dans ce dossier.\n")
            f.write("Le fichier doit contenir au minimum les colonnes suivantes :\n")
            f.write("- id: identifiant du tweet/texte\n")
            f.write("- text_raw (ou text): le contenu brut du texte\n")
            f.write("- label_original (ou label): le dialecte ou pays (ex: 'MSA', 'Morocco')\n\n")
            f.write("Si vous n'avez que les IDs Twitter, vous devrez écrire un script pour ré-hydrater les tweets.\n")
        logger.info("Fichier README_NADI.txt généré avec les instructions dans le dossier.")
    else:
        logger.info(f"Fichier NADI trouvé : {nadi_csv}")

def main():
    config = load_config()
    logger = setup_logger("01_download", log_level=config["project"]["log_level"])
    
    logger.info("Début de l'étape 1 : Téléchargement des datasets")
    download_atlaset(config, logger)
    download_fr_en(config, logger)
    verifier_nadi(config, logger)
    logger.info("Fin de l'étape 1")

if __name__ == "__main__":
    main()
