import os
import sys
import pandas as pd
from tqdm import tqdm
from datasets import Dataset

# Ajouter le dossier parent au PATH pour importer src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import load_config, setup_logger, ensure_dir
from src.normalizer import normalize, is_valid_text

def load_atlaset(config, logger) -> pd.DataFrame:
    path = config["paths"]["data"]["atlaset"]
    if not os.path.exists(path):
        logger.warning(f"Atlaset introuvable à {path}. Saute le chargement.")
        return pd.DataFrame()
    
    df = pd.read_parquet(path)
    
    # Équilibrer: Atlaset est extrêmement grand (~1.2M), on prend un échantillon
    sample_size = config.get("preprocessing", {}).get("atlaset_sample_size", 100000)
    if len(df) > sample_size:
        logger.info(f"Sous-échantillonnage de Atlaset de {len(df)} à {sample_size} lignes pour l'équilibrage.")
        # Utiliser un seed fixe pour la reproductibilité (ex: 42)
        df = df.sample(n=sample_size, random_state=42)

    # Atlaset peut avoir diverses colonnes (ex: text). On l'homogénéise.
    if "text" in df.columns:
        df = df.rename(columns={"text": "text_raw"})
    
    df["source"] = "Atlaset"
    # Atlaset = label original non défini forcément, mais c'est du Darija (Moroccan)
    df["label_original"] = "AR_DAR"
    
    # Génération d'un ID
    df["id"] = [f"atlaset_{i}" for i in range(len(df))]
    
    cols = [col for col in ["id", "text_raw", "label_original", "source"] if col in df.columns]
    return df[cols]

def load_fr_en(config, logger) -> pd.DataFrame:
    path = config["paths"]["data"]["fr_en"]
    if not os.path.exists(path):
        logger.warning(f"Dataset FR/EN introuvable à {path}. Saute le chargement.")
        return pd.DataFrame()
    
    df = pd.read_parquet(path)
    df["source"] = "papluca/language-identification"
    df["id"] = [f"fren_{i}" for i in range(len(df))]
    
    cols = [col for col in ["id", "text_raw", "label_original", "source"] if col in df.columns]
    return df[cols]

def load_nadi(config, logger) -> pd.DataFrame:
    path = config["paths"]["data"]["nadi"]
    nadi_csv = os.path.join(path, "nadi_texts.csv")
    if not os.path.exists(nadi_csv):
        logger.warning(f"NADI CSV introuvable à {nadi_csv}. Saute le chargement.")
        return pd.DataFrame()
    
    try:
        # Tente de charger le CSV, on suppose des colonnes id, text_raw, label_original
        # Utiliser un fallback pour lire txt, tsv si nécessaire
        df = pd.read_csv(nadi_csv)
        if "text" in df.columns and "text_raw" not in df.columns:
            df = df.rename(columns={"text": "text_raw"})
        if "label" in df.columns and "label_original" not in df.columns:
            df = df.rename(columns={"label": "label_original"})
            
        if "text_raw" not in df.columns:
            logger.error("Le fichier NADI n'a pas de colonne 'text' ou 'text_raw'.")
            return pd.DataFrame()
            
        if "id" not in df.columns:
            df["id"] = [f"nadi_{i}" for i in range(len(df))]
            
        df["source"] = "NADI2022"
        return df[["id", "text_raw", "label_original", "source"]]
        
    except Exception as e:
        logger.error(f"Erreur lors du chargement de NADI: {e}")
        return pd.DataFrame()

def main():
    config = load_config()
    logger = setup_logger("02_load_clean", log_level=config["project"]["log_level"])
    
    logger.info("Début de l'étape 2 : Nettoyage et Normalisation")
    
    tqdm.pandas()
    dfs = []
    
    df_atlaset = load_atlaset(config, logger)
    if not df_atlaset.empty: dfs.append(df_atlaset)
        
    df_fren = load_fr_en(config, logger)
    if not df_fren.empty: dfs.append(df_fren)
        
    df_nadi = load_nadi(config, logger)
    if not df_nadi.empty: dfs.append(df_nadi)
        
    if not dfs:
        logger.error("Aucun dataset n'a été chargé ! Arrêt du script.")
        return
        
    df_all = pd.concat(dfs, ignore_index=True)
    logger.info(f"Total phrases combinées : {len(df_all)}")
    
    # Nettoyer les nans
    df_all = df_all.dropna(subset=["text_raw"])
    
    logger.info("Filtrage des phrases invalides (trop courtes ou majoritairement des chiffres)...")
    len_before = len(df_all)
    df_all = df_all[df_all["text_raw"].apply(is_valid_text)]
    logger.info(f"Conservé {len(df_all)} sur {len_before} ({len(df_all)/len_before:.1%}) phrases.")
    
    logger.info("Application de la normalisation...")
    norm_config = config.get("preprocessing", {})
    
    def apply_norm(text):
        return normalize(
            text,
            lowercase_latin=norm_config.get("lowercase_latin", True),
            rm_urls=norm_config.get("remove_urls", True),
            rm_mentions=norm_config.get("remove_mentions", True),
            rm_hashtags=norm_config.get("remove_hashtags", False),
            norm_arabic=norm_config.get("normalize_arabic", True),
            norm_arabizi=norm_config.get("normalize_arabizi", True),
            keep_emojis=norm_config.get("keep_emojis", True)
        )
        
    df_all["text_norm"] = df_all["text_raw"].progress_apply(apply_norm)
    
    # Filtrer les textes vides après normalisation
    df_all = df_all[df_all["text_norm"].str.len() > 0]
    
    out_path = config["paths"]["data"]["processed"]
    ensure_dir(out_path)
    df_all.to_parquet(out_path, index=False)
    
    logger.info(f"Données nettoyées exportées ({len(df_all)} lignes) vers : {out_path}")
    logger.info("Fin de l'étape 2")

if __name__ == "__main__":
    main()
