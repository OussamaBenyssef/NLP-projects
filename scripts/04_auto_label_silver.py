import os
import sys
import re
import pandas as pd
from tqdm import tqdm
from ftlangdetect import detect

try:
    import joblib
except ImportError:
    joblib = None

# Ajouter le dossier parent au PATH pour importer src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import load_config, setup_logger, ensure_dir

# Lexique étendu de reconnaissance du Darija (Moroccan Arabic)
# Inclut des mots courants en arabe marocain et en Arabizi
DARIJA_KEYWORDS = set([
    # --- Arabizi (Latin script) ---
    # Verbes courants
    "daba", "dir", "gal", "galt", "galha", "brat", "mcha", "mchit", "ja", "jaw",
    "kan", "kant", "kayn", "kayna", "kayni", "bghit", "bgha", "bghina", "dar", "dart",
    "chaf", "chouf", "choufi", "sma3", "sme3", "kla", "chrab", "nta3", "hder", "hdert",
    "khdm", "khdma", "khedam", "l3b", "l3ib", "t3lem", "kteb", "9ra", "9rit",
    # Pronoms / particules
    "dyal", "dial", "dyali", "dyalna", "dyalkom", "dyalhom",
    "wach", "wesh", "ach", "achmen", "chmen", "fin", "feen", "chhal", "mnin",
    "ila", "iwa", "wakha", "yak", "yalah", "ewa", "7it", "hit", "bash", "bach",
    "hna", "hnak", "hadi", "hadak", "hadik", "hadchi", "hadouk",
    "m3a", "m3ak", "m3aya", "3and", "3andi", "3andek", "3andha", "3la", "3lash",
    # Noms / adjectifs courants
    "khdma", "khoya", "khouya", "sahbi", "sa7bi", "zwin", "zwina",
    "khayb", "khayba", "mzyan", "mzian", "mziana", "bzaf", "bzzaf",
    "flous", "flouss", "drari", "drari", "wlad", "bnat", "bent", "rajl", "mra",
    "chti", "lmghrib", "maghrib", "blad", "bladi", "mdina", "soug", "su9",
    "hanya", "walo", "walou", "wahd", "wa7d", "bhal", "b7al",
    "rani", "rak", "raha", "rahom", "raki", "rana",
    "kifash", "ki", "kidayr", "kidayra", "labas", "labass",
    "ghir", "ghadi", "gha", "mazal", "mazyan", "kolchi", "kolshi",
    # --- Arabe (Arabic script) ---
    # Verbes / particules typiquement Darija
    "علاش", "ديال", "ديالي", "ديالنا", "بزاف", "واش", "واشنو",
    "خويا", "صاحبي", "زوين", "زوينة", "خايب", "مزيان", "مزيانة",
    "دابا", "دير", "دارت", "بغيت", "بغا", "بغينا", "كاين", "كاينة",
    "مشيت", "جا", "جات", "جاو", "شاف", "شوف", "شوفي",
    "هانية", "والو", "كيفاش", "فين", "شحال", "منين",
    "غادي", "غير", "مازال", "كولشي", "بحال", "هادشي", "هادي",
    "خدمة", "فلوس", "دراري", "بنات", "مرا", "راجل",
    "لبلاد", "المغرب", "واخا", "يالاه", "إيوا", "حيت", "باش",
    "راني", "راك", "راها", "راهم", "عند", "عندي", "عندك",
    "كيدير", "كيديرة", "لاباس", "كيداير", "كيدايرة",
])

def get_script_ratio(tokens: list) -> tuple:
    """Retourne le ratio de tokens arabes et latins."""
    if not tokens:
        return 0.0, 0.0
    
    arabic_pattern = re.compile(r'[\u0600-\u06FF]+')
    latin_pattern = re.compile(r'[a-zA-Z]+')
    
    arab_count = sum(1 for t in tokens if arabic_pattern.search(t))
    latin_count = sum(1 for t in tokens if latin_pattern.search(t))
    
    return arab_count / len(tokens), latin_count / len(tokens)

def detect_latin_lang(text: str, default="FR") -> tuple:
    """Détecte la langue latine via fasttext (FR, EN, ES, PT, etc.)."""
    if not text.strip():
        return default, 0.5
    try:
        res = detect(text=text.replace('\n', ' '), low_memory=True)
        lang = res['lang'].upper()
        return lang, res['score']
    except:
        return default, 0.5

def heuristic_darija_score(tokens: list) -> float:
    """Calcule un score rudimentaire de présence de Darija."""
    if not tokens: return 0.0
    matches = sum(1 for t in tokens if t.lower() in DARIJA_KEYWORDS)
    return min(1.0, (matches * 2) / len(tokens))

# Mots FR/EN fréquents qui ne sont PAS des emprunts universels
# (les emprunts comme 'facebook', 'wifi', 'taxi' ne comptent pas comme MIX)
FR_SIGNAL_WORDS = set([
    "le", "la", "les", "un", "une", "des", "du", "de", "dans", "pour",
    "avec", "sur", "par", "est", "sont", "mais", "ou", "donc", "car",
    "je", "tu", "il", "elle", "nous", "vous", "ils", "elles",
    "mon", "ton", "son", "notre", "votre", "leur", "mes", "tes", "ses",
    "ce", "cette", "ces", "quel", "quelle",
    "pas", "ne", "plus", "très", "bien", "aussi", "comme", "tout",
    "faire", "avoir", "être", "aller", "venir", "voir", "savoir",
    "avant", "après", "encore", "toujours", "jamais", "peut",
    "problème", "solution", "inscription", "documents", "rapport",
    "application", "interface", "formation", "réunion", "bureau",
    "envoyer", "checker", "préparer", "commencer", "travailler",
])

EN_SIGNAL_WORDS = set([
    "the", "a", "an", "is", "are", "was", "were", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would",
    "can", "could", "should", "may", "might", "shall", "must",
    "i", "you", "he", "she", "we", "they", "it", "my", "your", "his",
    "for", "with", "from", "about", "into", "through", "during",
    "before", "after", "above", "below", "between", "under",
    "not", "no", "but", "or", "and", "if", "than", "too", "very",
    "just", "only", "also", "here", "there", "when", "where",
    "meeting", "deadline", "update", "email", "emails", "report",
    "problem", "working", "check", "send", "bug",
])

# Emprunts universels (mots intégrés dans le Darija, ne comptent PAS comme MIX)
BORROWED_UNIVERSAL = set([
    "facebook", "instagram", "youtube", "twitter", "whatsapp", "tiktok",
    "google", "wifi", "internet", "email", "password", "ok", "okay",
    "smartphone", "laptop", "pc", "usb", "bluetooth",
    "taxi", "bus", "parking", "pizza", "sandwich",
    "film", "match", "goal", "club", "sport", "covid", "vaccin",
    "video", "photo", "selfie", "url", "user", "live",
])

def detect_arabizi_mix(tokens: list) -> tuple:
    """Détecte le MIX Arabizi+FR/EN (tout en script latin).
    
    Retourne (is_mix, darija_count, fr_en_count) :
    - Compte les tokens Arabizi (matchent DARIJA_KEYWORDS)
    - Compte les tokens FR/EN (matchent FR/EN_SIGNAL_WORDS, hors emprunts)
    - Si les deux >= 2 tokens → MIX
    """
    darija_count = 0
    fr_en_count = 0
    
    for t in tokens:
        t_low = t.lower().strip(".,;:!?'\"()[]{}")
        if not t_low or not re.search(r'[a-zA-Z]', t_low):
            continue  # Skip non-latin tokens
        
        if t_low in DARIJA_KEYWORDS:
            darija_count += 1
        elif t_low in BORROWED_UNIVERSAL:
            pass  # Emprunts ne comptent pas
        elif t_low in FR_SIGNAL_WORDS or t_low in EN_SIGNAL_WORDS:
            fr_en_count += 1
        # Mots ambigus ou inconnus : on ne compte pas
    
    is_mix = darija_count >= 2 and fr_en_count >= 2
    return is_mix, darija_count, fr_en_count


def auto_annotate_row(row, config):
    text = row.get("text_norm", "")
    original = row.get("label_original", "")
    source = row.get("source", "")
    
    tokens = text.split()
    if not tokens:
        return pd.Series(["AR_MSA", 0.0, 0.0, 0.0, 0.0])
        
    arab_ratio, latin_ratio = get_script_ratio(tokens)
    cs_ratio = min(arab_ratio, latin_ratio) # Estimation simple du code-switching par mélange de scripts
    
    label_silver = None
    confidence = 0.5
    
    # Patterns compilés pour le comptage
    arabic_pattern = re.compile(r'[\u0600-\u06FF]+')
    latin_pattern = re.compile(r'[a-zA-Z]+')
    
    # 0. Vérifier si c'est du MIX Arabizi+FR/EN (tout en latin mais mélange de langues)
    if latin_ratio > 0.7:
        is_arabizi_mix, dar_cnt, fren_cnt = detect_arabizi_mix(tokens)
        if is_arabizi_mix:
            label_silver = "MIX"
            confidence = 0.80
            return pd.Series([label_silver, confidence, cs_ratio, arab_ratio, latin_ratio])
    
    # 1. Utilisation de la source / label d'origine comme fort signal
    if source == "Atlaset":
        if latin_ratio > 0.8:
            latin_text = " ".join([t for t in tokens if re.search(r'[a-zA-Z]', t)])
            lang, score = detect_latin_lang(latin_text)
            if lang in ["FR", "EN"]:
                label_silver = lang
                confidence = score * 0.8
            elif lang not in ["AR"] and score > 0.5:
                # Bruit espagnol, portugais, etc...
                label_silver = "OTHER"
                confidence = 0.0
            else:
                label_silver = "AR_DAR"
                confidence = 0.80
        else:
            # Fix 4: Appliquer les heuristiques DAR vs MSA au lieu de tout defaulter AR_DAR
            dar_score = heuristic_darija_score(tokens)
            if dar_score > 0.05:
                label_silver = "AR_DAR"
                confidence = 0.8 + min(dar_score, 0.15)
            else:
                label_silver = "AR_MSA"
                confidence = 0.6
    elif source == "papluca/language-identification":
        label_silver = original if original in ["FR", "EN"] else "FR"
        confidence = 0.95
    elif source == "NADI2022":
        # Fix 1: Mapping complet des labels NADI
        if str(original).upper() == "MSA":
            label_silver = "AR_MSA"
            confidence = 0.85
        elif str(original).upper() == "MOROCCO":
            label_silver = "AR_DAR"
            confidence = 0.85
        else:
            # Tous les autres dialectes arabes (egypt, iraq, ksa, etc.) → AR_MSA
            label_silver = "AR_MSA"
            confidence = 0.75

    # 2. Si pas de de label déterminé par la source, on utilise les heuristiques
    if label_silver is None:
        if latin_ratio > 0.8:
            # C'est principalement latin. Est-ce du FR/EN ou de l'Arabizi (DAR) ?
            # On détecte avec fasttext
            latin_text = " ".join([t for t in tokens if re.search(r'[a-zA-Z]', t)])
            lang, score = detect_latin_lang(latin_text)
            
            # On vérifie les mots clés arabizi
            dar_score = heuristic_darija_score(tokens)
            if dar_score > 0.1:
                label_silver = "AR_DAR"
                confidence = 0.6 + min(dar_score, 0.3)
            else:
                label_silver = lang
                confidence = score
                
        elif arab_ratio > 0.8:
            # Principalement arabe. MSA ou DAR ?
            dar_score = heuristic_darija_score(tokens)
            if dar_score > 0.05:
                label_silver = "AR_DAR"
                confidence = 0.7 + min(dar_score, 0.2)
            else:
                label_silver = "AR_MSA"
                confidence = 0.6
        else:
            # Mélange complexe — vérifier le code-switching avec contrainte de tokens minimum
            arab_tok_count = sum(1 for t in tokens if arabic_pattern.search(t))
            latin_tok_count = sum(1 for t in tokens if latin_pattern.search(t))
            min_tokens = config["auto_annotation"].get("mix_min_tokens_per_script", 2)
            
            if (cs_ratio > config["auto_annotation"]["cs_ratio_threshold"]
                    and arab_tok_count >= min_tokens and latin_tok_count >= min_tokens):
                label_silver = "MIX"
                confidence = 0.75
            else:
                label_silver = "AR_DAR"  # Fallback
                confidence = 0.4

    # 3. Forcer MIX si code-switching très évident, avec contrainte de tokens minimum
    if label_silver != "MIX" and cs_ratio > config["auto_annotation"]["cs_ratio_threshold"]:
        arab_tok_count = sum(1 for t in tokens if arabic_pattern.search(t))
        latin_tok_count = sum(1 for t in tokens if latin_pattern.search(t))
        min_tokens = config["auto_annotation"].get("mix_min_tokens_per_script", 2)
        
        if arab_tok_count >= min_tokens and latin_tok_count >= min_tokens:
            label_silver = "MIX"
            confidence = 0.8
        
    return pd.Series([label_silver, confidence, cs_ratio, arab_ratio, latin_ratio])

def process_split(split_name, in_path, out_path, config, logger, dar_msa_clf=None):
    logger.info(f"Traitement de la split '{split_name}': {in_path}")
    if not os.path.exists(in_path):
        logger.warning(f"Fichier {in_path} introuvable.")
        return
        
    df = pd.read_parquet(in_path)
    
    tqdm.pandas(desc=f"Annotation {split_name}")
    res = df.progress_apply(lambda row: auto_annotate_row(row, config), axis=1)
    df[["label_silver", "confidence", "cs_ratio", "arab_ratio", "latin_ratio"]] = res
    
    # Exclure les bruits (ex: textes espagnols marqués "OTHER")
    df = df[df["label_silver"] != "OTHER"]
    
    # Post-traitement : appliquer le classifieur DAR/MSA sur les cas ambigus
    if dar_msa_clf is not None:
        arabic_mask = df["label_silver"].isin(["AR_DAR", "AR_MSA"])
        ambiguous_mask = arabic_mask & (df["confidence"] < 0.85)
        
        if ambiguous_mask.sum() > 0:
            ambiguous_texts = df.loc[ambiguous_mask, "text_norm"].values
            clf_predictions = dar_msa_clf.predict(ambiguous_texts)
            
            n_changed = (df.loc[ambiguous_mask, "label_silver"].values != clf_predictions).sum()
            df.loc[ambiguous_mask, "label_silver"] = clf_predictions
            # Boost confidence for classifier-relabeled samples
            df.loc[ambiguous_mask, "confidence"] = df.loc[ambiguous_mask, "confidence"].clip(lower=0.70)
            
            logger.info(f"  Classifieur DAR/MSA : {ambiguous_mask.sum()} cas ambigus traités, {n_changed} relabelés")
    
    ensure_dir(out_path)
    df.to_parquet(out_path, index=False)
    
    # Affichage d'un bilan pour cette split
    bilan = df["label_silver"].value_counts(normalize=True).to_dict()
    logger.info(f"Bilan {split_name} : { {k: f'{v:.1%}' for k, v in bilan.items()} }")
    logger.info(f"Confiance moyenne : {df['confidence'].mean():.2f}")

def main():
    config = load_config()
    logger = setup_logger("04_silver", log_level=config["project"]["log_level"])
    
    logger.info("Début de l'étape 4 : Auto-annotation (Silver Labels)")
    
    # Charger le classifieur DAR/MSA si disponible
    dar_msa_clf = None
    clf_path = config["paths"]["models"].get("dar_msa_classifier", "models/dar_msa_classifier.pkl")
    if joblib is not None and os.path.exists(clf_path):
        dar_msa_clf = joblib.load(clf_path)
        logger.info(f"Classifieur DAR/MSA chargé depuis {clf_path}")
    else:
        logger.warning("Classifieur DAR/MSA non trouvé — utilisation des heuristiques seules")
    
    # Train
    process_split(
        "train",
        config["paths"]["data"]["train_split"],
        config["paths"]["data"]["train_silver"],
        config, logger, dar_msa_clf=dar_msa_clf
    )
    
    # Valid
    process_split(
        "valid",
        config["paths"]["data"]["valid_split"],
        config["paths"]["data"]["valid_silver"],
        config, logger, dar_msa_clf=dar_msa_clf
    )
    
    # Test (pour évaluation vs gold)
    test_silver_path = config["paths"]["data"].get("test_silver", "data/silver/test_silver.parquet")
    process_split(
        "test",
        config["paths"]["data"]["test_split"],
        test_silver_path,
        config, logger, dar_msa_clf=dar_msa_clf
    )
    
    logger.info("Auto-annotation terminée avec succès.")
    logger.info("Fin de l'étape 4")

if __name__ == "__main__":
    main()
