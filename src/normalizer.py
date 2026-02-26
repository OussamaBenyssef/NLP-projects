import re
import emoji

def remove_urls(text: str) -> str:
    # Suppression des URLs
    return re.sub(r'http\S+|www\.\S+', '', text)

def remove_mentions(text: str) -> str:
    # Suppression des mentions @
    return re.sub(r'@\w+', '', text)

def remove_hashtags(text: str) -> str:
    # Suppression des hashtags #
    return re.sub(r'#\w+', '', text)

def normalize_arabic_text(text: str) -> str:
    """Normalisation standard de l'arabe."""
    text = re.sub(r'[إأآا]', 'ا', text)
    text = re.sub(r'ة', 'ه', text)
    text = re.sub(r'ى', 'ي', text)
    # Tatweel
    text = re.sub(r'ـ', '', text)
    # Diacritics (harakat)
    text = re.sub(r'[\u064B-\u065F\u0670]', '', text)
    return text

def normalize_arabizi_text(text: str) -> str:
    """Règles heuristiques simples de normalisation de l'arabizi vers l'arabe (translittération basique).
    Attention: ceci est heuristique et imparfait."""
    # Mapping simpliste fréquent dans l'arabizi
    arabizi_map = {
        '7': 'ح',
        '3': 'ع',
        '9': 'ق',
        '2': 'ء',
        '5': 'خ',
        '6': 'ط',
        '8': 'غ',
    }
    
    # Remplacer uniquement si c'est entouré de lettres latines ou si ça ressemble à un mot arabizi
    # Pour faire simple on remplace ces chiffres dans le texte entier (heuristique basique).
    # Cela peut fausser les vrais chiffres 7, 3, etc., mais correspond aux consignes du projet.
    for k, v in arabizi_map.items():
        # Remplacer le chiffre si entouré par des lettres latines ou au bord d'un mot latin
        # ex: "sbah lkheer 7bibi" -> le 7 doit devenir ح
        text = re.sub(rf'(?<=[a-zA-Z]){k}|{k}(?=[a-zA-Z])', v, text)
        
    return text

def clean_whitespaces(text: str) -> str:
    return re.sub(r'\s+', ' ', text).strip()

def normalize(
    text: str,
    lowercase_latin: bool = True,
    rm_urls: bool = True,
    rm_mentions: bool = True,
    rm_hashtags: bool = False,
    norm_arabic: bool = True,
    norm_arabizi: bool = True,
    keep_emojis: bool = True
) -> str:
    """Pipeline de normalisation complète du texte."""
    if not isinstance(text, str):
        return ""
    
    if rm_urls:
        text = remove_urls(text)
    if rm_mentions:
        text = remove_mentions(text)
    if rm_hashtags:
        text = remove_hashtags(text)
        
    if not keep_emojis:
        text = emoji.replace_emoji(text, replace='')
        
    if lowercase_latin:
        text = text.lower()
        
    if norm_arabizi:
        text = normalize_arabizi_text(text)
        
    if norm_arabic:
        text = normalize_arabic_text(text)
        
    text = clean_whitespaces(text)
    return text

def is_valid_text(text: str) -> bool:
    """Vérifie si le texte contient au moins un mot valide (pas uniquement des chiffres/ponctuations)."""
    if not isinstance(text, str) or not text.strip():
        return False
    # Vérifie la présence d'au moins 2 lettres (arabes ou latines) pour être un texte valide
    letters = re.findall(r'[a-zA-Z\u0600-\u06FF]', text)
    if len(letters) < 2:
        return False
    # Évite les textes composés quasi-exclusivement de chiffres
    numbers = re.findall(r'\d', text)
    if len(numbers) > len(letters):
        return False
    return True
