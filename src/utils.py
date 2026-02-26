import yaml
import logging
import os
from pathlib import Path

def load_config(config_path: str = "config.yaml") -> dict:
    """Charge le fichier de configuration YAML."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Fichier config non trouvé : {config_path}")
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def setup_logger(name: str, log_level: str = "INFO", log_file: str = "pipeline.log") -> logging.Logger:
    """Configure et retourne un logger standard."""
    level = getattr(logging, log_level.upper(), logging.INFO)
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Éviter de dupliquer les handlers si le logger existe déjà
    if not logger.handlers:
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        # Console handler
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        # File handler
        fh = logging.FileHandler(log_file, encoding='utf-8')
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger

def ensure_dir(file_path: str):
    """S'assure que le dossier parent existe."""
    dir_path = os.path.dirname(file_path)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)
