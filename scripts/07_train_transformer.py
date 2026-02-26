import os
import sys
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

import torch
from torch.utils.data import Dataset as TorchDataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
from transformers import set_seed

# Ajouter le dossier parent au PATH pour importer src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import load_config, setup_logger, ensure_dir


# ---------------------------------------------------------------------------
# Dataset PyTorch
# ---------------------------------------------------------------------------
class LangDataset(TorchDataset):
    """Dataset Pytorch pour la classification de langues."""
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


# ---------------------------------------------------------------------------
# Métriques
# ---------------------------------------------------------------------------
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="macro")
    return {"accuracy": acc, "macro_f1": f1}


# ---------------------------------------------------------------------------
# Confusion Matrix Helper
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Fine-tune Transformer (XLM-R / mBERT)")
    parser.add_argument("--filter_confidence", type=float, default=0.6,
                        help="Filtrer les données train_silver avec confidence < X")
    parser.add_argument("--max_train_samples", type=int, default=0,
                        help="Limiter le nombre de samples d'entraînement (0 = tout)")
    args = parser.parse_args()

    config = load_config()
    logger = setup_logger("07_transformer", log_level=config["project"]["log_level"])
    seed = config["project"]["seed"]
    set_seed(seed)

    logger.info("Début de l'étape 7 : Fine-tuning Transformer")

    # -----------------------------------------------------------------------
    # 1. Chargement des données
    # -----------------------------------------------------------------------
    path_train = config["paths"]["data"]["train_silver"]
    path_valid = config["paths"]["data"]["valid_silver"]
    path_test_gold = config["paths"]["data"]["test_gold"]

    if not os.path.exists(path_train) or not os.path.exists(path_valid):
        logger.error("Fichiers silver introuvables. Lancez les étapes précédentes.")
        return

    df_train = pd.read_parquet(path_train)
    df_valid = pd.read_parquet(path_valid)

    # Filtrage confiance
    if args.filter_confidence > 0:
        len_before = len(df_train)
        df_train = df_train[df_train["confidence"] >= args.filter_confidence]
        logger.info(f"Filtrage confiance >= {args.filter_confidence}: {len(df_train)}/{len_before} conservés.")

    # Sous-échantillonage optionnel
    if args.max_train_samples > 0 and len(df_train) > args.max_train_samples:
        logger.info(f"Sous-échantillon de {len(df_train)} -> {args.max_train_samples}")
        df_train = df_train.sample(n=args.max_train_samples, random_state=seed)

    # -----------------------------------------------------------------------
    # 2. Encodage des labels
    # -----------------------------------------------------------------------
    all_labels = sorted(df_train["label_silver"].unique())
    label2id = {lbl: i for i, lbl in enumerate(all_labels)}
    id2label = {i: lbl for lbl, i in label2id.items()}
    num_labels = len(all_labels)

    logger.info(f"Classes : {label2id}")
    logger.info(f"Nombre de classes : {num_labels}")

    y_train = df_train["label_silver"].map(label2id).values
    y_valid = df_valid["label_silver"].map(label2id).values
    # Filtrer les labels inconnus dans valid
    valid_mask = pd.notna(df_valid["label_silver"].map(label2id))
    df_valid = df_valid[valid_mask]
    y_valid = df_valid["label_silver"].map(label2id).values

    X_train_texts = df_train["text_norm"].tolist()
    X_valid_texts = df_valid["text_norm"].tolist()

    logger.info(f"Train: {len(X_train_texts)} | Valid: {len(X_valid_texts)}")

    # -----------------------------------------------------------------------
    # 3. Tokenisation
    # -----------------------------------------------------------------------
    model_name = config["training"]["transformer"]["model_name"]
    logger.info(f"Modèle : {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    train_encodings = tokenizer(X_train_texts, truncation=True, padding=True, max_length=128)
    valid_encodings = tokenizer(X_valid_texts, truncation=True, padding=True, max_length=128)

    train_dataset = LangDataset(train_encodings, y_train)
    valid_dataset = LangDataset(valid_encodings, y_valid)

    # -----------------------------------------------------------------------
    # 4. Modèle
    # -----------------------------------------------------------------------
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    )

    # -----------------------------------------------------------------------
    # 5. Entraînement avec Trainer
    # -----------------------------------------------------------------------
    output_dir = config["paths"]["models"]["transformer"]
    epochs = config["training"]["transformer"]["epochs"]
    batch_size = config["training"]["transformer"]["batch_size"]
    lr = config["training"]["transformer"]["learning_rate"]

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        learning_rate=lr,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        logging_steps=100,
        seed=seed,
        report_to="none",
        fp16=torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    logger.info("Lancement de l'entraînement...")
    trainer.train()
    logger.info("Entraînement terminé.")

    # -----------------------------------------------------------------------
    # 6. Évaluation sur Valid (Silver)
    # -----------------------------------------------------------------------
    logger.info("=== Évaluation sur VALID_SILVER ===")
    valid_results = trainer.evaluate(valid_dataset)
    logger.info(f"Valid Silver Results: {valid_results}")

    preds_valid = trainer.predict(valid_dataset)
    y_pred_valid = np.argmax(preds_valid.predictions, axis=-1)
    y_true_valid = y_valid

    labels_str_valid = [id2label[i] for i in range(num_labels)]
    logger.info("\n" + classification_report(
        y_true_valid, y_pred_valid, target_names=labels_str_valid
    ))

    plot_cm_valid = "docs/figures/cm_transformer_valid.png"
    plot_confusion_matrix(
        y_true_valid, y_pred_valid,
        list(range(num_labels)), plot_cm_valid,
        title="Transformer - Validation (Silver)"
    )
    logger.info(f"Matrice de confusion Validation: {plot_cm_valid}")

    # -----------------------------------------------------------------------
    # 7. Évaluation sur Test Gold (si disponible)
    # -----------------------------------------------------------------------
    if os.path.exists(path_test_gold):
        logger.info("=== Évaluation sur TEST_GOLD ===")
        df_test = pd.read_parquet(path_test_gold)
        if not df_test.empty and "label_gold" in df_test.columns:
            df_test = df_test[df_test["label_gold"].isin(label2id.keys())]
            if not df_test.empty:
                y_test = df_test["label_gold"].map(label2id).values
                X_test_texts = df_test["text_norm"].tolist()

                test_encodings = tokenizer(X_test_texts, truncation=True, padding=True, max_length=128)
                test_dataset = LangDataset(test_encodings, y_test)

                preds_test = trainer.predict(test_dataset)
                y_pred_test = np.argmax(preds_test.predictions, axis=-1)

                acc_test = accuracy_score(y_test, y_pred_test)
                f1_test = f1_score(y_test, y_pred_test, average="macro")
                logger.info(f"Test Gold Accuracy: {acc_test:.2%}")
                logger.info(f"Test Gold Macro F1: {f1_test:.2%}")
                logger.info("\n" + classification_report(
                    y_test, y_pred_test, target_names=labels_str_valid
                ))

                plot_cm_test = "docs/figures/cm_transformer_test_gold.png"
                plot_confusion_matrix(
                    y_test, y_pred_test,
                    list(range(num_labels)), plot_cm_test,
                    title="Transformer - Test (Gold)"
                )
                logger.info(f"Matrice de confusion Test Gold: {plot_cm_test}")
    else:
        logger.warning(f"Test gold ({path_test_gold}) introuvable.")

    # -----------------------------------------------------------------------
    # 8. Sauvegarde du modèle final
    # -----------------------------------------------------------------------
    final_model_path = os.path.join(output_dir, "best")
    trainer.save_model(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    logger.info(f"Modèle final sauvegardé dans : {final_model_path}")

    logger.info("Fin de l'étape 7")


if __name__ == "__main__":
    main()
