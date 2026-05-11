"""
utils.py — Ortak yardımcı fonksiyonlar
Türkçe Haber Analiz Uygulaması
"""

import re
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")           # GUI gerektirmeyen backend (Colab uyumlu)
import seaborn as sns
import evaluate
from sklearn.metrics import classification_report, confusion_matrix


# ─────────────────────────────────────────────────────────────
# 1. Metin Temizleme
# ─────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
 
    if not isinstance(text, str):
        return ""

    # HTML etiketleri
    text = re.sub(r"<[^>]+>", " ", text)
    # URL'ler
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    # E-posta
    text = re.sub(r"\S+@\S+", " ", text)
    # Birden fazla boşluk
    text = re.sub(r"\s+", " ", text)
    # Baş/son boşluk
    text = text.strip()

    return text




accuracy_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")


def compute_metrics(eval_pred):
    """
    Trainer API'ye bağlanır.
    Hem accuracy hem de weighted F1 döner.
    """
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    acc = accuracy_metric.compute(predictions=predictions, references=labels)
    f1 = f1_metric.compute(
        predictions=predictions,
        references=labels,
        average="weighted"
    )
    return {
        "accuracy": acc["accuracy"],
        "f1": f1["f1"],
    }



def plot_confusion_matrix(y_true, y_pred, labels: list, save_path: str = None):
    """
    Confusion matrix oluştur ve isteğe bağlı PNG olarak kaydet.

    Args:
        y_true   : Gerçek etiketler (int listesi)
        y_pred   : Tahmin edilen etiketler (int listesi)
        labels   : Sınıf isimleri (string listesi)
        save_path: Kaydedilecek dosya yolu (None ise sadece gösterir)
    """
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(max(6, len(labels)), max(5, len(labels) - 1)))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax
    )
    ax.set_xlabel("Tahmin Edilen", fontsize=12)
    ax.set_ylabel("Gerçek", fontsize=12)
    ax.set_title("Confusion Matrix", fontsize=14)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
        print(f"Confusion matrix kaydedildi: {save_path}")

    plt.show()
    plt.close(fig)
    return fig




def plot_training_history(trainer, save_path: str = None, title: str = "Training History"):
    """
    HuggingFace Trainer'ın log geçmişinden loss grafiği çizer.

    Args:
        trainer  : Eğitim tamamlanmış Trainer nesnesi
        save_path: PNG kayıt yolu (None ise sadece gösterir)
        title    : Grafik başlığı
    """
    log_history = trainer.state.log_history

    train_losses, eval_losses = [], []
    train_epochs, eval_epochs = [], []

    for entry in log_history:
        if "loss" in entry and "eval_loss" not in entry:
            train_losses.append(entry["loss"])
            train_epochs.append(entry.get("epoch", len(train_losses)))
        if "eval_loss" in entry:
            eval_losses.append(entry["eval_loss"])
            eval_epochs.append(entry.get("epoch", len(eval_losses)))

    fig, ax = plt.subplots(figsize=(8, 5))
    if train_losses:
        ax.plot(train_epochs, train_losses, label="Train Loss", marker="o", color="#2563EB")
    if eval_losses:
        ax.plot(eval_epochs, eval_losses, label="Validation Loss", marker="s", color="#DC2626")

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
        print(f"Loss grafiği kaydedildi: {save_path}")

    plt.show()
    plt.close(fig)
    return fig



def save_classification_report(y_true, y_pred, labels: list, save_path: str):
    """
    sklearn classification_report'u hem konsola yazar hem JSON kaydeder.
    """
    report_str = classification_report(y_true, y_pred, target_names=labels)
    report_dict = classification_report(
        y_true, y_pred, target_names=labels, output_dict=True
    )

    print("\n" + "=" * 60)
    print(report_str)
    print("=" * 60)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(report_dict, f, ensure_ascii=False, indent=2)
    print(f"Rapor kaydedildi: {save_path}")

    return report_dict




def save_label_mapping(label2id: dict, id2label: dict, save_dir: str):
    """Label <-> ID eşlemesini JSON olarak kaydet."""
    os.makedirs(save_dir, exist_ok=True)
    mapping = {"label2id": label2id, "id2label": id2label}
    path = os.path.join(save_dir, "label_mapping.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)
    print(f"Label mapping kaydedildi: {path}")


def load_label_mapping(save_dir: str):
    """Kaydedilmiş label mapping'i yükle."""
    path = os.path.join(save_dir, "label_mapping.json")
    with open(path, "r", encoding="utf-8") as f:
        mapping = json.load(f)
    # id2label anahtarları JSON'da string olduğundan int'e çevir
    id2label = {int(k): v for k, v in mapping["id2label"].items()}
    label2id = mapping["label2id"]
    return label2id, id2label
