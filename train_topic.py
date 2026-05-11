"""
train_topic.py — Konu Siniflandirma Modeli Egitimi
Veri seti : savasy/ttc4900 (HuggingFace Hub) — 4900 Turkce haber, 7 kategori
Model     : dbmdz/bert-base-turkish-cased (BERTurk)
"""

import os
import numpy as np
from datasets import load_dataset, Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
)
from utils import (
    clean_text, compute_metrics, plot_confusion_matrix,
    plot_training_history, save_classification_report, save_label_mapping,
)

# ── Yapılandırma ──────────────────────────────────────────────
BASE_MODEL    = "dbmdz/bert-base-turkish-cased"
DATASET_NAME  = "savasy/ttc4900"
MODEL_SAVE    = "models/topic_model"
DATA_SAVE     = "data/topic"
MAX_LENGTH    = 256
BATCH_SIZE    = 8   # RTX 3050 4GB VRAM icin guvenliboyut
NUM_EPOCHS    = 3
LEARNING_RATE = 2e-5
SEED          = 42

os.makedirs(MODEL_SAVE, exist_ok=True)
os.makedirs(DATA_SAVE,  exist_ok=True)




# ── Sütun algılama ────────────────────────────────────────────
def _detect_column(sample, candidates):
    for col in candidates:
        if col in sample:
            return col
    for k, v in sample.items():
        if isinstance(v, str):
            return k
    raise ValueError(f"Sütun bulunamadı. Mevcut: {list(sample.keys())}")


# ── Ana akış ──────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  KONU SINIFLANDIRMA EĞİTİMİ")
    print("=" * 60)

    # 1. Veri yükleme
    print("\n[1/6] Veri seti yükleniyor...")
    dataset = load_dataset(DATASET_NAME)
    print(f"    Yuklendi: {DATASET_NAME}")
    # ttc4900 sadece 'train' split iceriyor — manuel bol
    if "test" not in dataset:
        split = dataset["train"].train_test_split(test_size=0.2, seed=SEED)
        dataset = DatasetDict({"train": split["train"], "test": split["test"]})
        print(f"    Train/test bölündü: {len(dataset['train'])} / {len(dataset['test'])}")

    # 2. Ön işleme
    print("\n[2/6] Ön işleme...")
    sample     = dataset["train"][0]
    # ttc4900 dataset sutunlari: "text", "category" (string etiket)
    text_col   = _detect_column(sample, ["text", "content", "sentence"])
    label_col  = _detect_column(sample, ["category", "label", "label_text", "cat"])

    # ttc4900 kategorileri (string olarak gelir)
    raw_labels = dataset["train"][label_col]
    first_label = raw_labels[0]
    if isinstance(first_label, int) or (isinstance(first_label, str) and str(first_label).isdigit()):
        CATEGORY_NAMES = [
            "kultursanat", "ekonomi", "siyaset", "egitim",
            "dunya", "spor", "teknoloji", "magazin", "saglik", "gundem"
        ]
        unique_ids = sorted(set(int(l) for l in raw_labels))
        id2label   = {i: CATEGORY_NAMES[i] if i < len(CATEGORY_NAMES) else str(i)
                      for i in unique_ids}
        label2id   = {v: k for k, v in id2label.items()}
    else:
        all_labels = sorted(set(str(l) for l in raw_labels))
        label2id   = {l: i for i, l in enumerate(all_labels)}
        id2label   = {i: l for i, l in enumerate(all_labels)}

    print(f"    Sınıf sayısı : {len(id2label)}")
    print(f"    Sınıflar     : {list(id2label.values())}")

    def preprocess_fn(examples):
        cleaned = [clean_text(str(t)) for t in examples[text_col]]
        if isinstance(examples[label_col][0], int) or (
                isinstance(examples[label_col][0], str) and examples[label_col][0].isdigit()):
            labels = [int(l) for l in examples[label_col]]
        else:
            labels = [label2id[str(l)] for l in examples[label_col]]
        return {"cleaned_text": cleaned, "labels": labels}

    dataset = dataset.map(preprocess_fn, batched=True)
    save_label_mapping(label2id, id2label, MODEL_SAVE)

    # 3. Tokenization
    print("\n[3/6] Tokenization...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    def tokenize_fn(examples):
        return tokenizer(examples["cleaned_text"], truncation=True, max_length=MAX_LENGTH)

    tokenized = dataset.map(tokenize_fn, batched=True)
    keep_cols = ["input_ids", "attention_mask", "token_type_ids", "labels"]
    remove_cols = [c for c in tokenized["train"].column_names if c not in keep_cols]
    tokenized = tokenized.remove_columns(remove_cols)
    tokenized.set_format("torch")

    # 4. Model & Trainer
    print("\n[4/6] Model eğitiliyor...")
    print(f"    Epoch: {NUM_EPOCHS} | Batch: {BATCH_SIZE} | LR: {LEARNING_RATE}")
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL, num_labels=len(label2id),
        id2label=id2label, label2id=label2id, ignore_mismatched_sizes=True,
    )
    args = TrainingArguments(
        output_dir=MODEL_SAVE,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
        warmup_ratio=0.1,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_steps=50,
        seed=SEED,
        report_to="none",
        fp16=True,
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["test"],
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )
    trainer.train()

    # 5. Loss grafiği
    plot_training_history(trainer,
        save_path=os.path.join(DATA_SAVE, "topic_loss.png"),
        title="Konu Sınıflandırma — Training History")

    # 6. Değerlendirme
    print("\n[5/6] Değerlendirme...")
    out    = trainer.predict(tokenized["test"])
    y_pred = np.argmax(out.predictions, axis=-1)
    y_true = out.label_ids
    labels = [id2label[i] for i in sorted(id2label.keys())]
    save_classification_report(y_true, y_pred, labels,
        save_path=os.path.join(DATA_SAVE, "topic_report.json"))
    plot_confusion_matrix(y_true, y_pred, labels,
        save_path=os.path.join(DATA_SAVE, "topic_confusion_matrix.png"))

    # 7. Model kaydet
    print(f"\n[6/6] Model kaydediliyor -> {MODEL_SAVE}")
    trainer.save_model(MODEL_SAVE)
    tokenizer.save_pretrained(MODEL_SAVE)
    print("\nKonu siniflandirma tamamlandi!")


if __name__ == "__main__":
    main()
