"""
train_sentiment.py — Duygu Analizi Modeli Eğitimi
Veri seti : winvoker/turkish-sentiment-analysis-dataset (HuggingFace Hub)
Model     : dbmdz/bert-base-turkish-cased (BERTurk)
Sınıflar  : pozitif, negatif, nötr
"""

import os 
import numpy as np
from datasets import load_dataset, Dataset, DatasetDict
from transformers import (
    AutoTokenizer, # tokenizasyon yapar
    AutoModelForSequenceClassification, # sınıflandırma yapar
    TrainingArguments, # eğitim argümanlarını ayarlar
    Trainer, # modeli eğitir
    DataCollatorWithPadding, # padding yapar [PAD]
    EarlyStoppingCallback, # erken durdurma callback
)
from utils import (
    clean_text, compute_metrics, plot_confusion_matrix,
    plot_training_history, save_classification_report, save_label_mapping,
)

# ── Yapılandırma ──────────────────────────────────────────────
BASE_MODEL    = "dbmdz/bert-base-turkish-cased"
DATASET_NAME  = "winvoker/turkish-sentiment-analysis-dataset"
MODEL_SAVE    = "models/sentiment_model"
DATA_SAVE     = "data/sentiment"
MAX_LENGTH    = 128  # token sayisi
BATCH_SIZE    = 8    # paket büyüklüğü 
NUM_EPOCHS    = 3    # tur sayısı , model veriyi kaç kere görecek
LEARNING_RATE = 2e-5 # öğrenme oranı , model türkçe sayı küçük
SEED          = 42   # rastgelelik tohumu

SENTIMENT_LABELS = ["negatif", "nötr", "pozitif"]  

os.makedirs(MODEL_SAVE, exist_ok=True)
os.makedirs(DATA_SAVE,  exist_ok=True)


LABEL_NORMALIZE_MAP = {
   
    "positive": "pozitif", "negative": "negatif", "neutral": "nötr",
    "pos": "pozitif",      "neg": "negatif",       "neu": "nötr",
    
    "pozitif": "pozitif",  "negatif": "negatif",   "nötr": "nötr",
    "notr": "nötr",
    # winvoker dataset: 0=negatif, 1=nötr, 2=pozitif
    "0": "negatif",  "1": "nötr",   "2": "pozitif",
    "0.0": "negatif","1.0": "nötr", "2.0": "pozitif",
    "-1": "negatif", "-1.0": "negatif",
}

def normalize_label(raw) -> str:
    s = str(raw).strip().lower()
    return LABEL_NORMALIZE_MAP.get(s, "nötr")   # bilinmeyen -> nötr

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
    print("  DUYGU ANALİZİ EĞİTİMİ")
    print("=" * 60)

    # Dictionary comprehension 
    label2id = {l: i for i, l in enumerate(SENTIMENT_LABELS)}
    id2label = {i: l for i, l in enumerate(SENTIMENT_LABELS)}

    # 1. Veri yükleme
    print("\n[1/6] Veri seti yükleniyor...")
    dataset = load_dataset(DATASET_NAME)
    print(f"    OK Yüklendi: {DATASET_NAME}")
    
    # Egitimi hizlandirmak icin veri setini %10 oranina kucult
    print("    Veri seti kucultuluyor (%10)...")
    import random
    for split in dataset.keys():
        size = len(dataset[split])
        subset_size = int(size * 0.1)
        indices = list(range(size))
        random.shuffle(indices)
        dataset[split] = dataset[split].select(indices[:subset_size])
    print(f"    Yeni egitim ornek sayisi: {len(dataset['train'])}")

    # 2. Ön işleme
    print("\n[2/6] Ön işleme ve etiket normalizasyonu...")
    sample    = dataset["train"][0]
    text_col  = _detect_column(sample, ["text", "content", "sentence", "review", "comment"])
    label_col = _detect_column(sample, ["label", "sentiment", "polarity", "class"])

    print(f"    Metin sütunu  : {text_col}")
    print(f"    Etiket sütunu : {label_col}")
    print(f"    Sınıflar      : {SENTIMENT_LABELS}")

    def preprocess_fn(examples):
        cleaned = [clean_text(str(t)) for t in examples[text_col]]
        labels  = [label2id[normalize_label(l)] for l in examples[label_col]]
        return {"cleaned_text": cleaned, "labels": labels}

    dataset = dataset.map(preprocess_fn, batched=True)
    save_label_mapping(label2id, id2label, MODEL_SAVE)

    # Etiket dağılımını göster
    train_labels = dataset["train"]["labels"]
    for i, name in id2label.items():
        count = sum(1 for l in train_labels if l == i)
        print(f"    {name:10s}: {count} örnek")

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
        BASE_MODEL, num_labels=3,
        id2label=id2label, label2id=label2id, ignore_mismatched_sizes=True,
    )
    args = TrainingArguments(
        output_dir=MODEL_SAVE,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        weight_decay=0.01, # aşırı öğrenmeyi önler
        warmup_ratio=0.1, # 10 %  eğitim verilerini artırır
        eval_strategy="epoch", # her epoch sonunda değerlendir
        save_strategy="epoch", # her epoch sonunda kaydet
        load_best_model_at_end=True, # en iyi modeli yükler
        metric_for_best_model="f1", # en iyi modeli seçmek için f1 kullan
        logging_steps=50, # 50 adımda bir log tutar
        seed=SEED, # rastgelelik tohumu
        report_to="none", # raporlama 
        fp16=True, # daha hızlı eğitim
    )
    trainer = Trainer(
        model=model, 
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["test"],
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer), # batch boyutuna göre tokenları ayarlar
        compute_metrics=compute_metrics, # metrikleri hesaplar
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)], # erken durdurma callback
    )
    trainer.train()

    # 5. Loss grafiği
    plot_training_history(trainer,
        save_path=os.path.join(DATA_SAVE, "sentiment_loss.png"),
        title="Duygu Analizi — Training History")

    # 6. Değerlendirme
    print("\n[5/6] Değerlendirme...")
    out    = trainer.predict(tokenized["test"])
    y_pred = np.argmax(out.predictions, axis=-1)
    y_true = out.label_ids
    save_classification_report(y_true, y_pred, SENTIMENT_LABELS,
        save_path=os.path.join(DATA_SAVE, "sentiment_report.json"))
    plot_confusion_matrix(y_true, y_pred, SENTIMENT_LABELS,
        save_path=os.path.join(DATA_SAVE, "sentiment_confusion_matrix.png"))

    # 7. Model kaydet
    print(f"\n[6/6] Model kaydediliyor -> {MODEL_SAVE}")
    trainer.save_model(MODEL_SAVE)
    tokenizer.save_pretrained(MODEL_SAVE)
    print("\nDuygu analizi egitimi tamamlandi!")


if __name__ == "__main__":
    main()
