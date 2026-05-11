# Türkçe Haber Analiz Uygulaması

BERTurk tabanlı iki ayrı NLP modeli kullanarak Türkçe haber metinleri üzerinde **konu sınıflandırma** ve **duygu analizi** gerçekleştiren uçtan uca bir makine öğrenmesi projesi.

---

## 📦 Kurulum

```bash
pip install -r requirements.txt
```

> **Not:** GPU varsa PyTorch'un CUDA versiyonunu ayrıca kurun:
> ```bash
> pip install torch --index-url https://download.pytorch.org/whl/cu118
> ```

---

## 🗂️ Proje Yapısı

```
LLM/
├── data/
│   ├── topic/           # Konu eğitim grafikleri, raporlar
│   └── sentiment/       # Duygu eğitim grafikleri, raporlar
├── models/
│   ├── topic_model/     # Fine-tune edilmiş konu modeli
│   └── sentiment_model/ # Fine-tune edilmiş duygu modeli
├── notebooks/
│   └── training_demo.ipynb   # Google Colab notebook
├── app.py               # Streamlit arayüzü
├── train_topic.py       # Konu sınıflandırma eğitimi
├── train_sentiment.py   # Duygu analizi eğitimi
├── utils.py             # Ortak yardımcı fonksiyonlar
├── requirements.txt
└── README.md
```

---

## 🚀 Eğitim

### Konu Sınıflandırma
```bash
python train_topic.py
```
- Veri seti: `savasy/ttc4900` (HuggingFace)
- Model kaydedilir: `models/topic_model/`
- Raporlar: `data/topic/`

### Duygu Analizi
```bash
python train_sentiment.py
```
- Veri seti: `winvoker/turkish-sentiment-analysis-dataset` (HuggingFace)
- Model kaydedilir: `models/sentiment_model/`
- Raporlar: `data/sentiment/`

> **Veri seti erişimi yoksa:** Her iki script de otomatik olarak dahili demo veri seti oluşturur ve eğitimi tamamlar.

> **Disk Alanı Uyarısı:** Eğitim tamamlandıktan sonra `models/topic_model/` ve `models/sentiment_model/` klasörleri içerisinde oluşan `checkpoint-*` isimli klasörler (her biri ~1.2 GB) yalnızca eğitimi kaldığı yerden devam ettirmek için yedek amaçlıdır. Projeyi (uygulamayı) çalıştırmak için bu dosyalara ihtiyaç yoktur ve disk alanından tasarruf etmek için güvenle silinebilirler.

---

## 💻 Uygulamayı Çalıştırma

```bash
streamlit run app.py
```
Tarayıcıda `http://localhost:8501` açılır.

---

## 🧠 Model Seçimi: Neden BERTurk?

| Kriter | Açıklama |
|---|---|
| Türkçe Ön Eğitim | BERTurk, yalnızca Türkçe metinlerle eğitilmiştir |
| Encoder Tabanlı | Sınıflandırma görevleri için ideal mimari |
| Case-Sensitive | Haber metinlerindeki özel isimler için önemli |
| HuggingFace Uyumu | Transformers kütüphanesiyle sorunsuz entegrasyon |
| Topluluk Desteği | Türkçe NLP'nin en yaygın kullanılan modeli |

---

## 📊 Veri Setleri

| Görev | Dataset | Kaynak |
|---|---|---|
| Konu Sınıflandırma | `savasy/ttc4900` | HuggingFace Hub |
| Duygu Analizi | `winvoker/turkish-sentiment-analysis-dataset` | HuggingFace Hub |

---

## ⚙️ Eğitim Parametreleri

```python
num_train_epochs        = 3
per_device_train_batch_size = 8
learning_rate           = 2e-5
weight_decay            = 0.01
warmup_ratio            = 0.1
evaluation_strategy     = "epoch"
metric_for_best_model   = "f1"
early_stopping_patience = 2
```

---

## 🧪 Google Colab

`notebooks/training_demo.ipynb` dosyasını Colab'da açın.
Notebook sırasıyla:
1. GPU kontrolü yapar
2. Bağımlılıkları kurar
3. Her iki eğitim scriptini çalıştırır
4. Grafikleri gösterir

---

## ⚠️ Bilinen Zorluklar

- **Dataset erişimi:** HuggingFace'ten dataset indirmek için internet bağlantısı gerekir. Offline ortamda demo veri seti devreye girer.
- **GPU belleği:** RTX 3050 gibi 4GB VRAM'li kartlar için kodda `BATCH_SIZE=8` olarak ayarlanmıştır.
- **fp16:** Bazı CPU ortamlarında fp16 hata verebilir; `TrainingArguments`'da `fp16=False` yapın.
- **Colab süresi:** 3 epoch eğitim Tesla T4 GPU ile ~15-30 dakika sürer.
