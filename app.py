"""
app.py — Streamlit Türkçe Haber Analiz Uygulaması

"""
import os
import json
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F


st.set_page_config(
    page_title="Türkçe Haber Analizi",
    page_icon="📰",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Başlık bloğu */
.hero-block {
    background: linear-gradient(135deg, #1e3a5f 0%, #2d6a9f 50%, #1a5276 100%);
    padding: 2.5rem 2rem;
    border-radius: 16px;
    color: white;
    text-align: center;
    margin-bottom: 2rem;
    box-shadow: 0 8px 32px rgba(29,53,87,0.3);
}
.hero-block h1 { font-size: 2rem; font-weight: 700; margin-bottom: 0.3rem; }
.hero-block p  { font-size: 1rem; opacity: 0.85; margin: 0; }

/* Sonuç kartları */
.result-card {
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    margin: 0.6rem 0;
    border-left: 5px solid;
    box-shadow: 0 2px 12px rgba(0,0,0,0.08);
}
.card-topic     { background: #f0f7ff; border-color: #2196F3; }
.card-sentiment { background: #f0fff4; border-color: #4CAF50; }
.card-warning   { background: #fff8e1; border-color: #FF9800; }

.card-label  { font-size: 0.78rem; font-weight: 600; text-transform: uppercase;
               letter-spacing: 0.08em; color: #666; margin-bottom: 0.3rem; }
.card-value  { font-size: 1.5rem; font-weight: 700; color: #1a1a2e; margin-bottom: 0.2rem; }
.card-conf   { font-size: 0.85rem; color: #666; }

/* Duygu renkleri */
.sentiment-pozitif { color: #2e7d32; }
.sentiment-negatif { color: #c62828; }
.sentiment-nötr    { color: #ef6c00; }

/* Güven bar */
.conf-bar-bg  { background: #e0e0e0; border-radius: 99px; height: 6px; margin-top: 6px; }
.conf-bar-fill{ height: 100%; border-radius: 99px;
                background: linear-gradient(90deg, #2196F3, #21CBF3); }

/* Gizle Streamlit footer */
footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)



# Model Yükleme 

TOPIC_MODEL_PATH     = "models/topic_model"
SENTIMENT_MODEL_PATH = "models/sentiment_model"
MAX_LENGTH           = 256


@st.cache_resource(show_spinner=False) 
def load_model(model_path: str):
    """Model ve tokenizer'ı diskten yükle (bir kez çalışır)."""
    if not os.path.isdir(model_path):
        return None, None, None, None

    # Label mapping 
    mapping_path = os.path.join(model_path, "label_mapping.json")
    if os.path.exists(mapping_path):
        with open(mapping_path, "r", encoding="utf-8") as f:
            mapping = json.load(f)
        id2label = {int(k): v for k, v in mapping["id2label"].items()}
        label2id = mapping["label2id"]
    else:
        id2label, label2id = None, None

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model     = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()
    return tokenizer, model, label2id, id2label


def predict(text: str, tokenizer, model, id2label: dict):
    """Tek metin için tahmin döndür."""
    inputs  = tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_LENGTH)
    with torch.no_grad(): # eğitim yok sadece tahmin 
        logits  = model(**inputs).logits 
    probs   = F.softmax(logits, dim=-1)[0] # yüzdelere çevirir
    pred_id = int(torch.argmax(probs)) # en yüksek olasılığa sahip sınıfın id'si
    label   = id2label.get(pred_id, str(pred_id)) # id'ye karşılık gelen sınıf
    confidence = float(probs[pred_id]) # güven skoru 
    # Tüm sınıf olasılıkları
    all_probs = {id2label.get(i, str(i)): float(p) for i, p in enumerate(probs)}
    return label, confidence, all_probs


def render_result_card(card_type: str, icon: str, title: str,
                       label: str, confidence: float, all_probs: dict):
    sentiment_class = ""
    if card_type == "sentiment":
        sentiment_class = f"sentiment-{label.lower().replace('ü','u').replace('ö','o')}"

    bar_width = int(confidence * 100)
    probs_html = " &nbsp;|&nbsp; ".join(
        f"<b>{k}</b>: %{v*100:.1f}" for k, v in
        sorted(all_probs.items(), key=lambda x: -x[1])
    )

    st.markdown(f"""
    <div class="result-card card-{card_type}">
        <div class="card-label">{icon} {title}</div>
        <div class="card-value {sentiment_class}">{label}</div>
        <div class="card-conf">Güven: %{confidence*100:.1f}</div>
        <div class="conf-bar-bg"><div class="conf-bar-fill" style="width:{bar_width}%"></div></div>
        <div style="margin-top:8px;font-size:0.78rem;color:#888;">{probs_html}</div>
    </div>
    """, unsafe_allow_html=True)



st.markdown("""
<div class="hero-block">
    <h1>📰 Türkçe Haber Analizi</h1>
    <p>BERTurk tabanlı yapay zeka ile <b>konu sınıflandırma</b> ve <b>duygu analizi</b></p>
</div>
""", unsafe_allow_html=True)



# Model Yükleme Durumu

with st.spinner("Modeller yükleniyor..."):
    tok_topic, mdl_topic, l2i_topic, i2l_topic = load_model(TOPIC_MODEL_PATH)
    tok_sent,  mdl_sent,  l2i_sent,  i2l_sent  = load_model(SENTIMENT_MODEL_PATH)

topic_ok     = tok_topic is not None
sentiment_ok = tok_sent  is not None

col1, col2 = st.columns(2)
with col1:
    icon = "✅" if topic_ok else "⚠️"
    st.markdown(f"{icon} **Konu Modeli**: {'Hazır' if topic_ok else 'Eğitim gerekli'}")
with col2:
    icon = "✅" if sentiment_ok else "⚠️"
    st.markdown(f"{icon} **Duygu Modeli**: {'Hazır' if sentiment_ok else 'Eğitim gerekli'}")

if not topic_ok or not sentiment_ok:
    st.markdown("""
    <div class="result-card card-warning">
        <div class="card-label">⚠️ Uyarı</div>
        <div style="font-size:0.9rem">
        Bir veya her iki model henüz eğitilmemiş.<br>
        Lütfen önce eğitim scriptlerini çalıştırın:<br><br>
        <code>python train_topic.py</code><br>
        <code>python train_sentiment.py</code>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.divider()



# Metin Giriş Alanı

st.subheader("📝 Haber Metni Girin")
example_texts = [
    "Örnek seçin...",
    "Borsa İstanbul, bugün açılışta yüzde iki artı ile kapandı. Analistler bu yükselişi olumlu karşıladı.",
    "Galatasaray, Şampiyonlar Ligi'nde rakibini 3-0 mağlup ederek grubunda lider konuma yükseldi.",
    "Hükümet, yeni ekonomi paketini bu hafta açıklayacağını belirtti. Detaylar henüz netlik kazanmadı.",
    "Depremin ardından yüzlerce kişi hayatını kaybetti. Yardım ekipleri bölgede çalışmalarını sürdürüyor.",
    "Türk bilim insanları, kanser tedavisinde çığır açacak yeni bir molekül keşfetti.",
]
selected = st.selectbox("Hızlı örnek metin seç:", example_texts)
default_text = "" if selected == "Örnek seçin..." else selected

news_text = st.text_area(
    "Türkçe haber metni:",
    value=default_text,
    height=150,
    max_chars=2000,
    placeholder="Buraya analiz etmek istediğiniz Türkçe haber metnini yapıştırın...",
    label_visibility="collapsed",
)
char_count = len(news_text)
st.caption(f"Karakter sayısı: {char_count}/2000")

analyze_btn = st.button(
    "🔍 Analiz Et",
    type="primary",
    use_container_width=True,
    disabled=(not topic_ok and not sentiment_ok),
)



# Tahmin

if analyze_btn:
    text = news_text.strip()
    if len(text) < 10:
        st.warning("Lütfen en az 10 karakterlik bir metin girin.")
    else:
        st.subheader("📊 Analiz Sonuçları")
        results_col1, results_col2 = st.columns(2)

        with results_col1:
            if topic_ok:
                with st.spinner("Konu analiz ediliyor..."):
                    lbl, conf, probs = predict(text, tok_topic, mdl_topic, i2l_topic)
                render_result_card("topic", "🗞️", "TAHMİN EDİLEN KONU", lbl, conf, probs)
            else:
                st.info("Konu modeli eğitilmedi.")

        with results_col2:
            if sentiment_ok:
                with st.spinner("Duygu analiz ediliyor..."):
                    lbl, conf, probs = predict(text, tok_sent, mdl_sent, i2l_sent)
                render_result_card("sentiment", "💬", "DUYGU ANALİZİ", lbl, conf, probs)
            else:
                st.info("Duygu modeli eğitilmedi.")

        # Analiz edilen metin önizleme
        with st.expander("📄 Analiz edilen metin"):
            st.write(text)



st.divider()
st.markdown("""
<div style="text-align:center;font-size:0.78rem;color:#aaa;">
    Model: <b>dbmdz/bert-base-turkish-cased (BERTurk)</b> &nbsp;|&nbsp;
    Framework: <b>HuggingFace Transformers</b> &nbsp;|&nbsp;
    UI: <b>Streamlit</b>
</div>
""", unsafe_allow_html=True)
