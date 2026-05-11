"""
 Tam Otomasyon Scripti

"""

import subprocess
import sys
import os
import time
from datetime import datetime

LOG_FILE = "training_log.txt"

def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")

def run_script(script_name):
    log(f"{'='*50}")
    log(f"BASLIYOR: {script_name}")
    log(f"{'='*50}")
    start = time.time()
    result = subprocess.run(
        [sys.executable, script_name],
        capture_output=False,   # ciktiyi terminale yansit
        text=True,
    )
    elapsed = time.time() - start
    dakika = int(elapsed // 60)
    saniye = int(elapsed % 60)
    if result.returncode == 0:
        log(f"TAMAMLANDI: {script_name} ({dakika}dk {saniye}sn)")
    else:
        log(f"HATA: {script_name} basarisiz oldu (returncode={result.returncode})")
    return result.returncode == 0

def main():
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write(f"Egitim basladi: {datetime.now()}\n")

    log("RTX 3050 GPU ile yerel egitim basliyor...")

    # 1. Konu siniflandirma
    ok1 = run_script("train_topic.py")
  

    # 2. Duygu analizi
    ok2 = run_script("train_sentiment.py")

    log("="*50)
    if ok1 and ok2:
        log("HER IKI MODEL BASARIYLA EGITILDI!")
        log(f"  Konu modeli   -> models/topic_model/")
        log(f"  Duygu modeli  -> models/sentiment_model/")
        log("Uygulamayi baslatmak icin: streamlit run app.py")
    else:
        log("BAZI MODELLER BASARISIZ OLDU  ")
    log("="*50)

if __name__ == "__main__":
    main()
