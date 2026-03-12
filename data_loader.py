import pandas as pd
import os

def load_data(file_path="data/creditcard.csv"):
    """
    Belirtilen yoldaki CSV dosyasını yükler.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Hata: {file_path} bulunamadı! Lütfen veriyi kontrol et.")
    
    df = pd.read_csv(file_path)
    print(f"✅ Veri başarıyla yüklendi. Satır sayısı: {len(df)}")
    return df