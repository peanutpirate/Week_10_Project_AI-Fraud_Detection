from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def preprocess_data(df):
    """
    Veriyi temizler, ölçeklendirir ve eğitim/test olarak ayırır.
    """
    # Amount sütununu ölçeklendir (Neural Network büyük sayıları sevmez)
    scaler = StandardScaler()
    df['Amount'] = scaler.fit_transform(df['Amount'].values.reshape(-1, 1))
    
    # Time sütunu genellikle gereksizdir, çıkaralım
    df = df.drop(['Time'], axis=1)
    
    X = df.drop('Class', axis=1) # Özellikler
    y = df['Class']              # Hedef (0 veya 1)
    
    # Stratify=y kullanarak her iki grupta da eşit oranda fraud olmasını sağlıyoruz
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print("✅ Ön işleme tamamlandı. Veri setleri hazır.")
    return X_train, X_test, y_train, y_test