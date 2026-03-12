import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def plot_training_history(history):
    """Eğitim kaybı (loss) ve metrik grafiklerini çizer."""
    plt.figure(figsize=(12, 4))
    
    # Loss Grafiği
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Eğitim Kaybı')
    plt.plot(history.history['val_loss'], label='Doğrulama Kaybı')
    plt.title('Model Kayıp (Loss) Analizi')
    plt.xlabel('Epoch')
    plt.legend()

    # Recall Grafiği
    plt.subplot(1, 2, 2)
    plt.plot(history.history['recall'], label='Eğitim Recall')
    plt.plot(history.history['val_recall'], label='Doğrulama Recall')
    plt.title('Model Duyarlılık (Recall) Analizi')
    plt.xlabel('Epoch')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(y_true, y_pred):
    """Hata matrisini (Confusion Matrix) görselleştirir."""
    cm = confusion_matrix(y_true, (y_pred > 0.5).astype(int))
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix (Hata Matrisi)')
    plt.ylabel('Gerçek Değer')
    plt.xlabel('Tahmin Edilen Değer')
    plt.show()