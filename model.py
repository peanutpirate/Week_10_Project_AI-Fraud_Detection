import tensorflow as tf

def build_fraud_model(input_shape):
    """
    Kredi kartı dolandırıcılığı için Deep Learning mimarisi oluşturur.
    """
    model = tf.keras.Sequential([
        # İlk katman ve Dropout (overfitting'i engellemek için)
        tf.keras.layers.Dense(32, activation='relu', input_shape=(input_shape,)),
        tf.keras.layers.Dropout(0.2),
        
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(8, activation='relu'),
        
        # Çıkış katmanı: Dolandırıcı mı değil mi? (0 ile 1 arası olasılık)
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=[tf.keras.metrics.Recall(name='recall'), 'accuracy']
    )
    
    print("✅ Model mimarisi oluşturuldu.")
    return model