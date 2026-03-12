import tensorflow as tf

def train_model(model, X_train, y_train, validation_data, epochs=20, batch_size=2048):
    """
    Modeli eğitir ve Early Stopping kullanarak overfitting'i engeller.
    """
    # Overfitting engellemek için sabır parametresi (patience)
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True
    )

    print("🚀 Eğitim başlıyor...")
    history = model.fit(
        X_train, y_train,
        validation_data=validation_data,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping],
        verbose=1
    )
    
    print("✅ Model eğitimi tamamlandı.")
    return history