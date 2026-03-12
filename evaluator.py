from sklearn.metrics import classification_report, roc_auc_score

def evaluate_model(model, X_test, y_test):
    """
    Test seti üzerinde modelin performansını detaylı analiz eder.
    """
    print("🔬 Model test ediliyor...")
    y_pred = model.predict(X_test)
    
    # Detaylı Rapor
    print("\n--- Sınıflandırma Raporu ---")
    print(classification_report(y_test, (y_pred > 0.5).astype(int)))
    
    # AUC Skoru (Fraud için çok önemli)
    auc = roc_auc_score(y_test, y_pred)
    print(f"ROC AUC Skoru: {auc:.4f}")
    
    return y_pred