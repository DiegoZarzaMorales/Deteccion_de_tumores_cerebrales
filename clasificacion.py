# ============================================================
# FUNCIONES DE CLASIFICACIÓN
# ============================================================

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score


def entrenar_clasificador(df):
    """
    Entrena un clasificador Random Forest para detectar tumores grandes
    
    Args:
        df: DataFrame con los datos
        
    Returns:
        tuple: (modelo entrenado, accuracy, DataFrame de importancias)
    """
    print("\n" + "="*60)
    print("PASO: Clasificacion con Machine Learning")
    print("="*60)
    
    features = ["MeanIntensity", "StdIntensity", "NumRegiones",
                "AreaTumor", "IntensidadTumor"]
    
    X = df[features]
    y = df["TumorGrande"]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.25, random_state=42, stratify=y
    )
    
    # Random Forest
    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf.fit(X_train, y_train)
    pred_rf = rf.predict(X_test)
    acc_rf = accuracy_score(y_test, pred_rf)
    
    print(f"\nRandom Forest Accuracy: {acc_rf:.2%}")
    print("\nReporte de clasificacion:")
    print(classification_report(y_test, pred_rf, target_names=['Tumor Pequeño', 'Tumor Grande']))
    
    # Importancia de características
    importancias = pd.DataFrame({
        'Característica': features,
        'Importancia': rf.feature_importances_
    }).sort_values('Importancia', ascending=False)
    
    return rf, acc_rf, importancias
