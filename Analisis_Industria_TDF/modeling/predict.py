# predict.py

import pandas as pd
import joblib

# Cargar el modelo y el imputador
model = joblib.load('path_to_save_model/random_forest_model.pkl')
imputer = joblib.load('path_to_save_model/imputer.pkl')

# Función para predecir usando el modelo
def predecir(data):
    # Convertir a DataFrame
    df = pd.DataFrame(data)
    
    # Aplicar imputador
    data_imputed = imputer.transform(df)
    
    # Hacer predicciones
    predicciones = model.predict(data_imputed)
    
    return predicciones

# Ejemplo de datos para predecir
nuevos_datos = {
    'Confeccionistas': [100, 150, 200],
    'Electrónicas': [120, 130, 140],
    'Plásticas': [80, 90, 100],
    'Textiles': [50, 60, 70],
    'Pesqueras': [30, 40, 50],
    'Otras': [20, 25, 30]
}

# Predecir usando el modelo
predicciones = predecir(nuevos_datos)
print("Predicciones:", predicciones)
