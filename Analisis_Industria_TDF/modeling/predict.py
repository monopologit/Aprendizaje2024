import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

# Cargar el dataset
personal_activo_df = pd.read_csv('../data/raw/Personal_Activo_Industria.csv', encoding='latin1', delimiter=';')

# Preprocesar el dataset
personal_activo_df.columns = personal_activo_df.columns.str.strip()
personal_activo_df['Mes'] = personal_activo_df['Mes'].apply(lambda x: meses.get(x.strip(), '01'))
personal_activo_df['Fecha'] = personal_activo_df.apply(lambda row: pd.to_datetime(f"{int(row['Año'])}-{row['Mes']}"), axis=1)
personal_activo_df.set_index('Fecha', inplace=True)
personal_activo_df.fillna(0, inplace=True)
personal_activo_df['Total'] = personal_activo_df[['Confeccionistas', 'Electrónicas', 'Plásticas', 'Textiles', 'Pesqueras', 'Otras']].sum(axis=1)

# Definir X (características) y y (variable objetivo)
X = personal_activo_df.drop(columns='Total')
y = personal_activo_df['Total']

# División de datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Imputar valores faltantes
imputer = SimpleImputer(strategy='median')
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

# Cargar el modelo entrenado
best_model = joblib.load('../models/ridge_model.pkl')
print(f"Modelo cargado desde '../models/ridge_model.pkl'")

# Predicciones
y_pred = best_model.predict(X_test_imputed)

# Evaluación del modelo
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Evaluación del Modelo")
print("Mean Squared Error:", mse)
print("Mean Absolute Error:", mae)
print("R2 Score:", r2)

# Gráfica de Actual vs Predicción
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))
plt.scatter(y_test, y_pred, color='blue', edgecolor='k', alpha=0.7)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs Predicted Values (Ridge)')
plt.show()

