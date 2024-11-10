import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split, GridSearchCV
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

# Ajuste de hiperparámetros usando GridSearchCV con Ridge
param_grid = {
    'alpha': [0.1, 1, 10, 100],
    'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sag']
}
grid_search = GridSearchCV(estimator=Ridge(), param_grid=param_grid, cv=5, scoring='r2', n_jobs=-1)
grid_search.fit(X_train_imputed, y_train)

# Mejor modelo
best_model = grid_search.best_estimator_
best_model.fit(X_train_imputed, y_train)

# Guardar el modelo entrenado
joblib.dump(best_model, '../models/ridge_model.pkl')
print(f"Modelo guardado en '../models/ridge_model.pkl'")

