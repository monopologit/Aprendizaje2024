# train.py

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib  # Para guardar el modelo

# Cargar los datos
personal_activo_df = pd.read_csv('path_to_data/Personal_Activo_Industria.csv', encoding='latin1', delimiter=';')

# Limpiar los nombres de las columnas eliminando espacios en blanco extra
personal_activo_df.columns = personal_activo_df.columns.str.strip()

# Diccionario para convertir los nombres de los meses en español a números
meses = {
    'Enero': '01', 'enero': '01', 'Febrero': '02', 'febrero': '02', 'Marzo': '03', 'marzo': '03',
    'Abril': '04', 'abril': '04', 'Mayo': '05', 'mayo': '05', 'Junio': '06', 'junio': '06',
    'Julio': '07', 'julio': '07', 'Agosto': '08', 'agosto': '08', 'Septiembre': '09', 'septiembre': '09',
    'Octubre': '10', 'octubre': '10', 'Noviembre': '11', 'noviembre': '11', 'Diciembre': '12', 'diciembre': '12'
}

# Función para reemplazar los nombres de los meses en español por números y manejar NaN
def convertir_fecha(row):
    if pd.isna(row['Año']) or pd.isna(row['Mes']):
        return pd.NaT
    año = str(int(row['Año']))
    mes = str(row['Mes']).strip()
    mes = meses.get(mes, '01')  # Asignar '01' como valor predeterminado si el mes no se encuentra en el diccionario
    return pd.to_datetime(f'{año}-{mes}', format='%Y-%m')

# Aplicar la función para crear la columna 'Fecha'
personal_activo_df['Fecha'] = personal_activo_df.apply(convertir_fecha, axis=1)

# Establecer la columna 'Fecha' como el índice
personal_activo_df.set_index('Fecha', inplace=True)

# Preparación de datos para el modelo
X = personal_activo_df[['Confeccionistas', 'Electrónicas', 'Plásticas', 'Textiles', 'Pesqueras', 'Otras']].values
personal_activo_df['Total'] = personal_activo_df[['Confeccionistas', 'Electrónicas', 'Plásticas', 'Textiles', 'Pesqueras', 'Otras']].sum(axis=1)
y = personal_activo_df['Total'].values

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear un imputador que reemplace los NaN con la mediana de cada columna
imputer = SimpleImputer(strategy='median')

# Aplicar el imputador a los datos de entrenamiento y prueba
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

# Definir el modelo
rf_model = RandomForestRegressor()

# Definir la cuadrícula de hiperparámetros
rf_param_grid = {
    'n_estimators': [100, 200, 300],
    'max_features': ['sqrt', 'log2'],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Configurar GridSearchCV
rf_grid_search = GridSearchCV(estimator=rf_model, param_grid=rf_param_grid, cv=5, scoring='r2', n_jobs=-1)

# Entrenar el modelo con GridSearchCV
rf_grid_search.fit(X_train_imputed, y_train)

# Obtener los mejores hiperparámetros
rf_best_params = rf_grid_search.best_params_
print("Mejores Hiperparámetros de Random Forest:", rf_best_params)

# Entrenar el modelo con los mejores hiperparámetros
rf_best_model = rf_grid_search.best_estimator_
rf_best_model.fit(X_train_imputed, y_train)

# Guardar el modelo y el imputador
joblib.dump(rf_best_model, 'path_to_save_model/random_forest_model.pkl')
joblib.dump(imputer, 'path_to_save_model/imputer.pkl')

# Evaluar el modelo
y_pred = rf_best_model.predict(X_test_imputed)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error (MSE) de Random Forest:", mse)
print("Mean Absolute Error (MAE) de Random Forest:", mae)
print("R2 Score de Random Forest:", r2)
