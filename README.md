# La Industria de Tierra del Fuego
# *Análisis y Predicción en la Industria de Tierra del Fuego*

![TDF_Industria](https://github.com/user-attachments/assets/870c3a72-71c6-4f68-a222-113939359bf7)


<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

### Industria Fueguina... Un análisis crucial para la economía de Tierra del Fuego.

Este proyecto aplica técnicas de Aprendizaje Automático para analizar datos relacionados con la industria en la provincia de Tierra del Fuego. El objetivo principal es predecir la demanda de personal en los establecimientos industriales, identificar factores que influyen en la producción de ciertos productos y clasificar los establecimientos según su eficiencia productiva. El análisis de estos datos puede proporcionar información valiosa para mejorar la eficiencia y la productividad en los sectores industriales locales


## Organización del Proyecto

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         Analisis_Industria_TDF and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│   └── video          <- Generated video  to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── Analisis_Industria_TDF   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes Analisis_Industria_TDF a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

--------

## Fuente de los datasets: 
https://ipiec.tierradelfuego.gob.ar/estadisticas-economicas-2/

## Orígenes de Datos

Este proyecto utiliza datos del Instituto Provincial de Estadísticas y Censos (IPIEC) de Tierra del Fuego. A continuación se detallan los datasets:

1. **Personal Activo en Establecimientos Industriales por Rama de Actividad**
   - **Periodo**: Enero 2001 - Agosto 2024
   - **Características**: Año, Mes, Confeccionistas, Electrónicas, Plásticas, Textiles, Pesqueras, Otras

2. **Producción Industrial de los Principales Productos por Sector**
   - **Periodo**: Enero 2001 - Agosto 2024
   - **Características**: Año, Mes, Sector, Productos Específicos
   - **Este dataset se transforma en 6 datasets.

3. **Establecimientos Industriales del Sector Manufacturero por Rama de Actividad**
   - **Periodo**: Enero 2001 - Agosto 2024
   - **Características**: Año, Mes, Confeccionistas, Electrónicas, Plásticas, Textiles, Pesqueras, Mecánicas, Otras, Total

## Configuración del Entorno
### Requisitos Previos
Asegúrate de tener instalado Python 3.x y pip.

Instalación del Entorno Virtual
1) Clona el repositorio en tu máquina local:
git clone https://github.com/tu-usuario/Analisis_Industria_TDF.git
cd Analisis_Industria_TDF
2) Crea y activa un entorno virtual:
python -m venv venv
source venv/bin/activate  # En Windows usa `venv\Scripts\activate`
3) Instala las dependencias:
pip install -r requirements.txt

## Ejecución de Notebooks
Los notebooks de Jupyter están disponibles en la carpeta notebooks. Aquí te indicamos cómo ejecutarlos:

1) Asegúrate de que el entorno virtual está activado.

2) Inicia Jupyter Notebook:
jupyter notebook
3) Abre y ejecuta los siguientes notebooks en el orden indicado:
    - 0.0_TDF_Industria_Aprendizaje.ipynb - Análisis Descriptivo y Visualización

    - 1.0_TDF_Industria_Aprendizaje.ipynb - Modelo de Regresión Lineal

    - 2.0_TDF_Industria_Aprendizaje.ipynb - Modelo de Ridge

    - 3.0_TDF_Industria_Aprendizaje.ipynb - Modelo de Random Forest

## Ejecución de Scripts de Modelado
Los scripts de modelado están ubicados en la carpeta Analisis_Industria_TDF/modeling. Aquí se explica cómo entrenar y predecir utilizando los modelos desarrollados:

1) Para entrenar un modelo, ejecuta train.py:
   python Analisis_Industria_TDF/modeling/train.py
2) Para hacer predicciones usando un modelo entrenado, ejecuta predict.py:
   python Analisis_Industria_TDF/modeling/predict.py

## Documentación
- Puedes encontrar más detalles sobre el proyecto en la carpeta "docs". Esta carpeta incluye todo el material explicativo adicional. Tambien la descripción de los orígenes de datos.
- En la carpeta "report" vas a encontrar material en pdf sobre las conclusiones claves derivadas del análisis exploratorio.
Detalle del modelo de Aprendizaje Automático que ha desarrollado, incluyendo la arquitectura, algoritmos utilizados, y ajuste de hiperparámetros.
Tambien se proporciona las métricas de evaluación del modelo, como precisión, recall, F1-score, y cualquier otra métrica relevante.
Se interpretan los resultados del modelo y se ofrecen conclusiones finales sobre cómo el modelo abordó el problema formulado en la primera entrega.

## VIDEO
- En la carpeta "report" hay otra carpeta con el nombre "video" ahí vas a encontrar el link al video explicativo del proyecto.

## MODELO SELECCIONADO
- El modelo de aprendizaje automatico elegido es: Modelo de Ridge con Hiperparámetros Ajustados
- En "docs" y "report" se encuentran los informes con muchos detalles al respecto.
- El archivo es: 2_0_TDF_Industria_Aprendizaje.ipynb   lo vas a encontrar en "notebook" y en "models"

## Cumplimiento final de nuestros objetivos:
### 1. Predicción de la Demanda de Personal en los Establecimientos Industriales
Modelo de Ridge con Hiperparámetros Ajustados: Hemos desarrollado y entrenado un modelo de Ridge, que ha demostrado ser altamente preciso con un R² casi perfecto. Este modelo permite predecir la cantidad de personal requerido en los establecimientos industriales.

### 2. Identificación de Factores que Influyen en la Producción de Ciertos Productos
Análisis Descriptivo y Exploratorio: A través de las estadísticas descriptivas y las visualizaciones, hemos identificado las tendencias y los factores clave que influyen en la producción industrial. Esto incluye el análisis de la cantidad de personal en diferentes sectores y su relación con la producción de productos específicos.

### 3. Clasificación de los Establecimientos Según su Eficiencia Productiva
Análisis Comparativo: Mediante el análisis de los datos de establecimientos industriales, hemos podido clasificar los establecimientos según su eficiencia productiva, observando la correlación entre el número de personal y la producción.

## ¿Quién soy?
Soy Carlos Alberto Gongora, estudiante de la carrera de Ciencia de Datos e IA en el Centro Politécnico Superior Malvinas Argentinas en la ciudad de Rio Grande en la Provincia de Tierra del Fuego. Actualmente cursando y presentando este proyecto para la materia de Aprendizaje Automatico dictada por el Profesor Martín Maximiliano Mirabete.
