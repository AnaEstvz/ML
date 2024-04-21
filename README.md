# Proyecto de Machine Learning: Predicción de Danceability en Canciones
En este proyecto de Machine Learning, hemos adquirido un dataset de Kaggle que contiene información sobre diferentes parámetros asociados a canciones. El objetivo principal es predecir la danceability de las canciones utilizando técnicas de clasificación.

![Texto alternativo]('app/baile.jpg')

## Descripción del Dataset
El dataset contiene una variedad de características para cada canción, como popularidad, acústica, instrumentalidad, valencia, entre otros. Sin embargo, nos enfocamos en predecir la danceability de las canciones, que es nuestra variable objetivo. Danceability es una medida de qué tan adecuada es una pista para bailar, representada como un número entre 0 y 1.

## Proceso de Trabajo
1. Análisis Exploratorio de Datos (EDA)
Realizamos un análisis exploratorio de datos para comprender la distribución de las características y la relación con la variable objetivo. Esto incluyó visualizaciones y estadísticas descriptivas para cada característica.

2. Limpieza de Datos
Durante la limpieza de datos, eliminamos valores nulos o faltantes y realizamos la transformación de datos necesaria. Además, definimos la variable predictora, que en este caso es la danceability de las canciones.

3. Definición de la Variable Objetivo
Decidimos definir dos grupos basados en un umbral de danceability: canciones con poca danceability y canciones con mucha danceability. Observamos que la proporción de muestras en estos grupos era de aproximadamente 0.4 para canciones con poca danceability y 0.6 para canciones con mucha danceability.

4. Mejora de Correlaciones
Exploramos la correlación entre las características y la variable objetivo y tratamos de aumentar las correlaciones de las características más relevantes para nuestro modelo. Utilizamos técnicas como agrupación por rangos, asignación de valores y agrupación por la media de danceability para reasignar nuevos rangos.

5. Selección y Evaluación de Modelos
Probamos varios modelos de clasificación, como Regresión Logística, Árboles de Decisión, y Random Forest. Hiperparametrizamos los modelos para mejorar su rendimiento y también exploramos la posibilidad de incluir pasos de aprendizaje no supervisado en nuestro flujo de trabajo.

## Despliegue de Aplicación en Streamlit
Creamos una aplicación web utilizando Streamlit para permitir a los usuarios realizar predicciones de danceability de canciones. Esto facilita la interacción con el modelo entrenado y permite a los usuarios obtener predicciones rápidas y fáciles.


## Estructura del Repositorio
**data/**: Carpeta que contiene el dataset original y los datos procesados. Así como, los datos de entrenamiento utilizados para entrenar el modelo, y los datos de test utilizados para evaluar el modelo a partir de los datos procesados.
**notebooks/**: Carpeta que contiene los notebooks Jupyter utilizados en el proyecto.
**models/**: Carpeta que contiene los modelos entrenados y el modelo final seleccionado.
**src/**: Contiene los archivos fuente de Python que implementan las funcionalidades clave del proyecto.
**app/**: Contiene los archivos necesarios para el despliegue del modelo en Streamlit.
**README.md**: Documento que proporciona una visión general del proyecto, incluyendo la descripción del dataset, el proceso de trabajo y los resultados obtenidos.

## Resultados y Conclusiones
Concluimos que, después de aplicar técnicas de preprocesamiento y selección de características, así como la evaluación de múltiples modelos, hemos desarrollado un modelo de Machine Learning capaz de predecir la danceability de las canciones con una precisión aceptable. Este modelo puede ser útil para diversas aplicaciones, como recomendación de música, análisis de tendencias musicales y más.




